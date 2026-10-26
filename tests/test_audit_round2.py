# -*- coding: utf-8 -*-
"""
test_audit_round2.py — 第二轮独立代码审计的回归 / 验证测试

覆盖审计重点：
  Q1 晋级评分与门槛：candidate_score / 复杂度惩罚 / 选择偏差惩罚 / 边界值 / 单位口径
  Q2 独立测试集隔离：晋级、指标打包、方向决策、计划生成都不吃测试集（含真实 run() 控制流）
  Q3 内存保护：真实小规模 XGBoost 训练 + 危险状态在“安全边界”停止（不并行启动第二个重型任务）
  Q4 AkShare 真实数据：字段/日期/停牌/成交量单位（成交额口径）
  Q5 失败实验确实影响下一轮方向与配置；配置去重 / 冷却 / 公平调度
  Q6 暂停/停止/重复启动/崩溃恢复/模型原子替换/不遗留后台线程
  Q7 与既有模型、历史记录的兼容性

运行：
  python -m unittest test_audit_round2 -v
真实训练的测试用 RUN_REAL_XGB=0 可跳过（默认执行真实训练）。
"""

import os
import sys
import json
import time
import types
import shutil
import queue
import tempfile
import threading
import unittest
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)                     # …\模型训练器后端文件夹（同时是后端代码所在目录）
DESKTOP = os.path.dirname(DATA_DIR)
# 后端代码已统一迁移到 模型训练器后端文件夹，优先从代码目录导入；
# 保留 DESKTOP 以兼容旧布局。
for _p in (DATA_DIR, DESKTOP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import safe_io                # noqa: E402
import memory_guard           # noqa: E402
import stock_pool as sp       # noqa: E402
import research_engine as re_  # noqa: E402
import train_xgb_gui as gui   # noqa: E402

RUN_REAL_XGB = os.environ.get("RUN_REAL_XGB", "1") != "0"
NET_OK = os.environ.get("AUDIT_NET", "1") != "0"


# ====================================================================
# 公共工具
# ====================================================================
def make_hist_df(n=300, low_liquidity=False, end=None):
    end = pd.Timestamp(end or datetime.now())
    dates = pd.bdate_range(end=end, periods=n)
    rng = np.random.RandomState(20260101)
    close = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    vol = np.full(n, 10.0) if low_liquidity else np.full(n, 100000.0)
    return pd.DataFrame({"日期": dates, "开盘": close, "收盘": close,
                         "最高": close, "最低": close, "成交量": vol})


def make_train_rows(n=240, seed=3):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=n)
    close = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    r = rng.randn(n)
    label = np.where(r > 0.5, 1, np.where(r < -0.5, -1, 0))
    df = pd.DataFrame({"日期": dates, "收盘": close, "label": label})
    for i in range(4):
        df[f"F{i}"] = rng.randn(n)
    return df, [f"F{i}" for i in range(4)]


def wait_until(pred, timeout=8.0, interval=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def good_metrics():
    return {"n_windows": 3, "n_seeds": 3, "positive_window_ratio": 0.67,
            "mean_ret_pct": 8.0, "max_dd_pct": 20.0, "mean_trades": 6.0,
            "seed_std_pct": 2.0, "n_features": 5, "max_depth": 3, "n_estimators": 200}


REJECT_RESULTS = {
    "best_features": ["f1", "f2"],
    "best_model_params": {"max_depth": 3, "n_estimators": 100},
    "best_strategy": {"val_ret_pct": -3.0, "val_dd": -0.1, "val_trades": 3,
                      "ma_period": 20, "stop_loss": 5.0, "take_profit": 15.0,
                      "ma_break_exit": True},
    "stability": {"mean_ret": -3.0, "std_ret": 1.0, "seed_rets": [-2.0, -4.0],
                  "seed_dds": [-0.1, -0.12]},
}


class _FakeRunner:
    def __init__(self, results, ok=True, last_error=""):
        self.results = results
        self.ok = ok
        self.last_error = last_error

    def run(self):
        return self.ok, dict(self.results)


class _DummyGui:
    DATA_DIR = DATA_DIR


class _RawGuard(memory_guard.MemoryGuard):
    def __init__(self, seq, **kw):
        super().__init__(**kw)
        self._seq = list(seq)
        self._i = 0

    def raw(self):
        v = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return v


# ====================================================================
# Q1 晋级评分与门槛：单位 / 计算顺序 / 边界
# ====================================================================
class TestScoringNumeric(unittest.TestCase):
    R = re_.DEFAULT_PROMOTION_RULES

    def test_candidate_score_exact_value(self):
        m = {"mean_ret_pct": 10.0, "max_dd_pct": 20.0, "seed_std_pct": 4.0,
             "mean_trades": 8.0, "n_features": 6, "max_depth": 4, "n_estimators": 500}
        # 手算：10 - 0.30*20 - 0.50*4 + 0.10*(8-2) - complexity - 0
        # complexity = 0.15*(6-5) + 0.2*(4-3) + 0.1*(500-400)/100 = 0.15+0.2+0.1 = 0.45
        #        = 10 - 6 - 2 + 0.6 - 0.45 = 2.15
        self.assertAlmostEqual(re_.candidate_score(m, self.R, 0), 2.15, places=4)

    def test_complexity_penalty_reference_values(self):
        self.assertAlmostEqual(re_.complexity_penalty(5, 3, 100, self.R), 0.0, places=4)
        self.assertAlmostEqual(re_.complexity_penalty(6, 3, 100, self.R), 0.15, places=4)
        self.assertAlmostEqual(re_.complexity_penalty(5, 4, 100, self.R), 0.20, places=4)
        self.assertAlmostEqual(re_.complexity_penalty(5, 3, 500, self.R), 0.10, places=4)
        self.assertAlmostEqual(re_.complexity_penalty(7, 5, 600, self.R),
                               0.15 * 2 + 0.2 * 2 + 0.1 * 2, places=4)

    def test_selection_bias_penalty_reference_values(self):
        self.assertEqual(re_.selection_bias_penalty(0, self.R), 0.0)
        self.assertEqual(re_.selection_bias_penalty(5, self.R), 0.0)
        self.assertAlmostEqual(re_.selection_bias_penalty(8, self.R), 0.3, places=4)
        self.assertAlmostEqual(re_.selection_bias_penalty(20, self.R), 1.5, places=4)

    def test_penalty_monotonic_and_nonnegative(self):
        self.assertGreaterEqual(re_.complexity_penalty(1, 1, 1, self.R), 0.0)
        self.assertGreaterEqual(re_.selection_bias_penalty(-5, self.R), 0.0)
        self.assertLessEqual(re_.complexity_penalty(5, 3, 100, self.R),
                             re_.complexity_penalty(8, 5, 799, self.R))

    def test_all_boundaries_pass_at_exact_limits(self):
        """正好落在每条硬门槛边界上的候选应当通过（>= / <= 语义）。"""
        m = {"n_windows": 3, "n_seeds": 3, "positive_window_ratio": 0.5,
             "mean_ret_pct": 0.01, "max_dd_pct": 60.0, "mean_trades": 2.0,
             "seed_std_pct": 8.0, "n_features": 8, "max_depth": 5, "n_estimators": 800}
        v = re_.evaluate_promotion(m, None, self.R, 0)
        self.assertTrue(v["promote"], v["reasons"])

    def test_boundary_just_over_limit_fails(self):
        cases = [
            ("max_dd", {"max_dd_pct": 60.01}),
            ("pos_ratio", {"positive_window_ratio": 0.499}),
            ("trades", {"mean_trades": 1.99}),
            ("seed_std", {"seed_std_pct": 8.01}),
            ("features", {"n_features": 9}),
            ("depth", {"max_depth": 6}),
            ("n_estimators", {"n_estimators": 801}),
        ]
        for name, patch in cases:
            m = {"n_windows": 3, "n_seeds": 3, "positive_window_ratio": 0.5,
                 "mean_ret_pct": 0.01, "max_dd_pct": 60.0, "mean_trades": 2.0,
                 "seed_std_pct": 8.0, "n_features": 8, "max_depth": 5, "n_estimators": 800}
            m.update(patch)
            v = re_.evaluate_promotion(m, None, self.R, 0)
            self.assertFalse(v["promote"], f"{name} 恰好越过上限本应被拒绝")

    def test_mean_ret_zero_is_rejected(self):
        m = good_metrics()
        m["mean_ret_pct"] = 0.0          # 要求严格 > 0
        v = re_.evaluate_promotion(m, None, self.R, 0)
        self.assertFalse(v["promote"])

    def test_missing_mean_ret_fails_safe(self):
        m = good_metrics()
        m["mean_ret_pct"] = None
        v = re_.evaluate_promotion(m, None, self.R, 0)
        self.assertFalse(v["promote"], "缺失收益必须按不通过处理，不能默认通过")

    def test_build_metrics_dd_is_positive_magnitude(self):
        res = dict(REJECT_RESULTS)
        res["best_strategy"] = {"val_ret_pct": -5.0, "val_dd": -0.35, "val_trades": 3}
        res["stability"] = {"mean_ret": -5.0, "std_ret": 1.0, "seed_rets": [-4.0, -6.0]}
        m = re_.build_metrics_from_results(res)
        self.assertGreater(m["max_dd_pct"], 0.0)
        self.assertAlmostEqual(m["max_dd_pct"], 35.0, places=3)

    def test_stable_upgrade_boundaries(self):
        # 稳定线：positive_window_ratio >= 0.5+0.15=0.65，seeds >= 2+1=3
        m = good_metrics()
        m["positive_window_ratio"] = 0.65
        m["n_seeds"] = 3
        ok, why = re_.evaluate_stable_upgrade(m, None, self.R)
        self.assertTrue(ok, why)
        m2 = good_metrics()
        m2["positive_window_ratio"] = 0.649
        m2["n_seeds"] = 3
        ok2, _ = re_.evaluate_stable_upgrade(m2, None, self.R)
        self.assertFalse(ok2)
        # 相对稳定基准还需 >= 2.0pp
        s = good_metrics()
        s["mean_ret_pct"] = 8.0
        c = good_metrics()
        c["mean_ret_pct"] = 9.5
        c["positive_window_ratio"] = 0.7
        c["n_seeds"] = 3
        ok3, why3 = re_.evaluate_stable_upgrade(c, s, self.R)
        self.assertFalse(ok3, why3)
        c["mean_ret_pct"] = 10.0
        ok4, _ = re_.evaluate_stable_upgrade(c, s, self.R)
        self.assertTrue(ok4)


# ====================================================================
# Q2 独立测试集隔离
# ====================================================================
class TestTestSetIsolation(unittest.TestCase):
    def test_build_metrics_source_never_reads_test(self):
        self.assertFalse(_uses_test_key(re_.build_metrics_from_results))

    def test_select_direction_source_never_reads_test(self):
        self.assertFalse(_uses_test_key(re_.select_direction))

    def test_plan_experiment_source_never_reads_test(self):
        self.assertFalse(_uses_test_key(re_.plan_experiment))

    def test_metrics_identical_with_poisoned_test_block(self):
        a = dict(REJECT_RESULTS)
        a["best_strategy"] = {"val_ret_pct": 4.0, "val_dd": -0.2, "val_trades": 5}
        a["stability"] = {"mean_ret": 4.0, "std_ret": 1.0, "seed_rets": [3.0, 5.0]}
        b = json.loads(json.dumps(a))
        b["test"] = {"ret_pct": 99999.0, "max_dd_pct": 0.0, "acc": 1.0,
                     "trades": 99999, "auc": 1.0}
        self.assertEqual(re_.build_metrics_from_results(a), re_.build_metrics_from_results(b))

    def test_direction_unchanged_by_poisoned_test_metrics(self):
        now = datetime(2026, 10, 16, 9, 0, 0)
        h1 = [{"goal": "strategy", "status": "done", "promoted": 0,
               "reject_reason": "回撤过大", "created_at": "2026-01-02 10:00:00"}]
        h2 = [dict(h1[0], metrics_json=json.dumps({"test": {"ret_pct": 999.0}}))]
        self.assertEqual(re_.select_direction(h1, now=now)["goal"],
                         re_.select_direction(h2, now=now)["goal"])

    def test_autonomous_execute_experiment_sets_skip_final_test(self):
        tmp = tempfile.mkdtemp(prefix="audit_skip_")
        reg = sp.ResearchRegistry(os.path.join(tmp, "q.db"))
        seen = {}

        class _CapRunner:
            last_error = ""

            def __init__(self, cfg):
                seen.update(cfg)

            def run(self):
                return True, {"status": "no_qualified", "no_qualified_reason": "x"}

        s = re_.AutonomousScheduler(reg, _DummyGui(), data_dir=tmp,
                                    cfg={"pause_poll_s": 0.01},
                                    runner_factory=lambda cfg: _CapRunner(cfg))
        plan = {"goal": "strategy", "budget": {"model_trials": 2, "feature_trials": 3,
                "strategy_trials": 6, "model_seeds": 2}, "param_priority": [0, 1],
                "label": None, "seed": 1, "train_start": "2020-01-01",
                "train_end": "2024-12-31", "test_start": "2025-01-01",
                "test_end": "2025-03-01"}
        s._execute_experiment({"code": "600000", "name": "x"}, plan, "E1", plan["budget"])
        self.assertTrue(seen.get("skip_final_test"))
        reg.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_auto_runner_control_flow_skips_final_test(self):
        """真实 AutoRunner.run() 在 skip_final_test=True 时【不调用】最终测试阶段。"""
        import auto_research

        class _Stub(auto_research.AutoRunner):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.s8_calls = 0

            def _gui(self):
                return _DummyGui()

            def _apply_budget(self):
                pass

            def _s1_data(self, gui):
                self.results["best_features"] = ["f1"]

            def _s2_features(self, gui):
                pass

            def _s3_model_search(self, gui):
                pass

            def _s4_feature_search(self, gui):
                pass

            def _s5_strategy_search(self, gui):
                pass

            def _s6_stability(self, gui):
                pass

            def _s7_final_train(self, gui):
                pass

            def _s8_final_test(self, gui):
                self.s8_calls += 1
                raise AssertionError("自主模式绝不应执行独立测试")

            def _s9_report(self, gui, aborted=""):
                pass

        r = _Stub({"skip_final_test": True, "ai_name": "T"}, lambda *a, **k: None)
        ok, results = r.run()
        self.assertTrue(ok)
        self.assertEqual(r.s8_calls, 0)
        self.assertIsNone(results.get("test"))
        self.assertTrue(results.get("test_skipped"))

    def test_auto_runner_control_flow_runs_final_test_when_not_skipped(self):
        import auto_research

        class _Stub2(auto_research.AutoRunner):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.s8_calls = 0

            def _gui(self):
                return _DummyGui()

            def _apply_budget(self):
                pass

            def _s1_data(self, gui):
                pass

            def _s2_features(self, gui):
                pass

            def _s3_model_search(self, gui):
                pass

            def _s4_feature_search(self, gui):
                pass

            def _s5_strategy_search(self, gui):
                pass

            def _s6_stability(self, gui):
                pass

            def _s7_final_train(self, gui):
                pass

            def _s8_final_test(self, gui):
                self.s8_calls += 1
                self.results["test"] = {"ret_pct": 1.0}

            def _s9_report(self, gui, aborted=""):
                pass

        r = _Stub2({"skip_final_test": False, "ai_name": "T"}, lambda *a, **k: None)
        ok, results = r.run()
        self.assertTrue(ok)
        self.assertEqual(r.s8_calls, 1)
        self.assertIsNotNone(results.get("test"))


def _func_source(fn):
    import inspect
    try:
        return inspect.getsource(fn)
    except Exception:
        return ""


def _uses_test_key(fn):
    """AST 检查：函数体（忽略文档字符串/注释）是否真的用到了 "test" 这个键。"""
    import ast
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(_func_source(fn)))
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and sl.value == "test":
                return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"):
            for a in node.args:
                if isinstance(a, ast.Constant) and a.value == "test":
                    return True
    return False


# ====================================================================
# Q3 内存保护 + 真实小规模 XGBoost
# ====================================================================
@unittest.skipUnless(RUN_REAL_XGB, "RUN_REAL_XGB=0 跳过真实训练")
class TestRealXGB(unittest.TestCase):
    def test_real_xgb_fit_works(self):
        rng = np.random.RandomState(0)
        X = rng.randn(160, 4)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(160) * 0.3 > 0).astype(int)
        params = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1,
                  "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"}
        m = re_.gui_mod_fit(None, X[:-40], y[:-40], X[-40:], y[-40:], params, 7)
        acc = float((m.predict(X[-40:]) == y[-40:]).mean())
        self.assertGreater(acc, 0.6, "真实 XGBoost 训练应产出可用模型")

    def test_real_rolling_windows_produce_metrics(self):
        rows, feats = make_train_rows(240, seed=3)
        out = re_.rolling_window_metrics(
            gui, rows, feats, {"n_estimators": 15, "max_depth": 2,
                               "learning_rate": 0.1, "eval_metric": "mlogloss"},
            {"ma_period": 10, "stop_loss": 5.0, "take_profit": 15.0, "ma_break_exit": True},
            n_windows=3, seeds_per_window=1, lookahead=5, min_train_rows=40)
        self.assertGreaterEqual(out["windows"], 1, out.get("note"))
        self.assertEqual(len(out["window_rets"]), out["windows"])
        self.assertTrue(len(out["seed_rets"]) >= 1)

    def test_real_training_stops_at_safe_boundary(self):
        """真实训练进行中收到危险/停止请求 → 只在窗口边界停止，不强行中断内核。"""
        rows, feats = make_train_rows(240, seed=5)
        calls = {"n": 0}

        def stop_fn():
            # 每个窗口有 2 个安全检查点（窗口级 + 种子级）；seeds_per_window=1，
            # 故第 1 个窗口共触发 2 次；第 3 次（下一窗口的窗口级检查）才请求停止。
            calls["n"] += 1
            return calls["n"] >= 3

        out = re_.rolling_window_metrics(
            gui, rows, feats, {"n_estimators": 15, "max_depth": 2,
                               "learning_rate": 0.1, "eval_metric": "mlogloss"},
            {"ma_period": 10, "stop_loss": 5.0, "take_profit": 15.0, "ma_break_exit": True},
            n_windows=3, seeds_per_window=1, lookahead=5, min_train_rows=40,
            stop_fn=stop_fn)
        self.assertEqual(out["windows"], 1, "应完成第一个窗口后在安全边界停止")
        self.assertIn("中断", out.get("note", ""))
        self.assertEqual(calls["n"], 3, "第 3 次检查点才停止，说明第一个窗口确实跑完了")

    def test_danger_guard_requests_safe_pause_via_stop_fn(self):
        tmp = tempfile.mkdtemp(prefix="audit_danger_")
        reg = sp.ResearchRegistry(os.path.join(tmp, "q.db"))

        class _DangerGuard:
            def should_abort(self):
                return True, "可用内存过低"

        s = re_.AutonomousScheduler(reg, _DummyGui(), data_dir=tmp, guard=_DangerGuard(),
                                    cfg={"pause_poll_s": 0.01})
        f = s._stop_fn_for_runner()
        self.assertTrue(f())
        self.assertTrue(s.pause_requested)
        reg.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_warn_does_not_start_second_heavy(self):
        """内存预警时 can_start_heavy 返回 False（不启动新重型任务），危险时可安全暂停。"""
        g = _RawGuard([(0, 0, 0, -1)])
        warn = int(g.cfg["warn_avail_mb"])
        danger = int(g.cfg["danger_avail_mb"])
        gw = _RawGuard([(8000.0, warn - 1, 10.0, -1.0)])
        ok, why = gw.can_start_heavy()
        self.assertFalse(ok)
        gd = _RawGuard([(8000.0, danger - 1, 10.0, -1.0)])
        abort, _ = gd.should_abort()
        self.assertTrue(abort)


# ====================================================================
# Q4 AkShare 真实数据
# ====================================================================
@unittest.skipUnless(NET_OK, "AUDIT_NET=0 跳过网络测试")
class TestRealData(unittest.TestCase):
    def test_universe_and_filter_real(self):
        try:
            rows = sp.fetch_universe_rows(retries=2)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"网络不可用，未验证：{e!r}")
        if not rows:
            self.skipTest("网络不可用（返回空），未验证")
        acc, rej = sp.filter_universe(rows)
        self.assertGreater(len(acc), 500)
        reasons = {r["reason"] for r in rej}
        self.assertTrue(any("ST" in x for x in reasons) or any("板块" in x for x in reasons))

    def test_volume_unit_normalized_matches_amount(self):
        """新浪兜底数据的成交量单位应被折算为‘手’，使 收盘×成交量×100 ≈ 成交额。"""
        try:
            import akshare as ak
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d")
            daily = ak.stock_zh_a_daily(symbol="sh600000", start_date=start,
                                        end_date=end, adjust="qfq")
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"网络不可用，未验证：{e!r}")
        if daily is None or len(daily) == 0:
            self.skipTest("网络不可用（无数据），未验证")
        try:
            df = gui.fetch_stock_data("600000", start, end)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"行情获取失败，未验证：{e!r}")
        self.assertIn("成交量", df.columns)
        # 用最后一天与新浪权威 amount 对照（新浪 volume 单位=股，amount=元）
        last_daily = daily.iloc[-1]
        amt_true = float(last_daily["amount"])
        row = df[df["日期"] == pd.Timestamp(last_daily["date"])]
        if len(row) == 0:
            self.skipTest("日期未对齐，无法比对")
        vol = float(row["成交量"].iloc[0])
        close = float(row["收盘"].iloc[0])
        amt_est = close * vol * 100.0        # 成交量按“手”口径
        ratio = amt_est / amt_true if amt_true else 0
        self.assertGreater(ratio, 0.9, "成交额口径偏小：成交量单位可能仍是股")
        self.assertLess(ratio, 1.1, "成交额口径偏大：成交量单位可能被重复放大")

    def test_check_stock_data_amount_plausible(self):
        try:
            info = sp.check_stock_data("600000",
                                       lambda c, s, e: gui.fetch_stock_data(c, s, e))
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"网络不可用，未验证：{e!r}")
        self.assertEqual(info["status"], "ready", info.get("reason"))
        # 浦发银行日均成交额应在数亿元量级（数千万~数百万元级），绝不应是 10^6 万元级
        self.assertGreater(info["avg_amount_wan"], 1000.0)
        self.assertLess(info["avg_amount_wan"], 500000.0)


# ====================================================================
# Q5 失败影响方向/配置；配置去重 / 冷却 / 公平
# ====================================================================
class TestFailureFeedback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit_fb_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failed_experiment_changes_direction_and_config(self):
        now = datetime(2026, 10, 16, 9, 0, 0)
        d0 = re_.select_direction([], now=now)
        self.assertEqual(d0["goal"], "strategy")
        failed = [{"goal": "strategy", "status": "failed", "promoted": 0,
                   "reject_reason": "", "created_at": now.strftime("%Y-%m-%d %H:%M:%S")}]
        d1 = re_.select_direction(failed, now=now)
        self.assertEqual(d1["goal"], "model_params", "执行失败应把方向引向更保守的模型参数")
        p0 = re_.plan_experiment("strategy", [], seed=1)
        p1 = re_.plan_experiment("model_params", failed, seed=1)
        self.assertGreater(p1["budget"]["model_trials"], p0["budget"]["model_trials"],
                           "失败后的配置预算应随之改变")

    def test_run_loop_failure_records_and_sets_next_direction(self):
        s = re_.AutonomousScheduler(
            self.reg, _DummyGui(), data_dir=self.tmp,
            cfg={"pause_poll_s": 0.01, "max_experiments_per_session": 1,
                 "max_experiment_retries": 0, "universe_refresh_hours": 12},
            scope={"fetch_retries": 1, "min_rows": 50},
            universe_fetch_fn=lambda: pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]}),
            fetch_fn=lambda c, a, b: make_hist_df(),
            runner_factory=lambda c: _FakeRunner({"status": "error"}, ok=False,
                                                 last_error="模拟执行失败"))
        s._sleep = lambda sec: None
        ok, note = s.run_loop()
        self.assertTrue(ok, note)
        recs = self.reg.list_experiments(code="600000", limit=10)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["status"], "failed")
        hist = self.reg.list_experiments(code="600000", limit=10)
        self.assertEqual(re_.select_direction(hist)["goal"], "model_params")

    def test_cooldown_after_three_failures_and_fair_pick(self):
        self.reg.upsert_stock("600001", name="失败股", status=sp.STATUS_READY)
        for _ in range(3):
            self.reg.update_after_research("600001", success=False)
        self.assertEqual(self.reg.get_stock("600001")["status"], sp.STATUS_COOLDOWN)
        self.reg.upsert_stock("600002", name="正常股", status=sp.STATUS_READY)
        picked = self.reg.pick_next(count=1, scope={"min_interval_hours": 6})
        self.assertEqual(picked[0]["stock"]["code"], "600002", "冷却中的股票不应被继续研究")

    def test_config_dedup_review_flag_ignores_random_seed(self):
        """修复：config_hash 不再把随机 seed 计入 → 相同结构配置可被识别为“复核”。"""
        p1 = re_.plan_experiment("features", [], seed=111, model_space_size=5)
        p2 = dict(p1)
        p2["seed"] = 999
        # 只改随机种子（其余结构相同）→ 哈希必须一致（修复前会不同）
        self.assertEqual(re_.config_hash_of(p1), re_.config_hash_of(p2))
        self.assertNotEqual(p1["seed"], p2["seed"])
        # 历史里已有同一“结构配置”（用空 param_priority 保证计划生成不变）→ 标记复核
        hist = [{"config_json": json.dumps({"param_priority": [], "label": p1["label"],
                                            "config_hash": p1["config_hash"]}),
                 "features": "", "goal": "features", "promoted": 0, "status": "done"}]
        p3 = re_.plan_experiment("features", hist, seed=111, model_space_size=5)
        self.assertEqual(p3["config_hash"], p1["config_hash"])
        self.assertTrue(p3["review"])
        # 反例：不同研究方向不应被误判为同一配置
        p4 = re_.plan_experiment("strategy", hist, seed=111, model_space_size=5)
        self.assertNotEqual(p4["config_hash"], p1["config_hash"])


# ====================================================================
# Q5b AkShare 内部异常（真实 GUI 报错：'NoneType' object has no attribute 'write'）
# ====================================================================
class TestUniverseFetchRobustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit_fetch_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_akshare_none_type_write_error_wrapped(self):
        """模拟 akshare 内部抛 'NoneType' object has no attribute 'write'（GUI 真实报错）。"""
        def bad_fetch():
            raise AttributeError("'NoneType' object has no attribute 'write'")
        with self.assertRaises(sp.UniverseError) as cm:
            sp.fetch_universe_rows(fetch_fn=bad_fetch, retries=3)
        msg = str(cm.exception)
        self.assertIn("股票列表获取失败", msg)
        self.assertIn("AttributeError", msg)
        self.assertIn("NoneType", msg)

    def test_default_universe_fetcher_wraps_akshare_exception(self):
        """default_universe_fetcher 把 akshare 内部异常包装成带类型名的 UniverseError。"""
        import akshare as ak
        real = ak.stock_info_a_code_name

        def boom():
            raise AttributeError("'NoneType' object has no attribute 'write'")

        ak.stock_info_a_code_name = boom
        try:
            with self.assertRaises(sp.UniverseError) as cm:
                sp.default_universe_fetcher()
            self.assertIn("AttributeError", str(cm.exception))
            self.assertIn("AkShare", str(cm.exception))
        finally:
            ak.stock_info_a_code_name = real

    def test_log_failure_does_not_mask_source_error(self):
        """日志记录失败不能掩盖原始数据源异常。"""
        def bad_fetch():
            raise RuntimeError("原始数据源错误")

        def bad_log(*a, **k):
            raise AttributeError("'NoneType' object has no attribute 'write'")
        with self.assertRaises(sp.UniverseError) as cm:
            sp.fetch_universe_rows(fetch_fn=bad_fetch, retries=2, log_fn=bad_log)
        self.assertIn("RuntimeError", str(cm.exception))
        self.assertIn("原始数据源错误", str(cm.exception))

    def test_scheduler_error_paused_after_repeated_universe_failure(self):
        """连续获取失败 → 错误暂停 + 状态保存（模拟 GUI 截图场景）。"""
        def bad_fetch():
            raise AttributeError("'NoneType' object has no attribute 'write'")
        s = re_.AutonomousScheduler(
            self.reg, _DummyGui(), data_dir=self.tmp,
            cfg={"max_consecutive_universe_failures": 2},
            universe_fetch_fn=bad_fetch)
        s._maybe_refresh_universe()
        self.assertEqual(s._consec_universe_fail, 1)
        with self.assertRaises(re_.ResearchError):
            s._maybe_refresh_universe()
        loop = self.reg.get_state("loop") or {}
        self.assertEqual(loop.get("status"), "error_paused")
        self.assertIn("NoneType", loop.get("reason", ""))


# ====================================================================
# Q6 暂停/停止/重复启动/崩溃恢复/原子替换/线程
# ====================================================================
class TestControlAndAtomic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit_ctrl_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sched(self, **kw):
        cfg = {"pause_poll_s": 0.02, "idle_sleep_s": 0.02,
               "max_experiments_per_session": 0, "max_experiment_retries": 0,
               "universe_refresh_hours": 12}
        cfg.update(kw.pop("cfg", {}))
        s = re_.AutonomousScheduler(
            self.reg, _DummyGui(), data_dir=self.tmp, cfg=cfg,
            scope={"fetch_retries": 1, "min_rows": 50},
            universe_fetch_fn=lambda: pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]}),
            fetch_fn=lambda c, a, b: make_hist_df(),
            runner_factory=kw.pop("runner_factory",
                                  lambda c: _FakeRunner(REJECT_RESULTS)), **kw)
        s._sleep = lambda sec: time.sleep(min(0.02, max(0.0, float(sec))))
        return s

    def _done(self):
        return int((self.reg.get_state("counters") or {}).get("experiments_done", 0) or 0)

    def test_stop_leaves_no_thread_and_releases_gate(self):
        s = self._sched()
        th = threading.Thread(target=s.run_loop, daemon=True)
        th.start()
        self.assertTrue(wait_until(lambda: self._done() >= 1, timeout=10))
        s.request_stop("测试停止")
        th.join(timeout=8)
        self.assertFalse(th.is_alive(), "停止后不得遗留后台研究线程")
        self.assertFalse(memory_guard.GLOBAL_HEAVY_GATE.is_busy(),
                         "停止后必须释放重型任务闸门")

    def test_gate_blocks_concurrent_heavy_task(self):
        """别的重型任务占用闸门时，调度器不得启动新的训练。"""
        gate = memory_guard.GLOBAL_HEAVY_GATE
        self.assertTrue(gate.try_acquire("other_heavy"))
        s = self._sched(cfg={"max_experiments_per_session": 1})
        th = threading.Thread(target=s.run_loop, daemon=True)
        th.start()
        try:
            time.sleep(1.0)
            self.assertEqual(self._done(), 0, "闸门被占用时不应启动新的重型实验")
        finally:
            gate.release()
        self.assertTrue(wait_until(lambda: self._done() >= 1, timeout=10),
                        "闸门释放后应能继续研究")
        s.request_stop("清理")
        th.join(timeout=6)
        self.assertFalse(th.is_alive())

    def test_duplicate_start_refused(self):
        recorded = []
        saved = gui.messagebox
        gui.messagebox = types.SimpleNamespace(
            showinfo=lambda *a, **k: recorded.append(("info", a)),
            showerror=lambda *a, **k: recorded.append(("error", a)),
            showwarning=lambda *a, **k: recorded.append(("warn", a)),
            askyesno=lambda *a, **k: True)
        try:
            class _App:
                working = False
                _auto_running = True

                def _autonomous_available(self):
                    return True

            gui.TrainApp._start_autonomous(_App())
        finally:
            gui.messagebox = saved
        self.assertTrue(recorded)
        self.assertEqual(recorded[0][0], "info")

    def test_manual_worker_defers_when_gate_busy(self):
        gate = memory_guard.GLOBAL_HEAVY_GATE
        self.assertTrue(gate.try_acquire("other"))
        try:
            stub = types.SimpleNamespace(q=queue.Queue(), stop_ev=threading.Event())
            gui.TrainApp._worker(stub, {"code": "600000"})
            kinds = []
            while not stub.q.empty():
                kinds.append(stub.q.get())
            self.assertTrue(any(k[0] == "done" and k[1] is False for k in kinds),
                            "闸门被占用时手动训练应被拒绝，而不是并行开跑")
        finally:
            gate.release()

    def test_promote_model_atomic_and_partial_write_safe(self):
        exp_dir = os.path.join(self.tmp, "experiments", "E1")
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "model.json"), "w", encoding="utf-8") as f:
            f.write("NEW_MODEL")
        dst = os.path.join(self.tmp, "model_600000_Best.json")
        with open(dst, "w", encoding="utf-8") as f:
            f.write("OLD_MODEL")
        s = re_.AutonomousScheduler(self.reg, _DummyGui(), data_dir=self.tmp)
        real_copy = shutil.copy2

        def boom(*a, **k):
            raise OSError("模拟磁盘错误")

        shutil.copy2 = boom
        try:
            with self.assertRaises(Exception):
                s._promote_model({"code": "600000"}, {}, "E1")
        finally:
            shutil.copy2 = real_copy
        with open(dst, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "OLD_MODEL", "复制失败时旧基准文件必须完好")
        # 正常路径：原子替换为新内容
        s._promote_model({"code": "600000"}, {}, "E1")
        with open(dst, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "NEW_MODEL")

    def test_crash_recovery_keeps_completed(self):
        self.reg.add_experiment({"exp_id": "run1", "code": "600000", "status": "running"})
        self.reg.add_experiment({"exp_id": "ok1", "code": "600000", "status": "promoted",
                                 "promoted": 1, "metrics_json": "{}"})
        re_.AutonomousScheduler(self.reg, _DummyGui(), data_dir=self.tmp)
        by = {r["exp_id"]: r for r in self.reg.list_experiments(code="600000", limit=10)}
        self.assertEqual(by["run1"]["status"], "interrupted")
        self.assertEqual(by["ok1"]["status"], "promoted")


# ====================================================================
# Q7 兼容性：不覆盖/不丢失既有模型与历史
# ====================================================================
class TestCompatibility(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit_compat_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_promotion_does_not_touch_unrelated_models(self):
        unrelated = os.path.join(self.tmp, "model_阳光电源.json")
        with open(unrelated, "w", encoding="utf-8") as f:
            f.write("USER_MODEL")
        stable = os.path.join(self.tmp, "model_600000.json")
        with open(stable, "w", encoding="utf-8") as f:
            f.write("USER_STABLE")
        exp_dir = os.path.join(self.tmp, "experiments", "E9")
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "model.json"), "w", encoding="utf-8") as f:
            f.write("NEW")
        s = re_.AutonomousScheduler(self.reg, _DummyGui(), data_dir=self.tmp)
        s._promote_model({"code": "600000"}, {}, "E9")
        with open(unrelated, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "USER_MODEL", "无关用户的模型文件绝不能被改写")
        with open(stable, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "USER_STABLE", "已存在的稳定基准文件不应被覆盖")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model_600000_Best.json")))

    def test_append_history_preserves_existing_rows(self):
        path = os.path.join(self.tmp, "history_600000_风控.csv")
        gui.ensure_history_file(path)
        gui.append_history(path, ["旧记录", "1", "2"])
        gui.append_history(path, ["新记录", "3", "4"])
        header, rows = gui.read_history_rows(path)
        joined = "\n".join(",".join(str(c) for c in r) for r in rows)
        self.assertIn("旧记录", joined)
        self.assertIn("新记录", joined)
        self.assertEqual(len(rows), 2)

    def test_existing_db_survives_reopen(self):
        path = os.path.join(self.tmp, "q.db")
        r1 = sp.ResearchRegistry(path)
        r1.upsert_stock("600123", name="保留", status=sp.STATUS_READY)
        r1.add_experiment({"exp_id": "keep1", "code": "600123", "status": "promoted",
                           "promoted": 1})
        r1.close()
        r2 = sp.ResearchRegistry(path)
        self.assertIsNotNone(r2.get_stock("600123"))
        ids = [x["exp_id"] for x in r2.list_experiments(code="600123", limit=10)]
        self.assertIn("keep1", ids)
        r2.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
