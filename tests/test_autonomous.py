# -*- coding: utf-8 -*-
"""
test_autonomous.py — 自主研究系统的离线测试（不联网、不长时间训练）

覆盖需求第九节要求的场景：
  1. 股票列表获取失败（保留现有队列 / 记录最后成功与错误 / 空队列连续失败→错误暂停）
  2. 单只股票数据失败 + 有限重试 → 冷却
  3. 重复启动拒绝（同一时刻只允许一个自主研究会话）
  4. 内存预警：不启动新重型任务；危险：请求在可控边界安全暂停；恢复后继续
  5. 暂停 / 恢复 / 停止 语义
  6. 崩溃恢复：running/planned → interrupted，已完成的实验不被重跑
  7. 文件写入失败：安全降级（返回 False，不抛异常，旧文件不被写坏）
  8. 晋级标准（窗口数/种子数/回撤/交易次数/标准差/复杂度/选择偏差）
  9. 研究方向决策确定性（select_direction）
 10. 实验计划避免重复配置（plan_experiment / config_hash）
 11. 晋级判定绝不读取独立测试集（results["test"]）

运行：python -m unittest discover -s <此目录> -p "test_*.py" -v
或   python test_autonomous.py
"""

import os
import sys
import json
import time
import types
import shutil
import tempfile
import threading
import unittest
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)                     # …\模型训练器后端文件夹（同时是后端代码所在目录）
DESKTOP = os.path.dirname(DATA_DIR)                  # …\Desktop
# 后端代码已统一迁移到 模型训练器后端文件夹，优先从代码目录导入；
# 保留 DESKTOP 以兼容旧布局。
for _p in (DATA_DIR, DESKTOP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import safe_io            # noqa: E402
import memory_guard       # noqa: E402
import stock_pool as sp   # noqa: E402
import research_engine as re_  # noqa: E402
import train_xgb_gui as gui    # noqa: E402


# ====================================================================
# 公共工具
# ====================================================================
def make_hist_df(n=300, low_liquidity=False, end=None):
    """合成 K 线：默认“数据质量良好”；low_liquidity=True 时流动性不足。"""
    import pandas as pd
    import numpy as np
    end = pd.Timestamp(end or datetime.now())
    dates = pd.bdate_range(end=end, periods=n)
    rng = np.random.RandomState(20260101)
    close = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    vol = np.full(n, 10.0) if low_liquidity else np.full(n, 100000.0)
    return pd.DataFrame({"日期": dates, "开盘": close, "收盘": close,
                         "最高": close, "最低": close, "成交量": vol})


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
    """一组在“无基准”时也能通过晋级门槛的训练期验证指标。"""
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
    """假的 AutoRunner：立即返回一份“训练期被拒”的结果，不训练、不联网。"""
    def __init__(self, results):
        self.results = results
        self.last_error = ""

    def run(self):
        return True, dict(self.results)


class _DummyGui:
    """调度器只需要极少 gui 能力；缺的属性由调度器自身 try/except 兜底。"""
    DATA_DIR = DATA_DIR


# ====================================================================
# 1. 股票列表 / 数据失败
# ====================================================================
class TestUniverseAndData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="auto_test_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sched(self, **kw):
        kw.setdefault("cfg", {"pause_poll_s": 0.01, "idle_sleep_s": 0.01,
                              "max_consecutive_universe_failures": 2})
        s = re_.AutonomousScheduler(self.reg, _DummyGui(), data_dir=self.tmp, **kw)
        s._sleep = lambda sec: None      # 测试里不真等
        return s

    def test_universe_failure_keeps_existing_queue(self):
        # 先放入一只“可研究”的股票，模拟已经运行过
        self.reg.upsert_stock("600000", name="浦发银行", board="主板(沪)", status=sp.STATUS_READY)
        self.reg.set_state("universe_last_success", "2026-10-01 09:00:00")

        def boom():
            raise RuntimeError("模拟数据源宕机")

        s = self._sched(universe_fetch_fn=boom)
        s._maybe_refresh_universe()      # 不应抛异常

        self.assertGreaterEqual(self.reg.pool_counts()["total"], 1)     # 旧队列保留
        ds = self.reg.get_data_source_status()
        self.assertFalse(ds["ok"])
        self.assertIn("模拟数据源宕机", ds["error"])
        self.assertEqual(self.reg.get_state("universe_last_success"), "2026-10-01 09:00:00")
        self.assertEqual(self.reg.get_stock("600000")["status"], sp.STATUS_READY)

    def test_universe_failure_empty_queue_error_pauses(self):
        def boom():
            raise RuntimeError("数据源不可用")

        s = self._sched(universe_fetch_fn=boom)
        # 第一次：累计 1 次，未达上限，不抛出
        s._maybe_refresh_universe()
        self.assertEqual(s._consec_universe_fail, 1)
        # 第二次：空队列 + 连续 2 次失败 → 抛出并置 error_paused
        with self.assertRaises(re_.ResearchError):
            s._maybe_refresh_universe()
        self.assertEqual((self.reg.get_state("loop") or {}).get("status"), "error_paused")

    def test_single_stock_data_failure_retries_then_cooldown(self):
        calls = {"n": 0}

        def bad_fetch(code, start, end):
            calls["n"] += 1
            raise RuntimeError("行情接口 500")

        s = self._sched(fetch_fn=bad_fetch,
                        scope={"fetch_retries": 2})
        self.reg.upsert_stock("600001", name="测试", status=sp.STATUS_PENDING)
        stock = self.reg.get_stock("600001")
        ok, info = s._ensure_stock_checked(stock)

        self.assertFalse(ok)
        self.assertEqual(info["status"], sp.STATUS_COOLDOWN)
        self.assertEqual(calls["n"], 2)                                  # 有限重试
        row = self.reg.get_stock("600001")
        self.assertEqual(row["status"], sp.STATUS_COOLDOWN)
        self.assertGreaterEqual(int(row["consecutive_fail"]), 1)

    def test_low_liquidity_downgraded_to_cooldown(self):
        s = self._sched(fetch_fn=lambda c, a, b: make_hist_df(low_liquidity=True),
                        scope={"min_avg_amount_wan": 2000})
        self.reg.upsert_stock("600002", name="低流动性", status=sp.STATUS_PENDING)
        ok, info = s._ensure_stock_checked(self.reg.get_stock("600002"))
        self.assertFalse(ok)
        self.assertEqual(info["status"], sp.STATUS_COOLDOWN)
        self.assertIn("流动性", info["reason"])


# ====================================================================
# 2. 重复启动拒绝
# ====================================================================
class TestDuplicateStart(unittest.TestCase):
    def test_start_refused_when_already_running(self):
        recorded = []
        saved_mb = gui.messagebox
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
            gui.messagebox = saved_mb

        self.assertTrue(recorded, "重复启动应给出提示")
        self.assertEqual(recorded[0][0], "info", "重复启动应被拒绝（不是启动，也不是报错）")

    def test_heavy_gate_allows_only_one(self):
        g = memory_guard.HeavyTaskGate()
        self.assertTrue(g.try_acquire("a"))
        self.assertFalse(g.try_acquire("b"))     # 同一时刻只能一个重型任务
        g.release()
        self.assertTrue(g.try_acquire("c"))
        g.release()


# ====================================================================
# 3. 内存保护
# ====================================================================
class _RawGuard(memory_guard.MemoryGuard):
    """用注入的 raw 序列替换真实采样，便于离线判断。"""
    def __init__(self, seq, **kw):
        super().__init__(**kw)
        self._seq = list(seq)
        self._i = 0

    def raw(self):
        v = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return v


class TestMemoryGuard(unittest.TestCase):
    def _guard(self, seq):
        return _RawGuard(seq)

    def test_warn_blocks_new_heavy_task(self):
        g = _RawGuard([(0, 0, 0, -1)])
        warn = int(g.cfg["warn_avail_mb"])
        danger = int(g.cfg["danger_avail_mb"])
        self.assertLess(danger, warn)
        g = _RawGuard([(8000.0, warn - 1, 50.0, -1.0)])
        ok, why = g.can_start_heavy()
        self.assertFalse(ok)
        self.assertIn("预警", why)

    def test_danger_triggers_abort(self):
        g = _RawGuard([(0, 0, 0, -1)])
        danger = int(g.cfg["danger_avail_mb"])
        g = _RawGuard([(8000.0, danger - 1, 50.0, -1.0)])
        ok, why = g.can_start_heavy()
        self.assertFalse(ok)
        self.assertIn("危险", why)
        abort, why2 = g.should_abort()
        self.assertTrue(abort)

    def test_recovery_requires_stable_samples(self):
        g = _RawGuard([(0, 0, 0, -1)])
        warn = int(g.cfg["warn_avail_mb"])
        danger = int(g.cfg["danger_avail_mb"])
        total = float(g.total_mb or 8000.0)
        ok_avail = warn + 2000.0
        g = _RawGuard([(total, danger - 1, 50.0, -1.0),
                       (total, ok_avail, 50.0, -1.0),
                       (total, ok_avail, 50.0, -1.0),
                       (total, ok_avail, 50.0, -1.0)])
        abort, _ = g.should_abort()
        self.assertTrue(abort)                     # 危险时请求安全暂停
        good, note = g.wait_for_safe(max_wait_s=5, poll_s=0.0)
        self.assertTrue(good)                      # 连续 3 次正常后才恢复
        self.assertIn("恢复", note)

    def test_leak_detection(self):
        g = _RawGuard([(0, 0, 0, -1)], cfg={"leak_window": 4, "leak_growth_mb": 50.0})
        warn = int(g.cfg["warn_avail_mb"])
        total = float(g.total_mb or 8000.0)
        avail = warn + 5000.0
        g = _RawGuard([(total, avail, 100.0, -1.0),
                       (total, avail, 200.0, -1.0),
                       (total, avail, 300.0, -1.0),
                       (total, avail, 400.0, -1.0)],
                      cfg={"leak_window": 4, "leak_growth_mb": 50.0})
        for _ in range(4):
            g.sample(force=True)
        self.assertTrue(g._leak_flag)
        self.assertIn("疑似内存泄漏", g._leak_note)

    def test_scheduler_danger_pauses_at_safe_boundary(self):
        tmp = tempfile.mkdtemp(prefix="auto_mem_")
        try:
            reg = sp.ResearchRegistry(os.path.join(tmp, "q.db"))

            class AbortGuard:
                def should_abort(self):
                    return True, "内存危险"

            s = re_.AutonomousScheduler(reg, _DummyGui(), data_dir=tmp, guard=AbortGuard(),
                                        cfg={"pause_poll_s": 0.01})
            f = s._stop_fn_for_runner()
            self.assertTrue(f())                              # 危险 → 请求在安全边界停止
            self.assertTrue(s.pause_requested)
            loop = reg.get_state("loop") or {}
            self.assertEqual(loop.get("status"), "paused")
            self.assertIn("内存保护", loop.get("reason", ""))
            reg.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ====================================================================
# 4. 暂停 / 恢复 / 停止 + 断点续跑
# ====================================================================
class TestSchedulerControl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="auto_ctrl_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self):
        cfg = {"pause_poll_s": 0.05, "idle_sleep_s": 0.05,
               "max_experiments_per_session": 0, "max_experiment_retries": 0,
               "universe_refresh_hours": 12}
        s = re_.AutonomousScheduler(
            self.reg, _DummyGui(), data_dir=self.tmp, cfg=cfg,
            scope={"fetch_retries": 1, "min_rows": 50},
            universe_fetch_fn=lambda: __import__("pandas").DataFrame(
                {"code": ["600000"], "name": ["浦发银行"]}),
            fetch_fn=lambda c, a, b: make_hist_df(),
            runner_factory=lambda c: _FakeRunner(REJECT_RESULTS))
        s._sleep = lambda sec: time.sleep(min(0.02, max(0.0, sec)))
        return s

    def _done(self):
        c = self.reg.get_state("counters") or {}
        return int(c.get("experiments_done", 0) or 0)

    def test_pause_resume_stop(self):
        s = self._build()
        th = threading.Thread(target=s.run_loop, daemon=True)
        th.start()
        try:
            # 等循环真正跑起来并完成至少 1 个实验
            self.assertTrue(wait_until(lambda: self._done() >= 1, timeout=8),
                            "自主循环未产出实验")
            # 暂停：不再启动新实验
            s.request_pause("测试暂停")
            self.assertTrue(wait_until(
                lambda: (self.reg.get_state("loop") or {}).get("status") == "paused", timeout=5))
            n_paused = self._done()
            time.sleep(0.4)
            self.assertEqual(self._done(), n_paused, "暂停后不应再启动新实验")

            # 恢复：继续产出实验
            s.request_resume()
            self.assertTrue(wait_until(lambda: self._done() > n_paused, timeout=8),
                            "恢复后应继续实验")

            # 停止 ≠ 暂停：彻底结束本次会话
            s.request_stop("测试停止")
            th.join(timeout=8)
            self.assertFalse(th.is_alive(), "停止后线程应退出")
            self.assertTrue(s.stop_requested)
            self.assertEqual((self.reg.get_state("loop") or {}).get("status"), "stopped")
        finally:
            s.request_stop("清理")
            th.join(timeout=5)

    def test_pause_blocks_new_experiments_deterministically(self):
        """暂停状态下（未恢复）不产生任何新实验。"""
        s = self._build()
        s.request_pause("先暂停")
        th = threading.Thread(target=s.run_loop, daemon=True)
        th.start()
        try:
            time.sleep(0.5)
            self.assertEqual((self.reg.get_state("loop") or {}).get("status"), "paused")
            self.assertEqual(self._done(), 0, "暂停状态下不应运行实验")
        finally:
            s.request_stop("清理")
            th.join(timeout=5)


# ====================================================================
# 5. 崩溃恢复
# ====================================================================
class TestCrashRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="auto_rec_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unfinished_marked_interrupted_and_completed_kept(self):
        self.reg.add_experiment({"exp_id": "a1", "code": "600000", "status": "running"})
        self.reg.add_experiment({"exp_id": "a2", "code": "600000", "status": "planned"})
        self.reg.add_experiment({"exp_id": "done1", "code": "600000", "status": "promoted",
                                 "promoted": 1, "metrics_json": "{}"})

        s = re_.AutonomousScheduler(self.reg, _DummyGui(), data_dir=self.tmp)   # 模拟重启

        by_id = {r["exp_id"]: r for r in self.reg.list_experiments(code="600000", limit=50)}
        self.assertEqual(by_id["a1"]["status"], "interrupted")
        self.assertEqual(by_id["a2"]["status"], "interrupted")
        self.assertEqual(by_id["done1"]["status"], "promoted")      # 已完成的绝不重跑/改写
        self.assertTrue(int(by_id["done1"]["promoted"]))
        counters = self.reg.get_state("counters") or {}
        self.assertGreaterEqual(int(counters.get("interrupted", 0)), 2)
        self.assertEqual(self.reg.unfinished_experiments(), [])

    def test_resume_after_recovery_adds_new_experiment(self):
        self.reg.add_experiment({"exp_id": "old_running", "code": "600000", "status": "running"})
        s = re_.AutonomousScheduler(
            self.reg, _DummyGui(), data_dir=self.tmp,
            cfg={"pause_poll_s": 0.01, "max_experiments_per_session": 1,
                 "max_experiment_retries": 0},
            scope={"fetch_retries": 1, "min_rows": 50},
            universe_fetch_fn=lambda: __import__("pandas").DataFrame(
                {"code": ["600000"], "name": ["浦发银行"]}),
            fetch_fn=lambda c, a, b: make_hist_df(),
            runner_factory=lambda c: _FakeRunner(REJECT_RESULTS))
        s._sleep = lambda sec: None
        ok, note = s.run_loop()
        self.assertTrue(ok)
        ids = [r["exp_id"] for r in self.reg.list_experiments(code="600000", limit=50)]
        self.assertIn("old_running", ids)
        self.assertNotEqual(set(ids), {"old_running"}, "恢复后应产生新的实验")
        self.assertGreaterEqual(len(ids), 2)


# ====================================================================
# 6. 安全写入 / 写入失败降级
# ====================================================================
class TestSafeIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="auto_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_json_roundtrip(self):
        p = os.path.join(self.tmp, "x.json")
        self.assertTrue(safe_io.atomic_write_json(p, {"a": 1, "中文": "值"}))
        self.assertEqual(safe_io.read_json_safe(p), {"a": 1, "中文": "值"})
        # 覆盖写不会写坏
        self.assertTrue(safe_io.atomic_write_json(p, {"b": 2}))
        self.assertEqual(safe_io.read_json_safe(p), {"b": 2})

    def test_read_missing_and_corrupt(self):
        self.assertIsNone(safe_io.read_json_safe(os.path.join(self.tmp, "nope.json")))
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{ 这不是 JSON")
        self.assertEqual(safe_io.read_json_safe(bad, {"d": 1}), {"d": 1})

    def test_append_csv_adds_header_once(self):
        p = os.path.join(self.tmp, "log.csv")
        self.assertTrue(safe_io.atomic_append_csv(p, ["r1"], header=["h1", "h2"]))
        self.assertTrue(safe_io.atomic_append_csv(p, ["r2"], header=["h1", "h2"]))
        with open(p, "r", encoding="utf-8-sig") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.assertEqual(lines[0], "h1,h2")
        self.assertEqual(lines[1], "r1")
        self.assertEqual(lines[2], "r2")

    def test_append_failure_falls_back_to_atomic_rewrite(self):
        """模拟追加失败（文件被占用）→ 退化为原子重写，旧内容 + 新行都不丢。"""
        import builtins
        p = os.path.join(self.tmp, "busy.csv")
        self.assertTrue(safe_io.atomic_append_csv(p, ["first"], header=["h"]))
        orig_open = builtins.open

        def flaky_open(file, mode="r", *a, **k):
            if "a" in mode and str(file).endswith("busy.csv"):
                raise PermissionError("模拟文件被占用")
            return orig_open(file, mode, *a, **k)

        builtins.open = flaky_open
        try:
            ok = safe_io.atomic_append_csv(p, ["second"], header=["h"])
        finally:
            builtins.open = orig_open
        self.assertTrue(ok)
        with open(p, "r", encoding="utf-8-sig") as f:
            content = f.read()
        self.assertIn("first", content)
        self.assertIn("second", content)

    def test_write_to_unwritable_path_returns_false(self):
        # 让“目录”位置上是一个文件 → 任何写入都必须优雅失败，不能抛异常
        blocker = os.path.join(self.tmp, "blocker")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("x")
        target = os.path.join(blocker, "sub", "x.json")
        self.assertFalse(safe_io.atomic_write_json(target, {"a": 1}))
        self.assertFalse(safe_io.atomic_append_csv(os.path.join(blocker, "x.csv"), ["a"]))
        self.assertFalse(safe_io.atomic_write_text(os.path.join(blocker, "y.txt"), "z"))


# ====================================================================
# 7. 晋级标准（纯函数）
# ====================================================================
class TestPromotion(unittest.TestCase):
    def test_pass_without_baseline(self):
        v = re_.evaluate_promotion(good_metrics(), None, re_.DEFAULT_PROMOTION_RULES, 0)
        self.assertTrue(v["promote"], v["reasons"])

    def test_fail_each_hard_gate(self):
        rules = re_.DEFAULT_PROMOTION_RULES
        cases = {
            "窗口不足": {"n_windows": 1},
            "种子不足": {"n_seeds": 1},
            "收益为负": {"mean_ret_pct": -0.5},
            "正收益窗口占比不足": {"positive_window_ratio": 0.2},
            "回撤过大": {"max_dd_pct": 90.0},
            "交易次数不足": {"mean_trades": 0.5},
            "种子标准差过大": {"seed_std_pct": 30.0},
            "复杂度超限": {"n_features": 50},
        }
        for name, patch in cases.items():
            m = good_metrics()
            m.update(patch)
            v = re_.evaluate_promotion(m, None, rules, 0)
            self.assertFalse(v["promote"], f"{name} 本应被拒绝")

    def test_selection_bias_raises_bar(self):
        base = good_metrics()
        cand = good_metrics()
        cand["mean_ret_pct"] = base["mean_ret_pct"] + 1.2     # 只比基准高 1.2pp
        rules = re_.DEFAULT_PROMOTION_RULES
        v_few = re_.evaluate_promotion(cand, base, rules, n_experiments=0)
        v_many = re_.evaluate_promotion(cand, base, rules, n_experiments=20)
        self.assertTrue(v_few["promote"], v_few["reasons"])
        self.assertFalse(v_many["promote"], "实验做得越多，改进门槛应越高")
        self.assertGreater(v_many["required_improvement_pp"],
                           v_few["required_improvement_pp"])

    def test_requires_improvement_over_baseline(self):
        base = good_metrics()
        cand = good_metrics()      # 与基准完全相同 → 无改进 → 拒绝
        v = re_.evaluate_promotion(cand, base, re_.DEFAULT_PROMOTION_RULES, 0)
        self.assertFalse(v["promote"])

    def test_first_baseline_requires_extra_seed(self):
        m = good_metrics()
        m["n_seeds"] = 2          # min_seeds=2，但首次建立基准要求 +1
        v = re_.evaluate_promotion(m, None, re_.DEFAULT_PROMOTION_RULES, 0)
        self.assertFalse(v["promote"])

    def test_complexity_penalty_monotonic(self):
        r = re_.DEFAULT_PROMOTION_RULES
        self.assertLess(re_.complexity_penalty(5, 3, 100, r),
                        re_.complexity_penalty(12, 6, 700, r))

    def test_stable_upgrade_stricter(self):
        # 一个能通过 best_validated 晋级、但达不到“稳定基准”的例子：
        # 正收益窗口占比 0.55 达到晋级线 0.50，但低于稳定线 0.50+0.15=0.65。
        m = good_metrics()
        m["positive_window_ratio"] = 0.55
        promoted = re_.evaluate_promotion(m, None, re_.DEFAULT_PROMOTION_RULES, 0)
        self.assertTrue(promoted["promote"], promoted["reasons"])
        ok, why = re_.evaluate_stable_upgrade(m, None, re_.DEFAULT_PROMOTION_RULES)
        self.assertFalse(ok, "稳定基准应比最佳验证更严格")
        # 严格达标时才允许升级
        good = good_metrics()
        good["positive_window_ratio"] = 0.70
        good["n_seeds"] = 3
        ok2, _ = re_.evaluate_stable_upgrade(good, None, re_.DEFAULT_PROMOTION_RULES)
        self.assertTrue(ok2)

    def test_promotion_ignores_test_set(self):
        c1 = good_metrics()
        c2 = good_metrics()
        c2["test"] = {"ret_pct": -999.0, "max_dd_pct": 999.0, "acc": 0.0}
        v1 = re_.evaluate_promotion(c1, None, re_.DEFAULT_PROMOTION_RULES, 0)
        v2 = re_.evaluate_promotion(c2, None, re_.DEFAULT_PROMOTION_RULES, 0)
        self.assertEqual(v1["promote"], v2["promote"])
        self.assertEqual(v1["candidate_score"], v2["candidate_score"])
        self.assertTrue(v2["promote"])

    def test_build_metrics_ignores_test_set(self):
        res = dict(REJECT_RESULTS)
        res["best_strategy"] = {"val_ret_pct": 4.0, "val_dd": -0.2, "val_trades": 5}
        res["stability"] = {"mean_ret": 4.0, "std_ret": 1.0, "seed_rets": [3.0, 5.0]}
        res["test"] = {"ret_pct": 999.0, "max_dd_pct": 1.0}
        m = re_.build_metrics_from_results(res)
        self.assertEqual(m["mean_ret_pct"], 4.0)      # 不受测试集影响


# ====================================================================
# 8. 方向决策 / 实验计划（纯函数）
# ====================================================================
class TestDirectionAndPlan(unittest.TestCase):
    def test_direction_deterministic(self):
        hist = [{"goal": "strategy", "status": "done", "promoted": 0,
                 "reject_reason": "回撤过大", "created_at": "2026-01-01 10:00:00"}]
        d1 = re_.select_direction(hist, now=datetime(2026, 10, 16, 9, 0, 0))
        d2 = re_.select_direction(hist, now=datetime(2026, 10, 16, 9, 0, 0))
        self.assertEqual(d1["goal"], d2["goal"])
        self.assertEqual(d1["priorities"], d2["priorities"])

    def test_direction_empty_history_stable(self):
        now = datetime(2026, 10, 16, 9, 0, 0)
        a = re_.select_direction([], now=now)
        b = re_.select_direction([], now=now)
        self.assertEqual(a["goal"], b["goal"])
        self.assertIn(a["goal"], re_.DIRECTION_ORDER)

    def test_direction_hint_strategy(self):
        now = datetime(2026, 10, 16, 9, 0, 0)
        hist = []
        for i, g in enumerate(re_.DIRECTION_ORDER):
            hist.append({"goal": g, "status": "done", "promoted": 0,
                         "reject_reason": "", "created_at": f"2026-01-0{i + 1} 10:00:00"})
        hist[-1]["reject_reason"] = "交易指标未达标：收益过低、回撤过大"
        hist[-1]["goal"] = re_.DIRECTION_ORDER[-1]
        d = re_.select_direction(hist, now=now)
        self.assertEqual(d["goal"], "strategy")

    def test_direction_hint_stability(self):
        now = datetime(2026, 10, 16, 9, 0, 0)
        hist = []
        for i, g in enumerate(re_.DIRECTION_ORDER):
            hist.append({"goal": g, "status": "done", "promoted": 0,
                         "reject_reason": "", "created_at": f"2026-01-0{i + 1} 10:00:00"})
        hist[-1]["reject_reason"] = "验证窗口数量不足，证据不够稳定"
        hist[-1]["goal"] = re_.DIRECTION_ORDER[-1]
        d = re_.select_direction(hist, now=now)
        self.assertEqual(d["goal"], "stability")

    def test_plan_hash_deterministic_and_review(self):
        p1 = re_.plan_experiment("features", [], seed=123, model_space_size=5)
        p2 = re_.plan_experiment("features", [], seed=123, model_space_size=5)
        self.assertEqual(p1["config_hash"], p2["config_hash"])
        self.assertEqual(p1["config_hash"], re_.config_hash_of(p1))
        # 历史里已有同一 config_hash → 标记为“复核”（历史条目本身不改变计划生成）
        hist_entry = {"config_json": json.dumps(
            {"param_priority": [], "label": None, "config_hash": p1["config_hash"]}),
            "features": "", "goal": "features"}
        p3 = re_.plan_experiment("features", [hist_entry], seed=123, model_space_size=5)
        self.assertEqual(p3["config_hash"], p1["config_hash"])
        self.assertTrue(p3["review"])

    def test_plan_avoids_tried_model_params(self):
        hist = [{"config_json": json.dumps({"param_priority": [0]}),
                 "goal": "strategy", "promoted": 0, "status": "done", "features": ""}]
        p = re_.plan_experiment("strategy", hist, seed=7, model_space_size=4)
        self.assertNotEqual(p["param_priority"][0], 0, "应优先尝试未试过的模型参数")

    def test_plan_label_mode_prefers_untried(self):
        hist = [{"config_json": json.dumps({"label": {"mode": "vol", "vol_k": 0.5}}),
                 "goal": "label_mode", "promoted": 0, "status": "done", "features": ""}]
        p = re_.plan_experiment("label_mode", hist, seed=1, model_space_size=4)
        self.assertIsNotNone(p["label"])
        self.assertNotEqual((p["label"]["mode"], p["label"]["vol_k"]), ("vol", 0.5))


# ====================================================================
# 9. 调度公平性（不饿死 / 不垄断）
# ====================================================================
class TestFairness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="auto_fair_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_never_researched_covered_first(self):
        self.reg.upsert_stock("600001", name="老的", status=sp.STATUS_READY,
                              research_count=10, quality=0.9, last_researched="2020-01-01 00:00:00")
        self.reg.upsert_stock("600002", name="新的", status=sp.STATUS_READY,
                              research_count=0, quality=0.5)
        picked = self.reg.pick_next(count=1, scope={"min_interval_hours": 6})
        self.assertEqual(picked[0]["stock"]["code"], "600002")
        self.assertEqual(picked[0]["schedule"]["sequence"], "首次覆盖")

    def test_cooldown_prefers_others(self):
        now = datetime(2026, 10, 16, 12, 0, 0)
        self.reg.upsert_stock("600011", name="刚研究过", status=sp.STATUS_READY,
                              research_count=3, quality=0.9, last_researched="2026-10-16 11:00:00")
        self.reg.upsert_stock("600012", name="很久没研究", status=sp.STATUS_READY,
                              research_count=3, quality=0.5, last_researched="2026-09-01 10:00:00")
        picked = self.reg.pick_next(count=1, scope={"min_interval_hours": 6}, now=now)
        self.assertEqual(picked[0]["stock"]["code"], "600012")


# ====================================================================
# ====================================================================
# 10. 端到端模拟（少量股票 / 少量实验，全部注入假数据，不联网、不真训练）
# ====================================================================
GOOD_RESULTS = {
    "best_features": ["f1", "f2", "f3", "f4", "f5"],
    "best_model_params": {"max_depth": 3, "n_estimators": 200},
    "best_strategy": {"val_ret_pct": 8.0, "val_dd": -0.2, "val_trades": 6,
                      "ma_period": 20, "stop_loss": 5.0, "take_profit": 15.0,
                      "ma_break_exit": True},
    "stability": {"mean_ret": 8.0, "std_ret": 2.0,
                  "seed_rets": [6.0, 8.0, 10.0], "seed_dds": [-0.2, -0.2, -0.2]},
}

# 训练期滚动验证结果（直接注入，避免真训练；窗口/种子都满足晋级门槛）
FAKE_ROLLING = {"window_rets": [6.0, 8.0, 10.0],
                "window_dds": [-0.10, -0.20, -0.15],
                "window_trades": [5.0, 6.0, 7.0],
                "seed_rets": [6.0, 8.0, 10.0, 7.0, 9.0, 11.0],
                "seed_dds": [-0.1, -0.2, -0.15, -0.12, -0.18, -0.2],
                "seeds_configured": 3, "seeds_per_window": 3,
                "seeds_attempted": 9, "seeds_succeeded": 9,
                "window_seed_details": [
                    {"window": 1, "attempted": 3, "succeeded": 3, "failed": 0,
                     "failures": [], "skipped": False, "skip_reason": "",
                     "seed_rets": [6.0, 7.0, 5.0]},
                    {"window": 2, "attempted": 3, "succeeded": 3, "failed": 0,
                     "failures": [], "skipped": False, "skip_reason": "",
                     "seed_rets": [8.0, 9.0, 7.0]},
                    {"window": 3, "attempted": 3, "succeeded": 3, "failed": 0,
                     "failures": [], "skipped": False, "skip_reason": "",
                     "seed_rets": [10.0, 11.0, 9.0]},
                ],
                "windows_planned": 3, "windows_attempted": 3, "windows_skipped": 0,
                "n_seeds": 3}


class _ModelRunner:
    """假 AutoRunner：写一个假模型文件（供晋级保存），返回“训练期良好”的结果。"""
    def __init__(self, data_dir, results, created):
        self.data_dir = data_dir
        self.results = results
        self.created = created
        self.last_error = ""

    def run(self):
        eid = self.created[-1]
        d = os.path.join(self.data_dir, "experiments", eid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "model.json"), "w", encoding="utf-8") as f:
            json.dump({"fake_model": True}, f)
        return True, dict(self.results)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="auto_e2e_")
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))
        self.created = []
        self._orig_make_id = re_._make_exp_id

    def tearDown(self):
        re_._make_exp_id = self._orig_make_id
        self.reg.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_id(self, code):
        eid = f"E{len(self.created) + 1}"
        self.created.append(eid)
        return eid

    def test_two_experiments_promote_then_reject(self):
        re_._make_exp_id = self._fake_id
        s = re_.AutonomousScheduler(
            self.reg, _DummyGui(), data_dir=self.tmp,
            cfg={"pause_poll_s": 0.01, "max_experiments_per_session": 2,
                 "max_experiment_retries": 0, "universe_refresh_hours": 12},
            scope={"fetch_retries": 1, "min_rows": 50},
            universe_fetch_fn=lambda: __import__("pandas").DataFrame(
                {"code": ["600000"], "name": ["浦发银行"]}),
            fetch_fn=lambda c, a, b: make_hist_df(),
            runner_factory=lambda c: _ModelRunner(self.tmp, GOOD_RESULTS, self.created))
        s._sleep = lambda sec: None
        s._rolling_for = lambda results, plan: dict(FAKE_ROLLING)   # 注入滚动验证结果

        ok, note = s.run_loop()
        self.assertTrue(ok, note)
        self.assertEqual(len(self.created), 2, "本次会话应正好跑 2 个实验")

        # 1) 数据源状态：成功
        self.assertTrue(self.reg.get_data_source_status()["ok"])

        # 2) 第一个实验应晋级并保存兼容命名的模型文件
        b = self.reg.get_baseline("600000")
        self.assertIsNotNone(b, "晋级后应保存基准")
        self.assertAlmostEqual(b["metrics"]["best_validated"]["mean_ret_pct"], 8.0, places=3)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model_600000_Best.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "model_600000.json")))

        rows = {r["exp_id"]: r for r in self.reg.list_experiments(code="600000", limit=10)}
        self.assertEqual(rows["E1"]["status"], "promoted")
        # 第二个实验与基准完全相同 → 无改进 → 必须被拒绝（不会自我晋级）
        self.assertEqual(rows["E2"]["status"], "rejected")
        self.assertFalse(int(rows["E2"]["promoted"]))

        # 3) 计数器与台账
        counters = self.reg.get_state("counters") or {}
        self.assertGreaterEqual(int(counters.get("promotions", 0)), 1)
        self.assertGreaterEqual(int(counters.get("stable_upgrades", 0)), 1)
        self.assertGreaterEqual(int(counters.get("experiments_rejected", 0)), 1)
        csv_path = os.path.join(self.tmp, "autonomous_experiments.csv")
        self.assertTrue(os.path.exists(csv_path), "应写出人类可读实验台账")
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
        self.assertIn("是否晋级", text)

        # 4) 每个实验都记录了“下一轮方向”，供后续研究使用
        self.assertTrue(rows["E1"]["next_direction"])
        # 5) 独立测试集从未被使用：自主实验的 test 区间仅占位，指标里没有测试收益
        m = json.loads(rows["E1"]["metrics_json"])
        self.assertNotIn("test_ret_pct", m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
