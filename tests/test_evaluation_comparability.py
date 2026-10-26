# -*- coding: utf-8 -*-
"""
test_evaluation_comparability.py — 批次 3B：候选/基准评估窗口可比性

不联网、不安装依赖、不训练真实模型。使用纯函数和假滚动结果验证：
  1. 同窗口、同成本、同标签口径时可正常晋级；
  2. 窗口/成本/标签不同时拒绝直接比较；
  3. 旧基准缺少评估元数据时不得默认可比；
  4. 同窗口复评使用新复评指标，而不是陈旧历史指标；
  5. evaluate_promotion / evaluate_stable_upgrade 均受保护；
  6. 不可比原因是证据问题，不应被记录成单纯收益/回撤不达标。

运行：
  python -m unittest discover -s tests -p "test_evaluation_comparability.py" -v
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

import research_engine as re  # noqa: E402
import stock_pool as sp       # noqa: E402


def _specs(offset=0):
    return [
        {"window": 1, "train_start": f"2020-01-0{1+offset}",
         "train_end": f"2020-03-3{1+offset}",
         "valid_start": f"2020-04-0{1+offset}", "valid_end": f"2020-06-3{1+offset}"},
        {"window": 2, "train_start": f"2020-04-0{1+offset}",
         "train_end": f"2020-06-3{1+offset}",
         "valid_start": f"2020-07-0{1+offset}", "valid_end": f"2020-09-3{1+offset}"},
        {"window": 3, "train_start": f"2020-07-0{1+offset}",
         "train_end": f"2020-09-3{1+offset}",
         "valid_start": f"2020-10-0{1+offset}", "valid_end": f"2020-12-3{1+offset}"},
    ]


def _eval(label=None, cost=None, specs=None, data_start="2020-01-01",
          data_end="2020-12-31", n_rows=240):
    return re.make_evaluation_contract(
        specs or _specs(), label or {"mode": "vol", "vol_k": 0.5, "horizon": 5},
        cost or dict(re.ROLLING_COST), 3, re.ROLLING_LOOKAHEAD, 40, 3,
        features=["f1", "f2", "f3"],
        data_start=data_start, data_end=data_end, n_rows=n_rows)


def _metrics(mean=8.0, evaluation=None, n_seeds=3):
    m = {
        "n_windows": 3, "n_seeds": n_seeds, "positive_window_ratio": 1.0,
        "mean_ret_pct": mean, "max_dd_pct": 10.0, "mean_trades": 6.0,
        "seed_std_pct": 1.0, "n_features": 3, "max_depth": 2,
        "n_estimators": 100, "features": ["f1", "f2", "f3"],
        "strategy": {"stop_loss": 5.0, "take_profit": 15.0, "ma_period": 20,
                     "ma_break_exit": True},
        "model_params": {"max_depth": 2, "n_estimators": 100,
                         "learning_rate": 0.05, "subsample": 0.8,
                         "colsample_bytree": 0.8, "reg_lambda": 1.0,
                         "objective": "multi:softprob", "eval_metric": "mlogloss"},
    }
    if evaluation is not None:
        m["evaluation"] = evaluation
    return m


def _rolling(mean=5.0):
    return {
        "window_rets": [mean, mean, mean],
        "window_dds": [-10.0, -10.0, -10.0],
        "window_trades": [5, 5, 5],
        "seed_rets": [mean, mean + 1.0, mean - 1.0] * 3,
        "seed_dds": [-10.0] * 9,
        "seeds_configured": 3, "seeds_per_window": 3,
        "seeds_attempted": 9, "seeds_succeeded": 9,
        "windows_planned": 3, "windows_attempted": 3, "windows_skipped": 0,
        "window_seed_details": [
            {"window": i, "attempted": 3, "succeeded": 3, "failed": 0,
             "failures": [], "skipped": False, "skip_reason": "",
             "seed_rets": [mean, mean + 1.0, mean - 1.0]}
            for i in range(1, 4)
        ],
        "n_seeds": 3, "windows": 3,
    }


class _DummyGui:
    DATA_DIR = DATA_DIR
    LABEL_MODE = "vol"
    LABEL_VOL_K = 0.5
    LABEL_HORIZON = 5


class _FakeRows:
    def __init__(self, columns):
        self.columns = list(columns)

    def copy(self):
        return self

    def dropna(self, subset=None):
        return self

    def reset_index(self, drop=True):
        return self

    def __len__(self):
        return 240


class TestDirectComparability(unittest.TestCase):
    R = re.DEFAULT_PROMOTION_RULES

    def test_same_window_cost_label_can_promote(self):
        ev = _eval()
        c = _metrics(mean=8.0, evaluation=ev)
        b = _metrics(mean=5.0, evaluation=dict(ev))
        v = re.evaluate_promotion(c, b, self.R, 0)
        self.assertTrue(v["promote"], v["reasons"])
        self.assertIsNotNone(v["candidate_score"])

    def test_different_model_configs_do_not_break_evaluation_comparability(self):
        ev = _eval()
        c = _metrics(mean=8.0, evaluation=ev)
        b = _metrics(mean=5.0, evaluation=dict(ev))
        b["features"] = ["other1", "other2", "other3"]
        b["model_params"] = {"max_depth": 4, "n_estimators": 300}
        b["strategy"] = {"stop_loss": 8.0, "take_profit": 30.0,
                         "ma_period": 30, "ma_break_exit": False}
        v = re.evaluate_promotion(c, b, self.R, 0)
        self.assertTrue(v["promote"], v["reasons"])

    def test_window_mismatch_blocks_comparison(self):
        c = _metrics(mean=8.0, evaluation=_eval(specs=_specs(0)))
        b = _metrics(mean=1.0, evaluation=_eval(specs=_specs(1),
                                                 data_start="2021-01-01",
                                                 data_end="2021-12-31"))
        v = re.evaluate_promotion(c, b, self.R, 0)
        self.assertFalse(v["promote"])
        self.assertIsNone(v["candidate_score"])
        self.assertTrue(any("评估不可比" in r for r in v["reasons"]), v["reasons"])
        self.assertFalse(any("平均净收益" in r for r in v["reasons"]),
                         "不可比不能被记成单纯收益不达标")

    def test_cost_mismatch_blocks_comparison(self):
        cost = dict(re.ROLLING_COST)
        cost["commission_rate"] = 0.001
        c = _metrics(evaluation=_eval())
        b = _metrics(evaluation=_eval(cost=cost))
        v = re.evaluate_promotion(c, b, self.R, 0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("评估不可比" in r for r in v["reasons"]), v["reasons"])

    def test_label_mismatch_blocks_comparison(self):
        c = _metrics(evaluation=_eval(label={"mode": "vol", "vol_k": 0.5, "horizon": 5}))
        b = _metrics(evaluation=_eval(label={"mode": "vol", "vol_k": 0.75, "horizon": 5}))
        v = re.evaluate_promotion(c, b, self.R, 0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("label" in r for r in v["reasons"]), v["reasons"])

    def test_old_baseline_without_metadata_is_incomparable(self):
        c = _metrics(mean=8.0, evaluation=_eval())
        b = _metrics(mean=1.0, evaluation=None)
        v = re.evaluate_promotion(c, b, self.R, 0)
        self.assertFalse(v["promote"])
        self.assertIsNone(v["candidate_score"])
        self.assertTrue(any("evaluation" in r or "元数据" in r for r in v["reasons"]),
                        v["reasons"])

    def test_stable_upgrade_is_protected(self):
        c = _metrics(mean=8.0, evaluation=_eval(), n_seeds=3)
        bad = _metrics(mean=1.0, evaluation=_eval(specs=_specs(1)), n_seeds=3)
        ok, why = re.evaluate_stable_upgrade(c, bad, self.R)
        self.assertFalse(ok)
        self.assertIn("评估不可比", why)

        old = _metrics(mean=1.0, evaluation=None, n_seeds=3)
        ok_old, why_old = re.evaluate_stable_upgrade(c, old, self.R)
        self.assertFalse(ok_old)
        self.assertIn("评估不可比", why_old)

        good = _metrics(mean=5.0, evaluation=_eval(), n_seeds=3)
        c["positive_window_ratio"] = 0.7
        ok2, why2 = re.evaluate_stable_upgrade(c, good, self.R)
        self.assertTrue(ok2, why2)


class TestBaselineReevaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eval_cmp_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.reg = sp.ResearchRegistry(os.path.join(self.tmp, "q.db"))
        self.addCleanup(self.reg.close)
        self.sched = re.AutonomousScheduler(
            self.reg, _DummyGui(), data_dir=self.tmp,
            cfg={"rolling_windows": 3, "rolling_seeds": 3,
                 "min_train_rows_rolling": 40})

    def _stored_baseline(self):
        ev = _eval(specs=_specs(0))
        best = _metrics(mean=99.0, evaluation=ev)
        self.reg.set_baseline("600000", {"best_validated": best, "stable": {}},
                              features="f1,f2,f3",
                              params=best["model_params"], source_exp="OLD")
        return best

    def test_reevaluation_uses_new_metrics_not_stale(self):
        self._stored_baseline()
        candidate_eval = _eval(specs=_specs(0))
        results = {"train_rows_all": _FakeRows(["日期", "收盘", "label", "f1", "f2", "f3"])}
        stale = _metrics(mean=99.0, evaluation=_eval(specs=_specs(0)))
        with mock.patch.object(re, "rolling_window_metrics", return_value=_rolling(5.0)):
            m, why = self.sched._reevaluate_config(
                stale, results, candidate_eval, {"model_seeds": 3}, "test")
        self.assertEqual(why, "")
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["mean_ret_pct"], 5.0, places=3)
        self.assertNotAlmostEqual(m["mean_ret_pct"], 99.0)
        self.assertTrue(m.get("re_evaluated"))
        self.assertEqual(m["evaluation"]["fingerprint"],
                         candidate_eval["fingerprint"])

    def test_build_comparison_marks_old_baseline_incomparable(self):
        # 旧基准缺少 evaluation 元数据
        old = _metrics(mean=99.0, evaluation=None)
        self.reg.set_baseline("600000", {"best_validated": old, "stable": {}},
                              features="f1,f2,f3", params=old["model_params"],
                              source_exp="OLD")
        candidate = _metrics(evaluation=_eval())
        result = self.sched._build_comparison(
            "600000",
            {"train_rows_all": _FakeRows(["日期", "收盘", "label", "f1", "f2", "f3"])},
            candidate, {"model_seeds": 3})
        self.assertFalse(result["comparable"])
        self.assertIn("无法同窗口复评", result["reason"])

    def test_build_comparison_returns_reevaluated_baseline(self):
        self._stored_baseline()
        candidate = _metrics(mean=8.0, evaluation=_eval())
        results = {"train_rows_all": _FakeRows(["日期", "收盘", "label", "f1", "f2", "f3"])}
        with mock.patch.object(re, "rolling_window_metrics", return_value=_rolling(5.0)):
            result = self.sched._build_comparison(
                "600000", results, candidate, {"model_seeds": 3})
        self.assertTrue(result["comparable"])
        self.assertIsNotNone(result["baseline"])
        self.assertAlmostEqual(result["baseline"]["mean_ret_pct"], 5.0, places=3)
        v = re.evaluate_promotion(candidate, result["baseline"],
                                  re.DEFAULT_PROMOTION_RULES, 0)
        self.assertTrue(v["promote"], v["reasons"])

    def test_build_comparison_rejects_cost_or_label_protocol_mismatch(self):
        cost = dict(re.ROLLING_COST)
        cost["slippage_rate"] = 0.002
        old_eval = _eval(cost=cost)
        old = _metrics(mean=99.0, evaluation=old_eval)
        self.reg.set_baseline("600000", {"best_validated": old, "stable": {}},
                              features="f1,f2,f3", params=old["model_params"],
                              source_exp="OLD")
        candidate = _metrics(mean=8.0, evaluation=_eval())
        results = {"train_rows_all": _FakeRows(["日期", "收盘", "label", "f1", "f2", "f3"])}
        result = self.sched._build_comparison(
            "600000", results, candidate, {"model_seeds": 3})
        self.assertFalse(result["comparable"])
        self.assertIn("无法同窗口复评", result["reason"])
        self.assertIn("cost", result["reason"])

    def test_finish_incomparable_does_not_reject_as_poor_performance(self):
        old = _metrics(mean=99.0, evaluation=None)
        self.reg.set_baseline("600000", {"best_validated": old, "stable": {}},
                              features="f1,f2,f3", params=old["model_params"],
                              source_exp="OLD")
        self.reg.upsert_stock("600000", name="测试", status=sp.STATUS_READY)
        self.reg.add_experiment({"exp_id": "E1", "code": "600000", "goal": "strategy",
                                 "status": "running", "created_at": re._now_str()})
        metrics = _metrics(mean=8.0, evaluation=_eval())
        metrics["_comparison"] = {"comparable": False,
                                  "reason": "旧基准缺少 evaluation 元数据"}
        plan = {"goal": "strategy", "seed": 1, "train_start": "2020-01-01",
                "train_end": "2020-12-31"}
        with mock.patch.object(self.sched, "_log", lambda *a, **k: None):
            self.sched._finish_experiment(
                {"code": "600000", "name": "测试"}, plan, "E1", metrics, [])
        rec = [r for r in self.reg.list_experiments(code="600000", limit=10)
               if r["exp_id"] == "E1"][0]
        self.assertEqual(rec["status"], "rejected")
        self.assertIn("评估不可比", rec["reject_reason"])
        self.assertNotIn("平均净收益", rec["reject_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
