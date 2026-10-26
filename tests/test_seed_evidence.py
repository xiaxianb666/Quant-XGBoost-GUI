# -*- coding: utf-8 -*-
"""
test_seed_evidence.py — 批次 3A/3A.1：滚动验证实际种子计数与晋级证据

覆盖：
  1. 配置 3 个种子，全部成功；
  2. 配置 3 个种子，仅 2 个成功；
  3. 配置 3 个种子，仅 1 个成功；
  4. 所有种子均失败；
  5. 多个窗口成功数量不同，n_seeds 取各窗口成功数最小值，不能被总数掩盖；
  6. 实际成功数不足晋级门槛时不得晋级；
  7. 成功场景的既有晋级逻辑不被破坏；
  8. 无滚动证据时使用稳定性阶段实际成功种子数，不用 budget 配置值补足；
  9. window_seed_details 保留每窗口尝试数、成功数、失败原因。

说明：
  * 使用纯逻辑/离线构造，只依赖标准库；
  * 末尾用最小 pandas 替身、假模型和假回测实际执行 rolling_window_metrics，
    不联网、不训练真实模型，也不要求安装 pandas。
运行：
  python -m unittest discover -s tests -p "test_seed_evidence.py" -v
"""

import os
import sys
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

import research_engine as re  # noqa: E402


# ----------------------------------------------------------------------
# 最小 pandas 替身：仅实现 rolling_window_metrics 实际用到的 DataFrame 操作。
# 这样当前环境没有 pandas 也能执行滚动循环逻辑，且不依赖第三方库。
# ----------------------------------------------------------------------
class _FakeMask:
    def __init__(self, values):
        self._values = list(values)

    def __and__(self, other):
        return _FakeMask([a and b for a, b in zip(self._values, other._values)])


class _FakeSeries:
    def __init__(self, values):
        self._values = list(values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]

    @property
    def iloc(self):
        return self._values

    def __ge__(self, other):
        return _FakeMask([v >= other for v in self._values])

    def __le__(self, other):
        return _FakeMask([v <= other for v in self._values])

    @property
    def values(self):
        return list(self._values)

    def map(self, mapping):
        return _FakeSeries([mapping.get(v, v) for v in self._values])

    def rolling(self, window):
        return _FakeRolling(self._values, int(window))


class _FakeRolling:
    def __init__(self, values, window):
        self.values = list(values)
        self.window = max(1, int(window))

    def mean(self):
        out = []
        for i in range(len(self.values)):
            if i + 1 < self.window:
                out.append(None)
            else:
                chunk = self.values[i + 1 - self.window:i + 1]
                out.append(sum(chunk) / float(self.window))
        return _FakeSeries(out)


class _FakeFrame:
    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = list(columns)

    @property
    def values(self):
        return [[row.get(c) for c in self._columns] for row in self._rows]


class _FakeILoc:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, item):
        return _FakeDataFrame(self.frame._rows[item])


class _FakeDataFrame:
    def __init__(self, rows):
        self._rows = [dict(r) for r in rows]

    def copy(self):
        return _FakeDataFrame(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, _FakeMask):
            return _FakeDataFrame(
                [r for r, keep in zip(self._rows, key._values) if keep])
        if isinstance(key, list):
            return _FakeFrame(self._rows, key)
        return _FakeSeries([r.get(key) for r in self._rows])

    @property
    def iloc(self):
        return _FakeILoc(self)

    def sort_values(self, column):
        return _FakeDataFrame(sorted(self._rows, key=lambda r: r.get(column)))

    def reset_index(self, drop=True):
        return _FakeDataFrame(self._rows)

    def dropna(self, subset=None):
        subset = list(subset or [])
        keep = []
        for r in self._rows:
            ok = True
            for c in subset:
                v = r.get(c)
                if v is None or (isinstance(v, float) and v != v):
                    ok = False
                    break
            if ok:
                keep.append(r)
        return _FakeDataFrame(keep)


def _fake_pandas():
    mod = types.ModuleType("pandas")
    mod.Series = _FakeSeries
    mod.to_datetime = lambda value: value
    return mod


def _results(stability_seed_count=3):
    """训练期结果骨架，不包含任何测试集字段。"""
    return {
        "best_features": ["f1", "f2", "f3", "f4", "f5"],
        "best_model_params": {"max_depth": 2, "n_estimators": 100},
        "best_strategy": {"val_ret_pct": 5.0, "val_dd": -0.1, "val_trades": 5,
                          "ma_period": 20, "stop_loss": 5.0, "take_profit": 15.0,
                          "ma_break_exit": True},
        "stability": {"mean_ret": 5.0, "std_ret": 1.0,
                      "seed_rets": [4.0 + i for i in range(stability_seed_count)],
                      "seed_dds": [-0.1] * stability_seed_count},
    }


def _details(success_counts, attempted=3, failures=None, seed_returns=None):
    """构造逐窗口明细；success_counts 是每个窗口实际成功数。"""
    out = []
    for i, s in enumerate(success_counts, 1):
        fails = list((failures or {}).get(i, []))
        rets = list((seed_returns or {}).get(i, [5.0] * s))
        out.append({"window": i, "attempted": attempted, "succeeded": s,
                    "failed": attempted - s, "failures": fails,
                    "skipped": False, "skip_reason": "",
                    "seed_rets": rets[:s]})
    return out


def _rolling(success_counts, attempted=3, window_rets=None, failures=None,
             seed_returns=None):
    details = _details(success_counts, attempted=attempted, failures=failures,
                       seed_returns=seed_returns)
    total_success = sum(success_counts)
    if window_rets is None:
        # 只有“至少有一个种子成功”的窗口才会产生 window_rets
        window_rets = [5.0 for s in success_counts if s > 0]
    return {
        "window_rets": list(window_rets),
        "window_dds": [-0.1] * len(window_rets),
        "window_trades": [5] * len(window_rets),
        "window_aucs": [0.6] * len(window_rets),
        "seed_rets": [5.0] * total_success,
        "seed_dds": [-0.1] * total_success,
        "seeds_configured": attempted,
        "seeds_per_window": attempted,
        "seeds_attempted": attempted * len(success_counts),
        "seeds_succeeded": total_success,
        "window_seed_details": details,
        "windows_planned": len(success_counts),
        "windows_attempted": len(success_counts),
        "windows_skipped": 0,
        "n_seeds": min(success_counts) if success_counts else 0,
        "windows": len(window_rets),
    }


def _metrics(success_counts, stability_seed_count=3, budget_seeds=3,
             attempted=3, window_rets=None, failures=None, seed_returns=None):
    return re.build_metrics_from_results(
        _results(stability_seed_count),
        rolling=_rolling(success_counts, attempted=attempted,
                         window_rets=window_rets, failures=failures,
                         seed_returns=seed_returns),
        budget={"model_seeds": budget_seeds},
    )


class TestSeedEvidenceSummary(unittest.TestCase):
    R = re.DEFAULT_PROMOTION_RULES

    def test_all_three_seeds_success(self):
        m = _metrics([3, 3, 3])
        self.assertEqual(m["n_seeds"], 3)
        self.assertEqual(m["seeds_configured"], 3)
        self.assertEqual(m["seeds_attempted"], 9)
        self.assertEqual(m["seeds_succeeded"], 9)
        self.assertEqual([d["succeeded"] for d in m["window_seed_details"]],
                         [3, 3, 3])
        # 成功场景不应被本次修复误伤
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertTrue(v["promote"], v["reasons"])

    def test_two_of_three_seeds_success_is_not_enough_for_first_baseline(self):
        m = _metrics([2, 2, 2])
        self.assertEqual(m["n_seeds"], 2)
        self.assertEqual(m["seeds_attempted"], 9)
        self.assertEqual(m["seeds_succeeded"], 6)
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("n_seeds=2" in r for r in v["reasons"]), v["reasons"])

    def test_one_of_three_seeds_success_is_not_enough(self):
        m = _metrics([1, 1, 1])
        self.assertEqual(m["n_seeds"], 1)
        self.assertEqual(m["seeds_succeeded"], 3)
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("n_seeds=1" in r for r in v["reasons"]), v["reasons"])

    def test_all_seeds_fail(self):
        m = _metrics([0, 0, 0], window_rets=[],
                     failures={1: ["seed x 训练失败：boom"],
                               2: ["seed y 回测失败：boom"],
                               3: ["seed z 训练失败：boom"]})
        self.assertEqual(m["n_seeds"], 0)
        self.assertEqual(m["seeds_attempted"], 9)
        self.assertEqual(m["seeds_succeeded"], 0)
        self.assertEqual(m["window_seed_details"][0]["failed"], 3)
        self.assertEqual(m["window_seed_details"][0]["failures"],
                         ["seed x 训练失败：boom"])
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("n_seeds=0" in r for r in v["reasons"]), v["reasons"])

    def test_mixed_windows_use_minimum_not_total(self):
        m = _metrics([3, 2, 3])
        # 跨窗口成功总数是 8，但晋级证据必须取每窗口最小值 2
        self.assertEqual(m["seeds_succeeded"], 8)
        self.assertEqual(m["n_seeds"], 2)
        self.assertNotEqual(m["n_seeds"], m["seeds_succeeded"])
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("n_seeds=2" in r for r in v["reasons"]), v["reasons"])

    def test_zero_success_window_cannot_be_masked_by_other_windows(self):
        m = _metrics([3, 0, 3], window_rets=[5.0, 5.0])
        self.assertEqual(m["n_seeds"], 0)
        self.assertEqual(m["seeds_succeeded"], 6)
        self.assertEqual(m["window_seed_details"][1]["succeeded"], 0)
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])

    def test_actual_success_below_stable_upgrade_threshold(self):
        m = _metrics([2, 2, 2])
        m["positive_window_ratio"] = 1.0
        ok, why = re.evaluate_stable_upgrade(m, None, self.R)
        self.assertFalse(ok)
        self.assertIn("实际成功种子数 2", why)
        m3 = _metrics([3, 3, 3])
        m3["positive_window_ratio"] = 1.0
        ok3, why3 = re.evaluate_stable_upgrade(m3, None, self.R)
        self.assertTrue(ok3, why3)

    def test_no_rolling_uses_actual_stability_seed_count_not_budget(self):
        m = re.build_metrics_from_results(
            _results(stability_seed_count=2),
            rolling=None,
            budget={"model_seeds": 3},
        )
        self.assertEqual(m["n_seeds"], 2)            # 实际成功 2，不是 budget 的 3
        self.assertEqual(m["seeds_configured"], 3)  # 配置值仅记录，不参与门槛
        self.assertEqual(m["stability_seeds_succeeded"], 2)
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])

    def test_no_seed_evidence_fails_closed(self):
        res = _results(stability_seed_count=0)
        res["stability"] = {}
        m = re.build_metrics_from_results(res, rolling=None,
                                          budget={"model_seeds": 3})
        self.assertEqual(m["n_seeds"], 0)
        self.assertIsNone(m["seed_std_pct"])
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"], "无任何种子证据时不得默认通过")

    def test_legacy_rolling_without_details_is_unknown_and_blocked(self):
        # 旧格式没有逐窗口明细：旧 n_seeds=3 可能是配置值，不能作为实际成功数
        legacy = {
            "window_rets": [5.0, 5.0, 5.0],
            "window_dds": [-0.1, -0.1, -0.1],
            "window_trades": [5, 5, 5],
            "seed_rets": [5.0] * 6,
            "seed_dds": [-0.1] * 6,
            "n_seeds": 3,
            "seeds_per_window": 3,
            "windows_planned": 3,
            "windows_attempted": 3,
        }
        m = re.build_metrics_from_results(_results(3), rolling=legacy,
                                          budget={"model_seeds": 3})
        self.assertEqual(m["n_seeds"], 0)
        self.assertNotEqual(m["n_seeds"], 3)
        self.assertFalse(m["seed_evidence_known"])
        self.assertFalse(m["seed_evidence_complete"])
        self.assertIn("缺少 window_seed_details", m["seed_evidence_note"])
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("种子证据可追溯" in r for r in v["reasons"]), v["reasons"])

    def test_empty_legacy_rolling_object_is_unknown(self):
        # 明确传入了旧滚动结果对象，但既无逐窗口明细也无结果 → 不能当成“未执行滚动”
        m = re.build_metrics_from_results(_results(3), rolling={},
                                          budget={"model_seeds": 3})
        self.assertEqual(m["n_seeds"], 0)
        self.assertFalse(m["seed_evidence_known"])
        self.assertFalse(m["seed_evidence_complete"])
        self.assertFalse(re.evaluate_promotion(
            m, None, self.R, 0)["promote"])

    def test_skipped_planned_window_blocks_promotion(self):
        roll = _rolling([3, 3], window_rets=[5.0, 5.0])
        roll["window_seed_details"].append({
            "window": 3, "attempted": 0, "succeeded": 0, "failed": 0,
            "failures": [], "skipped": True,
            "skip_reason": "训练或验证样本不足", "seed_rets": [],
        })
        roll["windows_planned"] = 3
        roll["windows_attempted"] = 2
        roll["windows_skipped"] = 1
        roll["n_seeds"] = 0
        m = re.build_metrics_from_results(_results(3), rolling=roll,
                                          budget={"model_seeds": 3})
        self.assertEqual(m["n_seeds"], 0)
        self.assertTrue(m["seed_evidence_known"])
        self.assertFalse(m["seed_evidence_complete"])
        self.assertEqual(m["windows_planned"], 3)
        self.assertEqual(m["windows_attempted"], 2)
        self.assertEqual(m["windows_skipped"], 1)
        self.assertEqual(m["window_seed_details"][2]["skip_reason"],
                         "训练或验证样本不足")
        self.assertIsNone(m["seed_std_pct"])
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])
        self.assertTrue(any("种子证据完整" in r for r in v["reasons"]), v["reasons"])

    def test_missing_planned_window_without_skip_also_blocks(self):
        # 例如滚动验证被中断：只产出了 2 个窗口，但计划是 3 个
        roll = _rolling([3, 3], window_rets=[5.0, 5.0])
        roll["windows_planned"] = 3
        roll["windows_attempted"] = 2
        roll["n_seeds"] = 3
        m = re.build_metrics_from_results(_results(3), rolling=roll,
                                          budget={"model_seeds": 3})
        self.assertEqual(m["n_seeds"], 0)
        self.assertFalse(m["seed_evidence_complete"])
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])

    def test_seed_std_does_not_mix_window_level_differences(self):
        # 每个窗口内种子收益完全相同（种子稳定），但窗口之间水平差异很大。
        # seed_std_pct 必须只反映窗口内种子差异，不能把 1 与 10 的窗口差异混入。
        m = _metrics([3, 3, 3], stability_seed_count=0, seed_returns={
            1: [1.0, 1.0, 1.0],
            2: [10.0, 10.0, 10.0],
            3: [1.0, 1.0, 1.0],
        })
        self.assertEqual(m["seed_std_pct"], 0.0)
        self.assertEqual(m["n_seeds"], 3)

    def test_seed_std_requires_two_seeds_in_every_attempted_window(self):
        # 第 1 个窗口只有 1 个成功种子：不能用其他窗口的种子离散度补足
        m = _metrics([1, 3, 3])
        self.assertEqual(m["n_seeds"], 1)
        self.assertIsNone(m["seed_std_pct"])
        v = re.evaluate_promotion(m, None, self.R, n_experiments=0)
        self.assertFalse(v["promote"])


class TestRollingWindowActualCounting(unittest.TestCase):
    """用假 pandas/假模型/假回测实际跑滚动循环，验证失败种子不会计作成功。"""

    @staticmethod
    def _rows(n=300):
        rows = []
        for i in range(n):
            row = {
                "日期": f"2020-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
                "收盘": 10.0 + i * 0.01,
                "label": 1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0),
            }
            for j in range(4):
                row[f"F{j}"] = ((i + j) % 5) - 2
            rows.append(row)
        return _FakeDataFrame(rows), ["F0", "F1", "F2", "F3"]

    class _Model:
        def __init__(self, n):
            self.n = n

        def predict(self, X):
            return [0] * len(X)

        def predict_proba(self, X):
            return None

    class _Engine:
        n_trades = 3
        equity = [100000.0, 101000.0]

        def result(self):
            return 100000.0, 0.01, -0.02

    class _Gui:
        def predictive_power(self, y_true, y_pred, y_proba=None):
            return 0.5, 0.5, 0.6, True, "fake"

        def run_backtest(self, *args, **kwargs):
            return TestRollingWindowActualCounting._Engine()

    def _run(self, fail_fit=None, fail_backtest=False, min_train_rows=40,
             window_specs=None):
        rows, feats = self._rows()
        calls = {"fit": 0, "bt": 0}

        def fake_fit(gui, X, y, X_es, y_es, params, seed):
            calls["fit"] += 1
            if fail_fit and fail_fit(calls["fit"]):
                raise RuntimeError("模拟训练失败")
            return self._Model(len(X))

        gui = self._Gui()
        orig_fit = re.gui_mod_fit
        orig_bt = gui.run_backtest

        def fake_bt(*args, **kwargs):
            calls["bt"] += 1
            if fail_backtest:
                raise RuntimeError("模拟回测失败")
            return self._Engine()

        re.gui_mod_fit = fake_fit
        gui.run_backtest = fake_bt
        try:
            with mock.patch.dict(sys.modules, {"pandas": _fake_pandas()}):
                return re.rolling_window_metrics(
                    gui, rows, feats,
                    {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1,
                     "eval_metric": "mlogloss"},
                    {"ma_period": 10, "stop_loss": 5.0, "take_profit": 15.0,
                     "ma_break_exit": True},
                    n_windows=3, seeds_per_window=3, lookahead=5,
                    min_train_rows=min_train_rows, window_specs=window_specs)
        finally:
            re.gui_mod_fit = orig_fit
            gui.run_backtest = orig_bt

    def test_all_success_counts_three_per_window(self):
        out = self._run()
        self.assertEqual(out["windows"], 3)
        self.assertEqual(out["seeds_attempted"], 9)
        self.assertEqual(out["seeds_succeeded"], 9)
        self.assertEqual(out["n_seeds"], 3)
        self.assertEqual([d["succeeded"] for d in out["window_seed_details"]],
                         [3, 3, 3])

    def test_two_of_three_success_counts_actual(self):
        out = self._run(fail_fit=lambda n: n % 3 == 0)
        self.assertEqual(out["seeds_attempted"], 9)
        self.assertEqual(out["seeds_succeeded"], 6)
        self.assertEqual(out["n_seeds"], 2)
        self.assertEqual([d["succeeded"] for d in out["window_seed_details"]],
                         [2, 2, 2])
        self.assertTrue(all(d["failed"] == 1 for d in out["window_seed_details"]))

    def test_one_of_three_success_counts_actual(self):
        out = self._run(fail_fit=lambda n: n % 3 != 1)
        self.assertEqual(out["seeds_attempted"], 9)
        self.assertEqual(out["seeds_succeeded"], 3)
        self.assertEqual(out["n_seeds"], 1)
        self.assertEqual([d["succeeded"] for d in out["window_seed_details"]],
                         [1, 1, 1])
        self.assertTrue(all(d["failed"] == 2 for d in out["window_seed_details"]))

    def test_all_failed_seeds_not_counted_success(self):
        out = self._run(fail_fit=lambda n: True)
        self.assertEqual(out["seeds_attempted"], 9)
        self.assertEqual(out["seeds_succeeded"], 0)
        self.assertEqual(out["n_seeds"], 0)
        self.assertEqual(out["windows"], 0)
        self.assertEqual(len(out["window_seed_details"]), 3)
        self.assertTrue(all(d["failures"] for d in out["window_seed_details"]))

    def test_backtest_failure_not_counted_success(self):
        out = self._run(fail_backtest=True)
        self.assertEqual(out["seeds_attempted"], 9)
        self.assertEqual(out["seeds_succeeded"], 0)
        self.assertEqual(out["n_seeds"], 0)
        self.assertTrue(all("回测失败" in " ".join(d["failures"])
                            for d in out["window_seed_details"]))

    def test_insufficient_data_skipped_windows_are_recorded(self):
        out = self._run(min_train_rows=1000)
        self.assertEqual(out["windows_planned"], 3)
        self.assertEqual(out["windows_attempted"], 0)
        self.assertEqual(out["windows_skipped"], 3)
        self.assertEqual(len(out["window_seed_details"]), 3)
        self.assertTrue(all(d["skipped"] for d in out["window_seed_details"]))
        self.assertTrue(all("样本不足" in d["skip_reason"]
                            for d in out["window_seed_details"]))
        self.assertEqual(out["n_seeds"], 0)

        m = re.build_metrics_from_results(_results(3), rolling=out,
                                          budget={"model_seeds": 3})
        self.assertFalse(m["seed_evidence_complete"])
        self.assertEqual(m["windows_skipped"], 3)
        self.assertFalse(re.evaluate_promotion(
            m, None, re.DEFAULT_PROMOTION_RULES, 0)["promote"])

    def test_precomputed_shared_window_specs_are_used(self):
        # 候选和基准应共用同一批日期窗口；这里验证显式 specs 路径
        rows, feats = self._rows()
        with mock.patch.dict(sys.modules, {"pandas": _fake_pandas()}):
            specs = re.build_rolling_window_specs(rows, n_windows=3,
                                                  lookahead=5, min_train_rows=40)
        self.assertEqual(len(specs), 3)
        out = self._run(window_specs=specs)
        self.assertEqual(out["windows"], 3)
        self.assertEqual(out["windows_planned"], 3)
        self.assertEqual(out["window_specs"], specs)
        self.assertEqual(out["n_seeds"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
