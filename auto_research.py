# -*- coding: utf-8 -*-
"""
auto_research.py — 一键自动化研究调度器（统一后台任务 + 阶段化流水线）

供 train_xgb_gui.py 的「一键自动训练」按钮调用。
- 只读行情 + 在数据目录写入本次实验产物；不联网交易、不删历史数据。
- 所有方案（候选特征 / 模型空间 / 策略范围 / 预算 / 入选规则）都在本文件用常量与
  纯函数决定，绝不依赖界面回调，方便追溯与离线测试。
- 严格时间纪律：
    * 用户填写的测试区间【绝不被修改】；
    * 训练段与测试段必须满足时间先后（允许中间有空档），不允许重叠；
    * 训练段 / 测试段各自裁掉末尾 LOOKAHEAD 行，防止标签 look-ahead 跨边界泄漏；
    * 测试段只在最后“独立测试回测”出现一次，绝不参与特征/模型/策略的任何选择。

预算控制（面向 4GB 老电脑）：有限试验次数、n_jobs=1、单条可靠训练链、无死循环。
"""

import os
import csv
import json
import time
import random
import traceback
from datetime import datetime, timedelta


# ====================================================================
# 0. 固定研究预算 / 方案常量（全部由后端决定，界面不可改）
# ====================================================================
CAPITAL = 100000.0            # 初始资金
RISK_CONTROL = True           # 一律启用风控（SL/TP/MA 由寻优决定）

# 真实交易成本（与界面「真实交易成本模拟」一致的默认口径）
FRICTION = True
COMMISSION_PCT = 0.03         # 佣金 %
STAMP_PCT = 0.10              # 印花税 %（仅卖出）
SLIPPAGE_PCT = 0.10           # 滑点 %

# 试验预算（老电脑保护：总训练次数 ≈ MODEL_TRIALS + FEATURE_TRIALS + 少量稳定性）
MODEL_TRIALS = 8              # XGBoost 参数组合试验数（= len(MODEL_SPACE)，取小值封顶）
FEATURE_TRIALS = 12           # 特征组合试验数
STRATEGY_TRIALS = 24          # SL/TP/MA 策略候选数（回测便宜，可多试）
MODEL_SEEDS = 2               # 稳定性验证时重复训练的随机种子数
N_JOBS = 1                    # 只用 1 个线程，照顾老电脑
MIN_FEATURES = 3              # 特征组合最少用几个特征
MAX_FEATURES = 7              # 特征组合最多用几个特征
FEATURE_POOL_MAX = 18         # 初筛后特征池上限

VALID_FRACTION = 0.30         # 训练期尾部作为“优化验证段”的比例
MODEL_ES_FRACTION = 0.15      # 训练段内部再切一小段做早停

MIN_TRAIN_ROWS = 80           # 训练段最少样本
MIN_TEST_ROWS = 15            # 测试段最少样本

LOOKAHEAD = 5                 # 标签未来天数（与主程序一致），用于边界防泄漏
L2N = {-1: 0, 0: 1, 1: 2}     # 标签 -> 类别编号
N2A = {0: -1, 1: 0, 2: 1}     # 类别编号 -> 交易信号


# ====================================================================
# 【标签连续化】XGBoost 要求分类标签必须是从 0 开始的连续整数。
# 某些股票/时间窗口可能缺失中间类别（例如只有 -1 和 +1，没有 0），
# 静态映射会得到 [0, 2]，直接 fit 会报：
#   "Invalid classes inferred from unique values of 'y'. Expected: [0 1], got [0 2]"
# 这里统一在送入模型前按实际出现的类别重新编码为 0..k-1，
# 并在模型预测后自动解码回原来的标签（-1/0/1 对应的 L2N 编号）。
# ====================================================================
def _fit_xgb_contiguous(X, y, X_es, y_es, params, seed, n_jobs=1):
    """训练 XGBClassifier：标签先做连续化编码，预测时自动解码回原标签。"""
    import numpy as np
    from xgboost import XGBClassifier

    y_arr = np.asarray(y).ravel()
    classes = np.unique(y_arr)
    if len(classes) == 0:
        raise ValueError("训练标签为空，无法训练")

    def _encode(arr):
        arr = np.asarray(arr).ravel()
        out = np.zeros(arr.shape, dtype=np.int64)
        for i, c in enumerate(classes):
            out[arr == c] = i
        return out

    y_enc = _encode(y_arr)

    # 早停集若出现训练集中没有的类别 → 过滤掉，避免 XGBoost 标签校验失败
    X_es2, y_es2 = X_es, y_es
    if X_es is not None and y_es is not None and len(X_es) > 0:
        y_es_arr = np.asarray(y_es).ravel()
        mask = np.isin(y_es_arr, classes)
        if not mask.all():
            X_es2 = np.asarray(X_es)[mask]
            y_es_arr = y_es_arr[mask]
        if len(y_es_arr) == 0:
            X_es2, y_es2 = None, None
        else:
            y_es2 = _encode(y_es_arr)

    p = dict(params)
    p["n_jobs"] = n_jobs
    # 多分类目标必须显式给出 num_class；否则当类别数不是 3 时
    # xgboost 可能报 "value 0 for Parameter num_class should be greater equal to 1"
    if str(p.get("objective", "")).startswith("multi"):
        p["num_class"] = len(classes)

    def _make_model():
        return XGBClassifier(random_state=seed, **p)

    model = _make_model()
    try:
        if X_es2 is not None and len(X_es2) > 0:
            model.fit(X, y_enc, eval_set=[(X_es2, y_es2)], verbose=False)
        else:
            p.pop("early_stopping_rounds", None)
            model = _make_model()
            model.fit(X, y_enc, verbose=False)
    except TypeError:
        # 兼容不同 xgboost 版本：早停参数不支持就退回普通训练
        p.pop("early_stopping_rounds", None)
        model = _make_model()
        if X_es2 is not None and len(X_es2) > 0:
            try:
                model.fit(X, y_enc, eval_set=[(X_es2, y_es2)], verbose=False)
            except Exception:
                model.fit(X, y_enc, verbose=False)
        else:
            model.fit(X, y_enc, verbose=False)

    # 保存编码表，并把 predict 包装成“自动解码回原标签”
    model._label_classes_ = classes
    _orig_predict = model.predict

    def _decoded_predict(data, *args, **kwargs):
        raw = np.asarray(_orig_predict(data, *args, **kwargs))
        # 某些 xgboost 版本在二分类 + multi:softprob 下 predict 会返回概率矩阵，
        # 这里统一取 argmax，保证输出一定是类别编号，再做解码。
        if raw.ndim > 1:
            raw = raw.argmax(axis=1)
        return classes[raw.astype(int)]

    model.predict = _decoded_predict
    return model

# 初筛候选特征池：所有指标开关全开（与主程序 make_features 对齐）
ALL_SWITCHES = {
    "MA5": True, "MA10": True, "MA20": True, "MA60": True,
    "RSI": True, "MACD": True, "KDJ": True, "BOLL": True,
    "VOL": True, "RET": True, "ATR": True, "ENH": True,
}

# XGBoost 候选参数空间（8 个由保守到激进、覆盖深度/学习率/正则的配置）
MODEL_SPACE = [
    dict(objective="multi:softprob", n_estimators=300, max_depth=3, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
         early_stopping_rounds=20, eval_metric="mlogloss"),
    dict(objective="multi:softprob", n_estimators=300, max_depth=2, learning_rate=0.05,
         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
         early_stopping_rounds=20, eval_metric="mlogloss"),
    dict(objective="multi:softprob", n_estimators=250, max_depth=4, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
         early_stopping_rounds=20, eval_metric="mlogloss"),
    dict(objective="multi:softprob", n_estimators=500, max_depth=3, learning_rate=0.03,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
         early_stopping_rounds=20, eval_metric="mlogloss"),
    dict(objective="multi:softprob", n_estimators=200, max_depth=3, learning_rate=0.10,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
         early_stopping_rounds=20, eval_metric="mlogloss"),
    dict(objective="multi:softprob", n_estimators=300, max_depth=3, learning_rate=0.05,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0,
         early_stopping_rounds=20, eval_metric="mlogloss"),
    dict(objective="multi:softprob", n_estimators=350, max_depth=4, learning_rate=0.04,
         subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
         early_stopping_rounds=20, eval_metric="mlogloss"),
    dict(objective="multi:softprob", n_estimators=400, max_depth=2, learning_rate=0.08,
         subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
         early_stopping_rounds=20, eval_metric="mlogloss"),
]

# 阶段名（界面阶段指示器使用）
STAGES = [
    "数据获取与质量检查",
    "特征生成与初筛",
    "XGBoost模型参数寻优",
    "特征组合搜索",
    "交易策略参数寻优(SL/TP/MA)",
    "训练期稳定性验证",
    "最终模型训练",
    "独立测试回测",
    "保存与报告生成",
]

# 【预算默认值】——自主研究调度器可对单次实验传入 cfg["budget"] 覆盖，
# 但每次运行都会先恢复成这里的默认值再覆盖，绝不让上一次的临时预算“污染”下一次。
_DEFAULT_BUDGET = {"MODEL_TRIALS": MODEL_TRIALS, "FEATURE_TRIALS": FEATURE_TRIALS,
                   "STRATEGY_TRIALS": STRATEGY_TRIALS, "MODEL_SEEDS": MODEL_SEEDS}


class AutoError(Exception):
    """可预期的研究失败（数据不足 / 日期非法 / 无合格候选等）。"""


def _digits(s):
    return "".join(ch for ch in str(s) if ch.isdigit())


def _make_exp_id(code):
    """生成唯一实验编号：{code}_{YYYYmmdd_HHMMSS}_{hex4}，绝不让实验互相覆盖。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salt = "%04x" % random.randint(0, 0xFFFF)
    return f"{code}_{stamp}_{salt}"


# ====================================================================
# 与主程序兼容的二次定界（train_xgb_gui 仍会调用本函数）
# ====================================================================
def compute_second_ranges(s1, t1, m1):
    """智能二次动态定界（纯计算，不跑回测、不改配置）。"""
    if t1 > 30:
        bias = "greedy"
        tp_hi = 30
        tp_lo = t1 - 10
        if tp_lo >= tp_hi:
            tp_lo = tp_hi - 10
    elif t1 < 15:
        bias = "conservative"
        tp_lo, tp_hi = t1, t1 + 15
    else:
        bias = "healthy"
        tp_lo, tp_hi = t1 - 5, t1 + 5

    if s1 < 5:
        sl_lo, sl_hi = 5, s1 + 5
    elif s1 > 10:
        sl_hi = 10
        sl_lo = s1 - 3
        if sl_lo >= sl_hi:
            sl_lo = sl_hi - 3
    else:
        sl_lo, sl_hi = s1 - 2, s1 + 2

    if m1 <= 20:
        ma_lo, ma_hi = 10, 30
    else:
        ma_lo, ma_hi = 15, 60

    sl_lo = max(1.0, sl_lo)
    if sl_hi <= sl_lo:
        sl_hi = sl_lo + 1
    tp_lo = max(1.0, tp_lo)
    if tp_hi <= tp_lo:
        tp_hi = tp_lo + 1
    ma_lo = max(5, int(round(ma_lo)))
    ma_hi = max(ma_lo + 1, int(round(ma_hi)))

    width_score = ((sl_hi - sl_lo) / 12.0 + (tp_hi - tp_lo) / 30.0 + (ma_hi - ma_lo) / 60.0) / 3.0
    width_score = max(0.0, min(1.0, width_score))
    dev = max(0.0, min(1.0, abs(t1 - 22.5) / 15.0))
    score = 0.5 * width_score + 0.5 * dev

    MAX_SEARCH_TRIALS = 200
    if bias == "conservative":
        trials = int(round(150 + 50 * score))
        trials = max(150, min(MAX_SEARCH_TRIALS, trials))
    elif bias == "greedy":
        trials = int(round(100 + 20 * score))
        trials = max(100, min(120, trials))
    else:
        trials = 150
    return (sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi, trials, bias)


# ====================================================================
# 调度器
# ====================================================================
class AutoRunner:
    def __init__(self, cfg, emit, stop_fn=None, state_dir=None):
        """
        cfg: dict —— code, train_start, train_end, test_start, test_end, ai_name
        emit: callable(kind, *args) —— 线程安全投递到界面队列
              kinds: log/stage/progress/status/busy/summary/box/refresh_history/done
        stop_fn: callable -> bool（True 表示请求停止）
        state_dir: 数据目录（默认取主程序 DATA_DIR）
        """
        self.cfg = dict(cfg or {})
        self.emit = emit or (lambda *a, **k: None)
        self.stop_fn = stop_fn or (lambda: False)
        self.results = {}
        self.exp_id = None
        self.exp_dir = None
        self._log_fh = None
        self.last_error = ""          # 供自主研究调度器读取的中止原因（不伪造成功）
        self.state_dir = state_dir
        self._test_registry_obj = None
        self._final_test_claim = None

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _l(self, msg, tag=None):
        try:
            self.emit("log", str(msg), tag)
        except Exception:
            pass
        try:
            if self._log_fh is not None:
                self._log_fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
                self._log_fh.flush()
        except Exception:
            pass

    def _stage(self, idx, status):
        try:
            self.emit("stage", int(idx), str(status))
        except Exception:
            pass

    def _progress(self, cur, total, label=""):
        try:
            self.emit("progress", int(cur), int(total), str(label))
        except Exception:
            pass

    def _status(self, text):
        try:
            self.emit("status", str(text))
        except Exception:
            pass

    def _check_stop(self):
        try:
            if self.stop_fn():
                raise AutoError("用户已请求停止研究")
        except AutoError:
            raise
        except Exception:
            pass

    def _gui(self):
        import train_xgb_gui as gui
        return gui

    # ------------------------------------------------------------------
    # 【预算覆盖】自主研究调度器可为单次实验指定预算；
    # 每次运行前先恢复默认预算，再应用本次覆盖，避免互相污染。
    # ------------------------------------------------------------------
    def _apply_budget(self):
        g = globals()
        for k, v in _DEFAULT_BUDGET.items():
            g[k] = v
        b = self.cfg.get("budget") or {}
        mapping = {"model_trials": "MODEL_TRIALS", "feature_trials": "FEATURE_TRIALS",
                   "strategy_trials": "STRATEGY_TRIALS", "model_seeds": "MODEL_SEEDS"}
        for src, dst in mapping.items():
            if src in b:
                try:
                    g[dst] = max(1, int(b[src]))
                except Exception:
                    pass

    def _ordered_model_space(self):
        """按 cfg["param_priority"]（自主研究给出的“未试过优先”顺序）排列候选参数。"""
        n = len(MODEL_SPACE)
        prio = self.cfg.get("param_priority") or []
        order = []
        for i in prio:
            try:
                i = int(i)
            except Exception:
                continue
            if 0 <= i < n and i not in order:
                order.append(i)
        for i in range(n):
            if i not in order:
                order.append(i)
        return [MODEL_SPACE[i] for i in order]

    def _skip_final_test(self):
        return bool(self.cfg.get("skip_final_test", False))

    # ------------------------------------------------------------------
    # 最终测试使用登记（持久化，手工/自动共享）
    # ------------------------------------------------------------------
    def _test_registry(self):
        if self._test_registry_obj is None:
            from stock_pool import ResearchRegistry
            data_dir = self.state_dir or getattr(self._gui(), "DATA_DIR", os.path.expanduser("~"))
            self._test_registry_obj = ResearchRegistry(
                os.path.join(data_dir, "research_queue.db"))
        return self._test_registry_obj

    def _final_test_block_reason(self):
        """只读预检：测试区间已消费/固定窗口不匹配时返回原因；不领取。"""
        code = _digits(self.cfg.get("code", "")) or str(self.cfg.get("code", ""))
        test_start = str(self.results.get("test_start") or
                         self.cfg.get("_test_start") or self.cfg.get("test_start") or "")
        test_end = str(self.results.get("test_end") or
                       self.cfg.get("_test_end") or self.cfg.get("test_end") or "")
        try:
            summary = self._test_registry().test_usage_summary(
                code, test_start, test_end)
        except Exception as e:      # noqa: BLE001
            return f"最终测试登记器不可用：{e}"
        status = str(summary.get("status") or "")
        rec = summary.get("record") or {}
        if status == "window_locked":
            return str(rec.get("note") or "测试区间与已固定窗口不一致")
        if status == "blocking":
            return (f"完整测试区间已被使用（状态={rec.get('status')}，"
                    f"实验={rec.get('exp_id') or '-'}，"
                    f"来源={rec.get('trigger') or '-'}），禁止重复消费")
        return ""

    def _claim_final_test_usage(self):
        """在真正执行最终测试前领取使用权；失败/已使用则不允许重复消费。"""
        code = _digits(self.cfg.get("code", "")) or str(self.cfg.get("code", ""))
        test_start = str(self.results.get("test_start") or
                         self.cfg.get("_test_start") or self.cfg.get("test_start") or "")
        test_end = str(self.results.get("test_end") or
                       self.cfg.get("_test_end") or self.cfg.get("test_end") or "")
        exp_id = str(self.exp_id or "")
        trigger = str(self.cfg.get("trigger") or "手动训练")
        try:
            reg = self._test_registry()
            ok, record = reg.claim_final_test(
                code, test_start, test_end, exp_id=exp_id, trigger=trigger)
        except Exception as e:      # noqa: BLE001
            return False, f"最终测试登记器不可用：{e}"
        if not ok:
            status = str((record or {}).get("status") or "unknown")
            old_exp = str((record or {}).get("exp_id") or "")
            old_trigger = str((record or {}).get("trigger") or "")
            if status == "window_locked":
                return False, (str((record or {}).get("note") or
                                   "测试区间与已固定窗口不一致")
                               + f"，禁止通过改日期绕过登记")
            return False, (f"完整测试区间已被使用（状态={status}，实验={old_exp or '-'}，"
                           f"来源={old_trigger or '-'}），禁止重复消费")
        self._final_test_claim = {
            "code": code, "test_start": test_start, "test_end": test_end,
            "exp_id": exp_id, "trigger": trigger,
        }
        return True, ""

    def _mark_final_test_usage(self, status, note=""):
        claim = self._final_test_claim
        if not claim:
            return
        try:
            reg = self._test_registry()
            reg.mark_final_test(
                claim["code"], claim["test_start"], claim["test_end"],
                exp_id=claim["exp_id"], status=status, note=note)
        except Exception:       # noqa: BLE001
            pass
        finally:
            if status in ("completed", "failed", "aborted"):
                self._final_test_claim = None

    def _close_test_registry(self):
        try:
            if self._test_registry_obj is not None:
                self._test_registry_obj.close()
        except Exception:       # noqa: BLE001
            pass
        self._test_registry_obj = None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self):
        gui = self._gui()
        t0 = time.time()
        self._apply_budget()
        skip_test = self._skip_final_test()
        if skip_test:
            self._l("自主研究模式：本次实验【不执行】独立测试（测试集只用于预先规定的最终评估）。",
                    "dim")
        try:
            self.emit("busy", True)
            self.emit("log", "正在执行数据检查...", "dim")
            self._s1_data(gui)
            self.emit("log", "正在执行指标计算与特征生成...", "dim")
            self._s2_features(gui)
            self.emit("log", "正在执行模型参数搜索...", "dim")
            self._s3_model_search(gui)
            self.emit("log", "正在执行特征组合搜索...", "dim")
            self._s4_feature_search(gui)
            self.emit("log", "正在执行交易策略研究...", "dim")
            self._s5_strategy_search(gui)
            self.emit("log", "正在执行训练期稳定性验证...", "dim")
            self._s6_stability(gui)
            self.emit("log", "正在训练最终模型...", "dim")
            self._s7_final_train(gui)
            if skip_test:
                self.results["test_skipped"] = True
                self.results["test_skip_reason"] = "自主研究模式：独立测试未执行"
                self._l("已跳过独立测试（自主研究只用训练期滚动验证做晋级判断）。", "dim")
            else:
                self.emit("log", "正在执行独立测试回测...", "dim")
                self._s8_final_test(gui)
            self.emit("log", "正在汇总报告...", "dim")
            self._s9_report(gui)
            self._l(f"一键自动训练全部完成，用时 {time.time() - t0:.1f}s", "title")
            return True, self.results
        except AutoError as e:
            self.last_error = str(e)
            self._mark_final_test_usage("failed", f"研究中止：{e}")
            self._l(f"研究中止：{e}", "red")
            # 尽力留下已产出的报告/历史（不伪造成功）
            try:
                self._s9_report(gui, aborted=str(e))
            except Exception:
                pass
            return False, self.results
        except Exception as e:
            self.last_error = str(e)
            self._mark_final_test_usage("failed", f"研究异常：{e}")
            self._l(f"研究异常：{e}\n{traceback.format_exc()}", "red")
            try:
                self._s9_report(gui, aborted=f"异常：{e}")
            except Exception:
                pass
            return False, self.results
        finally:
            try:
                self.emit("busy", False)
            except Exception:
                pass
            try:
                if self._log_fh:
                    self._log_fh.close()
                    self._log_fh = None
            except Exception:
                pass
            self._close_test_registry()

    # ==================================================================
    # 阶段 1：数据获取与质量检查
    # ==================================================================
    def _s1_data(self, gui):
        import pandas as pd

        self._stage(1, "running")
        self._status("数据获取与质量检查 ...")

        code = _digits(self.cfg.get("code", ""))
        if len(code) != 6:
            raise AutoError(f"股票代码必须为 6 位数字（当前：{self.cfg.get('code')!r}）")
        self.cfg["code"] = code

        def _pd(s):
            try:
                return datetime.strptime(str(s).strip(), "%Y-%m-%d")
            except Exception:
                raise AutoError(f"日期格式非法（应为 YYYY-MM-DD）：{s!r}")

        train_start = _pd(self.cfg.get("train_start"))
        train_end = _pd(self.cfg.get("train_end"))
        test_start = _pd(self.cfg.get("test_start"))
        test_end = _pd(self.cfg.get("test_end"))

        # 时间先后与重叠校验（用户填写的测试日期绝不修改）
        if not (train_start < train_end):
            raise AutoError("训练开始日期必须早于训练结束日期")
        if not (test_start <= test_end):
            raise AutoError("测试开始日期不能晚于测试结束日期")
        if not (train_end < test_start):
            raise AutoError(
                "训练结束日期必须早于测试开始日期（允许中间留空档，但不允许重叠）")

        self._l(f"日期校验通过：训练 {train_start.date()} ~ {train_end.date()} | "
                f"测试 {test_start.date()} ~ {test_end.date()}", "green")
        if (test_start - train_end).days > 400:
            self._l(f"提示：训练结束与测试开始之间空档 {(test_start - train_end).days} 天，"
                    f"属正常时间先后关系。", "dim")

        # 训练起始日期对齐上市首日（仅在不改变测试区间的前提下，向前取可用数据）
        try:
            first = gui.get_first_data_date(code)
        except Exception:
            first = None
        if first:
            try:
                first_d = datetime.strptime(str(first), "%Y-%m-%d")
                if train_start < first_d:
                    self._l(f"训练起始日期早于上市首日 {first}，已对齐至上市首日"
                            f"（测试区间未改动）", "yellow")
                    train_start = first_d
            except Exception:
                pass
        if not (train_start < train_end):
            raise AutoError("训练区间在上市后仍无有效数据，请调整训练日期")

        # 取数：训练入参前留 150 天预热（计算长周期均线），上界覆盖用户测试终止日
        warmup = (train_start - timedelta(days=150)).strftime("%Y%m%d")
        fetch_end = max(test_end, datetime.now()).strftime("%Y%m%d")
        self._l(f"正在获取行情：{code}  {warmup} ~ {fetch_end} ...")
        try:
            raw = gui.fetch_stock_data(code, warmup, fetch_end)
        except Exception as e:
            raise AutoError(f"行情获取失败：{e}")
        if raw is None or len(raw) < 30:
            raise AutoError("未获取到足够的日线数据，请检查股票代码或网络")

        raw = raw.copy()
        raw["日期"] = pd.to_datetime(raw["日期"])
        raw = raw.sort_values("日期").reset_index(drop=True)
        self._l(f"行情数据 {len(raw)} 条，范围 {raw['日期'].min().date()} ~ {raw['日期'].max().date()}")

        tr_mask = (raw["日期"] >= train_start) & (raw["日期"] <= train_end)
        te_mask = (raw["日期"] >= test_start) & (raw["日期"] <= test_end)
        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        self._l(f"可用样本：训练段 {n_tr} 行 | 测试段 {n_te} 行")
        if n_tr < MIN_TRAIN_ROWS:
            raise AutoError(f"训练段可用样本不足（{n_tr} < {MIN_TRAIN_ROWS}），请放宽训练区间")
        if n_te < MIN_TEST_ROWS and not self._skip_final_test():
            raise AutoError(f"测试段可用样本不足（{n_te} < {MIN_TEST_ROWS}），请放宽测试区间")
        if raw["日期"].max() < test_end:
            self._l(f"提示：行情最后一天 {raw['日期'].max().date()} 早于测试结束 "
                    f"{test_end.date()}，测试按实际可得数据评估。", "yellow")

        self.cfg["_train_start"] = train_start.strftime("%Y-%m-%d")
        self.cfg["_train_end"] = train_end.strftime("%Y-%m-%d")
        self.cfg["_test_start"] = test_start.strftime("%Y-%m-%d")
        self.cfg["_test_end"] = test_end.strftime("%Y-%m-%d")

        self.results["raw"] = raw
        self.results["train_start"] = self.cfg["_train_start"]
        self.results["train_end"] = self.cfg["_train_end"]
        self.results["test_start"] = self.cfg["_test_start"]
        self.results["test_end"] = self.cfg["_test_end"]

        # 建立实验目录 + 运行日志
        gui_mod = gui
        data_dir = getattr(gui_mod, "DATA_DIR", os.path.expanduser("~"))
        self.exp_id = _make_exp_id(code)
        self.exp_dir = os.path.join(data_dir, "experiments", self.exp_id)
        try:
            os.makedirs(self.exp_dir, exist_ok=True)
            self._log_fh = open(os.path.join(self.exp_dir, "run.log"), "a", encoding="utf-8")
            self._log_fh.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] exp={self.exp_id}\n")
            self._log_fh.write(f"cfg={json.dumps({k: str(v) for k, v in self.cfg.items()}, ensure_ascii=False)}\n")
            self._log_fh.flush()
            self._l(f"实验目录：{self.exp_dir}", "dim")
        except Exception as e:
            self._l(f"创建实验目录失败（仍继续，结论可能无法落盘）：{e}", "red")

        self.results["exp_id"] = self.exp_id
        self._stage(1, "done")

    # ==================================================================
    # 阶段 2：特征生成与初筛
    # ==================================================================
    def _s2_features(self, gui):
        import pandas as pd

        self._stage(2, "running")
        self._status("特征生成与初筛 ...")

        df = gui.add_indicators(self.results["raw"].copy())
        df = gui.build_labels(df)
        self.results["df"] = df

        X_all, names = gui.make_features(df, dict(ALL_SWITCHES))
        self.results["X_all"] = X_all
        self.results["feat_universe"] = list(names)
        self._l(f"候选特征 {len(names)} 个：{', '.join(names)}")

        # 行情 + 特征合并矩阵（后续训练 / 测试都从这里按日期切片，保证列一致）
        matrix_all = pd.concat([df[["日期", "收盘", "label"]], X_all], axis=1)
        self.results["matrix_all"] = matrix_all

        tr_mask = ((df["日期"] >= pd.to_datetime(self.results["train_start"]))
                   & (df["日期"] <= pd.to_datetime(self.results["train_end"])))
        train_rows = matrix_all[tr_mask].copy()

        kept, elim = [], []
        for c in names:
            if c not in train_rows.columns:
                elim.append((c, "列缺失"))
                continue
            s = train_rows[c].dropna()
            if len(s) < 20:
                elim.append((c, "样本不足"))
                continue
            if float(s.std()) == 0.0:
                elim.append((c, "常数"))
                continue
            if train_rows[c].isna().mean() > 0.30:
                elim.append((c, f"缺失率>30%"))
                continue
            kept.append(c)

        uniq = []
        for c in kept:
            dup = None
            for uc in uniq:
                try:
                    corr = train_rows[c].corr(train_rows[uc])
                    if abs(corr) > 0.95:
                        dup = (c, f"与{uc}强相关({corr:.2f})")
                        break
                except Exception:
                    pass
            if dup:
                elim.append(dup)
            else:
                uniq.append(c)

        self.results["feat_pool"] = uniq[:FEATURE_POOL_MAX]
        self._l(f"特征初筛：入选 {len(self.results['feat_pool'])} 个，淘汰 {len(elim)} 条")
        for name, why in elim[:12]:
            self._l(f"  淘汰 {name}：{why}", "dim")
        if len(self.results["feat_pool"]) < MIN_FEATURES:
            raise AutoError(f"初筛后有效特征不足（{len(self.results['feat_pool'])} < {MIN_FEATURES}）")

        self.results["train_rows_all"] = train_rows
        self._stage(2, "done")

    # ------------------------------------------------------------------
    # 训练 / 评估工具
    # ------------------------------------------------------------------
    def _prep_train_matrix(self, gui, feat_names):
        """构造训练期矩阵，并裁掉末尾 LOOKAHEAD 行，防止标签跨边界 look-ahead。"""
        rows = self.results["train_rows_all"].copy()
        if len(rows) > LOOKAHEAD:
            rows = rows.iloc[:-LOOKAHEAD].reset_index(drop=True)
        rows = rows.dropna(subset=list(feat_names) + ["label"]).reset_index(drop=True)
        return rows

    def _fit_model(self, X, y, X_es, y_es, params, seed):
        # 统一走连续标签编码：缺失中间类别也不会再触发 XGBoost Invalid classes
        return _fit_xgb_contiguous(X, y, X_es, y_es, params, seed, n_jobs=N_JOBS)

    def _split_valid(self, rows, feat_names):
        n = len(rows)
        cut = int(n * (1.0 - VALID_FRACTION))
        fit = rows.iloc[:cut]
        val = rows.iloc[cut:]
        return fit, val

    # ==================================================================
    # 阶段 3：XGBoost 模型参数寻优
    # ==================================================================
    def _s3_model_search(self, gui):
        self._stage(3, "running")
        self._status("XGBoost 模型参数寻优 ...")

        feat_names = list(self.results["feat_pool"])
        rows = self._prep_train_matrix(gui, feat_names)
        if len(rows) < MIN_TRAIN_ROWS:
            raise AutoError(f"裁剪边界后训练样本不足（{len(rows)} < {MIN_TRAIN_ROWS}）")
        fit, val = self._split_valid(rows, feat_names)
        # 早停用 fit 内部再切一小段，避免污染验证段
        cut2 = int(len(fit) * (1.0 - MODEL_ES_FRACTION))
        fit_core, es = fit.iloc[:cut2], fit.iloc[cut2:]

        Xf = fit_core[feat_names].values
        yf = fit_core["label"].map(L2N).values
        Xe = es[feat_names].values if len(es) else None
        ye = es["label"].map(L2N).values if len(es) else None
        Xv = val[feat_names].values
        yv = val["label"].map(L2N).values

        self._l(f"训练矩阵：fit核心 {len(fit_core)} | 早停 {len(es)} | 验证 {len(val)} | "
                f"特征 {len(feat_names)} 个")

        space = self._ordered_model_space()[:max(1, MODEL_TRIALS)]
        best = None
        log_rows = []
        for i, params in enumerate(space, 1):
            self._check_stop()
            self._progress(i, len(space), f"模型参数 {i}/{len(space)}")
            seed = 20260101 + i * 137
            try:
                model = self._fit_model(Xf, yf, Xe, ye, params, seed)
            except Exception as e:
                self._l(f"  第{i}组参数训练失败：{e}", "yellow")
                continue
            pred = model.predict(Xv)
            try:
                proba = model.predict_proba(Xv)
            except Exception:
                proba = None
            acc, majority, auc, has_edge, note = gui.predictive_power(yv, pred, proba)
            auc_v = auc if auc == auc else 0.0
            # 评分：AUC 为主，准确率相对多数类基准的边缘为辅
            score = auc_v + 0.5 * (acc - majority)
            tag = (params.get("max_depth"), params.get("learning_rate"),
                   params.get("n_estimators"), params.get("subsample"),
                   params.get("colsample_bytree"), params.get("reg_lambda"))
            self._l(f"  第{i}组 AUC={auc_v:.3f} 准确率={acc:.3f} 边缘={acc - majority:+.3f} "
                    f"score={score:.3f}  {note}")
            log_rows.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i,
                             json.dumps(tag, ensure_ascii=False), f"{acc:.4f}",
                             f"{auc_v:.4f}", f"{score:.4f}", note])
            if best is None or score > best["score"]:
                best = {"idx": i, "params": dict(params), "score": score,
                        "auc": auc_v, "acc": acc, "note": note, "seed": seed}

        if best is None:
            raise AutoError("所有 XGBoost 参数组合均训练失败")
        self.results["best_model_params"] = best["params"]
        self.results["model_search_best"] = best
        self.results["model_search_rows"] = log_rows
        self._write_csv(os.path.join(self._safe_exp_dir(), "model_search.csv"),
                        ["时间", "序号", "参数摘要", "准确率", "AUC", "得分", "说明"], log_rows)
        self._l(f"最佳模型参数：{best['params']}（AUC={best['auc']:.3f}, score={best['score']:.3f}）", "green")
        self._stage(3, "done")

    # ==================================================================
    # 阶段 4：特征组合搜索
    # ==================================================================
    def _s4_feature_search(self, gui):
        self._stage(4, "running")
        self._status("特征组合搜索 ...")

        pool = list(self.results["feat_pool"])
        params = self.results["best_model_params"]
        rng = random.Random(20260202)

        # 候选组合：先放几个锚点（全量池 / 前若干个），再随机补足
        combos, seen = [], set()

        def _add(cols):
            cols = tuple(sorted(set(cols)))
            if MIN_FEATURES <= len(cols) <= MAX_FEATURES and cols not in seen:
                seen.add(cols)
                combos.append(list(cols))

        _add(pool[:max(MIN_FEATURES, min(MAX_FEATURES, len(pool)))])
        _add(pool[:MIN_FEATURES])
        while len(combos) < FEATURE_TRIALS:
            k = rng.randint(MIN_FEATURES, min(MAX_FEATURES, len(pool)))
            _add(rng.sample(pool, k))
            if len(seen) >= _n_choose(len(pool), MIN_FEATURES):
                break

        best = None
        log_rows = []
        for i, cols in enumerate(combos, 1):
            self._check_stop()
            self._progress(i, len(combos), f"特征组合 {i}/{len(combos)}")
            rows = self._prep_train_matrix(gui, cols)
            if len(rows) < MIN_TRAIN_ROWS:
                continue
            fit, val = self._split_valid(rows, cols)
            cut2 = int(len(fit) * (1.0 - MODEL_ES_FRACTION))
            fit_core, es = fit.iloc[:cut2], fit.iloc[cut2:]
            try:
                model = self._fit_model(fit_core[cols].values, fit_core["label"].map(L2N).values,
                                        es[cols].values if len(es) else None,
                                        es["label"].map(L2N).values if len(es) else None,
                                        params, 777 + i)
                pred = model.predict(val[cols].values)
                proba = model.predict_proba(val[cols].values)
            except Exception as e:
                self._l(f"  组合{i} 训练失败：{e}", "yellow")
                continue
            acc, majority, auc, has_edge, note = gui.predictive_power(
                val["label"].map(L2N).values, pred, proba)
            auc_v = auc if auc == auc else 0.0
            score = auc_v + 0.5 * (acc - majority) - 0.01 * len(cols)  # 轻微惩罚特征数，防过拟合
            self._l(f"  组合{i}（{len(cols)}特征）AUC={auc_v:.3f} 准确率={acc:.3f} "
                    f"score={score:.3f}")
            log_rows.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i,
                             ",".join(cols), f"{acc:.4f}", f"{auc_v:.4f}",
                             f"{score:.4f}", note])
            if best is None or score > best["score"]:
                best = {"cols": list(cols), "score": score, "auc": auc_v, "acc": acc, "note": note}

        if best is None:
            raise AutoError("特征组合搜索未得到任何可用组合")
        self.results["best_features"] = best["cols"]
        self.results["feature_search_best"] = best
        self.results["feature_search_rows"] = log_rows
        self._write_csv(os.path.join(self._safe_exp_dir(), "feature_search.csv"),
                        ["时间", "序号", "特征组合", "准确率", "AUC", "得分", "说明"], log_rows)
        self._l(f"最佳特征组合（{len(best['cols'])}个）：{', '.join(best['cols'])}"
                f"（AUC={best['auc']:.3f}）", "green")
        self._stage(4, "done")

    # ==================================================================
    # 阶段 5：交易策略参数寻优（SL/TP/MA）
    # ==================================================================
    def _build_val_predictions(self, gui, feat_names, params, seed):
        """在训练期验证段上产出预测信号，供策略回测使用。"""
        rows = self._prep_train_matrix(gui, feat_names)
        fit, val = self._split_valid(rows, feat_names)
        cut2 = int(len(fit) * (1.0 - MODEL_ES_FRACTION))
        fit_core, es = fit.iloc[:cut2], fit.iloc[cut2:]
        model = self._fit_model(fit_core[feat_names].values, fit_core["label"].map(L2N).values,
                                es[feat_names].values if len(es) else None,
                                es["label"].map(L2N).values if len(es) else None,
                                params, seed)
        pred = model.predict(val[feat_names].values)
        return model, val, pred

    def _s5_strategy_search(self, gui):
        import pandas as pd

        self._stage(5, "running")
        self._status("交易策略参数寻优 ...")
        feat_names = self.results["best_features"]
        params = self.results["best_model_params"]

        # 在“训练期验证段”上得到信号（绝不碰测试段）
        model, val, pred = self._build_val_predictions(gui, feat_names, params, 4242)
        self.results["val_model"] = model
        self.results["val_pred"] = pred

        actions = pd.Series(pred).map(N2A).values
        dates = list(val["日期"])
        closes = list(val["收盘"])

        # 用整段行情计算策略范围（recommend_ranges 依赖 ATR / 波段统计）
        df_full = self.results["df"]
        try:
            sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi = gui.recommend_ranges(
                df_full[df_full["日期"] <= pd.to_datetime(self.results["train_end"])])
        except Exception as e:
            self._l(f"范围推荐失败，使用稳健默认值：{e}", "yellow")
            sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi = 6, 10, 20, 30, 10, 40
        self._l(f"策略搜索范围：止损 {sl_lo}-{sl_hi}% | 止盈 {tp_lo}-{tp_hi}% | "
                f"均线 {ma_lo}-{ma_hi}")

        anchors = []
        for sl in (sl_lo, round((sl_lo + sl_hi) / 2, 1), sl_hi):
            for tp in (tp_lo, round((tp_lo + tp_hi) / 2, 1), tp_hi):
                anchors.append((sl, tp, max(10, int(round((ma_lo + ma_hi) / 2))), 1))
        cands = gui.sample_strategy_candidates(sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi,
                                               STRATEGY_TRIALS, anchors=anchors, seed=20260303)

        # 预先缓存每种均线周期在验证段上的均线值（因果：仅用过去收盘价 rolling）
        ma_cache = {}

        qualified, all_rows = [], []
        best = None
        for i, (sl, tp, ma, mb) in enumerate(cands, 1):
            self._check_stop()
            self._progress(i, len(cands), f"策略 {i}/{len(cands)}  SL{sl} TP{tp} MA{int(ma)}")
            ma_period = int(ma)
            if ma_period not in ma_cache:
                ma_cache[ma_period] = list(val["收盘"].rolling(ma_period).mean())
            eng = gui.run_backtest(
                CAPITAL, RISK_CONTROL, float(sl), float(tp), ma_period,
                FRICTION, COMMISSION_PCT / 100.0, STAMP_PCT / 100.0, SLIPPAGE_PCT / 100.0,
                dates, closes, ma_cache[ma_period], list(actions), ma_break_exit=bool(mb))
            final, ret, dd = eng.result()
            ret_pct = ret * 100.0
            n_trades = eng.n_trades
            ok, reason = gui.is_strategy_qualified(ret_pct, dd, n_trades)
            score = ret_pct - 30.0 * abs(dd) - 1.0 * max(0, 4 - n_trades)
            all_rows.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i,
                             sl, tp, ma_period, int(bool(mb)), f"{ret_pct:.2f}",
                             f"{dd * 100:.2f}", n_trades, f"{score:.2f}",
                             "合格" if ok else reason])
            if ok:
                qualified.append((score, sl, tp, ma_period, bool(mb), ret_pct, dd, n_trades))
                if best is None or score > best[0]:
                    best = (score, sl, tp, ma_period, bool(mb), ret_pct, dd, n_trades)

        self.results["strategy_rows"] = all_rows
        self._write_csv(os.path.join(self._safe_exp_dir(), "strategy_search.csv"),
                        ["时间", "序号", "止损%", "止盈%", "均线", "跌破均线卖出",
                         "验证段收益%", "最大回撤%", "交易次数", "得分", "是否合格"], all_rows)

        # 只把“合格且得分最高”的候选交给下一阶段；不合格的位置允许全部不合格
        if best is None:
            self._l("所有策略候选均未通过可靠性检查（交易次数/回撤/净收益）。", "red")
            self.results["status"] = "no_qualified"
            self.results["no_qualified_reason"] = "训练期验证段无任何合格候选策略"
            raise AutoError("暂无合格候选模型")
        self.results["best_strategy"] = {
            "stop_loss": float(best[1]), "take_profit": float(best[2]),
            "ma_period": int(best[3]), "ma_break_exit": bool(best[4]),
            "val_ret_pct": float(best[5]), "val_dd": float(best[6]),
            "val_trades": int(best[7]), "score": float(best[0]),
        }
        self._l(f"最佳合格策略：止损{best[1]}% 止盈{best[2]}% MA{int(best[3])} "
                f"跌破均线卖出={'开' if best[4] else '关'} | "
                f"验证段收益 {best[5]:+.2f}% 回撤 {best[6] * 100:.1f}% 交易 {best[7]} 次", "green")
        self._stage(5, "done")

    # ==================================================================
    # 阶段 6：训练期稳定性验证
    # ==================================================================
    def _s6_stability(self, gui):
        import pandas as pd
        import statistics

        self._stage(6, "running")
        self._status("训练期稳定性验证 ...")
        feat_names = self.results["best_features"]
        params = self.results["best_model_params"]
        strat = self.results["best_strategy"]
        ma_period = strat["ma_period"]

        rets, dds, trades = [], [], []
        for s in range(MODEL_SEEDS):
            self._check_stop()
            self._progress(s + 1, MODEL_SEEDS, f"稳定性验证 {s + 1}/{MODEL_SEEDS}")
            model, val, pred = self._build_val_predictions(gui, feat_names, params, 9000 + s * 31)
            ma_ser = list(val["收盘"].rolling(ma_period).mean())
            eng = gui.run_backtest(
                CAPITAL, RISK_CONTROL, strat["stop_loss"], strat["take_profit"], ma_period,
                FRICTION, COMMISSION_PCT / 100.0, STAMP_PCT / 100.0, SLIPPAGE_PCT / 100.0,
                list(val["日期"]), list(val["收盘"]), ma_ser,
                list(pd.Series(pred).map(N2A).values), ma_break_exit=strat["ma_break_exit"])
            _, ret, dd = eng.result()
            rets.append(ret * 100.0)
            dds.append(dd * 100.0)
            trades.append(eng.n_trades)

        mean_ret = sum(rets) / len(rets)
        std_ret = statistics.pstdev(rets) if len(rets) > 1 else 0.0
        stability = mean_ret - 0.5 * std_ret        # 稳定性得分（越高越稳）
        self.results["stability"] = {
            "seed_rets": rets, "seed_dds": dds, "seed_trades": trades,
            "mean_ret": mean_ret, "std_ret": std_ret, "score": stability,
            "mean_trades": (sum(trades) / len(trades)) if trades else 0,
        }
        self._l(f"稳定性：{MODEL_SEEDS} 个种子 平均收益 {mean_ret:+.2f}% "
                f"标准差 {std_ret:.2f}% 平均交易 {self.results['stability']['mean_trades']:.1f} 次",
                "green")

        # 若多种子平均收益非正，视为不稳定 → 结束（不强行选好看的模型）
        ok, reason = gui.is_strategy_qualified(mean_ret, min(dds) / 100.0 if dds else 0.0,
                                               int(self.results["stability"]["mean_trades"]))
        if not ok:
            self._l(f"稳定性复核未通过：{reason}", "red")
            self.results["status"] = "no_qualified"
            self.results["no_qualified_reason"] = f"稳定性复核未通过：{reason}"
            raise AutoError("暂无合格候选模型")
        self.results["status"] = "qualified"
        self._stage(6, "done")

    # ==================================================================
    # 阶段 7：最终模型训练（用完整训练期）
    # ==================================================================
    def _s7_final_train(self, gui):
        self._stage(7, "running")
        self._status("最终模型训练 ...")
        feat_names = self.results["best_features"]
        params = self.results["best_model_params"]

        rows = self._prep_train_matrix(gui, feat_names)
        if len(rows) < MIN_TRAIN_ROWS:
            raise AutoError("最终训练样本不足")
        self._progress(1, 1, "训练最终模型")
        model = self._fit_model(rows[feat_names].values, rows["label"].map(L2N).values,
                                None, None, params, 20260606)
        self.results["final_model"] = model
        self._l(f"最终模型训练完成：样本 {len(rows)} 行，特征 {len(feat_names)} 个", "green")
        self._stage(7, "done")

    # ==================================================================
    # 阶段 8：独立测试回测（测试段只在此出现一次）
    # ==================================================================
    def _s8_final_test(self, gui):
        self._stage(8, "running")
        self._status("独立测试回测 ...")
        pre_reason = self._final_test_block_reason()
        if pre_reason:
            self.results["test_skipped"] = True
            self.results["test_skip_reason"] = pre_reason
            self._l(f"独立测试未执行：{pre_reason}", "yellow")
            self._stage(8, "done")
            return
        import pandas as pd

        matrix = self.results["matrix_all"]
        feat_names = self.results["best_features"]
        model = self.results["final_model"]
        strat = self.results["best_strategy"]

        te_mask = ((matrix["日期"] >= pd.to_datetime(self.results["test_start"]))
                   & (matrix["日期"] <= pd.to_datetime(self.results["test_end"])))
        test_rows = matrix[te_mask].copy()
        if len(test_rows) == 0:
            raise AutoError("测试段没有数据")

        # 标签评估：裁掉末尾 LOOKAHEAD 行，避免用到测试边界之外的未来数据
        te_label = test_rows.dropna(subset=feat_names + ["label"])
        if len(te_label) > LOOKAHEAD:
            te_label = te_label.iloc[:-LOOKAHEAD]

        # 交易回测：用整段测试数据（因果 T+1），均线用测试段 rolling
        ma_period = strat["ma_period"]
        bt_rows = test_rows.dropna(subset=feat_names).reset_index(drop=True)
        if len(bt_rows) == 0:
            raise AutoError("测试段特征全为空，无法回测")
        # 数据已确认可评估，才正式领取最终测试；此前的数据不足失败不消费测试集。
        claimed, claim_reason = self._claim_final_test_usage()
        if not claimed:
            self.results["test_skipped"] = True
            self.results["test_skip_reason"] = claim_reason
            self._l(f"独立测试未执行：{claim_reason}", "yellow")
            self._stage(8, "done")
            return

        test_acc = float("nan")
        test_auc = float("nan")
        test_note = "测试段无标签样本"
        if len(te_label) > 0:
            pred = model.predict(te_label[feat_names].values)
            try:
                proba = model.predict_proba(te_label[feat_names].values)
            except Exception:
                proba = None
            test_acc, majority, test_auc_raw, has_edge, test_note = gui.predictive_power(
                te_label["label"].map(L2N).values, pred, proba)
            test_auc = test_auc_raw if test_auc_raw == test_auc_raw else float("nan")

        bt_actions = pd.Series(model.predict(bt_rows[feat_names].values)).map(N2A).values
        bt_ma = list(bt_rows["收盘"].rolling(ma_period).mean())
        eng = gui.run_backtest(
            CAPITAL, RISK_CONTROL, strat["stop_loss"], strat["take_profit"], ma_period,
            FRICTION, COMMISSION_PCT / 100.0, STAMP_PCT / 100.0, SLIPPAGE_PCT / 100.0,
            list(bt_rows["日期"]), list(bt_rows["收盘"]), bt_ma, list(bt_actions),
            ma_break_exit=strat["ma_break_exit"])
        final, ret, dd = eng.result()
        ret_pct = ret * 100.0
        n_trades = eng.n_trades

        self.results["test"] = {
            "acc": test_acc, "auc": test_auc, "note": test_note,
            "final_capital": float(final), "ret_pct": float(ret_pct),
            "max_dd_pct": float(dd * 100.0), "n_trades": int(n_trades),
            "n_rows": int(len(bt_rows)), "start": str(bt_rows["日期"].iloc[0].date()),
            "end": str(bt_rows["日期"].iloc[-1].date()),
        }
        self._l(f"独立测试回测：区间 {bt_rows['日期'].iloc[0].date()} ~ "
                f"{bt_rows['日期'].iloc[-1].date()} | 收益 {ret_pct:+.2f}% | "
                f"回撤 {dd * 100:.1f}% | 交易 {n_trades} 次", "green")
        self._l(f"独立测试预测力：{test_note}")

        # 落盘：资金曲线 + 交易明细
        exp_dir = self._safe_exp_dir()
        try:
            pd.DataFrame({"日期": list(bt_rows["日期"]), "净值": list(eng.equity)}).to_csv(
                os.path.join(exp_dir, "backtest_test.csv"), index=False, encoding="utf-8-sig")
        except Exception as e:
            self._l(f"保存测试资金曲线失败：{e}", "yellow")
        try:
            if eng.trade_log:
                pd.DataFrame(eng.trade_log).to_csv(
                    os.path.join(exp_dir, "trades_test.csv"), index=False, encoding="utf-8-sig")
        except Exception as e:
            self._l(f"保存测试交易明细失败：{e}", "yellow")

        # 同时复制一份到全局资金曲线目录，供历史记录页查看（不改动旧文件）
        try:
            gui_mod = gui
            curve_dir = getattr(gui_mod, "CURVE_DIR", None)
            if curve_dir:
                os.makedirs(curve_dir, exist_ok=True)
                fname = f"curve_auto_{self.exp_id}.csv"
                pd.DataFrame({"日期": list(bt_rows["日期"]), "净值": list(eng.equity)}).to_csv(
                    os.path.join(curve_dir, fname), index=False, encoding="utf-8-sig")
                self.results["curve_rel"] = f"equity_curves/{fname}"
        except Exception as e:
            self._l(f"复制资金曲线到全局目录失败：{e}", "yellow")
        self._mark_final_test_usage("completed", f"exp={self.exp_id} 最终测试完成")
        self._stage(8, "done")

    # ==================================================================
    # 阶段 9：保存与报告生成
    # ==================================================================
    def _s9_report(self, gui, aborted=""):
        self._stage(9, "running")
        self._status("保存与报告生成 ...")
        exp_dir = self._safe_exp_dir()

        # ---- 保存模型 ----
        model_file = None
        try:
            if self.results.get("final_model") is not None and exp_dir:
                model_file = os.path.join(exp_dir, "model.json")
                self.results["final_model"].save_model(model_file)
                self._l(f"模型已保存：{model_file}", "green")
        except Exception as e:
            self._l(f"保存模型失败：{e}", "red")
            model_file = None
        self.results["model_file"] = model_file

        # ---- 统一写各类 JSON 产物 ----
        self._dump_json(os.path.join(exp_dir, "config.json"), {
            "exp_id": self.exp_id,
            "code": self.cfg.get("code"),
            "train_start": self.results.get("train_start"),
            "train_end": self.results.get("train_end"),
            "test_start": self.results.get("test_start"),
            "test_end": self.results.get("test_end"),
            "capital": CAPITAL, "friction": FRICTION,
            "commission_pct": COMMISSION_PCT, "stamp_pct": STAMP_PCT,
            "slippage_pct": SLIPPAGE_PCT,
            "budget": {"model_trials": MODEL_TRIALS, "feature_trials": FEATURE_TRIALS,
                       "strategy_trials": STRATEGY_TRIALS, "model_seeds": MODEL_SEEDS,
                       "n_jobs": N_JOBS},
        })
        self._dump_json(os.path.join(exp_dir, "data_check.json"), {
            "train_rows": int(len(self.results.get("train_rows_all", []))),
            "test_rows": int(self.results.get("test", {}).get("n_rows", 0)),
        })
        self._dump_json(os.path.join(exp_dir, "features.json"), {
            "feat_universe": self.results.get("feat_universe", []),
            "feat_pool": self.results.get("feat_pool", []),
            "best_features": self.results.get("best_features", []),
        })
        self._dump_json(os.path.join(exp_dir, "model_params.json"),
                        self.results.get("best_model_params", {}))
        self._dump_json(os.path.join(exp_dir, "strategy_params.json"),
                        self.results.get("best_strategy", {}))

        status = self.results.get("status", "unknown")
        save_ok = bool(exp_dir) and (bool(model_file) if status == "qualified" else True)
        summary = {
            "exp_id": self.exp_id,
            "status": status,
            "aborted": aborted,
            "save_ok": save_ok,
            "best_model_params": self.results.get("best_model_params"),
            "best_features": self.results.get("best_features"),
            "best_strategy": self.results.get("best_strategy"),
            "stability": {k: v for k, v in self.results.get("stability", {}).items()
                          if k in ("mean_ret", "std_ret", "score", "mean_trades")},
            "test": self.results.get("test"),
            "test_skipped": bool(self.results.get("test_skipped")),
            "test_skip_reason": str(self.results.get("test_skip_reason") or ""),
            "model_file": model_file,
        }
        self._dump_json(os.path.join(exp_dir, "summary.json"), summary)

        # ---- 历史记录（复用主程序格式，触发方式=一键自动）----
        self._write_history(gui)

        # ---- Markdown 报告 ----
        try:
            self._write_report(os.path.join(exp_dir, "report.md"), summary)
        except Exception as e:
            self._l(f"生成报告失败：{e}", "yellow")

        # ---- 界面摘要 ----
        self.results["ui_summary"] = summary
        try:
            self.emit("summary", summary)
        except Exception:
            pass

        if status == "no_qualified":
            self._l("结论：暂无合格候选模型（已按纪律结束，未强行选模型）", "red")
        elif aborted:
            self._l(f"结论：本次研究中止 —— {aborted}", "red")
        elif self.results.get("test_skipped"):
            st = self.results.get("stability", {}) or {}
            self._l(f"结论：训练期验证完成（独立测试未执行，留待预定的最终评估）。"
                    f"训练期多种子平均收益 {st.get('mean_ret', 0):+.2f}%", "title")
        else:
            t = self.results.get("test", {})
            self._l(f"结论：已在独立测试段完成一次性评估，收益 {t.get('ret_pct', 0):+.2f}%，"
                    f"回撤 {t.get('max_dd_pct', 0):.1f}%", "title")

        # ---- 保存失败明确提示（绝不显示假成功）----
        if exp_dir and not model_file and status == "qualified":
            self._l("⚠ 模型文件保存失败：本次结果不可复用，请在日志中查看原因。", "red")
        self._stage(9, "done")

    # ------------------------------------------------------------------
    # 历史记录写入（与主程序 HISTORY_HEADER 完全兼容）
    # ------------------------------------------------------------------
    def _write_history(self, gui):
        try:
            ai_name = self.cfg.get("ai_name") or "AI1"
            hist_path = gui.run_history_path_for(ai_name, True)
            test = self.results.get("test", {}) or {}
            strat = self.results.get("best_strategy", {}) or {}
            stab = self.results.get("stability", {}) or {}
            status = self.results.get("status", "unknown")

            if test:
                ret_str = f"测试{test.get('ret_pct', 0):+.2f}%"
                acc = test.get("acc", float("nan"))
                acc_str = f"测试准确率{acc:.4f}" if acc == acc else "测试准确率无"
            elif self.results.get("test_skipped"):
                ret_str = "验证通过(独立测试未执行)"
                acc_str = "-"
            elif status == "no_qualified":
                ret_str = "暂无合格候选模型"
                acc_str = "-"
            else:
                ret_str = "未完成(中止)"
                acc_str = "-"
            note = self.results.get("no_qualified_reason", "")
            skip_reason = str(self.results.get("test_skip_reason") or "")
            if skip_reason:
                note = (note + "；" if note else "") + f"独立测试未执行：{skip_reason}"
            stab_note = (f"稳定性均{stab.get('mean_ret', 0):+.2f}%/σ{stab.get('std_ret', 0):.2f}%"
                         if stab else "")
            if stab_note:
                note = (note + "；" if note else "") + stab_note
            if status == "qualified" and not self.results.get("model_file"):
                note = (note + "；" if note else "") + "模型保存失败"
            reuse = self._detect_test_reuse(gui, ai_name)
            if reuse:
                note = (note + "；" if note else "") + "该测试区间此前已被实验使用"

            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [time_str, self.cfg.get("code"), self.results.get("train_start", ""),
                   self.results.get("train_end", ""), self.results.get("test_start", ""),
                   f"{CAPITAL:.2f}", "", ret_str, acc_str,
                   ",".join(self.results.get("best_features", [])), note,
                   str(self.cfg.get("trigger") or "一键自动"),
                   _fmt(strat.get("stop_loss", "")), _fmt(strat.get("take_profit", "")),
                   _fmt(strat.get("ma_period", "")),
                   ",".join(self.results.get("best_features", [])),
                   self.results.get("curve_rel", ""), str(MODEL_TRIALS + FEATURE_TRIALS),
                   f"佣金{COMMISSION_PCT:g}%+印花税{STAMP_PCT:g}%+滑点{SLIPPAGE_PCT:g}%"]
            # 按标准表头长度补/截，保证列数一致
            ncol = len(getattr(gui, "HISTORY_HEADER", row))
            row = (row + [""] * ncol)[:ncol]
            gui.append_history(hist_path, row)
            self.results["history_path"] = hist_path
            self._l(f"已写入历史记录：{hist_path}", "green")
            try:
                self.emit("refresh_history")
            except Exception:
                pass
        except Exception as e:
            self._l(f"写入历史记录失败：{e}", "red")
            self.results["history_error"] = str(e)

    def _detect_test_reuse(self, gui, ai_name):
        try:
            path = gui.run_history_path_for(ai_name, True)
            header, data = gui.read_history_rows(path)
            if "股票代码" not in header or "测试起始日期" not in header:
                return False
            ci = header.index("股票代码")
            si = header.index("测试起始日期")
            for r in data:
                if ci < len(r) and si < len(r):
                    if (str(r[ci]).strip() == str(self.cfg.get("code")) and
                            str(r[si]).strip() == str(self.results.get("test_start"))):
                        return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    def _safe_exp_dir(self):
        if self.exp_dir:
            try:
                os.makedirs(self.exp_dir, exist_ok=True)
            except Exception:
                pass
        return self.exp_dir or ""

    def _dump_json(self, path, obj):
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self._l(f"写 JSON 失败 {os.path.basename(path)}：{e}", "yellow")

    def _write_csv(self, path, header, rows):
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                for r in rows:
                    w.writerow(r)
        except Exception as e:
            self._l(f"写 CSV 失败 {os.path.basename(path)}：{e}", "yellow")

    def _write_report(self, path, summary):
        if not path:
            return
        r = self.results
        lines = []
        lines.append(f"# 一键自动训练报告\n")
        lines.append(f"- 实验编号：{self.exp_id}")
        lines.append(f"- 股票代码：{self.cfg.get('code')}")
        lines.append(f"- 训练区间：{r.get('train_start')} ~ {r.get('train_end')}")
        lines.append(f"- 测试区间：{r.get('test_start')} ~ {r.get('test_end')}（仅最后评估一次）")
        lines.append(f"- 研究状态：{summary.get('status')}"
                     + (f"（{summary.get('aborted')}）" if summary.get("aborted") else ""))
        lines.append("")
        lines.append("## 数据边界纪律")
        lines.append(f"- 训练/测试各裁掉末尾 {LOOKAHEAD} 行，避免标签 look-ahead 跨边界泄漏。")
        lines.append("- 测试段未参与特征/模型/策略的任何选择。")
        lines.append("")
        lines.append("## 最佳模型参数")
        lines.append("```json")
        lines.append(json.dumps(r.get("best_model_params", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("## 最佳特征组合")
        lines.append(", ".join(r.get("best_features", [])) or "（无）")
        lines.append("")
        lines.append("## 最佳交易策略（训练期验证段选出）")
        lines.append("```json")
        lines.append(json.dumps(r.get("best_strategy", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        st = r.get("stability", {})
        if st:
            lines.append("## 稳定性（多种子）")
            lines.append(f"- 平均收益 {st.get('mean_ret', 0):+.2f}% | 标准差 {st.get('std_ret', 0):.2f}% | "
                         f"平均交易 {st.get('mean_trades', 0):.1f} 次")
        lines.append("")
        lines.append("## 独立测试结果（一次性）")
        if r.get("test_skipped"):
            reason = str(r.get("test_skip_reason") or "")
            lines.append("- 独立测试未执行："
                         + (reason or "自主研究模式按纪律跳过"))
        t = r.get("test") or {}
        if t:
            lines.append(f"- 区间 {t.get('start')} ~ {t.get('end')}（{t.get('n_rows')} 行）")
            lines.append(f"- 收益 {t.get('ret_pct', 0):+.2f}% | 最大回撤 {t.get('max_dd_pct', 0):.1f}% | "
                         f"交易 {t.get('n_trades')} 次 | 最终资金 {t.get('final_capital', 0):.2f}")
            acc = t.get("acc", float("nan"))
            if acc == acc:
                lines.append(f"- 测试准确率 {acc:.4f}")
            lines.append(f"- 说明：{t.get('note', '')}")
        elif not r.get("test_skipped"):
            lines.append("- 无（未产生合格候选模型，未进行独立测试）。")
        lines.append("")
        lines.append("> 结论仅基于训练期选择 + 一次性独立测试，不构成投资建议。")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _fmt(v):
    try:
        f = float(v)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:g}"
    except Exception:
        return str(v)


def _n_choose(n, k):
    import math
    if k > n:
        return 0
    return math.comb(n, k)
