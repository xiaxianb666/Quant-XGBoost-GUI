# -*- coding: utf-8 -*-
"""
research_engine.py — 【自主研究与进化循环】核心引擎

组成（对应需求第四、五、六、八、十节）：
  A. 晋级验收标准：可配置（promotion_rules.json）+ 纯函数 evaluate_promotion()，可离线单元测试；
  B. 研究方向决策：select_direction() 依据历史实验结果决定下一轮研究重点（纯函数）；
  C. 实验配置生成：plan_experiment() 在既有验证过的范围内生成配置，并避免重复相同随机组合；
  D. 滚动窗口验证：rolling_window_metrics() 在训练期内部做多窗口 + 多种子评价；
  E. 自主调度器：AutonomousScheduler —— 选股票 → 检查数据 → 生成实验 → 执行 → 评价 →
     晋级/拒绝 → 记录 → 依据结果调整方向 → 下一只；支持暂停/恢复/停止/错误暂停/内存保护/断点续跑。

纪律（硬性）：
  * 独立测试集绝不参与特征选择、参数优化、策略选择、晋级判断或下一轮方向选择；
    自主循环默认【不执行】独立测试（cfg["skip_final_test"]=True），只为“预先规定的最终评估”保留；
  * 一次只跑一个重型训练任务（HeavyTaskGate + MemoryGuard）；
  * 不执行模型生成的任意代码 / Shell 命令；
  * 所有结论只来自训练期滚动验证，收益类数字一律是“历史模拟”，不构成任何投资建议。
"""

import os
import csv
import json
import time
import shutil
import random
import hashlib
import traceback
from datetime import datetime, timedelta

try:
    from safe_io import (atomic_write_json, atomic_write_text, atomic_append_csv,
                         read_json_safe, ensure_dir)
except Exception:      # pragma: no cover
    def ensure_dir(p): pass
    atomic_write_json = lambda *a, **k: False
    atomic_write_text = lambda *a, **k: False
    atomic_append_csv = lambda *a, **k: False
    read_json_safe = lambda *a, **k: None

try:
    import stock_pool as sp
except Exception:      # pragma: no cover
    sp = None


# ====================================================================
# 0. 晋级标准默认配置（可被 promotion_rules.json 覆盖）
# ====================================================================
DEFAULT_PROMOTION_RULES = {
    # ---- 证据量（不允许凭单个窗口/单个种子晋级）----
    "min_windows": 3,
    "min_seeds": 2,
    "first_promotion_extra_seed": 1,      # 首次建立基准时额外要求的种子数
    # ---- 交易策略硬门槛（训练期验证段，含交易成本）----
    "min_mean_ret_pct": 0.0,              # 多窗口平均净收益必须为正
    "min_positive_window_ratio": 0.5,     # 至少一半窗口为正
    "max_worst_window_dd_pct": 60.0,      # 最差窗口最大回撤上限（正数表示回撤幅度）
    "min_mean_trades": 2.0,               # 平均交易次数下限
    "max_seed_std_pct": 8.0,              # 多种子收益标准差上限
    # ---- 相对基准的改进要求 ----
    "min_improvement_ret_pp": 1.0,        # 平均收益至少高出基准这么多“百分点”
    "min_improvement_score_abs": 0.5,     # 综合得分至少高出基准这么多
    "improvement_check_when_no_baseline": False,   # 无基准时跳过“相对改进”检查
    # ---- 选择偏差（跑得越多，门槛越高）----
    "selection_bias_free_experiments": 5,
    "selection_bias_pp_per_experiment": 0.1,
    # ---- 复杂度约束（4GB 老电脑）----
    "max_features": 8,
    "max_depth_allowed": 5,
    "max_n_estimators": 800,
    "complexity_penalty_per_extra_feature": 0.15,
    "complexity_penalty_per_depth_over_3": 0.2,
    # ---- 评分权重（只用于“验证段排序”，不是收益承诺）----
    "score_dd_weight": 0.30,
    "score_std_weight": 0.50,
    "score_trade_weight": 0.10,
    # ---- 稳定基准（stable）比最佳验证（best_validated）更严格 ----
    "stable_extra_positive_window_ratio": 0.15,
    "stable_extra_seeds": 1,
    "stable_min_improvement_pp": 2.0,
}


# ====================================================================
# A0. 评估口径与可比性（批次 3B）
# ====================================================================
# 这些值必须与 rolling_window_metrics 内部实际使用的回测成本一致。
ROLLING_METRIC_VERSION = "rolling_metrics_v2_3b"
ROLLING_LOOKAHEAD = 5
ROLLING_COST = {
    "friction": True,
    "commission_rate": 0.0003,
    "stamp_tax_rate": 0.001,
    "slippage_rate": 0.001,
}


def _date_text(value):
    """把日期/Timestamp 转成稳定字符串；失败则返回原值的安全字符串。"""
    try:
        return value.date().isoformat()
    except Exception:
        return str(value)[:10]


def _normalize_window_specs(specs):
    """规范化为可哈希/可保存的窗口边界列表。只保留边界与窗口编号。"""
    out = []
    for i, item in enumerate(specs or [], 1):
        if not isinstance(item, dict):
            continue
        out.append({
            "window": int(_num(item.get("window"), i) or i),
            "train_start": str(item.get("train_start") or ""),
            "train_end": str(item.get("train_end") or ""),
            "valid_start": str(item.get("valid_start") or ""),
            "valid_end": str(item.get("valid_end") or ""),
        })
    return out


def _window_fingerprint(window_specs):
    canonical = json.dumps(_normalize_window_specs(window_specs),
                           ensure_ascii=False, sort_keys=True)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()[:16]


def make_evaluation_contract(window_specs, label, cost, n_windows, lookahead,
                             min_train_rows, seeds_per_window, features=None,
                             metric_version=ROLLING_METRIC_VERSION,
                             data_start=None, data_end=None, n_rows=0):
    """构造候选/基准复评共用的评估口径元数据。

    指纹字段只包含“评估协议”：
      数据区间、窗口边界、标签定义、成本假设、指标版本、滚动参数。
    模型特征/参数/策略不进入指纹，因为它们正是候选与基准的差异本身；
    但会记录在 `features` / `model` 等字段供追溯。
    """
    specs = _normalize_window_specs(window_specs)
    meta = {
        "metric_version": str(metric_version),
        "window_specs": specs,
        "window_fingerprint": _window_fingerprint(specs),
        "data_start": str(data_start or ""),
        "data_end": str(data_end or ""),
        "n_rows": int(n_rows or 0),
        "label": dict(label or {}),
        "cost": dict(cost or {}),
        "n_windows": int(n_windows or len(specs) or 0),
        "lookahead": int(lookahead or ROLLING_LOOKAHEAD),
        "min_train_rows": int(min_train_rows or 0),
        "seeds_per_window": int(seeds_per_window or 0),
        "features": list(features or []),
    }
    core = {k: meta[k] for k in ("metric_version", "window_fingerprint",
                                 "data_start", "data_end", "n_rows",
                                 "label", "cost", "n_windows", "lookahead",
                                 "min_train_rows", "seeds_per_window")}
    meta["fingerprint"] = hashlib.md5(
        json.dumps(core, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return meta


def evaluation_protocol_compatible(candidate_eval, baseline_eval):
    """比较评估协议（不比较旧窗口；旧窗口会被同窗口复评替代）。

    返回 (bool, 原因)。旧记录缺少元数据时明确返回不可比，不做推断。
    """
    if not isinstance(candidate_eval, dict):
        return False, "候选缺少 evaluation 元数据"
    if not isinstance(baseline_eval, dict):
        return False, "基准缺少 evaluation 元数据（旧记录不可比）"
    for key in ("metric_version", "label", "cost", "lookahead"):
        if candidate_eval.get(key) != baseline_eval.get(key):
            return False, f"{key} 不一致"
    return True, ""


def evaluation_comparable(candidate, baseline):
    """比较两个当前评估结果是否真正同窗口、同口径。

    候选/基准都没有元数据时返回 True，仅用于保持纯函数旧调用兼容；
    只要有一侧带元数据而另一侧缺失，就判定不可比。
    """
    c = (candidate or {}).get("evaluation")
    b = (baseline or {}).get("evaluation") if baseline else None
    if c is None and b is None:
        return True, ""
    if not isinstance(c, dict) or not isinstance(b, dict):
        return False, "缺少 evaluation 元数据（旧记录或未复评结果不可比）"
    for key in ("metric_version", "label", "cost", "window_fingerprint",
                "data_start", "data_end", "n_rows",
                "n_windows", "lookahead", "min_train_rows", "seeds_per_window",
                "fingerprint"):
        if c.get(key) != b.get(key):
            return False, f"{key} 不一致"
    return True, ""


# 研究方向（goals）：每个方向只调“在既有接口内已验证的参数”，不生成任意代码
DIRECTION_ORDER = ["strategy", "features", "model_params", "stability", "label_mode"]

# 各方向的资源配比（相对于基础预算的倍率；老电脑保护：总训练次数有限）
GOAL_BUDGET_SCALE = {
    "strategy":     {"model": 0.75, "feature": 0.75, "strategy": 1.75, "seeds": 1.0},
    "features":     {"model": 0.75, "feature": 1.75, "strategy": 1.0, "seeds": 1.0},
    "model_params": {"model": 1.75, "feature": 0.75, "strategy": 1.0, "seeds": 1.0},
    "stability":    {"model": 0.5, "feature": 0.5, "strategy": 1.0, "seeds": 2.0},
    "label_mode":   {"model": 1.0, "feature": 1.0, "strategy": 1.0, "seeds": 1.0},
}

DEFAULT_ENGINE_CFG = {
    "experiments_per_run": 1,          # 每次循环执行几个实验（老电脑默认 1）
    "max_experiments_per_session": 0,  # 0 = 不限制（长期自主运行）
    "max_experiment_retries": 2,       # 单个实验失败后的重试上限（有限重试）
    "retry_wait_s": 20,
    "max_consecutive_failures": 5,     # 连续失败这么多 → 进入错误暂停
    "universe_refresh_hours": 12,
    "idle_sleep_s": 30,
    "pause_poll_s": 2,
    "max_consecutive_universe_failures": 2,
    "base_budget": {"model_trials": 8, "feature_trials": 12,
                    "strategy_trials": 24, "model_seeds": 3},
    "rolling_windows": 3,
    "rolling_seeds": 2,
    "min_train_rows_rolling": 120,     # 滚动验证最少训练行数
    "train_years": 5,                  # 每只股票默认取多少年训练数据
    "reserved_test_days": 60,          # 为“预定的最终评估”保留的测试天数（自主循环不使用）
    "reserved_test_start": "",         # 也可固定指定（YYYY-MM-DD），留空=按 reserved_test_days 推算
    "experiment_log_max_rows": 20000,
}


class ResearchError(Exception):
    """可预期的研究失败（数据不足 / 无合格候选等）。"""


# ====================================================================
# A. 晋级验收标准（纯函数 + 可配置）
# ====================================================================
def load_promotion_rules(path=None):
    """读取晋级规则配置；不存在则写入默认配置。返回 dict。"""
    rules = dict(DEFAULT_PROMOTION_RULES)
    if path:
        cfg = read_json_safe(path, None)
        if isinstance(cfg, dict):
            rules.update({k: v for k, v in cfg.items() if v is not None})
        else:
            atomic_write_json(path, DEFAULT_PROMOTION_RULES)
    return rules


def candidate_score(metrics, rules, n_experiments=0):
    """候选综合得分（只在训练期验证段上计算，用于排序与相对比较）。

    score = 平均净收益% − 回撤惩罚 × 最差窗口回撤% − 稳定性惩罚 × 种子标准差
            + 交易活跃度奖励 − 复杂度惩罚 − 选择偏差惩罚
    ⚠ 这是“历史模拟排序分”，不是收益预测，也不代表未来表现。
    """
    m = metrics or {}
    ret = _num(m.get("mean_ret_pct"), 0.0)
    dd = _num(m.get("max_dd_pct"), 0.0)
    std = _num(m.get("seed_std_pct"), 0.0)
    trades = _num(m.get("mean_trades"), 0.0)
    n_feat = int(_num(m.get("n_features"), 0))
    depth = int(_num(m.get("max_depth"), 0))
    n_est = int(_num(m.get("n_estimators"), 0))

    score = ret
    score -= float(rules["score_dd_weight"]) * dd
    score -= float(rules["score_std_weight"]) * std
    score += float(rules["score_trade_weight"]) * max(0.0, trades - float(rules["min_mean_trades"]))
    score -= complexity_penalty(n_feat, depth, n_est, rules)
    score -= selection_bias_penalty(n_experiments, rules)
    return round(float(score), 4)


def complexity_penalty(n_features, max_depth, n_estimators, rules):
    """复杂度惩罚：特征数 / 树深 / 树数超过配置上限的部分要付出代价。"""
    p = 0.0
    p += float(rules["complexity_penalty_per_extra_feature"]) * max(0, int(n_features) - 5)
    p += float(rules["complexity_penalty_per_depth_over_3"]) * max(0, int(max_depth) - 3)
    if int(n_estimators) > 400:
        p += 0.1 * (int(n_estimators) - 400) / 100.0
    return round(p, 4)


def selection_bias_penalty(n_experiments, rules):
    """选择偏差惩罚：同一只股票实验做得越多，“看起来最好”越可能只是运气。"""
    free_n = int(rules.get("selection_bias_free_experiments", 5))
    per = float(rules.get("selection_bias_pp_per_experiment", 0.1))
    extra = max(0, int(n_experiments) - free_n)
    return round(per * extra, 4)


def evaluate_promotion(candidate, baseline, rules, n_experiments=0):
    """【晋级判定】纯函数：只吃训练期验证段指标，绝不吃独立测试结果。

    candidate / baseline: metrics dict，字段见 build_metrics_from_results()。
    n_seeds 必须是“实际成功种子数的最小值”（由 build_metrics_from_results 生成），
    不是配置种子数，也不能用预算值补足。
    返回 dict：
      promote : bool
      verdict : "promoted" / "rejected"
      reasons : 未通过原因列表（人类可读，写入实验记录）
      checks  : 每条规则的明细（name / ok / detail）
      candidate_score / baseline_score / required_improvement_pp
    """
    rules = dict(DEFAULT_PROMOTION_RULES, **(rules or {}))
    c = candidate or {}
    b = baseline or None

    # 批次 3B：先过可比性，再看性能。
    # 不可比时只返回“证据不可比”，不能继续用旧指标算出一份性能拒绝。
    comp_error = c.get("_comparison_error") or (b.get("_comparison_error") if b else "")
    if comp_error:
        return {
            "promote": False, "verdict": "rejected",
            "reasons": [f"评估不可比：{comp_error}"],
            "checks": [{"name": "评估窗口/口径可比", "ok": False,
                        "detail": str(comp_error)}],
            "candidate_score": None, "baseline_score": None,
            "required_improvement_pp": None,
        }
    if b is not None:
        ok_comp, why_comp = evaluation_comparable(c, b)
        if not ok_comp:
            return {
                "promote": False, "verdict": "rejected",
                "reasons": [f"评估不可比：{why_comp}"],
                "checks": [{"name": "评估窗口/口径可比", "ok": False,
                            "detail": why_comp}],
                "candidate_score": None, "baseline_score": None,
                "required_improvement_pp": None,
            }
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    n_win = int(_num(c.get("n_windows"), 0))
    n_seed = int(_num(c.get("n_seeds"), 0))
    pos_ratio = _num(c.get("positive_window_ratio"), 0.0)
    mean_ret = _num(c.get("mean_ret_pct"), None)
    worst_dd = _num(c.get("max_dd_pct"), None)
    trades = _num(c.get("mean_trades"), 0.0)
    seed_std = _num(c.get("seed_std_pct"), None)
    n_feat = int(_num(c.get("n_features"), 0))
    depth = int(_num(c.get("max_depth"), 0))
    n_est = int(_num(c.get("n_estimators"), 0))

    # 1) 证据量
    add("验证窗口数量足够", n_win >= int(rules["min_windows"]),
        f"n_windows={n_win}，要求≥{int(rules['min_windows'])}")
    seed_known = bool(c.get("seed_evidence_known", True))
    seed_complete = bool(c.get("seed_evidence_complete", True))
    add("种子证据可追溯", seed_known,
        "旧滚动记录缺少逐窗口明细，无法确认实际成功种子数" if not seed_known
        else "实际成功种子数可追溯")
    add("种子证据完整", seed_complete,
        "存在跳过的计划窗口或滚动验证未完成，不能以剩余窗口补足" if not seed_complete
        else "所有计划窗口均有有效证据")
    need_seeds = int(rules["min_seeds"])
    if b is None:
        need_seeds += int(rules.get("first_promotion_extra_seed", 1))
    add("随机种子数量足够", n_seed >= need_seeds,
        f"n_seeds={n_seed}（实际成功种子数的最小值），要求≥{need_seeds}"
        + ("（首次建立基准要求更严）" if b is None else ""))

    # 2) 交易策略硬门槛（训练期验证段，含交易成本）
    add("平均净收益为正", (mean_ret is not None) and mean_ret > float(rules["min_mean_ret_pct"]),
        f"mean_ret={_fmt_n(mean_ret)}%，要求>{float(rules['min_mean_ret_pct'])}%")
    add("正收益窗口占比达标", pos_ratio >= float(rules["min_positive_window_ratio"]),
        f"positive_window_ratio={pos_ratio:.2f}，要求≥{float(rules['min_positive_window_ratio'])}")
    add("最差窗口回撤可接受", (worst_dd is not None) and worst_dd <= float(rules["max_worst_window_dd_pct"]),
        f"max_dd={_fmt_n(worst_dd)}%，上限{float(rules['max_worst_window_dd_pct'])}%")
    add("交易次数达标", trades >= float(rules["min_mean_trades"]),
        f"mean_trades={trades:.1f}，要求≥{float(rules['min_mean_trades'])}")
    add("多种子稳定性达标", (seed_std is not None) and seed_std <= float(rules["max_seed_std_pct"]),
        f"seed_std={_fmt_n(seed_std)}%，上限{float(rules['max_seed_std_pct'])}%")

    # 3) 复杂度约束（4GB 老电脑）
    add("复杂度在资源约束内",
        (n_feat <= int(rules["max_features"]) and depth <= int(rules["max_depth_allowed"])
         and n_est <= int(rules["max_n_estimators"])),
        f"features={n_feat}≤{int(rules['max_features'])}，depth={depth}≤{int(rules['max_depth_allowed'])}，"
        f"n_estimators={n_est}≤{int(rules['max_n_estimators'])}")

    # 4) 选择偏差：跑得越多，要求的改进越大
    bias_extra = selection_bias_penalty(n_experiments, rules)
    required_improvement = float(rules["min_improvement_ret_pp"]) + bias_extra
    c_score = candidate_score(c, rules, n_experiments)
    b_score = candidate_score(b, rules, 0) if b else None

    if b is None:
        if bool(rules.get("improvement_check_when_no_baseline", False)):
            add("相对基准改进", False, "没有可用基准（首次建立基准）")
        else:
            add("相对基准改进", True, "无历史基准：按绝对门槛与更严证据要求评估（首次建立基准）")
    else:
        b_ret = _num(b.get("mean_ret_pct"), None)
        ret_gain = (mean_ret - b_ret) if (mean_ret is not None and b_ret is not None) else None
        ok_gain = (ret_gain is not None) and (ret_gain >= required_improvement)
        score_gain = (c_score - b_score) if b_score is not None else None
        ok_score = (score_gain is not None) and (score_gain >= float(rules["min_improvement_score_abs"]))
        add("相对基准收益改进", ok_gain,
            f"收益改进={_fmt_n(ret_gain)}pp，要求≥{required_improvement:.2f}pp"
            + (f"（含选择偏差补偿 {bias_extra:.2f}pp）" if bias_extra else ""))
        add("相对基准得分改进", ok_score,
            f"得分改进={_fmt_n(score_gain)}，要求≥{float(rules['min_improvement_score_abs'])}")

    reasons = [f"{c_['name']}：{c_['detail']}" for c_ in checks if not c_["ok"]]
    promote = len(reasons) == 0
    return {
        "promote": promote,
        "verdict": "promoted" if promote else "rejected",
        "reasons": reasons,
        "checks": checks,
        "candidate_score": c_score,
        "baseline_score": b_score,
        "required_improvement_pp": round(required_improvement, 3),
    }


def evaluate_stable_upgrade(candidate, stable, rules):
    """是否达到“稳定基准”标准（比 best_validated 更严格）。返回 (bool, 原因)。"""
    rules = dict(DEFAULT_PROMOTION_RULES, **(rules or {}))
    c = candidate or {}
    s = stable or None
    comp_error = c.get("_comparison_error") or (s.get("_comparison_error") if s else "")
    if comp_error:
        return False, f"评估不可比：{comp_error}"
    if s is not None:
        ok_comp, why_comp = evaluation_comparable(c, s)
        if not ok_comp:
            return False, f"评估不可比：{why_comp}"
    why = []
    if not bool(c.get("seed_evidence_known", True)):
        why.append("种子证据不可追溯（旧记录缺少逐窗口明细）")
    if not bool(c.get("seed_evidence_complete", True)):
        why.append("种子证据不完整（存在跳过的计划窗口或滚动验证未完成）")
    need_pos = float(rules["min_positive_window_ratio"]) + float(rules["stable_extra_positive_window_ratio"])
    if _num(c.get("positive_window_ratio"), 0.0) < need_pos:
        why.append(f"正收益窗口占比 {_num(c.get('positive_window_ratio'), 0.0):.2f} < {need_pos:.2f}")
    need_seed = int(rules["min_seeds"]) + int(rules["stable_extra_seeds"])
    if int(_num(c.get("n_seeds"), 0)) < need_seed:
        why.append(f"实际成功种子数 {int(_num(c.get('n_seeds'), 0))} < {need_seed}"
                   "（按各窗口成功数最小值计）")
    if s is not None:
        gain = _num(c.get("mean_ret_pct"), None)
        base = _num(s.get("mean_ret_pct"), None)
        if gain is None or base is None or (gain - base) < float(rules["stable_min_improvement_pp"]):
            why.append(f"相对稳定基准收益改进不足（要求≥{float(rules['stable_min_improvement_pp'])}pp）")
    return (len(why) == 0), "；".join(why)


# ====================================================================
# B. 预测力评价（按任务类型选指标，不机械套用 AUC / IC）
# ====================================================================
def evaluate_prediction_metrics(task_kind, y_true, pred=None, proba=None, scores=None):
    """按任务类型选择评价指标。

    task_kind:
      "classification" → 准确率 / 多数类基准 / AUC（多分类用 macro OVR）
      "ranking"        → IC（Spearman 秩相关）/ 命中率；不适合用 AUC
    返回 dict；缺依赖/无样本时返回 None 值，绝不抛异常。
    """
    out = {"task_kind": task_kind, "acc": None, "majority": None, "auc": None,
           "ic": None, "n": 0, "note": ""}
    try:
        import numpy as np
        y = np.asarray(y_true).ravel()
        out["n"] = int(len(y))
        if len(y) == 0:
            out["note"] = "无样本"
            return out
        if task_kind == "ranking":
            if scores is None:
                out["note"] = "缺少连续得分，无法算 IC"
                return out
            sc = np.asarray(scores).ravel()
            if len(sc) != len(y):
                out["note"] = "得分与标签长度不一致"
                return out
            try:
                from scipy.stats import spearmanr
                out["ic"] = float(spearmanr(sc, y).correlation)
            except Exception:
                # 无 scipy → 用秩相关近似（皮尔逊）
                r1 = np.argsort(np.argsort(sc)).astype(float)
                r2 = np.argsort(np.argsort(y)).astype(float)
                out["ic"] = float(np.corrcoef(r1, r2)[0, 1]) if len(r1) > 2 else None
            out["note"] = "排序任务：以 IC 为主指标（不使用 AUC 作为唯一标准）"
            return out
        if pred is not None:
            p = np.asarray(pred).ravel()
            out["acc"] = float((p == y).mean()) if len(p) == len(y) else None
            vals, cnts = np.unique(y, return_counts=True)
            out["majority"] = float(cnts.max() / len(y)) if len(cnts) else None
        if proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                P = np.asarray(proba)
                if P.ndim == 2 and P.shape[1] == 2:
                    out["auc"] = float(roc_auc_score(y, P[:, 1]))
                elif P.ndim == 2 and P.shape[1] > 2:
                    out["auc"] = float(roc_auc_score(y, P, multi_class="ovr", average="macro"))
            except Exception:
                out["auc"] = None
        out["note"] = "分类任务：准确率相对多数类基准 + AUC（仅在有概率输出时）"
        return out
    except Exception as e:      # pragma: no cover
        out["note"] = f"预测指标计算异常：{e}"
        return out


# ====================================================================
# C. 研究方向决策（纯函数，依据历史实验结果）
# ====================================================================
def select_direction(history, now=None, order=None):
    """根据该股票的历史实验结果，决定下一轮研究重点。纯函数、可复现。

    history: list[dict]，每条至少含 goal / status / promoted / reject_reason / created_at
             （按时间倒序或正序都可，内部按时间排序）。
    now: datetime（默认当前）。
    返回 {"goal":..., "reason":..., "priorities": {goal: 分数}}
    """
    order = list(order or DIRECTION_ORDER)
    now = now or datetime.now()
    hist = list(history or [])
    if hist:
        hist.sort(key=lambda r: str(r.get("created_at") or ""))
    attempts = {g: 0 for g in order}
    promoted = {g: 0 for g in order}
    consec_fail = {g: 0 for g in order}
    last_time = {g: None for g in order}
    for r in hist:
        g = str(r.get("goal") or "")
        if g not in attempts:
            continue
        attempts[g] += 1
        last_time[g] = str(r.get("created_at") or "") or last_time[g]
        if r.get("promoted"):
            promoted[g] += 1
            consec_fail[g] = 0
        else:
            consec_fail[g] += 1

    # 最近一次拒绝的原因 → 提示下一步方向
    hint = None
    if hist:
        last = hist[-1]
        reason = str(last.get("reject_reason") or "")
        last_status = str(last.get("status") or "")
        if last_status in ("failed", "error"):
            hint = ("model_params", "上一次实验执行失败，先用更保守的模型/特征配置复跑，确认链路可靠")
        elif "稳定性" in reason or "证据" in reason or "窗口" in reason or "种子" in reason:
            hint = ("stability", "上一次因证据不足/不稳定被拒 → 增加窗口与种子做复核")
        elif "交易" in reason or "回撤" in reason or "收益" in reason:
            hint = ("strategy", "上一次交易指标未达标 → 重点搜索止损/止盈/均线策略")
        elif "预测" in reason or "AUC" in reason:
            hint = ("label_mode", "上一次预测力不足 → 尝试不同的标签定义")
        elif "复杂度" in reason or "资源" in reason:
            hint = ("model_params", "上一次复杂度超限 → 搜索更简单的模型参数")

    prios = {}
    for g in order:
        p = 1.0
        if attempts[g] == 0:
            p += 2.0                                  # 未覆盖的方向优先（保证覆盖面）
        if consec_fail[g] >= 2:
            p *= 0.4                                  # 连续失败 → 降权，避免死磕
        if attempts[g] > 0 and promoted[g] == 0:
            p -= 0.2
        lt = last_time[g]
        if lt:
            try:
                dt = datetime.strptime(lt[:19], "%Y-%m-%d %H:%M:%S")
                p += min(1.0, (now - dt).total_seconds() / 86400.0 / 3.0)   # 久未研究 → 升权
            except Exception:
                p += 0.2
        else:
            p += 0.5
        prios[g] = round(p, 3)
    if hint:
        prios[hint[0]] = round(prios.get(hint[0], 1.0) + 1.5, 3)

    goal = max(order, key=lambda g: (prios[g], -order.index(g)))
    if hint and hint[0] == goal:
        reason = hint[1]
    elif attempts[goal] == 0:
        reason = f"方向「{goal}」尚未研究过（优先覆盖）"
    else:
        reason = (f"按历史结果加权：{goal} 得分 {prios[goal]}（尝试 {attempts[goal]} 次，"
                  f"晋级 {promoted[goal]} 次，连续失败 {consec_fail[goal]} 次）")
    return {"goal": goal, "reason": reason, "priorities": prios,
            "attempts": attempts, "promoted": promoted, "consecutive_fail": consec_fail}


# ====================================================================
# D. 实验配置生成（限定在已验证范围内，避免重复相同随机组合）
# ====================================================================
def label_mode_candidates():
    return [{"mode": "vol", "vol_k": 0.5}, {"mode": "vol", "vol_k": 0.75},
            {"mode": "fixed", "vol_k": None}, {"mode": "binary", "vol_k": None},
            {"mode": "vol", "vol_k": 0.35}]


def plan_experiment(goal, history, base_budget=None, seed=None, model_space_size=8,
                    feature_pool=None, test_start=None, test_end=None,
                    train_start=None, train_end=None):
    """生成一次实验的配置（纯函数、可测试、可复现）。

    只在既有接口允许的参数范围内工作：
      * 预算（模型/特征/策略 试验次数、种子数）——上限受老电脑保护；
      * 标签定义（mode / vol_k）——对应 gui.build_labels 支持的三种模式；
      * 模型参数优先顺序——从 auto_research.MODEL_SPACE 里挑“没试过”的先试；
      * 特征子集提示——从历史失败组合里避开。
    返回 dict（含 config_hash）。若搜索空间已覆盖，会明确标注 review=True（复核而非新搜索）。
    """
    base = dict(DEFAULT_ENGINE_CFG["base_budget"])
    if base_budget:
        base.update(base_budget)
    scale = GOAL_BUDGET_SCALE.get(goal, GOAL_BUDGET_SCALE["strategy"])
    budget = {
        "model_trials": max(2, int(round(base["model_trials"] * scale["model"]))),
        "feature_trials": max(3, int(round(base["feature_trials"] * scale["feature"]))),
        "strategy_trials": max(6, int(round(base["strategy_trials"] * scale["strategy"]))),
        "model_seeds": max(2, int(round(base["model_seeds"] * scale["seeds"]))),
    }
    seed = int(seed if seed is not None else random.randint(1, 10 ** 6))

    hist = list(history or [])
    tried_models = set()
    tried_labels = set()
    failed_feature_sets = set()
    for r in hist:
        try:
            cfg = json.loads(r.get("config_json") or "{}")
        except Exception:
            cfg = {}
        for idx in (cfg.get("param_priority") or [])[:1]:
            tried_models.add(int(idx))
        lm = cfg.get("label")
        if isinstance(lm, dict):
            tried_labels.add((lm.get("mode"), lm.get("vol_k")))
        feat = str(r.get("features") or "").strip()
        if feat and not r.get("promoted") and str(r.get("status")) in ("done", "rejected"):
            failed_feature_sets.add(feat)

    # 模型参数优先顺序：未尝试过的在前（避免重复相同组合）
    rng = random.Random(seed)
    idxs = list(range(int(model_space_size)))
    untried = [i for i in idxs if i not in tried_models]
    rng.shuffle(untried)
    tried = [i for i in idxs if i in tried_models]
    rng.shuffle(tried)
    param_priority = (untried + tried)[:max(2, int(budget["model_trials"]))]

    # 标签定义：label_mode 方向优先挑没试过的
    labels = label_mode_candidates()
    label = None
    if goal == "label_mode" or goal == "stability":
        unknown = [l for l in labels if (l["mode"], l["vol_k"]) not in tried_labels]
        label = (unknown[0] if unknown else labels[seed % len(labels)])

    # 特征提示：优先用“没失败过”的特征全集里的前若干个（真正的组合搜索仍由 pipeline 完成）
    feature_hint = []
    if feature_pool:
        pool = [str(f) for f in feature_pool]
        feature_hint = pool[:min(len(pool), 12)]

    plan = {
        "goal": goal,
        "budget": budget,
        "param_priority": param_priority,
        "label": label,
        "feature_hint": feature_hint,
        "seed": seed,
        "train_start": train_start, "train_end": train_end,
        "test_start": test_start, "test_end": test_end,
        "review": False,
        "avoid_feature_sets": sorted(failed_feature_sets)[:20],
        "note": "",
    }
    plan["config_hash"] = config_hash_of(plan)
    known = set()
    for r in hist:
        try:
            known.add(json.loads(r.get("config_json") or "{}").get("config_hash"))
        except Exception:
            pass
    if plan["config_hash"] in known:
        plan["review"] = True
        plan["note"] = "该配置此前出现过：本次按“复核”记录（不再算作新搜索）"
    return plan


def config_hash_of(plan):
    # ⚠ 刻意【不】把随机 seed 计入哈希：seed 每次实验都随机生成，一旦计入，
    #   “该配置此前出现过（review）”将几乎永远无法命中，配置级去重形同失效。
    #   这里只对“结构性配置”（研究方向 / 预算 / 模型参数优先序 / 标签定义）做哈希，
    #   使相同结构的实验能够被识别并标记为“复核”。
    key = json.dumps({"goal": plan.get("goal"), "budget": plan.get("budget"),
                      "param_priority": plan.get("param_priority"),
                      "label": plan.get("label")},
                     ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


# ====================================================================
# E. 指标打包 / 滚动窗口验证
# ====================================================================
def _summarize_seed_evidence(rolling):
    """把滚动验证结果整理成“可用于晋级的实际成功种子证据”。

    返回字段：
      configured / attempted / succeeded / n_seeds / details / has_details
      / known / complete / windows_planned / windows_attempted / windows_skipped / note

    语义约定（本批次 3A.1 起）：
      * configured：每个窗口配置的种子数（seeds_per_window / seeds_configured）；
      * attempted / succeeded：所有实际尝试窗口内的种子尝试总数 / 成功数；
        训练失败、回测失败都不计入 succeeded；
      * n_seeds：所有实际尝试窗口中“成功种子数”的最小值（不是跨窗口结果总数，
        也不是配置值）。只要有一个窗口成功数为 0，n_seeds 就是 0；
      * 跳过的计划窗口会以 skipped=True / skip_reason 记录，成功数为 0；
        它们不能被其他成功窗口掩盖；
      * 旧格式滚动结果没有 window_seed_details 时，无法可靠还原逐窗口成功数，
        known=False、n_seeds=0，禁止凭旧 n_seeds 或预算值晋级；
      * 完全没有滚动结果（rolling 为空）时 known=True、complete=True，
        由调用方用稳定性阶段的实际成功种子数补充。
    """
    roll = rolling or {}
    roll_provided = rolling is not None
    raw_details = roll.get("window_seed_details") or []
    details = []
    min_success = None
    attempted = 0
    succeeded = 0
    windows_skipped = 0
    configured = int(_num(roll.get("seeds_configured"), 0) or
                     _num(roll.get("seeds_per_window"), 0) or 0)

    for i, item in enumerate(raw_details):
        if not isinstance(item, dict):
            continue
        a = max(0, int(_num(item.get("attempted"), 0) or 0))
        s = max(0, int(_num(item.get("succeeded"), 0) or 0))
        if s > a:
            s = a
        fails = item.get("failures")
        if not isinstance(fails, list):
            fails = []
        skip_reason = str(item.get("skip_reason") or "")
        skipped = bool(item.get("skipped")) or bool(skip_reason and a == 0)
        seed_rets = item.get("seed_rets")
        if not isinstance(seed_rets, list):
            seed_rets = []
        # 防御不一致输入：最多保留实际成功数对应的收益值
        seed_rets = list(seed_rets[:s]) if s else []
        if skipped:
            s = 0
            seed_rets = []
            windows_skipped += 1
        details.append({
            "window": item.get("window", i + 1),
            "attempted": a,
            "succeeded": s,
            "failed": a - s,
            "failures": [str(x) for x in fails],
            "skipped": skipped,
            "skip_reason": skip_reason,
            "seed_rets": [float(x) for x in seed_rets if _num(x, None) is not None],
        })
        attempted += a
        succeeded += s
        # 跳过窗口也以 0 成功参与最小值，防止被其他窗口掩盖
        min_success = s if min_success is None else min(min_success, s)

    if details:
        planned = int(_num(roll.get("windows_planned"), 0) or len(details))
        attempted_windows = sum(1 for d in details if not d["skipped"])
        complete = (windows_skipped == 0 and attempted_windows == planned
                    and len(details) >= planned)
        note = ""
        if windows_skipped:
            skipped = [d for d in details if d["skipped"]]
            note = "；".join(
                f"窗口 {d['window']} 跳过：{d['skip_reason'] or '未说明原因'}"
                for d in skipped)
        return {
            "configured": configured,
            "attempted": attempted,
            "succeeded": succeeded,
            "n_seeds": int(min_success or 0),
            "details": details,
            "has_details": True,
            "known": True,
            "complete": bool(complete),
            "windows_planned": planned,
            "windows_attempted": attempted_windows,
            "windows_skipped": windows_skipped,
            "note": note,
        }

    # 旧格式兼容：没有逐窗口明细时，旧 n_seeds 可能只是配置值，不能当作
    # 实际成功数；也不能用跨窗口 seed_rets 总数猜测每窗口成功数。
    # 只要存在旧滚动证据但缺明细，就标记 known=False 并阻止晋级。
    legacy_rets = list(roll.get("seed_rets") or [])
    legacy_window_rets = list(roll.get("window_rets") or [])
    claimed = max(0, int(_num(roll.get("n_seeds"), 0) or 0))
    planned = int(_num(roll.get("windows_planned"), 0) or 0)
    attempted_meta = int(_num(roll.get("windows_attempted"), 0) or 0)
    has_legacy_evidence = bool(legacy_rets or legacy_window_rets or claimed
                               or planned or attempted_meta)
    if has_legacy_evidence:
        return {
            "configured": configured,
            "attempted": 0,
            "succeeded": 0,
            "n_seeds": 0,
            "details": [],
            "has_details": False,
            "known": False,
            "complete": False,
            "windows_planned": planned,
            "windows_attempted": attempted_meta,
            "windows_skipped": int(_num(roll.get("windows_skipped"), 0) or 0),
            "note": "旧滚动结果缺少 window_seed_details，无法还原实际成功种子证据",
        }
    if roll_provided:
        # 明确传入了一个空滚动结果对象，但没有逐窗口明细：
        # 不能证明它等同于“未执行滚动验证”，按证据未知处理。
        return {
            "configured": configured,
            "attempted": 0,
            "succeeded": 0,
            "n_seeds": 0,
            "details": [],
            "has_details": False,
            "known": False,
            "complete": False,
            "windows_planned": planned,
            "windows_attempted": attempted_meta,
            "windows_skipped": int(_num(roll.get("windows_skipped"), 0) or 0),
            "note": "滚动结果为空且缺少 window_seed_details，无法确认实际成功种子证据",
        }
    # 完全没有滚动证据（rolling=None/空）：不是“未知旧记录”，
    # 允许调用方用稳定性阶段的实际成功数补充。
    return {
        "configured": configured,
        "attempted": 0,
        "succeeded": 0,
        "n_seeds": 0,
        "details": [],
        "has_details": False,
        "known": True,
        "complete": True,
        "windows_planned": 0,
        "windows_attempted": 0,
        "windows_skipped": 0,
        "note": "",
    }


def build_metrics_from_results(results, rolling=None, budget=None, extra=None):
    """把 AutoRunner 的结果 + 滚动窗口结果，整理成统一的“验证段指标”。

    ⚠ 只使用训练期（stage 3~6 + 滚动窗口）的数据，绝不读取 results["test"]。

    种子证据口径（批次 3A.1）：
      * `seeds_configured` 是每窗口配置种子数；
      * `seeds_attempted` / `seeds_succeeded` 是滚动窗口实际尝试/成功总数；
      * `window_seed_details` 是逐窗口尝试、成功、失败、失败原因、跳过原因；
      * `n_seeds` 是晋级使用的保守值：所有实际尝试窗口中成功数的最小值；
        没有滚动窗口时，使用稳定性阶段实际成功的种子数；
        绝不使用 budget.model_seeds 或 seeds_per_window 补足缺失证据；
      * 旧滚动结果缺 `window_seed_details` 时，`seed_evidence_known=False`、
        `n_seeds=0`，晋级必须失败；
      * `seed_std_pct` 只取各窗口内种子收益标准差与稳定性阶段种子收益标准差
        的最大值，不混入跨窗口收益水平差异；窗口成功种子不足 2 个时为 None。
    """
    r = results or {}
    stab = r.get("stability") or {}
    strat = r.get("best_strategy") or {}
    mbest = r.get("model_search_best") or {}
    fbest = r.get("feature_search_best") or {}
    feats = list(r.get("best_features") or [])
    params = r.get("best_model_params") or {}
    roll = rolling or {}

    window_rets = list(roll.get("window_rets") or [])
    window_dds = list(roll.get("window_dds") or [])
    window_trades = list(roll.get("window_trades") or [])
    window_aucs = [a for a in (roll.get("window_aucs") or []) if a is not None and a == a]

    seed_ev = _summarize_seed_evidence(rolling)
    stab_seed_rets = list(stab.get("seed_rets") or [])
    stab_seed_count = len(stab_seed_rets)
    seed_rets = list(stab_seed_rets)
    seed_rets += list(roll.get("seed_rets") or [])
    seed_dds = list(stab.get("seed_dds") or []) + list(roll.get("seed_dds") or [])

    # 晋级证据：取各实际证据源的最小成功种子数。
    # 新旧滚动记录都必须可追溯；旧记录缺逐窗口明细时 known=False，
    # 即使旧 n_seeds 或预算看起来足够，也不能晋级。
    seed_known = bool(seed_ev["known"])
    seed_complete = bool(seed_ev["complete"])
    evidence_counts = []
    if seed_ev["has_details"]:
        evidence_counts.append(seed_ev["n_seeds"])
    elif seed_ev["known"]:
        # rolling 为空：没有滚动证据，允许稳定性阶段实际成功数补充。
        pass
    if stab_seed_rets:
        evidence_counts.append(stab_seed_count)
    actual_n_seeds = 0 if not seed_known else (
        min(evidence_counts) if evidence_counts else 0)
    if not seed_complete:
        actual_n_seeds = 0

    mean_ret = _mean(window_rets) if window_rets else _num(strat.get("val_ret_pct"), None)
    if mean_ret is None:
        mean_ret = _num(stab.get("mean_ret"), None)
    pos_ratio = (sum(1 for v in window_rets if v > 0) / len(window_rets)) if window_rets else 0.0

    # 种子标准差只衡量“同一验证窗口内不同随机种子”的收益差异。
    # 不把不同窗口的收益水平差异混入，避免把行情窗口差异误读成种子不稳定。
    # 任一实际尝试窗口成功种子不足 2 个时，无法可靠衡量该窗口的种子离散度，
    # 不借用其他窗口或稳定性阶段的值来证明它“稳定”。
    detail_windows = [d for d in seed_ev["details"] if not d.get("skipped")]
    std_ready = (int(seed_ev["windows_skipped"]) == 0
                 and all(len(d.get("seed_rets") or []) >= 2 for d in detail_windows))
    std_candidates = []
    if std_ready:
        for item in detail_windows:
            rets = list(item.get("seed_rets") or [])
            if len(rets) >= 2:
                std_candidates.append(_pstdev(rets))
        if len(stab_seed_rets) >= 2:
            std_candidates.append(_pstdev(stab_seed_rets))
    std = max(std_candidates) if (std_ready and std_candidates) else None

    metrics = {
        "mean_ret_pct": _r(mean_ret, 4),
        "max_dd_pct": _r(abs(min(window_dds)) if window_dds else abs(_num(strat.get("val_dd"), 0.0) * 100.0), 4),
        "mean_dd_pct": _r(_mean([abs(d) for d in window_dds]) if window_dds else abs(_num(strat.get("val_dd"), 0.0) * 100.0), 4),
        "window_rets": [_r(v, 4) for v in window_rets],
        "window_dds": [_r(abs(v), 4) for v in window_dds],
        "window_aucs": [_r(a, 4) for a in window_aucs],
        "n_windows": int(len(window_rets) if window_rets else (1 if mean_ret is not None else 0)),
        "seed_rets": [_r(v, 4) for v in seed_rets],
        "seed_std_pct": _r(std, 4),
        "n_seeds": int(actual_n_seeds),
        "seeds_configured": int(seed_ev["configured"] or
                                (int(budget.get("model_seeds") or 0) if budget else 0)),
        "seeds_per_window": int(seed_ev["configured"] or
                                (int(budget.get("model_seeds") or 0) if budget else 0)),
        "seeds_attempted": int(seed_ev["attempted"]),
        "seeds_succeeded": int(seed_ev["succeeded"]),
        "stability_seeds_succeeded": int(stab_seed_count),
        "windows_planned": int(seed_ev["windows_planned"]),
        "windows_attempted": int(seed_ev["windows_attempted"]),
        "windows_skipped": int(seed_ev["windows_skipped"]),
        "seed_evidence_known": bool(seed_known),
        "seed_evidence_complete": bool(seed_complete),
        "seed_evidence_note": str(seed_ev["note"] or ""),
        "window_seed_details": seed_ev["details"],
        "positive_window_ratio": _r(pos_ratio, 4),
        "mean_trades": _r(_mean(window_trades) if window_trades else _num(strat.get("val_trades"), 0.0), 3),
        "auc": _r(mbest.get("auc") if mbest.get("auc") is not None else fbest.get("auc"), 4),
        "acc": _r(mbest.get("acc") if mbest.get("acc") is not None else fbest.get("acc"), 4),
        "n_features": int(len(feats)),
        "max_depth": int(_num(params.get("max_depth"), 0)),
        "n_estimators": int(_num(params.get("n_estimators"), 0)),
        "features": feats,
        "strategy": strat,
        "model_params": dict(params),
        "params": {k: params.get(k) for k in ("max_depth", "learning_rate", "n_estimators",
                                             "subsample", "colsample_bytree", "reg_lambda")},
    }
    if extra:
        metrics.update(extra)
    return metrics


def baseline_strategy_metrics(dates, closes, ma_period=20, capital=100000.0,
                              friction=True, commission=0.0003, stamp=0.001, slippage=0.001,
                              run_backtest_fn=None):
    """不使用 AI 的对照基准：买入持有 / 均线突破。

    返回 {"hold_ret_pct":.., "ma_ret_pct":.., "ma_trades":.., "incremental_vs_hold_pp":..}
    """
    out = {"hold_ret_pct": None, "ma_ret_pct": None, "ma_trades": None,
           "incremental_vs_hold_pp": None}
    try:
        if run_backtest_fn is None or not closes:
            return out
        import pandas as pd
        c = pd.Series(list(closes), dtype="float64")
        hold_ret = (c.iloc[-1] / c.iloc[0] - 1) * 100.0 if len(c) > 1 else 0.0
        out["hold_ret_pct"] = _r(hold_ret, 4)
        ma = c.rolling(int(ma_period)).mean()
        actions = []
        for i in range(len(c)):
            m = ma.iloc[i - 1] if i > 0 else None
            prev = c.iloc[i - 1] if i > 0 else None
            if m is None or m != m or prev is None:
                actions.append(0)
            else:
                actions.append(1 if prev > m else -1)
        eng = run_backtest_fn(capital, False, 0.0, 0.0, int(ma_period), friction,
                             commission, stamp, slippage, list(dates), list(closes),
                             list(ma), actions, True)
        _, ret, _dd = eng.result()
        out["ma_ret_pct"] = _r(ret * 100.0, 4)
        out["ma_trades"] = int(eng.n_trades)
        out["incremental_vs_hold_pp"] = _r((out["ma_ret_pct"] or 0.0) - hold_ret, 4)
        return out
    except Exception:
        return out


def prepare_rolling_rows(train_rows):
    """为候选/基准制备完全相同的评估行。

    只保留 label/收盘 以及所有特征列都不为空的行。候选与基准即使使用不同
    特征子集，也都从这份公共行集合开始，避免因各自 dropna 造成实际评估样本不同。
    """
    try:
        rows = train_rows.copy()
        feature_cols = [c for c in rows.columns
                        if c not in ("日期", "收盘", "label")]
        return rows.dropna(subset=feature_cols + ["label", "收盘"]).reset_index(drop=True)
    except Exception:
        try:
            return train_rows.dropna(subset=["label", "收盘"]).reset_index(drop=True)
        except Exception:
            return train_rows


def build_rolling_window_specs(train_rows, n_windows=3, lookahead=5,
                               min_train_rows=120):
    """基于“公共行”（只要求 label/收盘 非空）生成日期窗口，供候选与基准共用。

    窗口边界不依赖特征集合，因此候选与基准即使使用不同特征，也能在同一批
    日期窗口上复评。返回列表；数据不足时返回空列表（调用方应记录跳过/不可比）。
    """
    import pandas as pd
    try:
        rows = train_rows.copy()
        rows = rows.sort_values("日期").reset_index(drop=True)
        rows = rows.dropna(subset=["label", "收盘"]).reset_index(drop=True)
        n = len(rows)
        n_windows = int(max(1, n_windows))
        if n < max(int(min_train_rows), n_windows * 20):
            return []
        seg = n // (n_windows + 1)
        if seg < 20:
            return []
        specs = []
        for w in range(1, n_windows + 1):
            tr = rows.iloc[(w - 1) * seg:w * seg]
            va = rows.iloc[w * seg:min(n, w * seg + seg)]
            if len(tr) == 0 or len(va) == 0:
                continue
            specs.append({
                "window": w,
                "train_start": _date_text(tr["日期"].iloc[0]),
                "train_end": _date_text(tr["日期"].iloc[-1]),
                "valid_start": _date_text(va["日期"].iloc[0]),
                "valid_end": _date_text(va["日期"].iloc[-1]),
            })
        return specs
    except Exception:
        return []


def rolling_window_metrics(gui, train_rows, feat_names, params, strategy,
                           n_windows=3, seeds_per_window=2, lookahead=5,
                           min_train_rows=120, stop_fn=None, progress_fn=None,
                           l2n=None, n2a=None, window_specs=None):
    """在训练期内部做【多窗口 + 多种子】滚动验证（因果：只用窗口之前的数据训练）。

    切分：把训练行按时间顺序分成 n_windows+1 段；第 k 段的模型只用它之前的数据训练，
    在第 k+1 段上验证（均线等特征只用该段自身收盘价 rolling，不引入未来信息）。
    每个训练窗口末尾再裁掉 lookahead 行，防止标签跨越窗口边界泄漏。

    返回 {"window_rets","window_dds","window_trades","window_aucs",
          "seed_rets","seed_dds","n_seeds","seeds_per_window","windows"}；数据不足时给出空结果。

    种子证据口径（批次 3A.1）：
      * seeds_per_window / seeds_configured：每窗口配置种子数；
      * seeds_attempted / seeds_succeeded：所有实际尝试窗口的尝试总数 / 成功数，
        只有“训练成功且回测成功”的种子才计入成功；
      * window_seed_details：逐窗口 {window, attempted, succeeded, failed, failures,
        skipped, skip_reason, seed_rets}；
      * windows_planned / windows_attempted / windows_skipped：计划窗口数与实际执行/跳过数；
      * n_seeds：所有实际尝试窗口中成功种子数的最小值（不是跨窗口结果总数，
        也不是配置值）；任何窗口成功数或跳过窗口都会使 n_seeds=0。
    """
    import pandas as pd

    L2N = dict(l2n or {-1: 0, 0: 1, 1: 2})
    N2A = dict(n2a or {0: -1, 1: 0, 2: 1})
    configured_seeds = int(max(1, seeds_per_window))
    out = {"window_rets": [], "window_dds": [], "window_trades": [],
           "window_aucs": [], "seed_rets": [], "seed_dds": [],
           "seeds_configured": configured_seeds,
           "seeds_per_window": configured_seeds,
           "seeds_attempted": 0,
           "seeds_succeeded": 0,
           "window_seed_details": [],
           "window_specs": [],
           "windows_planned": 0,
           "windows_attempted": 0,
           "windows_skipped": 0,
           "n_seeds": 0,
           "windows": 0, "note": ""}
    min_success = None

    def _skip_window(w, reason):
        """记录计划窗口被跳过；成功数按 0 参与晋级证据。"""
        nonlocal min_success
        out["windows_skipped"] += 1
        out["window_seed_details"].append({
            "window": w,
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "failures": [],
            "skipped": True,
            "skip_reason": str(reason),
            "seed_rets": [],
        })
        min_success = 0
        out["n_seeds"] = 0
        msg = f"窗口 {w} 跳过：{reason}"
        out["note"] = f"{out['note']}；{msg}" if out["note"] else msg

    try:
        rows = train_rows.copy()
        rows = rows.sort_values("日期").reset_index(drop=True)
        specs = _normalize_window_specs(window_specs) if window_specs else []
        if specs:
            # 候选与基准共用同一批日期窗口：不在切分前按特征 dropna，
            # 而是在每个窗口内部按各自特征过滤，保证窗口边界一致。
            out["windows_planned"] = len(specs)
            out["window_specs"] = [dict(x) for x in specs]
            n_windows = len(specs)
            spec_iter = specs
        else:
            rows = rows.dropna(subset=list(feat_names) + ["label", "收盘"]).reset_index(drop=True)
            n = len(rows)
            n_windows = int(max(1, n_windows))
            out["windows_planned"] = n_windows
            if n < max(int(min_train_rows), n_windows * 20):
                for w in range(1, n_windows + 1):
                    _skip_window(w, f"样本不足（{n} 行）")
                return out
            seg = n // (n_windows + 1)
            if seg < 20:
                for w in range(1, n_windows + 1):
                    _skip_window(w, f"每段样本过少（{seg} 行）")
                return out
            spec_iter = [None] * n_windows

        for spec_index, spec in enumerate(spec_iter, 1):
            if stop_fn is not None and stop_fn():
                out["note"] = "滚动验证被停止请求中断"
                return out
            if spec:
                w = int(spec.get("window") or spec_index)
                tr_all = rows[(rows["日期"] >= pd.to_datetime(spec["train_start"]))
                              & (rows["日期"] <= pd.to_datetime(spec["train_end"]))]
                va_all = rows[(rows["日期"] >= pd.to_datetime(spec["valid_start"]))
                              & (rows["日期"] <= pd.to_datetime(spec["valid_end"]))]
                tr = tr_all.dropna(subset=list(feat_names) + ["label", "收盘"]).copy().reset_index(drop=True)
                va = va_all.dropna(subset=list(feat_names) + ["label", "收盘"]).copy().reset_index(drop=True)
            else:
                w = spec_index
                tr_end = w * seg
                start_tr = max(0, tr_end - seg)
                # 训练窗口：只保留 start_tr .. tr_end（保证只用过去数据 + 固定长度窗口）
                tr = rows.iloc[start_tr:tr_end].copy()
                va = rows.iloc[tr_end: min(n, tr_end + seg)].copy()
            if len(tr) - lookahead < int(min_train_rows) or len(va) < 10:
                _skip_window(w, "训练或验证样本不足")
                continue
            tr = tr.iloc[:-lookahead].reset_index(drop=True)      # 裁掉跨界标签
            cut2 = int(len(tr) * 0.85)
            fit_core, es = tr.iloc[:cut2], tr.iloc[cut2:]
            if len(fit_core) < 40:
                _skip_window(w, "训练核心样本不足")
                continue
            if not spec:
                # 旧调用/无预生成 specs 时，记录实际使用的窗口边界供追溯。
                out["window_specs"].append({
                    "window": w,
                    "train_start": _date_text(tr["日期"].iloc[0]),
                    "train_end": _date_text(tr["日期"].iloc[-1]),
                    "valid_start": _date_text(va["日期"].iloc[0]),
                    "valid_end": _date_text(va["日期"].iloc[-1]),
                })
            if progress_fn:
                progress_fn(w, n_windows, f"滚动窗口 {w}/{n_windows}")
            w_rets, w_dds, w_trades, w_aucs = [], [], [], []
            w_attempted, w_succeeded, w_failures = 0, 0, []
            for s in range(configured_seeds):
                if stop_fn is not None and stop_fn():
                    out["note"] = "滚动验证被停止请求中断"
                    return out
                seed = 31337 + w * 101 + s * 7
                w_attempted += 1
                try:
                    model = gui_mod_fit(gui, fit_core[feat_names].values,
                                        fit_core["label"].map(L2N).values,
                                        es[feat_names].values if len(es) else None,
                                        es["label"].map(L2N).values if len(es) else None,
                                        params, seed)
                    pred = model.predict(va[feat_names].values)
                    try:
                        proba = model.predict_proba(va[feat_names].values)
                    except Exception:
                        proba = None
                except Exception as e:      # noqa: BLE001
                    out["note"] = f"窗口 {w} 训练失败：{e}"
                    w_failures.append(f"seed {seed} 训练失败：{e}")
                    continue
                acc, majority, auc, _has_edge, _note = gui.predictive_power(
                    va["label"].map(L2N).values, pred, proba)
                if auc == auc:
                    w_aucs.append(float(auc))
                ma_period = int(strategy.get("ma_period") or 20)
                ma_ser = list(va["收盘"].rolling(ma_period).mean())
                try:
                    eng = gui.run_backtest(
                        100000.0, True, float(strategy.get("stop_loss") or 5.0),
                        float(strategy.get("take_profit") or 15.0), ma_period,
                        bool(ROLLING_COST["friction"]),
                        float(ROLLING_COST["commission_rate"]),
                        float(ROLLING_COST["stamp_tax_rate"]),
                        float(ROLLING_COST["slippage_rate"]),
                        list(va["日期"]), list(va["收盘"]), ma_ser,
                        list(pd.Series(pred).map(N2A).values),
                        bool(strategy.get("ma_break_exit", True)))
                    _final, ret, dd = eng.result()
                    w_rets.append(float(ret) * 100.0)
                    w_dds.append(float(dd) * 100.0)
                    w_trades.append(int(eng.n_trades))
                except Exception as e:      # noqa: BLE001
                    out["note"] = f"窗口 {w} 回测失败：{e}"
                    w_failures.append(f"seed {seed} 回测失败：{e}")
                    continue
                # 只有训练和回测都成功的种子才计入证据
                w_succeeded += 1

            out["windows_attempted"] += 1
            out["seeds_attempted"] += w_attempted
            out["seeds_succeeded"] += w_succeeded
            out["window_seed_details"].append({
                "window": w,
                "attempted": w_attempted,
                "succeeded": w_succeeded,
                "failed": w_attempted - w_succeeded,
                "failures": w_failures,
                "skipped": False,
                "skip_reason": "",
                "seed_rets": [_r(v, 4) for v in w_rets],
            })
            # 晋级用 n_seeds = 各实际尝试窗口中成功数的最小值；
            # 任何窗口成功数为 0 都会把 n_seeds 拉到 0，不能被其他窗口掩盖。
            min_success = w_succeeded if min_success is None else min(min_success, w_succeeded)
            out["n_seeds"] = int(min_success)

            if w_rets:
                avg_ret = _mean(w_rets)
                out["window_rets"].append(_r(avg_ret, 4))
                out["window_dds"].append(_r(min(w_dds) if w_dds else 0.0, 4))
                out["window_trades"].append(_r(_mean(w_trades), 2))
                out["seed_rets"].extend([_r(v, 4) for v in w_rets])
                out["seed_dds"].extend([_r(v, 4) for v in w_dds])
                if w_aucs:
                    out["window_aucs"].append(_r(_mean(w_aucs), 4))
                out["windows"] += 1
        return out
    except Exception as e:      # noqa: BLE001
        out["note"] = f"滚动验证异常：{e}"
        return out


def gui_mod_fit(gui, X, y, X_es, y_es, params, seed):
    """调用主程序的拟合工具（auto_research.AutoRunner._fit_model 同款），保持口径一致。

    统一使用连续标签编码，避免缺失中间类别时 XGBoost 报 Invalid classes。
    """
    from auto_research import _fit_xgb_contiguous, N_JOBS
    return _fit_xgb_contiguous(X, y, X_es, y_es, params, seed, n_jobs=N_JOBS)


# ====================================================================
# F. 自主调度器
# ====================================================================
class AutonomousScheduler:
    """长期自主运行的研究调度器。

    只做三件事：挑股票 → 跑一次实验 → 按验收标准决定晋级并记录；然后依据历史调整方向。
    所有资源/状态都受 MemoryGuard + HeavyTaskGate + SQLite 检查点约束。
    """

    def __init__(self, registry, gui_module, emit=None, guard=None, cfg=None,
                 scope=None, runner_factory=None, universe_fetch_fn=None,
                 fetch_fn=None, data_dir=None, log_fn=None):
        self.reg = registry
        self.gui = gui_module
        self.emit = emit or (lambda *a, **k: None)
        self.cfg = dict(DEFAULT_ENGINE_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.scope = dict(getattr(sp, "DEFAULT_SCOPE", {})) if sp else {}
        if scope:
            self.scope.update(scope)
        self.runner_factory = runner_factory
        self.universe_fetch_fn = universe_fetch_fn
        self.fetch_fn = fetch_fn
        self.data_dir = data_dir or getattr(gui_module, "DATA_DIR", os.path.expanduser("~"))
        self.log_fn = log_fn
        self.guard = guard
        self.rules = load_promotion_rules(os.path.join(self.data_dir, "promotion_rules.json"))

        self._stop = False
        self._pause = False
        self._heavy_held = False
        self._pause_reason = ""
        self._resumed = False
        self._consec_fail = 0
        self._consec_universe_fail = 0
        self._current_code = ""
        self._current_exp = ""
        self._last_save = ""
        self._last_universe_try = 0.0
        self._cycle = 0
        self._counters = self.reg.get_state("counters", None) or {
            "experiments_done": 0, "experiments_failed": 0, "experiments_rejected": 0,
            "promotions": 0, "stable_upgrades": 0, "interrupted": 0}
        if not isinstance(self._counters, dict):
            self._counters = {"experiments_done": 0, "experiments_failed": 0,
                              "experiments_rejected": 0, "promotions": 0,
                              "stable_upgrades": 0, "interrupted": 0}
        # 崩溃恢复：上次运行中/计划中的实验 → 标记 interrupted（不假装成功）
        try:
            for rec in self.reg.unfinished_experiments():
                self.reg.update_experiment(rec["exp_id"], status="interrupted",
                                           finished_at=_now_str(),
                                           error="程序重启，未完成的实验已标记为中断（可重新研究）")
                self._counters["interrupted"] = int(self._counters.get("interrupted", 0)) + 1
            self._save_counters()
        except Exception:
            pass

    # ---------------- 控制接口 ----------------
    def request_stop(self, reason="用户停止"):
        self._stop = True
        self._set_loop_state("stopped", reason)

    def request_pause(self, reason="用户暂停"):
        self._pause = True
        self._resumed = False
        self._pause_reason = reason
        self._set_loop_state("paused", reason)
        self._log(f"⏸ 已请求暂停：{reason}（当前实验会在可控边界结束并保存）", "yellow")

    def request_resume(self):
        self._resumed = True
        self._pause = False
        self._pause_reason = ""
        self._set_loop_state("running", "用户恢复")
        self._log("▶ 已恢复运行：从最后一个有效检查点继续", "green")

    @property
    def stop_requested(self):
        return self._stop

    @property
    def pause_requested(self):
        return self._pause

    def _stop_fn_for_runner(self):
        # runner（AutoRunner）只认识 stop_fn：暂停/停止/内存危险 都走同一条“安全边界停止”路径
        def _f():
            if self._stop or self._pause:
                return True
            if self.guard is not None:
                try:
                    abort, why = self.guard.should_abort()
                    if abort:
                        self._pause = True
                        self._pause_reason = f"内存保护：{why}"
                        self._set_loop_state("paused", self._pause_reason)
                        return True
                except Exception:
                    pass
            return False
        return _f

    # ---------------- 全局重型任务闸门 ----------------
    def _heavy_gate(self):
        # 惰性取得全局闸门（与手动一键训练共用同一把锁），导入失败则退化为无闸门
        try:
            import memory_guard as mg
            return getattr(mg, "GLOBAL_HEAVY_GATE", None)
        except Exception:
            return None

    def _acquire_heavy_gate(self, timeout_s=300.0):
        """获取全局重型任务闸门：保证“同一时刻只有一个重型训练任务在跑”。

        尊重停止/暂停请求；等待 timeout_s 秒仍拿不到（说明已有别的重型任务占用）
        则返回 False，由主循环稍后重试，绝不会强行并行启动第二个训练。
        """
        gate = self._heavy_gate()
        if gate is None:
            return True
        t0 = time.time()
        while True:
            if self._stop or self._pause:
                return False
            try:
                if gate.try_acquire(owner=f"research:{self._current_code or '-'}"):
                    self._heavy_held = True
                    return True
            except Exception:
                return True
            if timeout_s is not None and (time.time() - t0) >= float(timeout_s):
                return False
            time.sleep(0.5)

    def _release_heavy_gate(self):
        if getattr(self, "_heavy_held", False):
            gate = self._heavy_gate()
            try:
                if gate is not None:
                    gate.release()
            except Exception:
                pass
            self._heavy_held = False

    # ---------------- 日志 / 状态 ----------------
    def _log(self, msg, tag=None):
        try:
            self.emit("log", str(msg), tag)
        except Exception:
            pass
        if callable(self.log_fn):
            try:
                self.log_fn(f"{msg}")
            except Exception:
                pass

    def _status(self, text):
        try:
            self.emit("status", str(text))
        except Exception:
            pass

    def _progress(self, cur, total, label=""):
        try:
            self.emit("progress", int(cur), int(total), str(label))
        except Exception:
            pass

    def _set_loop_state(self, status, reason=""):
        self.reg.set_state("loop", {
            "status": status, "reason": reason, "current_code": self._current_code,
            "current_exp": self._current_exp, "cycle": self._cycle,
            "updated_at": _now_str(), "last_save": self._last_save,
        })
        self._emit_state()

    def _emit_state(self):
        try:
            self.emit("auto_status", self.snapshot())
        except Exception:
            pass

    def _save_counters(self):
        self.reg.set_state("counters", self._counters)

    def snapshot(self):
        """给界面用的完整状态快照。"""
        counts = self.reg.pool_counts()
        exp = self.reg.experiment_counts()
        ds = self.reg.get_data_source_status()
        loop = self.reg.get_state("loop", {}) or {}
        guard_txt = ""
        if self.guard is not None:
            try:
                guard_txt = self.guard.status_text(sample=False)
            except Exception:
                guard_txt = ""
        best = None
        if self._current_code:
            b = self.reg.get_baseline(self._current_code)
            if b:
                best = {"code": self._current_code,
                        "mean_ret_pct": (b.get("metrics") or {}).get("mean_ret_pct"),
                        "updated_at": b.get("updated_at")}
        return {
            "pool": counts, "exp_counts": exp, "data_source": ds, "loop": loop,
            "counters": dict(self._counters), "current_code": self._current_code,
            "current_exp": self._current_exp, "guard": guard_txt,
            "best": best, "last_save": self._last_save, "rules": self.rules,
        }

    # ---------------- 主循环 ----------------
    def run_loop(self):
        """自主研究主循环。返回 (ok, 说明)。stop_requested → 正常退出；错误暂停 → ok=False。"""
        self._log("🔭 自主研究系统启动：自动发现股票 → 检查数据 → 排研究队列 → 迭代训练", "title")
        self._set_loop_state("running", "启动")
        done_this_session = 0
        guard = self.guard
        try:
            self._recover_state()
            while not self._stop:
                # --- 暂停等待 ---
                if self._pause:
                    self._wait_while_paused()
                    if self._stop:
                        break
                # --- 数据源刷新（按间隔或股票池为空）---
                self._maybe_refresh_universe()
                # --- 内存保护：不满足就等，不启动重型任务 ---
                if guard is not None:
                    ok, why = guard.can_start_heavy()
                    if not ok:
                        self._log(f"⚠ 内存保护：{why} → 暂不启动新的重型任务，等待资源恢复", "yellow")
                        self._set_loop_state("paused", f"内存保护：{why}")
                        good, note = guard.wait_for_safe(
                            max_wait_s=1800, poll_s=5.0, should_stop=lambda: self._stop)
                        if self._stop:
                            break
                        if not good:
                            self._set_loop_state("error_paused", f"内存保护：{note}")
                            self._log(f"⛔ {note}：进入错误暂停（内存恢复后可手动恢复）", "red")
                            return False, note
                        self._set_loop_state("running", "内存已恢复")
                        continue
                # --- 选股票 ---
                picked = self.reg.pick_next(count=1, scope=self.scope)
                if not picked:
                    self._recycle_cooldowns()
                    picked = self.reg.pick_next(count=1, scope=self.scope)
                if not picked:
                    self._set_loop_state("idle", "没有可研究的股票（等待队列/数据更新）")
                    self._status("暂无待研究股票，等待数据源更新或队列释放")
                    self._sleep(self.cfg["idle_sleep_s"])
                    continue
                stock = picked[0]["stock"]
                sched = picked[0]["schedule"]
                self._cycle += 1
                self._log(f"📌 本轮研究股票：{stock['code']} {stock.get('name','')}"
                          f"（调度原因：{sched.get('sequence','优先级')} {sched}）")
                # --- 全局重型任务闸门：确保同一时刻只有一个重型训练在跑 ---
                if not self._acquire_heavy_gate():
                    if self._stop or self._pause:
                        continue
                    self._log("⚠ 检测到其它重型任务正在运行，暂不启动新实验，稍后重试", "yellow")
                    self._sleep(5)
                    continue
                try:
                    ok_exp, note_exp = self._research_one(stock)
                finally:
                    self._release_heavy_gate()
                done_this_session += 1
                limit = int(self.cfg.get("max_experiments_per_session") or 0)
                if ok_exp:
                    self._consec_fail = 0
                else:
                    self._consec_fail += 1
                    if self._consec_fail >= int(self.cfg["max_consecutive_failures"]):
                        msg = (f"连续 {self._consec_fail} 次研究失败（最后一次：{note_exp}）→ "
                               f"进入错误暂停，请检查数据源/配置后手动恢复")
                        self._set_loop_state("error_paused", msg)
                        self._log(f"⛔ {msg}", "red")
                        return False, msg
                if limit and done_this_session >= limit:
                    self._log(f"已达到本次会话实验上限（{limit}），安全结束本次自主研究", "title")
                    self._set_loop_state("stopped", "达到会话实验上限")
                    return True, "达到会话实验上限"
                self._sleep(1)
            self._set_loop_state("stopped", "用户停止")
            self._log("⏹ 自主研究已停止（状态与记录已保存）", "title")
            return True, "用户停止"
        except Exception as e:      # noqa: BLE001
            tb = traceback.format_exc()
            self._set_loop_state("error_paused", f"调度器异常：{e}")
            self._log(f"⛔ 调度器异常：{e}\n{tb}", "red")
            return False, f"调度器异常：{e}"

    def _wait_while_paused(self):
        self._set_loop_state("paused", self._pause_reason or "用户暂停")
        self._log(f"⏸ 暂停中：{self._pause_reason or '用户暂停'}（不启动任何新实验）", "yellow")
        while self._pause and not self._stop:
            if self.guard is not None:
                try:
                    abort, why = self.guard.should_abort()
                    if abort:
                        pass
                except Exception:
                    pass
            time.sleep(float(self.cfg["pause_poll_s"]))
        if not self._stop:
            self._set_loop_state("running", "已恢复")

    def _sleep(self, seconds):
        """可被停止/暂停打断的等待。"""
        t0 = time.time()
        while (time.time() - t0) < float(seconds):
            if self._stop or self._pause:
                return
            time.sleep(min(0.5, float(seconds)))

    def _recover_state(self):
        """重启恢复：把冷却到期的股票重新放回队列；明确当前进度。"""
        now = datetime.now()
        n = 0
        for r in self.reg.list_stocks(status=sp.STATUS_COOLDOWN if sp else "cooldown"):
            lc = str(r.get("last_checked") or r.get("last_researched") or "")
            try:
                dt = datetime.strptime(lc[:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = None
            if dt is None or (now - dt) > timedelta(hours=float(self.scope.get("min_interval_hours", 6)) * 2):
                self.reg.set_status(r["code"], sp.STATUS_PENDING if sp else "pending",
                                    "冷却到期，排队等待重新检查")
                n += 1
        if n:
            self._log(f"♻ 重启恢复：{n} 只冷却中的股票已重新排队")
        loop = self.reg.get_state("loop", {}) or {}
        self._log(f"恢复检查点：上次状态={loop.get('status')}，当前股票={loop.get('current_code','-')}，"
                  f"实验={loop.get('current_exp','-')}")

    def _recycle_cooldowns(self):
        """队列空了 → 把冷却中的股票重新纳入（有限重试，不无限重试同一只）。"""
        if not sp:
            return
        n = 0
        for r in self.reg.list_stocks(status=sp.STATUS_COOLDOWN):
            if int(r.get("consecutive_fail") or 0) >= 5:
                continue
            self.reg.set_status(r["code"], sp.STATUS_PENDING, "队列空闲，重新检查数据")
            n += 1
        if n:
            self._log(f"♻ 队列空闲：{n} 只冷却股票重新排入待检查队列")

    # ---------------- 数据源 ----------------
    def _maybe_refresh_universe(self):
        hours = float(self.cfg["universe_refresh_hours"])
        now = time.time()
        counts = self.reg.pool_counts()
        need = (counts.get("total", 0) == 0) or \
               (now - float(self._last_universe_try or 0) >= hours * 3600)
        if not need:
            return
        self._last_universe_try = now
        self._status("正在更新股票候选池 ...")
        try:
            rows = sp.fetch_universe_rows(self.universe_fetch_fn,
                                          retries=self.scope.get("universe_retries", 3),
                                          log_fn=self._log)
            accepted, rejected = sp.filter_universe(rows, self.scope)
            limit = int(self.scope.get("max_stocks_per_cycle", 200))
            accepted = accepted[:limit]
            for item in accepted:
                self.reg.upsert_stock(item["code"], name=item["name"], board=item["board"],
                                      status=sp.STATUS_PENDING)
            for item in rejected[:500]:
                self.reg.upsert_stock(item["code"], name=item["name"], board=item["board"],
                                      status=sp.STATUS_REJECTED, reason=item.get("reason", ""))
            retired = self.reg.mark_missing_as_retired([r["code"] for r in rows])
            self._consec_universe_fail = 0
            self.reg.set_state("data_source_last_success", _now_str())
            self.reg.set_data_source_status(True, "", count=len(accepted))
            self._log(f"✅ 股票候选池更新：纳入 {len(accepted)} 只，排除 {len(rejected)} 只"
                      f"（本次上限 {limit}），退市标记 {retired} 只")
            self.reg.set_state("universe_last_success", _now_str())
        except Exception as e:      # noqa: BLE001
            self._consec_universe_fail += 1
            last_ok = self.reg.get_state("universe_last_success", "") or ""
            self.reg.set_data_source_status(False, str(e), last_success=last_ok)
            self._log(f"⚠ 股票列表获取失败（连续 {self._consec_universe_fail} 次）：{e}；"
                      f"最后成功更新：{last_ok or '从未成功'}；已保留现有队列，稍后重试", "yellow")
            if counts.get("total", 0) == 0 and \
                    self._consec_universe_fail >= int(self.cfg["max_consecutive_universe_failures"]):
                msg = (f"股票列表连续 {self._consec_universe_fail} 次获取失败且本地无队列 → 错误暂停"
                       f"（最后成功：{last_ok or '从未成功'}，错误：{e}）")
                self._set_loop_state("error_paused", msg)
                raise ResearchError(msg)
            # 已有队列 → 不终止系统，继续用旧队列

    # ---------------- 单只股票：检查 + 研究 ----------------
    def _ensure_stock_checked(self, stock):
        """惰性数据检查（只在轮到它研究时执行）。返回 (ok, info)。"""
        code = stock["code"]
        if stock.get("status") == sp.STATUS_READY and int(stock.get("data_rows") or 0) > 0:
            if stock.get("last_checked"):
                return True, {"status": sp.STATUS_READY, "reason": "已有质量数据"}
        fetch = self.fetch_fn or (lambda c, s, e: self.gui.fetch_stock_data(c, s, e))
        retries = int(self.scope.get("fetch_retries", 3))
        last = None
        for i in range(1, max(1, retries) + 1):
            if self._stop or self._pause:
                return False, {"status": "interrupted", "reason": "收到暂停/停止请求"}
            info = sp.check_stock_data(code, fetch, self.scope)
            last = info
            if info["status"] != sp.STATUS_COOLDOWN:
                break
            self._log(f"  {code} 数据检查未通过（第 {i}/{retries} 次）：{info['reason']}", "yellow")
            if i < retries:
                self._sleep(2 * i)
        self.reg.update_after_check(code, last)
        self._log(f"  数据检查结果：{last['status']} —— {last['reason']}")
        if last["status"] != sp.STATUS_READY:
            self.reg.update_after_research(code, success=False)
            return False, last
        return True, last

    def _train_window(self, code, info=None):
        """为每只股票计算训练/保留测试区间。

        保留测试窗口首次计算后持久化到 test_windows；后续实验必须复用同一窗口，
        避免随日期滚动或通过改日期绕过最终测试登记。这里只使用持久化窗口，
        不改变用户配置的训练年数语义。
        """
        today = datetime.now()
        reserved_days = int(self.cfg.get("reserved_test_days", 60))
        fixed_start = str(self.cfg.get("reserved_test_start") or "").strip()
        if fixed_start:
            try:
                test_start = datetime.strptime(fixed_start, "%Y-%m-%d")
            except Exception:
                test_start = today - timedelta(days=reserved_days)
        else:
            test_start = today - timedelta(days=reserved_days)
        test_end = today

        # 固定并持久化保留窗口：一旦登记，不再随 today 改变。
        try:
            existing = self.reg.get_test_window(code)
            if existing:
                try:
                    test_start = datetime.strptime(
                        str(existing.get("test_start"))[:10], "%Y-%m-%d")
                    test_end = datetime.strptime(
                        str(existing.get("test_end"))[:10], "%Y-%m-%d")
                except Exception:
                    pass
            else:
                _created, rec = self.reg.ensure_test_window(
                    code, test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                    note="自主研究首次生成并固定的保留测试窗口")
                try:
                    test_start = datetime.strptime(str(rec.get("test_start"))[:10], "%Y-%m-%d")
                    test_end = datetime.strptime(str(rec.get("test_end"))[:10], "%Y-%m-%d")
                except Exception:
                    pass
        except Exception as e:      # noqa: BLE001
            self._log(f"  ⚠ 固定测试窗口读取/登记失败，使用本次计算值：{e}", "yellow")

        train_end = test_start - timedelta(days=1)
        years = float(self.cfg.get("train_years", 5))
        train_start = train_end - timedelta(days=int(365.25 * years))
        # 与上市首日对齐（不改变保留测试区间）
        try:
            first = self.gui.get_first_data_date(code)
            if first:
                fd = datetime.strptime(str(first), "%Y-%m-%d")
                if fd > train_start:
                    train_start = fd
        except Exception:
            pass
        return (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

    def _research_one(self, stock):
        """对一只股票执行一轮完整研究。返回 (ok, 说明)。"""
        code = stock["code"]
        self._current_code = code
        ok, info = self._ensure_stock_checked(stock)
        if not ok:
            self._counters["experiments_failed"] = int(self._counters.get("experiments_failed", 0)) + 1
            self._save_counters()
            self._emit_state()
            return True, f"数据检查未通过，已记录：{info.get('reason')}"   # 不算系统失败
        stock = self.reg.get_stock(code) or stock

        history = self.reg.list_experiments(code=code, limit=200)
        direction = select_direction(history)
        goal = direction["goal"]
        train_start, train_end, test_start, test_end = self._train_window(code, info)
        plan = plan_experiment(goal, history, base_budget=self.cfg["base_budget"],
                               seed=abs(hash((code, len(history), _now_str()))) % 10 ** 6,
                               train_start=train_start, train_end=train_end,
                               test_start=test_start, test_end=test_end)
        exp_id = _make_exp_id(code)
        self._current_exp = exp_id
        budget = plan["budget"]
        rec = {
            "exp_id": exp_id, "code": code, "name": stock.get("name", ""), "goal": goal,
            "config_json": json.dumps(plan, ensure_ascii=False, default=str),
            "status": "running", "seed": plan["seed"],
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "created_at": _now_str(),
        }
        self.reg.add_experiment(rec)
        # 自主研究不执行最终测试，但把保留区间登记为 reserved；
        # 这不会消费测试数据，也便于和手工训练的最终测试共用同一登记。
        try:
            ok_res, rec_res = self.reg.reserve_test_window(
                code, test_start, test_end, exp_id=exp_id, trigger="自主研究",
                note="自主研究仅预留，未执行最终评估")
            if not ok_res:
                self._log(f"  ⚠ 保留测试区间已有使用记录，本次仍不执行测试："
                          f"{rec_res.get('status')} / {rec_res.get('exp_id')}", "yellow")
        except Exception as e:      # noqa: BLE001
            self._log(f"  ⚠ 保留测试区间登记失败（继续研究）：{e}", "yellow")
        self._set_loop_state("running", f"研究 {code}：{goal}")
        self._log(f"🧪 实验 {exp_id}｜方向：{goal}（{direction['reason']}）")
        self._log(f"   配置：预算{plan['budget']} 标签{plan.get('label')} "
                  f"模型优先序{plan['param_priority'][:4]} 复核={plan['review']}")

        attempts = int(self.cfg["max_experiment_retries"]) + 1
        last_err = ""
        for attempt in range(1, attempts + 1):
            if self._stop or self._pause:
                self.reg.update_experiment(exp_id, status="interrupted", finished_at=_now_str(),
                                           error="收到暂停/停止请求，实验在安全边界中断，状态已保存")
                self._log(f"⏸ 实验 {exp_id} 在安全边界中断（第 {attempt} 次尝试），已保存状态", "yellow")
                self._current_exp = ""
                return True, "已暂停/停止"
            try:
                metrics, err = self._execute_experiment(stock, plan, exp_id, budget)
                if err:
                    last_err = err
                    if attempt < attempts:
                        self._log(f"  实验异常（第 {attempt}/{attempts} 次）：{err}；"
                                  f"{int(self.cfg['retry_wait_s'])}s 后重试", "yellow")
                        self.reg.update_experiment(exp_id, retries=attempt, error=str(err))
                        self._sleep(self.cfg["retry_wait_s"])
                        continue
                    self.reg.update_experiment(exp_id, status="failed", retries=attempt - 1,
                                               error=str(err), finished_at=_now_str())
                    self._counters["experiments_failed"] = int(self._counters.get("experiments_failed", 0)) + 1
                    self.reg.update_after_research(code, success=False)
                    self._save_counters()
                    self._current_exp = ""
                    self._log(f"✖ 实验 {exp_id} 失败（已重试 {attempts - 1} 次）：{err}", "red")
                    return False, str(err)
                # ---- 正常完成：晋级判定 ----
                self._finish_experiment(stock, plan, exp_id, metrics, history)
                self._current_exp = ""
                return True, "完成"
            except Exception as e:      # noqa: BLE001
                last_err = e
                if attempt < attempts:
                    self.reg.update_experiment(exp_id, retries=attempt, error=str(e))
                    self._sleep(self.cfg["retry_wait_s"])
                    continue
                self.reg.update_experiment(exp_id, status="failed", retries=attempt - 1,
                                           error=str(e), finished_at=_now_str())
                self._counters["experiments_failed"] = int(self._counters.get("experiments_failed", 0)) + 1
                self.reg.update_after_research(code, success=False)
                self._save_counters()
                self._current_exp = ""
                self._log(f"✖ 实验 {exp_id} 异常：{e}\n{traceback.format_exc()}", "red")
                return False, str(e)

    def _execute_experiment(self, stock, plan, exp_id, budget):
        """跑一次完整研究（复用现有 AutoRunner 流水线，跳过独立测试）。返回 (metrics, err)。"""
        code = stock["code"]
        label = plan.get("label") or {}
        restore = self._apply_label_patch(label)
        t0 = time.time()
        try:
            cfg = {
                "ai_name": code, "code": code,
                "train_start": plan["train_start"], "train_end": plan["train_end"],
                "test_start": plan["test_start"], "test_end": plan["test_end"],
                "skip_final_test": True,
                "budget": budget,
                "param_priority": plan["param_priority"],
                "goal": plan["goal"],
                "trigger": "自主研究",
            }
            runner = self._make_runner(cfg)
            ok, results = runner.run()
            if not ok and str(results.get("status") or "") != "no_qualified":
                return None, str(getattr(runner, "last_error", "") or
                                 (results.get("ui_summary") or {}).get("aborted") or "研究未完成")
            # 无合格候选 → 合法的“负结果”，照常记录（不算系统失败）
            if str(results.get("status") or "") == "no_qualified":
                metrics = build_metrics_from_results(results, rolling=None, budget=budget)
                metrics["reason_no_candidate"] = results.get("no_qualified_reason", "")
                metrics["failure_reason"] = "训练期无合格候选策略"
                metrics["duration_s"] = round(time.time() - t0, 2)
                return metrics, None
            # ---- 滚动窗口验证（训练期内部，测试集不参与）----
            rolling = None
            try:
                rolling = self._rolling_for(results, plan)
            except Exception as e:      # noqa: BLE001
                self._log(f"  滚动验证异常（按证据不足处理）：{e}", "yellow")
            metrics = build_metrics_from_results(results, rolling=rolling, budget=budget)
            # 非 AI 对照基准（同一验证区间）
            try:
                extra = self._baseline_strategies(results, metrics)
                metrics.update(extra)
            except Exception:
                pass
            # 评估口径元数据 + 在同一批窗口上复评历史基准
            metrics["evaluation"] = self._evaluation_contract(
                rolling, results, metrics.get("features"))
            metrics["_comparison"] = self._build_comparison(
                code, results, metrics, budget)
            metrics["duration_s"] = round(time.time() - t0, 2)
            return metrics, None
        finally:
            self._restore_label_patch(restore)

    def _apply_label_patch(self, label):
        """按实验计划临时切换标签定义（只在 gui 模块常量的已验证范围内）。"""
        g = self.gui
        old = {}
        try:
            if not label:
                return old
            for k in ("LABEL_MODE", "LABEL_VOL_K"):
                if hasattr(g, k):
                    old[k] = getattr(g, k)
            if label.get("mode"):
                setattr(g, "LABEL_MODE", label["mode"])
            if label.get("vol_k") is not None and hasattr(g, "LABEL_VOL_K"):
                setattr(g, "LABEL_VOL_K", float(label["vol_k"]))
        except Exception:
            pass
        return old

    def _restore_label_patch(self, old):
        try:
            for k, v in (old or {}).items():
                setattr(self.gui, k, v)
        except Exception:
            pass

    def _make_runner(self, cfg):
        if self.runner_factory is not None:
            return self.runner_factory(cfg)
        from auto_research import AutoRunner
        return AutoRunner(cfg, self.emit, stop_fn=self._stop_fn_for_runner(),
                          state_dir=self.data_dir)

    def _rolling_for(self, results, plan):
        rows = results.get("train_rows_all")
        feats = results.get("best_features") or []
        params = results.get("best_model_params") or {}
        strat = results.get("best_strategy") or {}
        if rows is None or not feats or not strat:
            return None
        # 候选与基准共用同一份“所有特征列均非空”的公共行，保证实际评估样本一致。
        rows = prepare_rolling_rows(rows)
        if rows is None or len(rows) == 0:
            return None
        n_windows = int(self.cfg["rolling_windows"])
        min_train_rows = int(self.cfg["min_train_rows_rolling"])
        specs = build_rolling_window_specs(
            rows, n_windows=n_windows, lookahead=ROLLING_LOOKAHEAD,
            min_train_rows=min_train_rows)
        return rolling_window_metrics(
            self.gui, rows, feats, params, strat,
            n_windows=n_windows,
            seeds_per_window=int(self.cfg["rolling_seeds"]),
            lookahead=ROLLING_LOOKAHEAD,
            min_train_rows=min_train_rows,
            stop_fn=lambda: (self._stop or self._pause),
            window_specs=specs or None)

    def _label_evaluation_config(self):
        """记录当前生效的标签定义（候选实验期间临时 patch 后）。"""
        return {
            "mode": getattr(self.gui, "LABEL_MODE", None),
            "vol_k": getattr(self.gui, "LABEL_VOL_K", None),
            "horizon": int(getattr(self.gui, "LABEL_HORIZON", ROLLING_LOOKAHEAD)
                           or ROLLING_LOOKAHEAD),
        }

    def _evaluation_contract(self, rolling, results, features):
        specs = (rolling or {}).get("window_specs") or []
        rows = (results or {}).get("train_rows_all")
        common = prepare_rolling_rows(rows) if rows is not None else None
        return make_evaluation_contract(
            specs, self._label_evaluation_config(), ROLLING_COST,
            n_windows=len(specs) or int(self.cfg["rolling_windows"]),
            lookahead=ROLLING_LOOKAHEAD,
            min_train_rows=int(self.cfg["min_train_rows_rolling"]),
            seeds_per_window=int(self.cfg["rolling_seeds"]),
            features=features,
            data_start=(results or {}).get("train_start"),
            data_end=(results or {}).get("train_end"),
            n_rows=(len(common) if common is not None else 0),
        )

    def _reevaluate_config(self, config_metrics, results, candidate_eval, budget, source):
        """在候选当前窗口上复评基准配置，返回 (metrics, 原因)。

        只允许使用带完整评估元数据的基准配置；旧记录缺元数据时直接不可比。
        """
        if not isinstance(config_metrics, dict) or not config_metrics:
            return None, "基准配置指标为空"
        ok, why = evaluation_protocol_compatible(
            candidate_eval, config_metrics.get("evaluation"))
        if not ok:
            return None, why
        feats = list(config_metrics.get("features") or [])
        params = config_metrics.get("model_params") or config_metrics.get("params") or {}
        strat = config_metrics.get("strategy") or {}
        specs = (candidate_eval or {}).get("window_specs") or []
        rows = (results or {}).get("train_rows_all")
        if not feats or not params or not strat:
            return None, "基准配置缺少特征/模型参数/策略"
        if rows is None or not specs:
            return None, "缺少同窗口复评所需的训练数据或窗口边界"
        rows = prepare_rolling_rows(rows)
        if rows is None or len(rows) == 0:
            return None, "同窗口复评没有公共评估样本"
        missing = [c for c in feats if c not in rows.columns]
        if missing:
            return None, f"基准特征在当前数据中缺失：{','.join(missing[:5])}"
        try:
            roll = rolling_window_metrics(
                self.gui, rows, feats, params, strat,
                n_windows=len(specs),
                seeds_per_window=int((candidate_eval or {}).get("seeds_per_window")
                                     or self.cfg["rolling_seeds"]),
                lookahead=int((candidate_eval or {}).get("lookahead")
                              or ROLLING_LOOKAHEAD),
                min_train_rows=int((candidate_eval or {}).get("min_train_rows")
                                   or self.cfg["min_train_rows_rolling"]),
                stop_fn=lambda: (self._stop or self._pause),
                window_specs=specs)
        except Exception as e:      # noqa: BLE001
            return None, f"基准同窗口复评异常：{e}"
        # 基准复评不能借用候选的稳定性阶段指标，只使用同窗口滚动结果。
        baseline_results = {
            "train_rows_all": rows,
            "best_features": feats,
            "best_model_params": params,
            "best_strategy": strat,
            "stability": {},
            "model_search_best": {},
            "feature_search_best": {},
        }
        m = build_metrics_from_results(baseline_results, rolling=roll, budget=budget)
        # 复评使用与候选相同的评估协议；指纹因此一致。
        m["evaluation"] = dict(candidate_eval)
        m["evaluation"]["re_evaluated_from"] = source
        m["evaluation"]["evaluated_features"] = feats
        m["re_evaluated"] = True
        if not m.get("seed_evidence_known") or not m.get("seed_evidence_complete"):
            return None, "基准同窗口复评的窗口/种子证据不完整"
        if not m.get("n_windows"):
            return None, "基准同窗口复评没有有效窗口"
        return m, ""

    def _build_comparison(self, code, results, metrics, budget):
        """在候选当前滚动窗口上复评基准/稳定基准；返回可供晋级使用的对比结果。"""
        baseline_row = self.reg.get_baseline(code)
        if not baseline_row:
            return {"comparable": True, "baseline": None, "stable": None,
                    "reason": "无历史基准"}
        bm = baseline_row.get("metrics") or {}
        best = bm.get("best_validated") or bm
        stable = bm.get("stable")
        candidate_eval = metrics.get("evaluation")
        baseline_metrics, why = self._reevaluate_config(
            best, results, candidate_eval, budget, "best_validated")
        if baseline_metrics is None:
            return {"comparable": False, "baseline": None, "stable": None,
                    "reason": f"基准无法同窗口复评：{why}"}
        stable_metrics = None
        stable_reason = ""
        if stable:
            stable_metrics, stable_reason = self._reevaluate_config(
                stable, results, candidate_eval, budget, "stable")
            if stable_metrics is None:
                stable_metrics = {"_comparison_error":
                                  f"稳定基准无法同窗口复评：{stable_reason}"}
        return {"comparable": True, "baseline": baseline_metrics,
                "stable": stable_metrics, "reason": stable_reason}

    def _baseline_strategies(self, results, metrics):
        matrix = results.get("matrix_all")
        strat = results.get("best_strategy") or {}
        if matrix is None:
            return {}
        import pandas as pd
        tr = matrix[(matrix["日期"] >= pd.to_datetime(results.get("train_start")))
                    & (matrix["日期"] <= pd.to_datetime(results.get("train_end")))]
        if len(tr) < 30:
            return {}
        base = baseline_strategy_metrics(
            list(tr["日期"]), list(tr["收盘"]),
            ma_period=int(strat.get("ma_period") or 20),
            run_backtest_fn=self.gui.run_backtest)
        base["incremental_vs_ai_pp"] = None
        if metrics.get("mean_ret_pct") is not None and base.get("hold_ret_pct") is not None:
            base["incremental_vs_ai_pp"] = _r(metrics["mean_ret_pct"] - base["hold_ret_pct"], 4)
        return base

    # ---------------- 实验收尾：晋级 / 记录 / 方向 ----------------
    def _finish_experiment(self, stock, plan, exp_id, metrics, history):
        code = stock["code"]
        n_prior = len(history)
        baseline_row = self.reg.get_baseline(code)
        comparison = metrics.pop("_comparison", None)
        baseline_metrics = None
        stable_metrics = None
        if baseline_row:
            if not comparison:
                comparison = {"comparable": False,
                              "reason": "缺少同窗口复评记录，不能使用历史基准指标"}
            if not comparison.get("comparable"):
                err = comparison.get("reason") or "基准与候选评估不可比"
                metrics["_comparison_error"] = err
                baseline_metrics = {"_comparison_error": err}
                stable_metrics = {"_comparison_error": err}
                metrics["baseline_comparison"] = {"status": "incomparable",
                                                  "reason": err}
            else:
                baseline_metrics = comparison.get("baseline")
                stable_metrics = comparison.get("stable")
                if baseline_metrics is None:
                    err = "同窗口复评未产出基准指标"
                    metrics["_comparison_error"] = err
                    baseline_metrics = {"_comparison_error": err}
                    stable_metrics = {"_comparison_error": err}
                    metrics["baseline_comparison"] = {"status": "incomparable",
                                                      "reason": err}
                else:
                    metrics["baseline_comparison"] = {
                        "status": "comparable",
                        "reason": comparison.get("reason") or "",
                    }
                    if (isinstance(stable_metrics, dict)
                            and stable_metrics.get("_comparison_error")):
                        metrics["baseline_comparison"]["stable_reason"] = \
                            stable_metrics.get("_comparison_error")
        else:
            metrics["baseline_comparison"] = {"status": "no_baseline",
                                              "reason": "首次建立基准"}
        verdict = evaluate_promotion(metrics, baseline_metrics, self.rules,
                                     n_experiments=n_prior)
        promoted = bool(verdict["promote"])
        next_dir = select_direction(history + [{
            "goal": plan["goal"], "status": "done" if not metrics.get("reason_no_candidate") else "rejected",
            "promoted": promoted, "reject_reason": "；".join(verdict["reasons"]),
            "created_at": _now_str()}])["goal"]

        model_path = ""
        stable_upgraded = False
        if promoted:
            try:
                model_path = self._promote_model(stock, metrics, exp_id)
            except Exception as e:      # noqa: BLE001
                self._log(f"  ⚠ 晋级模型保存失败（本次不计入基准）：{e}", "red")
                promoted = False
                verdict["reasons"].append(f"模型保存失败：{e}")
        if promoted:
            payload = {
                "best_validated": metrics,
                "stable": stable_metrics or {},
                "validation_only": True,
                "note": "全部指标来自训练期滚动验证；独立测试集未参与任何选择",
            }
            up_ok, up_why = evaluate_stable_upgrade(metrics, stable_metrics, self.rules)
            if up_ok or stable_metrics is None:
                payload["stable"] = metrics
                stable_upgraded = True
            self.reg.set_baseline(code, payload, model_path=model_path,
                                  features=",".join(metrics.get("features") or []),
                                  params=metrics.get("params") or {}, source_exp=exp_id)
            self._counters["promotions"] = int(self._counters.get("promotions", 0)) + 1
            if stable_upgraded:
                self._counters["stable_upgrades"] = int(self._counters.get("stable_upgrades", 0)) + 1
            self._log(f"🏆 晋级成功：{code}｜验证段平均收益 {_fmt_n(metrics.get('mean_ret_pct'))}% "
                      f"（窗口 {metrics.get('n_windows')}，种子 {metrics.get('n_seeds')}，"
                      f"得分 {verdict['candidate_score']}）"
                      + ("；同时升级为稳定基准" if stable_upgraded else ""), "green")
        else:
            self._counters["experiments_rejected"] = int(self._counters.get("experiments_rejected", 0)) + 1
            self._log(f"❎ 未晋级（保留原最佳版本）：{'；'.join(verdict['reasons']) or '证据不足'}", "yellow")
        if not metrics.get("reason_no_candidate"):
            self._counters["experiments_done"] = int(self._counters.get("experiments_done", 0)) + 1
        self._save_counters()

        # ---- 落盘实验记录（可追踪：配置/区间/种子/指标/内存/耗时/是否晋级/原因/下一方向）----
        snap = {}
        if self.guard is not None:
            try:
                snap = self.guard.snapshot()
            except Exception:
                snap = {}
        metrics["memory_mb"] = float(snap.get("proc_mb") or 0.0)
        self.reg.update_experiment(
            exp_id,
            status=("promoted" if promoted else ("rejected" if not metrics.get("reason_no_candidate") else "done")),
            promoted=promoted,
            reject_reason=("" if promoted else "；".join(verdict["reasons"]) or
                           metrics.get("failure_reason", "证据不足")),
            next_direction=next_dir,
            metrics_json=json.dumps(metrics, ensure_ascii=False, default=str),
            features=",".join(metrics.get("features") or []),
            params_json=json.dumps(metrics.get("params") or {}, ensure_ascii=False, default=str),
            memory_mb=float(snap.get("proc_mb") or 0.0),
            duration_s=float(metrics.get("duration_s") or 0.0),
            finished_at=_now_str())
        self._append_experiment_csv(stock, plan, exp_id, metrics, verdict, promoted, next_dir)
        self.reg.update_after_research(
            code, success=True, promoted=promoted,
            new_data_date=str(stock.get("last_data_date") or ""))
        self._last_save = _now_str()
        self._set_loop_state("running", f"实验 {exp_id} 完成（{'晋级' if promoted else '未晋级'}）")
        self._progress(1, 1, f"{code} {plan['goal']} 完成")
        self._emit_state()

    def _promote_model(self, stock, metrics, exp_id):
        """把晋级模型保存成兼容命名（model_{AI}_Best.json），原子替换，绝不覆盖旧基准文件。"""
        code = stock["code"]
        src = os.path.join(self.data_dir, "experiments", exp_id, "model.json")
        if not os.path.exists(src):
            raise ResearchError("实验模型文件不存在，无法晋级")
        dst = os.path.join(self.data_dir, f"model_{code}_Best.json")
        tmp = dst + ".tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        # 稳定基准模型（第一个通过稳定门槛的版本）单独留一份
        try:
            stable_dst = os.path.join(self.data_dir, f"model_{code}.json")
            if not os.path.exists(stable_dst):
                tmp2 = stable_dst + ".tmp"
                shutil.copy2(src, tmp2)
                os.replace(tmp2, stable_dst)
        except Exception:
            pass
        self._log(f"  💾 晋级模型已保存：{os.path.basename(dst)}（来源实验 {exp_id}）")
        return dst

    def _append_experiment_csv(self, stock, plan, exp_id, metrics, verdict, promoted, next_dir):
        """人类可读的实验台账（CSV，增量追加，原子安全）。"""
        path = os.path.join(self.data_dir, "autonomous_experiments.csv")
        header = ["时间", "股票代码", "股票名称", "实验编号", "研究目标", "执行状态",
                  "数据起止", "特征集合", "模型参数", "随机种子", "验证窗口数", "种子数",
                  "验证段平均收益%", "最差窗口回撤%", "平均交易次数", "AUC", "候选得分",
                  "基准得分", "是否晋级", "拒绝/失败原因", "下一轮方向", "内存MB", "耗时秒"]
        row = [_now_str(), stock["code"], stock.get("name", ""), exp_id, plan["goal"],
               "promoted" if promoted else ("no_candidate" if metrics.get("reason_no_candidate") else "rejected"),
               f"{plan['train_start']}~{plan['train_end']}",
               ",".join(metrics.get("features") or []),
               json.dumps(metrics.get("params") or {}, ensure_ascii=False),
               plan["seed"], metrics.get("n_windows"), metrics.get("n_seeds"),
               _fmt_n(metrics.get("mean_ret_pct")), _fmt_n(metrics.get("max_dd_pct")),
               _fmt_n(metrics.get("mean_trades")), _fmt_n(metrics.get("auc")),
               _fmt_n(verdict.get("candidate_score")), _fmt_n(verdict.get("baseline_score")),
               "是" if promoted else "否",
               "" if promoted else ("；".join(verdict["reasons"]) or metrics.get("failure_reason", "")),
               next_dir, _fmt_n(metrics.get("memory_mb")), _fmt_n(metrics.get("duration_s"))]
        atomic_append_csv(path, row, header=header)


# ====================================================================
# 小工具
# ====================================================================
def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_exp_id(code):
    return f"{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 0xFFFF):04x}"


def _num(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _r(v, nd=4):
    try:
        if v is None:
            return None
        f = round(float(v), nd)
        return f
    except Exception:
        return None


def _mean(xs):
    xs = [x for x in (xs or []) if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pstdev(xs):
    xs = [float(x) for x in (xs or []) if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _fmt_n(v, nd=2):
    try:
        if v is None:
            return "—"
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)
