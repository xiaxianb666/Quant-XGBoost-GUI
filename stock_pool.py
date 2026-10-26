# -*- coding: utf-8 -*-
"""
stock_pool.py — 【自主发现股票 + 数据检查 + 研究队列】

职责（对应需求第三、四节）：
  1. 从 AkShare 公开行情数据自动获取沪深 A 股代码+名称（候选池），绝不要求人工填代码；
  2. 用明确的规则排除/降级不适合研究的股票（ST、退市风险、上市过短、特殊板块、数据缺失、
     长期停牌、流动性不足、异常价格）；
  3. 惰性检查：默认不把整个股票池的历史行情一次性拉进内存，只有轮到它研究时才检查；
  4. 轻量可恢复存储：SQLite（股票注册表 / 研究队列 / 实验记录 / 基准 / 运行状态）；
  5. 公平调度 + 优先级：既不让热门股票永久占据队列，也不让某些股票永远得不到机会。

本模块不训练模型、不写 Tkinter、不执行任何用户命令；所有网络访问都可注入（便于离线测试）。
"""

import os
import math
import sqlite3
import threading
from datetime import datetime, timedelta

try:
    from safe_io import atomic_write_json, read_json_safe
except Exception:      # 兜底：允许单文件运行
    atomic_write_json = lambda *a, **k: False
    read_json_safe = lambda *a, **k: None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# AkShare 异常诊断文件（项目内独立文件；pythonw 下不依赖 stdout/stderr）
AKSHARE_TRACE_FILE = os.path.join(BASE_DIR, "_akshare_traceback.txt")


# ====================================================================
# 0. 默认研究范围与过滤规则（可配置，默认沪深 A 股，无需人工选股）
# ====================================================================
DEFAULT_SCOPE = {
    # ---- 板块范围（默认覆盖沪深主板 + 创业板；科创板/北交所默认排除，可打开）----
    "include_main_sh": True,     # 沪市主板 60xxxx
    "include_main_sz": True,     # 深市主板 00xxxx
    "include_gem": True,         # 创业板 30xxxx
    "include_star": False,       # 科创板 68xxxx（20% 涨跌幅，默认排除）
    "include_bj": False,         # 北交所（默认排除）
    # ---- 名称规则 ----
    "exclude_st": True,          # ST / *ST / 退市风险
    "exclude_delisting": True,   # 名称含“退”
    # ---- 数据质量门槛 ----
    "lookback_days": 420,        # 惰性检查时回看多少自然日
    "min_rows": 220,             # 最少有效日线
    "max_missing_ratio": 0.10,   # 交易日缺失率上限
    "max_zero_vol_ratio": 0.25,  # 零成交占比上限
    "min_listed_days": 250,      # 上市天数下限（不足直接排除）
    "max_stale_days": 30,        # 行情最后一天距今超过它就视为停牌/无有效成交
    "min_avg_amount_wan": 2000,  # 日均成交额下限（万元），流动性不足则降级
    # ---- 调度 ----
    "min_interval_hours": 6,     # 同一只股票两次研究的最小间隔（防反复研究同一只）
    "max_stocks_per_cycle": 200, # 单次刷新最多处理多少只（保护内存）
    # ---- 优先级权重 ----
    "weights": {
        "quality": 3.0, "samples": 1.0, "liquidity": 1.5,
        "gap": 2.0, "fresh": 1.0, "cost": 0.6, "fail": 1.5,
    },
    # ---- 重试 ----
    "fetch_retries": 3,          # 单只股票行情获取失败重试次数
    "universe_retries": 3,       # 股票列表获取失败重试次数
}

STATUS_PENDING = "pending"     # 待检查（刚发现，还没有行情质量信息）
STATUS_READY = "ready"         # 可研究
STATUS_COOLDOWN = "cooldown"   # 暂时失败，等待重试
STATUS_REJECTED = "rejected"   # 明确不合格（数据不足/流动性差/ST 等）
STATUS_RETIRED = "retired"     # 已退市 / 已从列表消失

# 最终保留测试的使用状态。reserved 只是登记，不阻止后续正式评估；
# 其余状态代表测试数据已经被消费或正在消费，默认禁止无提示重复执行。
TEST_USAGE_RESERVED = "reserved"
TEST_USAGE_RUNNING = "running"
TEST_USAGE_COMPLETED = "completed"
TEST_USAGE_FAILED = "failed"
TEST_USAGE_ABORTED = "aborted"
TEST_USAGE_BLOCKING = (TEST_USAGE_RUNNING, TEST_USAGE_COMPLETED,
                       TEST_USAGE_FAILED, TEST_USAGE_ABORTED, "used")


class UniverseError(Exception):
    """股票列表获取失败（数据源不可用）。"""


def _now():
    return datetime.now()


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _digits(s, n=6):
    d = "".join(ch for ch in str(s) if ch.isdigit())
    return d if len(d) == n else None


# ====================================================================
# 1. 股票池发现（AkShare）
# ====================================================================
def default_universe_fetcher():
    """默认数据源：AkShare 全市场股票代码+名称（一次请求，只拿代码/名称，不拿历史行情）。"""
    import akshare as ak
    import sys
    import io
    import contextlib

    # 【防御】在 GUI 环境（stdout/stderr 为 None）下，拦截 akshare 内部的 tqdm 进度条输出
    # 防止 tqdm.write() 调用失败导致崩溃
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()

    try:
        # 检测并拦截标准输出流
        if sys.stdout is None or not hasattr(sys.stdout, 'write'):
            sys.stdout = fake_stdout
        if sys.stderr is None or not hasattr(sys.stderr, 'write'):
            sys.stderr = fake_stderr

        try:
            df = ak.stock_info_a_code_name()
        except Exception as e:      # noqa: BLE001 —— akshare 内部异常（如网络/缓存）统一包装
            # 【防御】清除 lru_cache，避免缓存损坏状态导致后续重试持续失败
            try:
                ak.stock_info_a_code_name.cache_clear()
            except Exception:
                pass
            # 【诊断】把"当时看到的"完整 traceback 写入项目内独立文件。
            # 该记录只用于后续定位：现有生产 traceback 可能不完整，且不能据此
            # 断定 AkShare 内部就是根因。写日志失败绝不能掩盖原始异常。
            try:
                _write_akshare_diagnostic(e)
            except Exception:
                pass
            raise UniverseError(f"AkShare 股票列表异常（{type(e).__name__}）：{e}") from e
        if df is None or len(df) == 0:
            raise UniverseError("AkShare 返回空股票列表")
        return df
    finally:
        # 【恢复】无论成功失败，都恢复标准输出流
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup
        # 清空假输出流（防止污染）
        fake_stdout.truncate(0)
        fake_stdout.seek(0)
        fake_stderr.truncate(0)
        fake_stderr.seek(0)


def _err_desc(e):
    """把异常转成含类型名的可读描述（str(e) 可能为空，类型名不能丢）。"""
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


def _date_ordinal(value):
    """把 YYYY-MM-DD 转成可比较序数；非法日期返回 None。"""
    try:
        return datetime.strptime(str(value or "").strip()[:10], "%Y-%m-%d").toordinal()
    except Exception:
        return None


def _ranges_overlap(start_a, end_a, start_b, end_b):
    """判断两个完整日期区间是否重叠（含边界）。"""
    a1, a2 = _date_ordinal(start_a), _date_ordinal(end_a)
    b1, b2 = _date_ordinal(start_b), _date_ordinal(end_b)
    if None not in (a1, a2, b1, b2):
        return a1 <= b2 and b1 <= a2
    # 日期格式不标准时退化为字符串比较，仍避免明显重叠
    sa, ea = str(start_a or ""), str(end_a or "")
    sb, eb = str(start_b or ""), str(end_b or "")
    return sa <= eb and sb <= ea


def _err_traceback(e):
    """返回异常的完整堆栈字符串（用于日志诊断，不抛异常）。"""
    import traceback
    try:
        return "".join(traceback.format_exception(type(e), e, e.__traceback__))
    except Exception:       # noqa: BLE001
        return ""


def _write_akshare_diagnostic(exc):
    """把 AkShare 异常堆栈追加到项目内诊断文件；返回是否成功，绝不抛异常。

    仅记录调用处当时能取得的信息。文件写入失败或堆栈不完整时，调用方仍会
    抛出原始异常（UniverseError 链），不会因日志问题改变故障语义。
    """
    import traceback
    try:
        header = f"\n===== {datetime.now()} =====\n"
        tb_text = ""
        try:
            tb_text = traceback.format_exc()
        except Exception:
            tb_text = ""
        if not tb_text.strip():
            try:
                tb_text = f"{type(exc).__name__}: {exc}\n（无可用堆栈）\n"
            except Exception:
                tb_text = "（无可用的异常信息）\n"
        try:
            with open(AKSHARE_TRACE_FILE, "a", encoding="utf-8",
                      errors="backslashreplace") as f:
                f.write(header)
                f.write(tb_text)
                if not tb_text.endswith("\n"):
                    f.write("\n")
            return True
        except Exception:
            return False
    except Exception:
        return False


def fetch_universe_rows(fetch_fn=None, retries=None, sleep_fn=None, log_fn=None):
    """获取全市场股票列表。失败有限重试，超限抛 UniverseError（绝不假装发现了新股票）。

    返回 [{code, name}, ...]。
    """
    fn = fetch_fn or default_universe_fetcher
    tries = int(retries if retries is not None else DEFAULT_SCOPE["universe_retries"])
    tries = max(1, tries)
    log = log_fn if callable(log_fn) else (lambda *a, **k: None)
    last_err = None
    for i in range(1, tries + 1):
        try:
            df = fn()
            rows = _normalize_universe_df(df)
            if not rows:
                raise UniverseError("股票列表为空（字段解析失败或数据源异常）")
            return rows
        except Exception as e:      # noqa: BLE001 —— 数据源任何异常都按“获取失败”处理
            last_err = e
            # 日志记录本身失败绝不能掩盖原始数据源异常
            try:
                log(f"股票列表获取失败（第 {i}/{tries} 次）：{_err_desc(e)}")
                tb = _err_traceback(e)
                if tb:
                    log(f"  堆栈：{tb.strip().splitlines()[-1] if tb.strip() else ''}")
            except Exception:       # noqa: BLE001
                pass
            if i < tries:
                delay = min(30, 2 ** i)
                (sleep_fn or (lambda s: None))(delay) if sleep_fn else None
    raise UniverseError(f"股票列表获取失败（已重试 {tries} 次）：{_err_desc(last_err)}")


def _normalize_universe_df(df):
    """把数据源 DataFrame 归一化成 [{code,name}]（兼容中英文列名）。"""
    if df is None:
        return []
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return []
    cols = {str(c).strip().lower(): c for c in df.columns}
    c_code = cols.get("code") or cols.get("代码") or cols.get("symbol") or cols.get("ts_code")
    c_name = cols.get("name") or cols.get("名称") or cols.get("股票简称") or cols.get("简称")
    if c_code is None:
        return []
    out = []
    for _, r in df.iterrows():
        code = _digits(r[c_code])
        if not code:
            continue
        name = "" if c_name is None else str(r[c_name]).strip()
        out.append({"code": code, "name": name})
    return out


# ====================================================================
# 2. 板块识别 / 基础过滤（纯函数，可测试）
# ====================================================================
def classify_board(code):
    """返回板块名：主板(沪)/主板(深)/创业板/科创板/北交所/其他。"""
    code = _digits(code) or str(code)
    if code.startswith("60"):
        return "主板(沪)"
    if code.startswith("00"):
        return "主板(深)"
    if code.startswith("30"):
        return "创业板"
    if code.startswith("68"):
        return "科创板"
    if code.startswith(("43", "83", "87", "88", "92", "8", "4")):
        return "北交所"
    return "其他"


def basic_filter(code, name, scope=None):
    """代码/名称级过滤（不联网）。返回 (是否保留, 原因)。"""
    s = dict(DEFAULT_SCOPE)
    if scope:
        s.update(scope)
    code = _digits(code)
    if not code:
        return False, "代码非法"
    nm = str(name or "").strip()
    nm_upper = nm.upper()
    board = classify_board(code)
    if s.get("exclude_delisting") and ("退" in nm):
        return False, "退市/退市风险"
    if s.get("exclude_st") and ("ST" in nm_upper or "*" in nm):
        return False, "ST/*ST（默认不纳入研究池）"
    if board == "主板(沪)" and not s.get("include_main_sh"):
        return False, "板块已排除：沪市主板"
    if board == "主板(深)" and not s.get("include_main_sz"):
        return False, "板块已排除：深市主板"
    if board == "创业板" and not s.get("include_gem"):
        return False, "板块已排除：创业板"
    if board == "科创板" and not s.get("include_star"):
        return False, "板块已排除：科创板（默认需专门数据能力）"
    if board == "北交所" and not s.get("include_bj"):
        return False, "板块已排除：北交所"
    if board == "其他":
        return False, "非沪深 A 股（特殊证券）"
    return True, "通过基础过滤"


def filter_universe(rows, scope=None):
    """对股票列表做基础过滤。返回 (accepted, rejected)，元素都是 dict。"""
    accepted, rejected = [], []
    seen = set()
    for r in rows or []:
        code = _digits(r.get("code"))
        if not code or code in seen:
            continue
        seen.add(code)
        name = str(r.get("name", "")).strip()
        ok, why = basic_filter(code, name, scope)
        item = {"code": code, "name": name, "board": classify_board(code)}
        if ok:
            accepted.append(item)
        else:
            item["reason"] = why
            rejected.append(item)
    return accepted, rejected


# ====================================================================
# 3. 单只股票数据质量检查（惰性调用，写进注册表）
# ====================================================================
def check_stock_data(code, fetch_fn, scope=None, today=None):
    """检查单只股票的历史行情质量。fetch_fn(code, start_yyyymmdd, end_yyyymmdd) -> DataFrame。

    返回 dict：
      status   : ready / rejected / cooldown
      reason   : 人类可读原因
      rows / last_date / missing_ratio / zero_vol_ratio / avg_amount_wan / quality / listed_days
    绝不抛异常（数据源异常 → status=cooldown，交给调度器有限重试）。
    """
    s = dict(DEFAULT_SCOPE)
    if scope:
        s.update(scope)
    today = today or _now()
    lookback = int(s["lookback_days"])
    start = (today - timedelta(days=lookback)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    out = {"status": STATUS_COOLDOWN, "reason": "", "rows": 0, "last_date": "",
           "missing_ratio": 1.0, "zero_vol_ratio": 1.0, "avg_amount_wan": 0.0,
           "quality": 0.0, "listed_days": 0}

    df = None
    try:
        df = fetch_fn(code, start, end)
    except Exception as e:          # noqa: BLE001
        out["reason"] = f"行情获取失败：{e}"
        return out
    if df is None or len(df) == 0:
        out["reason"] = "无有效历史行情"
        return out
    try:
        import pandas as pd
        df = df.copy()
        col_date = "日期" if "日期" in df.columns else df.columns[0]
        df[col_date] = pd.to_datetime(df[col_date])
        df = df.sort_values(col_date).reset_index(drop=True)
        n = int(len(df))
        first_d = df[col_date].iloc[0]
        last_d = df[col_date].iloc[-1]
        out["rows"] = n
        out["last_date"] = str(last_d.date())
        listed_days = int((today - first_d.to_pydatetime()).days)
        out["listed_days"] = max(0, listed_days)

        # 交易日缺失率：用区间内工作日估算应有多少个交易日
        expected = max(1, int(np_business_days(first_d, last_d)))
        out["missing_ratio"] = float(max(0.0, min(1.0, 1.0 - n / expected)))

        # 价格异常
        ccol = "收盘" if "收盘" in df.columns else None
        if ccol is None:
            out["status"] = STATUS_REJECTED
            out["reason"] = "行情缺少收盘价字段"
            return out
        close = pd.to_numeric(df[ccol], errors="coerce")
        if close.isna().any() or (close <= 0).any():
            out["status"] = STATUS_REJECTED
            out["reason"] = "存在异常价格（NaN 或非正）"
            return out

        # 零成交 / 流动性
        zero_ratio = 0.0
        amount_wan = 0.0
        if "成交量" in df.columns:
            vol = pd.to_numeric(df["成交量"], errors="coerce").fillna(0.0)
            zero_ratio = float((vol <= 0).mean())
            amount = (close * vol * 100.0) / 1e4      # 成交量单位：手（1 手=100 股）→ 万元
            amount_wan = float(amount.tail(min(n, 60)).mean())
        out["zero_vol_ratio"] = zero_ratio
        out["avg_amount_wan"] = amount_wan

        # 结论判定（顺序：数据量 → 上市时长 → 停牌 → 流动性 → 质量）
        if n < int(s["min_rows"]):
            out["status"] = STATUS_REJECTED
            out["reason"] = f"历史数据不足（{n} < {int(s['min_rows'])} 条）"
            return out
        if listed_days < int(s["min_listed_days"]):
            out["status"] = STATUS_REJECTED
            out["reason"] = f"上市时间不足（{listed_days} < {int(s['min_listed_days'])} 天）"
            return out
        if (today - last_d.to_pydatetime()).days > int(s["max_stale_days"]):
            out["status"] = STATUS_REJECTED
            out["reason"] = (f"停牌/长期无有效成交（最后行情 {out['last_date']} 距今 "
                             f"{(today - last_d.to_pydatetime()).days} 天）")
            return out
        if out["missing_ratio"] > float(s["max_missing_ratio"]):
            out["status"] = STATUS_REJECTED
            out["reason"] = f"行情缺失严重（缺失率 {out['missing_ratio'] * 100:.1f}%）"
            return out
        if zero_ratio > float(s["max_zero_vol_ratio"]):
            out["status"] = STATUS_REJECTED
            out["reason"] = f"零成交占比过高（{zero_ratio * 100:.1f}%）"
            return out

        quality = 1.0
        quality -= min(0.5, out["missing_ratio"] * 2.0)
        quality -= min(0.3, zero_ratio)
        quality = max(0.0, min(1.0, quality))
        out["quality"] = quality

        if amount_wan < float(s["min_avg_amount_wan"]):
            # 流动性不足：不直接淘汰，降级为低优先级研究队列（cooldown 由调度器控制）
            out["status"] = STATUS_COOLDOWN
            out["reason"] = (f"流动性不足（日均成交额约 {amount_wan:.0f} 万元 < "
                             f"{float(s['min_avg_amount_wan']):.0f} 万元），已降低研究优先级")
            return out

        out["status"] = STATUS_READY
        out["reason"] = (f"数据质量良好（{n} 条，缺失率 {out['missing_ratio'] * 100:.1f}%，"
                         f"日均成交额约 {amount_wan:.0f} 万元）")
        return out
    except Exception as e:          # noqa: BLE001
        out["status"] = STATUS_COOLDOWN
        out["reason"] = f"数据质量检查异常：{e}"
        return out


def np_business_days(d0, d1):
    """两个日期之间的工作日数（不含端点重复计数）。不依赖 numpy。"""
    try:
        days = max(0, int((d1 - d0).days))
    except Exception:
        return 1
    if days <= 0:
        return 1
    weeks, rem = divmod(days, 7)
    count = weeks * 5
    wd0 = d0.weekday()
    for i in range(rem):
        if (wd0 + i) % 7 < 5:
            count += 1
    return max(1, count)


# ====================================================================
# 4. 调度优先级（纯函数，可测试）
# ====================================================================
def priority_score(stock, scope=None, now=None):
    """研究优先级评分（越高越先研究）。返回 (分数, 明细 dict)。

    因素：数据质量、样本充足度、流动性、距上次研究的时间（公平性）、
          是否有新行情、研究成本（已研究次数）、连续失败惩罚。
    """
    s = dict(DEFAULT_SCOPE)
    if scope:
        s.update(scope)
    w = s.get("weights", {})
    now = now or _now()

    quality = float(stock.get("quality") or 0.0)
    rows = int(stock.get("data_rows") or 0)
    amount = float(stock.get("avg_amount_wan") or 0.0)
    count = int(stock.get("research_count") or 0)
    fails = int(stock.get("consecutive_fail") or 0)

    sample_score = min(1.0, rows / max(1.0, float(s["min_rows"]) * 2.0))
    liq_score = min(1.0, max(0.0, math.log10(max(amount, 1.0) / 100.0) / 2.0))
    gap_days = 999.0
    last = str(stock.get("last_researched") or "").strip()
    if last:
        try:
            gap_days = max(0.0, (now - datetime.strptime(last, "%Y-%m-%d %H:%M:%S")).days)
        except Exception:
            gap_days = 999.0
    gap_score = min(3.0, gap_days / 7.0)
    fresh_score = 0.0
    ldd = str(stock.get("last_data_date") or "").strip()
    try:
        if ldd and str(stock.get("last_research_data_date") or "").strip() != ldd:
            fresh_score = 1.0
    except Exception:
        fresh_score = 0.0
    cost = min(1.0, count / 20.0)
    fail_penalty = min(3.0, float(fails))

    score = (float(w.get("quality", 3.0)) * quality
             + float(w.get("samples", 1.0)) * sample_score
             + float(w.get("liquidity", 1.5)) * liq_score
             + float(w.get("gap", 2.0)) * gap_score
             + float(w.get("fresh", 1.0)) * fresh_score
             - float(w.get("cost", 0.6)) * cost
             - float(w.get("fail", 1.5)) * fail_penalty)
    detail = {"quality": round(quality, 3), "sample": round(sample_score, 3),
              "liquidity": round(liq_score, 3), "gap_days": round(gap_days, 1),
              "fresh": fresh_score, "research_count": count,
              "consecutive_fail": fails}
    return score, detail


def is_in_cooldown(stock, scope=None, now=None):
    """同一只股票是否还在“最小研究间隔”内（防反复研究同一只）。"""
    s = dict(DEFAULT_SCOPE)
    if scope:
        s.update(scope)
    now = now or _now()
    last = str(stock.get("last_researched") or "").strip()
    if not last:
        return False, 0.0
    try:
        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return False, 0.0
    hours = (now - last_dt).total_seconds() / 3600.0
    return hours < float(s["min_interval_hours"]), hours


# ====================================================================
# 5. 轻量注册表（SQLite：股票 / 队列 / 实验记录 / 基准 / 运行状态）
# ====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks(
  code TEXT PRIMARY KEY, name TEXT, board TEXT,
  status TEXT DEFAULT 'pending', reason TEXT DEFAULT '',
  quality REAL DEFAULT 0, data_rows INTEGER DEFAULT 0,
  last_data_date TEXT DEFAULT '', last_research_data_date TEXT DEFAULT '',
  avg_amount_wan REAL DEFAULT 0, missing_ratio REAL DEFAULT 1.0,
  zero_vol_ratio REAL DEFAULT 0, listed_days INTEGER DEFAULT 0,
  research_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0,
  fail_count INTEGER DEFAULT 0, consecutive_fail INTEGER DEFAULT 0,
  last_researched TEXT DEFAULT '', last_checked TEXT DEFAULT '',
  first_seen TEXT DEFAULT '', updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS experiments(
  exp_id TEXT PRIMARY KEY, code TEXT, name TEXT, goal TEXT,
  config_json TEXT DEFAULT '{}', status TEXT DEFAULT 'planned',
  promoted INTEGER DEFAULT 0, reject_reason TEXT DEFAULT '',
  next_direction TEXT DEFAULT '', metrics_json TEXT DEFAULT '{}',
  seed INTEGER DEFAULT 0, train_start TEXT DEFAULT '', train_end TEXT DEFAULT '',
  test_start TEXT DEFAULT '', test_end TEXT DEFAULT '',
  features TEXT DEFAULT '', params_json TEXT DEFAULT '{}',
  memory_mb REAL DEFAULT 0, duration_s REAL DEFAULT 0,
  retries INTEGER DEFAULT 0, created_at TEXT DEFAULT '', finished_at TEXT DEFAULT '',
  error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS baselines(
  code TEXT PRIMARY KEY, metrics_json TEXT DEFAULT '{}', model_path TEXT DEFAULT '',
  features TEXT DEFAULT '', params_json TEXT DEFAULT '{}',
  source_exp TEXT DEFAULT '', updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS test_usage(
  code TEXT NOT NULL,
  test_start TEXT NOT NULL,
  test_end TEXT NOT NULL,
  exp_id TEXT NOT NULL DEFAULT '',
  trigger TEXT DEFAULT '',
  status TEXT DEFAULT 'reserved',
  note TEXT DEFAULT '',
  created_at TEXT DEFAULT '',
  updated_at TEXT DEFAULT '',
  PRIMARY KEY(code, test_start, test_end, exp_id)
);
CREATE TABLE IF NOT EXISTS test_windows(
  code TEXT PRIMARY KEY,
  test_start TEXT NOT NULL,
  test_end TEXT NOT NULL,
  status TEXT DEFAULT 'active',
  note TEXT DEFAULT '',
  created_at TEXT DEFAULT '',
  updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS test_claims(
  code TEXT NOT NULL,
  test_start TEXT NOT NULL,
  test_end TEXT NOT NULL,
  exp_id TEXT NOT NULL DEFAULT '',
  trigger TEXT DEFAULT '',
  status TEXT DEFAULT 'running',
  note TEXT DEFAULT '',
  created_at TEXT DEFAULT '',
  updated_at TEXT DEFAULT '',
  PRIMARY KEY(code, test_start, test_end)
);
CREATE INDEX IF NOT EXISTS idx_test_usage_code ON test_usage(code);
CREATE INDEX IF NOT EXISTS idx_test_usage_status ON test_usage(status);
CREATE INDEX IF NOT EXISTS idx_test_claims_code ON test_claims(code);
CREATE TABLE IF NOT EXISTS runtime_state(
  key TEXT PRIMARY KEY, value TEXT DEFAULT '', updated_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_exp_code ON experiments(code);
CREATE INDEX IF NOT EXISTS idx_stocks_status ON stocks(status);
"""


class ResearchRegistry:
    """所有自主研究状态落盘在这里（轻量、可恢复、进程重启后可直接续跑）。"""

    def __init__(self, db_path):
        self.db_path = db_path
        d = os.path.dirname(os.path.abspath(db_path))
        if d:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, timeout=15, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------------- 基础 ----------------
    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def _exec(self, sql, args=()):
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _query(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # ---------------- 股票 ----------------
    def upsert_stock(self, code, name="", board="", **fields):
        code = _digits(code) or str(code)
        row = self.get_stock(code)
        now = _now_str()
        if row is None:
            self._exec(
                "INSERT INTO stocks(code,name,board,status,first_seen,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (code, name, board, fields.pop("status", STATUS_PENDING), now, now))
            row = self.get_stock(code) or {"code": code}
        sets, args = [], []
        for k in ("name", "board", "status", "reason", "quality", "data_rows",
                  "last_data_date", "last_research_data_date", "avg_amount_wan",
                  "missing_ratio", "zero_vol_ratio", "listed_days", "research_count",
                  "success_count", "fail_count", "consecutive_fail", "last_researched",
                  "last_checked"):
            if k in fields and fields[k] is not None:
                sets.append(f"{k}=?")
                args.append(fields[k])
        if name and not fields.get("name"):
            sets.append("name=?")
            args.append(name)
        sets.append("updated_at=?")
        args.append(now)
        args.append(code)
        self._exec(f"UPDATE stocks SET {','.join(sets)} WHERE code=?", tuple(args))
        return self.get_stock(code)

    def get_stock(self, code):
        rows = self._query("SELECT * FROM stocks WHERE code=?", (_digits(code) or str(code),))
        return rows[0] if rows else None

    def list_stocks(self, status=None, limit=None):
        sql = "SELECT * FROM stocks"
        args = []
        if status:
            if isinstance(status, (list, tuple)):
                sql += " WHERE status IN (%s)" % ",".join("?" for _ in status)
                args.extend(list(status))
            else:
                sql += " WHERE status=?"
                args.append(status)
        sql += " ORDER BY code"
        if limit:
            sql += " LIMIT ?"
            args.append(int(limit))
        return self._query(sql, tuple(args))

    def pool_counts(self):
        rows = self._query("SELECT status, COUNT(*) AS n FROM stocks GROUP BY status")
        out = {STATUS_PENDING: 0, STATUS_READY: 0, STATUS_COOLDOWN: 0,
               STATUS_REJECTED: 0, STATUS_RETIRED: 0}
        total = 0
        for r in rows:
            out[str(r["status"])] = int(r["n"])
            total += int(r["n"])
        out["total"] = total
        out["blocked"] = out[STATUS_REJECTED] + out[STATUS_RETIRED]
        return out

    def mark_missing_as_retired(self, seen_codes):
        """股票从最新列表里消失 → 标记 retired（不删除记录，保留可追溯性）。"""
        seen = set(_digits(c) for c in (seen_codes or []) if _digits(c))
        if not seen:
            return 0
        n = 0
        for r in self.list_stocks():
            if r["code"] not in seen and r["status"] != STATUS_RETIRED:
                self._exec("UPDATE stocks SET status=?, reason=?, updated_at=? WHERE code=?",
                           (STATUS_RETIRED, "已从最新股票列表消失（疑似退市）", _now_str(), r["code"]))
                n += 1
        return n

    # ---------------- 队列调度 ----------------
    def pick_next(self, count=1, scope=None, now=None, allow_pending=True):
        """挑出下一批要研究的股票（公平调度 + 优先级）。返回 dict 列表。

        顺序：
          1) 从未研究过、且不在冷却中的 ready 股票 → 先覆盖一轮（防止某些股票永远没机会）；
          2) 其余按 priority_score 降序，但跳过“最小研究间隔”内的股票（防止反复研究同一只，
             除非没有别的可选）。
        """
        s = dict(DEFAULT_SCOPE)
        if scope:
            s.update(scope)
        now = now or _now()
        count = max(1, int(count))
        ready = [r for r in self.list_stocks(status=STATUS_READY)]
        picked = []
        used = set()

        never = [r for r in ready if int(r.get("research_count") or 0) == 0
                 and not is_in_cooldown(r, s, now)[0]]
        never.sort(key=lambda r: (-float(r.get("quality") or 0), r["code"]))
        for r in never[:count]:
            picked.append((r, {"sequence": "首次覆盖"}))
            used.add(r["code"])

        if len(picked) < count:
            rest = [r for r in ready if r["code"] not in used]
            scored = []
            for r in rest:
                cd, _hrs = is_in_cooldown(r, s, now)
                sc, detail = priority_score(r, s, now)
                scored.append((sc, cd, r, detail))
            scored.sort(key=lambda t: -t[0])
            not_cd = [t for t in scored if not t[1]]
            pool = not_cd if not_cd else scored        # 都还在间隔内就放宽（避免空队列）
            for sc, _cd, r, detail in pool:
                if len(picked) >= count:
                    break
                detail = dict(detail)
                detail["score"] = round(sc, 3)
                detail["sequence"] = "优先级调度"
                picked.append((r, detail))
                used.add(r["code"])

        if len(picked) < count and allow_pending:
            pend = self.list_stocks(status=STATUS_PENDING)
            for r in pend:
                if len(picked) >= count:
                    break
                if r["code"] in used:
                    continue
                picked.append((r, {"sequence": "待数据检查"}))
                used.add(r["code"])
        return [{"stock": r, "schedule": d} for r, d in picked]

    def update_after_check(self, code, info):
        return self.upsert_stock(
            code, status=info.get("status"), reason=info.get("reason", ""),
            quality=info.get("quality", 0.0), data_rows=int(info.get("rows", 0)),
            last_data_date=info.get("last_date", ""),
            avg_amount_wan=info.get("avg_amount_wan", 0.0),
            missing_ratio=info.get("missing_ratio", 1.0),
            zero_vol_ratio=info.get("zero_vol_ratio", 0.0),
            listed_days=int(info.get("listed_days", 0)),
            last_checked=_now_str())

    def update_after_research(self, code, success, promoted=False, new_data_date=None):
        row = self.get_stock(code) or {}
        rc = int(row.get("research_count") or 0) + 1
        ok = int(row.get("success_count") or 0) + (1 if success else 0)
        fail = int(row.get("fail_count") or 0) + (0 if success else 1)
        consec = 0 if success else int(row.get("consecutive_fail") or 0) + 1
        fields = {"research_count": rc, "success_count": ok, "fail_count": fail,
                  "consecutive_fail": consec, "last_researched": _now_str()}
        if new_data_date:
            fields["last_research_data_date"] = new_data_date
        if success:
            fields["status"] = STATUS_READY
            fields["reason"] = "上一次研究成功完成"
        # 连续失败 3 次 → 降到冷却（避免死磕同一只股票），冷却后自然变为可再试
        if consec >= 3:
            fields["status"] = STATUS_COOLDOWN
            fields["reason"] = f"连续 {consec} 次研究失败，已转入冷却"
        return self.upsert_stock(code, **fields)

    def set_status(self, code, status, reason=""):
        return self.upsert_stock(code, status=status, reason=reason)

    def ready_count(self):
        return int(self.pool_counts().get(STATUS_READY, 0))

    # ---------------- 实验记录 ----------------
    def add_experiment(self, rec):
        rec = dict(rec or {})
        self._exec(
            "INSERT OR REPLACE INTO experiments(exp_id,code,name,goal,config_json,status,"
            "promoted,reject_reason,next_direction,metrics_json,seed,train_start,train_end,"
            "test_start,test_end,features,params_json,memory_mb,duration_s,retries,"
            "created_at,finished_at,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(rec.get("exp_id", "")), str(rec.get("code", "")), str(rec.get("name", "")),
             str(rec.get("goal", "")), str(rec.get("config_json", "{}")),
             str(rec.get("status", "planned")), 1 if rec.get("promoted") else 0,
             str(rec.get("reject_reason", "")), str(rec.get("next_direction", "")),
             str(rec.get("metrics_json", "{}")), int(rec.get("seed") or 0),
             str(rec.get("train_start", "")), str(rec.get("train_end", "")),
             str(rec.get("test_start", "")), str(rec.get("test_end", "")),
             str(rec.get("features", "")), str(rec.get("params_json", "{}")),
             float(rec.get("memory_mb") or 0.0), float(rec.get("duration_s") or 0.0),
             int(rec.get("retries") or 0), str(rec.get("created_at") or _now_str()),
             str(rec.get("finished_at", "")), str(rec.get("error", ""))))
        return rec.get("exp_id")

    def update_experiment(self, exp_id, **fields):
        if not fields:
            return
        sets, args = [], []
        for k, v in fields.items():
            if k in ("promoted",):
                v = 1 if v else 0
            sets.append(f"{k}=?")
            args.append(v)
        args.append(exp_id)
        self._exec(f"UPDATE experiments SET {','.join(sets)} WHERE exp_id=?", tuple(args))

    def list_experiments(self, code=None, limit=100, status=None):
        sql = "SELECT * FROM experiments"
        args, where = [], []
        if code:
            where.append("code=?")
            args.append(_digits(code) or str(code))
        if status:
            where.append("status=?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(int(limit))
        return self._query(sql, tuple(args))

    def experiment_counts(self, code=None):
        sql = "SELECT status, COUNT(*) AS n, SUM(promoted) AS p FROM experiments"
        args = []
        if code:
            sql += " WHERE code=?"
            args.append(_digits(code) or str(code))
        sql += " GROUP BY status"
        out = {"total": 0, "promoted": 0, "failed": 0, "done": 0, "rejected": 0,
               "interrupted": 0, "running": 0, "planned": 0}
        for r in self._query(sql, tuple(args)):
            st = str(r["status"])
            out[st] = int(r["n"])
            out["total"] += int(r["n"])
            out["promoted"] += int(r["p"] or 0)
        return out

    def unfinished_experiments(self):
        return self.list_experiments(limit=200, status="running") + \
               self.list_experiments(limit=200, status="planned")

    # ---------------- 基准（best validated per stock） ----------------
    def get_baseline(self, code):
        rows = self._query("SELECT * FROM baselines WHERE code=?", (_digits(code) or str(code),))
        if not rows:
            return None
        r = rows[0]
        r["metrics"] = read_json_safe_text(r.get("metrics_json"))
        r["params"] = read_json_safe_text(r.get("params_json"))
        return r

    def set_baseline(self, code, metrics, model_path="", features="", params=None,
                     source_exp=""):
        import json
        self._exec(
            "INSERT OR REPLACE INTO baselines(code,metrics_json,model_path,features,"
            "params_json,source_exp,updated_at) VALUES(?,?,?,?,?,?,?)",
            (_digits(code) or str(code), json.dumps(metrics or {}, ensure_ascii=False, default=str),
             str(model_path or ""), str(features or ""),
             json.dumps(params or {}, ensure_ascii=False, default=str),
             str(source_exp or ""), _now_str()))
        return self.get_baseline(code)

    # ---------------- 最终保留测试使用登记 ----------------
    def get_test_window(self, code):
        """返回该股票已固定的保留测试窗口（若存在）。"""
        code = _digits(code) or str(code)
        rows = self._query("SELECT * FROM test_windows WHERE code=?", (code,))
        return rows[0] if rows else None

    def ensure_test_window(self, code, test_start, test_end, note=""):
        """持久化固定测试窗口；一旦存在，绝不因调用方改日期而改变。

        返回 (created, record)。created=False 表示已有固定窗口；调用方应使用
        record 中的日期，而不是继续使用自己传入的日期。
        """
        code = _digits(code) or str(code)
        ts, te = str(test_start or ""), str(test_end or "")
        if not ts or not te or _date_ordinal(ts) is None or _date_ordinal(te) is None:
            return False, {"code": code, "test_start": ts, "test_end": te,
                           "status": "invalid", "note": "测试窗口日期非法"}
        if _date_ordinal(ts) > _date_ordinal(te):
            return False, {"code": code, "test_start": ts, "test_end": te,
                           "status": "invalid", "note": "测试窗口起止顺序错误"}
        now = _now_str()
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT * FROM test_windows WHERE code=?", (code,)).fetchall()]
            if rows:
                return False, rows[0]
            try:
                self._conn.execute(
                    "INSERT INTO test_windows(code,test_start,test_end,status,note,"
                    "created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (code, ts, te, "active", str(note or ""), now, now))
                self._conn.commit()
            except sqlite3.IntegrityError:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                recs = [dict(r) for r in self._conn.execute(
                    "SELECT * FROM test_windows WHERE code=?", (code,)).fetchall()]
                if recs:
                    return False, recs[0]
                raise
            recs = [dict(r) for r in self._conn.execute(
                "SELECT * FROM test_windows WHERE code=?", (code,)).fetchall()]
        return True, (recs[0] if recs else {
            "code": code, "test_start": ts, "test_end": te,
            "status": "active", "note": str(note or ""),
        })

    def get_overlapping_test_usage(self, code, test_start, test_end):
        """返回与给定完整区间重叠的所有 usage 记录。"""
        code = _digits(code) or str(code)
        rows = self._query("SELECT * FROM test_usage WHERE code=?", (code,))
        return [r for r in rows
                if _ranges_overlap(r.get("test_start"), r.get("test_end"),
                                   test_start, test_end)]

    def get_test_usage(self, code, test_start, test_end):
        """返回某股票完整测试区间的所有登记记录（含 reserved/running/...）。"""
        code = _digits(code) or str(code)
        return self._query(
            "SELECT * FROM test_usage WHERE code=? AND test_start=? AND test_end=? "
            "ORDER BY updated_at DESC, exp_id DESC",
            (code, str(test_start or ""), str(test_end or "")))

    def get_blocking_test_usage(self, code, test_start, test_end):
        """返回与给定区间重叠、且已经消费/正在消费测试数据的记录。"""
        rows = self.get_overlapping_test_usage(code, test_start, test_end)
        return [r for r in rows if str(r.get("status") or "") in TEST_USAGE_BLOCKING]

    def reserve_test_window(self, code, test_start, test_end, exp_id="",
                            trigger="", note=""):
        """登记一个“保留但未评估”的测试区间；不消费测试数据。

        固定窗口已存在且日期不同、或该区间已有重叠的消费记录时，返回
        (False, 已存在记录)，不覆盖、不降级。
        """
        code = _digits(code) or str(code)
        exp_id = str(exp_id or "")
        ts, te = str(test_start or ""), str(test_end or "")
        # 固定窗口不能被改日期；已有重叠消费也不能留下错误 reservation。
        win = self.get_test_window(code)
        if win and (str(win.get("test_start") or "") != ts
                    or str(win.get("test_end") or "") != te):
            return False, dict(win, status="window_locked",
                               note=f"保留测试窗口已固定为 "
                                    f"{win.get('test_start')}~{win.get('test_end')}")
        blocking = self.get_blocking_test_usage(code, ts, te)
        if blocking:
            return False, blocking[0]
        created, win = self.ensure_test_window(
            code, ts, te, note="固定保留测试窗口")
        if str(win.get("test_start") or "") != ts or str(win.get("test_end") or "") != te:
            return False, dict(win, status="window_locked",
                               note=f"保留测试窗口已固定为 "
                                    f"{win.get('test_start')}~{win.get('test_end')}")
        now = _now_str()
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT * FROM test_usage WHERE code=? AND test_start=? AND test_end=?",
                (code, ts, te)).fetchall()]
            existing = next((r for r in rows if str(r.get("exp_id") or "") == exp_id), None)
            if existing is None:
                self._conn.execute(
                    "INSERT INTO test_usage(code,test_start,test_end,exp_id,trigger,"
                    "status,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (code, ts, te, exp_id, str(trigger or ""), TEST_USAGE_RESERVED,
                     str(note or ""), now, now))
            else:
                # 同一实验重复登记：只更新 reserved 的说明，不降级其他状态
                if str(existing.get("status") or "") == TEST_USAGE_RESERVED:
                    self._conn.execute(
                        "UPDATE test_usage SET trigger=?, note=?, updated_at=? "
                        "WHERE code=? AND test_start=? AND test_end=? AND exp_id=?",
                        (str(trigger or ""), str(note or ""), now, code, ts, te, exp_id))
            self._conn.commit()
        recs = self.get_test_usage(code, ts, te)
        same = [r for r in recs if str(r.get("exp_id") or "") == exp_id]
        return True, (same[0] if same else (recs[0] if recs else {
            "code": code, "test_start": ts, "test_end": te,
            "exp_id": exp_id, "status": TEST_USAGE_RESERVED,
        }))

    def get_test_claim(self, code, test_start, test_end):
        code = _digits(code) or str(code)
        rows = self._query(
            "SELECT * FROM test_claims WHERE code=? AND test_start=? AND test_end=?",
            (code, str(test_start or ""), str(test_end or "")))
        return rows[0] if rows else None

    def claim_final_test(self, code, test_start, test_end, exp_id="",
                         trigger=""):
        """原子领取最终测试使用权。

        固定窗口已存在且日期不同、或已有重叠的 running/completed/failed/
        aborted/used 记录时拒绝；只有 reserved/无记录才允许领取。
        test_claims 的主键 + BEGIN IMMEDIATE 保证跨连接并发只有一个领取者。
        """
        code = _digits(code) or str(code)
        exp_id = str(exp_id or "")
        ts, te = str(test_start or ""), str(test_end or "")
        now = _now_str()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                win_rows = [dict(r) for r in self._conn.execute(
                    "SELECT * FROM test_windows WHERE code=?", (code,)).fetchall()]
                if win_rows:
                    win = win_rows[0]
                    if (str(win.get("test_start") or "") != ts
                            or str(win.get("test_end") or "") != te):
                        self._conn.rollback()
                        return False, dict(win, status="window_locked",
                                           note=f"保留测试窗口已固定为 "
                                                f"{win.get('test_start')}~{win.get('test_end')}")
                else:
                    usage = [dict(r) for r in self._conn.execute(
                        "SELECT * FROM test_usage WHERE code=?", (code,)).fetchall()]
                    blocking = [r for r in usage
                                if str(r.get("status") or "") in TEST_USAGE_BLOCKING
                                and _ranges_overlap(r.get("test_start"), r.get("test_end"), ts, te)]
                    if blocking:
                        self._conn.rollback()
                        return False, blocking[0]
                    self._conn.execute(
                        "INSERT INTO test_windows(code,test_start,test_end,status,note,"
                        "created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                        (code, ts, te, "active", "首次最终测试固定窗口", now, now))

                existing_claim = self._conn.execute(
                    "SELECT * FROM test_claims WHERE code=? AND test_start=? AND test_end=?",
                    (code, ts, te)).fetchone()
                if existing_claim is not None:
                    rec = dict(existing_claim)
                    self._conn.rollback()
                    return False, rec

                usage = [dict(r) for r in self._conn.execute(
                    "SELECT * FROM test_usage WHERE code=?", (code,)).fetchall()]
                blocking = [r for r in usage
                            if str(r.get("status") or "") in TEST_USAGE_BLOCKING
                            and _ranges_overlap(r.get("test_start"), r.get("test_end"), ts, te)]
                if blocking:
                    self._conn.rollback()
                    return False, blocking[0]

                self._conn.execute(
                    "INSERT INTO test_claims(code,test_start,test_end,exp_id,trigger,"
                    "status,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (code, ts, te, exp_id, str(trigger or ""), TEST_USAGE_RUNNING,
                     "最终测试已领取", now, now))
                existing = next((r for r in usage
                                 if str(r.get("exp_id") or "") == exp_id
                                 and str(r.get("test_start") or "") == ts
                                 and str(r.get("test_end") or "") == te), None)
                if existing is None:
                    self._conn.execute(
                        "INSERT INTO test_usage(code,test_start,test_end,exp_id,trigger,"
                        "status,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (code, ts, te, exp_id, str(trigger or ""), TEST_USAGE_RUNNING,
                         "最终测试已领取", now, now))
                else:
                    self._conn.execute(
                        "UPDATE test_usage SET status=?, trigger=?, note=?, updated_at=? "
                        "WHERE code=? AND test_start=? AND test_end=? AND exp_id=?",
                        (TEST_USAGE_RUNNING, str(trigger or ""), "最终测试已领取",
                         now, code, ts, te, exp_id))
                self._conn.commit()
            except sqlite3.IntegrityError:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                rec = self.get_test_claim(code, ts, te) or {
                    "code": code, "test_start": ts, "test_end": te,
                    "exp_id": exp_id, "status": TEST_USAGE_RUNNING,
                }
                return False, rec
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise
        rec = self.get_test_claim(code, ts, te)
        return True, (rec or {
            "code": code, "test_start": ts, "test_end": te,
            "exp_id": exp_id, "status": TEST_USAGE_RUNNING, "trigger": trigger,
        })

    def mark_final_test(self, code, test_start, test_end, exp_id="",
                        status=TEST_USAGE_COMPLETED, note=""):
        """更新最终测试状态（completed/failed/aborted）；保留消费记录。"""
        code = _digits(code) or str(code)
        exp_id = str(exp_id or "")
        ts, te = str(test_start or ""), str(test_end or "")
        status = str(status or TEST_USAGE_COMPLETED)
        now = _now_str()
        with self._lock:
            self._conn.execute(
                "UPDATE test_claims SET status=?, note=?, updated_at=? "
                "WHERE code=? AND test_start=? AND test_end=?",
                (status, str(note or ""), now, code, ts, te))
            cur = self._conn.execute(
                "UPDATE test_usage SET status=?, note=?, updated_at=? "
                "WHERE code=? AND test_start=? AND test_end=? AND exp_id=?",
                (status, str(note or ""), now, code, ts, te, exp_id))
            if cur.rowcount == 0:
                self._conn.execute(
                    "INSERT INTO test_usage(code,test_start,test_end,exp_id,trigger,"
                    "status,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (code, ts, te, exp_id, "", status, str(note or ""), now, now))
            self._conn.commit()
        return True

    def test_usage_summary(self, code, test_start, test_end):
        """便于日志/测试读取的简要状态：blocking / reserved / none。"""
        win = self.get_test_window(code)
        if win and (str(win.get("test_start") or "") != str(test_start or "")
                    or str(win.get("test_end") or "") != str(test_end or "")):
            return {"status": "window_locked", "record": win,
                    "records": self.get_overlapping_test_usage(code, test_start, test_end),
                    "window": win}
        rows = self.get_overlapping_test_usage(code, test_start, test_end)
        blocking = [r for r in rows if str(r.get("status") or "") in TEST_USAGE_BLOCKING]
        if blocking:
            return {"status": "blocking", "record": blocking[0], "records": rows,
                    "window": win}
        if rows:
            return {"status": "reserved", "record": rows[0], "records": rows,
                    "window": win}
        return {"status": "none", "record": None, "records": [], "window": win}

    # ---------------- 运行状态（暂停/恢复/停止/数据源） ----------------
    def set_state(self, key, value):
        import json
        try:
            val = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
        except Exception:
            val = str(value)
        self._exec("INSERT OR REPLACE INTO runtime_state(key,value,updated_at) VALUES(?,?,?)",
                   (str(key), val, _now_str()))

    def get_state(self, key, default=None):
        rows = self._query("SELECT value FROM runtime_state WHERE key=?", (str(key),))
        if not rows:
            return default
        v = rows[0]["value"]
        if isinstance(v, str) and v[:1] in "[{":
            try:
                import json
                return json.loads(v)
            except Exception:
                return v
        return v

    def set_data_source_status(self, ok, error="", last_success=None, count=None):
        self.set_state("data_source", {
            "ok": bool(ok), "error": str(error or ""),
            "last_success": last_success or self.get_state("data_source_last_success", ""),
            "checked_at": _now_str(), "count": count,
        })

    def get_data_source_status(self):
        return self.get_state("data_source", {"ok": None, "error": "", "last_success": ""})


def read_json_safe_text(s):
    try:
        import json
        return json.loads(s) if s else {}
    except Exception:
        return {}
