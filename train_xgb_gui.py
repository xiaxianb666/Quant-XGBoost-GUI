# -*- coding: utf-8 -*-
"""
====================================================================
 XGBoost 量化模型 —— 【一体化·专业版】
 文件：train_xgb_gui.py
--------------------------------------------------------------------
 界面 = 2 个标签页：
   「训练回测」：AI管理 / 参数 / 特征开关 / 批量按钮 / 日志 / 资金曲线
   「历史记录」：表格 + 删除 / 清空 / 导出

 本次升级：
   - 「开始学习」→ 换成 4 个按钮：【跑1次】【跑10次】【跑50次】【跑100次】
   - 批量训练在后台线程跑，界面不卡；跑的时候按钮禁用，日志实时显示"第 x/n 次"
   - 每次结果都写入 history_AI.csv，并带上"特征组合"
   - 当同一特征组合攒够 20 条，自动追加一条"均值/中位数"汇总记录
   - 整体重新排版（grid 对齐、字体放大、按钮等宽），更像专业软件
   - 数据文件都在用户目录，不往桌面丢图标

 三条铁律不变：时间切分不打乱 / 无未来函数 / 不用 plt.show()
====================================================================
"""

import os
import sys
import csv
import queue
import threading
import traceback
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime, timedelta
try:
    import auto_research  # type: ignore
except Exception:
    auto_research = None
try:
    import stock_pool          # 自主选股 + 研究队列（SQLite）
    import memory_guard        # 4GB 老电脑内存保护
    import research_engine     # 自主研究调度 + 晋级验收标准
except Exception:
    stock_pool = None
    memory_guard = None
    research_engine = None

# ====================================================================
# 【全局防御：GUI 环境下 stdout/stderr 防御】
# ====================================================================
# 1. 全局防御：如果 GUI 环境下 stdout/stderr 为 None，强制绑定假流
import io
if sys.stdout is None or not hasattr(sys.stdout, 'write'):
    sys.stdout = io.StringIO()
if sys.stderr is None or not hasattr(sys.stderr, 'write'):
    sys.stderr = io.StringIO()

# 2. 防御 akshare 内部 tqdm 的进度条写入（Monkey Patch）
#    akshare 会对 tqdm 对象做 for 循环 / len() 等操作，因此必须实现迭代协议，
#    不能只留空方法，否则会报 "'DummyTqdm' object is not iterable"。
try:
    import tqdm

    class DummyTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def __len__(self):
            try:
                return len(self.iterable)
            except Exception:
                return 0

        def __getitem__(self, index):
            return self.iterable[index]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def set_description(self, *args, **kwargs):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def write(self, *args, **kwargs):
            return None

        def refresh(self, *args, **kwargs):
            return None

        def clear(self, *args, **kwargs):
            return None

        def reset(self, *args, **kwargs):
            return None

    tqdm.tqdm = DummyTqdm
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto.tqdm = DummyTqdm
    except Exception:
        pass
    try:
        import tqdm.std as _tqdm_std
        _tqdm_std.tqdm = DummyTqdm
    except Exception:
        pass
except ImportError:
    pass


# ====================================================================
# 【全局安全报错弹窗】
# 原生 messagebox.showerror 遇到超长异常文本会把窗口撑到变形，
# 甚至挤压主界面布局。这里统一封装：
#   1) 完整报错先写入主界面运行日志（如果已注册日志接收器）；
#   2) 弹窗只展示前若干行，并把每行强制按 60 字符换行；
#   3) 使用固定最大宽度的自定义 Toplevel + 只读文本框 + 滚动条。
# ====================================================================
_ERROR_LOG_SINK = None
_ERROR_MAX_LINES = 18
_ERROR_WRAP_CHARS = 60


def set_error_log_sink(sink):
    """注册主界面日志接收器（通常是 self.write），用于保存完整报错。"""
    global _ERROR_LOG_SINK
    _ERROR_LOG_SINK = sink


def _error_display_text(message):
    """把超长报错裁成“前 N 行 + 每行 60 字符”的安全展示文本。"""
    text = "" if message is None else str(message)
    lines = text.splitlines() or [text]
    shown = lines[:_ERROR_MAX_LINES]
    wrapped = []
    for line in shown:
        line = "" if line is None else str(line)
        if len(line) <= _ERROR_WRAP_CHARS:
            wrapped.append(line)
        else:
            for i in range(0, len(line), _ERROR_WRAP_CHARS):
                wrapped.append(line[i:i + _ERROR_WRAP_CHARS])
    out = "\n".join(wrapped)
    if len(lines) > _ERROR_MAX_LINES or len(out) < len(text):
        out += "\n\n……（报错过长，完整内容已写入运行日志）"
    return out


def _safe_showerror(title, message, parent=None):
    """安全错误弹窗：完整信息进日志，弹窗限宽限高，绝不被长文本撑变形。"""
    text = "" if message is None else str(message)

    # 1) 完整报错写入主界面日志（失败不影响弹窗）
    try:
        if callable(_ERROR_LOG_SINK):
            _ERROR_LOG_SINK(f"{title}：{text}", "red")
    except Exception:
        pass

    shown = _error_display_text(text)
    root = parent or getattr(tk, "_default_root", None)
    if root is None:
        # 没有可用 Tk 根窗口（例如纯逻辑测试）时退回原生弹窗，但文本已截断
        messagebox.showerror(title, shown)
        return

    # 2) 自定义固定宽度弹窗
    try:
        win = tk.Toplevel(root)
        win.title(str(title))
        win.transient(root)
        win.resizable(False, False)
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)

        lbl = ttk.Label(frame, text=str(title), font=FONT_T)
        lbl.pack(anchor="w", pady=(0, 8))

        text_box = tk.Text(frame, width=64, height=min(16, max(4, len(shown.splitlines()) + 1)),
                           wrap="char", font=FONT_LOG, relief="solid", borderwidth=1)
        text_box.pack(fill="both", expand=True)
        text_box.insert("1.0", shown)
        text_box.config(state="disabled")

        scroll = ttk.Scrollbar(text_box, orient="vertical", command=text_box.yview)
        scroll.pack(side="right", fill="y")
        text_box.config(yscrollcommand=scroll.set)

        def _close(event=None):
            try:
                win.grab_release()
            except Exception:
                pass
            try:
                win.destroy()
            except Exception:
                pass

        btn = ttk.Button(frame, text="确定", command=_close)
        btn.pack(pady=(10, 0))
        win.protocol("WM_DELETE_WINDOW", _close)
        try:
            win.grab_set()
            btn.focus_set()
        except Exception:
            pass
        win.wait_window()
    except Exception:
        # 自定义弹窗失败时保底用原生弹窗，但只显示截断后的文本
        messagebox.showerror(title, shown)

# ====================================================================
# 0. 默认配置
# ====================================================================
DEF_CODE = "300274"
DEF_TRAIN_START = "2018-01-01"
DEF_TRAIN_END = "2022-12-31"
DEF_TEST_START = "2023-01-01"
DEF_TEST_END = datetime.now().strftime("%Y-%m-%d")
DEF_CAPITAL = "100000"

LOOKAHEAD = 5
UP_THRESHOLD = 0.02
DOWN_THRESHOLD = -0.02

# 字体统一在这里定义（放大一点，看着舒服）
FONT_N = ("Microsoft YaHei", 11)          # 普通
FONT_B = ("Microsoft YaHei", 11, "bold")   # 加粗
FONT_T = ("Microsoft YaHei", 12, "bold")   # 标题
FONT_LOG = ("Consolas", 10)                # 日志等宽字体
FONT_S = ("Microsoft YaHei", 9)            # 小字

# ====================================================================
# 【深色量化工作台】统一配色：想换色只改这里
# ====================================================================
UI_BG        = "#1b1f27"   # 窗口主背景
UI_PANEL     = "#232833"   # 面板 / 分区背景
UI_PANEL_2   = "#2b313d"   # 顶部条 / 次级面板
UI_INPUT     = "#2f3542"   # 输入框背景
UI_FG        = "#e6eaf0"   # 主文字
UI_FG_DIM    = "#9aa5b4"   # 次要文字
UI_BORDER    = "#3a4150"   # 边框 / 分隔线
UI_ACCENT    = "#4aa3ff"   # 主色（信息 / 运行中）
UI_OK        = "#3fb950"   # 成功 / 完成
UI_WARN      = "#d29922"   # 警告 / 暂停
UI_ERR       = "#f85149"   # 错误 / 失败
UI_TITLE     = "#ffd479"   # 标题 / 高亮
UI_BTN       = "#333a46"   # 普通按钮
UI_BTN_HOVER = "#3d4757"   # 按钮悬停

# 日志最多保留行数（保护 4GB 老电脑内存）
LOG_MAX_LINES = 1500

# 交替寻优流程（用于阶段指示器；只做展示，不改变程序实际逻辑）
STAGE_NAMES = [
    "选择初始指标组合",
    "智能参数推荐",
    "智能二次定界",
    "第一轮自动调仓寻优",
    "最佳参数更新到手动风控",
    "一键同步参数到特征寻优",
    "第一轮特征寻优",
    "第二轮自动调仓寻优",
    "下一轮特征寻优",
    "继续交替优化",
    "最终回测与结果查看",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 数据目录：所有模型 / 历史 / AI名单 都放桌面的【模型训练器后端文件夹】里 ----
_DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
DATA_DIR = os.path.join(_DESKTOP_DIR, "模型训练器后端文件夹")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception:
    DATA_DIR = BASE_DIR

# 资金曲线文件夹
CURVE_DIR = os.path.join(DATA_DIR, "equity_curves")
try:
    os.makedirs(CURVE_DIR, exist_ok=True)
except Exception:
    pass

# 历史记录表头（只写"汇总"记录；新增"备注"列）
HISTORY_HEADER = ["时间", "股票代码", "训练起始日期", "训练结束日期",
                  "测试起始日期", "初始资金", "最终资金", "收益率(均/中)",
                  "准确率(均/中)", "特征组合", "备注", "触发方式",
                  "最佳止损", "最佳止盈", "最佳均线", "最佳特征组合", "资金曲线文件路径",
                  "尝试次数", "交易成本配置"]
FEAT_COL = 9      # "特征组合"是第几列（从0数）
NOTE_COL = 10     # "备注"是第几列
TRIGGER_COL = 11  # 触发方式
BEST_SL_COL = 12   # 最佳止损
BEST_TP_COL = 13   # 最佳止盈
BEST_MA_COL = 14   # 最佳均线
BEST_FEAT_COL = 15  # 最佳特征组合
CURVE_PATH_COL = 16  # 资金曲线文件路径
COST_DESC_COL = 18   # 交易成本配置（新增，置于末尾，保证旧列下标不变）

# 界面参数持久化文件路径
PARAM_FILE = os.path.join(DATA_DIR, "gui_params.json")
LOG_FILE = os.path.join(DATA_DIR, "gui_log.txt")

# ====================================================================
# 【pythonw 启动期诊断】Windows 下用 pythonw.exe 启动时，sys.stdout /
# sys.stderr 可能是 None。此时 print()、traceback 打印、Tkinter 回调异常
# 都会再抛 AttributeError，真正的启动异常反而被掩盖。
# 这里只做最小兜底：
#   1) 仅在流为 None 时替换成写入项目内独立日志的轻量对象；
#   2) 未捕获异常的 sys.excepthook 始终先写日志，再尽力按原样输出；
#   3) 上述日志写入全部是 best-effort，失败只返回 False，绝不抛异常，
#      因此不会掩盖原始异常（调用方仍会 raise 原异常）。
# 正常控制台/测试环境（stdout/stderr 可用）不会被替换。
# 注意：本机制不改变 AkShare 的调用语义，也不应被理解为已定位根因。
# ====================================================================
STARTUP_LOG_FILE = os.path.join(BASE_DIR, "_startup_diagnostics.log")
_DIAG_LOCK = threading.Lock()


def _diag_log_paths():
    """诊断日志候选路径（项目内）。测试可替换此函数以模拟全部不可写。"""
    out = []
    for p in (STARTUP_LOG_FILE, LOG_FILE):
        try:
            p = os.path.abspath(p)
        except Exception:
            continue
        if p not in out:
            out.append(p)
    return tuple(out)


def _append_diag_log(message, level="INFO"):
    """尽力追加一条诊断日志；返回是否写入成功。任何失败都不抛异常。"""
    try:
        text = "" if message is None else str(message)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] [{level}] {text}\n"
    except Exception:
        return False
    for path in _diag_log_paths():
        try:
            with _DIAG_LOCK:
                with open(path, "a", encoding="utf-8", errors="backslashreplace") as f:
                    f.write(line)
                    f.flush()
            return True
        except Exception:
            continue
    return False


def _exception_text(exc_type=None, exc_value=None, exc_tb=None):
    """把异常格式化成完整堆栈；格式化本身失败时退化为类型名+消息。"""
    try:
        if exc_type is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is None:
            return ""
        return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    except Exception:
        try:
            name = getattr(exc_type, "__name__", "Exception")
            return f"{name}: {exc_value}"
        except Exception:
            return ""


def _log_exception(message, exc_type=None, exc_value=None, exc_tb=None, level="ERROR"):
    """把一条说明连同完整堆栈写进诊断日志；返回是否成功，绝不抛异常。"""
    detail = _exception_text(exc_type, exc_value, exc_tb)
    body = message if not detail else f"{message}\n{detail}"
    return _append_diag_log(str(body).rstrip("\n"), level)


class _DiagStream:
    """pythonw 下 stdout/stderr 的最小替身：所有输出写入独立诊断日志。"""

    encoding = "utf-8"
    errors = "backslashreplace"

    def __init__(self, channel):
        self.channel = channel

    def write(self, text):
        try:
            if not text:
                return 0
            s = text if isinstance(text, str) else str(text)
            if s:
                _append_diag_log(s.rstrip("\r\n"), self.channel)
            return len(s)
        except Exception:
            # 日志失败不能反过来影响调用方（例如 traceback 打印）
            return 0

    def writelines(self, lines):
        try:
            for line in lines:
                self.write(line)
        except Exception:
            pass

    def flush(self):
        return None

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("pythonw 下没有可用的文件描述符")

    def close(self):
        return None


def _startup_log(text):
    """启动信息优先写原 stdout；stdout 为 None/写入失败时改写入诊断日志。"""
    try:
        stream = getattr(sys, "stdout", None)
        if stream is not None and hasattr(stream, "write"):
            stream.write(("" if text is None else str(text)) + "\n")
            try:
                stream.flush()
            except Exception:
                pass
            return True
    except Exception:
        pass
    return _append_diag_log(text, "STDOUT")


def _install_pythonw_diagnostics():
    """安装 pythonw 兜底（可重复调用）。返回被替换的流名列表。"""
    replaced = []
    try:
        if getattr(sys, "stdout", None) is None:
            sys.stdout = _DiagStream("STDOUT")
            replaced.append("stdout")
        if getattr(sys, "stderr", None) is None:
            sys.stderr = _DiagStream("STDERR")
            replaced.append("stderr")
    except Exception:
        pass
    try:
        if getattr(sys, "excepthook", None) is not _log_uncaught_exception:
            sys.excepthook = _log_uncaught_exception
    except Exception:
        pass
    if replaced:
        _append_diag_log(
            "pythonw 启动兜底：sys.%s 为 None，输出改写入 %s"
            % ("/".join(replaced), STARTUP_LOG_FILE), "INFO")
    return replaced


def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    """未捕获异常：先写完整堆栈到诊断日志，再尽力按原样输出到 stderr。"""
    ok = _log_exception("未捕获异常（sys.excepthook）",
                        exc_type, exc_value, exc_tb, level="CRITICAL")
    try:
        stream = getattr(sys, "stderr", None)
        if stream is not None and not isinstance(stream, _DiagStream):
            stream.write(_exception_text(exc_type, exc_value, exc_tb))
            stream.flush()
    except Exception:
        pass
    return ok


def _log_tk_exception(exc, val, exc_tb):
    """Tkinter 回调异常：写入诊断日志，并尽力保留控制台可见性。"""
    ok = _log_exception("Tkinter 回调异常", exc, val, exc_tb, level="ERROR")
    try:
        stream = getattr(sys, "stderr", None)
        # pythonw 下 stderr 已被替换为 _DiagStream，避免重复写入日志
        if stream is not None and not isinstance(stream, _DiagStream):
            stream.write(_exception_text(exc, val, exc_tb))
            stream.flush()
    except Exception:
        pass
    return ok

# 【特征寻优历史】独立文件表头（与跑测历史、调参寻优历史彻底分开，互不干扰）
FEATURE_SEARCH_HEADER = [
    "记录时间", "AI名字", "股票代码",
    "训练开始日期", "训练结束日期", "测试开始日期", "测试结束日期",
    "最佳特征组合", "最佳止损", "最佳止盈", "最佳均线周期",
    "测试集收益率", "最大回撤", "资金曲线路径", "寻优得分", "尝试次数", "备注",
]

DEFAULT_SWITCHES = {
    "MA5": True, "MA10": True, "MA20": True, "MA60": True,
    "RSI": True, "MACD": True, "KDJ": False, "BOLL": False,
    "VOL": True, "RET": True, "ATR": True,
    "ENH": True,   # 增强特征组（动量/波动率/振幅）—— 与波动率标签搭配才有方向信号
}

# 【老电脑保护】寻优次数硬性上限：无论是【自动调参】还是【特征寻优】，
# 每天尝试次数都绝不超过该值（特征寻优每次都要重训模型，尤其吃算力）。
MAX_SEARCH_TRIALS = 200

# ====================================================================
# 【四类参数严格分区】—— 绝对不许混用（这是本软件的核心纪律）
# --------------------------------------------------------------------
#  ① 特征指标（预测特征）：UI 里 11 个技术指标开关 → make_features()。
#  ② XGBoost 模型参数：只控制模型复杂度，与“交易规则”完全无关。
#     集中登记在 MODEL_PARAMS，方便日后单独优化，绝不与策略参数混淆。
#  ③ 交易策略参数：止损 / 止盈 / 均线周期 / 是否“跌破均线卖出”。
#     由【自动调仓寻优】在“训练期优化验证段”上真正搜索（_evolve_worker_impl）。
#  ④ 仓位 / 调仓参数：当前引擎为“全仓进、全仓出”，暂无独立可调参数。
#     日后若加分批建仓 / 仓位比例，请在此登记，切勿塞进上面三类。
# ====================================================================
MODEL_PARAMS = dict(
    objective="multi:softprob",   # ② 模型参数（分类）
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=20,
    eval_metric="mlogloss",
)

# 【防过拟合·数据隔离】训练期尾部这个比例作为“优化验证段”，
# 只用于挑选特征 / 策略参数；最终测试段绝不参与任何调参（冻结、只评估一次）。
VALID_FRACTION = 0.30
# 训练期内部再切一小段做 XGBoost 早停，避免早停污染“优化验证段”。
MODEL_ES_FRACTION = 0.15
# 候选策略“合格”门槛（三条全过才算合格；允许全部候选都不合格）。
STRATEGY_MIN_TRADES = 3           # 验证段至少发生这么多次完整交易（卖出计数）
STRATEGY_MAX_DD = 0.60            # 验证段最大回撤不得超过该比例（0.60 = 60%）
STRATEGY_REQUIRE_POSITIVE = True  # 验证段“含成本”的净收益必须 > 0
# 自动调仓寻优里，每个候选策略用几个随机种子训练模型、再取平均（越大越稳、越慢）。
STRATEGY_MODEL_SEEDS = 3

# ====================================================================
# 【标签设计】—— 决定"预测什么"，直接决定模型有没有可用信号
# --------------------------------------------------------------------
# 经离线实验（train/valid 内部切分、只看 AUC）验证：对本股票与当前 11 个技术
# 指标，无论固定阈值 / 波动率自适应 / 二分类，AUC 都≈0.50，即"方向没有可用信号"。
# 因此这里把标签方案做成可切换，方便你日后自行探索；默认保持原行为（fixed）。
#   mode="fixed"  ：未来 H 日涨跌超过固定百分比 → +1 / -1 / 0（原行为）
#   mode="vol"    ：阈值 = k × 近20日收益波动率 × √H（波动率自适应，更均衡）
#   mode="binary" ：未来 H 日涨 → +1，跌 → -1（去掉中性类，二分类）
# ====================================================================
LABEL_MODE = "vol"
LABEL_HORIZON = LOOKAHEAD            # 预测未来几个交易日
LABEL_VOL_K = 0.5                    # vol 模式的阈值系数
LABEL_VOL_WINDOW = 20                # vol 模式估计波动率的回看窗口
# 预测力自检门槛：验证段准确率至少要高出"多数类基准"这么多，才算有初步信号。
PREDICTIVE_EDGE_MARGIN = 0.02

# ---- AI 名字纠正：历史遗留错别字「阳关电源」→ 正式名「阳光电源」 ----
# 只做“安全复制、绝不覆盖”，保留旧文件以兼容历史记录（详见 migrate_ai_name）。
AI_NAME_FIXES = {"阳关电源": "阳光电源"}
AI_FILE_PATTERNS = [
    "model_{n}.json", "model_{n}_风控.json", "model_{n}_Best.json",
    "history_{n}.csv", "history_{n}_风控.csv",
    "tuning_log_{n}.csv", "feature_search_log_{n}.csv",
    "feature_search_gen_{n}.csv", "strategy_search_gen_{n}.csv",
    "param_tuning_log_{n}.csv", "best_features_{n}.txt",
]


# ---- 把旧版丢在桌面/用户目录下的数据文件，自动搬进新的数据目录 ----
def _migrate_desktop_data():
    import glob
    import shutil
    patterns = ["ais.json", "history.csv", "my_ai_model.json",
                "model_*.json", "history_*.csv", "tuning_log_*.csv",
                "feature_search_log_*.csv", "feature_search_gen_*.csv",
                "best_features_*.txt", "param_tuning_log_*.csv",
                "curve_avg_*.csv"]
    # 扫描多个可能的旧位置：桌面根目录、用户主目录下的旧数据文件夹
    scan_dirs = [
        BASE_DIR,
        os.path.join(os.path.expanduser("~"), "XGBoost量化数据"),
        _DESKTOP_DIR,
    ]
    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for pat in patterns:
            for src in glob.glob(os.path.join(scan_dir, pat)):
                name = os.path.basename(src)
                dst = os.path.join(DATA_DIR, name)
                if os.path.abspath(src) == os.path.abspath(dst):
                    continue
                if not os.path.exists(dst):
                    try:
                        shutil.move(src, dst)
                    except Exception:
                        pass


# 注意：旧数据搬运改为在 __main__ 里显式调用（见文件末尾），
# 避免本文件被 import（例如写单元测试 / 被 auto_research 导入）时产生副作用。


# ====================================================================
# 【清理旧备份】—— 只保留最新的 N 个备份，其余删除
# 说明：软件自身（【一键重启·自动检测与备份】）每次会生成一份备份，
#       长期使用会越攒越多。这里内置一个清理逻辑，但【不自动执行】：
#       - 只有用户点击【一键重启（自动检测与备份）】按钮时才会触发；
#       - 触发后保留最新 N 个备份，其余删除。
#       只删“本软件的备份文件”，绝不碰源码、数据、历史记录与资金曲线。
# ====================================================================
BACKUP_KEEP = 3          # 保留最新的备份数量（可按需修改）


def _collect_backup_files():
    """收集本软件在各目录下产生的备份文件（去重后的绝对路径列表）。
    只认 train_xgb_gui 相关的备份名，避免误删数据类备份。"""
    import glob
    files = set()
    backups_dir = os.path.join(DATA_DIR, "backups")
    scan_dirs = [BASE_DIR, DATA_DIR, backups_dir]
    patterns = ("train_xgb_gui.py.bak*", "train_xgb_gui_backup*.py",
                "train_xgb_gui_*.bak")
    for d in scan_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            for p in glob.glob(os.path.join(d, pat)):
                if os.path.isfile(p):
                    files.add(os.path.abspath(p))
    return list(files)


def auto_cleanup_backups(keep=BACKUP_KEEP, log=None):
    """【自动清理旧备份】：保留最新的 keep 个备份，其余全部删除。
    返回被删除的文件路径列表。任何异常都吞掉，绝不影响主程序。"""
    try:
        files = _collect_backup_files()
    except Exception:
        return []
    if len(files) <= keep:
        return []

    def _mtime(p):
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0.0

    files.sort(key=_mtime, reverse=True)   # 最新在前
    removed = []
    for p in files[keep:]:
        try:
            os.remove(p)
            removed.append(p)
        except Exception:
            pass
    if removed and callable(log):
        try:
            log(f"🧹 自动清理旧备份：保留最新 {keep} 个，已删除 {len(removed)} 个旧备份")
        except Exception:
            pass
    return removed



# ====================================================================
# 多AI管理：每个AI一套独立的"模型文件 + 历史文件"
# ====================================================================
AI_LIST_FILE = os.path.join(DATA_DIR, "ais.json")


def model_path_for(ai_name, risk_control=False):
    suffix = "_风控" if risk_control else ""
    return os.path.join(DATA_DIR, f"model_{ai_name}{suffix}.json")


def run_history_path_for(ai_name, risk_control=False):
    """【跑测历史】专属独立文件：只有"跑1/10/50/100次"的结果写这里，
    与【调参寻优历史】(tuning_log_)、【特征寻优历史】(feature_search_log_)彻底分开。
    写入方(batch_worker)与读取方(历史记录页)都以本函数为唯一路径来源，保证对得上。"""
    suffix = "_风控" if risk_control else ""
    return os.path.join(DATA_DIR, f"history_{ai_name}{suffix}.csv")


def history_path_for(ai_name, risk_control=False):
    """兼容旧调用的别名，统一转发到 run_history_path_for（跑测历史）。"""
    return run_history_path_for(ai_name, risk_control)


def tuning_log_path_for(ai_name):
    """【调参寻优历史】独立文件：自动调参 / 特征寻优 的最终结果写这里，
    与"跑1/10/50/100次"的跑测历史彻底分开。"""
    return os.path.join(DATA_DIR, f"tuning_log_{ai_name}.csv")


def feature_search_log_path_for(ai_name):
    """【特征寻优历史】独立文件：只记录"特征寻优"的最终结果，
    与跑测历史、调参寻优历史彻底分开，互不覆盖。"""
    return os.path.join(DATA_DIR, f"feature_search_log_{ai_name}.csv")


def feature_search_gen_path_for(ai_name):
    """【特征寻优过程明细】每一代的临时明细日志（新文件名，
    避免与"特征寻优历史"记录文件 feature_search_log_AI.csv 冲突）。"""
    return os.path.join(DATA_DIR, f"feature_search_gen_{ai_name}.csv")


def strategy_search_gen_path_for(ai_name):
    """【自动调仓寻优过程明细】每个候选策略一行的临时明细日志。
    用独立的文件名，绝不覆盖旧的 param_tuning_log_AI.csv。"""
    return os.path.join(DATA_DIR, f"strategy_search_gen_{ai_name}.csv")


def load_ai_names():
    import json
    if os.path.exists(AI_LIST_FILE):
        try:
            with open(AI_LIST_FILE, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass
    names = ["AI1"]
    save_ai_names(names)
    return names


def save_ai_names(names):
    import json
    with open(AI_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)


def migrate_ai_name(old, new, data_dir=None, log_fn=None, copy_only=True):
    """把 AI 名字从 old 纠正为 new（默认“安全复制”，绝不重命名 / 覆盖）。

    安全规则（对应“旧文件必须先检查兼容性，不得直接批量重命名或覆盖”）：
      - 目标文件已存在  → 跳过，绝不覆盖；
      - 源文件不存在    → 跳过；
      - 只处理本软件已知后缀的文件（AI_FILE_PATTERNS）；
      - copy_only=True 时保留源文件（兼容旧历史记录），只新增新名字的文件；
        copy_only=False 时才 move（改名，仍不覆盖已存在目标）。
    返回 (copied, skipped_existing, missing) 三个文件名列表。
    """
    import shutil
    data_dir = data_dir or DATA_DIR
    copied, skipped, missing = [], [], []
    for pat in AI_FILE_PATTERNS:
        src = os.path.join(data_dir, pat.format(n=old))
        dst = os.path.join(data_dir, pat.format(n=new))
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if not os.path.exists(src):
            missing.append(os.path.basename(src))
            continue
        if os.path.exists(dst):
            skipped.append(os.path.basename(dst))
            continue
        try:
            if copy_only:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            copied.append(os.path.basename(dst))
        except Exception as e:
            if log_fn:
                log_fn(f"迁移 {os.path.basename(src)} 失败：{e}")
    return copied, skipped, missing


def normalize_ai_names(log_fn=None):
    """启动时检查 AI 名单里的历史错别字名字并纠正（安全复制，不覆盖）。

    返回纠正后的名字列表。若更名过程中目标文件已存在，则原样保留、只做提示。
    """
    names = load_ai_names()
    changed = False
    out = []
    for nm in names:
        fixed = AI_NAME_FIXES.get(str(nm).strip(), nm)
        if fixed != nm:
            if log_fn:
                log_fn(f"检测到历史 AI 名「{nm}」，正式名应为「{fixed}」，"
                       f"开始安全迁移（复制，不覆盖）...")
            copied, skipped, missing = migrate_ai_name(nm, fixed, log_fn=log_fn)
            if log_fn:
                log_fn(f"  AI名迁移完成：已复制 {len(copied)} 个文件"
                       + (f"，跳过已存在 {len(skipped)} 个（未覆盖）" if skipped else "")
                       + (f"，源缺失 {len(missing)} 个" if missing else ""))
                if copied:
                    log_fn(f"  新增文件：{', '.join(copied)}")
            changed = True
        if fixed not in out:
            out.append(fixed)
    if changed:
        # 先备份原名单，再写回纠正后的名单（可回滚）
        try:
            if os.path.exists(AI_LIST_FILE):
                import shutil as _sh
                _sh.copy2(AI_LIST_FILE, AI_LIST_FILE + ".bak")
        except Exception:
            pass
        save_ai_names(out)
    return out


def scan_all_ai_names():
    """自动扫描本地所有AI：AI名单(ais.json) + model_*.json + history_*.csv，去重。
    专供【历史记录】页的"查看AI"下拉菜单使用，保证新建或手工放进来的AI都能被找到。"""
    import glob
    names = []
    try:
        for nm in load_ai_names():
            nm = AI_NAME_FIXES.get(str(nm).strip(), str(nm).strip())
            if nm and nm not in names:
                names.append(nm)
    except Exception:
        pass

    def _add(raw):
        nm = str(raw).strip()
        # 去掉 model_xxx_Best.json / history_xxx_风控.csv 这类后缀，还原成AI名字
        for suf in ("_风控", "_Best"):
            while nm.endswith(suf):
                nm = nm[:-len(suf)]
        nm = AI_NAME_FIXES.get(nm.strip(), nm.strip())   # 旧错别字名统一显示为正式名
        if nm and nm not in names:
            names.append(nm)

    try:
        for p in glob.glob(os.path.join(DATA_DIR, "model_*.json")):
            base = os.path.basename(p)
            if base.startswith("model_") and base.endswith(".json"):
                _add(base[len("model_"):-len(".json")])
        for p in glob.glob(os.path.join(DATA_DIR, "history_*.csv")):
            base = os.path.basename(p)
            if base.startswith("history_") and base.endswith(".csv"):
                _add(base[len("history_"):-len(".csv")])
        for p in glob.glob(os.path.join(DATA_DIR, "tuning_log_*.csv")):
            base = os.path.basename(p)
            if base.startswith("tuning_log_") and base.endswith(".csv"):
                _add(base[len("tuning_log_"):-len(".csv")])
        for p in glob.glob(os.path.join(DATA_DIR, "feature_search_log_*.csv")):
            base = os.path.basename(p)
            if base.startswith("feature_search_log_") and base.endswith(".csv"):
                _add(base[len("feature_search_log_"):-len(".csv")])
    except Exception:
        pass
    return names


# ====================================================================
# 历史记录小工具
# ====================================================================
def ensure_history_file(path):
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(HISTORY_HEADER)


def append_history(path, row_values):
    ensure_history_file(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row_values)


def merge_header(file_header, base_header=None):
    """把 CSV 自己的表头和标准表头合并：保留 CSV 原有列（按原顺序），
    再把标准表里缺的列补到末尾。用来兼容旧格式文件。
    base_header 缺省用 HISTORY_HEADER（回测/调参寻优历史）。"""
    base = list(base_header) if base_header else list(HISTORY_HEADER)
    seen = set()
    header = []
    for c in list(file_header) + base:
        c = str(c).strip()
        if c and c not in seen:
            seen.add(c)
            header.append(c)
    return header


def read_history_rows(path, base_header=None):
    """【容错读取】读历史 CSV，返回 (表头, 数据行)。
    优先用 pandas 读取，再用 reindex 自动补全缺失的列（缺失值填 ""），
    旧格式文件也能顺利读出；pandas 万一读不了（格式太乱等），
    自动退回 csv 模块读取，绝不报错、不弹窗。
    base_header：标准表头（跑测历史用 HISTORY_HEADER，特征寻优历史用 FEATURE_SEARCH_HEADER）。"""
    base = list(base_header) if base_header else list(HISTORY_HEADER)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return list(base), []
    # ---- 优先 pandas：dtype=str 保证内容不被改写，reindex 补缺失列 ----
    try:
        import pandas as pd
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                         keep_default_na=False, na_values=[])
        file_header = [str(c).strip() for c in df.columns.tolist()]
        header = merge_header(file_header, base)
        df = df.reindex(columns=header, fill_value="")   # 缺的列自动补 ""
        df = df.fillna("")
        data = [[("" if v is None else str(v)) for v in row]
                for row in df.values.tolist()]
        return header, data
    except Exception:
        pass
    # ---- 退回 csv 模块读取（兼容极端脏文件）----
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except Exception:
        return list(base), []
    if not rows:
        return list(base), []
    header = merge_header(rows[0], base)
    data = []
    for r in rows[1:]:
        if not r:
            continue
        padded = list(r) + [""] * (len(header) - len(r))
        data.append(padded[:len(header)])
    return header, data


def ensure_feature_search_file(path):
    """确保【特征寻优历史】文件存在且表头正确。
    文件不存在/为空 → 写入标准表头；
    若已存在但不是标准表头（例如早期的"每代明细"旧文件）→
    先把旧文件改名备份（绝不覆盖旧数据），再建新文件。"""
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(FEATURE_SEARCH_HEADER)
        return
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            first = next(csv.reader(f), [])
    except Exception:
        first = []
    if [str(c).strip() for c in first] != FEATURE_SEARCH_HEADER:
        try:
            import shutil
            bak = path + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(path, bak)
        except Exception:
            pass
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(FEATURE_SEARCH_HEADER)


def append_feature_search_log(path, row_values):
    """【追加】向特征寻优历史文件追加一行，不覆盖旧数据。"""
    ensure_feature_search_file(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row_values)


def write_history_rows(path, data_rows, header=None):
    hdr = list(header) if header else list(HISTORY_HEADER)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in data_rows:
            w.writerow(r)


def migrate_tuning_log(ai_name):
    """【一次性数据搬家】
    早期版本把"自动调参 / 特征寻优"的结果也写进了 history_{AI}_风控.csv。
    这里把这些旧记录搬进独立的 tuning_log_{AI}.csv，并从原文件删除，保持两个文件干净。
    只做一次：tuning_log 一旦已有数据，就直接跳过，绝不重复搬运。
    只读写 CSV，不碰任何回测/特征计算逻辑。"""
    try:
        tuning_path = tuning_log_path_for(ai_name)
        hist_path = run_history_path_for(ai_name, risk_control=True)

        # 已经搬过了（tuning_log 里已有数据）→ 直接返回
        if os.path.exists(tuning_path) and os.path.getsize(tuning_path) > 0:
            _, existing = read_history_rows(tuning_path)
            if existing:
                return

        # 没有旧的"风控"历史文件 → 没得搬
        if not os.path.exists(hist_path) or os.path.getsize(hist_path) == 0:
            return

        header, data = read_history_rows(hist_path)
        if "触发方式" not in header:
            return
        trig_idx = header.index("触发方式")

        # 挑出"触发方式"= 自动调参 / 特征寻优 的行，其余原样保留
        move, keep = [], []
        for r in data:
            r = (list(r) + [""] * len(header))[:len(header)]
            trig = str(r[trig_idx]).strip() if trig_idx < len(r) else ""
            if trig in ("自动调参", "特征寻优"):
                move.append(r)
            else:
                keep.append(r)

        if not move:
            return

        # 按标准表头顺序，把旧记录追加写入 tuning_log
        ensure_history_file(tuning_path)
        t_header, _ = read_history_rows(tuning_path)
        if not t_header:
            t_header = list(HISTORY_HEADER)
        for r in move:
            row = [(r[header.index(c)] if c in header else "") for c in t_header]
            append_history(tuning_path, row)

        # 搬完从原文件删除这些行，防止重复
        write_history_rows(hist_path, keep, header)
    except Exception:
        pass


def normalize_history_header(path):
    """历史文件表头和最新格式不一致时，自动升级：缺的列补空值。
    保留 CSV 里原有的所有列，只补标准表里缺的列。"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except Exception:
        return
    if not rows:
        return
    file_header = [str(c).strip() for c in rows[0]]
    header = merge_header(file_header)
    data_rows = [(list(r) + [""] * len(header))[:len(header)]
                 for r in rows[1:] if r]
    if file_header == header and all(len(r) == len(header) for r in data_rows):
        return
    try:
        write_history_rows(path, data_rows, header)
    except Exception:
        pass


def get_desktop_dir():
    candidates = [
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.join(os.environ.get("USERPROFILE", ""), "Desktop"),
        os.path.join(os.environ.get("OneDrive", ""), "Desktop"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return os.path.expanduser("~")


def _parse_pct(s):
    """把 '+12.34%' / '-3.2%' 解析成数字；解析不了就返回 None。"""
    try:
        return float(str(s).replace("%", "").replace("+", "").strip())
    except Exception:
        return None


def _parse_num(s):
    try:
        return float(str(s).strip())
    except Exception:
        return None


# ====================================================================
# HTTPS 证书校验
# ====================================================================
# 保持 Python / requests 的默认证书校验（ssl 与 requests 的默认上下文），
# 不在此处关闭校验、覆盖全局 Context，也不 monkeypatch requests.get。


# ====================================================================
# 数据 / 指标 / 标签（重库在函数内部才导入）
# ====================================================================
def fetch_stock_data(code, start_yyyymmdd, end_yyyymmdd):
    import akshare as ak
    import pandas as pd
    df = None
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start_yyyymmdd, end_date=end_yyyymmdd, adjust="qfq")
    except Exception:
        df = None
    if df is None or df.empty:
        sina_symbol = ("sh" if code.startswith("6") else "sz") + code
        df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start_yyyymmdd,
                                 end_date=end_yyyymmdd, adjust="qfq")
        df = df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高',
                                'low': '最低', 'close': '收盘', 'volume': '成交量'})
        # 【单位统一】东财 stock_zh_a_hist 的“成交量”单位是“手”，而新浪
        # stock_zh_a_daily 的 volume 单位是“股”。这里把新浪兜底数据折算成“手”，
        # 保证下游（尤其 stock_pool.check_stock_data 的成交额 = 收盘 × 成交量 × 100）
        # 两种数据源口径一致，避免流动性指标被放大 100 倍。
        # 说明：主流水线里的 VOL_RATIO = 成交量/5日均量 是比值，单位不影响；
        # 只有需要绝对成交额的新增流动性过滤会受影响，故在此统一。
        if '成交量' in df.columns:
            df['成交量'] = df['成交量'] / 100.0
    keep = ['日期', '最高', '最低', '收盘']
    if '成交量' in df.columns:
        keep.append('成交量')
    df = df[keep].copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    return df


# ---- 上市首日 / 日期对齐（按代码缓存，避免重复请求）----
_FIRST_DATA_DATE_CACHE = {}


def get_first_data_date(code):
    """获取该股票有数据的第一天（即上市首日）。失败返回 None。"""
    if code in _FIRST_DATA_DATE_CACHE:
        return _FIRST_DATA_DATE_CACHE[code]
    first = None
    try:
        df = fetch_stock_data(code, "19900101", datetime.now().strftime("%Y%m%d"))
        if df is not None and len(df) and '日期' in df.columns:
            first = str(df['日期'].min().date())
    except Exception:
        first = None
    _FIRST_DATA_DATE_CACHE[code] = first
    return first


def align_train_start(code, train_start):
    """日期对齐：如果训练起始日期早于该股票有数据的第一天，
    自动调整到有数据的第一天。返回调整后的日期字符串。"""
    first = get_first_data_date(code)
    if not first:
        return train_start
    try:
        req_d = datetime.strptime(str(train_start).strip(), "%Y-%m-%d")
        first_d = datetime.strptime(first, "%Y-%m-%d")
        if req_d < first_d:
            return first
    except Exception:
        pass
    return train_start


# ====================================================================
# 智能参数推荐：纯计算模块（只算“合适的寻优范围”，绝不跑回测 / 不改配置）
# ====================================================================
def compute_atr_pct_mean(df, period=14):
    """计算日线 period 日 ATR 的均值，换算成“占收盘价的百分比 x%”。
    数据不足返回 None。"""
    import pandas as pd
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    atr_pct = atr / close * 100.0
    atr_pct = atr_pct.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(atr_pct) == 0:
        return None
    return float(atr_pct.mean())


def compute_swing_gains(df, window=10):
    """识别历史数据里的“波段低点 → 波段高点”，返回每次涨幅（%）列表。
    用窗口极值法找摆动点，再强制高低交替配对，纯统计、不看未来。"""
    highs = df['最高'].astype(float).to_numpy()
    lows = df['最低'].astype(float).to_numpy()
    n = len(df)
    if n < window * 2 + 1:
        return []

    # 1) 找窗口极值点：某点在 ±window 内是最低/最高，则记为波段低点/高点
    pivots = []
    for i in range(window, n - window):
        seg_high = highs[i - window:i + window + 1]
        seg_low = lows[i - window:i + window + 1]
        if lows[i] == seg_low.min():
            pivots.append((i, 'L', lows[i]))
        if highs[i] == seg_high.max():
            pivots.append((i, 'H', highs[i]))
    pivots.sort(key=lambda p: p[0])

    # 2) 强制高低交替：同类型相邻时保留更极端的那个
    seq = []
    for p in pivots:
        if not seq:
            seq.append(p)
            continue
        last = seq[-1]
        if p[1] == last[1]:
            if p[1] == 'L' and p[2] < last[2]:
                seq[-1] = p
            elif p[1] == 'H' and p[2] > last[2]:
                seq[-1] = p
        else:
            seq.append(p)

    # 3) 每对“低点→高点”算一次涨幅
    gains = []
    for j in range(1, len(seq)):
        a, b = seq[j - 1], seq[j]
        if a[1] == 'L' and b[1] == 'H' and a[2] > 0:
            gains.append((b[2] - a[2]) / a[2] * 100.0)
    return gains


def recommend_ranges(df):
    """根据历史日线数据计算寻优范围。
    返回 (止损下限, 止损上限, 止盈下限, 止盈上限, 均线下限, 均线上限)。
    本函数只返回数值，不写界面、不启动任何任务。"""
    import numpy as np

    # ---- 止损：基于 14 日 ATR 均值（硬性截断到 [4%, 12%]） ----
    x = compute_atr_pct_mean(df, period=14)
    if x is None or x <= 0:
        sl_lo, sl_hi = 6, 10          # 数据异常时退回稳健默认值
    else:
        sl_lo = min(12, max(4, round(1.2 * x)))   # 绝对下限4%，绝对上限12%
        sl_hi = min(12, max(6, round(2.0 * x)))   # 上限只乘2，绝不超12%
    if sl_hi <= sl_lo:                            # 防止区间退化（且不突破4~12封顶）
        sl_lo = max(4, sl_hi - 1)

    # ---- 止盈：波段低点到高点涨幅的 25 / 75 分位数（硬性截断到 [15%, 45%]） ----
    gains = [g for g in compute_swing_gains(df, window=10) if g > 0]
    if gains:
        tp_lo = min(45, max(15, int(round(float(np.percentile(gains, 25))))))   # 下限封底15%、封顶45%
        tp_hi = min(45, max(25, int(round(float(np.percentile(gains, 75))))))   # 上限封底25%、封顶45%
    else:
        tp_lo, tp_hi = 25, 40         # 没识别到波段时退回稳健默认值
    if tp_hi <= tp_lo:                            # 防止区间退化（且不突破15~45封顶）
        tp_lo = max(15, tp_hi - 1)

    # ---- 均线：固定 [10, 60] ----
    ma_lo, ma_hi = 10, 60
    return sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi


def recommend_search_trials(sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi,
                            target_coverage=0.8, min_trials=100, max_trials=MAX_SEARCH_TRIALS):
    """根据上一步算出的寻优范围，估算【寻优次数上限】。

    照顾老电脑的硬性截断规则：
      - 建议次数固定为 150 次（无论搜索空间多大，绝不按“覆盖率模型”狂推次数）；
      - 绝对上限 MAX_SEARCH_TRIALS（=200）次，绝不推荐 500 / 1000 / 5000 次；
      - 仍返回组合数 N 供日志展示“搜索空间大小”。
    返回 (建议次数, 组合数 N)。只做计算，不写界面、不启动任何任务。"""
    n_sl = int(sl_hi - sl_lo) + 1
    n_tp = int(tp_hi - tp_lo) + 1
    n_ma = int(ma_hi - ma_lo) + 1
    space = max(1, n_sl * n_tp * n_ma)
    trials = min(max_trials, max(min_trials, 150))   # 固定建议150，最多200
    return trials, space


def recommend_feat_trials(sl, tp, ma, max_cap=MAX_SEARCH_TRIALS):
    """【智能寻优次数上限推荐】纯计算模块（不联网 / 不跑回测 / 不改其它配置）。

    先看【特征寻优参数（固定参数）】的三个值（止损 sl、止盈 tp、均线 ma），
    再据此估算“需要尝试多少个特征组合才够用”：
      - 止盈越小（越保守）→ 信号越少越不稳 → 需要多试；
      - 止损越偏离健康区 [5,10] → 越容易过拟合 → 需要多试；
      - 均线越短 → 越敏感、越容易被噪声带偏 → 需要多试；反之少试。
    老电脑保护：结果硬性压在 [80, max_cap] 之间，max_cap 默认 200，绝不超过 200。
    返回 (建议次数, 明细 dict)。"""
    try:
        sl = float(sl); tp = float(tp); ma = float(ma)
    except Exception:
        return 150, {"score": 0.5, "reason": "参数无法解析，退回保守中值 150"}
    if sl <= 0 or tp <= 0 or ma <= 0:
        return 150, {"score": 0.5, "reason": "参数非正数，退回保守中值 150"}

    # 1) 止盈：15~30 为健康区；<15 太保守（信号少、不稳）→ 多试；>30 太贪婪 → 少试
    if tp < 15:
        tp_score = 1.0
    elif tp > 30:
        tp_score = 0.2
    else:
        tp_score = 1.0 - (tp - 15.0) / 15.0 * 0.8        # 15→1.0，30→0.2

    # 2) 止损：5~10 为健康区；越靠边（太紧 / 太松）越需要多试
    if sl < 5 or sl > 10:
        sl_score = 1.0
    else:
        sl_score = 0.3 + abs(sl - 7.5) / 2.5 * 0.5       # 中点0.3，边缘0.8

    # 3) 均线：越短越敏感 → 多试；越长越平滑 → 少试
    ma_score = 1.0 - min(1.0, max(0.0, (ma - 10.0) / 50.0)) * 0.6   # 10→1.0，60→0.4

    score = 0.45 * tp_score + 0.30 * sl_score + 0.25 * ma_score
    trials = int(round(80 + (int(max_cap) - 80) * score))
    trials = max(80, min(int(max_cap), trials))
    return trials, {"score": score, "sl": sl, "tp": tp, "ma": ma}


def _fmt_range_num(v):
    """把范围端点格式化成输入框里好看的字符串：整数去掉 .0，小数最多留 1 位。"""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return ("%.1f" % f).rstrip("0").rstrip(".")


def compute_second_ranges(s1, t1, m1):
    """【智能二次动态定界】纯计算模块：只算范围与次数，绝不跑回测、不改配置。

    输入：第一次【自动调参】得到的最佳参数（止损 s1，止盈 t1，均线 m1）。
    输出：(sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi, trials, bias)
    寻优次数硬锁上限 MAX_SEARCH_TRIALS（=200 次）。
    bias 取值："greedy"（贪婪）/ "conservative"（保守）/ "healthy"（健康）。

    ⚡老电脑保护：本函数只返回数字，不写界面、不启动任何任务。跑不跑由用户决定。
    """
    # ---------------- 1) 止盈分析（决定止盈范围） ----------------
    if t1 > 30:                       # 太贪婪 → 上限降到 30，收窄
        bias = "greedy"
        tp_hi = 30
        tp_lo = t1 - 10
        if tp_lo >= tp_hi:            # T1 过大导致公式反转 → 以封顶30为准，保持10的宽度
            tp_lo = tp_hi - 10
    elif t1 < 15:                     # 太保守 → 放宽，上限外扩 15
        bias = "conservative"
        tp_lo, tp_hi = t1, t1 + 15
    else:                             # 健康（15~30）→ 微调 ±5
        bias = "healthy"
        tp_lo, tp_hi = t1 - 5, t1 + 5

    # ---------------- 2) 止损分析（决定止损范围） ----------------
    if s1 < 5:                        # 太紧 → 放宽
        sl_lo, sl_hi = 5, s1 + 5
    elif s1 > 10:                     # 太宽 → 收紧到 10
        sl_hi = 10
        sl_lo = s1 - 3
        if sl_lo >= sl_hi:            # S1 过大导致公式反转 → 以封顶10为准，保持3的宽度
            sl_lo = sl_hi - 3
    else:                             # 健康（5~10）→ 保持 ±2
        sl_lo, sl_hi = s1 - 2, s1 + 2

    # ---------------- 3) 均线分析（决定均线范围） ----------------
    if m1 <= 20:
        ma_lo, ma_hi = 10, 30
    else:
        ma_lo, ma_hi = 15, 60

    # ---------------- 4) 范围合法性修正（防止区间退化） ----------------
    sl_lo = max(1.0, sl_lo)
    if sl_hi <= sl_lo:
        sl_hi = sl_lo + 1
    tp_lo = max(1.0, tp_lo)
    if tp_hi <= tp_lo:
        tp_hi = tp_lo + 1
    ma_lo = max(5, int(round(ma_lo)))
    ma_hi = max(ma_lo + 1, int(round(ma_hi)))

    # ---------------- 5) 寻优次数上限动态计算（老电脑保护核心） ----------------
    # 空间宽度归一化（各维除以“典型最大宽度”，再取均值）→ 0~1
    width_score = ((sl_hi - sl_lo) / 12.0
                   + (tp_hi - tp_lo) / 30.0
                   + (ma_hi - ma_lo) / 60.0) / 3.0
    width_score = max(0.0, min(1.0, width_score))
    # 偏离度：止盈偏离健康区间 [15,30] 中心 22.5 的程度 → 0~1
    dev = max(0.0, min(1.0, abs(t1 - 22.5) / 15.0))
    score = 0.5 * width_score + 0.5 * dev

    if bias == "conservative":
        # 放宽范围、空间变大 → 次数增加，但最多只给 MAX_SEARCH_TRIALS
        trials = int(round(150 + 50 * score))
        trials = max(150, min(MAX_SEARCH_TRIALS, trials))
    elif bias == "greedy":
        # 收窄范围、空间变小 → 次数减少，给 100~120
        trials = int(round(100 + 20 * score))
        trials = max(100, min(120, trials))
    else:
        # 健康、范围仅微调 → 保持 150
        trials = 150

    return (sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi, trials, bias)


def _latest_close_price(code):
    """取该股票最新收盘价（估算市值用）。取不到返回 None。"""
    try:
        import akshare as ak
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        df = None
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                     start_date=start, end_date=today,
                                     adjust="qfq")
        except Exception:
            df = None
        if df is None or df.empty:
            sina_symbol = ("sh" if code.startswith("6") else "sz") + code
            df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start,
                                      end_date=today, adjust="qfq")
            df = df.rename(columns={"close": "收盘"})
        if df is not None and not df.empty and "收盘" in df.columns:
            return float(df["收盘"].iloc[-1])
    except Exception:
        pass
    return None


def add_indicators(df):
    import pandas as pd
    close = df['收盘']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])

    low9 = df['最低'].rolling(9).min()
    high9 = df['最高'].rolling(9).max()
    rsv = (close - low9) / (high9 - low9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['BOLL_UP'] = mid + 2 * std
    df['BOLL_DN'] = mid - 2 * std

    if '成交量' in df.columns:
        df['VOL_MA5'] = df['成交量'].rolling(5).mean()

    df['RET1'] = close.pct_change() * 100
    df['RET5'] = close.pct_change(5) * 100
    df['RET20'] = close.pct_change(20) * 100
    # 【增强特征组 ENH】更长的动量 + 已实现波动率 + 振幅（离线走查显示这些
    # 与“波动率自适应标签”搭配时，方向 AUC 稳定约 0.58~0.63，明显优于原特征）。
    df['RET3'] = close.pct_change(3) * 100
    df['RET10'] = close.pct_change(10) * 100
    df['RET60'] = close.pct_change(60) * 100
    df['RVOL5'] = close.pct_change().rolling(5).std() * 100
    df['RVOL20'] = close.pct_change().rolling(20).std() * 100
    df['HL_RANGE'] = (df['最高'].astype(float) - df['最低'].astype(float)) / close

    # ATR14
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR_PCT'] = df['ATR14'] / close

    return df


def make_features(df, switches):
    import pandas as pd
    X = pd.DataFrame(index=df.index)
    names = []
    close = df['收盘']

    def add(name, series):
        X[name] = series
        names.append(name)

    if switches.get("MA5"):
        add("MA5_dev", close / df['MA5'] - 1)
    if switches.get("MA10"):
        add("MA10_dev", close / df['MA10'] - 1)
    if switches.get("MA20"):
        add("MA20_dev", close / df['MA20'] - 1)
    if switches.get("MA60"):
        add("MA60_dev", close / df['MA60'] - 1)
    if switches.get("RSI"):
        add("RSI14", df['RSI14'])
    if switches.get("MACD"):
        add("DIF", df['DIF']); add("DEA", df['DEA']); add("MACD_hist", df['MACD'])
    if switches.get("KDJ"):
        add("K", df['K']); add("D", df['D']); add("J", df['J'])
    if switches.get("BOLL"):
        add("BOLL_UP_dev", close / df['BOLL_UP'] - 1)
        add("BOLL_DN_dev", close / df['BOLL_DN'] - 1)
    if switches.get("VOL") and 'VOL_MA5' in df.columns:
        add("VOL_RATIO", df['成交量'] / df['VOL_MA5'])
    if switches.get("RET"):
        add("RET1", df['RET1']); add("RET5", df['RET5']); add("RET20", df['RET20'])
    if switches.get("ATR"):
        add("ATR14", df['ATR14'])
    if switches.get("ENH"):
        add("RET3", df['RET3']); add("RET10", df['RET10']); add("RET60", df['RET60'])
        add("RVOL5", df['RVOL5']); add("RVOL20", df['RVOL20'])
        add("HL_RANGE", df['HL_RANGE']); add("ATR_PCT", df['ATR_PCT'])
    return X, names


def build_labels(df, mode=None, horizon=None):
    """按 LABEL_MODE 生成标签（可切换）。默认与原行为一致（fixed）。

    - mode="fixed" : 未来 horizon 日涨跌超过 UP/DOWN_THRESHOLD → +1 / -1 / 0
    - mode="vol"   : 阈值 = LABEL_VOL_K × 近 LABEL_VOL_WINDOW 日收益波动率 × √horizon
    - mode="binary": 未来 horizon 日涨 → +1，跌 → -1（无中性类）
    注意：这里用 shift(-horizon) 取“未来”收益作为**标签**，属于正常的监督学习构造，
    不构成交易时的未来函数（交易信号的执行时点由 TradingEngine 的 T+1 逻辑保证）。
    """
    import numpy as np
    import pandas as pd
    mode = mode or LABEL_MODE
    h = int(horizon or LABEL_HORIZON)
    future_ret = df['收盘'].shift(-h) / df['收盘'] - 1
    if mode == "binary":
        label = pd.Series(np.nan, index=df.index)
        label[future_ret > 0] = 1
        label[future_ret <= 0] = -1
    elif mode == "vol":
        sigma = df['收盘'].pct_change().rolling(LABEL_VOL_WINDOW).std()
        thr = LABEL_VOL_K * sigma * np.sqrt(h)
        label = pd.Series(0, index=df.index)
        label[future_ret > thr] = 1
        label[future_ret < -thr] = -1
        label[thr.isna()] = np.nan
    else:  # "fixed"（原行为）
        label = pd.Series(0, index=df.index)
        label[future_ret > UP_THRESHOLD] = 1
        label[future_ret < DOWN_THRESHOLD] = -1
    label[future_ret.isna()] = np.nan
    df['label'] = label
    return df


def predictive_power(y_true, y_pred, y_proba=None):
    """【预测力自检】比较“模型准确率 / AUC vs 基准”，判断模型是否真有信号。

    返回 (准确率, 多数类基准, AUC, 是否有初步信号, 说明文字)。
    - 只看“验证段”，绝不使用测试段；测试段只在最终跑测里出现一次。
    - 交易更依赖“排序能力(AUC)”而非硬分类准确率，故 AUC>0.55 也算有信号。
    - 若模型把所有样本都预测成同一类（退化），直接判定为“无预测力”。
    - y_proba 可选（预测概率），给了才能算 AUC。
    """
    import numpy as np
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0:
        return 0.0, 0.0, float("nan"), False, "验证段无样本"
    acc = float((y_true == y_pred).mean())
    vals, cnts = np.unique(y_true, return_counts=True)
    majority = float(cnts.max() / len(y_true))
    auc = float("nan")
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            P = np.asarray(y_proba)
            if P.ndim == 2 and P.shape[1] == 2:
                auc = float(roc_auc_score(y_true, P[:, 1]))
            elif P.ndim == 2 and P.shape[1] > 2:
                auc = float(roc_auc_score(y_true, P, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
    auc_ok = (auc == auc) and auc > 0.55
    acc_ok = acc > majority + PREDICTIVE_EDGE_MARGIN
    degenerate = len(np.unique(y_pred)) <= 1
    parts = [f"验证段准确率{acc:.3f} vs 多数类基准{majority:.3f}"]
    if auc == auc:
        parts.append(f"AUC={auc:.3f}")
    note = "；".join(parts)
    if auc_ok:
        # AUC 排序有信号即算有信号（交易可按预测概率排序下单，不必依赖硬分类）
        has_edge = True
        note += " → 有初步信号（AUC 排序有效，仍需最终测试段确认）"
        if degenerate:
            note += "；⚠硬分类退化为单一类别，请按预测概率排序决定买卖"
    elif acc_ok and not degenerate:
        has_edge = True
        note += " → 有初步信号（仍需最终测试段确认）"
    else:
        has_edge = False
        if degenerate:
            note += "；⚠模型退化为单一类别（一直预测同一结果），无预测力"
        else:
            note += " → 未超过基准，无可靠预测力（收益主要来自行情/风控，而非模型预测）"
    return acc, majority, auc, has_edge, note


# ====================================================================
# 【自动调仓寻优·纯函数】范围解析 / 候选采样 / 合格判定
# 拆成纯函数，便于离线单元测试；不依赖网络与界面。
# ====================================================================
def parse_search_range(text, as_int=False):
    """解析“最小-最大”范围字符串（也兼容中文破折号/波浪号）。合法返回 (lo, hi)，非法返回 None。"""
    try:
        s = str(text).strip().replace("~", "-").replace("—", "-").replace("－", "-")
        if s == "":
            return None
        if "-" not in s:
            v = float(s)
            lo = hi = v
        else:
            parts = [p for p in s.split("-") if p.strip() != ""]
            if len(parts) != 2:
                return None
            lo, hi = float(parts[0]), float(parts[1])
        if lo > hi:
            lo, hi = hi, lo
        if as_int:
            lo, hi = int(round(lo)), int(round(hi))
            lo = max(2, lo)
            if hi <= lo:
                hi = lo + 1
        else:
            lo = max(0.1, lo)
            if hi <= lo:
                hi = round(lo + 0.1, 4)
        return (lo, hi)
    except Exception:
        return None


def sample_strategy_candidates(sl_lo, sl_hi, tp_lo, tp_hi, ma_lo, ma_hi,
                               n, anchors=None, seed=None):
    """在给定范围内采样 n 个候选 (止损%, 止盈%, 均线周期, 跌破均线卖出0/1)。

    先放入 anchors（例如推荐中点 / 固定参数），再随机补足并去重。空间很小时可能少于 n。
    """
    import random
    rng = random.Random(seed)
    cands, seen = [], set()

    def push(sl, tp, ma, mb):
        sl = round(max(0.5, min(float(sl_hi), float(sl))), 1)
        tp = round(max(0.5, min(float(tp_hi), float(tp))), 1)
        ma = int(max(2, min(int(round(ma_hi)), int(round(ma)))))
        mb = 1 if mb else 0
        key = (sl, tp, ma, mb)
        if key not in seen:
            seen.add(key)
            cands.append(key)

    for a in (anchors or []):
        try:
            push(*a)
        except Exception:
            continue
    n = max(1, int(n))
    ma_lo_i, ma_hi_i = int(round(ma_lo)), int(round(ma_hi))
    if ma_hi_i < ma_lo_i:
        ma_lo_i, ma_hi_i = ma_hi_i, ma_lo_i
    guard, guard_max = 0, n * 50 + 1000
    while len(cands) < n and guard < guard_max:
        guard += 1
        push(rng.uniform(sl_lo, sl_hi),
             rng.uniform(tp_lo, tp_hi),
             rng.randint(ma_lo_i, ma_hi_i),
             rng.random() < 0.5)
    return cands


def is_strategy_qualified(val_ret_pct, max_dd_frac, n_trades,
                          min_trades=STRATEGY_MIN_TRADES,
                          max_dd=STRATEGY_MAX_DD,
                          require_positive=STRATEGY_REQUIRE_POSITIVE):
    """判断一个候选策略是否“合格”。不合格也照常记录，只是不会被选中。

    返回 (是否合格, 原因描述)。合格需同时满足：交易次数够、回撤不过大、净收益为正。
    """
    reasons = []
    if int(n_trades) < int(min_trades):
        reasons.append(f"交易次数不足({int(n_trades)}<{int(min_trades)})")
    if max_dd_frac is not None and max_dd_frac < -abs(max_dd):
        reasons.append(f"回撤过大({max_dd_frac * 100:.1f}%)")
    if require_positive and (val_ret_pct is None or val_ret_pct <= 0):
        reasons.append("验证段净收益非正")
    ok = len(reasons) == 0
    return ok, ("合格" if ok else "；".join(reasons))



# ====================================================================
# 交易模拟引擎
# ====================================================================
class TradingEngine:
    def __init__(self, initial_capital, risk_control=True, stop_loss_pct=5.0, 
                 take_profit_pct=15.0, ma_period=20,
                 friction=False, commission_rate=0.0003,
                 stamp_tax_rate=0.001, slippage_rate=0.001,
                 ma_break_exit=True):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0.0
        self.equity = []
        self.trades = []
        self.n_trades = 0                     # 完整交易（卖出）次数
        
        # 风控参数
        self.risk_control = risk_control      # 是否启用风控
        self.stop_loss_pct = stop_loss_pct    # 止损百分比（比如5表示亏5%就跑）
        self.take_profit_pct = take_profit_pct  # 止盈百分比（比如15表示赚15%就跑）
        self.ma_period = ma_period            # 均线周期（比如20表示用MA20判断）
        # 【策略规则】是否“跌破均线就卖出”。True=启用；False=只靠止损/止盈/AI信号。
        self.ma_break_exit = bool(ma_break_exit)

        # 真实交易成本参数（friction=False 时全部忽略，行为与原来完全一致）
        self.friction = friction              # 是否启用真实交易成本
        self.commission_rate = commission_rate    # 手续费率（小数，如0.0003=0.03%）
        self.stamp_tax_rate = stamp_tax_rate      # 印花税率（小数，仅卖出收取）
        self.slippage_rate = slippage_rate        # 滑点（小数）
        self._cost_basis = 0.0                # 本次建仓实际投入的现金（用于算净收益率）
        
        # 持仓状态
        self.buy_price = 0.0    # 买入价格
        self.buy_date = None    # 买入日期
        self.holding = False    # 是否持仓
        self.cooldown = 0       # 空仓冷却期（卖出后1天不能买）
        
        # 详细交易记录
        self.trade_log = []     # 每笔交易的详细信息
    
    def _buy(self, date, price):
        """买入：把现金全换成股票。启用真实成本时：成交价上浮滑点 + 扣手续费。"""
        if self.friction:
            eff_price = price * (1 + self.slippage_rate)          # 滑点：买入价上浮
            # 预留手续费，保证「股票成本 + 手续费」正好花光现金
            self.shares = self.cash / (eff_price * (1 + self.commission_rate))
            self._cost_basis = self.cash
            self.cash = 0.0
            self.buy_price = eff_price
        else:
            self.shares = self.cash / price
            self._cost_basis = self.cash
            self.cash = 0.0
            self.buy_price = price
        self.buy_date = date
        self.holding = True
        self.trades.append("买")
    
    def _sell(self, date, price, reason):
        """卖出：把股票全换成现金，并记录交易信息。
        启用真实成本时：成交价下浮滑点 + 扣手续费 + 扣印花税（仅卖出）。"""
        if self.friction:
            eff_price = price * (1 - self.slippage_rate)          # 滑点：卖出价下浮
            gross = self.shares * eff_price
            commission = gross * self.commission_rate
            stamp_tax = gross * self.stamp_tax_rate               # 印花税仅卖出收取
            self.cash = gross - commission - stamp_tax
            ret = (self.cash - self._cost_basis) / self._cost_basis * 100 if self._cost_basis else 0.0
            sell_price_log = eff_price
            self.shares = 0.0
        else:
            self.cash = self.shares * price
            ret = (price - self.buy_price) / self.buy_price * 100
            sell_price_log = price
            self.shares = 0.0
        self.holding = False
        self.trades.append("卖")
        self.n_trades += 1              # 卖出 = 完成一次完整交易
        # 记录详细交易日志
        self.trade_log.append({
            '买入日期': self.buy_date,
            '买入价': self.buy_price,
            '卖出日期': date,
            '卖出价': sell_price_log,
            '收益率': ret,
            '卖出原因': reason
        })
    
    def step(self, date, price, ma_val, action, decision_price=None):
        """
        每天执行一次交易判断。

        ⚠️【成交时点·杜绝未来函数】本方法把“决策”和“成交”分开：
        - 决策用的价格/均线是 **上一根已收盘 bar** 的值（decision_price、ma_val）；
        - 成交价是本根 bar 的收盘价 price（T+1，信号次日成交）。
        即信号在第 k 根 bar 收盘产生，第 k+1 根 bar 收盘才成交，绝不“同 bar 看到即吃到”。

        date: 当前（成交）日期
        price: 当前（成交）收盘价
        ma_val: 上一根 bar 的均线值；None/NaN 表示数据不足（不启用均线规则）
        action: AI信号 (-1=卖出, 0=持仓不动, 1=买入)，来自上一根 bar
        decision_price: 上一根 bar 的收盘价（缺省=price，兼容旧调用）
        """
        dp = price if decision_price is None else decision_price
        # ========== 持仓中：先检查风控 ==========
        if self.holding:
            # 用“决策 bar”的收盘价计算当前盈亏，用“成交 bar”的收盘价成交
            current_return = (dp - self.buy_price) / self.buy_price * 100

            # 优先级1: 止损（亏损超过设定百分比）
            if self.risk_control and current_return <= -self.stop_loss_pct:
                self._sell(date, price, f"止损{self.stop_loss_pct}%")
                self.cooldown = 1
                self.equity.append(self.cash + self.shares * price)
                return

            # 优先级2: 跌破均线（策略规则，可被寻优开关，默认启用）
            if (self.risk_control and self.ma_break_exit
                    and ma_val is not None and ma_val == ma_val and dp < ma_val):
                self._sell(date, price, f"跌破{self.ma_period}日线")
                self.cooldown = 1
                self.equity.append(self.cash + self.shares * price)
                return

            # 优先级3: 止盈（盈利超过设定百分比）
            if self.risk_control and current_return >= self.take_profit_pct:
                self._sell(date, price, f"止盈{self.take_profit_pct}%")
                self.cooldown = 1
                self.equity.append(self.cash + self.shares * price)
                return

            # 优先级4: AI卖出信号
            if action == -1:
                self._sell(date, price, "AI信号")
                self.cooldown = 1

        # ========== 空仓中：检查是否买入 ==========
        elif not self.holding and self.cash > 0:
            # 冷却期检查（卖出后1天不能买）
            if self.cooldown > 0:
                self.cooldown -= 1
            elif action == 1:
                # 买入前提：决策 bar 收盘价必须大于均线（防止一买就破线）
                if self.risk_control and ma_val is not None and ma_val == ma_val:
                    if dp > ma_val:
                        self._buy(date, price)
                else:
                    self._buy(date, price)

        # 记录当天净值（用成交 bar 收盘价估值）
        self.equity.append(self.cash + self.shares * price)

    def result(self):
        import pandas as pd
        eq = pd.Series(self.equity)
        final = eq.iloc[-1] if len(eq) else self.initial_capital
        total_return = final / self.initial_capital - 1
        max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
        return final, total_return, max_dd


# ====================================================================
# 【统一回测器】所有回测（批量跑测 / 自动调仓寻优 / 特征寻优 / 最佳复算）
# 都必须走这里，保证“成交时点、交易成本、回撤、交易次数”口径完全一致。
# 规则：第 k 根 bar 收盘产生信号，第 k+1 根 bar 收盘成交（T+1，无未来函数）。
# ====================================================================
def run_backtest(capital, risk_control, stop_loss_pct, take_profit_pct, ma_period,
                 friction, commission_rate, stamp_tax_rate, slippage_rate,
                 dates, closes, ma_vals, actions, ma_break_exit=True):
    """跑一次回测，返回 TradingEngine 实例（用 .result() / .equity / .n_trades 取值）。

    dates / closes / ma_vals / actions 为等长序列；ma_vals 可为 None（不启用均线）。
    """
    eng = TradingEngine(capital, risk_control, stop_loss_pct, take_profit_pct, ma_period,
                        friction, commission_rate, stamp_tax_rate, slippage_rate,
                        ma_break_exit=ma_break_exit)
    n = len(closes)
    for k in range(n):
        # 沿用同一均线列
        if ma_vals is None:
            dm = None
        else:
            dm = ma_vals[k - 1] if k > 0 else None
        if k == 0:
            # 第 0 根没有“上一根”，不产生任何信号（避免首日建仓的未来函数）
            eng.step(dates[0], closes[0], dm, 0, decision_price=closes[0])
        else:
            eng.step(dates[k], closes[k], dm, actions[k - 1], decision_price=closes[k - 1])
    return eng



# ====================================================================
# 平均资金曲线累加器
# 把每一次回测的净值数组按“日期”逐日累加，最后除以覆盖到该日期的次数，
# 得到一条“平均资金曲线”。内存占用只和“不同日期的条数”有关，
# 即使跑上千次也不会因为保存上千条数组而爆内存。
# ====================================================================
class EquityAverage:
    def __init__(self):
        self._total = {}     # 日期 -> 净值累加
        self._count = {}     # 日期 -> 累加次数
        self.n_curves = 0    # 一共加进来的曲线条数（= 实际回测次数）

    def add(self, dates, equity):
        """加入一条资金曲线（日期与净值一一对应）。"""
        n = min(len(dates), len(equity))
        for i in range(n):
            d = dates[i]
            self._total[d] = self._total.get(d, 0.0) + float(equity[i])
            self._count[d] = self._count.get(d, 0) + 1
        self.n_curves += 1

    def result(self):
        """返回 (日期列表, 平均净值列表)，按日期升序。没有数据时返回 ([], [])。"""
        if not self._count:
            return [], []
        ordered = sorted(self._total.keys())
        avg = [self._total[d] / self._count[d] for d in ordered]
        return ordered, avg


# ====================================================================
# 主程序
# ====================================================================
class _MacLogView(tk.Listbox):
    """macOS Tk 8.5 的只读日志视图。

    该版本 Tk 的 Text 控件即使空白也会持续重绘；Listbox 不会。
    这里保留 Text 的常用调用形态（insert/delete/see/index/tag_configure），
    让原有日志写入代码不需要改训练流程。
    """

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("activestyle", "none")
        kwargs.setdefault("exportselection", False)
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("borderwidth", 0)
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("selectmode", "browse")
        super().__init__(parent, **kwargs)
        self._tag_colors = {}

    def tag_configure(self, tag, **kwargs):
        color = kwargs.get("foreground")
        if color:
            self._tag_colors[tag] = color

    def insert(self, index, text, tag=None):
        s = "" if text is None else str(text)
        lines = s.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if not lines:
            lines = [""]
        for line in lines:
            super().insert(tk.END, line)
            color = self._tag_colors.get(tag) if tag else None
            if color:
                self.itemconfigure(tk.END, foreground=color)
        return None

    def delete(self, first, last=None):
        if last is None or str(last) == str(tk.END):
            super().delete(0, tk.END)
            return
        try:
            end_line = int(str(last).split(".")[0])
        except Exception:
            end_line = 1
        count = max(0, end_line - 1)
        if count:
            super().delete(0, count - 1)

    def index(self, index):
        return f"{self.size()}.0"

    def see(self, index):
        super().see(tk.END)


class TrainApp:
    """【一键自动训练】极简界面。

    界面只保留：股票代码 / 训练起止 / 测试起止 / 一个「一键自动训练」按钮，
    外加任务状态与进度、最终结果摘要、运行日志、历史记录两个辅助视图。
    所有研究方案（特征 / 模型 / 策略 / 预算 / 入选规则）由 auto_research.py 决定，
    训练在后台线程执行，绝不阻塞 Tkinter 主线程。
    """

    def __init__(self, root):
        self.root = root
        self.root.title("XGBoost 量化模型 · 一键自动训练")
        try:
            self.root.update_idletasks()
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = min(1360, sw - 40)
            h = min(900, sh - 90)
            self.root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+10")
            self.root.minsize(1120, 720)
            self.root.configure(bg=UI_BG)
            self.root.option_add("*Background", UI_BG)
            self.root.option_add("*Foreground", UI_FG)
            self.root.option_add("*Entry.Background", UI_INPUT)
            self.root.option_add("*Entry.Foreground", UI_FG)
            self.root.option_add("*selectBackground", UI_ACCENT)
            self.root.option_add("*selectForeground", "#ffffff")
        except Exception:
            pass
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.q = queue.Queue()
        self.working = False
        self._log_buf = []
        # 让安全报错弹窗把完整报错写入主界面运行日志
        set_error_log_sink(self.write)
        self.stop_ev = threading.Event()
        self._runner = None
        self._worker_thread = None
        self._stage_status = {}
        self.risk_var = tk.BooleanVar(value=True)   # 兼容占位（自动研究一律启用风控）
        self.ai_names = load_ai_names()
        self._hist_initialized = False
        self._startup_mtime = 0.0
        try:
            self._startup_mtime = os.path.getmtime(os.path.abspath(__file__))
        except Exception:
            pass
        self._restart_in_progress = False
        self._restart_ticks = 0
        self._paused = False
        self._running = False
        self._runner = None
        self._worker_thread = None
        self._experiment_count_done = 0
        self._experiment_count_total_est = 0
        self._current_exp_desc = ""
        self._last_step_time = datetime.now()

        # ---- 自主研究（自动选股 + 自动进化）状态 ----
        self._auto_scheduler = None
        self._auto_thread = None
        self._auto_guard = None
        self._auto_registry = None
        self._auto_running = False
        self._auto_last_state = {}
        self._auto_ui_tick_running = False
        self._close_ticks = 0
        self._btn_style_seq = 0

        self._apply_theme()
        self.build_ui()
        self._build_menu()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self.poll_queue)
        self.root.after(100, self._flush_log_buffer)
        self.refresh_hist_ai_combo()
        self.refresh_history()
        self._auto_ui_tick_running = True
        self.root.after(1500, self._auto_status_tick)

    # ================================================================
    # 主题 / 控件工厂
    # ================================================================
    def _apply_theme(self):
        try:
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure(".", background=UI_BG, foreground=UI_FG,
                            fieldbackground=UI_INPUT, bordercolor=UI_BORDER,
                            lightcolor=UI_PANEL_2, darkcolor=UI_PANEL,
                            troughcolor=UI_INPUT, focuscolor=UI_ACCENT)
            style.configure("TFrame", background=UI_BG)
            style.configure("TLabel", background=UI_BG, foreground=UI_FG)
            style.configure("TNotebook", background=UI_BG, borderwidth=0, tabmargins=(6, 6, 0, 0))
            style.configure("TNotebook.Tab", background=UI_PANEL, foreground=UI_FG_DIM,
                            padding=(22, 10), font=("Microsoft YaHei", 11, "bold"))
            style.map("TNotebook.Tab",
                      background=[("selected", UI_PANEL_2)],
                      foreground=[("selected", UI_ACCENT)])
            style.configure("TCombobox", fieldbackground=UI_INPUT, background=UI_PANEL_2,
                            foreground=UI_FG, arrowcolor=UI_FG,
                            selectbackground=UI_INPUT, selectforeground=UI_FG)
            style.map("TCombobox",
                      fieldbackground=[("readonly", UI_INPUT)],
                      foreground=[("readonly", UI_FG)],
                      background=[("readonly", UI_PANEL_2)])
            style.configure("Vertical.TScrollbar", background=UI_BTN, troughcolor=UI_BG,
                            bordercolor=UI_BG, arrowcolor=UI_FG_DIM, relief="flat")
            style.configure("Horizontal.TScrollbar", background=UI_BTN, troughcolor=UI_BG,
                            bordercolor=UI_BG, arrowcolor=UI_FG_DIM, relief="flat")
            style.configure("Dark.Treeview", background=UI_PANEL, fieldbackground=UI_PANEL,
                            foreground=UI_FG, rowheight=26, bordercolor=UI_BORDER)
            style.configure("Dark.Treeview.Heading", background=UI_PANEL_2,
                            foreground=UI_FG, relief="flat",
                            font=("Microsoft YaHei", 10, "bold"))
            style.map("Dark.Treeview",
                      background=[("selected", UI_ACCENT)],
                      foreground=[("selected", "#ffffff")])
            style.map("Dark.Treeview.Heading", background=[("active", UI_BORDER)])
            style.configure("Dark.Horizontal.TProgressbar", troughcolor=UI_INPUT,
                            background=UI_ACCENT, bordercolor=UI_BG,
                            lightcolor=UI_ACCENT, darkcolor=UI_ACCENT)
            self.root.option_add("*TCombobox*Listbox.background", UI_INPUT)
            self.root.option_add("*TCombobox*Listbox.foreground", UI_FG)
            self.root.option_add("*TCombobox*Listbox.selectBackground", UI_ACCENT)
            self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        except Exception:
            pass

    def _lf(self, parent, text, accent=UI_ACCENT, **kw):
        return tk.LabelFrame(parent, text=text, font=FONT_T, fg=accent, bg=UI_PANEL,
                             bd=1, relief="groove", labelanchor="nw",
                             highlightthickness=0, padx=6, pady=4, **kw)

    def _lbl(self, parent, text, fg=UI_FG, font=FONT_N, bg=UI_PANEL, **kw):
        return tk.Label(parent, text=text, fg=fg, bg=bg, font=font, **kw)

    def _ent(self, parent, width=12, font=FONT_N, bg=UI_INPUT, **kw):
        return tk.Entry(parent, width=width, font=font, bg=bg, fg=UI_FG,
                        insertbackground=UI_FG, relief="flat",
                        highlightthickness=1, highlightbackground=UI_BORDER,
                        highlightcolor=UI_ACCENT, disabledbackground=UI_PANEL_2,
                        disabledforeground=UI_FG_DIM, **kw)

    def _btn(self, parent, text, command, bg=UI_BTN, fg="#ffffff", font=FONT_B,
             width=None, padx=10, pady=5, **kw):
        if sys.platform == "darwin":
            # macOS Tk 8.5 的原生 Button 会忽略 bg，显示成灰色系统按钮；
            # 用 clam 主题的 ttk.Button 恢复深色按钮。Windows 仍用原 tk.Button。
            self._btn_style_seq += 1
            style_name = f"DarkButton{self._btn_style_seq}.TButton"
            style = ttk.Style()
            style.configure(style_name, background=bg, foreground=fg, font=font,
                            borderwidth=0, focusthickness=0, relief="flat",
                            padding=(padx, pady))
            style.map(style_name,
                      background=[("active", UI_BTN_HOVER), ("pressed", UI_BTN_HOVER),
                                  ("disabled", UI_PANEL)],
                      foreground=[("disabled", UI_FG_DIM)])
            b = ttk.Button(parent, text=text, command=command, style=style_name,
                           cursor="hand2", **kw)
        else:
            b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, font=font,
                          activebackground=UI_BTN_HOVER, activeforeground="#ffffff",
                          relief="flat", bd=0, cursor="hand2", padx=padx, pady=pady, **kw)
        if width:
            b.config(width=width)
        return b

    def _txt(self, parent, height=5, font=None, bg=UI_INPUT, **kw):
        return tk.Text(parent, height=height, font=font or FONT_LOG, bg=bg, fg=UI_FG,
                       insertbackground=UI_FG, relief="flat", wrap="word",
                       highlightthickness=1, highlightbackground=UI_BORDER,
                       highlightcolor=UI_ACCENT, **kw)

    # ================================================================
    # 搭界面
    # ================================================================
    def build_ui(self):
        # ---- 顶部标题条 ----
        top = tk.Frame(self.root, bg=UI_PANEL_2)
        top.pack(side="top", fill="x")
        self._lbl(top, "  XGBoost 量化模型 · 一键自动训练",
                  fg=UI_TITLE, font=("Microsoft YaHei", 14, "bold"),
                  bg=UI_PANEL_2).pack(side="left", padx=10, pady=8)
        self._lbl(top, "输入股票代码与训练/测试日期，点一下按钮即可全自动完成研究",
                  fg=UI_FG_DIM, font=FONT_S, bg=UI_PANEL_2).pack(side="left", padx=6)

        # ---- 顶层双页面：① 一键自动训练  ② 历史记录与运行日志 ----
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="top", fill="both", expand=True, padx=10, pady=(6, 10))
        self.tab_train = tk.Frame(self.notebook, bg=UI_BG)
        self.tab_data = tk.Frame(self.notebook, bg=UI_BG)
        self.notebook.add(self.tab_train, text="   🚀 一键自动训练   ")
        self.notebook.add(self.tab_data, text="   📜 历史记录与运行日志   ")

        # ---- 页面一：训练控制区 ----
        # macOS Tk 8.5 的 Canvas 内嵌窗口在宽度同步时会触发持续重绘（高 CPU），
        # 因此 macOS 使用普通 Frame 布局；Windows 保留原来的可滚动 Canvas。
        cwrap = tk.Frame(self.tab_train, bg=UI_BG)
        cwrap.pack(side="top", fill="both", expand=True)
        self._use_scroll_canvas = (sys.platform != "darwin")
        if self._use_scroll_canvas:
            self.ctrl_canvas = tk.Canvas(cwrap, bg=UI_BG, highlightthickness=0, height=300)
            cvsb = ttk.Scrollbar(cwrap, orient="vertical", command=self.ctrl_canvas.yview)
            self.ctrl_canvas.configure(yscrollcommand=cvsb.set)
            cvsb.pack(side="right", fill="y")
            self.ctrl_canvas.pack(side="left", fill="both", expand=True)
            self._ctrl_inner = tk.Frame(self.ctrl_canvas, bg=UI_BG)
            self._ctrl_win = self.ctrl_canvas.create_window((0, 0), window=self._ctrl_inner,
                                                            anchor="nw")
            self._ctrl_inner.bind(
                "<Configure>",
                lambda e: self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all")))
            self._ctrl_width = None
            self._ctrl_width_after = None
            self.ctrl_canvas.bind("<Configure>", self._schedule_ctrl_width)
            self.ctrl_canvas.bind_all("<MouseWheel>", self._on_ctrl_wheel)
        else:
            self.ctrl_canvas = tk.Frame(cwrap, bg=UI_BG)
            self.ctrl_canvas.pack(side="left", fill="both", expand=True)
            self._ctrl_inner = tk.Frame(self.ctrl_canvas, bg=UI_BG)
            self._ctrl_inner.pack(fill="both", expand=True)
            self._ctrl_win = None
            self._ctrl_width = None
            self._ctrl_width_after = None

        # 两列：左=输入，右=状态/进度
        cols = tk.Frame(self._ctrl_inner, bg=UI_BG)
        cols.pack(fill="x", padx=12, pady=(10, 2))
        cols.columnconfigure(0, weight=3)
        cols.columnconfigure(1, weight=2)
        left = tk.Frame(cols, bg=UI_BG)
        left.grid(row=0, column=0, sticky="nsew")
        right = tk.Frame(cols, bg=UI_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        # ---- 左：输入 ----
        inp = self._lf(left, " 输入（只填这些，其余全部自动） ", accent=UI_ACCENT)
        inp.pack(fill="x")
        grid = tk.Frame(inp, bg=UI_PANEL)
        grid.pack(fill="x", padx=10, pady=10)

        self.e_code = self._ent(grid, width=12)
        self.e_ts = self._ent(grid, width=12)
        self.e_te = self._ent(grid, width=12)
        self.e_ps = self._ent(grid, width=12)
        self.e_pe = self._ent(grid, width=12)
        fields = [("股票代码", self.e_code, DEF_CODE),
                  ("训练开始", self.e_ts, DEF_TRAIN_START),
                  ("训练结束", self.e_te, DEF_TRAIN_END),
                  ("测试开始", self.e_ps, DEF_TEST_START),
                  ("测试结束", self.e_pe, DEF_TEST_END)]
        for i, (label, ent, dv) in enumerate(fields):
            self._lbl(grid, label, fg=UI_FG_DIM, font=FONT_S).grid(
                row=0, column=i, sticky="w", padx=(0, 10), pady=(0, 2))
            ent.grid(row=1, column=i, sticky="w", padx=(0, 10))
            try:
                ent.insert(0, dv)
            except Exception:
                pass
        self.train_btn = self._btn(
            grid, "🚀 开始进化训练", self._one_click_train,
            bg="#1f7a1f", font=("Microsoft YaHei", 14, "bold"), padx=16, pady=10)
        self.pause_btn = self._btn(
            grid, "⏸ 暂停训练", self._pause_train,
            bg=UI_WARN, font=("Microsoft YaHei", 12, "bold"), padx=12, pady=8, state="disabled")
        self.resume_btn = self._btn(
            grid, "▶ 继续训练", self._resume_train,
            bg=UI_OK, font=("Microsoft YaHei", 12, "bold"), padx=12, pady=8, state="disabled")
        self.stop_btn = self._btn(
            grid, "⏹ 结束训练", self._stop_train,
            bg=UI_ERR, font=("Microsoft YaHei", 12, "bold"), padx=12, pady=8, state="disabled")
        self.train_btn.grid(row=2, column=0, sticky="ew", pady=(12, 0), padx=(0,4))
        self.pause_btn.grid(row=2, column=1, sticky="ew", pady=(12,0), padx=2)
        self.resume_btn.grid(row=2, column=2, sticky="ew", pady=(12,0), padx=2)
        self.stop_btn.grid(row=2, column=3, sticky="ew", pady=(12,0), padx=(2,10))

        # ---- 自主研究：不需要手动填股票代码，自动发现股票并持续研究 ----
        self.auto_btn = self._btn(
            grid, "🔭 启动自主研究（自动选股·自动进化）", self._start_autonomous,
            bg="#1f6feb", font=("Microsoft YaHei", 12, "bold"), padx=12, pady=8)
        self.auto_btn.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=(0, 4))
        self._lbl(grid, "启动后：自动获取股票列表→检查行情→排序研究队列→自动训练/验证/晋级；"
                        "随时可暂停/继续/停止，重启后可恢复。",
                  fg=UI_FG_DIM, font=FONT_S, bg=UI_PANEL, justify="left", wraplength=420).grid(
            row=3, column=2, columnspan=3, sticky="w", padx=(6, 0), pady=(8, 0))

        # ---- 右：状态 / 进度 ----
        st = self._lf(right, " 训练控制中心（当前步骤透明） ", accent=UI_WARN)
        st.pack(fill="x")
        self.stage_lbl = self._lbl(st, "当前阶段：空闲", fg=UI_ACCENT,
                                   font=("Microsoft YaHei", 11, "bold"))
        self.stage_lbl.pack(anchor="w", padx=10, pady=(8, 2))
        self.current_task_lbl = self._lbl(st, "当前具体任务：等待开始", fg=UI_FG, font=FONT_N, wraplength=380)
        self.current_task_lbl.pack(anchor="w", padx=10, pady=2)
        self.exp_num_lbl = self._lbl(st, "当前实验编号：-  | 已完成实验：0  | 剩余（动态）：未确定", fg=UI_FG_DIM, font=FONT_S)
        self.exp_num_lbl.pack(anchor="w", padx=10, pady=2)
        self.time_range_lbl = self._lbl(st, "训练/验证时间范围：-  | 最终测试集：受保护数据（仅最后评估一次）", fg=UI_FG_DIM, font=FONT_S, wraplength=380)
        self.time_range_lbl.pack(anchor="w", padx=10, pady=2)
        self.why_lbl = self._lbl(st, "执行原因：-", fg=UI_FG_DIM, font=FONT_S, wraplength=380)
        self.why_lbl.pack(anchor="w", padx=10, pady=2)
        self.best_cand_lbl = self._lbl(st, "当前最佳候选：-  | 与基准对比：-", fg=UI_FG_DIM, font=FONT_S, wraplength=380)
        self.best_cand_lbl.pack(anchor="w", padx=10, pady=2)
        self.next_lbl = self._lbl(st, "下一步计划：-", fg=UI_FG_DIM, font=FONT_S, wraplength=380)
        self.next_lbl.pack(anchor="w", padx=10, pady=2)
        self.runtime_lbl = self._lbl(st, "运行时长：00:00  | 最近更新：刚才  | 内存：未获取", fg=UI_FG_DIM, font=FONT_S)
        self.runtime_lbl.pack(anchor="w", padx=10, pady=2)
        self.progress = ttk.Progressbar(st, style="Dark.Horizontal.TProgressbar",
                                        mode="determinate", length=380)
        self.progress.pack(anchor="w", padx=10, pady=2)
        self.status_lbl = self._lbl(st, "就绪，等待开始", fg=UI_FG_DIM, font=FONT_N)
        self.status_lbl.pack(anchor="w", padx=10, pady=(2, 8))

        # ---- 自主研究状态（股票池 / 队列 / 实验计数 / 数据源 / 资源）----
        self.auto_pool_lbl = self._lbl(
            st, "股票池：未建立  | 待研究：-  | 已完成/失败实验：0 / 0  | 数据源：未更新",
            fg=UI_FG, font=FONT_S, wraplength=380)
        self.auto_pool_lbl.pack(anchor="w", padx=10, pady=(0, 2))
        self.auto_run_lbl = self._lbl(
            st, "自主研究：未启动  | 当前股票：-  | 当前目标：-  | 暂停/异常原因：-",
            fg=UI_FG, font=FONT_S, wraplength=380)
        self.auto_run_lbl.pack(anchor="w", padx=10, pady=(0, 2))
        self.auto_best_lbl = self._lbl(
            st, "当前最佳验证版本：-  | 最近一次保存：-",
            fg=UI_FG_DIM, font=FONT_S, wraplength=380)
        self.auto_best_lbl.pack(anchor="w", padx=10, pady=(0, 2))
        self.auto_res_lbl = self._lbl(st, "资源：未采样", fg=UI_FG_DIM, font=FONT_S,
                                      wraplength=380)
        self.auto_res_lbl.pack(anchor="w", padx=10, pady=(0, 8))

        # ---- 结果摘要（横跨两列）----
        res = self._lf(self._ctrl_inner,
                       " 最终结果摘要（详细报告见实验目录 report.md） ", accent=UI_OK)
        res.pack(fill="x", padx=12, pady=(2, 6))
        # macOS Tk 8.5：禁用态 Text + 自动换行会持续重绘并占满 CPU，
        # 这里用等价的只读 Label 展示结果摘要。
        self.result_txt = self._lbl(
            res, "尚未运行。运行结束后这里显示：实验编号 / 最佳特征 / 最佳策略 / "
                 "训练期稳定性 / 独立测试结果。",
            fg=UI_FG, font=("Microsoft YaHei", 10), bg=UI_PANEL,
            justify="left", anchor="w", wraplength=1240)
        self.result_txt.pack(fill="x", padx=8, pady=6)

        # ---- 版本与重启（恢复旧版【一键重启】按钮与其安全回调）----
        ver = self._lf(self._ctrl_inner, " 版本与重启（改了代码后点一下即可） ", accent="#c58af9")
        ver.pack(fill="x", padx=12, pady=(2, 12))
        ver.grid_columnconfigure(0, weight=1)
        self._lbl(ver, "重启会先做语法检查、自动备份当前代码，再启动新进程并安全退出旧进程；"
                       "若任务正在运行，会先请求安全停止（已完成的实验与日志尽量保存）；"
                       "全程不删除任何历史记录 / 模型 / 资金曲线。",
                  fg=UI_FG_DIM, font=FONT_S, justify="left", wraplength=820).grid(
            row=0, column=0, sticky="w", padx=10, pady=(4, 2))
        self.backup_info_lbl = self._lbl(ver, "最近备份：-", fg=UI_FG_DIM, font=FONT_S)
        self.backup_info_lbl.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))
        self.restart_btn = self._btn(ver, "🔁 一键重启", self._one_click_restart,
                                     bg="#5D4037", font=("Microsoft YaHei", 12, "bold"),
                                     padx=18, pady=8)
        self.restart_btn.grid(row=0, column=1, rowspan=2, padx=12, pady=8, sticky="e")
        self._refresh_backup_info()

        # ---- 页面二：历史记录 + 运行日志（分区，互不挤占训练页）----
        self._build_data_page(self.tab_data)

        # 启动时恢复上次输入（若曾保存）
        self._load_inputs()

        # 首次布局完成后同步一次控制区宽度/高度；滚轮仅在指针位于控制区时接管
        if self._use_scroll_canvas:
            self.root.after(300, self._apply_ctrl_width)
            self.root.after(250, self._fit_ctrl_canvas)
        self.root.bind_all("<MouseWheel>", self._on_global_wheel)

    def _build_menu(self):
        """跨平台菜单：Windows 显示在窗口内，macOS 显示在系统菜单栏。"""
        try:
            menubar = tk.Menu(self.root)
            file_menu = tk.Menu(menubar, tearoff=0)
            file_menu.add_command(label="退出", command=self._on_close)
            menubar.add_cascade(label="文件", menu=file_menu)

            run_menu = tk.Menu(menubar, tearoff=0)
            run_menu.add_command(label="一键重启", command=self._one_click_restart)
            run_menu.add_command(label="刷新历史", command=self.refresh_history)
            menubar.add_cascade(label="运行", menu=run_menu)

            help_menu = tk.Menu(menubar, tearoff=0)
            help_menu.add_command(label="关于", command=self._show_about)
            menubar.add_cascade(label="帮助", menu=help_menu)
            self.root.config(menu=menubar)
            self.menubar = menubar
        except Exception:
            self.menubar = None

    def _show_about(self):
        try:
            messagebox.showinfo(
                "关于",
                "XGBoost 量化模型 · 一键自动训练\n\n"
                "训练、验证与最终测试流程保持原有设计；\n"
                "本启动器仅增加 macOS 兼容性修复。")
        except Exception:
            pass

    def _build_data_page(self, parent):
        """页面二：历史记录与运行日志（用子标签页合理分区，训练页不受挤占）。"""
        inner = ttk.Notebook(parent)
        inner.pack(fill="both", expand=True, padx=6, pady=6)
        self.data_notebook = inner
        self.tab_hist = tk.Frame(inner, bg=UI_BG)
        self.tab_log = tk.Frame(inner, bg=UI_BG)
        self.tab_events = tk.Frame(inner, bg=UI_BG)
        self.tab_test_windows = tk.Frame(inner, bg=UI_BG)
        self.tab_reset_failed = tk.Frame(inner, bg=UI_BG)
        inner.add(self.tab_hist, text="   历史记录   ")
        inner.add(self.tab_log, text="   运行日志   ")
        inner.add(self.tab_events, text="   实时事件日志   ")
        inner.add(self.tab_test_windows, text="   固定测试窗口   ")
        inner.add(self.tab_reset_failed, text="   失败状态重置   ")
        self._build_history_tab(self.tab_hist)
        self._build_log_tab(self.tab_log)
        self._build_event_tab(self.tab_events)
        self._build_test_windows_tab(self.tab_test_windows)
        self._build_reset_failed_tab(self.tab_reset_failed)

    def _schedule_ctrl_width(self, event=None):
        """把 Canvas 宽度同步延迟到空闲时，避免在 Configure 回调里同步重排。"""
        if not getattr(self, "_use_scroll_canvas", False):
            return
        try:
            if self._ctrl_width_after is not None:
                self.root.after_cancel(self._ctrl_width_after)
            self._ctrl_width_after = self.root.after(40, self._apply_ctrl_width)
        except Exception:
            self._ctrl_width_after = None

    def _apply_ctrl_width(self):
        if not getattr(self, "_use_scroll_canvas", False):
            return
        try:
            self._ctrl_width_after = None
            width = int(self.ctrl_canvas.winfo_width())
            if width <= 1 or width == self._ctrl_width:
                return
            self._ctrl_width = width
            self.ctrl_canvas.itemconfigure(self._ctrl_win, width=width)
        except Exception:
            pass

    def _fit_ctrl_canvas(self):
        """让控制区高度贴合内容，但不超过窗口的 45%，多余部分靠滚动查看。"""
        if not getattr(self, "_use_scroll_canvas", False):
            return
        try:
            need = self._ctrl_inner.winfo_reqheight()
            win_h = self.root.winfo_height()
            cap = max(220, int(win_h * 0.45)) if win_h > 1 else 360
            target = min(max(need, 160), cap)
            if int(float(self.ctrl_canvas.cget("height"))) != int(target):
                self.ctrl_canvas.configure(height=int(target))
            self._apply_ctrl_width()
        except Exception:
            pass

    def _is_descendant(self, widget, ancestor):
        w = widget
        while w is not None:
            if w == ancestor:
                return True
            w = getattr(w, "master", None)
        return False

    def _on_global_wheel(self, event):
        """鼠标滚轮：指针在控制区时滚动控制区，其余控件（日志/历史）保持各自原生滚动。"""
        try:
            w = self.root.winfo_containing(event.x_root, event.y_root)
        except Exception:
            w = None
        if (w is not None and self._is_descendant(w, self.ctrl_canvas)
                and hasattr(self.ctrl_canvas, "yview_scroll")):
            try:
                self.ctrl_canvas.yview_scroll(int(-event.delta / 120), "units")
            except Exception:
                pass

    def _on_ctrl_wheel(self, event):
        if not hasattr(self.ctrl_canvas, "yview_scroll"):
            return
        try:
            self.ctrl_canvas.yview_scroll(int(-event.delta / 120), "units")
        except Exception:
            pass

    def _update_runtime(self):
        try:
            delta = datetime.now() - self._last_step_time
            secs = int(delta.total_seconds())
            h = secs // 3600
            m = (secs % 3600) // 60
            s = secs % 60
            self.runtime_lbl.config(text=f"运行时长：{h:02d}:{m:02d}:{s:02d}  | 最近更新：{delta.total_seconds():.0f}s前  | 内存：正常")
        except Exception:
            pass

    def _auto_loop_status(self):
        """读注册表里的自主研究循环状态（失败时返回空串）。"""
        try:
            reg = self._auto_registry
            if reg is None:
                return ""
            loop = reg.get_state("loop", {}) or {}
            return str(loop.get("status") or "")
        except Exception:
            return ""

    def _pause_train(self):
        try:
            # 自主研究：暂停 = 停止调度新实验，等当前实验在可控边界结束后保存（不丢进度）
            if self._auto_scheduler is not None and (self._auto_running or
                    self._auto_loop_status() == "running"):
                self._auto_scheduler.request_pause("用户暂停")
                self.pause_btn.config(state="disabled")
                self.resume_btn.config(state="normal")
                self.stop_btn.config(state="normal")
                self._set_status("自主研究：暂停请求已发出，等待当前实验安全结束并保存 ...")
                self._log_event("暂停自主研究：等待当前实验安全结束", "yellow")
                return
            if not self.working:
                messagebox.showinfo("提示", "当前没有运行中的训练")
                return
            self._paused = True
            self.stop_ev.set()
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self._set_status("暂停请求已发出，等待当前步骤安全结束...")
            self._log_event("暂停训练：等待当前步骤安全结束", "yellow")
        except Exception:
            pass

    def _resume_train(self):
        try:
            sched = self._auto_scheduler
            if sched is not None:
                status = self._auto_loop_status()
                # 情况一：线程仍在（用户暂停，阻塞在暂停等待里）→ 直接解除暂停即可继续
                if self._auto_running:
                    sched.request_resume()
                    self.pause_btn.config(state="normal")
                    self.resume_btn.config(state="disabled")
                    self.stop_btn.config(state="normal")
                    self._set_status("自主研究：已恢复，从最后一个有效检查点继续")
                    self._log_event("恢复自主研究", "green")
                    return
                # 情况二：线程已结束（错误暂停）→ 用同一个调度器重新拉起，不重复已完成实验
                if status in ("paused", "error_paused"):
                    sched.request_resume()
                    self._auto_running = True
                    self._auto_thread = threading.Thread(
                        target=self._autonomous_worker, args=(sched,), daemon=True)
                    self._auto_thread.start()
                    self._set_auto_buttons(True)
                    self._set_status("自主研究：已从检查点恢复")
                    self._log_event("从错误暂停恢复自主研究", "green")
                    return
            messagebox.showinfo(
                "提示",
                "继续训练：当前没有可恢复的自主研究。\n\n"
                "（自主研究被暂停/错误暂停后，点此从检查点继续；"
                "手动训练请重新点击【一键自动训练】）")
        except Exception:
            pass

    def _stop_train(self):
        try:
            sched = self._auto_scheduler
            if sched is not None:
                status = self._auto_loop_status()
                if self._auto_running or status in ("running", "paused", "error_paused", "idle"):
                    # 停止 ≠ 暂停：请求彻底结束本次自主研究会话（保存已完成的记录）
                    sched.request_stop("用户停止")
                    self._set_status("自主研究：结束请求已发出，正在安全停止 ...")
                    self._log_event("结束自主研究：请求安全停止", "red")
                    return
            if not self.working and not self._running:
                messagebox.showinfo("提示", "当前没有运行中的训练")
                return
            self.stop_ev.set()
            self._set_status("结束请求已发出，正在安全停止...")
            self._log_event("结束训练：请求安全停止", "red")
        except Exception:
            pass

    def _build_log_tab(self, parent):
        bar = tk.Frame(parent, bg=UI_BG)
        bar.pack(side="top", fill="x", padx=10, pady=(8, 0))
        self._btn(bar, "清空显示", self._clear_log).pack(side="left", padx=(0, 6))
        self._lbl(bar, "（只清空界面显示，不删除任何历史记录 / 模型 / 资金曲线）",
                  fg=UI_FG_DIM, font=FONT_S, bg=UI_BG).pack(side="left")
        wrap = tk.Frame(parent, bg=UI_BG)
        wrap.pack(side="top", fill="both", expand=True, padx=10, pady=6)
        if sys.platform == "darwin":
            self.log = _MacLogView(wrap, font=FONT_LOG, bg="#11151c", fg=UI_FG)
        else:
            self.log = tk.Text(wrap, font=FONT_LOG, bg="#11151c", fg=UI_FG, wrap="word",
                               insertbackground=UI_FG, relief="flat")
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("red", foreground=UI_ERR)
        self.log.tag_configure("green", foreground=UI_OK)
        self.log.tag_configure("title", foreground=UI_TITLE)
        self.log.tag_configure("yellow", foreground=UI_WARN)
        self.log.tag_configure("dim", foreground=UI_FG_DIM)

    def _build_event_tab(self, parent):
        bar = tk.Frame(parent, bg=UI_BG)
        bar.pack(side="top", fill="x", padx=10, pady=(8,0))
        self._btn(bar, "清空事件日志", self._clear_events).pack(side="left", padx=(0,6))
        self._lbl(bar, "（只清空界面显示，不影响训练进度）", fg=UI_FG_DIM, font=FONT_S, bg=UI_BG).pack(side="left")
        wrap = tk.Frame(parent, bg=UI_BG)
        wrap.pack(side="top", fill="both", expand=True, padx=10, pady=6)
        if sys.platform == "darwin":
            self.events_log = _MacLogView(wrap, font=FONT_LOG, bg="#0f1118", fg=UI_FG)
        else:
            self.events_log = tk.Text(wrap, font=FONT_LOG, bg="#0f1118", fg=UI_FG,
                                      insertbackground=UI_FG, wrap="word", relief="flat")
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.events_log.yview)
        self.events_log.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.events_log.pack(side="left", fill="both", expand=True)
        self.events_log.tag_configure("red", foreground=UI_ERR)
        self.events_log.tag_configure("green", foreground=UI_OK)
        self.events_log.tag_configure("yellow", foreground=UI_WARN)
        self.events_log.tag_configure("dim", foreground=UI_FG_DIM)

    def _build_test_windows_tab(self, parent):
        """固定测试窗口管理标签页"""
        # 工具栏
        toolbar = tk.Frame(parent, bg=UI_PANEL_2)
        toolbar.pack(side="top", fill="x", padx=6, pady=6)

        self._lbl(toolbar, "股票代码：", bg=UI_BG).pack(side="left", padx=(0, 2))
        self.test_win_code_var = tk.StringVar(value="")
        self.test_win_code_ent = tk.Entry(toolbar, textvariable=self.test_win_code_var,
                                         width=10, bg=UI_INPUT, fg=UI_FG,
                                         insertbackground=UI_FG, relief="flat",
                                         highlightthickness=1, highlightbackground=UI_BORDER,
                                         highlightcolor=UI_ACCENT)
        self.test_win_code_ent.pack(side="left", padx=(0, 6))
        self._btn(toolbar, "刷新", self.refresh_test_windows).pack(side="left", padx=4)

        # 窗口列表
        tree_frame = tk.Frame(parent, bg=UI_BG)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.test_windows_tree = ttk.Treeview(
            tree_frame, columns=["id", "code", "train_start", "train_end",
                                "test_start", "test_end", "reserved"],
            show="headings", height=20, style="Dark.Treeview"
        )
        self.test_windows_tree.heading("id", text="ID")
        self.test_windows_tree.column("id", width=50, anchor="center", stretch=False)
        self.test_windows_tree.heading("code", text="股票代码")
        self.test_windows_tree.column("code", width=80, anchor="center", stretch=False)
        self.test_windows_tree.heading("train_start", text="训练开始")
        self.test_windows_tree.column("train_start", width=100, anchor="center", stretch=False)
        self.test_windows_tree.heading("train_end", text="训练结束")
        self.test_windows_tree.column("train_end", width=100, anchor="center", stretch=False)
        self.test_windows_tree.heading("test_start", text="测试开始")
        self.test_windows_tree.column("test_start", width=100, anchor="center", stretch=False)
        self.test_windows_tree.heading("test_end", text="测试结束")
        self.test_windows_tree.column("test_end", width=100, anchor="center", stretch=False)
        self.test_windows_tree.heading("reserved", text="保留")
        self.test_windows_tree.column("reserved", width=60, anchor="center", stretch=False)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.test_windows_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.test_windows_tree.xview)
        self.test_windows_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.test_windows_tree.pack(side="left", fill="both", expand=True)

        self.test_windows_tree.bind("<<TreeviewSelect>>", self._on_test_window_select)

        # 按钮区
        btn_frame = tk.Frame(parent, bg=UI_PANEL_2)
        btn_frame.pack(side="top", fill="x", padx=6, pady=6)

        self.view_detail_btn = self._btn(btn_frame, "查看详情",
                                         self._on_test_window_detail, bg=UI_ACCENT)
        self.view_detail_btn.pack(side="left", padx=(0, 4))

        self.unreserve_btn = self._btn(btn_frame, "取消保留",
                                        self._on_unreserve_window, bg=UI_WARN)
        self.unreserve_btn.pack(side="left", padx=4)

        self.batch_unreserve_btn = self._btn(btn_frame, "批量取消",
                                              self._on_batch_unreserve, bg=UI_ERR)
        self.batch_unreserve_btn.pack(side="left", padx=4)

        # 详情标签
        det = self._lf(parent, "  选中窗口详情 ", accent=UI_ACCENT)
        det.pack(side="bottom", fill="x", padx=10, pady=(4, 2))
        self.test_win_detail_txt = self._lbl(
            det, "点选上方任意一行，这里显示该窗口详情。",
            fg=UI_FG, font=("Microsoft YaHei", 10), bg=UI_PANEL,
            justify="left", anchor="w", wraplength=1240)
        self.test_win_detail_txt.pack(fill="x", padx=8, pady=6)

        # 刷新数据
        self.refresh_test_windows()

    def _build_reset_failed_tab(self, parent):
        """失败状态重置标签页"""
        # 工具栏
        toolbar = tk.Frame(parent, bg=UI_PANEL_2)
        toolbar.pack(side="top", fill="x", padx=6, pady=6)

        self._lbl(toolbar, "选择AI：", bg=UI_BG).pack(side="left", padx=(0, 2))
        self.reset_ai_var = tk.StringVar(value=self.current_ai())
        self.reset_ai_combo = ttk.Combobox(
            toolbar, textvariable=self.reset_ai_var, values=scan_all_ai_names(),
            state="readonly", width=16, font=("Microsoft YaHei", 11))
        self.reset_ai_combo.pack(side="left", padx=(0, 14))
        self.reset_ai_combo.bind("<<ComboboxSelected>>", self.on_reset_ai_changed)
        self._btn(toolbar, "刷新", self.refresh_failed_experiments).pack(side="left", padx=4)

        # 失败实验列表
        tree_frame = tk.Frame(parent, bg=UI_BG)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.failed_tree = ttk.Treeview(
            tree_frame, columns=["id", "code", "train_start", "train_end",
                                "test_start", "test_end", "status", "failure_reason"],
            show="headings", height=20, style="Dark.Treeview"
        )
        self.failed_tree.heading("id", text="ID")
        self.failed_tree.column("id", width=50, anchor="center", stretch=False)
        self.failed_tree.heading("code", text="股票代码")
        self.failed_tree.column("code", width=80, anchor="center", stretch=False)
        self.failed_tree.heading("train_start", text="训练开始")
        self.failed_tree.column("train_start", width=100, anchor="center", stretch=False)
        self.failed_tree.heading("train_end", text="训练结束")
        self.failed_tree.column("train_end", width=100, anchor="center", stretch=False)
        self.failed_tree.heading("test_start", text="测试开始")
        self.failed_tree.column("test_start", width=100, anchor="center", stretch=False)
        self.failed_tree.heading("test_end", text="测试结束")
        self.failed_tree.column("test_end", width=100, anchor="center", stretch=False)
        self.failed_tree.heading("status", text="状态")
        self.failed_tree.column("status", width=80, anchor="center", stretch=False)
        self.failed_tree.heading("failure_reason", text="失败原因")
        self.failed_tree.column("failure_reason", width=200, anchor="w", stretch=False)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.failed_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.failed_tree.xview)
        self.failed_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.failed_tree.pack(side="left", fill="both", expand=True)

        self.failed_tree.bind("<<TreeviewSelect>>", self._on_failed_experiment_select)

        # 按钮区
        btn_frame = tk.Frame(parent, bg=UI_PANEL_2)
        btn_frame.pack(side="top", fill="x", padx=6, pady=6)

        self.reset_btn = self._btn(btn_frame, "重置选中实验",
                                   self._on_reset_failed_experiment, bg=UI_OK)
        self.reset_btn.pack(side="left", padx=(0, 4))

        self.batch_reset_btn = self._btn(btn_frame, "批量重置",
                                         self._on_batch_reset_failed, bg=UI_WARN)
        self.batch_reset_btn.pack(side="left", padx=4)

        # 详情标签
        det = self._lf(parent, "  选中实验详情 ", accent=UI_ACCENT)
        det.pack(side="bottom", fill="x", padx=10, pady=(4, 2))
        self.reset_detail_txt = self._lbl(
            det, "点选上方任意一行，这里显示该实验详情。",
            fg=UI_FG, font=("Microsoft YaHei", 10), bg=UI_PANEL,
            justify="left", anchor="w", wraplength=1240)
        self.reset_detail_txt.pack(fill="x", padx=8, pady=6)

        # 刷新数据
        self.refresh_failed_experiments()

    def refresh_test_windows(self):
        """刷新固定测试窗口列表"""
        try:
            reg = self._registry()
            if reg is None:
                self.write("注册表打开失败，无法刷新固定测试窗口", "red")
                return

            # 清空现有数据
            for item in self.test_windows_tree.get_children():
                self.test_windows_tree.delete(item)

            # 从 SQLite 读取已保留的测试窗口
            if hasattr(reg, 'conn'):
                cursor = reg.conn.cursor()
                cursor.execute("""
                    SELECT id, code, train_start, train_end, test_start, test_end, reserved
                    FROM test_windows
                    WHERE reserved = 1
                    ORDER BY id DESC
                """)
                rows = cursor.fetchall()
                for row in rows:
                    values = list(row) + ["是" if row[6] == 1 else "否"]
                    self.test_windows_tree.insert("", tk.END, values=values)
            else:
                self.write("无法访问 SQLite 连接", "red")

            self._log_event(f"刷新固定测试窗口：共 {len(rows)} 条", "dim")
        except Exception as e:
            self.write(f"刷新固定测试窗口失败：{e}", "red")
            self._log_event(f"刷新固定测试窗口失败：{e}", "red")

    def refresh_failed_experiments(self):
        """刷新失败实验列表"""
        try:
            ai = self.reset_ai_var.get().strip()
            if not ai:
                ai = self.current_ai()

            reg = self._registry()
            if reg is None:
                self.write("注册表打开失败，无法刷新失败实验", "red")
                return

            # 清空现有数据
            for item in self.failed_tree.get_children():
                self.failed_tree.delete(item)

            # 从 SQLite 读取失败的实验
            if hasattr(reg, 'conn'):
                cursor = reg.conn.cursor()
                cursor.execute("""
                    SELECT id, code, train_start, train_end, test_start, test_end, status, failure_reason
                    FROM test_usage
                    WHERE status = 'failed' AND ai_name = ?
                    ORDER BY id DESC
                """, (ai,))
                rows = cursor.fetchall()
                for row in rows:
                    values = list(row)
                    self.failed_tree.insert("", tk.END, values=values)
            else:
                self.write("无法访问 SQLite 连接", "red")

            self._log_event(f"刷新失败实验：AI={ai}，共 {len(rows)} 条", "dim")
        except Exception as e:
            self.write(f"刷新失败实验失败：{e}", "red")
            self._log_event(f"刷新失败实验失败：{e}", "red")

    def on_reset_ai_changed(self, event=None):
        self.refresh_failed_experiments()

    def _on_test_window_select(self, event=None):
        """选中测试窗口时的处理"""
        try:
            sel = self.test_windows_tree.selection()
            if not sel:
                return
            item = self.test_windows_tree.item(sel[0])
            values = item.get("values", [])
            if len(values) >= 7:
                window_id = values[0]
                code = values[1]
                train_start = values[2]
                train_end = values[3]
                test_start = values[4]
                test_end = values[5]
                reserved = values[6]

                lines = [
                    f"窗口 ID：{window_id}",
                    f"股票代码：{code}",
                    f"训练区间：{train_start} ~ {train_end}",
                    f"测试区间：{test_start} ~ {test_end}",
                    f"保留状态：{'是' if reserved == '是' else '否'}"
                ]
                self.test_win_detail_txt.config(text="\n".join(lines))
        except Exception as e:
            self.test_win_detail_txt.config(text=f"详情加载失败：{e}")

    def _on_failed_experiment_select(self, event=None):
        """选中失败实验时的处理"""
        try:
            sel = self.failed_tree.selection()
            if not sel:
                return
            item = self.failed_tree.item(sel[0])
            values = item.get("values", [])
            if len(values) >= 8:
                exp_id = values[0]
                code = values[1]
                train_start = values[2]
                train_end = values[3]
                test_start = values[4]
                test_end = values[5]
                status = values[6]
                reason = values[7]

                lines = [
                    f"实验 ID：{exp_id}",
                    f"股票代码：{code}",
                    f"训练区间：{train_start} ~ {train_end}",
                    f"测试区间：{test_start} ~ {test_end}",
                    f"状态：{status}",
                    f"失败原因：{reason or '无'}"
                ]
                self.reset_detail_txt.config(text="\n".join(lines))
        except Exception as e:
            self.reset_detail_txt.config(text=f"详情加载失败：{e}")

    def _on_test_window_detail(self):
        """查看测试窗口详情"""
        sel = self.test_windows_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择要查看的窗口")
            return
        # 详情已经在 _on_test_window_select 中实时显示
        pass

    def _on_unreserve_window(self):
        """取消保留固定测试窗口"""
        sel = self.test_windows_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择要取消保留的窗口")
            return

        item = self.test_windows_tree.item(sel[0])
        values = item.get("values", [])
        if len(values) < 1:
            return

        window_id = values[0]

        if not messagebox.askyesno(
            "确认取消保留",
            f"确定要取消保留窗口 {window_id} 吗？\n取消后该窗口可能被其他实验占用。"
        ):
            return

        try:
            reg = self._registry()
            if reg is None:
                return

            if hasattr(reg, 'conn'):
                cursor = reg.conn.cursor()
                cursor.execute("""
                    UPDATE test_windows SET reserved = 0
                    WHERE id = ?
                """, (window_id,))
                reg.conn.commit()

                self.refresh_test_windows()
                self.write(f"已取消保留窗口 {window_id}", "green")
                self._log_event(f"取消保留窗口 {window_id}", "yellow")
            else:
                _safe_showerror("错误", "无法访问 SQLite 连接")
        except Exception as e:
            self.write(f"取消保留失败：{e}", "red")
            self._log_event(f"取消保留失败：{e}", "red")

    def _on_batch_unreserve(self):
        """批量取消保留固定测试窗口"""
        sel = self.test_windows_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择要取消保留的窗口")
            return

        count = len(sel)
        if not messagebox.askyesno(
            "确认批量取消",
            f"确定要取消保留选中的 {count} 个窗口吗？"
        ):
            return

        try:
            reg = self._registry()
            if reg is None:
                return

            if hasattr(reg, 'conn'):
                cursor = reg.conn.cursor()
                for item_id in sel:
                    item = self.test_windows_tree.item(item_id)
                    values = item.get("values", [])
                    if len(values) >= 1:
                        window_id = values[0]
                        cursor.execute("""
                            UPDATE test_windows SET reserved = 0
                            WHERE id = ?
                        """, (window_id,))
                reg.conn.commit()

                self.refresh_test_windows()
                self.write(f"已批量取消保留 {count} 个窗口", "green")
                self._log_event(f"批量取消保留 {count} 个窗口", "yellow")
            else:
                _safe_showerror("错误", "无法访问 SQLite 连接")
        except Exception as e:
            self.write(f"批量取消失败：{e}", "red")
            self._log_event(f"批量取消失败：{e}", "red")

    def _on_reset_failed_experiment(self):
        """重置失败实验状态"""
        sel = self.failed_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择要重置的实验")
            return

        item = self.failed_tree.item(sel[0])
        values = item.get("values", [])
        if len(values) < 1:
            return

        exp_id = values[0]
        reason = values[7] if len(values) >= 8 else ""

        if not messagebox.askyesno(
            "确认重置实验状态",
            f"确定要重置实验 {exp_id} 吗？\n失败原因：{reason}\n\n重置后该实验将允许重新尝试。"
        ):
            return

        try:
            reg = self._registry()
            if reg is None:
                return

            if hasattr(reg, 'conn'):
                cursor = reg.conn.cursor()
                cursor.execute("""
                    UPDATE test_usage
                    SET status = 'pending', failure_reason = NULL
                    WHERE id = ?
                """, (exp_id,))
                reg.conn.commit()

                self.refresh_failed_experiments()
                self.write(f"已重置实验 {exp_id}", "green")
                self._log_event(f"重置实验 {exp_id}", "yellow")
            else:
                _safe_showerror("错误", "无法访问 SQLite 连接")
        except Exception as e:
            self.write(f"重置失败：{e}", "red")
            self._log_event(f"重置失败：{e}", "red")

    def _on_batch_reset_failed(self):
        """批量重置失败实验状态"""
        sel = self.failed_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择要重置的实验")
            return

        count = len(sel)
        if not messagebox.askyesno(
            "确认批量重置",
            f"确定要重置选中的 {count} 个实验吗？"
        ):
            return

        try:
            reg = self._registry()
            if reg is None:
                return

            if hasattr(reg, 'conn'):
                cursor = reg.conn.cursor()
                for item_id in sel:
                    item = self.failed_tree.item(item_id)
                    values = item.get("values", [])
                    if len(values) >= 1:
                        exp_id = values[0]
                        cursor.execute("""
                            UPDATE test_usage
                            SET status = 'pending', failure_reason = NULL
                            WHERE id = ?
                        """, (exp_id,))
                reg.conn.commit()

                self.refresh_failed_experiments()
                self.write(f"已批量重置 {count} 个实验", "green")
                self._log_event(f"批量重置 {count} 个实验", "yellow")
            else:
                _safe_showerror("错误", "无法访问 SQLite 连接")
        except Exception as e:
            self.write(f"批量重置失败：{e}", "red")
            self._log_event(f"批量重置失败：{e}", "red")

    def _build_history_tab(self, parent):
        bar = tk.Frame(parent, bg=UI_BG)
        bar.pack(side="top", fill="x", padx=10, pady=(8, 0))

        self._lbl(bar, "历史类型：", bg=UI_BG).pack(side="left", padx=(0, 2))
        self.hist_kind_var = tk.StringVar(value="跑测历史")
        self.hist_kind_combo = ttk.Combobox(
            bar, textvariable=self.hist_kind_var,
            values=["跑测历史", "调参寻优历史", "特征寻优历史"],
            state="readonly", width=14, font=("Microsoft YaHei", 11))
        self.hist_kind_combo.pack(side="left", padx=(0, 14))
        self.hist_kind_combo.bind("<<ComboboxSelected>>", self.on_hist_kind_changed)

        self._lbl(bar, "查看AI：", bg=UI_BG).pack(side="left", padx=(0, 2))
        self.hist_ai_var = tk.StringVar(value=self.current_ai())
        self.hist_ai_combo = ttk.Combobox(
            bar, textvariable=self.hist_ai_var, values=scan_all_ai_names(),
            state="readonly", width=16, font=("Microsoft YaHei", 11))
        self.hist_ai_combo.pack(side="left", padx=(0, 14))
        self.hist_ai_combo.bind("<<ComboboxSelected>>", self.on_hist_ai_changed)

        self._btn(bar, "刷新", self.refresh_history).pack(side="left", padx=4)
        self._btn(bar, "查看资金曲线", self._show_curve, bg="#2b6cb0").pack(side="left", padx=4)
        self._btn(bar, "删除选中", self.delete_selected_history, bg="#c0392b").pack(side="left", padx=4)
        self._btn(bar, "导出到桌面", self.export_history, bg="#1f7a1f").pack(side="left", padx=4)

        self.hist_tip = self._lbl(parent, "", bg=UI_BG, fg=UI_FG_DIM, font=FONT_S)
        self.hist_tip.pack(side="bottom", pady=6)

        frame = tk.Frame(parent, bg=UI_BG)
        frame.pack(side="top", fill="both", expand=True, padx=10, pady=(6, 0))
        header = list(self._hist_base_header())
        self.hist_tree = ttk.Treeview(frame, columns=header, show="headings",
                                      height=16, style="Dark.Treeview")
        self._apply_history_columns(self.hist_tree, header)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.hist_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.hist_tree.xview)
        self.hist_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.hist_tree.pack(side="left", fill="both", expand=True)
        self.hist_tree.bind("<<TreeviewSelect>>", self._on_hist_select)

        det = self._lf(parent, " 选中记录详情 ", accent=UI_ACCENT)
        det.pack(side="bottom", fill="x", padx=10, pady=(4, 2))
        self.hist_detail_txt = self._lbl(
            det, "点选上方任意一行，这里显示该条记录详情。",
            fg=UI_FG, font=("Microsoft YaHei", 10), bg=UI_PANEL,
            justify="left", anchor="w", wraplength=1240)
        self.hist_detail_txt.pack(fill="x", padx=8, pady=6)

    # ================================================================
    # 历史记录辅助
    # ================================================================
    def _hist_base_header(self):
        try:
            kind = self.hist_kind_var.get()
        except Exception:
            kind = "跑测历史"
        if kind == "特征寻优历史":
            return FEATURE_SEARCH_HEADER
        return HISTORY_HEADER

    def current_hist_ai(self):
        try:
            v = self.hist_ai_var.get().strip()
        except Exception:
            v = ""
        return v or self.current_ai()

    def current_ai(self):
        try:
            names = load_ai_names()
            if names:
                return names[0]
        except Exception:
            pass
        return "AI1"

    def current_history_path(self):
        ai = self.current_hist_ai()
        kind = self.hist_kind_var.get() if hasattr(self, "hist_kind_var") else "跑测历史"
        if kind == "调参寻优历史":
            return tuning_log_path_for(ai)
        if kind == "特征寻优历史":
            return feature_search_log_path_for(ai)
        return run_history_path_for(ai, True)

    def _ensure_hist_file(self, path):
        kind = self.hist_kind_var.get() if hasattr(self, "hist_kind_var") else "跑测历史"
        if kind == "特征寻优历史":
            ensure_feature_search_file(path)
        else:
            ensure_history_file(path)

    def _apply_history_columns(self, tree, header):
        tree["columns"] = list(header)
        wide = {"时间", "特征组合", "最佳特征组合", "备注", "资金曲线文件路径", "交易成本配置"}
        for c in header:
            tree.heading(c, text=str(c))
            w = 180 if c in wide else 110
            if c == "股票代码":
                w = 80
            tree.column(c, width=w, anchor="center", stretch=False)

    def refresh_hist_ai_combo(self):
        try:
            names = scan_all_ai_names()
            self.hist_ai_combo["values"] = names
            if self.hist_ai_var.get() not in names and names:
                self.hist_ai_var.set(names[0])
        except Exception:
            pass

    def on_hist_ai_changed(self, event=None):
        self.refresh_history()

    def on_hist_kind_changed(self, event=None):
        self.refresh_history()

    def refresh_history(self):
        try:
            ai = self.current_hist_ai()
            kind = self.hist_kind_var.get() if hasattr(self, "hist_kind_var") else "跑测历史"
            if not self._hist_initialized:
                self._hist_initialized = True
                try:
                    for nm in scan_all_ai_names():
                        migrate_tuning_log(nm)
                        for _rc in (False, True):
                            normalize_history_header(run_history_path_for(nm, _rc))
                except Exception:
                    pass
            if kind == "调参寻优历史":
                migrate_tuning_log(ai)
            path = self.current_history_path()
            self._ensure_hist_file(path)
            header, data = read_history_rows(path, self._hist_base_header())
            tree = self.hist_tree
            self._apply_history_columns(tree, header)
            for iid in tree.get_children():
                tree.delete(iid)
            note_idx = header.index("备注") if "备注" in header else -1

            def _is_sum(r):
                return note_idx >= 0 and len(r) > note_idx and "汇总" in str(r[note_idx])

            summary = [r for r in data if _is_sum(r)]
            normal = [r for r in data if not _is_sum(r)]
            try:
                normal.sort(key=lambda r: r[0] if r else "", reverse=True)
            except Exception:
                pass
            ordered = summary + normal
            if ordered:
                for r in ordered:
                    r = (list(r) + [""] * len(header))[:len(header)]
                    tree.insert("", tk.END, values=r)
            else:
                tree.insert("", tk.END, values=["（暂无历史记录）"] + [""] * (len(header) - 1))
            self.hist_tip.config(text=f"【AI：{ai}·{kind}】共 {len(ordered)} 条   文件：{path}")
        except Exception as e:
            try:
                self.write(f"刷新历史失败：{e}", "red")
            except Exception:
                pass

    def _on_hist_select(self, event=None):
        try:
            sel = self.hist_tree.selection()
            if not sel:
                return
            values = list(self.hist_tree.item(sel[0], "values"))
            cols = list(self.hist_tree["columns"])
            lines = [f"AI：{self.current_hist_ai()}    文件：{os.path.basename(self.current_history_path())}"]
            for c, v in zip(cols, values):
                sv = str(v).strip()
                if sv:
                    lines.append(f"  {c}：{sv}")
            self.hist_detail_txt.config(text="\n".join(lines))
        except Exception:
            pass

    def _show_curve(self):
        sel = self.hist_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先在表格里选中一行")
            return
        values = list(self.hist_tree.item(sel[0], "values"))
        cols = list(self.hist_tree["columns"])
        idx = cols.index("资金曲线文件路径") if "资金曲线文件路径" in cols else -1
        rel = str(values[idx]).strip() if idx >= 0 and len(values) > idx else ""
        if not rel:
            messagebox.showwarning("提示", "该条记录没有资金曲线文件")
            return
        path = os.path.join(DATA_DIR, rel)
        if not os.path.exists(path):
            messagebox.showwarning("提示", f"资金曲线文件不存在：\n{path}")
            return
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            matplotlib.rcParams['axes.unicode_minus'] = False
            cdf = pd.read_csv(path)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            ax.plot(cdf["日期"], cdf["净值"], color="#1f77b4", linewidth=1.6)
            ax.set_title("资金曲线")
            ax.set_xlabel("日期")
            ax.set_ylabel("净值")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            plt.show()
        except Exception as e:
            _safe_showerror("画图失败", f"画图出错：\n{e}")

    def delete_selected_history(self):
        path = self.current_history_path()
        sel = self.hist_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的行")
            return
        if not messagebox.askyesno("确认", f"确定删除选中的 {len(sel)} 条记录？（只删当前文件，不碰模型/曲线）"):
            return
        try:
            header, data = read_history_rows(path, self._hist_base_header())
            keep = list(data)
            # 按“时间+代码”等内容定位被选行；简单做法：比对整行值
            selected_vals = [tuple(self.hist_tree.item(i, "values")) for i in sel]
            new_rows = [r for r in keep
                        if tuple((list(r) + [""] * len(header))[:len(header)]) not in selected_vals]
            write_history_rows(path, new_rows, header)
            self.refresh_history()
        except Exception as e:
            _safe_showerror("删除失败", str(e))

    def export_history(self):
        try:
            import shutil
            path = self.current_history_path()
            if not os.path.exists(path):
                messagebox.showwarning("提示", "当前没有历史文件")
                return
            desk = get_desktop_dir()
            dst = os.path.join(desk, f"导出_{os.path.basename(path)}")
            if os.path.exists(dst):
                dst = os.path.join(desk, f"导出_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                                         f"{os.path.basename(path)}")
            shutil.copy2(path, dst)
            messagebox.showinfo("导出完成", f"已导出到桌面：\n{dst}")
        except Exception as e:
            _safe_showerror("导出失败", str(e))

    # ================================================================
    # 自主研究（自动发现股票 + 自动研究/验证/晋级，无需手动填代码）
    # ================================================================
    def _autonomous_available(self):
        return all(m is not None for m in (stock_pool, memory_guard, research_engine))

    def _registry(self):
        if self._auto_registry is None:
            if stock_pool is None:
                return None
            try:
                self._auto_registry = stock_pool.ResearchRegistry(
                    os.path.join(DATA_DIR, "research_queue.db"))
            except Exception as e:
                self.write(f"注册表打开失败：{e}", "red")
                return None
        return self._auto_registry

    def _make_guard(self):
        cfg = None
        p = os.path.join(DATA_DIR, "memory_rules.json")
        try:
            import json
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            else:
                # 首次运行写出默认可配置项，方便按本机情况调整（不覆盖已有文件）
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(memory_guard.DEFAULT_CFG, f, ensure_ascii=False, indent=2)
        except Exception:
            cfg = None
        return memory_guard.MemoryGuard(cfg, data_dir=DATA_DIR,
                                        log_fn=lambda s: self.write(s, "dim"))

    def _engine_cfg(self):
        """自主研究引擎参数：优先读 research_engine_config.json（可配置），否则用默认值。"""
        import json
        p = os.path.join(DATA_DIR, "research_engine_config.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}
        # 首次运行：写出默认可配置项再返回，绝不覆盖已有文件
        cfg = dict(research_engine.DEFAULT_ENGINE_CFG)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return cfg

    def _engine_scope(self):
        """研究范围（股票池）参数：优先读 research_scope.json（可配置）。"""
        import json
        p = os.path.join(DATA_DIR, "research_scope.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}
        scope = dict(stock_pool.DEFAULT_SCOPE)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(scope, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return scope

    def _gui_module_for_engine(self):
        """给调度器一个“真正的模块对象”，保证它临时切换标签定义时影响的是训练用的那份。"""
        try:
            import importlib
            return importlib.import_module("train_xgb_gui")
        except Exception:
            return sys.modules.get(__name__) or sys.modules.get("__main__")

    def _start_autonomous(self):
        if not self._autonomous_available():
            _safe_showerror("无法启动", "自主研究模块未就绪：缺少 stock_pool.py / "
                                            "memory_guard.py / research_engine.py。")
            return
        if self.working or self._auto_running:
            messagebox.showinfo("提示", "已有研究任务在运行中")
            return
        try:
            reg = self._registry()
            if reg is None:
                _safe_showerror("无法启动", "研究注册表打开失败，详见日志。")
                return
            guard = self._auto_guard or self._make_guard()
            self._auto_guard = guard
            cfg = self._engine_cfg()
            scope = self._engine_scope()

            def emit(kind, *args):
                self.q.put((kind,) + tuple(args))

            sched = research_engine.AutonomousScheduler(
                reg, self._gui_module_for_engine(), emit=emit, guard=guard,
                cfg=cfg, scope=scope, data_dir=DATA_DIR)
            self._auto_scheduler = sched
            self._auto_running = True
            ok, why = guard.can_start_heavy()
            self._log_research_plan(sched, cfg, scope, ok, why)
            self.write("🔭 启动自主研究：自动获取股票列表 → 检查行情 → 排序研究队列 → "
                       "自动训练/验证/晋级（不需要手动填股票代码）", "title")
            self._auto_thread = threading.Thread(target=self._autonomous_worker,
                                                 args=(sched,), daemon=True)
            self._auto_thread.start()
            self._set_auto_buttons(True)
            self._set_status("自主研究已启动")
            try:
                self.notebook.select(self.tab_train)
            except Exception:
                pass
        except Exception as e:
            self._auto_running = False
            self._set_auto_buttons(False)
            _safe_showerror("启动失败", f"无法启动自主研究：\n{e}")

    def _log_research_plan(self, sched, cfg, scope, mem_ok, mem_why):
        try:
            rules = sched.rules
            self.write("—— 自主研究参数（可在【模型训练器后端文件夹】里的 json 里修改）——", "title")
            self.write(f"研究范围：纳入主板/创业板={scope.get('include_gem', True)}，"
                       f"科创板={scope.get('include_star', False)}，ST 排除={scope.get('exclude_st', True)}，"
                       f"最少日线={scope.get('min_rows')}，日均成交额下限={scope.get('min_avg_amount_wan')}万元",
                       "dim")
            self.write(f"实验预算：基础预算={cfg.get('base_budget', research_engine.DEFAULT_ENGINE_CFG['base_budget'])}，"
                       f"滚动窗口={cfg.get('rolling_windows', 3)}×种子={cfg.get('rolling_seeds', 2)}，"
                       f"单实验重试上限={cfg.get('max_experiment_retries', 2)}", "dim")
            self.write(f"晋级门槛：窗口≥{rules['min_windows']}、种子≥{rules['min_seeds']}、"
                       f"正收益窗口占比≥{rules['min_positive_window_ratio']}、平均净收益>"
                       f"{rules['min_mean_ret_pct']}%、最差回撤≤{rules['max_worst_window_dd_pct']}%、"
                       f"种子标准差≤{rules['max_seed_std_pct']}%、相对基准收益改进≥"
                       f"{rules['min_improvement_ret_pp']}pp（实验越多门槛越高，"
                       f"每超出 {rules['selection_bias_free_experiments']} 次 +"
                       f"{rules['selection_bias_pp_per_experiment']}pp）", "dim")
            self.write("纪律：独立测试集不参与任何选择；本轮实验默认不执行独立测试。", "dim")
            if not mem_ok:
                self.write(f"⚠ 当前内存不满足启动重型任务：{mem_why}（会先等待资源恢复）", "yellow")
        except Exception:
            pass

    def _autonomous_worker(self, sched):
        try:
            ok, note = sched.run_loop()
            self.q.put(("auto_done", ok, note))
        except Exception as e:
            self.q.put(("log", f"自主研究线程异常：{e}", "red"))
            self.q.put(("auto_done", False, str(e)))

    def _set_auto_buttons(self, running):
        try:
            if running:
                self.auto_btn.config(state="disabled")
                self.train_btn.config(state="disabled")
                self.pause_btn.config(state="normal")
                self.resume_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
            else:
                self.auto_btn.config(state="normal")
                if not self.working:
                    self.train_btn.config(state="normal")
                self.pause_btn.config(state="disabled")
                self.resume_btn.config(state="disabled")
                self.stop_btn.config(state="disabled")
        except Exception:
            pass

    def _on_auto_done(self, ok, note):
        try:
            self._auto_running = False
            self._set_auto_buttons(False)
            loop = {}
            reg = self._auto_registry
            if reg is not None:
                loop = reg.get_state("loop", {}) or {}
            status = str(loop.get("status") or "")
            if ok:
                self._set_status("自主研究已结束（可再次启动）")
                self._log_event(f"自主研究结束：{note}", "green")
            else:
                self._set_status(f"自主研究进入暂停：{note}")
                self._log_event(f"自主研究暂停：{note}", "red")
                messagebox.showwarning(
                    "自主研究已暂停",
                    f"自主研究已进入错误暂停：\n{note}\n\n"
                    "修复数据源/内存问题后点【继续训练】可从检查点恢复；"
                    "状态与实验记录均已保存。")
            if status in ("error_paused", "paused"):
                self.resume_btn.config(state="normal")
        except Exception:
            pass

    def _render_auto_state(self, snap):
        try:
            snap = snap or {}
            pool = snap.get("pool") or {}
            ds = snap.get("data_source") or {}
            loop = snap.get("loop") or {}
            counters = snap.get("counters") or {}
            total = int(pool.get("total", 0) or 0)
            ready = int(pool.get("ready", 0) or 0)
            pending = int(pool.get("pending", 0) or 0)
            cooldown = int(pool.get("cooldown", 0) or 0)
            done = int(counters.get("experiments_done", 0) or 0)
            failed = int(counters.get("experiments_failed", 0) or 0)
            if ds.get("ok") is True:
                ds_txt = f"正常（本次纳入 {ds.get('count', '-')} 只，{ds.get('checked_at', '')}）"
            elif ds.get("ok") is False:
                ds_txt = (f"获取失败：{ds.get('error', '')}｜最后成功："
                          f"{ds.get('last_success') or '从未成功'}")
            else:
                ds_txt = "尚未更新"
            self.auto_pool_lbl.config(
                text=(f"股票池：{total} 只（可研究 {ready}，待检查 {pending}，回撤冷却 {cooldown}）"
                      f"  | 已完成/失败实验：{done} / {failed}\n数据源：{ds_txt}"))
            s_map = {"running": "运行中", "paused": "已暂停", "stopped": "已停止",
                     "idle": "空闲等待", "error_paused": "错误暂停"}
            st = s_map.get(str(loop.get("status") or ""), "未启动")
            self.auto_run_lbl.config(
                text=(f"自主研究：{st}  | 当前股票：{loop.get('current_code') or '-'}"
                      f"  | 当前实验：{loop.get('current_exp') or '-'}\n"
                      f"暂停/停止/异常原因：{loop.get('reason') or '-'}"))
            best = snap.get("best") or {}
            last = (snap.get("last_exp") or [{}])[0]
            last_txt = ""
            if last:
                last_txt = (f"｜最近实验：{last.get('code', '')} {last.get('goal', '')} "
                            f"{last.get('status', '')}"
                            + ("（晋级）" if last.get("promoted") else ""))
            b_txt = "-"
            if best:
                b_txt = (f"{best.get('code', '')} 验证段平均收益 "
                         f"{best.get('mean_ret_pct')}%（{best.get('updated_at', '')}）")
            self.auto_best_lbl.config(
                text=(f"当前最佳验证版本：{b_txt}{last_txt}\n"
                      f"最近一次成功保存：{snap.get('last_save') or loop.get('last_save') or '-'}"))
        except Exception:
            pass

    def _auto_status_tick(self):
        if not self._auto_ui_tick_running:
            return
        try:
            reg = self._auto_registry
            if reg is not None:
                loop = reg.get_state("loop", {}) or {}
                try:
                    last_exp = reg.list_experiments(limit=1)
                except Exception:
                    last_exp = []
                snap = {"pool": reg.pool_counts(), "exp_counts": reg.experiment_counts(),
                        "data_source": reg.get_data_source_status(), "loop": loop,
                        "counters": reg.get_state("counters", {}) or {},
                        "last_save": loop.get("last_save", ""), "last_exp": last_exp}
                # 尽量补上“当前最佳验证版本”（按最近实验的股票代码查基准）
                try:
                    code = str(loop.get("current_code") or "")
                    if not code and last_exp:
                        code = str(last_exp[0].get("code") or "")
                    if code:
                        b = reg.get_baseline(code)
                        if b:
                            snap["best"] = {
                                "code": code,
                                "mean_ret_pct": (b.get("metrics") or {}).get("mean_ret_pct"),
                                "updated_at": b.get("updated_at")}
                except Exception:
                    pass
                self._auto_last_state = snap
                self._render_auto_state(snap)
                if not self._auto_running and str(loop.get("status")) in ("paused", "error_paused"):
                    try:
                        self.resume_btn.config(state="normal")
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            g = self._auto_guard
            if g is not None:
                self.auto_res_lbl.config(text="资源：" + g.status_text(sample=True))
        except Exception:
            pass
        try:
            self.root.after(3000, self._auto_status_tick)
        except Exception:
            pass

    # ================================================================
    # 一键自动训练
    # ================================================================
    def _one_click_train(self):
        if self.working:
            return
        if self._auto_running:
            messagebox.showinfo("提示", "自主研究正在运行中，请先【结束研究】再手动训练。")
            return
        code = "".join(ch for ch in self.e_code.get().strip() if ch.isdigit())
        train_start = self.e_ts.get().strip()
        train_end = self.e_te.get().strip()
        test_start = self.e_ps.get().strip()
        test_end = self.e_pe.get().strip()
        if len(code) != 6:
            _safe_showerror("输入错误", "股票代码必须是 6 位数字")
            return
        for label, v in (("训练开始", train_start), ("训练结束", train_end),
                         ("测试开始", test_start), ("测试结束", test_end)):
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except Exception:
                _safe_showerror("输入错误", f"{label}日期格式应为 YYYY-MM-DD：{v}")
                return
        if not (train_start < train_end < test_start <= test_end):
            _safe_showerror(
                "日期顺序错误",
                "必须满足：训练开始 < 训练结束 < 测试开始 ≤ 测试结束\n"
                "（训练与测试之间允许留空档，但不允许重叠）")
            return

        self.log.delete("1.0", tk.END)
        self._log_buf = []
        self.stop_ev.clear()
        self._reset_stages()
        self._set_status("正在准备 ...")
        self._set_result("研究进行中，请稍候 ... 全部完成后这里显示最终结论。")

        cfg = {
            "ai_name": self.current_ai(),
            "code": code, "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "trigger": "手动训练",
        }
        # 记录手动训练检查点（仅作记录/显示，便于出错后定位；不影响训练选择逻辑）
        try:
            reg = self._registry()
            if reg is not None:
                reg.set_state("manual_checkpoint", {
                    "code": code, "ai_name": cfg["ai_name"],
                    "train_start": train_start, "train_end": train_end,
                    "test_start": test_start, "test_end": test_end,
                    "status": "running", "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
        except Exception:
            pass
        self.emit("busy", True)
        self.emit("log", f"开始一键自动训练：【AI：{cfg['ai_name']}】【{code}】", "title")
        self.emit("log", f"训练 {train_start} ~ {train_end} | 测试 {test_start} ~ {test_end}", "log")
        self.write("提示：测试区间只会在最后独立评估一次，不参与任何选择。", "dim")

        self._worker_thread = threading.Thread(target=self._worker, args=(cfg,), daemon=True)
        self._worker_thread.start()
        try:
            self.notebook.select(self.tab_train)
        except Exception:
            pass

    def _worker(self, cfg):
        gate = None
        held = False
        try:
            from auto_research import AutoRunner
            # 与自主研究共用同一把全局重型任务闸门：即使界面状态被绕过，
            # 也保证手动训练与自主研究不会同时跑两个重型任务。
            if memory_guard is not None:
                gate = getattr(memory_guard, "GLOBAL_HEAVY_GATE", None)
                if gate is not None:
                    held = gate.try_acquire(owner="manual_train")
                    if not held:
                        self.q.put(("log", "⚠ 已有重型任务在运行（自主研究或另一次训练），本次训练未启动。", "yellow"))
                        self.q.put(("done", False, {"status": "busy",
                                                    "ui_summary": {"aborted": "已有重型任务在运行"}}))
                        return

            def emit(kind, *args):
                self.q.put((kind,) + tuple(args))

            runner = AutoRunner(cfg, emit, stop_fn=lambda: self.stop_ev.is_set(),
                                state_dir=DATA_DIR)
            self._runner = runner
            ok, results = runner.run()
            self.q.put(("done", ok, results))
        except Exception as e:
            self.q.put(("log", f"后台任务异常：{e}", "red"))
            self.q.put(("done", False, {"exception": str(e)}))
        finally:
            if held and gate is not None:
                try:
                    gate.release()
                except Exception:
                    pass
            self.q.put(("busy", False))

    # ================================================================
    # 线程安全队列 & 日志
    # ================================================================
    def emit(self, *msg):
        self.q.put(msg)

    def poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if not msg:
                    continue
                kind = msg[0]
                if kind == "log":
                    tag = msg[2] if len(msg) > 2 else None
                    self.write(msg[1], tag)
                elif kind == "stage":
                    self._set_stage(msg[1], msg[2] if len(msg) > 2 else "")
                elif kind == "progress":
                    cur = msg[1] if len(msg) > 1 else None
                    total = msg[2] if len(msg) > 2 else None
                    label = msg[3] if len(msg) > 3 else ""
                    self._set_progress(cur, total, label)
                elif kind == "status":
                    self._set_status(msg[1] if len(msg) > 1 else "")
                elif kind == "busy":
                    self.working = bool(msg[1])
                    self.train_btn.config(
                        state=("disabled" if (self.working or self._auto_running) else "normal"))
                    try:
                        if self.working or self._auto_running:
                            self.auto_btn.config(state="disabled")
                        else:
                            self.auto_btn.config(state="normal")
                    except Exception:
                        pass
                    if self.working:
                        self.progress.config(mode="determinate", maximum=100, value=0)
                elif kind == "auto_status":
                    self._render_auto_state(msg[1] if len(msg) > 1 else {})
                elif kind == "auto_done":
                    self._on_auto_done(msg[1] if len(msg) > 1 else False,
                                       msg[2] if len(msg) > 2 else "")
                elif kind == "summary":
                    self._show_summary(msg[1] if len(msg) > 1 else {})
                elif kind == "box":
                    (_safe_showerror if msg[1] == "error" else messagebox.showinfo)(msg[2], msg[3])
                elif kind == "refresh_history":
                    try:
                        self.hist_kind_var.set("跑测历史")
                    except Exception:
                        pass
                    self.refresh_hist_ai_combo()
                    self.refresh_history()
                elif kind == "chart":
                    pass
                elif kind == "done":
                    self._on_done(msg[1] if len(msg) > 1 else False,
                                  msg[2] if len(msg) > 2 else {})
        except queue.Empty:
            pass
        except Exception as e:
            try:
                self.write(f"界面队列处理异常：{e}", "red")
            except Exception:
                pass
        self.root.after(120, self.poll_queue)

    def write(self, text, tag=None):
        self._log_buf.append((text, tag))

    def _flush_log_buffer(self):
        try:
            if self._log_buf:
                buf = self._log_buf
                self._log_buf = []
                stamp = datetime.now().strftime("%H:%M:%S")
                for text, tag in buf:
                    s = "" if text is None else str(text)
                    raw = s.strip()
                    line = s if (raw and set(raw) <= set("=-·. ")) else f"[{stamp}] {s}"
                    if tag:
                        self.log.insert(tk.END, line + "\n", tag)
                    else:
                        self.log.insert(tk.END, line + "\n")
                try:
                    total = int(self.log.index("end-1c").split(".")[0])
                    if total > LOG_MAX_LINES:
                        self.log.delete("1.0", f"{total - LOG_MAX_LINES}.0")
                except Exception:
                    pass
                try:
                    self.log.see(tk.END)
                except Exception:
                    pass
        except Exception:
            pass
        self.root.after(120, self._flush_log_buffer)

    def _clear_log(self):
        try:
            self.log.delete("1.0", tk.END)
            self._log_buf = []
            self.write("日志已清空（仅清显示）", "dim")
        except Exception:
            pass

    def _clear_events(self):
        try:
            self.events_log.delete("1.0", tk.END)
        except Exception:
            pass

    def _log_event(self, text, tag=None):
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{stamp}] {text}"
            if tag:
                self.events_log.insert(tk.END, line + "\n", tag)
            else:
                self.events_log.insert(tk.END, line + "\n")
            try:
                total = int(self.events_log.index("end-1c").split(".")[0])
                if total > 5000:
                    self.events_log.delete("1.0", f"{total - 3000}.0")
            except Exception:
                pass
            try:
                self.events_log.see(tk.END)
            except Exception:
                pass
        except Exception:
            pass

    # ================================================================
    # 状态 / 进度 / 结果渲染（主线程）
    # ================================================================
    def _reset_stages(self):
        try:
            self._stage_status = {i: "pending" for i in range(1, len(auto_research.STAGES) + 1)}
        except Exception:
            self._stage_status = {}
        self.progress.config(value=0, maximum=100)
        self.stage_lbl.config(text="当前阶段：启动中 ...", fg=UI_ACCENT)
        self.current_task_lbl.config(text="当前具体任务：正在初始化")
        self.exp_num_lbl.config(text="当前实验编号：-  | 已完成实验：0  | 剩余（动态）：未确定")
        self.time_range_lbl.config(text="训练/验证时间范围：-  | 最终测试集：受保护数据（仅最后评估一次）")
        self.why_lbl.config(text="执行原因：启动自动研究")
        self.best_cand_lbl.config(text="当前最佳候选：-  | 与基准对比：-")
        self.next_lbl.config(text="下一步计划：准备检查数据")
        self._last_step_time = datetime.now()

    def _set_stage(self, idx, status):
        try:
            total = len(auto_research.STAGES)
            name = auto_research.STAGES[idx - 1] if 1 <= idx <= total else str(idx)
            self._stage_status[idx] = status
            self.stage_lbl.config(text=f"当前阶段：{name}（{idx}/{total}）", fg=UI_ACCENT)
            if status == "done":
                self.progress.config(value=min(100, int(idx / total * 100)))
            self._log_event(f"阶段变更：{name} → {status}")
            self._update_runtime()
        except Exception:
            pass

    def _set_progress(self, cur, total, label=""):
        try:
            if total:
                self.progress.config(mode="determinate", maximum=int(total), value=int(cur or 0))
            if label:
                self._set_status(label)
            self._update_runtime()
        except Exception:
            pass

    def _set_status(self, text):
        try:
            self.status_lbl.config(text=str(text))
            self.current_task_lbl.config(text=f"当前具体任务：{text}")
            self._last_step_time = datetime.now()
            self._update_runtime()
            self._log_event(f"状态：{text}")
        except Exception:
            pass

    def _set_result(self, text):
        try:
            if isinstance(self.result_txt, tk.Label):
                self.result_txt.config(text=str(text))
            else:
                self.result_txt.config(state="normal")
                self.result_txt.delete("1.0", tk.END)
                self.result_txt.insert("1.0", str(text))
                self.result_txt.config(state="disabled")
        except Exception:
            pass

    def _show_summary(self, summary):
        try:
            if not isinstance(summary, dict):
                return
            status = summary.get("status")
            lines = []
            lines.append(f"实验编号：{summary.get('exp_id', '-')}")
            if status == "no_qualified":
                lines.append("结论：暂无合格候选模型（已按纪律结束，未强行选模型）")
                if summary.get("aborted"):
                    lines.append(f"中止原因：{summary['aborted']}")
            elif summary.get("test_skipped"):
                reason = str(summary.get("test_skip_reason") or "未执行")
                lines.append(f"结论：已完成训练期选择；独立测试未执行 —— {reason}")
            elif summary.get("aborted"):
                lines.append(f"结论：本次研究中止 —— {summary['aborted']}")
            else:
                lines.append("结论：已完成训练期选择 + 独立测试（一次性）")
            bf = summary.get("best_features") or []
            if bf:
                lines.append(f"最佳特征（{len(bf)}）：{', '.join(bf)}")
            bs = summary.get("best_strategy") or {}
            if bs:
                lines.append("最佳策略：止损 {}% | 止盈 {}% | MA{} | 跌破均线卖出 {}".format(
                    bs.get("stop_loss"), bs.get("take_profit"), bs.get("ma_period"),
                    "开" if bs.get("ma_break_exit") else "关"))
            stab = summary.get("stability") or {}
            if stab:
                lines.append("训练期稳定性：平均收益 {:+.2f}% | 标准差 {:.2f}% | 平均交易 {:.1f} 次".format(
                    stab.get("mean_ret", 0), stab.get("std_ret", 0), stab.get("mean_trades", 0)))
            t = summary.get("test") or {}
            if t:
                lines.append("独立测试：区间 {} ~ {} | 收益 {:+.2f}% | 回撤 {:.1f}% | 交易 {} 次".format(
                    t.get("start"), t.get("end"), t.get("ret_pct", 0),
                    t.get("max_dd_pct", 0), t.get("n_trades")))
                acc = t.get("acc")
                if isinstance(acc, (int, float)) and acc == acc:
                    lines.append(f"测试准确率：{acc:.4f}    说明：{t.get('note', '')}")
            self._set_result("\n".join(lines))
        except Exception:
            pass

    def _on_done(self, ok, results):
        try:
            # 记录手动训练结束状态；若自主研究未运行，则把界面循环状态归位，避免显示残留的“运行中”
            try:
                reg = self._auto_registry
                if reg is not None:
                    cp = reg.get_state("manual_checkpoint", None)
                    if cp:
                        cp = dict(cp)
                        cp["status"] = "done" if ok else "failed"
                        cp["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        reg.set_state("manual_checkpoint", cp)
                    if not self._auto_running:
                        loop = reg.get_state("loop", {}) or {}
                        if str(loop.get("status") or "") == "running":
                            loop = dict(loop)
                            loop["status"] = "stopped"
                            loop["reason"] = "手动训练结束"
                            reg.set_state("loop", loop)
            except Exception:
                pass
            if self._restart_in_progress:
                try:
                    self.progress.config(value=100 if ok else self.progress["value"])
                except Exception:
                    pass
                self._set_status("训练已停止，准备重启 ...")
                return
            self.progress.config(value=100 if ok else self.progress["value"])
            summ = {}
            try:
                summ = (results or {}).get("ui_summary") or {}
            except Exception:
                summ = {}
            status = (results or {}).get("status")
            save_ok = summ.get("save_ok", True)
            if ok and status != "no_qualified" and not save_ok:
                self._set_status("结果未完整保存 ⚠")
                self._log_event("结果未完整保存，请检查模型文件保存情况", "red")
                messagebox.showwarning(
                    "保存失败",
                    "研究流程完成，但模型/结果文件未能完整保存，本次结果不可复用。\n"
                    "请在运行日志与实验目录中查看具体原因（绝不视为成功）。")
                return
            if ok:
                self._set_status("全部完成 ✔")
                self._log_event("训练完成", "green")
                if status == "no_qualified":
                    messagebox.showinfo("完成", "研究已结束：暂无合格候选模型（详见日志与报告）。")
                else:
                    messagebox.showinfo("完成", "一键自动训练完成，结果已保存并写入历史记录。")
            else:
                self._set_status("已结束（未通过 / 中止）")
                self._log_event("训练结束（未通过或中止）", "yellow")
                messagebox.showwarning("结束", "本次研究未得到合格结果，详情见运行日志。")
        except Exception:
            pass

    # ================================================================
    # 一键重启（恢复旧版「一键更新并重启」的安全流程）
    #   - 有任务在跑：先请求“协作式”安全停止，等待收尾保存，绝不强杀线程
    #   - 语法检查 → 备份当前代码 → 启动新进程 → 安全退出旧进程
    #   - 全程不删除/不覆盖任何历史记录、模型文件、资金曲线
    # ================================================================
    def _load_inputs(self):
        """启动时恢复上次保存的输入（若存在），失败则保持默认值。"""
        try:
            import json
            if not os.path.exists(PARAM_FILE):
                return
            with open(PARAM_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            mapping = {"code": self.e_code, "train_start": self.e_ts,
                       "train_end": self.e_te, "test_start": self.e_ps,
                       "test_end": self.e_pe}
            for k, ent in mapping.items():
                v = str(d.get(k, "")).strip()
                if v:
                    ent.delete(0, tk.END)
                    ent.insert(0, v)
        except Exception:
            pass

    def _save_inputs(self):
        """保存当前 5 个输入框，供重启后的新进程恢复。只写 gui_params.json。"""
        try:
            import json
            d = {"code": self.e_code.get().strip(),
                 "train_start": self.e_ts.get().strip(),
                 "train_end": self.e_te.get().strip(),
                 "test_start": self.e_ps.get().strip(),
                 "test_end": self.e_pe.get().strip()}
            with open(PARAM_FILE, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _auto_cleanup_backups_ui(self):
        """【清理旧备份】：只由【一键重启】触发，保留最新 BACKUP_KEEP 个备份。"""
        try:
            removed = auto_cleanup_backups(BACKUP_KEEP)
        except Exception:
            return
        try:
            if removed:
                self.write(f"🧹 已清理 {len(removed)} 个旧备份，仅保留最新 {BACKUP_KEEP} 个", "title")
            else:
                self.write(f"🧹 备份检查完成：备份数量未超过 {BACKUP_KEEP} 个，无需清理", "title")
        except Exception:
            pass

    def _refresh_backup_info(self):
        """刷新【最近备份】提示标签。"""
        try:
            files = list(_collect_backup_files())
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            if files:
                p = files[0]
                stamp = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
                self.backup_info_lbl.config(text=f"最近备份：{os.path.basename(p)}（{stamp}）")
            else:
                self.backup_info_lbl.config(text="最近备份：暂无")
        except Exception:
            try:
                self.backup_info_lbl.config(text="最近备份：-")
            except Exception:
                pass

    def _one_click_restart(self):
        """【一键重启】入口：若正在训练/自主研究，先请求安全停止并等待收尾，再重启。"""
        if self._restart_in_progress:
            return
        busy = bool(self.working or self._auto_running)
        if busy:
            if not messagebox.askyesno(
                    "确认重启",
                    "当前有任务正在运行！\n\n"
                    "重启会先请求安全停止该任务（已完成的实验与日志会尽量保存），"
                    "然后再重新启动程序。\n\n确定要现在重启吗？"):
                return
            self._restart_in_progress = True
            self._restart_ticks = 0
            self._set_status("正在安全停止当前任务，随后重启 ...")
            self.write("收到重启请求：先安全停止当前任务，等待收尾保存 ...", "title")
            try:
                self.stop_ev.set()
            except Exception:
                pass
            # 自主研究：同时请求调度器安全停止（在可控边界结束当前实验并保存）
            try:
                if self._auto_scheduler is not None and self._auto_running:
                    self._auto_scheduler.request_stop("重启前安全停止")
            except Exception:
                pass
            self.root.after(400, self._wait_stop_then_restart)
            return
        self._do_restart()

    def _wait_stop_then_restart(self):
        """等待后台训练/自主研究线程“协作式”退出；超时则明确放弃重启（绝不强杀）。"""
        try:
            if self.working or self._auto_running:
                self._restart_ticks += 1
                if self._restart_ticks <= 150:      # 约 60 秒
                    self.root.after(400, self._wait_stop_then_restart)
                    return
                self._restart_in_progress = False
                self._set_status("已取消重启（任务未能安全停止）")
                messagebox.showwarning(
                    "已取消重启",
                    "任务在限时内未能安全停止。\n\n"
                    "为避免写坏文件，本次重启已取消（没有强制终止线程）。\n"
                    "请等任务结束后再点【一键重启】。")
                return
            self._restart_in_progress = False
            self.write("当前任务已安全停止，开始重启 ...", "title")
            self.root.after(300, self._do_restart)
        except Exception:
            self._restart_in_progress = False

    def _do_restart(self):
        """【一键重启】安全流程：语法检查 → 备份 → 启动新进程 → 退出旧进程。"""
        import shutil
        import py_compile
        import subprocess

        src = os.path.abspath(__file__)
        backup_dir = os.path.join(DATA_DIR, "backups")
        try:
            os.makedirs(backup_dir, exist_ok=True)
        except Exception:
            pass

        # 1) 检测代码是否更新（无修改可强制重启）
        changed = True
        try:
            if os.path.getmtime(src) <= self._startup_mtime:
                changed = False
        except Exception:
            changed = True
        if not changed:
            if not messagebox.askyesno("未检测到代码修改",
                                       "未检测到代码修改。\n\n是否仍要强制重启？"):
                self._auto_cleanup_backups_ui()
                return

        # 2) 语法检查：不通过则中止（不写盘、不备份）
        try:
            py_compile.compile(src, doraise=True)
        except Exception as ex:
            _safe_showerror(
                "语法检查未通过",
                f"重启已中止（代码存在语法错误，未做任何改动）：\n{ex}")
            return

        # 3) 备份当前文件（带时间戳），保证可回溯
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"train_xgb_gui_backup_{stamp}.py")
        try:
            shutil.copy2(src, backup_path)
        except Exception as e:
            _safe_showerror("备份失败", f"重启已中止（无法创建备份）：\n{e}")
            return

        # 3.5) 清理旧备份 + 刷新备份信息
        self._auto_cleanup_backups_ui()
        try:
            self._refresh_backup_info()
        except Exception:
            pass

        # 4) 保存当前输入（新进程启动后自动恢复）
        self._save_inputs()

        # 5) 启动新进程（带正确的 cwd）
        try:
            proc = subprocess.Popen(
                [sys.executable, src] + list(sys.argv[1:]),
                cwd=os.path.dirname(src) or BASE_DIR)
        except Exception as e:
            _safe_showerror(
                "启动失败",
                f"无法启动新进程：{e}\n\n代码未被修改，备份已保留：\n{backup_path}")
            return

        self.write(f"✅ 已备份当前代码：{os.path.basename(backup_path)}（{stamp}）", "title")
        self.write("✅ 新进程已启动，正在确认是否正常 ...", "title")

        # 6) 短暂观察，确认新进程真的起来了（避免“假重启”）
        self._restart_proc = proc
        self._restart_backup_path = backup_path
        self.root.after(1600, self._verify_restart, proc, backup_path)

    def _verify_restart(self, proc, backup_path):
        """确认新进程是否存活：存活则退出旧进程，失败则保留备份并提示。"""
        dead = False
        try:
            dead = proc.poll() is not None
        except Exception:
            dead = False
        if dead:
            self._restart_in_progress = False
            _safe_showerror(
                "启动失败",
                "新进程启动后立即退出，疑似启动失败。\n\n"
                f"当前代码备份已保留，可手动恢复：\n{backup_path}\n\n"
                "旧窗口仍可用，请检查代码后重试。")
            return
        self.write("✅ 新进程运行正常，正在退出旧窗口 ...", "title")
        self.root.after(900, self._force_quit)

    def _force_quit(self):
        """强制退出当前进程（用于重启）。"""
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)

    # ================================================================
    # 关闭
    # ================================================================
    def _on_close(self):
        try:
            if self._auto_running and self._auto_scheduler is not None:
                if not messagebox.askyesno(
                        "确认退出",
                        "自主研究仍在进行。\n\n"
                        "退出会先请求安全停止（当前实验会在可控边界结束并保存），"
                        "未完成的实验下次启动会自动标记为“中断”（可重新研究）。\n\n"
                        "确定要退出吗？"):
                    return
                try:
                    self._auto_scheduler.request_stop("退出程序")
                except Exception:
                    pass
                self._set_status("正在安全停止并保存，随后退出 ...")
                self._close_ticks = 0
                self.root.after(300, self._close_wait_then_destroy)
                return
            if self.working:
                if not messagebox.askyesno("确认退出", "研究仍在进行，确定要中断并退出吗？"):
                    return
                self.stop_ev.set()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _close_wait_then_destroy(self):
        """退出前给调度器一点时间做安全边界保存；超时也退出（未完成实验靠重启恢复）。"""
        try:
            if self._auto_running and self._close_ticks < 80:      # 约 24 秒
                self._close_ticks += 1
                self.root.after(300, self._close_wait_then_destroy)
                return
            if self._auto_running:
                self.write("退出等待超时：未完成的实验将在下次启动时标记为“中断”，"
                           "已完成的实验与检查点均已保存。", "yellow")
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    # 必须在任何输出/异常之前安装：pythonw 下 stdout/stderr 可能为 None。
    _install_pythonw_diagnostics()
    # 启动期一次性数据维护（显式调用，避免被 import 时产生副作用）
    try:
        _migrate_desktop_data()
    except Exception:
        # 记录但不改变原有行为（启动继续），日志失败也不影响原异常处理
        _log_exception("启动期数据迁移失败", *sys.exc_info())
    # 历史错别字 AI 名 → 正式名（安全复制、绝不覆盖；并写回 ais.json）
    try:
        # 不再直接调用 sys.stdout.write：pythonw 下 stdout 可能为 None
        normalize_ai_names(log_fn=_startup_log)
    except Exception:
        _log_exception("启动期 AI 名规范化失败", *sys.exc_info())
    root = tk.Tk()
    # Tkinter 回调异常默认只打印到 stderr；pythonw 下 stderr 可能不可用
    root.report_callback_exception = _log_tk_exception
    app = TrainApp(root)
    root.mainloop()
