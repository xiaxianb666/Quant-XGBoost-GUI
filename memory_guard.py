# -*- coding: utf-8 -*-
"""
memory_guard.py — 【4GB 老电脑资源保护】内存 / CPU 监控 + 安全闸门

设计目标（对应需求第七节）：
  * 训练前评估可用内存，决定能不能启动重型任务；
  * 训练期间定期采样，内存到预警阈值→不再启动新任务；
  * 到达危险阈值→请求在当前任务“可控边界”安全停止并保存（不强行杀进程）；
  * 记录进程内存增长，连续采样 + 诊断信息判定“疑似泄漏”，不把正常缓存误判成泄漏；
  * 周期记录 Python 进程内存、系统可用物理内存、CPU 使用率（CSV 增量追加，自动轮转）。

本模块只做“测量 + 判定”，不做任何训练/回测；所有阈值都可配置，并按本机真实内存自适应。
不依赖 psutil 也能工作（Windows ctypes 兜底）。
"""

import os
import csv
import time
import threading
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------- 默认阈值 ----
# 注意：以下数值是“比例自适应”的兜底默认值，实际生效值会在 __init__ 里按本机物理内存重算，
# 绝不假设所有 4GB 电脑有相同的可用内存。
DEFAULT_CFG = {
    # 系统可用物理内存阈值（MB）：低于 warn → 不再启动新重型任务；低于 danger → 请求安全暂停
    "warn_avail_ratio": 0.15,      # 占总内存比例
    "danger_avail_ratio": 0.08,
    "warn_avail_mb_min": 450,
    "danger_avail_mb_min": 280,
    # 本进程内存阈值（MB）
    "proc_warn_mb": 900,
    "proc_danger_mb": 1400,
    "proc_warn_ratio": 0.25,
    "proc_danger_ratio": 0.40,
    # 采样 / 泄漏检测
    "sample_interval_s": 4.0,      # 采样最小间隔（避免高频采样本身占资源）
    "log_interval_s": 15.0,        # 写 CSV 的最小间隔
    "leak_window": 8,              # 连续采样点数
    "leak_growth_mb": 200.0,       # 窗口内进程内存净增长超过它才算可疑
    "log_max_bytes": 2 * 1024 * 1024,   # 资源日志超过 2MB 自动轮转一次
}

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_DANGER = "danger"


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 底层测量 ----
def _psutil():
    try:
        import psutil  # noqa
        return psutil
    except Exception:
        return None


def _win_mem_via_ctypes():
    """Windows 兜底：GlobalMemoryStatusEx + GetProcessMemoryInfo，返回 (total_mb, avail_mb, proc_mb)。"""
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        total_mb = st.ullTotalPhys / (1024.0 * 1024.0)
        avail_mb = st.ullAvailPhys / (1024.0 * 1024.0)

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        proc_mb = 0.0
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                proc_mb = pmc.WorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            proc_mb = 0.0
        return total_mb, avail_mb, proc_mb
    except Exception:
        return 0.0, 0.0, 0.0


class MemoryGuard:
    """内存 / CPU 保护器。线程安全：内部加锁，任何方法都不会抛异常。"""

    def __init__(self, cfg=None, data_dir=None, log_fn=None):
        self.cfg = dict(DEFAULT_CFG)
        if isinstance(cfg, dict):
            self.cfg.update({k: v for k, v in cfg.items() if v is not None})
        self.data_dir = data_dir or os.path.expanduser("~")
        self.log_fn = log_fn if callable(log_fn) else (lambda *a, **k: None)
        self._lock = threading.RLock()
        self._hist = deque(maxlen=64)          # 最近采样 (ts, avail_mb, proc_mb, cpu)
        self._last_sample_t = 0.0
        self._last_log_t = 0.0
        self._last = {"total_mb": 0.0, "avail_mb": 0.0, "proc_mb": 0.0,
                      "cpu_pct": -1.0, "ts": 0.0, "level": LEVEL_OK, "note": ""}
        self._leak_flag = False
        self._leak_note = ""

        # ---- 按本机真实内存自适应阈值 ----
        total_mb = self._detect_total_mb()
        self.total_mb = total_mb
        cfgd = self.cfg
        if total_mb > 0:
            cfgd["warn_avail_mb"] = max(cfgd["warn_avail_mb_min"],
                                        int(total_mb * cfgd["warn_avail_ratio"]))
            cfgd["danger_avail_mb"] = max(cfgd["danger_avail_mb_min"],
                                          int(total_mb * cfgd["danger_avail_ratio"]))
            cfgd["proc_warn_mb"] = min(cfgd["proc_warn_mb"],
                                       int(max(450, total_mb * cfgd["proc_warn_ratio"])))
            cfgd["proc_danger_mb"] = min(cfgd["proc_danger_mb"],
                                         int(max(700, total_mb * cfgd["proc_danger_ratio"])))
        else:
            cfgd["warn_avail_mb"] = cfgd["warn_avail_mb_min"]
            cfgd["danger_avail_mb"] = cfgd["danger_avail_mb_min"]
        # 危险阈值必须小于预警阈值
        if cfgd["danger_avail_mb"] >= cfgd["warn_avail_mb"]:
            cfgd["danger_avail_mb"] = max(64, int(cfgd["warn_avail_mb"] * 0.6))
        if cfgd["proc_danger_mb"] <= cfgd["proc_warn_mb"]:
            cfgd["proc_danger_mb"] = int(cfgd["proc_warn_mb"] * 1.5)

    # ------------------------------------------------------------ 测量 ----
    def _detect_total_mb(self):
        p = _psutil()
        if p is not None:
            try:
                return float(p.virtual_memory().total) / (1024.0 * 1024.0)
            except Exception:
                pass
        t, _a, _pr = _win_mem_via_ctypes()
        return t

    def raw(self):
        """返回 (total_mb, avail_mb, proc_mb, cpu_pct)。cpu_pct 取不到返回 -1。"""
        p = _psutil()
        if p is not None:
            try:
                vm = p.virtual_memory()
                total_mb = float(vm.total) / (1024.0 * 1024.0)
                avail_mb = float(vm.available) / (1024.0 * 1024.0)
                proc_mb = 0.0
                try:
                    proc_mb = float(p.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
                except Exception:
                    proc_mb = 0.0
                cpu = -1.0
                try:
                    cpu = float(p.cpu_percent(interval=None))
                except Exception:
                    cpu = -1.0
                return total_mb, avail_mb, proc_mb, cpu
            except Exception:
                pass
        t, a, pr = _win_mem_via_ctypes()
        return t, a, pr, -1.0

    def sample(self, force=False):
        """采样一次（节流：默认最短间隔 sample_interval_s）。返回最新快照 dict。"""
        with self._lock:
            now = time.time()
            if not force and (now - self._last_sample_t) < float(self.cfg["sample_interval_s"]):
                return dict(self._last)
            self._last_sample_t = now
            total_mb, avail_mb, proc_mb, cpu = self.raw()
            if total_mb <= 0:
                total_mb = self.total_mb
            level, note = self._judge(avail_mb, proc_mb=proc_mb)
            self._last = {"total_mb": total_mb, "avail_mb": avail_mb, "proc_mb": proc_mb,
                          "cpu_pct": cpu, "ts": now, "level": level, "note": note}
            self._hist.append((now, avail_mb, proc_mb, cpu))
            self._update_leak()
            if (now - self._last_log_t) >= float(self.cfg["log_interval_s"]):
                self._last_log_t = now
                self._append_log(level, note)
            return dict(self._last)

    def _judge(self, avail_mb, proc_mb=0.0):
        cfgd = self.cfg
        notes = []
        level = LEVEL_OK
        if avail_mb > 0:
            if avail_mb <= cfgd["danger_avail_mb"]:
                level = LEVEL_DANGER
                notes.append(f"系统可用内存 {avail_mb:.0f}MB ≤ 危险阈值 {cfgd['danger_avail_mb']}MB")
            elif avail_mb <= cfgd["warn_avail_mb"]:
                level = LEVEL_WARN
                notes.append(f"系统可用内存 {avail_mb:.0f}MB ≤ 预警阈值 {cfgd['warn_avail_mb']}MB")
        if proc_mb > 0:
            if proc_mb >= cfgd["proc_danger_mb"]:
                level = LEVEL_DANGER
                notes.append(f"本进程内存 {proc_mb:.0f}MB ≥ 危险阈值 {cfgd['proc_danger_mb']}MB")
            elif proc_mb >= cfgd["proc_warn_mb"] and level == LEVEL_OK:
                level = LEVEL_WARN
                notes.append(f"本进程内存 {proc_mb:.0f}MB ≥ 预警阈值 {cfgd['proc_warn_mb']}MB")
        if self._leak_flag:
            notes.append(self._leak_note)
        return level, "；".join(notes)

    def _update_leak(self):
        """连续采样判定“疑似内存泄漏”：窗口内进程内存持续上升且净增长超阈值。"""
        w = int(self.cfg["leak_window"])
        if len(self._hist) < w:
            self._leak_flag = False
            self._leak_note = ""
            return
        seg = list(self._hist)[-w:]
        procs = [s[2] for s in seg]
        if min(procs) <= 0:
            self._leak_flag = False
            self._leak_note = ""
            return
        growth = procs[-1] - procs[0]
        # 单调性：允许极小抖动（0.5%）
        monotonic = all(procs[i + 1] >= procs[i] * 0.995 for i in range(len(procs) - 1))
        if monotonic and growth >= float(self.cfg["leak_growth_mb"]):
            self._leak_flag = True
            self._leak_note = (f"疑似内存泄漏：最近 {w} 次采样进程内存从 {procs[0]:.0f}MB "
                               f"持续升到 {procs[-1]:.0f}MB（+{growth:.0f}MB），请查看 resource_log.csv")
        else:
            self._leak_flag = False
            self._leak_note = ""

    def _append_log(self, level, note):
        """增量追加资源日志（自动轮转，绝不无限增长）。"""
        try:
            path = os.path.join(self.data_dir, "resource_log.csv")
            try:
                if os.path.exists(path) and os.path.getsize(path) > int(self.cfg["log_max_bytes"]):
                    old = os.path.join(self.data_dir, "resource_log_old.csv")
                    try:
                        if os.path.exists(old):
                            os.remove(old)
                    except Exception:
                        pass
                    os.replace(path, old)
            except Exception:
                pass
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            s = self._last
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["时间", "系统总内存MB", "系统可用内存MB", "系统内存使用率%",
                                "进程内存MB", "CPU%", "级别", "说明"])
                used_pct = (100.0 * (s["total_mb"] - s["avail_mb"]) / s["total_mb"]
                            if s["total_mb"] else 0.0)
                cpu = "" if s["cpu_pct"] < 0 else f"{s['cpu_pct']:.1f}"
                w.writerow([_now_str(), f"{s['total_mb']:.0f}", f"{s['avail_mb']:.0f}",
                            f"{used_pct:.1f}", f"{s['proc_mb']:.0f}", cpu, level, note])
        except Exception:
            pass

    # ------------------------------------------------------------ 判定 ----
    def can_start_heavy(self):
        """能不能启动新的重型任务。返回 (bool, 原因)。"""
        s = self.sample(force=True)
        if s["level"] == LEVEL_DANGER:
            return False, "内存已达危险阈值：" + (s["note"] or "资源不足")
        if s["level"] == LEVEL_WARN:
            return False, "内存已达预警阈值：" + (s["note"] or "资源紧张")
        return True, "内存充足"

    def should_abort(self):
        """当前任务是否应在可控边界停止（危险阈值）。返回 (bool, 原因)。"""
        s = self.sample()
        if s["level"] == LEVEL_DANGER:
            return True, (s["note"] or "内存达危险阈值")
        return False, ""

    def wait_for_safe(self, max_wait_s=1800, poll_s=5.0, should_stop=None):
        """等待内存回到安全范围（且经过稳定性检查）。返回 (ok, 说明)。

        稳定性检查：连续 3 次采样都回到 ok 级别才认为恢复（避免刚好一时抖动就恢复）。
        """
        t0 = time.time()
        ok_streak = 0
        while time.time() - t0 < max_wait_s:
            if should_stop is not None and should_stop():
                return False, "等待恢复期间收到停止请求"
            s = self.sample(force=True)
            if s["level"] == LEVEL_OK:
                ok_streak += 1
                if ok_streak >= 3:
                    return True, "内存已恢复到安全范围"
            else:
                ok_streak = 0
            time.sleep(poll_s)
        return False, f"等待内存恢复超时（{max_wait_s}s）"

    # ------------------------------------------------------------ 展示 ----
    def status_text(self, sample=True):
        s = self.sample() if sample else dict(self._last)
        lvl = {"ok": "正常", "warn": "预警", "danger": "危险"}.get(s["level"], s["level"])
        cpu = "—" if s["cpu_pct"] < 0 else f"{s['cpu_pct']:.0f}%"
        txt = (f"内存：可用 {s['avail_mb']:.0f}MB / 总 {s['total_mb']:.0f}MB（{lvl}）"
               f" | 本进程 {s['proc_mb']:.0f}MB | CPU {cpu}")
        if s["note"]:
            txt += f" | {s['note']}"
        return txt

    def snapshot(self):
        return dict(self._last)

    # ------------------------------------------------------------ 学习闸门 ----
class HeavyTaskGate:
    """重型任务闸门：保证同一时刻只有一个重型训练任务在跑（4GB 老电脑硬性要求）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    def try_acquire(self, owner="heavy"):
        got = self._lock.acquire(blocking=False)
        if got:
            self._owner = owner
        return got

    def release(self):
        try:
            self._owner = None
            self._lock.release()
        except Exception:
            pass

    def is_busy(self):
        return self._lock.locked()

    def owner(self):
        return self._owner


GLOBAL_HEAVY_GATE = HeavyTaskGate()
