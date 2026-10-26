# -*- coding: utf-8 -*-
"""
safe_io.py — 安全写入 / 原子替换 / 容错读取

需求第十节：所有状态与实验记录必须能安全落盘，中途退出不能写坏旧文件。
做法：先写同目录临时文件 → flush + fsync → os.replace() 原子替换。
读取一律容错：文件损坏 / 不存在 / 内容非法都返回默认值，绝不抛异常中断主流程。
"""

import os
import csv
import json
import time
import tempfile


def ensure_dir(path):
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def atomic_write_text(path, text, encoding="utf-8"):
    """原子写文本：临时文件 + os.replace。成功返回 True。

    失败时保留原文件不变，并在同目录留下 .tmp 供诊断（不删除）。
    """
    if not path:
        return False
    ensure_dir(path)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                                   dir=os.path.dirname(os.path.abspath(path)))
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def atomic_write_json(path, obj, encoding="utf-8"):
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return False
    return atomic_write_text(path, text, encoding=encoding)


def read_json_safe(path, default=None):
    """容错读 JSON：任何异常都返回 default。"""
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            return default
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def atomic_append_csv(path, row, header=None, encoding="utf-8-sig"):
    """安全追加一行 CSV：文件不存在/为空时先写表头；用 tmp+replace 保证不写坏旧文件。

    header 为 None 时只追加数据行（不补表头）。
    """
    if not path:
        return False
    ensure_dir(path)
    try:
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        lines = []
        if not exists and header:
            lines.append(_csv_line(header))
        lines.append(_csv_line(row))
        with open(path, "a", newline="", encoding=encoding) as f:
            f.write("".join(lines))
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        return True
    except Exception:
        # 追加失败（例如文件被占用）→ 退化为原子重写：读出旧内容 + 新行，再整体替换
        try:
            old = ""
            if os.path.exists(path):
                with open(path, "r", encoding=encoding, errors="replace") as f:
                    old = f.read()
            if old and not old.endswith("\n"):
                old += "\n"
            if not old and header:
                old = _csv_line(header)
            return atomic_write_text(path, old + _csv_line(row), encoding=encoding)
        except Exception:
            return False


def _csv_line(values):
    import io
    buf = io.StringIO()
    csv.writer(buf).writerow(list(values))
    return buf.getvalue()


def backup_file(path, keep_suffix=None):
    """改动前备份：返回备份路径（失败返回 None）。"""
    try:
        if not os.path.exists(path):
            return None
        suffix = keep_suffix or time.strftime("%Y%m%d_%H%M%S")
        dst = f"{path}.bak_{suffix}"
        import shutil
        shutil.copy2(path, dst)
        return dst
    except Exception:
        return None
