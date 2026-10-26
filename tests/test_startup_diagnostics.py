# -*- coding: utf-8 -*-
"""
test_startup_diagnostics.py — 批次 1B：pythonw 空 stdout/stderr 与诊断日志回归测试

覆盖：
  1. sys.stdout / sys.stderr 为 None 时，启动兜底会替换为写入项目内日志的流；
  2. print / Tkinter 回调异常 / sys.excepthook 都能落到独立日志文件；
  3. 日志路径不可写时，写日志函数返回 False 且不抛异常；
  4. 日志失败不会掩盖原始异常（原异常仍按原样抛出）；
  5. stock_pool 的 AkShare 诊断文件写入失败时，仍抛出原始 UniverseError 链；
  6. fetch_universe_rows 的 log_fn 自身失败时，不掩盖数据源原始异常。

不联网、不安装依赖、不启动 GUI 主循环、不训练。仅依赖标准库 + tkinter。
运行：
  python -m unittest discover -s tests -p "test_startup_diagnostics.py" -v
"""

import os
import io
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

import stock_pool as sp        # noqa: E402
import train_xgb_gui as gui    # noqa: E402


class StartupDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="diag_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.log_path = os.path.join(self.tmp, "startup.log")
        self.ak_path = os.path.join(self.tmp, "_akshare_traceback.txt")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _read(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _fake_akshare(self, message="synthetic upstream failure"):
        """构造离线假 AkShare：调用即抛异常，且带 cache_clear。"""
        fake = types.ModuleType("akshare")

        def boom():
            raise RuntimeError(message)

        boom.cache_clear = lambda: None
        fake.stock_info_a_code_name = boom
        return fake

    def _install_fake_akshare(self, fake):
        missing = object()
        old = sys.modules.get("akshare", missing)

        def restore():
            if old is missing:
                sys.modules.pop("akshare", None)
            else:
                sys.modules["akshare"] = old

        self.addCleanup(restore)
        sys.modules["akshare"] = fake

    # ------------------------------------------------------------------
    # 1) 空 stdout/stderr
    # ------------------------------------------------------------------
    def test_none_stdout_stderr_replaced_and_output_logged(self):
        old_out, old_err, old_hook = sys.stdout, sys.stderr, sys.excepthook

        def restore():
            sys.stdout, sys.stderr, sys.excepthook = old_out, old_err, old_hook

        self.addCleanup(restore)

        with mock.patch.object(gui, "_diag_log_paths", lambda: (self.log_path,)):
            sys.stdout = None
            sys.stderr = None
            replaced = gui._install_pythonw_diagnostics()

            self.assertIn("stdout", replaced)
            self.assertIn("stderr", replaced)
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)

            # 普通 print 不再因 stdout 为 None 抛 AttributeError
            print("hello from pythonw")
            sys.stderr.write("stderr line\n")

            # Tkinter 回调异常应写入日志文件
            try:
                raise ZeroDivisionError("tk callback boom")
            except ZeroDivisionError:
                gui._log_tk_exception(*sys.exc_info())

            # 启动信息走 _startup_log（stdout 已被替换）
            gui._startup_log("startup info")

        text = self._read(self.log_path)
        self.assertIn("hello from pythonw", text)
        self.assertIn("stderr line", text)
        self.assertIn("Tkinter 回调异常", text)
        self.assertIn("ZeroDivisionError", text)
        self.assertIn("startup info", text)
        self.assertIn("sys.stdout", text)  # 记录“已替换”的说明

    def test_startup_log_falls_back_when_stdout_is_none(self):
        old_out = sys.stdout
        self.addCleanup(setattr, sys, "stdout", old_out)
        sys.stdout = None
        with mock.patch.object(gui, "_diag_log_paths", lambda: (self.log_path,)):
            self.assertTrue(gui._startup_log("fallback line"))
        self.assertIn("fallback line", self._read(self.log_path))

    # ------------------------------------------------------------------
    # 2) 未捕获异常钩子
    # ------------------------------------------------------------------
    def test_excepthook_records_full_traceback(self):
        old_out, old_err, old_hook = sys.stdout, sys.stderr, sys.excepthook

        def restore():
            sys.stdout, sys.stderr, sys.excepthook = old_out, old_err, old_hook

        self.addCleanup(restore)

        with mock.patch.object(gui, "_diag_log_paths", lambda: (self.log_path,)):
            sys.stdout = None
            sys.stderr = None
            gui._install_pythonw_diagnostics()
            try:
                raise ValueError("uncaught-original-error")
            except ValueError:
                ok = sys.excepthook(*sys.exc_info())

        self.assertTrue(ok)
        text = self._read(self.log_path)
        self.assertIn("未捕获异常", text)
        self.assertIn("Traceback (most recent call last)", text)
        self.assertIn("ValueError: uncaught-original-error", text)

    # ------------------------------------------------------------------
    # 3) 日志写入失败：不抛异常、不掩盖原异常
    # ------------------------------------------------------------------
    def test_log_write_failure_returns_false_and_never_raises(self):
        bad_path = os.path.join(self.tmp, "missing_dir", "sub", "diag.log")

        with mock.patch.object(gui, "_diag_log_paths", lambda: (bad_path,)):
            self.assertFalse(gui._append_diag_log("cannot write"))
            self.assertFalse(gui._log_exception("cannot write exception"))
            # 替身流的 write 也不能抛异常
            stream = gui._DiagStream("STDOUT")
            self.assertEqual(stream.write("still-safe"), len("still-safe"))
            self.assertIsNone(stream.flush())
            self.assertFalse(stream.isatty())
            with self.assertRaises(OSError):
                stream.fileno()

            # 原始异常必须原样抛出，不能被日志故障替换
            with mock.patch.object(sys, "stderr", io.StringIO()):
                with self.assertRaisesRegex(ValueError, "original-error"):
                    try:
                        raise ValueError("original-error")
                    except ValueError:
                        gui._log_uncaught_exception(*sys.exc_info())
                        raise

    def test_failed_log_does_not_change_startup_log_return(self):
        bad_path = os.path.join(self.tmp, "missing_dir", "sub", "diag.log")
        old_out, old_err, old_hook = sys.stdout, sys.stderr, sys.excepthook

        def restore():
            sys.stdout, sys.stderr, sys.excepthook = old_out, old_err, old_hook

        self.addCleanup(restore)
        sys.stdout = None
        with mock.patch.object(gui, "_diag_log_paths", lambda: (bad_path,)):
            self.assertFalse(gui._startup_log("dropped"))
            # stdout 为 None + 日志失败：安装兜底本身不应抛异常
            gui._install_pythonw_diagnostics()
            self.assertIsNotNone(sys.stdout)

    # ------------------------------------------------------------------
    # 4) stock_pool：AkShare 诊断文件
    # ------------------------------------------------------------------
    def test_akshare_diagnostic_written_without_network(self):
        fake = self._fake_akshare()
        self._install_fake_akshare(fake)
        with mock.patch.object(sp, "AKSHARE_TRACE_FILE", self.ak_path):
            with self.assertRaises(sp.UniverseError) as ctx:
                sp.default_universe_fetcher()

        self.assertIn("RuntimeError", str(ctx.exception))
        self.assertIn("synthetic upstream failure", str(ctx.exception))
        text = self._read(self.ak_path)
        self.assertIn("Traceback (most recent call last)", text)
        self.assertIn("RuntimeError: synthetic upstream failure", text)
        # 保留不确定性：测试只断言“记录到了调用处看到的异常”，
        # 不断言 AkShare 内部就是根因。
        self.assertIn("=====", text)

    def test_akshare_diagnostic_write_failure_does_not_mask_original(self):
        fake = self._fake_akshare("feed exploded")
        self._install_fake_akshare(fake)
        bad_path = os.path.join(self.tmp, "missing_dir", "sub", "_akshare_traceback.txt")
        with mock.patch.object(sp, "AKSHARE_TRACE_FILE", bad_path):
            with self.assertRaises(sp.UniverseError) as ctx:
                sp.default_universe_fetcher()

        self.assertIn("RuntimeError", str(ctx.exception))
        self.assertIn("feed exploded", str(ctx.exception))
        self.assertFalse(os.path.exists(bad_path))

    def test_akshare_diagnostic_helper_returns_false_on_bad_path(self):
        bad_path = os.path.join(self.tmp, "missing_dir", "sub", "diag.txt")
        with mock.patch.object(sp, "AKSHARE_TRACE_FILE", bad_path):
            try:
                raise RuntimeError("diagnostic-only")
            except RuntimeError as e:
                self.assertFalse(sp._write_akshare_diagnostic(e))

    # ------------------------------------------------------------------
    # 5) fetch_universe_rows：log_fn 失败不影响原始错误
    # ------------------------------------------------------------------
    def test_fetch_universe_rows_log_failure_preserves_error(self):
        def fail_fetch():
            raise RuntimeError("data source down")

        def broken_log(_msg):
            raise IOError("log sink broken")

        with self.assertRaises(sp.UniverseError) as ctx:
            sp.fetch_universe_rows(fetch_fn=fail_fetch, retries=1, log_fn=broken_log)

        self.assertIn("RuntimeError", str(ctx.exception))
        self.assertIn("data source down", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
