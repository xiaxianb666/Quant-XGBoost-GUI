# -*- coding: utf-8 -*-
"""
test_test_usage_registry.py — 批次 4：最终测试使用登记与重复消费保护

不联网、不训练、不跑真实回测。覆盖：
  1. reserved 只是登记，不阻止首次最终测试；
  2. 同一完整测试区间重复 claim 被拒绝；
  3. completed / failed / running 状态在进程重开后仍然阻止重复消费；
  4. 手工训练与自主研究共用同一个 SQLite 登记；
  5. 不同测试区间互不阻塞（普通研究/回测可以继续）；
  6. AutoRunner 在最终测试执行前检查登记，重复时跳过并记录原因；
  7. 旧数据库没有 usage 记录时按“未使用”处理。

运行：
  python -m unittest discover -s tests -p "test_test_usage_registry.py" -v
"""

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

import stock_pool as sp          # noqa: E402
from auto_research import AutoRunner  # noqa: E402
import research_engine as re      # noqa: E402


class _WindowGui:
    def get_first_data_date(self, code):
        return None


class TestUsageRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_usage_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = os.path.join(self.tmp, "research_queue.db")
        self.reg = sp.ResearchRegistry(self.db)
        self.addCleanup(self.reg.close)

    def test_reserve_does_not_block_first_final_test(self):
        self.reg.reserve_test_window(
            "600000", "2023-01-01", "2023-12-31", exp_id="R1",
            trigger="自主研究", note="预留未评估")
        summary = self.reg.test_usage_summary("600000", "2023-01-01", "2023-12-31")
        self.assertEqual(summary["status"], "reserved")
        ok, rec = self.reg.claim_final_test(
            "600000", "2023-01-01", "2023-12-31", exp_id="M1", trigger="手动训练")
        self.assertTrue(ok)
        self.assertEqual(rec["status"], sp.TEST_USAGE_RUNNING)

    def test_duplicate_claim_and_completed_survive_reopen(self):
        ok, _ = self.reg.claim_final_test(
            "600000", "2023-01-01", "2023-12-31", exp_id="M1", trigger="手动训练")
        self.assertTrue(ok)
        ok2, rec2 = self.reg.claim_final_test(
            "600000", "2023-01-01", "2023-12-31", exp_id="M2", trigger="手动训练")
        self.assertFalse(ok2)
        self.assertEqual(rec2["status"], sp.TEST_USAGE_RUNNING)
        self.reg.mark_final_test(
            "600000", "2023-01-01", "2023-12-31", exp_id="M1",
            status=sp.TEST_USAGE_COMPLETED, note="done")
        self.reg.close()

        reg2 = sp.ResearchRegistry(self.db)
        self.addCleanup(reg2.close)
        summary = reg2.test_usage_summary("600000", "2023-01-01", "2023-12-31")
        self.assertEqual(summary["status"], "blocking")
        self.assertEqual(summary["record"]["status"], sp.TEST_USAGE_COMPLETED)
        ok3, _ = reg2.claim_final_test(
            "600000", "2023-01-01", "2023-12-31", exp_id="M3", trigger="手动训练")
        self.assertFalse(ok3)

    def test_failed_usage_blocks_after_restart(self):
        self.reg.claim_final_test("600001", "2024-01-01", "2024-12-31",
                                  exp_id="M1", trigger="手动训练")
        self.reg.mark_final_test("600001", "2024-01-01", "2024-12-31",
                                 exp_id="M1", status=sp.TEST_USAGE_FAILED,
                                 note="模拟中断")
        self.reg.close()
        reg2 = sp.ResearchRegistry(self.db)
        self.addCleanup(reg2.close)
        ok, rec = reg2.claim_final_test("600001", "2024-01-01", "2024-12-31",
                                        exp_id="M2", trigger="手动训练")
        self.assertFalse(ok)
        self.assertEqual(rec["status"], sp.TEST_USAGE_FAILED)

    def test_cross_entry_reservation_shares_registry(self):
        # 自主研究登记同一个 DB；手工训练的独立 ResearchRegistry 能看见
        self.reg.reserve_test_window("600002", "2023-01-01", "2023-12-31",
                                     exp_id="A1", trigger="自主研究")
        other = sp.ResearchRegistry(self.db)
        self.addCleanup(other.close)
        summary = other.test_usage_summary("600002", "2023-01-01", "2023-12-31")
        self.assertEqual(summary["status"], "reserved")
        ok, _ = other.claim_final_test("600002", "2023-01-01", "2023-12-31",
                                       exp_id="M1", trigger="手动训练")
        self.assertTrue(ok)

    def test_different_interval_same_code_is_window_locked(self):
        self.reg.claim_final_test("600003", "2023-01-01", "2023-12-31",
                                  exp_id="M1", trigger="手动训练")
        self.reg.mark_final_test("600003", "2023-01-01", "2023-12-31",
                                 exp_id="M1", status=sp.TEST_USAGE_COMPLETED)
        # 同一股票的保留测试窗口已经固定，不能通过改日期绕过
        ok, rec = self.reg.claim_final_test("600003", "2024-01-01", "2024-12-31",
                                            exp_id="M2", trigger="手动训练")
        self.assertFalse(ok)
        self.assertEqual(rec["status"], "window_locked")
        # 另一只股票有各自的固定窗口，不受影响
        ok2, _ = self.reg.claim_final_test("600006", "2024-01-01", "2024-12-31",
                                           exp_id="M3", trigger="手动训练")
        self.assertTrue(ok2)

    def test_overlapping_interval_is_blocked_even_if_not_identical(self):
        self.reg.claim_final_test("600007", "2023-01-01", "2023-12-31",
                                  exp_id="M1", trigger="手动训练")
        # 手工删除固定窗口模拟旧记录/历史数据，只留下消费记录
        self.reg._exec("DELETE FROM test_windows WHERE code=?", ("600007",))
        ok, rec = self.reg.claim_final_test("600007", "2023-06-01", "2024-06-30",
                                            exp_id="M2", trigger="手动训练")
        self.assertFalse(ok)
        self.assertIn(rec["status"], (sp.TEST_USAGE_RUNNING, sp.TEST_USAGE_COMPLETED))

    def test_reserved_overlap_does_not_block_normal_research(self):
        # reserved 不是消费；普通研究可继续，正式 claim 仍按固定窗口检查
        self.reg.reserve_test_window("600008", "2023-01-01", "2023-12-31",
                                     exp_id="A1", trigger="自主研究")
        ok, _ = self.reg.claim_final_test("600008", "2023-01-01", "2023-12-31",
                                          exp_id="M1", trigger="手动训练")
        self.assertTrue(ok)

    def test_reserve_does_not_downgrade_completed(self):
        self.reg.claim_final_test("600004", "2023-01-01", "2023-12-31",
                                  exp_id="M1", trigger="手动训练")
        self.reg.mark_final_test("600004", "2023-01-01", "2023-12-31",
                                 exp_id="M1", status=sp.TEST_USAGE_COMPLETED)
        ok, rec = self.reg.reserve_test_window(
            "600004", "2023-01-01", "2023-12-31", exp_id="A1", trigger="自主研究")
        self.assertFalse(ok)
        self.assertEqual(rec["status"], sp.TEST_USAGE_COMPLETED)

    def test_missing_old_record_is_treated_as_unused(self):
        self.assertEqual(
            self.reg.test_usage_summary("600005", "2025-01-01", "2025-12-31")["status"],
            "none")
        ok, _ = self.reg.claim_final_test("600005", "2025-01-01", "2025-12-31",
                                          exp_id="M1", trigger="手动训练")
        self.assertTrue(ok)

    def test_concurrent_claim_only_one_wins(self):
        import threading
        reg2 = sp.ResearchRegistry(self.db)
        self.addCleanup(reg2.close)
        barrier = threading.Barrier(2)
        results = []
        lock = threading.Lock()

        def worker(reg, exp):
            barrier.wait(timeout=5)
            ok, rec = reg.claim_final_test(
                "600010", "2023-01-01", "2023-12-31",
                exp_id=exp, trigger="手动训练")
            with lock:
                results.append((ok, rec))

        t1 = threading.Thread(target=worker, args=(self.reg, "T1"))
        t2 = threading.Thread(target=worker, args=(reg2, "T2"))
        t1.start(); t2.start(); t1.join(10); t2.join(10)
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(1 for ok, _ in results if ok), 1, results)
        self.assertTrue(any(not ok for ok, _ in results), results)

    def test_concurrent_overlapping_ranges_only_one_window(self):
        import threading
        reg2 = sp.ResearchRegistry(self.db)
        self.addCleanup(reg2.close)
        barrier = threading.Barrier(2)
        results = []

        def worker(reg, exp, start, end):
            barrier.wait(timeout=5)
            results.append(reg.claim_final_test(
                "600011", start, end, exp_id=exp, trigger="手动训练")[0])

        t1 = threading.Thread(target=worker, args=(self.reg, "T1", "2023-01-01", "2023-12-31"))
        t2 = threading.Thread(target=worker, args=(reg2, "T2", "2023-06-01", "2024-06-30"))
        t1.start(); t2.start(); t1.join(10); t2.join(10)
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(1 for ok in results if ok), 1, results)

    def test_reserved_window_is_fixed_across_config_changes(self):
        sched = re.AutonomousScheduler(
            self.reg, _WindowGui(), data_dir=self.tmp,
            cfg={"reserved_test_days": 60, "train_years": 5})
        first = sched._train_window("600009")
        self.assertIsNotNone(self.reg.get_test_window("600009"))
        sched.cfg["reserved_test_days"] = 10
        sched.cfg["reserved_test_start"] = "2035-01-01"
        second = sched._train_window("600009")
        self.assertEqual(first[2:], second[2:])
        win = self.reg.get_test_window("600009")
        self.assertEqual(win["test_start"], first[2])
        self.assertEqual(win["test_end"], first[3])


class TestAutoRunnerFinalTestGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_guard_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = os.path.join(self.tmp, "research_queue.db")

    def _runner(self):
        return AutoRunner(
            {"code": "600000", "test_start": "2023-01-01",
             "test_end": "2023-12-31", "trigger": "手动训练"},
            lambda *a, **k: None, state_dir=self.tmp)

    def test_duplicate_claim_is_rejected(self):
        r = self._runner()
        r.results = {"test_start": "2023-01-01", "test_end": "2023-12-31"}
        r.exp_id = "E1"
        self.assertTrue(r._claim_final_test_usage()[0])
        ok, reason = r._claim_final_test_usage()
        self.assertFalse(ok)
        self.assertIn("禁止重复消费", reason)
        r._close_test_registry()

    def test_s8_skips_when_interval_already_used(self):
        reg = sp.ResearchRegistry(self.db)
        reg.claim_final_test("600000", "2023-01-01", "2023-12-31",
                             exp_id="OLD", trigger="手动训练")
        reg.close()
        r = self._runner()
        r.results = {"test_start": "2023-01-01", "test_end": "2023-12-31"}
        r.exp_id = "E2"
        # 登记已占用时，_s8 必须在访问 matrix_all / 回测之前跳过
        r._s8_final_test(object())
        self.assertTrue(r.results.get("test_skipped"))
        self.assertIn("禁止重复消费", r.results.get("test_skip_reason", ""))
        r._close_test_registry()

    def test_pre_start_failure_does_not_consume_test_window(self):
        r = self._runner()
        r.results = {"test_start": "2023-01-01", "test_end": "2023-12-31"}
        r.exp_id = "E1"
        with self.assertRaises(Exception):
            # 没有 matrix_all / 有效测试行，失败发生在正式 claim 之前
            r._s8_final_test(object())
        self.assertIsNone(r._final_test_claim)
        reg = sp.ResearchRegistry(self.db)
        self.addCleanup(reg.close)
        self.assertEqual(
            reg.test_usage_summary("600000", "2023-01-01", "2023-12-31")["status"],
            "none")
        r._close_test_registry()

    def test_failed_mark_is_persisted(self):
        r = self._runner()
        r.results = {"test_start": "2023-01-01", "test_end": "2023-12-31"}
        r.exp_id = "E1"
        self.assertTrue(r._claim_final_test_usage()[0])
        r._mark_final_test_usage("failed", "模拟失败")
        r._close_test_registry()
        reg = sp.ResearchRegistry(self.db)
        self.addCleanup(reg.close)
        summary = reg.test_usage_summary("600000", "2023-01-01", "2023-12-31")
        self.assertEqual(summary["status"], "blocking")
        self.assertEqual(summary["record"]["status"], sp.TEST_USAGE_FAILED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
