"""CPU contracts for a proposed replacement G budget; no real launch or files."""
from pathlib import Path
from types import SimpleNamespace
import ast
import time
import unittest
from unittest.mock import patch

import torch
import train_ttmre_dwie_generic_budget_replacement as new
import train_ttmre_dwie_generic_control as old


class Contracts(unittest.TestCase):
    def test_complete_target_protocol_unchanged_and_failed_updates_visible(self):
        a, b = new.protocol(), old.protocol()
        for key in ("arms", "seed", "epochs_per_arm", "train_documents", "dev_documents",
                    "optimizer", "loss", "alpha", "precision", "input_protocol", "decoder", "selection"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a["total_optimizer_updates"], 4530)
        self.assertEqual(a["prior_failed_optimizer_updates"], 151)
        self.assertEqual(a["expected_total_G_updates_across_attempts"], 4681)

    def test_pending_budget_refuses_before_predecessor_or_writes(self):
        pending = {"arms": ["G"], "seed": 75, "epochs": 30, "optimizer_updates": 4530,
                   "approved": False, "hard_seconds": None, "user_confirmation": None,
                   **{k: False for k in ("automatic_retry", "resume", "extension", "test_data_read", "new_seeds", "model_download", "deletion", "dependency_install")}}
        with patch.object(new.base, "digest", return_value=new.AUTHORIZATION_SHA), \
             patch.object(new.base, "read_json", return_value=pending), \
             patch.object(new, "verify_budget_stop_predecessor", side_effect=AssertionError("Past pending budget")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("Pending created directory")):
            for method, args in ((new.admit, ()), (new.start, (Path("__not_created__"),)), (new.worker, (Path("__not_created__"),))):
                with self.assertRaises(PermissionError): method(*args)

    def test_pending_fixture_does_not_read_live_approved_authorization(self):
        with patch.object(new, "AUTHORIZATION", Path("__synthetic_already_approved__")), \
             patch.object(new.base, "read_json", side_effect=AssertionError("Read live authorization")):
            result = unittest.TestResult()
            Contracts("test_pending_budget_refuses_before_predecessor_or_writes").run(result)
        self.assertTrue(result.wasSuccessful(), repr(result.errors) + repr(result.failures))

    def test_exact_partial151_budgetstop_required(self):
        f = {"passed": False, "error_type": "BudgetStop", "optimizer_steps": 151,
             "error": "G measured throughput exceeds approved remaining total cap",
             "test_data_read": False, "automatic_retry": False, "resume": False, "extension": False}
        new.verify_budget_stop_report(f)
        for value in (0, 152):
            with self.assertRaises(ValueError): new.verify_budget_stop_report({**f, "optimizer_steps": value})

    def test_old_core_model_and_training_functions_preserved(self):
        functions = lambda path: {n.name: ast.dump(n, include_attributes=False)
                                 for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        a, b = functions(Path(new.__file__)), functions(Path(old.__file__))
        for name in ("build_generic", "default_backbone_loader", "task_state_digest", "disk_admission", "inventory", "reference_comparison"):
            self.assertEqual(a[name], b[name], name)
        self.assertIs(new.base.train_arm, old.base.train_arm)
        self.assertIs(new.base.resource_admission, old.base.resource_admission)

    def test_new_budget_and_claim_cannot_silently_reuse_old_authorization(self):
        self.assertNotEqual(new.AUTHORIZATION, old.AUTHORIZATION)
        self.assertNotEqual(new.AUTHORIZATION_SHA, old.AUTHORIZATION_SHA)
        self.assertNotEqual(new.CLAIM, old.CLAIM)
        start = time.monotonic() - 1
        with patch.object(new, "read_authorization", return_value={"hard_seconds": 5400}), \
             patch.dict(new.os.environ, {"TTMRE_DWIE_G_START_MONOTONIC": str(start),
                 "TTMRE_DWIE_G_BOOT_ID": "synthetic", "TTMRE_DWIE_G_START_UTC": "synthetic"}), \
             patch.object(new.base, "check_budget"):
            b = new.budget_from_environment()
        self.assertEqual(b["authorization_sha256"], new.AUTHORIZATION_SHA)
        self.assertAlmostEqual(b["deadline_monotonic"] - start, 5400)

    def test_actual_replacement_worker_dispatch_uses_inherited_spine(self):
        import check_ttmre_dwie_generic_claim_isolation as fixture
        with patch.object(fixture, "run", new), \
             patch.object(new, "verify_budget_stop_predecessor", return_value={"prior_failed_optimizer_updates": 151}):
            result = unittest.TestResult()
            fixture.Contracts("test_worker_positive_dispatch_and_callbacks_without_real_GPU_or_files").run(result)
        self.assertTrue(result.wasSuccessful(), repr(result.errors) + repr(result.failures))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
