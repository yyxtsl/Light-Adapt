"""CPU regression tests for the failed runtime audit, never a cloud result."""
import hashlib
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import transformers

import ttmre_dwie_runtime_fix as fixed
import recover_ttmre_dwie_runtime_check as recovery


class Contracts(unittest.TestCase):
    def test_actual_installed_inherited_step_and_source_are_audited(self):
        import torch.optim.adam as adam
        import torch.optim.adamw as adamw
        self.assertFalse(hasattr(adamw, "_single_tensor_adamw"))
        result = fixed.adamw_implementation_identity()
        self.assertEqual(result["AdamW_single_tensor_implementation"], "torch.optim.adam._single_tensor_adam")
        self.assertEqual(result["AdamW_single_tensor_source_sha256"],
                         hashlib.sha256(inspect.getsource(adam._single_tensor_adam).encode()).hexdigest())
        self.assertTrue(result["AdamW_decoupled_weight_decay"])
        self.assertEqual(len(result["AdamW_implementation_source_sha256"]), 4)

    def test_successful_runtime_path_with_mocked_cloud_version_labels(self):
        # Labels are mocked; implementation source is from the installed local
        # torch. This proves the full control path, not an actual 2.7.1 run.
        with patch.object(torch, "__version__", "2.7.1+cu128"), \
             patch.object(np, "__version__", "1.24.3"), \
             patch.object(transformers, "__version__", "4.34.0"), \
             patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("Audit constructed optimizer")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("Audit queried CUDA")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("Audit initialized CUDA")):
            result = fixed.runtime_identity()
        self.assertEqual(result["torch"], "2.7.1+cu128")
        self.assertFalse(result["optimizer_created_for_audit"])
        self.assertEqual(result["optimizer_updates_for_audit"], 0)
        self.assertEqual(result["AdamW_step_module"], "torch.optim.adam")

    def test_wrong_torch_still_fails(self):
        with patch.object(torch, "__version__", "wrong"), self.assertRaisesRegex(RuntimeError, "no install/version fallback"):
            fixed.runtime_identity()

    def test_wrong_numpy_still_fails(self):
        with patch.object(torch, "__version__", "2.7.1+cu128"), patch.object(np, "__version__", "wrong"), \
             self.assertRaisesRegex(RuntimeError, "no install/version fallback"):
            fixed.runtime_identity()

    def test_missing_actual_scalar_helper_is_not_silently_skipped(self):
        import torch.optim.adam as adam
        with patch.object(adam, "_single_tensor_adam", None), \
             self.assertRaisesRegex(RuntimeError, "admission refuses"):
            fixed.adamw_implementation_identity()

    def test_unexpected_step_layout_is_not_silently_skipped(self):
        with patch.object(torch.optim.AdamW, "step", lambda self: None), \
             self.assertRaisesRegex(RuntimeError, "Unexpected AdamW step layout"):
            fixed.adamw_implementation_identity()

    def recovery_records(self):
        failure = {"error":"module 'torch.optim.adamw' has no attribute '_single_tensor_adamw'",
                   "error_type":"AttributeError", "completed_arms":{}, "test_data_read":False,
                   "progress":{"current_arm":None,"current_arm_optimizer_steps":0,"completed_optimizer_steps":0}}
        budget = {"start_monotonic":100., "deadline_monotonic":28900., "hard_seconds":28800,
                  "authorization_sha256":recovery.ORIGINAL_AUTH_SHA256,"resume":False,"extension":False}
        return failure, budget

    def test_exact_zero_update_failure_accepts_unchanged_budget(self):
        failure,budget = self.recovery_records()
        recovery.verify_failure_budget(failure,budget)
        self.assertEqual(budget["deadline_monotonic"],28900.)

    def test_any_formal_update_refuses_recovery(self):
        failure,budget = self.recovery_records()
        failure["progress"]["current_arm_optimizer_steps"] = 1
        with self.assertRaisesRegex(ValueError,"zero-update"):
            recovery.verify_failure_budget(failure,budget)

    def test_new_or_nonfinite_deadline_refuses_recovery(self):
        failure,budget = self.recovery_records()
        for deadline in (28901.,float('nan'),float('inf'),True):
            with self.subTest(deadline=deadline),self.assertRaisesRegex(ValueError,"budget changed"):
                recovery.verify_failure_budget(failure,{**budget,"deadline_monotonic":deadline})

    def test_changed_original_authorization_refuses_recovery(self):
        failure,budget = self.recovery_records()
        with self.assertRaisesRegex(ValueError,"budget changed"):
            recovery.verify_failure_budget(failure,{**budget,"authorization_sha256":"different"})

    def test_pending_recovery_refuses_before_lock_or_files(self):
        with patch.object(recovery.run,"digest",return_value=recovery.PERMISSION_SHA256), \
             patch.object(recovery.run,"read_json",return_value={"approved":False,"user_confirmation":None}), \
             patch.object(recovery.run,"require_lock",side_effect=AssertionError("Premature lock")), \
             patch.object(Path,"mkdir",side_effect=AssertionError("Premature mkdir")):
            with self.assertRaisesRegex(PermissionError,"not been explicitly confirmed"):
                recovery.start(Path("__synthetic_never_created__/workspace"))

    def test_recovery_authorization_cannot_add_budget(self):
        permission = {"approved":True,"user_confirmation":"synthetic only",
                      "original_launch":recovery.ORIGINAL_LAUNCH.relative_to(recovery.ROOT).as_posix(),
                      "original_hard_seconds":28800,"new_budget_seconds":0,"maximum_manual_recoveries":1,
                      "test_data_read":False,"automatic_retry":False,"extension":False,
                      "deletion":False,"dependency_install":False}
        with patch.object(recovery.run,"digest",return_value=recovery.PERMISSION_SHA256), \
             patch.object(recovery.run,"read_json",return_value=permission):
            self.assertEqual(recovery.read_permission(),permission)
            permission["new_budget_seconds"] = 1
            with self.assertRaisesRegex(ValueError,"scope/budget"):
                recovery.read_permission()


if __name__ == "__main__":
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contracts attempted CUDA")):
        unittest.main(verbosity=2)
