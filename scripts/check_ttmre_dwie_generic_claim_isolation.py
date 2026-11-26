# Isolated replacement suite; original approved G checker remains unchanged.
"""CPU-only G contracts; synthetic tensors and in-memory artifact mocks only.

No real generic/source weights, CUDA, dataset records, disk checkpoints,
training launch or unlink. Tiny optimizer steps validate the inherited target
training implementation, never establish DWIE quality or cloud admission.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import train_ttmre_dwie_generic_control as run
from check_ttmre_dwie_adaptation import TinyModel, InMemoryArtifacts, feature, document, REL2ID, METRIC_SOURCE
from check_ttmre_dwie_gpu_probe_v4 import TinyBackbone


def factory(_):
    return TinyModel(), {"synthetic": True}


def loader(_, config):
    model = TinyBackbone()
    with torch.no_grad():
        model.embeddings.word_embeddings.weight.fill_(.125)
    return model, {"synthetic_generic": True}


def tiny_generic():
    with patch.object(run.base, "NUM_CLASSES", 3):
        return run.build_generic({"model_config_dir": "__synthetic_generic__"}, torch.device("cpu"),
                                 model_factory=factory, backbone_loader=loader)


class Contracts(unittest.TestCase):
    def test_protocol_is_one_fixed_target_only_control(self):
        p = run.protocol()
        self.assertEqual((p["arms"], p["seed"], p["epochs_per_arm"], p["total_optimizer_updates"]), (["G"], 75, 30, 4530))
        self.assertEqual((p["logical_batch_documents"], p["microbatch_documents"], p["last_group_documents"]), (4, 1, 2))
        self.assertFalse(p["source_relation_training"])
        self.assertFalse(p["additional_benchmark"])
        self.assertNotIn("source_parents", p)

    def test_pending_refuses_before_any_mutation_or_asset_loading(self):
        auth = {"arms": ["G"], "seed": 75, "epochs": 30, "optimizer_updates": 4530,
                **{k: False for k in ("automatic_retry", "resume", "extension", "test_data_read", "new_seeds", "model_download", "deletion", "dependency_install")},
                "approved": False, "hard_seconds": None}
        with patch.object(run.base, "digest", return_value=run.AUTHORIZATION_SHA), \
             patch.object(run.base, "read_json", return_value=auth), \
             patch.object(Path, "mkdir", side_effect=AssertionError("mkdir before approval")), \
             patch.object(run, "validate_local", side_effect=AssertionError("assets before approval")):
            for method, args in ((run.admit, ()), (run.start, (Path("__not_created__"),)),
                                 (run.worker, (Path("__not_created__"),))):
                with self.assertRaises(PermissionError):
                    method(*args)

    def test_approved_budget_cannot_expand_scope(self):
        auth = {"arms": ["G"], "seed": 75, "epochs": 30, "optimizer_updates": 4530,
                **{k: False for k in ("automatic_retry", "resume", "extension", "test_data_read", "new_seeds", "model_download", "deletion", "dependency_install")},
                "approved": True, "hard_seconds": 5400, "user_confirmation": "synthetic approval"}
        with patch.object(run.base, "digest", return_value=run.AUTHORIZATION_SHA), patch.object(run.base, "read_json", return_value=auth):
            self.assertEqual(run.read_authorization(True)["hard_seconds"], 5400)
            auth["hard_seconds"] = 7200
            with self.assertRaises(PermissionError): run.read_authorization(True)
            auth["hard_seconds"] = 5400
            auth["optimizer_updates"] = 9060
            with self.assertRaises(ValueError): run.read_authorization(True)

    def test_single_checkpoint_disk_bound_and_shortfall(self):
        required = run.CHECKPOINT_BOUND + run.base.REPORT_MARGIN
        with patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=required - 1)):
            with self.assertRaisesRegex(RuntimeError, "shortfall=1"):
                run.disk_admission()
        with patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=required)):
            report = run.disk_admission()
        self.assertEqual(report["checkpoints_to_write"], 1)
        self.assertEqual(report["required_bytes"], 1990716847)

    def test_admission_disk_failure_precedes_generic_hash_and_CUDA(self):
        with patch.object(run, "CLAIM", SimpleNamespace(exists=lambda: False)), \
             patch.object(run, "read_authorization"), patch.object(run.base, "digest", return_value="fake"), \
             patch.dict(run.os.environ, {"TTMRE_DWIE_G_SELF_SHA256": "fake"}), \
             patch.object(run, "budget_from_environment", return_value={}), \
             patch.object(run, "validate_local", return_value=({}, [], [], [], {})), \
             patch.object(run, "disk_admission", side_effect=RuntimeError("synthetic disk")), \
             patch.object(run, "inventory", side_effect=AssertionError("hash after low disk")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("mkdir after low disk")):
            with self.assertRaisesRegex(RuntimeError, "synthetic disk"):
                run.admit()

    def test_admission_full_positive_path_is_read_only(self):
        budget = {"deadline_monotonic": time.monotonic() + 1000}
        with ExitStack() as stack:
            stack.enter_context(patch.object(run, "CLAIM", SimpleNamespace(exists=lambda: False)))
            stack.enter_context(patch.object(run, "read_authorization"))
            stack.enter_context(patch.object(run.base, "digest", return_value="fake"))
            stack.enter_context(patch.dict(run.os.environ, {"TTMRE_DWIE_G_SELF_SHA256": "fake"}))
            stack.enter_context(patch.object(run, "budget_from_environment", return_value=budget))
            stack.enter_context(patch.object(run, "validate_local", return_value=({}, [], [], [], {})))
            stack.enter_context(patch.object(run, "disk_admission", return_value={"sufficient": True}))
            stack.enter_context(patch.object(run, "inventory", return_value={"all_generic_assets_present": True, "weight_SHA_verified": True}))
            stack.enter_context(patch.object(run.runtime_fix, "runtime_identity", return_value={"audited": True}))
            stack.enter_context(patch.object(run.base, "host_memory_admission", return_value={"available": True}))
            stack.enter_context(patch.object(run.base.probe.gpu_preflight, "inspect_gpu", return_value={"passed": True}))
            stack.enter_context(patch.object(run.base, "check_budget"))
            stack.enter_context(patch.object(Path, "mkdir", side_effect=AssertionError("admission wrote directory")))
            result = run.admit()
        self.assertTrue(result["runtime"]["audited"])
        self.assertEqual(result["budget"], budget)

    def test_clock_includes_prior_admission_elapsed_time(self):
        start = time.monotonic() - 321
        with patch.object(run, "read_authorization", return_value={"hard_seconds": 5400}), \
             patch.dict(run.os.environ, {"TTMRE_DWIE_G_START_MONOTONIC": str(start),
                       "TTMRE_DWIE_G_BOOT_ID": "synthetic-boot", "TTMRE_DWIE_G_START_UTC": "synthetic-utc"}), \
             patch.object(run.base, "check_budget"):
            b = run.budget_from_environment()
        self.assertEqual(b["start_monotonic"], start)
        self.assertAlmostEqual(b["deadline_monotonic"] - time.monotonic(), 5079, delta=2)

    def test_invalid_future_clock_rejected(self):
        with patch.object(run, "read_authorization", return_value={"hard_seconds": 5400}), \
             patch.dict(run.os.environ, {"TTMRE_DWIE_G_START_MONOTONIC": str(time.monotonic() + 20),
                       "TTMRE_DWIE_G_BOOT_ID": "fake", "TTMRE_DWIE_G_START_UTC": "fake"}):
            with self.assertRaises(ValueError): run.budget_from_environment()

    def test_generic_loading_allows_only_unused_MLM_pooler_difference(self):
        info = {"missing_keys": ["pooler.dense.weight", "pooler.dense.bias"],
                "unexpected_keys": ["lm_head.dense.weight"], "mismatched_keys": [], "error_msgs": []}
        with patch("transformers.AutoModel.from_pretrained", return_value=(TinyBackbone(), info)) as load:
            _, report = run.default_backbone_loader(Path("__synthetic__"), SimpleNamespace())
        self.assertEqual(report["loading_info"], info)
        self.assertTrue(load.call_args.kwargs["local_files_only"])
        self.assertTrue(load.call_args.kwargs["use_safetensors"])

    def test_missing_real_encoder_parameter_is_not_silently_random(self):
        info = {"missing_keys": ["encoder.layer.0.attention.self.query.weight"], "unexpected_keys": [],
                "mismatched_keys": [], "error_msgs": []}
        with patch("transformers.AutoModel.from_pretrained", return_value=(TinyBackbone(), info)):
            with self.assertRaises(ValueError): run.default_backbone_loader(Path("__synthetic__"), SimpleNamespace())

    def test_cloud434_prefixed_unused_pooler_names_are_accepted(self):
        info = {"missing_keys": ["roberta.pooler.dense.weight", "roberta.pooler.dense.bias"],
                "unexpected_keys": ["lm_head.decoder.weight"], "mismatched_keys": [], "error_msgs": []}
        with patch("transformers.AutoModel.from_pretrained", return_value=(TinyBackbone(), info)):
            _, report = run.default_backbone_loader(Path("__synthetic__"), SimpleNamespace())
        self.assertEqual(report["loading_info"], info)
        self.assertEqual(report["unused_pooler_may_be_fresh"], sorted(info["missing_keys"]))

    def test_prefixed_real_encoder_missing_key_is_still_rejected(self):
        info = {"missing_keys": ["roberta.encoder.layer.0.attention.self.query.weight"],
                "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        with patch("transformers.AutoModel.from_pretrained", return_value=(TinyBackbone(), info)):
            with self.assertRaises(ValueError): run.default_backbone_loader(Path("__synthetic__"), SimpleNamespace())

    def test_unexpected_task_weight_is_rejected(self):
        info = {"missing_keys": [], "unexpected_keys": ["head_extractor.weight"], "mismatched_keys": [], "error_msgs": []}
        with patch("transformers.AutoModel.from_pretrained", return_value=(TinyBackbone(), info)):
            with self.assertRaises(ValueError): run.default_backbone_loader(Path("__synthetic__"), SimpleNamespace())

    def test_generic_construction_does_not_read_any_task_checkpoint(self):
        with patch.object(run.base, "_load_parent", side_effect=AssertionError("Distant tensors read")), \
             patch.object(torch, "load", side_effect=AssertionError("Task tensor load")):
            model, report = tiny_generic()
        self.assertTrue(report["all_task_parameters_fresh"])
        self.assertEqual(report["source_task_tensors_read"], 0)
        self.assertFalse(report["replacement"]["memory_participates_in_forward"])
        self.assertTrue(torch.equal(model.model.embeddings.word_embeddings.weight, torch.full_like(model.model.embeddings.word_embeddings.weight, .125)))

    def test_construction_preserves_callers_RNG_and_is_deterministic(self):
        before = torch.random.get_rng_state().clone()
        first, a = tiny_generic()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second, b = tiny_generic()
        self.assertEqual(a["fresh_task_state_sha256"], b["fresh_task_state_sha256"])
        self.assertTrue(torch.equal(first.head_extractor.weight, second.head_extractor.weight))

    def test_fresh_target_head_matches_prior_initialization_rule(self):
        model, report = tiny_generic()
        with patch.object(run.base, "NUM_CLASSES", 3):
            _, expected = run.base._target_head(model.bilinear.in_features)
        self.assertEqual(report["fresh_target_head_sha256"], expected)

    def test_backbone_loader_cannot_replace_task_parameters(self):
        captured = {}
        def capture(path):
            model, execution = factory(path)
            captured["model"] = model
            return model, execution
        def bad_loader(_, config):
            with torch.no_grad(): captured["model"].head_extractor.weight.add_(1)
            return loader(_, config)
        with patch.object(run.base, "NUM_CLASSES", 3):
            with self.assertRaisesRegex(RuntimeError, "task-specific"):
                run.build_generic({"model_config_dir": "fake"}, torch.device("cpu"), model_factory=capture, backbone_loader=bad_loader)

    def test_actual_tiny_G_train_save_and_replay_uses_inherited_functions(self):
        model, _ = tiny_generic()
        samples = [feature(2 + i % 2, "synthetic-G-" + str(i)) for i in range(6)]
        dev = [feature(3, "synthetic-G-dev")]
        dev[0]["gold"].append({"h": 2, "r": "unknown", "t": 8})
        artifacts = InMemoryArtifacts()
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "mkdir"))
            stack.enter_context(patch.object(run.base, "save_state_once", side_effect=artifacts.save))
            stack.enter_context(patch.object(run.base, "load_saved_state", side_effect=artifacts.load))
            stack.enter_context(patch.object(run.base, "write_json", side_effect=artifacts.write_json))
            stack.enter_context(patch.object(run.base, "digest", return_value="synthetic-hash"))
            result = run.base.train_arm(model, samples, dev, [document(dev[0])], REL2ID,
                torch.device("cpu"), Path("__G_memory_only__"), epochs=2, official_source=METRIC_SOURCE)
        self.assertTrue(result["passed"] and result["synthetic_config"] and result["strict_best_replay"])
        self.assertEqual(result["optimizer_steps"], 4)
        self.assertEqual(len(artifacts.saves), 1)
        self.assertEqual(result["best_dev"]["slices"]["train_unknown"]["fn"], 1)
        self.assertTrue(result["positive_lr_learning_witness"]["witness"]["changed"])

    def test_missing_cache_inventory_reports_zero_writes_and_no_tensor_load(self):
        with patch.object(Path, "is_file", return_value=False), patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=2**31)), \
             patch.object(torch, "load", side_effect=AssertionError("inventory read tensors")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("inventory created directory")):
            report = run.inventory()
        self.assertEqual(set(report["missing_files"]), set(run.GENERIC_FILES))
        self.assertEqual(report["files_written"], 0)
        self.assertFalse(report["training_started"])

    def test_canonical_output_claim_and_old_run_are_separate(self):
        self.assertNotEqual(run.RUN_ROOT, run.base.RUN_ROOT)
        self.assertNotEqual(run.CLAIM.name, run.base.CLAIM_NAME)
        self.assertEqual(run.RUN_ROOT.name, "2026-10-25-dwie-generic-control")

    def test_real_constructor_fresh63_head_shared_projector_backward(self):
        from transformers import RobertaConfig, RobertaModel
        config = RobertaConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, max_position_embeddings=514)
        def synthetic_generic(_, config):
            return RobertaModel(config), {"synthetic_random_encoder_only": True}
        with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
             patch.object(run.base, "_load_parent", side_effect=AssertionError("Distant tensor read")):
            model, report = run.build_generic({"model_config_dir": "__synthetic_config__"},
                torch.device("cpu"), model_factory=run.base.probe.build_model, backbone_loader=synthetic_generic)
        # Local torch2.11 random-head bytes differ from cloud2.7.1. Exercise the
        # real geometry and prior initialization rule here; production keeps
        # the fixed cloud head SHA check under the strict cloud version gate.
        relations = {"TH": 0, "r1": 1, "r2": 2, **{"r" + str(i): i for i in range(3, 63)}}
        logits, labels, _ = run.base.probe.feature_logits(model, feature(3), relations, torch.device("cpu"))
        loss = run.base.probe.atgl_loss(logits, labels)
        loss.backward()
        self.assertEqual(tuple(logits.shape), (9, 63))
        _, expected = run.base._target_head(768 * 64)
        self.assertEqual(report["fresh_target_head_sha256"], expected)
        self.assertGreater(model.head_extractor.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.tail_extractor.weight.grad)
        self.assertFalse(any(n.startswith("mu_encoder.") for n, p in model.named_parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_production_constructor_does_not_relax_cloud_head_SHA(self):
        with patch.object(run.base.probe, "build_model", side_effect=factory), \
             patch.object(run.base, "NUM_CLASSES", 3), \
             patch.object(run, "TARGET_HEAD_SHA", "deliberately-wrong-fixed-head"):
            with self.assertRaisesRegex(ValueError, "target head"):
                run.build_generic({"model_config_dir": "__synthetic__"}, torch.device("cpu"), backbone_loader=loader)

    def test_production_parameter_geometry_is_enforced(self):
        with patch.object(run.base, "NUM_CLASSES", 3):
            _, head_sha = run.base._target_head(8)
            with patch.object(run.base.probe, "build_model", side_effect=factory), \
                 patch.object(run, "TARGET_HEAD_SHA", head_sha):
                with self.assertRaisesRegex(ValueError, "parameter geometry"):
                    run.build_generic({"model_config_dir": "__synthetic__"}, torch.device("cpu"), backbone_loader=loader)

    def test_worker_positive_dispatch_and_callbacks_without_real_GPU_or_files(self):
        workspace = (run.RUN_ROOT / "generic75-launch-synthetic/workspace").resolve()
        budget = {"start_monotonic": time.monotonic(), "deadline_monotonic": time.monotonic() + 5400}
        claim = {"workspace": str(workspace), "budget_sha256": "synthetic-sha",
                 "authorization_sha256": run.AUTHORIZATION_SHA, "entry_sha256": "synthetic-sha"}
        writes, events = {}, []
        def fake_train(*args, **kwargs):
            kwargs["step_hook"](2, {"loss": 1., "adam_step": 2, "learning_rates_used": [.1], "witness": {"changed": True}})
            kwargs["epoch_hook"]({"epoch": 1, "epoch_seconds": 1.})
            return {"production_config": True, "optimizer_steps": 4530, "synthetic_fixture_only": True}
        with ExitStack() as stack:
            stack.enter_context(patch.object(run, "read_authorization"))
            stack.enter_context(patch.object(run.base, "require_lock", return_value=9))
            stack.enter_context(patch.object(run, "budget_from_environment", return_value=budget))
            stack.enter_context(patch.object(run.base, "read_json", side_effect=lambda p: claim if p == run.CLAIM else budget))
            stack.enter_context(patch.object(run.base, "digest", return_value="synthetic-sha"))
            stack.enter_context(patch.dict(run.os.environ, {"TTMRE_DWIE_G_SELF_SHA256": "synthetic-sha"}))
            stack.enter_context(patch.object(run, "run_contracts", return_value={"synthetic": True}))
            stack.enter_context(patch.object(run, "validate_local", return_value=({}, [], [], [], {})))
            stack.enter_context(patch.object(run, "inventory", return_value={"all_generic_assets_present": True, "weight_SHA_verified": True}))
            stack.enter_context(patch.object(run, "disk_admission", return_value={}))
            stack.enter_context(patch.object(run.runtime_fix, "runtime_identity", return_value={}))
            stack.enter_context(patch.object(run.base, "host_memory_admission", return_value={}))
            stack.enter_context(patch.object(run.base.probe.gpu_preflight, "inspect_gpu", return_value={"passed": True, "name": "syntheticGPU"}))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(torch.cuda, "current_device", return_value=0))
            stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="syntheticGPU")))
            stack.enter_context(patch.object(run, "build_generic", return_value=(TinyModel(), {})))
            stack.enter_context(patch.object(run.base, "resource_admission", return_value={"synthetic": True}))
            stack.enter_context(patch.object(run.base, "train_arm", side_effect=fake_train))
            stack.enter_context(patch.object(run, "reference_comparison", return_value={"synthetic": True}))
            stack.enter_context(patch.object(run.base, "check_budget"))
            stack.enter_context(patch.object(run.base, "write_json", side_effect=lambda p, v: writes.__setitem__(p.name, v)))
            stack.enter_context(patch.object(run.base, "event", side_effect=lambda p, name, **fields: events.append(name)))
            stack.enter_context(patch.object(run.base, "pack_reports", return_value={"sha256": "synthetic"}))
            stack.enter_context(patch("builtins.print"))
            stack.enter_context(patch.object(Path, "mkdir", side_effect=AssertionError("Worker contract wrote directory")))
            run.worker(workspace)
        self.assertTrue(writes["completion.json"]["passed"])
        self.assertIn("throughput_admission", events)
        self.assertIn("formal_training_progress", events)
        self.assertIn("DWIE_GENERIC_CONTROL_COMPLETE", events)
        self.assertFalse(torch.cuda.is_initialized())


    def test_two_admission_fixtures_ignore_existing_real_claim(self):
        original = run.CLAIM
        sentinel = SimpleNamespace(name="synthetic-claimed.json", exists=Mock(side_effect=AssertionError("Test touched real caller claim")))
        with patch.object(run, "CLAIM", sentinel):
            suite = unittest.TestSuite(Contracts(name) for name in (
                "test_admission_disk_failure_precedes_generic_hash_and_CUDA",
                "test_admission_full_positive_path_is_read_only"))
            result = unittest.TestResult()
            suite.run(result)
            self.assertTrue(result.wasSuccessful(), repr(result.errors) + repr(result.failures))
            sentinel.exists.assert_not_called()
        self.assertIs(run.CLAIM, original)

    def test_production_existing_claim_refusal_is_not_weakened(self):
        with patch.object(run, "read_authorization"), patch.object(run.base, "digest", return_value="fake"), \
             patch.dict(run.os.environ, {"TTMRE_DWIE_G_SELF_SHA256": "fake"}), \
             patch.object(run, "budget_from_environment", return_value={}), \
             patch.object(run, "CLAIM", SimpleNamespace(exists=lambda: True)), \
             patch.object(run, "disk_admission", side_effect=AssertionError("Guard bypassed")):
            with self.assertRaisesRegex(ValueError, "already claimed"):
                run.admit()


class RecoveryGuards(unittest.TestCase):
    def samples(self):
        import recover_ttmre_dwie_generic_contracts as recovery
        failure = {"passed": False, "optimizer_steps": 0,
            "error": "G CPU contracts failed before real generic tensor load", "error_type": "RuntimeError",
            "test_data_read": False, "automatic_retry": False, "resume": False, "extension": False}
        budget = {"start_monotonic": 100., "deadline_monotonic": 5500., "hard_seconds": 5400,
            "authorization_sha256": recovery.ORIGINAL_AUTH_SHA, "automatic_retry": False,
            "resume": False, "extension": False, "boot_id": "synthetic", "start_utc": "synthetic"}
        return recovery, failure, budget

    def test_exact_zero_update_original_budget_accepted(self):
        recovery, failure, budget = self.samples()
        recovery.verify_failure_budget(failure, budget)

    def test_any_actual_update_forbids_recovery(self):
        recovery, failure, budget = self.samples()
        with self.assertRaises(ValueError):
            recovery.verify_failure_budget({**failure, "optimizer_steps": 1}, budget)

    def test_extended_or_reset_budget_is_not_accepted(self):
        recovery, failure, budget = self.samples()
        with self.assertRaises(ValueError):
            recovery.verify_failure_budget(failure, {**budget, "deadline_monotonic": 5501.})
        with self.assertRaises(ValueError):
            recovery.verify_failure_budget(failure, {**budget, "hard_seconds": 7200})

    def test_different_original_authorization_rejected(self):
        recovery, failure, budget = self.samples()
        with self.assertRaises(ValueError):
            recovery.verify_failure_budget(failure, {**budget, "authorization_sha256": "different"})

    def test_pending_permission_refuses_before_source_or_write(self):
        recovery, _, _ = self.samples()
        with patch.object(run.base, "digest", return_value=recovery.PERMISSION_SHA), \
             patch.object(run.base, "read_json", return_value={"approved": False}), \
             patch.object(recovery, "verify_sources", side_effect=AssertionError("past pending permission")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("pending writes")):
            for method, args in ((recovery.admit, ()), (recovery.start, (Path("__not_created__"),)),
                                 (recovery.worker, (Path("__not_created__"),))):
                with self.assertRaises(PermissionError): method(*args)

    def test_restored_environment_keeps_original_clock(self):
        recovery, _, budget = self.samples()
        with patch.dict(run.os.environ, {}), patch.object(run.base, "check_budget"):
            recovery.restore_clock_environment(budget)
            self.assertEqual(run.budget_from_environment(), budget)

    def test_original_G_entry_bytes_are_preserved(self):
        recovery, _, _ = self.samples()
        import hashlib
        self.assertEqual(hashlib.sha256((run.ROOT / "scripts/train_ttmre_dwie_generic_control.py").read_bytes()).hexdigest(), recovery.ORIGINAL_ENTRY_SHA)

    def test_recovery_claim_keeps_original_budget_digest(self):
        recovery, _, budget = self.samples()
        context = {"budget": budget, "original_claim": {"budget_sha256": "original-digest"},
                   "old_workspace": Path("__old_read_only__"), "self_sha": "synthetic-source"}
        result = recovery.claim_record(context, Path("__new_synthetic__"))
        self.assertEqual(result["budget_sha256"], "original-digest")
        self.assertEqual(result["new_budget_seconds"], 0)
        self.assertEqual(result["entry_sha256"], recovery.ORIGINAL_ENTRY_SHA)


if __name__ == "__main__":
    unittest.main()
