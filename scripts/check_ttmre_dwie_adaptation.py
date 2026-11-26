"""CPU synthetic contracts for DWIE adaptation, never formal training evidence.

Tiny random models use three synthetic output classes instead of the production
63. Real AdamW steps and in-memory serialization test implementation behavior;
no corpus rows, real parent tensors, CUDA, downloads, or disk checkpoints are
used. Existing pinned metric *source code* may be read for metric-unit tests.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import io
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import torch
from torch import nn

import train_ttmre_dwie_adaptation as run
import ttmre_dwie_input as dwie
import ttmre_dwie_gpu_probe_v4 as probe
import ttmre_pair_replacement as replacement
from check_ttmre_dwie_gpu_probe_v4 import TinyModel as ProbeTinyModel


DEVICE = torch.device("cpu")
REL2ID = {"TH": 0, "r1": 1, "r2": 2}
METRIC_SOURCE = (Path(__file__).resolve().parents[1] /
                 "refine-logs/SECOND_DATASET_ADMISSION_20261023/sources/"
                 "dwie_official_evaluator.py.txt")


class TinyModel(ProbeTinyModel):
    """Real tensor path plus intentionally disconnected writer/tail weights."""
    def __init__(self):
        super().__init__()
        self.unused_writer = nn.Linear(4, 4)
        self.unused_tail = nn.Linear(4, 4)


def feature(entities=3, ident="synthetic-train"):
    if entities < 1 or entities > 4:
        raise ValueError("Synthetic fixture supports 1..4 entities")
    ids = [2 + 3 * j for j in range(entities)]
    positions = [[[2*j, 2*j+1]] for j in range(entities-1)] + [[]]
    return {"id": ident, "inputs": {
        "content_ids": [3, 4, 5, 6, 7, 8, 9, 10],
        "entity_ids": ids, "entity_pos": positions,
        "candidate_rule": "cartesian_including_self"},
        "gold": [{"h": ids[0], "r": "r1", "t": ids[0]},
                 {"h": ids[0], "r": "r2", "t": ids[-1]}]}


def document(sample):
    return {"id": sample["id"], "entities": [
        {"id": ident, "mentions": [{"begin": a, "end": b} for a, b in positions]}
        for ident, positions in zip(sample["inputs"]["entity_ids"],
                                    sample["inputs"]["entity_pos"])],
        "relations": copy.deepcopy(sample["gold"])}


def make_model(kind="C", dtype=torch.float32):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(750024)
        model = TinyModel()
    if kind == "C":
        model.mu_encoder.memory_tokens.requires_grad_(False)
    elif kind in ("B", "D"):
        replacement.install(model, "bypass")
    else:
        raise ValueError("Synthetic unknown arm")
    return model.to(dtype=dtype)


class InMemoryArtifacts:
    """Exclusive torch serialization without filesystem writes or cleanup."""
    def __init__(self):
        self.states = {}
        self.json = {}
        self.saves = []

    def save(self, path, value, *args, **kwargs):
        key = str(path)
        if key in self.states:
            raise FileExistsError(key)
        stream = io.BytesIO()
        torch.save(value, stream, *args, **kwargs)
        self.states[key] = stream.getvalue()
        self.saves.append(key)

    def load(self, path, *args, **kwargs):
        return torch.load(io.BytesIO(self.states[str(path)]), *args, **kwargs)

    def write_json(self, path, value):
        key = str(path)
        if key in self.json:
            raise FileExistsError(key)
        self.json[key] = copy.deepcopy(value)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_gradients_equal(self, first, second):
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
                first.named_parameters(), second.named_parameters()):
            self.assertEqual(name_a, name_b)
            if parameter_a.grad is None or parameter_b.grad is None:
                self.assertIs(parameter_a.grad, parameter_b.grad, name_a)
            else:
                torch.testing.assert_close(parameter_a.grad, parameter_b.grad,
                                           atol=2e-10, rtol=2e-9, msg=name_a)

    def check_accumulation(self, counts):
        first = make_model(dtype=torch.float64)
        second = copy.deepcopy(first)
        samples = [feature(n, "pair-weight-"+str(j)) for j, n in enumerate(counts)]
        actual = run.accumulate_group(first, samples, REL2ID, DEVICE)
        logits, labels = [], []
        for sample in samples:
            output, target, _ = probe.feature_logits(second, sample, REL2ID, DEVICE)
            logits.append(output)
            labels.append(target)
        expected = probe.atgl_loss(torch.cat(logits), torch.cat(labels))
        expected.backward()
        self.assert_gradients_equal(first, second)
        self.assertEqual(actual["pair_counts"], [n*n for n in counts])
        self.assertEqual(actual["total_pairs"], sum(n*n for n in counts))
        self.assertAlmostEqual(actual["loss"], expected.item(), places=10)

    def test_four_unequal_documents_accumulate_to_pair_mean(self):
        self.check_accumulation([2, 3, 4, 2])

    def test_final_two_document_group_uses_actual_pair_denominator(self):
        self.check_accumulation([2, 4])

    def test_602_documents_once_per_epoch_and_4530_steps(self):
        groups = []
        for epoch in range(30):
            one = run.epoch_groups(602, epoch)
            self.assertEqual(len(one), 151)
            self.assertEqual([len(group) for group in one], [4]*150+[2])
            self.assertEqual(sorted(index for group in one for index in group), list(range(602)))
            groups.extend(one)
        self.assertEqual(len(groups), 4530)

    def test_document_order_rng_is_independent_of_model_rng(self):
        expected = run.epoch_groups(602, 7)
        cpu_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        run.epoch_groups(602, 7)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
        self.assertEqual(random.getstate(), python_state)
        random.seed(123)
        torch.manual_seed(123)
        _ = torch.randn(100)
        _ = [random.random() for _ in range(80)]
        self.assertEqual(run.epoch_groups(602, 7), expected)
        self.assertNotEqual(run.epoch_groups(602, 8), expected)

    def test_one_clip_optimizer_and_scheduler_step_per_logical_group(self):
        model = make_model()
        optimizer = run.optimizer_for_model(model)
        scheduler = run.make_scheduler(optimizer, total_steps=4, warmup=1)
        clip = torch.nn.utils.clip_grad_norm_
        with patch.object(torch.nn.utils, "clip_grad_norm_", wraps=clip) as clipping, \
             patch.object(optimizer, "step", wraps=optimizer.step) as stepping, \
             patch.object(scheduler, "step", wraps=scheduler.step) as scheduling:
            row = run.train_step(model, [feature(2, "one"), feature(4, "two")],
                                 REL2ID, DEVICE, optimizer, scheduler)
        self.assertEqual((clipping.call_count, stepping.call_count, scheduling.call_count), (1, 1, 1))
        self.assertEqual(row["documents"], ["one", "two"])
        self.assertEqual(row["total_pairs"], 20)
        self.assertFalse(optimizer.defaults["foreach"])
        self.assertFalse(optimizer.defaults["fused"])

    def test_disconnected_writer_and_tail_stay_none_and_do_not_decay(self):
        model = make_model()
        optimizer = run.optimizer_for_model(model)
        scheduler = run.make_scheduler(optimizer, total_steps=3, warmup=0)
        before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()
                  if name.startswith("unused_")}
        for _ in range(2):
            run.train_step(model, [feature(3)], REL2ID, DEVICE, optimizer, scheduler)
        for name, parameter in model.named_parameters():
            if name in before:
                self.assertIsNone(parameter.grad)
                self.assertNotIn(parameter, optimizer.state)
                self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_C_memory_is_frozen_and_B_memory_buffer_has_no_forward_effect(self):
        c_model = make_model("C")
        original = c_model.mu_encoder.memory_tokens.detach().clone()
        optimizer = run.optimizer_for_model(c_model)
        scheduler = run.make_scheduler(optimizer, total_steps=2, warmup=0)
        run.train_step(c_model, [feature(3)], REL2ID, DEVICE, optimizer, scheduler)
        self.assertIsNone(c_model.mu_encoder.memory_tokens.grad)
        self.assertTrue(torch.equal(c_model.mu_encoder.memory_tokens, original))
        b_model = make_model("B")
        self.assertNotIn("mu_encoder.memory_tokens", dict(b_model.named_parameters()))
        self.assertIn("mu_encoder.memory_tokens", dict(b_model.named_buffers()))
        first, _, _ = probe.feature_logits(b_model, feature(3), REL2ID, DEVICE)
        b_model.mu_encoder.memory_tokens.add_(1000)
        second, _, _ = probe.feature_logits(b_model, feature(3), REL2ID, DEVICE)
        self.assertTrue(torch.equal(first, second))

    def test_positive_LR_step_changes_active_weights_after_zero_LR_step(self):
        model = make_model("B")
        optimizer = run.optimizer_for_model(model)
        scheduler = run.make_scheduler(optimizer, total_steps=4, warmup=1)
        parameter = model.head_extractor.weight
        before = parameter.detach().clone()
        run.train_step(model, [feature(3)], REL2ID, DEVICE, optimizer, scheduler)
        self.assertTrue(torch.equal(parameter, before))
        self.assertEqual(int(optimizer.state[parameter]["step"]), 1)
        self.assertGreater(optimizer.param_groups[0]["lr"], 0)
        run.train_step(model, [feature(3)], REL2ID, DEVICE, optimizer, scheduler)
        self.assertFalse(torch.equal(parameter, before))
        self.assertEqual(int(optimizer.state[parameter]["step"]), 2)

    def test_nonfinite_loss_stops_before_any_optimizer_update(self):
        model = make_model()
        optimizer = run.optimizer_for_model(model)
        scheduler = run.make_scheduler(optimizer, total_steps=2, warmup=0)
        def nonfinite(current, sample, rel2id, device):
            logits, labels, audit = probe.feature_logits(current, sample, rel2id, device)
            return logits*float("nan"), labels, audit
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with patch.object(optimizer, "step", wraps=optimizer.step) as stepping:
            with self.assertRaises((ValueError, RuntimeError, FloatingPointError)):
                run.train_step(model, [feature(3)], REL2ID, DEVICE, optimizer,
                               scheduler, logits_fn=nonfinite)
        stepping.assert_not_called()
        self.assertFalse(optimizer.state)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))

    def test_finite_loss_with_nonfinite_gradient_stops_before_step(self):
        model = make_model()
        optimizer = run.optimizer_for_model(model)
        scheduler = run.make_scheduler(optimizer, total_steps=2, warmup=0)
        def overflow_gradient(current, sample, rel2id, device):
            logits, labels, audit = probe.feature_logits(current, sample, rel2id, device)
            logits.register_hook(lambda grad: torch.full_like(grad, float("inf")))
            return logits, labels, audit
        with patch.object(optimizer, "step", wraps=optimizer.step) as stepping:
            with self.assertRaises((ValueError, RuntimeError, FloatingPointError)):
                run.train_step(model, [feature(3)], REL2ID, DEVICE, optimizer,
                               scheduler, logits_fn=overflow_gradient)
        stepping.assert_not_called()
        self.assertFalse(optimizer.state)

    def test_train_unknown_gold_fails_instead_of_becoming_TH(self):
        sample = feature(3)
        sample["gold"].append({"h":2, "r":"unknown", "t":8})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            run.accumulate_group(make_model(), [sample], REL2ID, DEVICE)

    def test_zero_pair_schema_and_expired_group_budget_stop(self):
        sample = feature(2)
        sample["inputs"]["entity_ids"] = []
        sample["inputs"]["entity_pos"] = []
        sample["gold"] = []
        with self.assertRaises((ValueError, RuntimeError)):
            run.accumulate_group(make_model(), [sample], REL2ID, DEVICE)
        model = make_model()
        optimizer = run.optimizer_for_model(model)
        scheduler = run.make_scheduler(optimizer, total_steps=2, warmup=0)
        def expired():
            raise TimeoutError("fixed synthetic deadline")
        with patch.object(optimizer, "step", wraps=optimizer.step) as stepping:
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                run.train_step(model, [feature(3)], REL2ID, DEVICE, optimizer,
                               scheduler, check_fn=expired)
        stepping.assert_not_called()

    def test_active_gradient_reservation_never_allocates_disconnected_gradients(self):
        model = make_model()
        run.accumulate_group(model, [feature(3)], REL2ID, DEVICE)
        connected = {name for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        pointers = {name: parameter.grad.data_ptr() for name, parameter in model.named_parameters()
                    if parameter.grad is not None}
        active = run.reserve_active_gradients(model)
        self.assertEqual({name for name, _ in active}, connected)
        for name, parameter in model.named_parameters():
            if name in connected:
                self.assertEqual(parameter.grad.data_ptr(), pointers[name])
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
            else:
                self.assertIsNone(parameter.grad)
        self.assertFalse(any(name.startswith("unused_") for name, _ in active))

    def test_each_arm_reloads_correct_parent_and_identical_fresh_head(self):
        # Manufacture source layouts from tiny random tensors, never load a
        # real parent checkpoint or claim its SHA has been verified here.
        source_c = TinyModel()
        probe.set_classifier(source_c, 97)
        source_d = copy.deepcopy(source_c)
        replacement.install(source_d, "bypass")
        with torch.no_grad():
            source_d.head_extractor.weight.add_(.25)
        states = {probe.C_SHA256: copy.deepcopy(source_c.state_dict()),
                  probe.D_SHA256: copy.deepcopy(source_d.state_dict())}
        identity = {"model_config_dir":"synthetic-no-files",
                    "parent_paths":{"C_B":"synthetic-C", "D":"synthetic-D"}}
        loaded = []
        def factory(config_dir):
            self.assertEqual(config_dir, "synthetic-no-files")
            model = TinyModel()
            probe.set_classifier(model, 97)
            return model, {"synthetic_only":True}
        def loader(model, path, expected_sha, check_fn, source_name):
            self.assertEqual(model.bilinear.out_features, 97)
            loaded.append(expected_sha)
            result = probe.strict_load(model, states[expected_sha], "synthetic literal state")
            return {**result, "checkpoint_sha256":expected_sha, "synthetic_literal_state":True}
        models = {}
        with patch.object(run, "NUM_CLASSES", 3):
            models["C"], _ = run.build_arm("C", identity, DEVICE,
                                           model_factory=factory, state_loader=loader)
            head = copy.deepcopy(models["C"].bilinear.state_dict())
            with torch.no_grad():
                models["C"].head_extractor.weight.add_(100)
            models["B"], _ = run.build_arm("B", identity, DEVICE,
                                           model_factory=factory, state_loader=loader)
            models["D"], _ = run.build_arm("D", identity, DEVICE,
                                           model_factory=factory, state_loader=loader)
        self.assertEqual(loaded, [probe.C_SHA256, probe.C_SHA256, probe.D_SHA256])
        self.assertTrue(torch.equal(models["B"].head_extractor.weight,
                                    states[probe.C_SHA256]["head_extractor.weight"]))
        self.assertTrue(torch.equal(models["D"].head_extractor.weight,
                                    states[probe.D_SHA256]["head_extractor.weight"]))
        self.assertFalse(models["C"].mu_encoder.memory_tokens.requires_grad)
        for arm in ("C", "B", "D"):
            self.assertEqual(models[arm].bilinear.out_features, 3)
            self.assertTrue(all(torch.equal(value, head[key])
                                for key, value in models[arm].bilinear.state_dict().items()))
        self.assertNotEqual(id(models["C"].model), id(models["B"].model))

    def test_zero_update_admission_restores_parameters_all_RNG_and_training_mode(self):
        import numpy as np
        model = make_model("C")
        model.eval()
        samples = [feature(n, "admission-"+str(j)) for j, n in enumerate([2,3,4,2,3])]
        state = copy.deepcopy(model.state_dict())
        cpu_rng, py_rng, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        def random_logits(current, sample, rel2id, device):
            # Deliberately consume all three RNG streams inside admission.
            random.random()
            np.random.random()
            torch.rand(1)
            return probe.feature_logits(current, sample, rel2id, device)
        with patch.object(torch.optim, "AdamW", side_effect=AssertionError("Admission created optimizer")):
            report = run.resource_admission(model, samples, REL2ID, DEVICE,
                                            logits_fn=random_logits)
        self.assertTrue(report["passed"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertTrue(report["RNG_restored"])
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(value, state[key]) for key, value in model.state_dict().items()))
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertEqual(random.getstate(), py_rng)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], np_rng[0])
        self.assertTrue(np.array_equal(after_numpy[1], np_rng[1]))
        self.assertEqual(after_numpy[2:], np_rng[2:])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(any(name.startswith("unused_") for name in report["active_parameter_names"]))
        self.assertGreater(report["reserved_moment_bytes"], 0)
        self.assertGreater(report["reserved_scratch_bytes"], 0)
        self.assertTrue(report["representative_logical_groups"])

    def test_admission_allocation_failure_restores_state_without_optimizer(self):
        model = make_model("B")
        model.eval()
        before = copy.deepcopy(model.state_dict())
        cpu_rng, py_rng = torch.get_rng_state().clone(), random.getstate()
        with patch.object(torch, "empty_like", side_effect=MemoryError("synthetic moment reservation")), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Admission created optimizer")):
            with self.assertRaisesRegex(MemoryError, "reservation"):
                run.resource_admission(model, [feature(3)], REL2ID, DEVICE)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertEqual(random.getstate(), py_rng)

    def test_real_tiny_C_B_D_training_best_earliest_tie_unique_save_strict_replay(self):
        samples = [feature(n, "train-"+str(j)) for j, n in enumerate([2,3,4,2,3,4,2,3])]
        dev = [feature(3, "dev-a"), feature(4, "dev-b")]
        dev[0]["gold"].append({"h":2,"r":"unknown","t":8})
        docs = [document(sample) for sample in dev]
        originals = copy.deepcopy(samples + dev)
        artifacts, results = InMemoryArtifacts(), {}
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "mkdir"))
            stack.enter_context(patch.object(run, "save_state_once", side_effect=artifacts.save))
            stack.enter_context(patch.object(run, "load_saved_state", side_effect=artifacts.load))
            stack.enter_context(patch.object(run, "write_json", side_effect=artifacts.write_json))
            stack.enter_context(patch.object(run, "digest", side_effect=lambda path, check_fn=None:
                              hashlib.sha256(artifacts.states[str(path)]).hexdigest()))
            for arm in ("C", "B", "D"):
                model = make_model(arm)
                # A large deterministic margin makes both epoch predictions tie,
                # while real tensor losses/gradients/Adam updates still execute.
                with torch.no_grad():
                    model.bilinear.weight.zero_()
                    model.bilinear.bias.copy_(torch.tensor([-10.,10.,-10.]))
                memory = model.mu_encoder.memory_tokens.detach().clone()
                rows = []
                result = run.train_arm(model, samples, dev, docs, REL2ID, DEVICE,
                                       Path("__dwie_synthetic_no_disk__") / arm,
                                       epochs=2, epoch_hook=rows.append,
                                       official_source=METRIC_SOURCE)
                results[arm] = result
                self.assertTrue(result["passed"])
                self.assertTrue(result["synthetic_config"])
                self.assertFalse(result["production_config"])
                self.assertEqual((result["optimizer_steps"], result["complete_epochs"]), (4, 2))
                self.assertEqual(result["best_epoch"], 1)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["dev"], rows[1]["dev"])
                self.assertTrue(result["strict_best_replay"])
                self.assertEqual(result["checkpoint_saves"], 1)
                self.assertTrue(result["positive_lr_learning_witness"]["witness"]["changed"])
                self.assertEqual(result["positive_lr_learning_witness"]["adam_step"], 1)
                self.assertTrue(result["frozen_memory_unchanged"])
                self.assertFalse(result["test_data_read"])
                self.assertEqual(result["best_dev"]["slices"]["train_unknown"]["fn"], 1)
                self.assertTrue(torch.equal(model.mu_encoder.memory_tokens, memory))
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(samples + dev, originals)
        self.assertEqual(len(artifacts.saves), 3)
        self.assertEqual(len(set(artifacts.saves)), 3)
        self.assertEqual(len({result["sample_order_sha256"] for result in results.values()}), 1)
        self.assertEqual(len({result["initial_cpu_rng_sha256"] for result in results.values()}), 1)

    def test_prediction_clears_gold_and_preserves_all_input_pairs(self):
        sample = feature(3, "synthetic-dev")
        sample["gold"].append({"h":2, "r":"unknown", "t":8})
        original = copy.deepcopy(sample)
        seen = []
        def checked(current, forwarded, rel2id, device):
            self.assertEqual(forwarded["gold"], [])
            self.assertEqual(forwarded["inputs"], original["inputs"])
            seen.append(forwarded["id"])
            return probe.feature_logits(current, forwarded, rel2id, device)
        first = run.predict_documents(make_model(), [sample], REL2ID, DEVICE,
                                      logits_fn=checked)
        sample["gold"] = []
        second = run.predict_documents(make_model(), [sample], REL2ID, DEVICE,
                                       logits_fn=checked)
        self.assertEqual(first, second)
        self.assertEqual(seen, ["synthetic-dev", "synthetic-dev"])
        self.assertEqual(original["inputs"], sample["inputs"])

    def test_decoder_uses_strict_threshold_for_all_self_and_empty_pairs(self):
        sample = feature(3, "decoder")
        def logits_fn(model, sample, rel2id, device):
            logits = torch.zeros(9, 3)
            logits[0, 1] = 1  # self pair at first ID
            logits[2, 2] = 1  # first ID to final empty concept
            logits[8, 1] = 0  # equality must not pass
            return logits, torch.zeros_like(logits), {}
        prediction = run.predict_documents(make_model(), [sample], REL2ID, DEVICE,
                                           logits_fn=logits_fn)
        self.assertEqual(set(prediction), {("decoder",2,"r1",2), ("decoder",2,"r2",8)})

    def test_unknown_gold_and_empty_self_entities_remain_in_scores(self):
        sample = feature(3, "score")
        sample["gold"].append({"h":8, "r":"unknown", "t":8})
        score = dwie.score_documents([document(sample)],
                    [("score",2,"r1",2), ("score",2,"r2",8)], REL2ID)
        self.assertEqual((score["primary_full_id"]["tp"], score["primary_full_id"]["fn"]), (2, 1))
        self.assertEqual(score["slices"]["train_unknown"]["fn"], 1)
        self.assertEqual(score["slices"]["self"]["fn"], 1)
        self.assertEqual(score["slices"]["unanchored_endpoint"]["fn"], 1)

    def test_official_soft_hard_are_supplemental_distinct_units(self):
        docs = [{"id":"metric", "entities":[
            {"id":0,"mentions":[{"begin":0,"end":1}]},
            {"id":1,"mentions":[{"begin":2,"end":3}]},
            {"id":2,"mentions":[]}],
            "relations":[{"h":0,"r":"r1","t":1}, {"h":2,"r":"r1","t":1}]}]
        score = dwie.score_documents(docs, [("metric",0,"r1",1)], REL2ID, METRIC_SOURCE)
        self.assertAlmostEqual(score["primary_full_id"]["f1"], 2/3)
        self.assertEqual(score["official_expanded_clusters"]["soft"]["f1"], 1)
        self.assertAlmostEqual(score["official_expanded_clusters"]["hard"]["f1"], 2/3)
        self.assertTrue(score["official_metrics_are_supplemental_not_id_equivalent"])

    def test_unapproved_training_and_changed_authorization_fail_before_writes(self):
        pending = {"approved":False, "hard_seconds":None,
                   "authorization_scope":"DWIE C/B/D seed75 30 epochs training and Dev only",
                   "arms":["C","B","D"], "seed":75, "epochs":30,
                   "optimizer_updates_per_arm":4530, "total_optimizer_updates":13590,
                   "test_data_read":False, "automatic_retry":False, "resume":False,
                   "extension":False, "user_confirmation":None}
        with patch.object(run, "digest", return_value=run.AUTHORIZATION_SHA256), \
             patch.object(run, "read_json", return_value=pending):
            self.assertEqual(run.read_authorization(require_approved=False), pending)
            with self.assertRaises(PermissionError):
                run.read_authorization(require_approved=True)
            with patch.object(run, "require_lock", side_effect=AssertionError("Lock before authorization")), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("Directory before authorization")):
                with self.assertRaises(PermissionError):
                    run.start(Path("__dwie_unapproved_no_disk__/workspace"))
            pending["seed"] = 76
            with self.assertRaisesRegex(ValueError, "protocol"):
                run.read_authorization(require_approved=False)

    def test_invalid_hard_caps_cannot_authorize_training(self):
        approved = {"approved":True, "hard_seconds":3600,
                    "arms":["C","B","D"], "seed":75, "epochs":30,
                    "optimizer_updates_per_arm":4530, "total_optimizer_updates":13590,
                    "test_data_read":False, "automatic_retry":False, "resume":False,
                    "extension":False, "user_confirmation":"synthetic authorization only"}
        with patch.object(run, "digest", return_value=run.AUTHORIZATION_SHA256), \
             patch.object(run, "read_json", return_value=approved):
            for seconds in (None, False, -1, 0, 100, float("inf"), float("nan"), "3600"):
                approved["hard_seconds"] = seconds
                with self.subTest(seconds=seconds), self.assertRaises(PermissionError):
                    run.read_authorization(require_approved=True)
            approved["hard_seconds"] = 3600
            approved["user_confirmation"] = ""
            with self.assertRaises(PermissionError):
                run.read_authorization(require_approved=True)

    def test_expired_deadline_and_changed_host_boot_do_not_reset_budget(self):
        now = time.monotonic()
        budget = {"deadline_monotonic":now+3600, "boot_id":"synthetic-boot"}
        with patch.object(Path, "read_text", return_value="synthetic-boot"):
            run.check_budget(budget)
            with self.assertRaises(run.BudgetStop):
                run.check_budget({**budget, "deadline_monotonic":now-1})
            with self.assertRaisesRegex(run.BudgetStop, "boot"):
                run.check_budget({**budget, "boot_id":"changed"})

    def test_disk_shortfall_reports_and_stops_without_deletion(self):
        identity = {"parent_paths":{"C_B":"synthetic-C", "D":"synthetic-D"}}
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1024)), \
             patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1)), \
             patch.object(Path, "unlink", side_effect=AssertionError("Deletion forbidden")):
            with self.assertRaisesRegex(RuntimeError, "shortfall=.*no deletion"):
                run.disk_admission(Path("__synthetic_disk_only__"), identity, ("C","B","D"))
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1024)), \
             patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=5*1024**3)):
            report = run.disk_admission(Path("__synthetic_disk_only__"), identity, ("C","B","D"))
        self.assertEqual(report["required_bytes"], 5*1024**3)
        self.assertFalse(report["old_files_deleted"])

    def test_changed_runtime_fails_without_dependency_install_or_GPU(self):
        with patch.object(torch, "__version__", "synthetic-wrong-version"), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("Premature CUDA query")):
            with self.assertRaisesRegex(RuntimeError, "no install/version fallback"):
                run.runtime_identity()

    def test_changed_source_hash_refuses_before_data_or_parent_reads(self):
        with patch.object(run, "digest", return_value="synthetic-wrong-source-sha"), \
             patch.object(run, "read_json", side_effect=AssertionError("Data read before source admission")), \
             patch.object(torch, "load", side_effect=AssertionError("Parent load before source admission")):
            with self.assertRaisesRegex(ValueError, "Pinned source/metadata mismatch"):
                run.validate_inputs(cloud=True)

    def test_container_remaining_memory_can_reject_large_host_memory(self):
        identity = {"parent_paths":{"C_B":"synthetic-C", "D":"synthetic-D"}}
        v2_max = "/sys/fs/cgroup/memory.max"
        v2_used = "/sys/fs/cgroup/memory.current"
        content = {"/proc/meminfo":"MemAvailable: 67108864 kB\n",
                   v2_max:str(2*1024**3), v2_used:str(1900*1024**2)}
        def read(path, *args, **kwargs):
            return content[path.as_posix()]
        with patch.object(Path, "read_text", new=read), \
             patch.object(Path, "is_file", new=lambda path: path.as_posix() in (v2_max,v2_used)), \
             patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1024**3)):
            with self.assertRaisesRegex(RuntimeError, "Effective CPU memory admission failed"):
                run.host_memory_admission(identity)
            content[v2_max] = str(16*1024**3)
            report = run.host_memory_admission(identity)
        self.assertLess(report["effective_available_bytes"], report["host_MemAvailable_bytes"])
        self.assertEqual(report["required_bytes"], 4*1024**3+512*1024**2)
        self.assertTrue(report["host_total_is_not_container_quota"])

    def test_production_checkpoint_benchmark_rotates_one_model_and_retains_raw_differences(self):
        features = [feature(3,"bench-a"), feature(4,"bench-b")]
        original = copy.deepcopy(features)
        templates = {arm:make_model(arm) for arm in ("C","B")}
        states = {arm:copy.deepcopy(model.state_dict()) for arm,model in templates.items()}
        expected = {arm:run.predict_documents(model, features, REL2ID, DEVICE)
                    for arm,model in templates.items()}
        built, weak, counts = [], [], {"C":0,"B":0}
        def builder(arm, identity, device, check_fn):
            self.assertTrue(all(ref() is None for ref in weak), "Two benchmark models reside together")
            model = make_model(arm)
            model._synthetic_round = counts[arm]
            counts[arm] += 1
            built.append(arm)
            weak.append(weakref.ref(model))
            return model, {"synthetic_only":True}
        def logits_fn(model, sample, rel2id, device):
            self.assertEqual(sample["gold"], [])
            output, labels, audit = probe.feature_logits(model, sample, rel2id, device)
            # A known common logit offset exceeds numeric tolerance without
            # changing threshold decisions. The report must retain it.
            if model._synthetic_round > 0:
                output = output+.01
            return output, labels, audit
        def load(path):
            return copy.deepcopy(states[Path(path).parent.name])
        with patch.object(run, "load_saved_state", side_effect=load), \
             patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CPU benchmark touched CUDA")), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Benchmark created optimizer")):
            report = run.benchmark_checkpoints(Path("__dwie_benchmark_no_disk__"), {},
                        features, REL2ID, DEVICE, model_builder=builder,
                        logits_fn=logits_fn, expected_predictions=expected)
        self.assertEqual(built, ["C","B","B","C","C","B"])
        self.assertTrue(report["single_resident_model"])
        self.assertTrue(report["model_load_excluded"])
        self.assertFalse(report["test_data_read"])
        self.assertEqual(len(report["rounds"]), 6)
        for row in report["rounds"]:
            self.assertEqual(row["documents"], 2)
            self.assertEqual(row["pairs"], 25)
            self.assertGreater(row["forward_seconds"], 0)
            self.assertGreaterEqual(row["e2e_seconds"], row["forward_seconds"])
            self.assertEqual(row["prediction_difference_vs_best"], 0)
            self.assertEqual(row["decision_flips_vs_first_round"], 0)
            self.assertEqual(row["numeric_tolerances"], {"atol":1e-4,"rtol":1e-4})
            self.assertTrue(row["all_original_differences_retained"])
            if row["round"] == 0:
                self.assertEqual(row["raw_numeric_violations"], 0)
            else:
                self.assertGreater(row["raw_numeric_violations"], 0)
                self.assertGreater(row["max_absolute_logit_difference"], .009)
        self.assertEqual(features, original)


if __name__ == "__main__":
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contracts attempted CUDA")):
        unittest.main(verbosity=2)
