"""Synthetic CPU contracts for the one-use DWIE GPU probe; no real assets."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import ttmre_dwie_gpu_probe_v4 as probe
import ttmre_pair_replacement as replacement

torch.set_num_threads(2)


class Output:
    def __init__(self, sequence, attention):
        self.last_hidden_state = sequence
        self.attentions = (attention,)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = TinyEmbeddings()

    def forward(self, input_ids, attention_mask, output_attentions, return_dict):
        sequence = self.embeddings.word_embeddings(input_ids)
        states = sequence[0]
        weights = torch.softmax(states @ states.T, dim=-1)
        return Output(sequence, weights[None, None])


class TinyEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(16, 4)


class TinyEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_tokens = nn.Parameter(torch.randn(2, 4))
        self.projection = nn.Linear(4, 4)

    def forward(self, values, train=False):
        return values + self.projection(values) + self.memory_tokens.mean()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.emb_size, self.block_size = 4, 2
        self.head_extractor = nn.Linear(8, 4)
        self.bilinear = nn.Linear(8, 3)
        self.mu_encoder = TinyEnhancer()
        self.args = SimpleNamespace(num_class=3)
        self.config = SimpleNamespace(num_labels=3)
        self.rels = 2


def feature():
    return {"id": "synthetic-train", "inputs": {
        "content_ids": [3, 4, 5, 6], "entity_ids": [2, 8, 9],
        "entity_pos": [[[0, 1]], [[2, 3]], []],
        "candidate_rule": "cartesian_including_self"},
        "gold": [{"h": 2, "r": "r1", "t": 8}, {"h": 8, "r": "r2", "t": 8}]}


class Contracts(unittest.TestCase):
    def test_protocol_fixed_scope_and_no_training_artifacts(self):
        contract = probe.protocol()
        self.assertEqual(contract["version"], "dwie-gpu-admission-probe-v4")
        self.assertEqual(contract["forward_backward_runs"], 6)
        self.assertEqual(contract["hard_seconds"], 885)
        self.assertEqual(contract["candidate_pairs"], "all directed entity pairs including self pairs; retain empty-mention entities")
        self.assertFalse(contract["optimizer_created"])
        self.assertEqual(contract["optimizer_updates"], 0)
        self.assertEqual(contract["checkpoint_saves"], 0)
        self.assertFalse(contract["test_data_read"])

    def test_atgl_targets_include_threshold_for_all_pairs_and_multiple_relations(self):
        labels = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
        logits = torch.tensor([[2., 1., -1.], [0., -1., 2.]])
        loss = probe.atgl_loss(logits, labels)
        expected_target = torch.tensor([[1., 6., 0.], [1., 0., 6.]])
        expected = -(torch.log_softmax(logits, -1) * expected_target).sum(-1).mean()
        torch.testing.assert_close(loss, expected)

    def test_fixed_classifier_initialization_is_identical(self):
        first, first_sha = probe.fixed_target_head(8)
        second, second_sha = probe.fixed_target_head(8)
        self.assertEqual(first_sha, second_sha)
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_strict_state_loader_fails_closed_on_missing_keys(self):
        model = nn.Linear(3, 2)
        state = copy.deepcopy(model.state_dict())
        report = probe.strict_load(model, state, "synthetic")
        self.assertTrue(report["strict"])
        with self.assertRaisesRegex(RuntimeError, "missing=.*bias"):
            probe.strict_load(model, {"weight": state["weight"]}, "synthetic-broken")

    def test_bypass_transition_removes_only_ttm_parameters_and_keeps_memory_buffer(self):
        model = TinyModel()
        c_state = copy.deepcopy(model.state_dict())
        probe.strict_load(model, c_state, "C synthetic")
        model.mu_encoder.memory_tokens.requires_grad_(False)
        self.assertFalse(model.mu_encoder.memory_tokens.requires_grad)
        model.requires_grad_(True)
        self.assertTrue(model.mu_encoder.memory_tokens.requires_grad)
        before = set(model.state_dict())
        replacement.install(model, "bypass")
        after = set(model.state_dict())
        self.assertEqual(before - after, {"mu_encoder.projection.weight", "mu_encoder.projection.bias"})
        self.assertIn("mu_encoder.memory_tokens", after)
        self.assertNotIn("mu_encoder.memory_tokens", dict(model.named_parameters()))
        self.assertIn("mu_encoder.memory_tokens", dict(model.named_buffers()))
        d_target = TinyModel()
        replacement.install(d_target, "bypass")
        report = probe.strict_load(model, copy.deepcopy(d_target.state_dict()), "D synthetic")
        self.assertTrue(report["strict"])

    def test_tiny_forward_backward_uses_self_pairs_and_keeps_empty_entity(self):
        model = TinyModel()
        model.requires_grad_(True)
        with patch.object(probe, "check_budget"):
            report = probe.probe_document(model, feature(), {"r1": 1, "r2": 2},
                                          torch.device("cpu"), seed=75, budget={})
        self.assertEqual(report["candidate_pairs_including_self"], 9)
        self.assertEqual(report["zero_attention_context_pairs"], 5)
        self.assertTrue(report["logits_finite"])
        self.assertTrue(report["loss_finite"])
        self.assertTrue(report["all_gradients_finite"])
        self.assertTrue(all(report["critical_gradient_nonzero"].values()))
        self.assertTrue(report["parameters_unchanged"])
        self.assertEqual(report["optimizer_updates"], 0)

    def test_unknown_gold_is_rejected_instead_of_becoming_threshold(self):
        broken = feature()
        broken["gold"].append({"h": 2, "r": "unknown", "t": 9})
        with self.assertRaisesRegex(ValueError, "Unknown train relation"):
            probe.target_labels(broken, {"r1": 1, "r2": 2}, torch.device("cpu"))

    def test_busy_gpu_stops_before_cuda_queries_or_parent_load(self):
        busy = {"passed": False, "memory_used_bytes": 501 * 1024**2,
                "memory_total_bytes": 32607 * 1024**2}
        with patch.object(probe.gpu_preflight, "inspect_gpu", return_value=busy), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA touched")), \
             patch.object(probe, "build_model", side_effect=AssertionError("Model built")), \
             patch.object(probe, "load_checkpoint", side_effect=AssertionError("Weights loaded")):
            with self.assertRaisesRegex(RuntimeError, "501.0MiB externally used"):
                probe.run_probe(None, {}, {}, {})

    def test_external_idle_check_precedes_all_cuda_queries(self):
        order = []
        def external_check():
            order.append("external")
            return {"passed": True, "name": "synthetic-gpu"}
        def available():
            order.append("cuda")
            return False
        with patch.object(probe.gpu_preflight, "inspect_gpu", side_effect=external_check), \
             patch.object(torch.cuda, "is_available", side_effect=available):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                probe.run_probe(None, {}, {}, {})
        self.assertEqual(order, ["external", "cuda"])

    def test_real_model_constructor_ttm_checkpoint_backward_and_c_b_d_transition(self):
        from transformers import RobertaConfig
        config = RobertaConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=16,
                               max_position_embeddings=514, pad_token_id=1)
        with patch("transformers.AutoConfig.from_pretrained", return_value=config):
            model, report = probe.build_model("synthetic-config-no-files")
        self.assertTrue(report["pair_independent_ttm_axis_correction"])
        self.assertTrue(report["ttm_activation_checkpointing"]["preserve_rng_state"])
        encoder = model.mu_encoder
        self.assertTrue(encoder.token_turing_machine_unit.vit_encoder._ttmre_pair_independent)
        self.assertTrue(encoder._ttmre_unused_write_bypassed)
        self.assertTrue(hasattr(encoder, "_ttmre_uncheckpointed_forward"))
        # Check the real model's tensor key/shape contract without writing a checkpoint.
        original_state = copy.deepcopy(model.state_dict())
        self.assertTrue(probe.strict_load(model, original_state, "synthetic C")["strict"])
        del original_state
        model.mu_encoder.memory_tokens.requires_grad_(False)
        probe.set_classifier(model, 3)
        with patch.object(probe, "check_budget"):
            c_report = probe.probe_document(model, feature(), {"r1": 1, "r2": 2},
                                            torch.device("cpu"), seed=75, budget={})
        self.assertTrue(c_report["frozen_memory_gradient_none"])
        self.assertTrue(c_report["all_gradients_finite"])
        self.assertTrue(c_report["parameters_unchanged"])
        model.requires_grad_(True)
        replacement.install(model, "bypass")
        with patch.object(probe, "check_budget"):
            b_report = probe.probe_document(model, feature(), {"r1": 1, "r2": 2},
                                            torch.device("cpu"), seed=75, budget={})
        self.assertTrue(b_report["all_gradients_finite"])
        probe.set_classifier(model, probe.SOURCE_CLASSES)
        # Plain tensor state and strict loading use the same no-TTM key layout as D.
        state = copy.deepcopy(model.state_dict())
        self.assertTrue(probe.strict_load(model, state, "synthetic D")["strict"])
        probe.set_classifier(model, 3)
        with patch.object(probe, "check_budget"):
            d_report = probe.probe_document(model, feature(), {"r1": 1, "r2": 2},
                                            torch.device("cpu"), seed=75, budget={})
        self.assertTrue(d_report["all_gradients_finite"])
        self.assertTrue(d_report["parameters_unchanged"])

    def test_long_window_merge_backpropagates_to_both_document_ends(self):
        model = TinyModel()
        sample = feature()
        sample["inputs"]["content_ids"] = [3] * 519 + [7]
        sample["inputs"]["entity_pos"] = [[[0, 1]], [[519, 520]], []]
        with patch.object(probe, "check_budget"):
            result = probe.probe_document(model, sample, {"r1": 1, "r2": 2},
                                          torch.device("cpu"), seed=75, budget={})
        self.assertEqual(result["windows"], 2)
        # probe_document clears gradients; inspect a separate graph at the boundary tokens.
        logits, labels, _ = probe.feature_logits(model, sample, {"r1": 1, "r2": 2}, torch.device("cpu"))
        probe.atgl_loss(logits, labels).backward()
        gradient = model.model.embeddings.word_embeddings.weight.grad
        self.assertGreater(gradient[3].abs().sum().item(), 0)
        self.assertGreater(gradient[7].abs().sum().item(), 0)

    def test_readonly_source_validation_accepts_no_budget(self):
        with patch.object(probe, "digest", side_effect=lambda path, budget=None: probe.PINNED_SOURCES[path.relative_to(probe.ROOT).as_posix()]), \
             patch.object(probe, "check_budget", side_effect=AssertionError("Unexpected mutable budget")):
            self.assertEqual(probe.check_source_inputs(None), probe.PINNED_SOURCES)


if __name__ == "__main__":
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden")), \
         patch.object(torch.optim, "AdamW", side_effect=AssertionError("Optimizer forbidden")), \
         patch.object(torch, "save", side_effect=AssertionError("Checkpoint save forbidden")):
        unittest.main()
