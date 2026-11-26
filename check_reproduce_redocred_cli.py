"""Standard-library CLI contracts. No tensors, model loading or dataset reads."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ENTRY = Path(__file__).with_name("reproduce_redocred.py")
SPEC = importlib.util.spec_from_file_location("portable_redocred", ENTRY)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EntryPointContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ttmre-portable-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "roberta"
        self.data = self.directory / "data"
        self.model.mkdir()
        self.data.mkdir()
        (self.model / "config.json").write_text(
            json.dumps({"model_type": "roberta", "hidden_size": 1024}), encoding="utf-8")
        for name in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
                     "model.safetensors"):
            (self.model / name).write_bytes(b"deliberately not a model")
        for name in ("train_revised.json", "dev_revised.json", "test_revised.json"):
            (self.data / name).write_bytes(b"deliberately not a dataset")
        self.checkpoint = self.directory / "source.pth"
        self.checkpoint.write_bytes(b"deliberately not a checkpoint")

    def arguments(self, *extra):
        values = ["--mode", "validate", "--model-dir", str(self.model),
                  "--data-dir", str(self.data), "--checkpoint", str(self.checkpoint), *extra]
        return MODULE.normalized_args(MODULE.parser().parse_args(values))

    def test_no_arguments_only_prints_usage(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(MODULE.main([]), 0)
        self.assertIn("--mode", stream.getvalue())

    def test_validate_does_not_read_model_checkpoint_or_dataset_contents(self):
        result = MODULE.validate(self.arguments())
        self.assertFalse(result["checkpoint_contents_read"])
        self.assertFalse(result["dataset_contents_read"])
        self.assertFalse(result["model_execution"])
        self.assertFalse(result["checkpoint_hash_checked"])

    def test_validation_does_not_create_output_directory(self):
        output = self.directory / "new-output"
        MODULE.validate(self.arguments("--output-dir", str(output)))
        self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        output = self.directory / "existing-output"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            MODULE.validate(self.arguments("--output-dir", str(output)))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_training_cannot_select_on_test(self):
        with self.assertRaisesRegex(ValueError, "Dev"):
            MODULE.validate(self.arguments("--validate-for", "train", "--split", "test"))

    def test_c_off_is_evaluation_only(self):
        with self.assertRaisesRegex(ValueError, "C-off"):
            MODULE.validate(self.arguments("--validate-for", "train", "--variant", "C-off"))

    def test_missing_split_is_reported(self):
        (self.data / "test_revised.json").unlink()
        with self.assertRaises(FileNotFoundError):
            MODULE.validate(self.arguments("--split", "test"))

    def test_average_requires_three_distinct_inputs(self):
        values = ["--mode", "validate", "--validate-for", "average", "--checkpoints",
                  *([str(self.checkpoint)] * 3)]
        arguments = MODULE.normalized_args(MODULE.parser().parse_args(values))
        with self.assertRaisesRegex(ValueError, "distinct"):
            MODULE.validate(arguments)

    def test_average_validation_does_not_open_tensor_files(self):
        paths = []
        for seed in (75, 76, 77):
            path = self.directory / (str(seed) + ".pth")
            path.write_bytes(b"not a tensor file")
            paths.append(str(path))
        values = ["--mode", "validate", "--validate-for", "average", "--checkpoints", *paths]
        arguments = MODULE.normalized_args(MODULE.parser().parse_args(values))
        result = MODULE.validate(arguments)
        self.assertEqual(len(result["files"]), 3)
        self.assertFalse(result["checkpoint_contents_read"])

    def test_selection_is_after_penultimate_update(self):
        self.assertFalse(MODULE.selection_due(761, 763))
        self.assertTrue(MODULE.selection_due(762, 763))
        self.assertFalse(MODULE.selection_due(763, 763))


if __name__ == "__main__":
    if "torch" in sys.modules:
        raise RuntimeError("Run CLI contracts in a fresh standard-library process")
    unittest.main()
