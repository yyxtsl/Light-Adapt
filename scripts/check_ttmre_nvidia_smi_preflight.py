"""Pure standard-library tests for the external GPU check; no torch/CUDA."""
import unittest
from unittest.mock import Mock

import ttmre_nvidia_smi_preflight as preflight


class NvidiaSmiPreflightContracts(unittest.TestCase):
    def fake_runner(self, stdout):
        result = Mock(stdout=stdout)
        return Mock(return_value=result)

    def test_idle_gpu_is_selected_and_passes_below_limit(self):
        runner = self.fake_runner("0, NVIDIA GeForce RTX 5090, 2, 32607, 0, GPU-test-0\n")
        report = preflight.inspect_gpu(run=runner)
        self.assertTrue(report["passed"])
        self.assertEqual(report["memory_used_bytes"], 2 * preflight.MIB)
        self.assertEqual(report["source"], "nvidia-smi-before-torch-cuda-init")
        runner.assert_called_once_with(preflight.QUERY, capture_output=True, text=True, check=True, timeout=10)

    def test_exactly_five_hundred_mib_fails_closed(self):
        runner = self.fake_runner("0, NVIDIA GeForce RTX 5090, 500, 32607, 0, GPU-test-0\n")
        report = preflight.inspect_gpu(run=runner)
        self.assertFalse(report["passed"])

    def test_selects_visible_physical_gpu_instead_of_always_gpu_zero(self):
        runner = self.fake_runner(
            "0, GPU A, 10, 32607, 0, GPU-a\n"
            "1, GPU B, 700, 32607, 30, GPU-b\n"
        )
        report = preflight.inspect_gpu(visible_devices="1", run=runner)
        self.assertEqual(report["index"], 1)
        self.assertFalse(report["passed"])

    def test_selects_visible_gpu_by_uuid(self):
        runner = self.fake_runner("2, GPU C, 2, 32607, 0, GPU-c\n")
        report = preflight.inspect_gpu(visible_devices="GPU-c", run=runner)
        self.assertEqual(report["uuid"], "GPU-c")
        self.assertTrue(report["passed"])

    def test_malformed_and_empty_reports_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unexpected nvidia-smi"):
            preflight.parse_rows("0, GPU\n")
        with self.assertRaisesRegex(RuntimeError, "no GPU rows"):
            preflight.parse_rows("\n")

    def test_disabled_or_unmappable_visible_device_fails_closed(self):
        rows = preflight.parse_rows("0, GPU A, 2, 32607, 0, GPU-a\n")
        with self.assertRaisesRegex(RuntimeError, "disables CUDA"):
            preflight.select_visible_gpu(rows, "-1")
        with self.assertRaisesRegex(RuntimeError, "Cannot map"):
            preflight.select_visible_gpu(rows, "GPU-missing")


if __name__ == "__main__":
    unittest.main()
