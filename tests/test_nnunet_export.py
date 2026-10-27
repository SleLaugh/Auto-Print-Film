"""External export adapter: bounded work, upstream equivalence, and spawn."""

from functools import partial
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from src.inference import nnunet_export as adapter


ADAPTER_SCRIPT = Path(adapter.__file__).resolve()


class ExportAdapterTest(unittest.TestCase):
    def test_environment_defaults_and_bounds(self):
        for value, expected in [(None, 1), ("bad", 1), ("0", 1), ("-1", 1),
                                ("2", 2), ("4", 4), ("999", 4)]:
            with self.subTest(value=value), patch.dict(os.environ, clear=True):
                if value is not None:
                    os.environ[adapter.EXPORT_THREADS_ENV] = value
                self.assertEqual(adapter.export_threads(), expected)

    def test_import_needs_neither_numpy_torch_nor_nnunet(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import nnunet_export; "
            "assert not any(x in sys.modules for x in ('numpy', 'torch', 'nnunetv2'))"
        )
        result = subprocess.run([sys.executable, "-S", "-c", code, str(ADAPTER_SCRIPT.parent)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_channels_are_bounded_views_and_original_quantization_is_preserved(self):
        data = np.arange(12 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 3, 4)
        calls = []
        gate = threading.Barrier(3)
        lock = threading.Lock()
        active = 0
        maximum = 0

        def original(channel, shape, is_seg, axis, order, separate, order_z, dtype_out):
            nonlocal active, maximum
            self.assertTrue(np.shares_memory(channel, data))
            self.assertEqual(channel.dtype, data.dtype)
            self.assertEqual(channel.shape[0], 1)
            self.assertEqual((is_seg, axis, order, separate, order_z), (False, 1, 1, True, 2))
            with lock:
                calls.append(float(channel[0, 0, 0, 0]))
                active += 1
                maximum = max(maximum, active)
            gate.wait(timeout=5)
            result = np.full((1, *shape), channel[0, 0, 0, 0] + 0.8,
                             dtype=np.uint8 if dtype_out is None else dtype_out)
            with lock:
                active -= 1
            return result

        actual = adapter.resample_data_or_seg(data, (3, 4, 5), False, 1, 1, True, 2,
                                              np.uint8, original_function=original, threads=3)
        self.assertEqual(maximum, 3)
        self.assertEqual(len(calls), 12)
        self.assertEqual(actual.dtype, np.dtype("uint8"))
        for index in range(12):
            self.assertTrue(np.all(actual[index] == np.uint8(data[index, 0, 0, 0] + 0.8)))

    def test_unsupported_and_no_resize_paths_delegate_unchanged(self):
        data = np.ones((3, 2, 3, 4), dtype=np.float32)
        marker = object()
        cases = [dict(threads=1, order=1), dict(threads=3, order=3),
                 dict(threads=3, order=1, is_seg=True),
                 dict(threads=3, order=1, shape=data.shape[1:]),
                 dict(threads=3, order=1, data=object()),
                 dict(threads=3, order=1, data=data[:1])]
        for case in cases:
            with self.subTest(case=case):
                passed = []

                def original(*args):
                    passed.append(args)
                    return marker

                options = dict(case)
                given = options.pop("data", data)
                shape = options.pop("shape", (3, 4, 5))
                self.assertIs(adapter.resample_data_or_seg(
                    given, shape, original_function=original, **options), marker)
                self.assertEqual(len(passed), 1)
                self.assertIs(passed[0][0], given)

    def test_upstream_without_optional_dtype_out_keeps_its_call_contract(self):
        data = np.ones((3, 2, 3, 4), dtype=np.float16)

        def original(value, shape, is_seg=False, axis=None, order=3,
                     do_separate_z=False, order_z=0):
            return np.full((value.shape[0], *shape), 0.5, dtype=value.dtype)

        for threads in (1, 3):
            with self.subTest(threads=threads):
                actual = adapter.resample_data_or_seg(
                    data, (3, 4, 5), order=1,
                    original_function=original, threads=threads,
                )
                np.testing.assert_array_equal(actual, original(data, (3, 4, 5), order=1))

    def test_diagnostics_failure_does_not_change_success_or_exception(self):
        with patch.object(adapter.time, "perf_counter", side_effect=RuntimeError("clock")):
            self.assertEqual(adapter._timed_call("stage", lambda: 9), 9)
        with patch("builtins.print", side_effect=BrokenPipeError()):
            self.assertEqual(adapter._timed_call("stage", lambda: 7), 7)
            with self.assertRaisesRegex(ValueError, "original error"):
                adapter._timed_call("stage", lambda: (_ for _ in ()).throw(ValueError("original error")))

    def test_worker_resampling_patch_is_restored_when_original_raises(self):
        fake_module = type("FakeModule", (), {})()

        def original(*args, **kwargs):
            return None

        fake_module.resample_data_or_seg = original

        def to_shape(*args, **kwargs):
            raise ValueError("upstream error")

        to_shape.__module__ = adapter._DEFAULT_RESAMPLING_MODULE
        to_shape.__name__ = "resample_data_or_seg_to_shape"
        with patch.object(adapter.importlib, "import_module", return_value=fake_module):
            with self.assertRaisesRegex(ValueError, "upstream error"):
                with adapter._parallel_default_resampling(partial(to_shape, order=1), 2):
                    self.assertIsNot(fake_module.resample_data_or_seg, original)
                    to_shape()
        self.assertIs(fake_module.resample_data_or_seg, original)

    def test_cli_spawn_imports_stable_wrapper_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "nnunetv2/inference"
            package.mkdir(parents=True)
            (package.parent / "__init__.py").write_text("")
            (package / "__init__.py").write_text("")
            (package / "export_prediction.py").write_text(
                "import json\n"
                "def export_prediction_from_logits(data, props, config, plans, dataset, output, save=False, **kwargs):\n"
                "    value = config.resampling_fn_probabilities(data)\n"
                "    value = plans.get_label_manager(dataset).convert_logits_to_segmentation(value)\n"
                "    plans.image_reader_writer_class().write_seg(value, output, props)\n"
                "    return value\n"
            )
            (package / "predict_from_raw_data.py").write_text(
                "import json, multiprocessing, sys\n"
                "def resample(value): return value + 2\n"
                "class Config:\n"
                "    resampling_fn_probabilities = staticmethod(resample)\n"
                "class Label:\n"
                "    def convert_logits_to_segmentation(self, value): return value + 3\n"
                "class Writer:\n"
                "    def write_seg(self, value, output, props):\n"
                "        with open(output, 'w') as handle: json.dump(value, handle)\n"
                "class Plans:\n"
                "    image_reader_writer_class = Writer\n"
                "    def get_label_manager(self, dataset): return Label()\n"
                "def predict_entry_point():\n"
                "    from nnunetv2.inference import export_prediction as upstream\n"
                "    assert upstream.export_prediction_from_logits is not export_prediction_from_logits\n"
                "    with multiprocessing.get_context('spawn').Pool(1) as pool:\n"
                "        value = pool.apply(export_prediction_from_logits, (4, {}, Config(), Plans(), {}, sys.argv[1]))\n"
                "    print(json.dumps({'value': value, 'args': sys.argv, 'module': export_prediction_from_logits.__module__}))\n"
                "    return 17\n"
            )
            environment = os.environ.copy()
            environment.pop("PYTHONHOME", None)
            environment["PYTHONPATH"] = str(root)
            environment[adapter.EXPORT_THREADS_ENV] = "3"
            output = root / "中文 output.json"
            arguments = [str(output), "-f", "0", "--continue_prediction"]
            result = subprocess.run([sys.executable, "-S", str(ADAPTER_SCRIPT), *arguments],
                                    env=environment, capture_output=True, text=True, timeout=30,
                                    encoding="utf-8", check=False)
            self.assertEqual(result.returncode, 17, result.stderr)
            payload = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(payload, {"value": 9, "args": ["nnUNetv2_predict", *arguments],
                                       "module": "nnunet_export"})
            self.assertEqual(json.loads(output.read_text()), 9)
            for stage in ("export_resampling", "export_label_decision", "export_write_seg", "export_total"):
                self.assertIn(f"[AutoPrintFilm nnUNet] {stage} seconds=", result.stdout)


class UpstreamEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.upstream = importlib.import_module(adapter._DEFAULT_RESAMPLING_MODULE)
            cls.torch = importlib.import_module("torch")
        except ImportError as error:
            raise unittest.SkipTest(f"optional external nnUNet dependencies unavailable: {error}")

    def test_real_resampler_matches_exactly_across_axes_orders_and_dtypes(self):
        rng = np.random.default_rng(781)
        for dtype in (np.float16, np.float32, np.float64):
            for axis in (None, 0, 1, 2):
                for order_z in (0, 1, 3):
                    with self.subTest(dtype=dtype, axis=axis, order_z=order_z):
                        data = rng.normal(size=(4, 4, 5, 6)).astype(dtype)
                        options = dict(axis=axis, order=1, do_separate_z=axis is not None, order_z=order_z)
                        expected = self.upstream.resample_data_or_seg(data, (6, 7, 8), **options)
                        actual = adapter.resample_data_or_seg(data, (6, 7, 8), **options,
                                                              original_function=self.upstream.resample_data_or_seg,
                                                              threads=3)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.dtype, expected.dtype)
                        np.testing.assert_array_equal(actual.argmax(0), expected.argmax(0))

    def test_real_no_resize_identity_and_tensor_contract(self):
        data = np.ones((4, 4, 5, 6), dtype=np.float32)
        with patch.dict(os.environ, {adapter.EXPORT_THREADS_ENV: "3"}):
            for given in (data, self.torch.from_numpy(data)):
                with self.subTest(kind=type(given)):
                    function = partial(self.upstream.resample_data_or_seg_to_shape, order=1)
                    with adapter._parallel_default_resampling(function, 3):
                        actual = function(given, (4, 5, 6), (1, 1, 1), (1, 1, 1))
                    self.assertIsInstance(actual, np.ndarray)
                    self.assertTrue(np.shares_memory(actual, data))
                    with adapter._parallel_default_resampling(function, 3):
                        actual = function(given, (6, 7, 8), (1, 1, 1), (1, 1, 1))
                    expected = function(given, (6, 7, 8), (1, 1, 1), (1, 1, 1))
                    np.testing.assert_array_equal(actual, expected)

    def test_real_nan_and_ties_preserve_values_and_label_decision(self):
        data = np.zeros((4, 4, 5, 6), dtype=np.float32)
        data[0, 0, 0, 0] = np.nan
        data[1:3] = 0.5
        options = dict(order=1)
        expected = self.upstream.resample_data_or_seg(data, (6, 7, 8), **options)
        actual = adapter.resample_data_or_seg(data, (6, 7, 8), **options,
                                              original_function=self.upstream.resample_data_or_seg, threads=4)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.argmax(0), expected.argmax(0))


if __name__ == "__main__":
    unittest.main()
