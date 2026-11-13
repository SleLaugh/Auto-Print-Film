"""Observe nnUNet boundaries without importing an engine or touching CUDA."""

from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.inference import nnunet_export as adapter


def events(stream):
    prefix = "[AutoPrintFilm nnUNet timing] "
    return [json.loads(line[len(prefix):]) for line in stream.getvalue().splitlines()
            if line.startswith(prefix)]


class PredictionTimingTest(unittest.TestCase):
    def test_nested_boundaries_preserve_signature_result_and_iteration(self):
        marker = object()
        order = []

        class Predictor:
            def initialize_from_trained_model_folder(self, folder, *, checkpoint="final"):
                order.append((folder, checkpoint))
                return marker

            def predict_logits_from_preprocessed_data(self, data):
                self.assertion = data is marker
                order.append("prediction")
                return data

            def predict_from_data_iterator(self, data_iterator):
                values = [self.predict_logits_from_preprocessed_data(x) for x in data_iterator]
                order.append("cleanup")
                return values

            def predict_from_files(self, files, *, overwrite=False):
                # Upstream uses this exact introspection to save its arguments.
                self.arguments = tuple(inspect.signature(self.predict_from_files).parameters)
                def source():
                    order.append("preprocess")
                    yield files
                    order.append("preprocess_finish")
                return self.predict_from_data_iterator(source())

        signature = inspect.signature(Predictor.predict_from_files)
        engine = SimpleNamespace(nnUNetPredictor=Predictor)
        adapter.install_prediction_timing(engine)
        measured = Predictor.predict_from_files
        adapter.install_prediction_timing(engine)
        self.assertIs(Predictor.predict_from_files, measured)
        self.assertEqual(inspect.signature(Predictor.predict_from_files), signature)
        stream = io.StringIO()
        predictor = Predictor()
        with redirect_stdout(stream), patch.object(adapter.time, "perf_counter", side_effect=range(100, 140)):
            self.assertIs(predictor.initialize_from_trained_model_folder("model", checkpoint="best"), marker)
            self.assertEqual(predictor.predict_from_files(marker, overwrite=True), [marker])
        self.assertTrue(predictor.assertion)
        self.assertEqual(predictor.arguments, ("files", "overwrite"))
        self.assertEqual(order, [("model", "best"), "preprocess", "prediction", "preprocess_finish", "cleanup"])
        records = events(stream)
        self.assertEqual([r["stage"] for r in records], [
            "model_initialization", "preprocessing_next", "predict_logits",
            "preprocessing_next", "prediction_iterator_total", "prediction_files_total",
        ])
        self.assertEqual([r["status"] for r in records], [
            "completed", "ready", "completed", "exhausted", "completed", "completed",
        ])
        outer = records[-1]
        for child in records[1:-1]:
            self.assertLess(outer["start_monotonic"], child["start_monotonic"])
            self.assertGreater(outer["end_monotonic"], child["end_monotonic"])
        self.assertTrue(all("no_added_cuda_sync" in r["measurement"] for r in records))

    def test_custom_iterator_identity_and_its_cleanup_are_untouched(self):
        class CustomIterator:
            pass
        iterator = CustomIterator()

        class Predictor:
            def predict_from_data_iterator(self, data_iterator):
                assert data_iterator is iterator
                assert isinstance(data_iterator, CustomIterator)
                return "original cleanup path"

        adapter.install_prediction_timing(SimpleNamespace(nnUNetPredictor=Predictor))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(Predictor().predict_from_data_iterator(data_iterator=iterator), "original cleanup path")

    def test_failure_identity_and_generator_close_are_preserved(self):
        error = ValueError("engine error")

        class Predictor:
            def predict_logits_from_preprocessed_data(self, data):
                raise error

        adapter.install_prediction_timing(SimpleNamespace(nnUNetPredictor=Predictor))
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaises(ValueError) as caught:
            Predictor().predict_logits_from_preprocessed_data(None)
        self.assertIs(caught.exception, error)
        self.assertEqual(events(stream)[0]["status"], "failed")
        finalized = []
        def source():
            try:
                yield 3
                raise error
            finally:
                finalized.append(True)
        observed = adapter._TimedPreprocessingIterator(source())
        with redirect_stdout(stream):
            self.assertEqual(next(observed), 3)
            observed.close()
        self.assertEqual(finalized, [True])
        observed = adapter._TimedPreprocessingIterator(source())
        with redirect_stdout(stream):
            next(observed)
            with self.assertRaises(ValueError) as caught:
                next(observed)
        self.assertIs(caught.exception, error)
        self.assertEqual(events(stream)[-1]["status"], "failed")

    def test_diagnostics_cannot_change_result_or_original_exception(self):
        marker = object()
        with patch.object(adapter.time, "perf_counter", side_effect=RuntimeError("clock")):
            self.assertIs(adapter.timed_phase("example", lambda: marker), marker)
        error = LookupError("original")
        def fail():
            raise error
        with patch("builtins.print", side_effect=BrokenPipeError()):
            self.assertIs(adapter.timed_phase("example", lambda: marker), marker)
            with self.assertRaises(LookupError) as caught:
                adapter.timed_phase("example", fail)
        self.assertIs(caught.exception, error)

    def test_cpu_entry_preserves_exit_value_and_records_inclusive_import_boundary(self):
        engine = SimpleNamespace(predict_entry_point=lambda: 17)
        stream = io.StringIO()
        with patch.object(adapter, "install_export_adapter", return_value=engine), redirect_stdout(stream):
            self.assertEqual(adapter.main(entry_started=adapter._clock()), 17)
        self.assertEqual([r["stage"] for r in events(stream)], [
            "entry_imports", "engine_import", "cli_total", "entry_script_total",
        ])

    def test_gpu_entry_installs_same_timing_without_loading_gpu_dependencies(self):
        script = Path(adapter.__file__).with_name("nnunet_gpu_export.py")
        spec = importlib.util.spec_from_file_location("timing_test_gpu_adapter", script)
        gpu = importlib.util.module_from_spec(spec)
        # Only adapter control flow runs. No torch, CuPy, weights or CUDA calls.
        with patch.dict("sys.modules", {"nnunet_export": adapter,
                                        "nnunet_gpu_resampling": SimpleNamespace(), "numpy": SimpleNamespace()}):
            spec.loader.exec_module(gpu)
        engine = SimpleNamespace(predict_entry_point=lambda: 23)
        stream = io.StringIO()
        with (patch.dict("os.environ", {"AUTOPRINTFILM_NNUNET_ACCELERATION_SCHEME": "scheme2",
                                        "AUTOPRINTFILM_NNUNET_RESAMPLING_DEVICE": "cpu",
                                        "NNUNET_GPU_RESAMPLING": "0"}),
              patch.object(gpu.importlib, "import_module", return_value=engine),
              redirect_stdout(stream)):
            self.assertEqual(gpu.main(entry_started=adapter._clock()), 23)
        self.assertIs(engine.export_prediction_from_logits, gpu.export_prediction_from_logits)
        self.assertEqual([r["stage"] for r in events(stream)], [
            "entry_imports", "engine_import", "cli_total", "entry_script_total",
        ])


if __name__ == "__main__":
    unittest.main()
