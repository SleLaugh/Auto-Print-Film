"""Official nnUNet CLI with bounded, equivalent CPU export parallelism.

This file is an external-Python entry point, not an application dependency on
torch or nnUNet. It deliberately leaves model prediction and export semantics
to nnUNet. Only independent channels of its default order-1 resampler run in
parallel, and each channel still uses the installed nnUNet implementation.
"""

import time

# This starts at script execution, after Python's own interpreter/site startup.
try:
    _ENTRY_STARTED = time.perf_counter()
except Exception:
    _ENTRY_STARTED = None

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import partial, wraps
import importlib
import json
import os
from pathlib import Path
import sys
import threading
from types import GeneratorType


EXPORT_THREADS_ENV = "AUTOPRINTFILM_NNUNET_EXPORT_THREADS"
MAX_EXPORT_THREADS = 4
_DEFAULT_RESAMPLING_MODULE = "nnunetv2.preprocessing.resampling.default_resampling"
_RESAMPLING_LOCK = threading.RLock()


def export_threads():
    """Invalid/missing external configuration always uses the original path."""
    try:
        return max(1, min(MAX_EXPORT_THREADS, int(os.environ.get(EXPORT_THREADS_ENV, "1"))))
    except (TypeError, ValueError):
        return 1


def _clock():
    try:
        return time.perf_counter()
    except Exception:
        return None


def _timing(stage, started):
    # A closed stdout or failed diagnostic timer must not fail a valid export.
    try:
        if started is not None:
            elapsed = time.perf_counter() - started
            print(f"[AutoPrintFilm nnUNet] {stage} seconds={elapsed:.6f}", flush=True)
    except Exception:
        pass


def _timed_call(stage, function, *args, **kwargs):
    started = _clock()
    status = "failed"
    try:
        result = function(*args, **kwargs)
        status = "completed"
        return result
    finally:
        _timing(stage, started)
        phase_timing(stage, started, status=status, scope="export_worker; nested export timings overlap")


def phase_timing(stage, started, *, status="completed", scope=""):
    """Record host wall time only; never synchronize CUDA for diagnostics.

    These are inclusive intervals, potentially in different processes. A parent
    waiting for preprocessing/export overlaps the worker's own measurements;
    neither durations nor their differences are independent phase totals.
    """
    try:
        if started is None:
            return
        ended = time.perf_counter()
        print("[AutoPrintFilm nnUNet timing] " + json.dumps({
            "stage": stage, "pid": os.getpid(), "status": status,
            "seconds": round(ended - started, 6),
            "start_monotonic": started, "end_monotonic": ended,
            "measurement": "inclusive_host_wall; no_added_cuda_sync",
            "scope": scope,
        }), flush=True)
    except Exception:
        # Observation cannot replace the original result or exception.
        pass


def timed_phase(stage, function, *args, scope="", **kwargs):
    started = _clock()
    status = "failed"
    try:
        result = function(*args, **kwargs)
        status = "completed"
        return result
    finally:
        phase_timing(stage, started, status=status, scope=scope)


class _TimedPreprocessingIterator:
    """Observe next() on the official file generator without changing its data.

    Only native generators are wrapped. In particular, leave custom augmenters
    unchanged because upstream uses isinstance() to decide how to close them.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __iter__(self):
        return self

    def __next__(self):
        started = _clock()
        status = "failed"
        try:
            value = next(self._wrapped)
            status = "ready"
            return value
        except StopIteration:
            status = "exhausted"
            raise
        finally:
            phase_timing(
                "preprocessing_next", started, status=status,
                scope="parent wait; includes worker startup/read/preprocess/IPC or iterator finish; overlaps worker timings",
            )

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def install_prediction_timing(prediction):
    """Wrap stable public boundaries; preserve signatures used by nnUNet itself."""
    predictor = getattr(prediction, "nnUNetPredictor", None)
    if predictor is None:
        return
    boundaries = {
        "initialize_from_trained_model_folder": (
            "model_initialization", "checkpoint reading, network construction and initialization; not weights-only",
        ),
        "predict_from_files": (
            "prediction_files_total", "includes input setup, preprocessing, prediction, export wait and worker cleanup",
        ),
        "predict_logits_from_preprocessed_data": (
            "predict_logits", "whole predictor call including folds/transfers; host wall time, not CUDA kernel time",
        ),
    }
    for method, (stage, scope) in boundaries.items():
        original = getattr(predictor, method, None)
        if original is None or getattr(original, "_autoprintfilm_timed", False):
            continue

        def wrap(function, stage, scope):
            @wraps(function)
            def measured(*args, **kwargs):
                return timed_phase(stage, function, *args, scope=scope, **kwargs)
            measured._autoprintfilm_timed = True
            return measured

        setattr(predictor, method, wrap(original, stage, scope))

    original = getattr(predictor, "predict_from_data_iterator", None)
    if original is not None and not getattr(original, "_autoprintfilm_timed", False):
        @wraps(original)
        def measured_iterator(self, data_iterator, *args, **kwargs):
            if isinstance(data_iterator, GeneratorType):
                data_iterator = _TimedPreprocessingIterator(data_iterator)
            return timed_phase(
                "prediction_iterator_total", original, self, data_iterator, *args,
                scope="includes preprocessing waits, prediction, export IPC/wait and worker cleanup; nested intervals overlap",
                **kwargs,
            )
        measured_iterator._autoprintfilm_timed = True
        predictor.predict_from_data_iterator = measured_iterator


def resample_data_or_seg(data, new_shape, is_seg=False, axis=None, order=3,
                         do_separate_z=False, order_z=0, dtype_out=None, *,
                         original_function, threads=1):
    """Parallelize only a supported resize, preserving per-channel quantization.

    There are at most ``threads`` outstanding one-channel calls. In particular,
    no float64 copy of the entire multi-channel prediction is made here. The
    installed implementation handles separate-z interpolation, NaNs and ties.
    """
    import numpy as np

    arguments = (new_shape, is_seg, axis, order, do_separate_z, order_z)
    # Older nnUNet releases do not expose dtype_out. An omitted/None value
    # already means the original dtype, so keep their seven-argument contract.
    if dtype_out is not None:
        arguments += (dtype_out,)
    threads = max(1, min(MAX_EXPORT_THREADS, int(threads)))
    if (threads == 1 or is_seg or order != 1 or not isinstance(data, np.ndarray)
            or data.ndim != 4 or data.shape[0] < 2 or len(new_shape) != 3
            or not np.any(np.asarray(data.shape[1:]) != np.asarray(new_shape))):
        return original_function(data, *arguments)

    # Keep nnUNet's output allocation/dtype contract, including its error on an
    # invalid output shape, rather than deriving labels before dtype conversion.
    output = np.zeros((data.shape[0], *new_shape), dtype=data.dtype if dtype_out is None else dtype_out)
    channel_iterator = iter(range(data.shape[0]))
    with ThreadPoolExecutor(max_workers=min(threads, data.shape[0]),
                            thread_name_prefix="nnunet-export") as executor:
        pending = {}

        def submit_next():
            channel = next(channel_iterator, None)
            if channel is not None:
                future = executor.submit(original_function, data[channel:channel + 1], *arguments)
                pending[future] = channel

        for _ in range(min(threads, data.shape[0])):
            submit_next()
        while pending:
            completed = wait(pending, return_when=FIRST_COMPLETED).done
            # Pop finished futures before submitting replacements so completed
            # results do not accumulate in a retained futures list.
            while completed:
                future = completed.pop()
                channel = pending.pop(future)
                output[channel] = future.result()[0]
                del future
                submit_next()
    return output


@contextmanager
def _parallel_default_resampling(function, threads):
    """Patch only inside a worker's default probability-resampling call."""
    base_function = function
    while isinstance(base_function, partial):
        base_function = base_function.func
    if (threads == 1 or getattr(base_function, "__module__", None) != _DEFAULT_RESAMPLING_MODULE
            or getattr(base_function, "__name__", None) != "resample_data_or_seg_to_shape"):
        yield
        return
    module = importlib.import_module(_DEFAULT_RESAMPLING_MODULE)
    with _RESAMPLING_LOCK:
        original = module.resample_data_or_seg
        module.resample_data_or_seg = partial(
            resample_data_or_seg, original_function=original, threads=threads,
        )
        try:
            yield
        finally:
            module.resample_data_or_seg = original


class _Proxy:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _ConfigurationProxy(_Proxy):
    @property
    def resampling_fn_probabilities(self):
        original = self._wrapped.resampling_fn_probabilities

        def resample(*args, **kwargs):
            with _parallel_default_resampling(original, export_threads()):
                return _timed_call("export_resampling", original, *args, **kwargs)

        return resample


class _LabelProxy(_Proxy):
    def convert_logits_to_segmentation(self, *args, **kwargs):
        return _timed_call("export_label_decision", self._wrapped.convert_logits_to_segmentation,
                           *args, **kwargs)

    def apply_inference_nonlin(self, *args, **kwargs):
        return _timed_call("export_inference_nonlin", self._wrapped.apply_inference_nonlin,
                           *args, **kwargs)

    def convert_probabilities_to_segmentation(self, *args, **kwargs):
        return _timed_call("export_label_decision", self._wrapped.convert_probabilities_to_segmentation,
                           *args, **kwargs)


class _WriterProxy(_Proxy):
    def write_seg(self, *args, **kwargs):
        return _timed_call("export_write_seg", self._wrapped.write_seg, *args, **kwargs)


class _PlansProxy(_Proxy):
    def get_label_manager(self, *args, **kwargs):
        return _LabelProxy(self._wrapped.get_label_manager(*args, **kwargs))

    def image_reader_writer_class(self, *args, **kwargs):
        return _WriterProxy(self._wrapped.image_reader_writer_class(*args, **kwargs))


def export_prediction_from_logits(predicted_array_or_file, properties_dict, configuration_manager,
                                  plans_manager, dataset_json_dict_or_file, output_file_truncated,
                                  save_probabilities=False, *args, **kwargs):
    """Picklable spawn-worker entry; original export performs all data handling."""
    from nnunetv2.inference.export_prediction import export_prediction_from_logits as original_export

    return _timed_call(
        "export_total", original_export, predicted_array_or_file, properties_dict,
        _ConfigurationProxy(configuration_manager), _PlansProxy(plans_manager),
        dataset_json_dict_or_file, output_file_truncated, save_probabilities, *args, **kwargs,
    )


def install_export_adapter():
    """Install the stable-module callable that official nnUNet sends to spawn."""
    prediction = importlib.import_module("nnunetv2.inference.predict_from_raw_data")
    prediction.export_prediction_from_logits = export_prediction_from_logits
    return prediction


def main(entry_started=None):
    started = _ENTRY_STARTED if entry_started is None else entry_started
    phase_timing("entry_imports", started, scope="script entry to main; excludes interpreter/site startup")
    status = "failed"
    try:
        prediction = timed_phase("engine_import", install_export_adapter,
                                 scope="import installed nnUNet and its dependencies")
        install_prediction_timing(prediction)
        sys.argv[0] = "nnUNetv2_predict"
        result = timed_phase("cli_total", prediction.predict_entry_point,
                             scope="official CLI including model initialization and prediction; inclusive")
        status = "completed"
        return result
    finally:
        phase_timing("entry_script_total", started, status=status,
                     scope="script entry until CLI returns; excludes interpreter startup and exit teardown")


if __name__ == "__main__":
    # Workers import this file as nnunet_export, never as a transient __main__.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(importlib.import_module("nnunet_export").main(entry_started=_ENTRY_STARTED))
