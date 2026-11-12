"""External nnUNet entry with configurable CPU/CUDA logit resampling.

Reuse the CPU adapter for CPU requests and unsupported plans. Model weights,
plans and installed nnUNet files are never overwritten.
"""
from functools import partial
import importlib
import json
import os
import sys
import time

import numpy as np
import nnunet_export as baseline
import nnunet_gpu_resampling as gpu_resampling

_MODULE = 'nnunetv2.preprocessing.resampling.default_resampling'
_SUPPORTED_KWARGS = {'is_seg', 'order', 'order_z', 'force_separate_z',
                     'separate_z_anisotropy_threshold'}
_DEVICE_ENV = 'AUTOPRINTFILM_NNUNET_RESAMPLING_DEVICE'
_SCHEME_ENV = 'AUTOPRINTFILM_NNUNET_ACCELERATION_SCHEME'


def _scheme():
    value = os.environ.get(_SCHEME_ENV, 'scheme2')
    if value not in ('scheme2', 'scheme6'):
        raise ValueError(f'{_SCHEME_ENV} must be scheme2 or scheme6')
    return value


def _device():
    value = os.environ.get(_DEVICE_ENV, 'cpu')
    if value not in ('cpu', 'gpu'):
        raise ValueError(f'{_DEVICE_ENV} must be cpu or gpu')
    return value

def _enabled():
    value = os.environ.get('NNUNET_GPU_RESAMPLING', '1')
    if value not in ('0', '1'):
        raise ValueError('NNUNET_GPU_RESAMPLING must be 0 or 1')
    return value == '1'

def _threads():
    return baseline.export_threads()

class ConfigurationProxy(baseline._Proxy):
    @property
    def resampling_fn_probabilities(self):
        original = self._wrapped.resampling_fn_probabilities
        function = original
        while isinstance(function, partial):
            function = function.func
        options = dict(self._wrapped.configuration.get('resampling_fn_probabilities_kwargs', {}))

        def resample(data, new_shape, current_spacing, new_spacing, **kwargs):
            started = time.perf_counter()
            effective = dict(options)
            effective.update(kwargs)
            compatible = (getattr(function, '__module__', None) == _MODULE
                and getattr(function, '__name__', None) == 'resample_data_or_seg_to_shape'
                and not (set(effective) - _SUPPORTED_KWARGS)
                and effective.get('is_seg', False) is False
                and effective.get('order', 3) == 1)
            if not _enabled() or not compatible:
                with baseline._parallel_default_resampling(original, _threads()):
                    result = original(data, new_shape, current_spacing, new_spacing, **kwargs)
                info = dict(backend='configured_resampler',
                            reason='disabled' if not _enabled() else 'unsupported plans function or options')
            else:
                upstream = importlib.import_module(_MODULE)
                separate, axis = upstream.determine_do_sep_z_and_axis(
                    effective.get('force_separate_z', False), current_spacing, new_spacing,
                    effective.get('separate_z_anisotropy_threshold', upstream.ANISO_THRESHOLD))
                if not isinstance(data, np.ndarray):
                    data = data.detach().cpu().numpy()
                result = gpu_resampling.resample_logits(
                    data, new_shape, is_seg=False, axis=axis, order=1,
                    do_separate_z=separate, order_z=effective.get('order_z', 0),
                    threads=_threads())
                info = dict(gpu_resampling.LAST_RUN_INFO)
            print(f'[AutoPrintFilm nnUNet] export_resampling seconds={time.perf_counter()-started:.6f}', flush=True)
            print('[GPU resampling] ' + json.dumps(info), flush=True)
            return result
        return resample

def export_prediction_from_logits(predicted, properties, configuration_manager, plans_manager,
                                  dataset_json_dict_or_file, output_file_truncated,
                                  save_probabilities=False, *args, **kwargs):
    # The application selects this per process from the actual model ID.
    # Workers inherit it through spawn; no dataset ID is hardcoded here.
    requested_device = _device()
    if requested_device == 'cpu' or not _enabled():
        print('[GPU resampling] ' + json.dumps({
            'backend': 'configured_resampler',
            'reason': 'CPU requested' if requested_device == 'cpu' else 'disabled',
            'requested_device': requested_device,
        }), flush=True)
        return baseline.export_prediction_from_logits(
            predicted, properties, configuration_manager, plans_manager,
            dataset_json_dict_or_file, output_file_truncated, save_probabilities, *args, **kwargs)
    from nnunetv2.inference.export_prediction import export_prediction_from_logits as original
    return baseline._timed_call('export_total', original, predicted, properties,
        ConfigurationProxy(configuration_manager), baseline._PlansProxy(plans_manager),
        dataset_json_dict_or_file, output_file_truncated, save_probabilities, *args, **kwargs)

def main():
    # Spawn workers must be able to import this real module and its top-level
    # callable. No lambdas/closures are submitted to the multiprocessing pool.
    _enabled()
    _device()
    scheme = _scheme()
    # Explicit CPU/MPS prediction must not start CUDA in the export workers.
    # Leave parsing/validation of the complete CLI to nnUNet itself.
    device = 'cuda'
    for index, argument in enumerate(sys.argv[1:], 1):
        if argument == '-device' and index + 1 < len(sys.argv):
            device = sys.argv[index + 1]
        elif argument.startswith('-device='):
            device = argument.split('=', 1)[1]
    if device in ('cpu', 'mps'):
        os.environ['NNUNET_GPU_RESAMPLING'] = '0'
    os.environ.setdefault(baseline.EXPORT_THREADS_ENV, '4')
    prediction = importlib.import_module('nnunetv2.inference.predict_from_raw_data')
    prediction.export_prediction_from_logits = export_prediction_from_logits
    active_scheme = 'scheme2'
    if scheme == 'scheme6' and _device() == 'gpu' and _enabled():
        from nnunet_s6_adapter import install_scheme6
        install_scheme6(prediction)
        active_scheme = 'scheme6'
    print('[AutoPrintFilm nnUNet] ' + json.dumps({
        'requested_scheme': scheme, 'active_scheme': active_scheme,
        'requested_device': _device(), 'gpu_enabled': _enabled(),
    }), flush=True)
    sys.argv[0] = 'nnUNetv2_predict'
    return prediction.predict_entry_point()


if __name__ == '__main__':
    # Spawn workers import a stable module instead of pickling __main__.
    sys.exit(importlib.import_module('nnunet_gpu_export').main())
