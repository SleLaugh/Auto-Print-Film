"""CUDA cubic input resampling using the installed nnU-Net image conventions.

Matches this installation's skimage.resize(mode='edge', anti_aliasing=False):
float64 spline work and output, half-pixel/grid_mode=True coordinates, and
range clipping before casting to the original dtype. Separate-axis resizing
uses the same per-plane operation and float64 half-pixel nearest indices.
No installed modules are modified and no CPU fallback is performed.
"""
from __future__ import annotations

from functools import partial
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

LAST_RUN_INFO = {}
_CUPY = None
_SUPPORTED_OPTIONS = {
    'is_seg', 'order', 'order_z', 'force_separate_z',
    'separate_z_anisotropy_threshold',
}


def _load_cupy():
    global _CUPY
    if _CUPY is None:
        dependency_path = os.environ.get('GPU_INPUT_CUPY_PATH')
        if dependency_path:
            dependency_directory = Path(dependency_path).expanduser().resolve()
            if not dependency_directory.is_dir():
                raise RuntimeError(f'GPU_INPUT_CUPY_PATH is not a directory: {dependency_directory}')
            if str(dependency_directory) not in sys.path:
                sys.path.insert(0, str(dependency_directory))
        try:
            _CUPY = (importlib.import_module('cupy'),
                     importlib.import_module('cupyx.scipy.ndimage'))
        except ImportError as error:
            raise RuntimeError('CuPy backend unavailable; no CPU substitution allowed') from error
    return _CUPY


def nearest_indices_float64(old_length, new_length):
    """Indices from installed SciPy map_coordinates(order=0, mode=nearest)."""
    coordinates = (float(old_length) / new_length
                   * (np.arange(new_length, dtype=np.float64) + 0.5) - 0.5)
    return np.clip(np.floor(coordinates + 0.5), 0, old_length - 1).astype(np.intp)


def estimate_cubic_work_bytes(source_shape, target_shape, *, do_separate_z=False, axis=None):
    source_shape = tuple(int(n) for n in source_shape)
    target_shape = tuple(int(n) for n in target_shape)
    if do_separate_z:
        if axis not in (0, 1, 2):
            raise ValueError('Separate-axis mode requires axis 0, 1, or 2')
        source_shape = tuple(n for i, n in enumerate(source_shape) if i != axis)
        target_shape = tuple(n for i, n in enumerate(target_shape) if i != axis)
    source_bytes = math.prod(source_shape) * 8
    padded_bytes = math.prod(n + 24 for n in source_shape) * 8
    target_bytes = math.prod(target_shape) * 8
    return {
        'source_float64_bytes': source_bytes,
        'padded_float64_bytes': padded_bytes,
        'target_float64_bytes': target_bytes,
        'estimated_peak_bytes': source_bytes + padded_bytes * 2 + target_bytes + 64 * 1024**2,
        'budget_kind': 'one_plane' if do_separate_z else 'one_volume',
        'padding_per_side': 12,
        'estimate_is_not_allocator_guarantee': True,
    }


def _gpu_resize(cp, ndi, array, shape):
    values = result = None
    try:
        # Explicit float64 output matters: CPU clips in float64 and casts only
        # when assigning to reshaped_final; rounding before clip is different.
        values = cp.asarray(np.ascontiguousarray(array, dtype=np.float64))
        factors = [1.0 / (float(old) / new) for old, new in zip(array.shape, shape)]
        result = ndi.zoom(values, factors, output=cp.float64, order=3,
                          mode='nearest', prefilter=True, grid_mode=True)
        if tuple(result.shape) != tuple(shape):
            raise RuntimeError(f'Unexpected CUDA resize shape {result.shape}; requested {shape}')
        lower = np.min(array)
        if np.isnan(lower):
            lower, upper = np.nanmin(array), np.nanmax(array)
        else:
            upper = np.max(array)
        cp.clip(result, lower, upper, out=result)
        return cp.asnumpy(result)
    finally:
        values = result = None


def resample_data_or_seg_gpu(data, new_shape, is_seg=False, axis=None, order=3,
                             do_separate_z=False, order_z=0, dtype_out=None, *,
                             max_pool_gib=8.0):
    """CUDA cubic replacement with a hard CuPy allocator cap of at most 8 GiB."""
    global LAST_RUN_INFO
    started = time.perf_counter()
    source = np.asarray(data)
    shape = tuple(int(n) for n in new_shape)
    info = {
        'backend': 'pending', 'input_shape': list(source.shape),
        'target_shape': list(shape), 'input_dtype': str(source.dtype),
        'order': int(order), 'order_z': int(order_z),
        'separate_z': bool(do_separate_z), 'axis': None if axis is None else int(axis),
        'coordinates': 'skimage_half_pixel_grid_mode_true',
        'separate_axis_coordinates': 'float64_half_pixel_nearest',
        'coefficient_precision': 'float64', 'interpolated_precision': 'float64',
        'clip': True, 'anti_aliasing': False, 'cpu_fallback_allowed': False,
    }
    LAST_RUN_INFO = info
    cp = None
    try:
        if source.ndim != 4 or len(shape) != 3 or any(n <= 0 for n in shape):
            raise ValueError('Expected CXYZ data and positive 3D target shape')
        if (is_seg or order != 3
                or (do_separate_z and (axis not in (0, 1, 2) or order_z != 0))):
            raise ValueError('Only cubic image resize with optional separate-axis order_z=0 is supported')
        if not np.issubdtype(source.dtype, np.floating):
            raise ValueError('Normalized floating image input required')
        if tuple(source.shape[1:]) == shape:
            info['backend'] = 'identity'
            return data
        requested_gib = float(max_pool_gib)
        if not math.isfinite(requested_gib) or requested_gib <= 0:
            raise ValueError('max_pool_gib must be finite and positive')
        budget = int(min(8.0, requested_gib) * 1024**3)
        estimate = estimate_cubic_work_bytes(source.shape[1:], shape,
                                            do_separate_z=do_separate_z, axis=axis)
        info.update(work_memory_estimate=estimate, requested_work_gib=requested_gib)
        if estimate['estimated_peak_bytes'] > budget:
            raise RuntimeError('Estimated cubic work exceeds configured limit (at most 8 GiB)')
        cp, ndi = _load_cupy()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        budget = min(budget, int(free_bytes * 0.70))
        info.update(cuda_free_before_bytes=int(free_bytes), cuda_total_bytes=int(total_bytes),
                    max_cupy_pool_bytes=budget, cupy_version=cp.__version__)
        if estimate['estimated_peak_bytes'] > budget:
            raise RuntimeError('Estimated cubic work exceeds 70% of free GPU memory')
        cp.get_default_memory_pool().set_limit(size=budget)
        info.update(backend='cupyx_cuda_cubic_skimage', gpu_calls=0)
        output = np.empty((source.shape[0], *shape), dtype=source.dtype if dtype_out is None else dtype_out)
        if do_separate_z:
            # Cubic is independent between source planes; nearest sampling
            # needs only the source planes selected by the float64 coordinates.
            ids = nearest_indices_float64(source.shape[axis + 1], shape[axis])
            plane_shape = tuple(n for i, n in enumerate(shape) if i != axis)
            for channel in range(source.shape[0]):
                old_planes = np.moveaxis(source[channel], axis, 0)
                new_planes = np.moveaxis(output[channel], axis, 0)
                for old_id in np.unique(ids):
                    resized = _gpu_resize(cp, ndi, old_planes[old_id], plane_shape)
                    for new_id in np.flatnonzero(ids == old_id):
                        new_planes[new_id] = resized
                    info['gpu_calls'] += 1
        else:
            for channel in range(source.shape[0]):
                output[channel] = _gpu_resize(cp, ndi, source[channel], shape)
                info['gpu_calls'] += 1
        cp.cuda.get_current_stream().synchronize()
        info.update(cupy_pool_used_bytes_after=int(cp.get_default_memory_pool().used_bytes()),
                    cupy_pool_reserved_bytes_after=int(cp.get_default_memory_pool().total_bytes()))
        return output
    except Exception as error:
        info.update(backend='failed_no_substitution', error=str(error))
        raise
    finally:
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
        info['seconds'] = time.perf_counter() - started
        print('[S6 GPU input resampling] ' + json.dumps(info), flush=True)


def resample_to_shape(data, new_shape, current_spacing, new_spacing, is_seg=False,
                      order=3, order_z=0, force_separate_z=False,
                      separate_z_anisotropy_threshold=3, *, max_pool_gib=8.0):
    from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis
    separate, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold)
    return resample_data_or_seg_gpu(data, new_shape, is_seg=is_seg, order=order,
                                    do_separate_z=separate, axis=axis, order_z=order_z,
                                    max_pool_gib=max_pool_gib)


class InputConfigurationProxy:
    """Spawn-safe proxy; the returned resampling callable is module-level."""
    def __init__(self, wrapped, *, max_pool_gib=8.0):
        self._wrapped = wrapped
        self.max_pool_gib = max_pool_gib

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_wrapped'), name)

    @property
    def resampling_fn_data(self):
        original = self._wrapped.resampling_fn_data
        options = {}
        function = original
        while isinstance(function, partial):
            options = {**function.keywords, **options}
            if function.args:
                raise ValueError('Positional partial resampling arguments are unsupported')
            function = function.func
        if (function.__module__ != 'nnunetv2.preprocessing.resampling.default_resampling'
                or function.__name__ != 'resample_data_or_seg_to_shape'
                or set(options) - _SUPPORTED_OPTIONS or options.get('is_seg', False)
                or options.get('order', 3) != 3):
            raise ValueError('Unsupported input resampler; refusing silent algorithm substitution')
        return partial(resample_to_shape, **options, max_pool_gib=self.max_pool_gib)
