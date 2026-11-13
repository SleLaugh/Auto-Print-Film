"""GPU label merge preserving the AutoPrintFilm GPU resampler's arithmetic.

No installed nnUNet files are changed. Unsupported requests explicitly invoke
the existing export entry and are reported as fallback, never as S6 success.
"""
from __future__ import annotations

from functools import partial
import gc
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import nnunet_gpu_export as old_export
import nnunet_gpu_resampling as reference


LAST_EXPORT_INFO = {}
_MIB = 1024**2


class MergeFallback(RuntimeError):
    """A deliberate request to retain the old export's behavior."""


def _finite_array(data):
    # Bound temporary boolean arrays to one channel.
    return all(np.isfinite(channel).all() for channel in data)


def _merge_block(values, first_channel, best, labels, invalid):
    invalid.logical_or_(~torch.isfinite(values).all())
    for local in range(values.shape[0]):
        channel = first_channel + local
        if channel == 0:
            best.copy_(values[local])
        else:
            update = values[local] > best
            best.copy_(torch.where(update, values[local], best))
            labels.masked_fill_(update, channel)


def _upload(array, info, *, preserve_dtype=False):
    started = time.perf_counter()
    result = torch.from_numpy(np.ascontiguousarray(array)).to(
        device='cuda', dtype=None if preserve_dtype else torch.float32,
    )
    info['h2d_host_seconds'] += time.perf_counter() - started
    info['upload_calls'] += 1
    return result


def _sample_tiles(uploaded, shape, dtype, best, labels, channel, invalid, info,
                  reference_budget):
    # Exact operations and tile selection from reference._linear_tiled.
    residual = max(_MIB, reference_budget - uploaded.numel() * uploaded.element_size())
    maximum = max(1, min(reference._MAX_TILE_VOXELS, residual // (4 * (1 + 8))))
    tile = reference._tile_shape(shape, maximum)
    info.update(spatial_tile_shape=list(tile), gpu_tiled=True)
    for begin in itertools.product(*(range(0, n, t) for n, t in zip(shape, tile))):
        end = tuple(min(b + t, n) for b, t, n in zip(begin, tile, shape))
        coordinates = [
            (2.0 * (torch.arange(b, e, dtype=torch.float32, device='cuda') + 0.5) / n) - 1.0
            for b, e, n in zip(begin, end, shape)
        ]
        zz, yy, xx = torch.meshgrid(*coordinates, indexing='ij')
        grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(uploaded, grid, mode='bilinear',
                                padding_mode='border', align_corners=False)[0].to(dtype=dtype)
        slices = tuple(slice(b, e) for b, e in zip(begin, end))
        _merge_block(sampled, channel, best[slices], labels[slices], invalid)
        info['gpu_blocks'] += 1
        del coordinates, zz, yy, xx, grid, sampled


@torch.inference_mode()
def merge_logits(data, new_shape, *, do_separate_z=False, axis=None, info=None):
    """Return a uint8 mask equal to old resample_logits + NumPy argmax.

Only finite float16/32/64 logits are accepted. Nonfinite intermediate results
also request fallback so NumPy's original NaN argmax semantics remain intact.
"""
    info = {} if info is None else info
    info.update(upload_calls=0, h2d_host_seconds=0.0, gpu_blocks=0, gpu_tiled=False)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if not isinstance(data, np.ndarray) or data.ndim != 4:
        raise MergeFallback('expected CXYZ NumPy or Torch logits')
    shape = tuple(int(value) for value in new_shape)
    if len(shape) != 3 or min(shape) < 1 or not 1 <= data.shape[0] <= 255:
        raise MergeFallback('positive 3D shape and 1..255 channels required')
    dtype = np.dtype(data.dtype)
    if dtype not in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')):
        raise MergeFallback('only float16/32/64 logits supported')
    if not _finite_array(data):
        raise MergeFallback('nonfinite logits retain original NumPy argmax behavior')
    axis = reference._axis_int(axis)
    if do_separate_z and axis not in (0, 1, 2):
        raise MergeFallback('separate-z requires a single spatial axis')
    if not torch.cuda.is_available():
        raise MergeFallback('CUDA unavailable')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    reference_budget = int(min(reference._MAX_WORK_BYTES, free * 0.65))
    try:
        requested_gib = float(os.environ.get('S6_MERGE_WORK_GIB', '4'))
    except ValueError as error:
        raise MergeFallback('S6_MERGE_WORK_GIB must be numeric') from error
    if not math.isfinite(requested_gib) or requested_gib <= 0:
        raise MergeFallback('S6_MERGE_WORK_GIB must be positive and finite')
    budget = min(reference_budget, int(requested_gib * 1024**3))
    identity = tuple(data.shape[1:]) == shape
    if not identity and not do_separate_z:
        minimum = math.prod(data.shape[1:]) * (4 + dtype.itemsize) + 64 * _MIB
        if minimum > reference_budget:
            raise MergeFallback('old resampler would use its CPU work-budget fallback')
    torch_dtype = reference._dtype_to_torch(dtype)
    inverse = None
    if do_separate_z and not identity:
        indices = reference.nearest_indices(data.shape[axis + 1], shape[axis])
        unique, inverse = np.unique(indices, return_inverse=True)
        plane_shape = tuple(n for index, n in enumerate(shape) if index != axis)
        source_plane = tuple(n for index, n in enumerate(data.shape[1:]) if index != axis)
        merge_shape = (len(unique), *plane_shape)
        reference_plane_bytes = 4 * (math.prod(source_plane) + 2 * math.prod(plane_shape))
        reference_batch = max(1, min(32, reference_budget // max(1, reference_plane_bytes)))
        scratch_unit = 4 * math.prod(source_plane) + (4 + dtype.itemsize * 2 + 1) * math.prod(plane_shape)
    else:
        merge_shape = shape
        ninput, noutput = math.prod(data.shape[1:]), math.prod(shape)
        reference_channel_bytes = 4 * (ninput + 2 * noutput)
        reference_batch = max(1, min(4, reference_budget // max(1, reference_channel_bytes)))
        scratch_unit = (dtype.itemsize * ninput if identity else 4 * ninput) + (4 + dtype.itemsize * 2 + 1) * noutput
    persistent = math.prod(merge_shape) * (dtype.itemsize + 1)
    available = budget - persistent - 64 * _MIB
    if available <= 0:
        raise MergeFallback('GPU merge score/label buffers exceed work budget')
    tiled = not identity and not do_separate_z and reference_channel_bytes > reference_budget
    if tiled:
        residual = max(_MIB, reference_budget - 4 * ninput)
        max_voxels = max(1, min(reference._MAX_TILE_VOXELS, residual // 36))
        # Preserve the old sampler's exact tile geometry; decline if it cannot fit.
        scratch_needed = 4 * ninput + max_voxels * (32 + dtype.itemsize * 2)
        if scratch_needed > available:
            raise MergeFallback('reference spatial tiles plus GPU merge exceed work budget')
        batch = 1
    else:
        batch = min(reference_batch, available // max(1, scratch_unit))
        if batch < 1:
            raise MergeFallback('one interpolation block plus GPU merge exceeds work budget')
    info.update(input_shape=list(data.shape), output_shape=list(shape), input_dtype=str(dtype),
                quantization_dtype=str(dtype), work_budget_bytes=budget,
                reference_work_budget_bytes=reference_budget, persistent_bytes=persistent,
                cuda_free_before_bytes=int(free), cuda_total_bytes=int(total),
                channel_block_size=1 if do_separate_z and not identity else int(batch),
                plane_batch_size=int(batch) if do_separate_z and not identity else None,
                reference_batch_size=int(reference_batch), identity=identity,
                separate_z=bool(do_separate_z), anisotropic_axis=axis,
                interpolation='identity' if identity else 'bilinear' if do_separate_z else 'grid_sample' if tiled else 'trilinear',
                align_corners=False, class_comparison='strict > after original dtype quantization',
                original_logits_download_bytes=math.prod(shape) * data.shape[0] * dtype.itemsize,
                final_mask_download_bytes=math.prod(merge_shape))
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    best = labels = invalid = uploaded = sampled = values = None
    started = time.perf_counter()
    try:
        best = torch.empty(merge_shape, dtype=torch_dtype, device='cuda')
        labels = torch.zeros(merge_shape, dtype=torch.uint8, device='cuda')
        invalid = torch.zeros((), dtype=torch.bool, device='cuda')
        if do_separate_z and not identity:
            for channel in range(data.shape[0]):
                source = np.moveaxis(data[channel], axis, 0)
                for offset in range(0, len(unique), batch):
                    ids = unique[offset:offset + batch]
                    uploaded = _upload(source[ids], info).unsqueeze(1)
                    sampled = F.interpolate(uploaded, size=plane_shape, mode='bilinear', align_corners=False)
                    values = sampled[:, 0].to(dtype=torch_dtype).unsqueeze(0)
                    _merge_block(values, channel, best[offset:offset + len(ids)],
                                 labels[offset:offset + len(ids)], invalid)
                    info['gpu_blocks'] += 1
                    uploaded = sampled = values = None
        else:
            for channel in range(0, data.shape[0], batch):
                uploaded = _upload(data[channel:channel + batch], info, preserve_dtype=identity).unsqueeze(0)
                if tiled:
                    _sample_tiles(uploaded, shape, torch_dtype, best, labels, channel,
                                  invalid, info, reference_budget)
                else:
                    sampled = uploaded if identity else F.interpolate(
                        uploaded, size=shape, mode='trilinear', align_corners=False)
                    values = sampled[0].to(dtype=torch_dtype)
                    _merge_block(values, channel, best, labels, invalid)
                    info['gpu_blocks'] += 1
                uploaded = sampled = values = None
        if invalid.item():
            raise MergeFallback('nonfinite interpolated logits require original argmax')
        torch.cuda.synchronize()
        info['resample_merge_seconds'] = time.perf_counter() - started
        started_download = time.perf_counter()
        result = labels.cpu().numpy()
        info['mask_download_seconds'] = time.perf_counter() - started_download
        info['mask_download_calls'] = 1
        info['cuda_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        info['cuda_peak_extra_bytes'] = max(0, info['cuda_peak_allocated_bytes'] - before)
        if inverse is not None:
            result = np.moveaxis(result[inverse], 0, axis)
        return result
    finally:
        best = labels = invalid = uploaded = sampled = values = None


def _configuration(configuration, plans, properties):
    original = configuration.resampling_fn_probabilities
    function = original
    while isinstance(function, partial):
        function = function.func
    options = dict(configuration.configuration.get('resampling_fn_probabilities_kwargs', {}))
    if (getattr(function, '__module__', None) != old_export._MODULE
            or getattr(function, '__name__', None) != 'resample_data_or_seg_to_shape'
            or set(options) - old_export._SUPPORTED_KWARGS
            or options.get('is_seg', False) is not False or options.get('order', 3) != 1):
        raise MergeFallback('unsupported configured probability resampler')
    spacing = [properties['spacing'][i] for i in plans.transpose_forward]
    current = configuration.spacing if len(configuration.spacing) == len(
        properties['shape_after_cropping_and_before_resampling']) else [spacing[0], *configuration.spacing]
    upstream = importlib.import_module(old_export._MODULE)
    separate, axis = upstream.determine_do_sep_z_and_axis(
        options.get('force_separate_z', False), current, spacing,
        options.get('separate_z_anisotropy_threshold', upstream.ANISO_THRESHOLD))
    if separate and options.get('order_z', 0) != 0:
        raise MergeFallback('separate-z order_z must be zero')
    return separate, axis


def export_prediction_from_logits(predicted, properties, configuration_manager, plans_manager,
                                  dataset_json_dict_or_file, output_file_truncated,
                                  save_probabilities=False, *args, **kwargs):
    global LAST_EXPORT_INFO
    started = time.perf_counter()
    info = {'backend': 'pending', 'fallback': False, 'completed': False}
    LAST_EXPORT_INFO = info
    fallback_reason = None
    segmentation = None
    try:
        if save_probabilities:
            raise MergeFallback('probability export retains original path')
        if old_export._device() != 'gpu' or not old_export._enabled():
            raise MergeFallback('GPU resampling disabled by original entry configuration')
        if len(args) > 1 or set(kwargs) - {'num_threads_torch'}:
            raise MergeFallback('unrecognized export arguments')
        dataset = dataset_json_dict_or_file
        if isinstance(dataset, (str, Path)):
            dataset = json.loads(Path(dataset).read_text(encoding='utf-8'))
        manager = plans_manager.get_label_manager(dataset)
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        if type(manager) is not LabelManager or manager.has_regions or manager.has_ignore_label:
            raise MergeFallback('requires standard mutually exclusive LabelManager')
        if manager.all_labels != list(range(manager.num_segmentation_heads)):
            raise MergeFallback('requires continuous labels starting at zero')
        if not hasattr(predicted, 'shape') or predicted.shape[0] != manager.num_segmentation_heads:
            raise MergeFallback('logits and label count differ')
        separate, axis = _configuration(configuration_manager, plans_manager, properties)
        segmentation = merge_logits(predicted, properties['shape_after_cropping_and_before_resampling'],
                                    do_separate_z=separate, axis=axis, info=info)
    except (MergeFallback, torch.cuda.OutOfMemoryError) as error:
        fallback_reason = str(error)
        error.__traceback__ = None
    if fallback_reason is not None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info.update(backend='original_export_fallback', fallback=True, fallback_reason=fallback_reason)
        print('[S6 GPU merge] ' + json.dumps(info), flush=True)
        try:
            result = old_export.export_prediction_from_logits(
                predicted, properties, configuration_manager, plans_manager,
                dataset_json_dict_or_file, output_file_truncated, save_probabilities, *args, **kwargs)
            info['completed'] = True
            return result
        finally:
            info['export_seconds'] = time.perf_counter() - started
            old_export.baseline.phase_timing('s6_export_total', started,
                                            status='completed' if info['completed'] else 'failed',
                                            scope='worker export including fallback; nested export timings overlap')
            print('[S6 GPU merge] ' + json.dumps(info), flush=True)
    try:
        restore_started = time.perf_counter()
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        restored = np.zeros(properties['shape_before_cropping'], dtype=np.uint8)
        restored = insert_crop_into_image(restored, segmentation, properties['bbox_used_for_cropping'])
        restored = restored.transpose(plans_manager.transpose_backward)
        info['geometry_restore_seconds'] = time.perf_counter() - restore_started
        write_started = time.perf_counter()
        plans_manager.image_reader_writer_class().write_seg(
            restored, str(output_file_truncated) + dataset['file_ending'], properties)
        info['write_seconds'] = time.perf_counter() - write_started
        info.update(backend='s6_gpu_resample_dtype_quantized_merge', completed=True)
    finally:
        info['export_seconds'] = time.perf_counter() - started
        old_export.baseline.phase_timing('s6_export_total', started,
                                        status='completed' if info['completed'] else 'failed',
                                        scope='worker export including geometry restore and write; overlaps parent export wait')
        print('[S6 GPU merge] ' + json.dumps(info), flush=True)
