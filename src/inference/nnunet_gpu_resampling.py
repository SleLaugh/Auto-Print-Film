"""Bounded CUDA logit resampling used by the external export adapter.

This file never edits the installed nnUNet implementation. It accelerates only
linear floating-point logit resampling. Unsupported requests use the original
nnUNet function, with the reason recorded in LAST_RUN_INFO. CUDA arithmetic is
float32, so agreement with the float64 SciPy reference must be measured.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.preprocessing.resampling.default_resampling import (
    resample_data_or_seg as _REFERENCE,
)
from nnunet_export import resample_data_or_seg as _CPU_RESAMPLE


LAST_RUN_INFO: dict[str, Any] = {}
_GIB = 1024**3
_MAX_WORK_BYTES = 4 * _GIB
_MAX_TILE_VOXELS = 4 * 1024**2


def nearest_indices(nin: int, nout: int) -> np.ndarray:
    """SciPy order=0, mode='nearest', using the reference's float64 mapping."""
    coords = (float(nin) / nout) * (np.arange(nout, dtype=np.float64) + 0.5) - 0.5
    return np.clip(np.floor(coords + 0.5), 0, nin - 1).astype(np.int64)


def _dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    return {np.dtype('float16'): torch.float16,
            np.dtype('float32'): torch.float32,
            np.dtype('float64'): torch.float64}[np.dtype(dtype)]


def _axis_int(axis: Any) -> int | None:
    if axis is None:
        return None
    arr = np.asarray(axis).reshape(-1)
    return int(arr[0]) if len(arr) == 1 else None


def _upload(arr: np.ndarray, info: dict[str, Any]) -> torch.Tensor:
    start = time.perf_counter()
    # Contiguous copying is restricted to the current channel/slice block.
    uploaded = torch.from_numpy(np.ascontiguousarray(arr)).to(
        device='cuda', dtype=torch.float32
    )
    torch.cuda.synchronize()
    info['h2d_seconds'] += time.perf_counter() - start
    return uploaded


def _download(value: torch.Tensor, dtype: np.dtype, info: dict[str, Any]) -> np.ndarray:
    start = time.perf_counter()
    result = value.to(dtype=_dtype_to_torch(dtype)).cpu().numpy()
    torch.cuda.synchronize()
    info['d2h_seconds'] += time.perf_counter() - start
    return result


def _tile_shape(shape: tuple[int, ...], max_voxels: int) -> tuple[int, ...]:
    tile = list(shape)
    while math.prod(tile) > max_voxels:
        biggest = max(range(len(tile)), key=tile.__getitem__)
        tile[biggest] = max(1, (tile[biggest] + 1) // 2)
    return tuple(tile)


def _linear_tiled(inp: torch.Tensor, shape: tuple[int, int, int], out: np.ndarray,
                  dtype: np.dtype, info: dict[str, Any], budget: int) -> None:
    """Sample tiles with global output coordinates; never resize local tiles.

    inp has all input voxels for the current channel block. A tile grid uses
    global j in normalized align_corners=False coordinates, and border padding
    is the equivalent of the reference's edge clamping.
    """
    channels = inp.shape[1]
    # Grid has three float32 components plus C float32 output components. Keep
    # substantial spare memory for grid construction and interpolation internals.
    tile_voxels = max(1, min(_MAX_TILE_VOXELS, budget // (4 * (channels + 8))))
    tile = _tile_shape(shape, tile_voxels)
    info['spatial_tile_shape'] = list(tile)
    info['gpu_tiled'] = True
    ranges = [range(0, n, t) for n, t in zip(shape, tile)]
    for begin in itertools.product(*ranges):
        end = tuple(min(b + t, n) for b, t, n in zip(begin, tile, shape))
        start = time.perf_counter()
        coords = [
            (2.0 * (torch.arange(b, e, dtype=torch.float32, device='cuda') + 0.5) / n) - 1.0
            for b, e, n in zip(begin, end, shape)
        ]
        zz, yy, xx = torch.meshgrid(*coords, indexing='ij')
        grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(inp, grid, mode='bilinear', padding_mode='border',
                                align_corners=False)
        torch.cuda.synchronize()
        info['gpu_compute_seconds'] += time.perf_counter() - start
        copied = _download(sampled[0], dtype, info)
        slices = tuple(slice(b, e) for b, e in zip(begin, end))
        out[(slice(None), *slices)] = copied
        del coords, zz, yy, xx, grid, sampled, copied
        info['gpu_blocks'] += 1


def _resample_volume(data: np.ndarray, shape: tuple[int, int, int], out: np.ndarray,
                     dtype: np.dtype, info: dict[str, Any], budget: int) -> None:
    # Whole input channels remain on-device, while output channels are streamed
    # back to CPU. The full 27-channel output is never placed on the GPU.
    nin = math.prod(data.shape[1:])
    nout = math.prod(shape)
    per_channel = 4 * (nin + 2 * nout)  # output plus dtype-conversion allowance
    block = max(1, min(4, budget // max(1, per_channel)))
    info['channel_block_size'] = block
    for c in range(0, data.shape[0], block):
        stop = min(c + block, data.shape[0])
        inp = _upload(data[c:stop], info).unsqueeze(0)
        if 4 * (nin + 2 * nout) * (stop - c) <= budget:
            start = time.perf_counter()
            result = F.interpolate(inp, size=shape, mode='trilinear',
                                   align_corners=False)
            torch.cuda.synchronize()
            info['gpu_compute_seconds'] += time.perf_counter() - start
            out[c:stop] = _download(result[0], dtype, info)
            del result
            info['gpu_blocks'] += 1
        else:
            residual = max(1024**2, budget - inp.numel() * inp.element_size())
            _linear_tiled(inp, shape, out[c:stop], dtype, info, residual)
        del inp


def _resample_separate(data: np.ndarray, shape: tuple[int, int, int], axis: int,
                       out: np.ndarray, dtype: np.dtype, info: dict[str, Any],
                       budget: int) -> None:
    # First interpolate each required source plane; then restore the anisotropic
    # axis using exactly the reference's order=0 integer source index. Planes
    # omitted by the nearest-neighbor step need not be computed.
    indices = nearest_indices(data.shape[axis + 1], shape[axis])
    plane_shape = tuple(n for i, n in enumerate(shape) if i != axis)
    src_plane_shape = tuple(n for i, n in enumerate(data.shape[1:]) if i != axis)
    unique = np.unique(indices)
    per_plane = 4 * (math.prod(src_plane_shape) + 2 * math.prod(plane_shape))
    plane_batch = max(1, min(32, budget // max(1, per_plane)))
    info['channel_block_size'] = 1
    info['plane_batch_size'] = plane_batch
    info['separate_z'] = True
    info['anisotropic_axis'] = axis
    info['required_source_planes'] = int(len(unique))
    info['source_planes'] = int(data.shape[axis + 1])
    for c in range(data.shape[0]):
        source = np.moveaxis(data[c], axis, 0)
        target = np.moveaxis(out[c], axis, 0)
        for offset in range(0, len(unique), plane_batch):
            ids = unique[offset:offset + plane_batch]
            inp = _upload(source[ids], info).unsqueeze(1)
            start = time.perf_counter()
            result = F.interpolate(inp, size=plane_shape, mode='bilinear',
                                   align_corners=False)
            torch.cuda.synchronize()
            info['gpu_compute_seconds'] += time.perf_counter() - start
            copied = _download(result[:, 0], dtype, info)
            # Loop over planes avoids a volume-size advanced-indexing temporary.
            for local, src_id in enumerate(ids):
                for target_id in np.flatnonzero(indices == src_id):
                    target[target_id] = copied[local]
            del inp, result, copied
            info['gpu_blocks'] += 1


def resample_logits(data: np.ndarray, new_shape, is_seg=False, axis=None, order=3,
                     do_separate_z=False, order_z=0, dtype_out=None, *, threads=4):
    """Drop-in experimental replacement for nnUNet resample_data_or_seg.

    Float32 CUDA linear interpolation intentionally differs from the float64
    reference. Final dtype quantization happens before the unchanged nnUNet
    argmax/export. Unsupported options explicitly fall back to the reference.
    """
    global LAST_RUN_INFO
    start = time.perf_counter()
    info: dict[str, Any] = {
        'backend': 'pending', 'fallback_reason': None, 'gpu_tiled': False,
        'h2d_seconds': 0.0, 'gpu_compute_seconds': 0.0, 'd2h_seconds': 0.0,
        'gpu_blocks': 0, 'threads_argument': int(threads),
    }
    LAST_RUN_INFO = info
    shape = tuple(int(n) for n in new_shape)
    dtype = np.dtype(data.dtype if dtype_out is None else dtype_out)
    axis = _axis_int(axis)
    reason = None
    if data.ndim != 4 or len(shape) != 3 or min(shape) <= 0:
        reason = 'expected CXYZ array and positive three-dimensional target'
    elif is_seg:
        reason = 'segmentation label interpolation is unsupported'
    elif order != 1:
        reason = f'only linear logit interpolation is accelerated; order={order}'
    elif do_separate_z and (order_z != 0 or axis not in (0, 1, 2)):
        reason = 'separate_z requires one spatial axis and order_z=0'
    elif data.dtype.kind != 'f' or dtype not in (
        np.dtype('float16'), np.dtype('float32'), np.dtype('float64')
    ):
        reason = 'floating-point input/output required'
    elif not torch.cuda.is_available():
        reason = 'CUDA unavailable'

    if reason is not None:
        info.update(backend='reference_fallback', fallback_reason=reason)
        result = _CPU_RESAMPLE(data, shape, is_seg, axis, order, do_separate_z,
                               order_z, dtype_out, original_function=_REFERENCE, threads=threads)
        info['wall_seconds'] = time.perf_counter() - start
        return result
    if tuple(data.shape[1:]) == shape:
        info.update(backend='identity', wall_seconds=time.perf_counter() - start)
        return data

    torch.cuda.synchronize()
    # Inference may have left several GiB in the allocator cache. Release unused
    # blocks before choosing this stage's bounded working set; live model weights
    # remain allocated. The release cost is included in wall_seconds.
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    budget = int(min(_MAX_WORK_BYTES, free * 0.65))
    # Input is a single channel in the tiled path. Reserve enough extra room
    # for source dtype conversion and at least one small output/grid tile.
    min_input_bytes = math.prod(data.shape[1:]) * (4 + data.dtype.itemsize)
    if not do_separate_z and min_input_bytes + 64 * 1024**2 > budget:
        info.update(backend='reference_fallback',
                    fallback_reason='single input channel exceeds CUDA work budget')
        result = _CPU_RESAMPLE(data, shape, is_seg, axis, order, do_separate_z,
                               order_z, dtype_out, original_function=_REFERENCE, threads=threads)
        info['wall_seconds'] = time.perf_counter() - start
        return result

    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    info.update(backend='cuda_float32', work_budget_bytes=budget,
                cuda_free_before_bytes=int(free), cuda_total_bytes=int(total),
                cuda_allocated_before_bytes=int(baseline_allocated),
                input_shape=list(data.shape), output_shape=[data.shape[0], *shape],
                input_dtype=str(data.dtype), output_dtype=str(dtype))
    out = np.empty((data.shape[0], *shape), dtype=dtype)
    with torch.inference_mode():
        if do_separate_z:
            _resample_separate(data, shape, axis, out, dtype, info, budget)
        else:
            _resample_volume(data, shape, out, dtype, info, budget)
    torch.cuda.synchronize()
    info.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                wall_seconds=time.perf_counter() - start)
    info['cuda_peak_extra_allocated_bytes'] = (
        info['cuda_peak_allocated_bytes'] - baseline_allocated
    )
    return out
