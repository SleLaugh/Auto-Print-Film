"""Scheme 6 preprocessing proxy; installed nnU-Net stays untouched."""
from __future__ import annotations

import json
import time

import numpy as np
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

from nnunet_s6_input import InputConfigurationProxy

LAST_RUN_INFO = {}


class SkipUnusedMaskPreprocessor(DefaultPreprocessor):
    """Keep dummy creation/crop/normalization; omit its unused resize/cast."""
    def run_case_npy(self, data, seg, properties, plans_manager,
                     configuration_manager, dataset_json):
        global LAST_RUN_INFO
        started = time.perf_counter()
        has_previous_stage = getattr(configuration_manager, 'previous_stage_name', None) is not None
        if seg is not None or has_previous_stage:
            LAST_RUN_INFO = {
                'backend': 'original_default_preprocessor',
                'skipped_dummyseg_resampling': False,
                'reason': 'real segmentation or cascade; preserve original processing',
            }
            self.last_run_info = LAST_RUN_INFO
            print('[S6 preprocessor] ' + json.dumps(LAST_RUN_INFO), flush=True)
            return super().run_case_npy(data, seg, properties, plans_manager,
                                        configuration_manager, dataset_json)
        if getattr(configuration_manager, 'preprocessor_name', None) != 'DefaultPreprocessor':
            raise ValueError('Unused-mask shortcut requires the standard DefaultPreprocessor')
        LAST_RUN_INFO = info = {
            'backend': 'skip_unused_mask', 'state': 'started',
            'skipped_dummyseg_creation': False, 'skipped_dummyseg_resampling': True,
            'skipped_dummyseg_final_cast': True, 'real_segmentation_supplied': False,
            'cascade': False, 'returns_segmentation': False,
            'crop_and_normalization': 'unchanged installed DefaultPreprocessor operations',
        }
        self.last_run_info = info
        try:
            # These operations and their order mirror this installation's
            # DefaultPreprocessor.run_case_npy up to image resampling.
            data = data.astype(np.float32)
            data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
            original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
            properties['shape_before_cropping'] = data.shape[1:]
            data, dummyseg, bbox = crop_to_nonzero(data, None)
            properties['bbox_used_for_cropping'] = bbox
            properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]
            target_spacing = configuration_manager.spacing
            if len(target_spacing) < len(data.shape[1:]):
                target_spacing = [original_spacing[0]] + target_spacing
            new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
            data = self._normalize(data, dummyseg, configuration_manager,
                                   plans_manager.foreground_intensity_properties_per_channel)
            old_shape = data.shape[1:]
            data = configuration_manager.resampling_fn_data(data, new_shape,
                                                           original_spacing, target_spacing)
            del dummyseg
            info.update(state='completed',
                        source_shape=list(properties['shape_before_cropping']),
                        cropped_shape=list(old_shape), output_shape=list(data.shape))
            return data, None, properties
        except Exception as error:
            info.update(state='failed', error=str(error))
            raise
        finally:
            info['seconds'] = time.perf_counter() - started
            print('[S6 preprocessor] ' + json.dumps(info), flush=True)


class ConfigurationProxy(InputConfigurationProxy):
    """Use in predictor.configuration_manager before spawning preprocessors."""
    def __init__(self, wrapped, *, gpu_input=True, skip_unused_mask=True, max_pool_gib=8.0):
        super().__init__(wrapped, max_pool_gib=max_pool_gib)
        self.gpu_input = bool(gpu_input)
        self.skip_unused_mask = bool(skip_unused_mask)

    @property
    def preprocessor_class(self):
        original = self._wrapped.preprocessor_class
        if not self.skip_unused_mask:
            return original
        if original is not DefaultPreprocessor:
            raise ValueError('Unused-mask shortcut only supports the installed DefaultPreprocessor')
        return SkipUnusedMaskPreprocessor

    @property
    def resampling_fn_data(self):
        if not self.gpu_input:
            return self._wrapped.resampling_fn_data
        return super().resampling_fn_data
