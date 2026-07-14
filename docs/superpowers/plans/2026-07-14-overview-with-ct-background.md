# Overview With CT Background Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace refined-grid label-only overviews with presentation NIfTI files containing resampled CT, thickness bands, and center lines.

**Architecture:** Reuse the existing per-plane overview reference and label painting logic. Resample `raw_for_reconstruction_path` onto that reference, burn label 15 regions above the CT maximum and label 14 regions higher still, then write a float32 presentation image while retaining native-grid label overviews.

**Tech Stack:** Python 3, NumPy, SimpleITK, unittest.

## Global Constraints

- Presentation files are debug-only and never become stage-5 inputs.
- CT resampling uses linear interpolation and the source minimum as the default value.
- Existing native-grid overview files remain unchanged.
- Do not commit unless the user explicitly requests a commit.

---

### Task 1: Burn CT and overlay labels into one presentation image

**Files:**
- Modify: `src/stages/stage_mapping.py`
- Test: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Produces: `_build_overview_with_image(raw_image, reference_image, overlay_array) -> sitk.Image`
- Consumes: label values ending in 5 for thickness bands and values ending in 4 for center lines.

- [ ] **Step 1: Write the failing unit test**

Construct a small signed-int16 CT with a gradient, an overview reference with a refined stack axis, and an overlay containing labels 15 and 14. Assert output metadata matches the reference, untouched voxels equal linearly resampled CT values, band voxels exceed the CT maximum, and center voxels exceed band voxels.

- [ ] **Step 2: Verify RED**

Run the targeted unittest and expect failure because `_build_overview_with_image` does not exist.

- [ ] **Step 3: Implement the helper**

Use `sitk.Resample(raw_image, reference_image, sitk.Transform(), sitk.sitkLinear, source_min, sitk.sitkFloat32)`, convert to NumPy, choose `band_value = source_max + max(source_max - source_min, 1)` and `center_value = band_value + max(source_max - source_min, 1)`, apply masks, and copy reference metadata to the result.

- [ ] **Step 4: Verify GREEN**

Run the targeted unittest and expect `OK`.

### Task 2: Replace refined label-only overview outputs

**Files:**
- Modify: `src/stages/stage_mapping.py`
- Test: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Changes: `_write_slice_plan_overview_overlays` reads `raw_for_reconstruction_path`, writes `切片总览含原图_*`, and returns those paths as presentation overview paths.
- Preserves: `切片原始网格总览_*` and `debug_slice_plan_native_overview_paths`.

- [ ] **Step 1: Write failing integration tests**

Provide matching raw and segmentation images, execute stage 4 for all three planes, and assert returned paths are named `切片总览含原图_横/矢/冠_整.nii.gz`, the old refined `切片总览_*` files do not exist, the native overview files still match the segmentation grid, and the output dictionary is also stored under `debug_slice_plan_overview_with_image_paths`.

- [ ] **Step 2: Verify RED**

Run the new integration test and expect failure because the current implementation writes label-only `切片总览_*` files.

- [ ] **Step 3: Implement output wiring**

Read the raw image once, build each presentation image with Task 1's helper, write it using the new filename, return the new path dictionary through both `debug_slice_plan_overview_paths` and `debug_slice_plan_overview_with_image_paths`, and point the legacy single path at the horizontal or first available new file.

- [ ] **Step 4: Add missing-raw behavior**

When the raw path is missing, return no presentation paths while still writing native-grid overview files and preserving the rest of stage 4 output.

- [ ] **Step 5: Verify stage-4 tests**

Run `python -m unittest tests.test_stage4_slice_planning -v` and expect all stage-4 tests to pass.

### Task 3: Keep the full CT extent with int16 storage

**Files:**
- Modify: `src/stages/stage_mapping.py`
- Test: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Changes: `_build_overview_with_image` returns a full-reference-grid `sitkInt16` image.
- Preserves: the complete CT field of view and all nonzero overlay layers.

- [ ] **Step 1: Write a failing unit test**

Assert the presentation image has exactly the refined reference size, origin, spacing, and direction, and has pixel ID `sitkInt16`.

- [ ] **Step 2: Verify RED**

Run the targeted unittests and expect failure because the current output is cropped and uses `sitkFloat32`.

- [ ] **Step 3: Restore the full reference grid**

Pass the full refined reference and overlay directly to `_build_overview_with_image`; remove the minimum-slab crop helper.

- [ ] **Step 4: Store the presentation image as int16**

Resample with output type `sitkInt16`, clamp the default fill and calculated band/center highlight values to the signed 16-bit range, and keep the center value above the band value.

- [ ] **Step 5: Verify stage-4 tests**

Run `python -m unittest tests.test_stage4_slice_planning -q` and expect all stage-4 tests to pass.

### Task 4: Full regression verification

**Files:**
- Verify: `src/stages/stage_mapping.py`
- Verify: `tests/test_stage4_slice_planning.py`
- Verify: `docs/superpowers/specs/2026-07-14-overview-with-ct-background-design.md`

**Interfaces:**
- No new interfaces.

- [ ] **Step 1: Run whitespace validation**

Run `git diff --check` and expect no output.

- [ ] **Step 2: Run the full suite**

Run `python -m unittest discover -s tests -v` and expect all tests to pass.

- [ ] **Step 3: Review the final diff**

Confirm only stage-4 implementation, stage-4 tests, specification, and plan changed. Leave the worktree uncommitted.
