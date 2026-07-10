# Slice Thickness and Spacing Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make axial whole-range slices use true thickness coverage, strictly uniform center spacing, and a diagnostic overview that shows both centers and thickness bands.

**Architecture:** Keep generic center interpolation in `src/planning/geometry.py`, but pass it the valid center interval computed by the axial bbox planner. Remove endpoint injection from interpolation. Extend mapping diagnostics from serialized slice centers and paint thickness bands beside center lines in the overview overlay.

**Tech Stack:** Python 3, unittest, NumPy, SimpleITK

## Global Constraints

- `slice_thickness` is physical coverage along the slice normal in millimetres.
- `slice_spacing` is the physical distance between adjacent output centers in millimetres.
- Never append a terminal center that violates `slice_spacing`.
- Preserve spacing, origin, and direction-aware coordinate conversion.
- Do not change request configuration values.
- Do not run `git commit`; leave changes for user review.

---

### Task 1: Strictly spaced center interpolation

**Files:**
- Modify: `tests/test_planning_geometry.py`
- Modify: `src/planning/geometry.py`

**Interfaces:**
- Consumes: `interpolate_points(start: Point3D, end: Point3D, spacing: float)`
- Produces: a non-empty list beginning at `start`, with every adjacent distance equal to `spacing`, and no forced terminal point

- [ ] **Step 1: Write the failing non-integer endpoint test**

Add a test calling `interpolate_points(Point3D(0, 0, 0), Point3D(0, 0, 12.6), 5.0)` and assert z centers are `[0.0, 5.0, 10.0]`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_planning_geometry.PlanningGeometryTest.test_interpolate_points_does_not_append_non_uniform_endpoint -v`

Expected: FAIL because the current result includes `12.6`.

- [ ] **Step 3: Remove forced endpoint insertion**

Delete the terminal append block in `interpolate_points`; retain `floor(distance / spacing) + 1` regularly spaced centers.

- [ ] **Step 4: Run geometry tests and verify GREEN**

Run: `.venv/bin/python -m unittest tests/test_planning_geometry.py -v`

Expected: all geometry tests pass.

### Task 2: Put axial centers inside the target thickness bounds

**Files:**
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `src/planning/whole_range.py`

**Interfaces:**
- Consumes: `bbox_axial_range_points(localization_result, slice_thickness)`
- Produces: first and last valid slice-center bounds, each inset by half thickness from the physical bbox boundaries; for targets shorter than thickness both bounds equal the target midpoint

- [ ] **Step 1: Replace outward-expansion expectations with inset-center expectations**

For bbox z `0..10`, thickness `5`, spacing `5`, assert center bounds and slice centers are `2.5..7.5` and `[2.5, 7.5]`. For bbox z `-2..12`, assert bounds are `0.5..9.5` and centers `[0.5, 5.5]`.

- [ ] **Step 2: Add a short-target test**

For bbox z `4..6` and thickness `5`, assert exactly one center at physical bbox midpoint z `5.0`.

- [ ] **Step 3: Run the focused planner tests and verify RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_axial_whole_range_expands_center_range_by_half_thickness tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_axial_whole_range_short_target_uses_midpoint -v`

Expected: existing test fails against outward centers and the new short-target test fails before implementation.

- [ ] **Step 4: Implement direction-aware inset center bounds**

Convert bbox endpoints to physical points without margins, compute the physical axis vector and target length, and return either the midpoint pair for a short target or endpoints inset by `slice_thickness / 2` along the normalized vector.

- [ ] **Step 5: Run stage-four planning tests and verify GREEN**

Run: `.venv/bin/python -m unittest tests/test_stage4_slice_planning.py -v`

Expected: all updated stage-four planning tests pass.

### Task 3: Report actual spacing and uncovered boundaries

**Files:**
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `src/stages/stage_mapping.py`

**Interfaces:**
- Consumes: serialized plan `slices`, `slice_thickness`, `slice_spacing`, `range_start`, and `range_end`
- Produces diagnostics keys `actual_spacing_min_mm`, `actual_spacing_max_mm`, `has_nonuniform_spacing`, `leading_uncovered_mm`, `trailing_uncovered_mm`, and existing theoretical gap fields

- [ ] **Step 1: Write failing diagnostics assertions**

Assert uniform 5 mm plans report min/max 5.0, `has_nonuniform_spacing == false`, and no endpoint-injection flag. Add a 12.6 mm regression fixture and assert no 2.6 mm actual interval exists.

- [ ] **Step 2: Run the focused diagnostics test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_mapping_stage_writes_slice_plans_json -v`

Expected: FAIL because the new diagnostics keys do not exist.

- [ ] **Step 3: Calculate diagnostics from adjacent serialized centers**

Compute Euclidean distances between adjacent centers, use zero for min/max when fewer than two slices exist, compare with `slice_spacing` using a `1e-6` tolerance, and calculate leading/trailing uncovered lengths from valid center bounds and half thickness.

- [ ] **Step 4: Run the focused diagnostics test and verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_mapping_stage_writes_slice_plans_json -v`

Expected: PASS.

### Task 4: Paint true thickness bands in the overview

**Files:**
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `src/stages/stage_mapping.py`

**Interfaces:**
- Consumes: plan centers and `slice_thickness`
- Produces: overview label `plan_index * 10 + 4` for center lines and `plan_index * 10 + 5` for thickness coverage bands

- [ ] **Step 1: Write a failing band-coverage test**

For a 1 mm z-spacing image with thickness 3 mm, assert the overview contains value `15` around each center, value `14` on the center row, and zero in the configured 2 mm gap between bands.

- [ ] **Step 2: Run the focused overlay test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_stage4_slice_planning.Stage4SlicePlanningTest.test_mapping_stage_overview_draws_thickness_bands -v`

Expected: FAIL because the current overview never writes label 15.

- [ ] **Step 3: Implement physical thickness-band painting**

For every overview line, transform physical points at center minus/plus half thickness along the plan normal into continuous or bounded index coverage, paint each intersecting z row across the same y extent with label 15, then paint the single center line with label 14 so it remains visible.

- [ ] **Step 4: Verify overlay behavior including image direction**

Run: `.venv/bin/python -m unittest tests/test_stage4_slice_planning.py -v`

Expected: all overlay and direction tests pass.

### Task 5: Full regression verification

**Files:**
- Verify: `src/planning/geometry.py`
- Verify: `src/planning/whole_range.py`
- Verify: `src/stages/stage_mapping.py`
- Verify: `tests/test_planning_geometry.py`
- Verify: `tests/test_stage4_slice_planning.py`

**Interfaces:**
- Consumes: all changes from Tasks 1–4
- Produces: verified workspace diff ready for user review

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 2: Check patch hygiene**

Run: `git diff --check`

Expected: exit code 0 with no output.

- [ ] **Step 3: Review the final diff against the design**

Run: `git diff -- src/planning/geometry.py src/planning/whole_range.py src/stages/stage_mapping.py tests/test_planning_geometry.py tests/test_stage4_slice_planning.py docs/superpowers/specs/2026-07-10-slice-thickness-spacing-correction-design.md docs/superpowers/plans/2026-07-10-slice-thickness-spacing-correction.md`

Expected: only the approved slice-placement, diagnostics, overlay, tests, and documentation changes are present.
