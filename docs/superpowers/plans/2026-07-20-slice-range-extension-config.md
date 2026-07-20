# Slice Range Extension Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional per-body-part, per-cut-mode, per-plane start/end slice-range extension values in millimetres while preserving all existing behavior when omitted.

**Architecture:** Keep configuration in each existing `localization_config.json` cut-mode object. Stage 3 validates all cut-mode objects for the selected body part and forwards them as `cut_configs`; Stage 4 resolves values by the current plan's cut mode and plane, then extends the target segment before the existing equal-spacing center-coverage calculation. Stage 5 remains unchanged.

**Tech Stack:** Python 3, JSON configuration, `unittest`, NumPy/SimpleITK geometry models already used by the project.

## Global Constraints

- The configuration key is exactly `slice_range_extension`.
- Plane keys use existing `PlaneTypeEnum` values: `横状位`, `矢状位`, `冠状位`.
- Per-plane keys are exactly `start` and `end`, interpreted as millimetres.
- Missing configuration or missing endpoint fields resolve to `0.0` and preserve existing planning behavior.
- Only `肺部.whole_range.横状位` is configured in the repository, with `start=3.0` and `end=3.0`.
- Negative, boolean, non-numeric and non-finite extension values are rejected.
- `SlicePlan.target_start/target_end` remain the unextended anatomical target boundaries.
- Slice spacing, thickness, angle, localizer behavior and Stage 5 execution are unchanged.
- New and modified source classes and functions require normal Chinese comments/docstrings.
- Per-plane extension objects may omit either `start` or `end`, but reject every key other than those two names.

---

### Task 1: Validate and apply configured slice-range extensions

**Files:**
- Modify: `src/config/localization_config.json`
- Modify: `src/stages/stage_localization.py`
- Modify: `src/models/localization.py`
- Modify: `src/planning/straight.py`
- Modify: `src/planning/whole_range.py`
- Modify: `tests/test_stage3_target_axial_correction.py`
- Modify: `tests/test_planning_geometry.py`
- Modify: `tests/test_stage4_slice_planning.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `localization_result["cut_configs"][cut_mode]["slice_range_extension"][plane.value]`, with the legacy `cut_config` retained for compatibility.
- Produces: `resolve_slice_range_extension(cut_config, plane) -> SliceRangeExtension` and optional `start_extension_mm`/`end_extension_mm` inputs to `cover_segment_range`.

- [x] **Step 1: Write failing configuration and planning tests**

Add tests asserting the checked-in lung configuration is `3.0/3.0`, no other checked-in combination is nonzero, omitted planes and omitted endpoint fields resolve to zero, invalid values fail validation, mixed cut modes select their own configuration, and range extension works for forward, reverse and oblique target segments without changing spacing.

- [x] **Step 2: Run focused tests and verify the expected failures**

Run:

```bash
.venv/bin/python -m unittest tests.test_stage3_target_axial_correction tests.test_planning_geometry tests.test_stage4_slice_planning -v
```

Expected: new tests fail because `slice_range_extension` resolution and geometric application do not exist yet.

- [x] **Step 3: Implement minimal configuration validation and geometry**

Add a documented frozen value object and resolver in `src/planning/straight.py`. Extend the segment along its own normalized direction before calling the existing center-coverage calculation. Pass the resolved values through both the generic straight planner and the anatomical-frame whole-range planner. Validate the JSON structure in `LocalizationStage._validate_cut_config`.

- [x] **Step 4: Run focused tests and verify they pass**

Run the same focused `unittest` command. Expected: all focused tests pass.

- [x] **Step 5: Document the configuration**

Add a README example explaining the fixed millimetre unit, endpoint direction, omitted-value behavior and the distinction between configured target extension and the final spacing-aligned center range.

- [x] **Step 6: Run full verification**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: all tests pass and both static checks exit with status 0.
