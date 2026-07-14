# Numeric DICOM Names And Localizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name reconstructed slices `1.dcm`, `2.dcm`, and so on, and add an unnumbered `Localizer.dcm` beside each window series.

**Architecture:** Stage 5 keeps DICOM `InstanceNumber` unchanged while changing only the numbered slice filenames. For each plan independently, it selects the median fixed-axis original CT layer, burns that plan's thickness bands and center lines into the real orthogonal slice, applies the current window, and writes it as a separate localizer DICOM. Stage 6 numerically validates only files whose complete filename is `<positive integer>.dcm` and ignores `Localizer.dcm`.

**Tech Stack:** Python 3, SimpleITK, NumPy, unittest.

## Global Constraints

- Formal slice filenames contain only their one-based decimal index and `.dcm`.
- `InstanceNumber (0020,0013)` remains one-based and continuous.
- `Localizer.dcm` is not counted as a reconstructed slice and is not validated by stage 6.
- The localizer filename is identical for every plane and window.
- Do not commit unless the user explicitly requests it.

---

### Task 1: Rename formal DICOM slices and update reporting validation

**Files:**
- Modify: `src/stages/stage_reconstruction.py`
- Modify: `src/stages/stage_reporting.py`
- Test: `tests/test_stage5_reconstruction.py`
- Test: `tests/test_reporting_plane_configs.py`

**Interfaces:**
- Produces: formal slice paths named `<index>.dcm`.
- Preserves: DICOM `InstanceNumber` and numeric sequence order.

- [ ] Write failing tests expecting `1.dcm`, `2.dcm`, and numeric reporting order.
- [ ] Run targeted tests and confirm they fail on the old `切片_0001.dcm` contract.
- [ ] Change stage 5 naming and make stage 6 match `^(\d+)\.dcm$`, then sort by parsed integer.
- [ ] Run targeted tests and confirm they pass.

### Task 2: Write one localizer per window directory

**Files:**
- Modify: `src/reconstruction/dicom.py`
- Modify: `src/stages/stage_reconstruction.py`
- Test: `tests/test_stage5_reconstruction.py`

**Interfaces:**
- Consumes: the original reconstruction CT and current plan.
- Produces: `write_localizer_dicom(image, output_path, window, study_uid, series_uid, source_metadata)` and `Localizer.dcm`.

- [ ] Write a failing integration test with a CT-backed plane overview and assert every window directory contains `Localizer.dcm` plus numbered slices.
- [ ] Run the test and confirm failure because no localizer is written.
- [ ] Extract the plan-center median orthogonal CT layer, draw only the current plan's cut lines, apply the selected window, and write a separate derived CT localizer with its own series UID.
- [ ] Run stage-5 tests and confirm the localizer is readable while reconstruction counts still exclude it.

### Task 3: Full verification

**Files:**
- Verify all files above.

**Interfaces:**
- No new interfaces.

- [ ] Run `python -m unittest discover -s tests -q`.
- [ ] Run `python -m compileall -q src tests`.
- [ ] Run `git diff --check` and review `git status --short`.
