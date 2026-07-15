# Window Presets Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the JSON configuration the only source of supported window preset names and values.

**Architecture:** `src/models/window.py` loads and validates `src/config/window_presets.json`, then exposes immutable `WindowPreset` values through the existing lookup API. Request models carry plain preset-name strings so adding a configured preset never requires a Python enum change.

**Tech Stack:** Python dataclasses, JSON, unittest.

## Global Constraints

- External requests continue to pass only preset names.
- Each configured entry contains only `name`, `window_width`, and `window_center`.
- `WindowPresetEnum` is removed.
- Line color is outside this change.

---

### Task 1: Configuration loader

**Files:**
- Create: `src/config/window_presets.json`
- Modify: `src/models/window.py`
- Test: `tests/test_window_presets.py`

**Interfaces:**
- Produces: `load_window_presets(path: Path | None = None) -> dict[str, WindowPreset]`
- Produces: `get_window_preset(preset: str) -> WindowPreset`

- [ ] Add failing tests proving existing values load from JSON, a temporary configuration can introduce a new name, and duplicate or invalid entries raise `ValueError`.
- [ ] Run `python -m unittest tests.test_window_presets -v` and confirm the new configuration tests fail.
- [ ] Move all existing preset values into `window_presets.json` and implement strict JSON loading and validation in `src/models/window.py`.
- [ ] Run `python -m unittest tests.test_window_presets -v` and confirm all tests pass.

### Task 2: Remove the enum from request and pipeline contracts

**Files:**
- Modify: `src/models/enums.py`
- Modify: `src/models/__init__.py`
- Modify: `src/models/request.py`
- Modify: request, planning, reconstruction, and reporting tests importing `WindowPresetEnum`

**Interfaces:**
- `PlaneConfig.window_presets: list[str]`
- `PlaneConfig.from_dict()` preserves configured names as strings.
- `Request.validate()` resolves every name through `get_window_preset()`.

- [ ] Add failing request tests proving configured string names round-trip and unknown names fail validation.
- [ ] Run `python -m unittest tests.test_request_plane_config -v` and confirm failures are caused by enum conversion.
- [ ] Remove `WindowPresetEnum`, migrate defaults and all typed call sites to strings, and validate names through the configuration lookup.
- [ ] Update existing tests to use string preset names without changing external JSON expectations.
- [ ] Run `python -m unittest discover -s tests -q`, `python -m compileall -q src tests`, and `git diff --check`.
