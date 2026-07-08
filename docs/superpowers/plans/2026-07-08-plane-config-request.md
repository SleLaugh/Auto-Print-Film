# Plane Config Request Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add request parameters for per-plane slice thickness, slice spacing, and window presets so whole-range reconstruction can be driven by structured input.

**Architecture:** Keep the parameter contract in `src/models`: enums define legal string values, `PlaneConfig` owns per-plane parsing/serialization, and `Request` owns cross-plane validation. `main.py` parses the CLI JSON string into `Request`; `ReportingStage` serializes `Request.to_dict()` so reports automatically include the new fields.

**Tech Stack:** Python dataclasses, stdlib `json`, existing `argparse`, existing `pytest`-style test execution via `python -m pytest` if pytest is available.

## Global Constraints

- Add `WindowPresetEnum` with exactly `骨窗`, `肺窗`, and `纵隔窗`.
- Reuse existing `PlaneTypeEnum` values: `横状位`, `矢状位`, `冠状位`.
- Add `Request.plane_configs: list[PlaneConfig]`.
- Default `plane_configs` to three planes, each with `slice_thickness=1`, `slice_spacing=1`, and `window_presets=["骨窗"]`.
- Reject empty `plane_configs`, duplicate planes, non-positive `slice_thickness`, non-positive `slice_spacing`, and empty `window_presets`.
- CLI parameter is `--plane_configs` with alias `--plane-configs`; value is a JSON array string.
- Do not add `--plane_config_file` in this implementation.
- Do not implement MPR reconstruction or DICOM output directories in this implementation.
- Keep explanatory comments reasonably full and in Chinese where practical.

---

## File Structure

- Modify `src/models/enums.py`: add `WindowPresetEnum`.
- Modify `src/models/request.py`: add `PlaneConfig`, default plane config factory, parsing, serialization, and validation.
- Modify `src/models/__init__.py`: export `PlaneConfig` and `WindowPresetEnum`.
- Modify `main.py`: add `--plane_configs/--plane-configs`, parse JSON, pass it to `Request`.
- Modify `src/stages/stage_reporting.py`: use `request.to_dict()` in report output so `plane_configs` is included.
- Create `tests/test_request_plane_config.py`: model-level tests.
- Create `tests/test_main_plane_configs.py`: CLI parser tests.

### Task 1: Model Enums and Request Parsing

**Files:**
- Modify: `src/models/enums.py`
- Modify: `src/models/request.py`
- Modify: `src/models/__init__.py`
- Create: `tests/test_request_plane_config.py`

**Interfaces:**
- Produces: `WindowPresetEnum(Enum)` with values `骨窗`, `肺窗`, `纵隔窗`.
- Produces: `PlaneConfig(plane: PlaneTypeEnum, slice_thickness: float, slice_spacing: float, window_presets: list[WindowPresetEnum])`.
- Produces: `PlaneConfig.from_dict(data: dict) -> PlaneConfig`.
- Produces: `PlaneConfig.to_dict(self) -> dict[str, object]`.
- Produces: `Request.plane_configs: list[PlaneConfig]`.
- Produces: `Request.to_dict() -> dict` including JSON-friendly `plane_configs`.
- Produces: `Request.from_dict(data: dict) -> Request` parsing `plane_configs` if present.
- Produces: `Request.validate() -> tuple[bool, Optional[str]]` validating the new fields.

- [ ] **Step 1: Write failing model tests**

Create `tests/test_request_plane_config.py` with:

```python
import pytest

from src.models import PlaneConfig, Request, WindowPresetEnum
from src.models.enums import BodyPartEnum, PlaneTypeEnum


def test_plane_config_from_dict_converts_strings_to_enums():
    config = PlaneConfig.from_dict({
        "plane": "冠状位",
        "slice_thickness": 5,
        "slice_spacing": 10,
        "window_presets": ["骨窗", "肺窗", "纵隔窗"],
    })

    assert config.plane is PlaneTypeEnum.冠状位
    assert config.slice_thickness == 5.0
    assert config.slice_spacing == 10.0
    assert config.window_presets == [
        WindowPresetEnum.骨窗,
        WindowPresetEnum.肺窗,
        WindowPresetEnum.纵隔窗,
    ]


def test_request_from_dict_parses_plane_configs_and_to_dict_serializes_values():
    request = Request.from_dict({
        "dicom_dir": "/tmp/dicom",
        "output_dir": "/tmp/output",
        "body_part": BodyPartEnum.腰椎.value,
        "plane_configs": [
            {
                "plane": "冠状位",
                "slice_thickness": 5,
                "slice_spacing": 10,
                "window_presets": ["骨窗", "肺窗"],
            }
        ],
    })

    assert request.plane_configs[0].plane is PlaneTypeEnum.冠状位
    assert request.to_dict()["plane_configs"] == [
        {
            "plane": "冠状位",
            "slice_thickness": 5.0,
            "slice_spacing": 10.0,
            "window_presets": ["骨窗", "肺窗"],
        }
    ]


def test_request_uses_three_default_bone_window_plane_configs():
    request = Request(dicom_dir="/tmp/dicom", output_dir="/tmp/output")

    assert request.to_dict()["plane_configs"] == [
        {
            "plane": "横状位",
            "slice_thickness": 1.0,
            "slice_spacing": 1.0,
            "window_presets": ["骨窗"],
        },
        {
            "plane": "矢状位",
            "slice_thickness": 1.0,
            "slice_spacing": 1.0,
            "window_presets": ["骨窗"],
        },
        {
            "plane": "冠状位",
            "slice_thickness": 1.0,
            "slice_spacing": 1.0,
            "window_presets": ["骨窗"],
        },
    ]


@pytest.mark.parametrize(
    ("plane_configs", "message"),
    [
        ([], "plane_configs 不能为空"),
        (
            [
                {"plane": "横状位", "slice_thickness": 1, "slice_spacing": 1, "window_presets": ["骨窗"]},
                {"plane": "横状位", "slice_thickness": 1, "slice_spacing": 1, "window_presets": ["骨窗"]},
            ],
            "plane_configs 中存在重复状位",
        ),
        (
            [{"plane": "横状位", "slice_thickness": 0, "slice_spacing": 1, "window_presets": ["骨窗"]}],
            "层厚必须大于 0",
        ),
        (
            [{"plane": "横状位", "slice_thickness": 1, "slice_spacing": 0, "window_presets": ["骨窗"]}],
            "层间距必须大于 0",
        ),
        (
            [{"plane": "横状位", "slice_thickness": 1, "slice_spacing": 1, "window_presets": []}],
            "窗宽窗位不能为空",
        ),
    ],
)
def test_request_validate_rejects_invalid_plane_configs(plane_configs, message):
    request = Request.from_dict({
        "dicom_dir": "/tmp/dicom",
        "output_dir": "/tmp/output",
        "plane_configs": plane_configs,
    })

    is_valid, error_message = request.validate()

    assert is_valid is False
    assert message in error_message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_request_plane_config.py -v`

Expected: FAIL with import errors for `PlaneConfig` and `WindowPresetEnum`, proving the new interface does not exist yet.

- [ ] **Step 3: Implement enum and model changes**

Modify `src/models/enums.py` to add:

```python
class WindowPresetEnum(Enum):
    """窗宽窗位预设枚举"""
    骨窗 = "骨窗"
    肺窗 = "肺窗"
    纵隔窗 = "纵隔窗"
```

Modify `src/models/request.py` so the imports and dataclasses include:

```python
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from .enums import (
    BodyPartEnum,
    CheckTypeEnum,
    ScanThicknessEnum,
    CutModeEnum,
    PlaneTypeEnum,
    WindowPresetEnum,
)


@dataclass
class PlaneConfig:
    """单个重建状位的采样和显示配置。"""
    plane: PlaneTypeEnum
    slice_thickness: float
    slice_spacing: float
    window_presets: List[WindowPresetEnum]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlaneConfig':
        return cls(
            plane=PlaneTypeEnum(data['plane']),
            slice_thickness=float(data['slice_thickness']),
            slice_spacing=float(data['slice_spacing']),
            window_presets=[
                WindowPresetEnum(value)
                for value in data.get('window_presets', [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'plane': self.plane.value,
            'slice_thickness': self.slice_thickness,
            'slice_spacing': self.slice_spacing,
            'window_presets': [
                window.value
                for window in self.window_presets
            ],
        }
```

Add a default factory:

```python
def default_plane_configs() -> List[PlaneConfig]:
    """
    默认输出三个状位的骨窗配置。

    旧命令没有传 plane_configs 时仍然能构造 Request；后续重建阶段
    可以直接读取这个列表，不需要再判断 None。
    """
    return [
        PlaneConfig(PlaneTypeEnum.横状位, 1.0, 1.0, [WindowPresetEnum.骨窗]),
        PlaneConfig(PlaneTypeEnum.矢状位, 1.0, 1.0, [WindowPresetEnum.骨窗]),
        PlaneConfig(PlaneTypeEnum.冠状位, 1.0, 1.0, [WindowPresetEnum.骨窗]),
    ]
```

Update `Request` field and methods:

```python
plane_configs: List[PlaneConfig] = field(default_factory=default_plane_configs)
```

```python
'plane_configs': [
    plane_config.to_dict()
    for plane_config in self.plane_configs
]
```

```python
plane_configs=[
    PlaneConfig.from_dict(item)
    for item in data.get('plane_configs', [
        plane_config.to_dict()
        for plane_config in default_plane_configs()
    ])
]
```

Add validation after existing path checks:

```python
if not self.plane_configs:
    return False, "plane_configs 不能为空"

seen_planes = set()
for plane_config in self.plane_configs:
    plane_value = plane_config.plane.value
    if plane_value in seen_planes:
        return False, f"plane_configs 中存在重复状位: {plane_value}"
    seen_planes.add(plane_value)

    if plane_config.slice_thickness <= 0:
        return False, f"{plane_value} 层厚必须大于 0"
    if plane_config.slice_spacing <= 0:
        return False, f"{plane_value} 层间距必须大于 0"
    if not plane_config.window_presets:
        return False, f"{plane_value} 窗宽窗位不能为空"
```

Modify `src/models/__init__.py` to import/export:

```python
from .request import Request, PlaneConfig
```

```python
WindowPresetEnum
```

- [ ] **Step 4: Run model tests to verify they pass**

Run: `python -m pytest tests/test_request_plane_config.py -v`

Expected: PASS for all tests in this file.

- [ ] **Step 5: Commit model task**

```bash
git add src/models/enums.py src/models/request.py src/models/__init__.py tests/test_request_plane_config.py
git commit -m "feat: add plane config request model"
```

### Task 2: CLI JSON Parsing

**Files:**
- Modify: `main.py`
- Create: `tests/test_main_plane_configs.py`

**Interfaces:**
- Consumes: `Request(..., plane_configs=list[PlaneConfig])` from Task 1.
- Produces: `parse_plane_configs(value: str | None) -> list[PlaneConfig] | None`.
- Produces: `parse_arguments()` namespace with `plane_configs`.

- [ ] **Step 1: Write failing CLI parser tests**

Create `tests/test_main_plane_configs.py` with:

```python
import sys

import pytest

import main
from src.models.enums import PlaneTypeEnum, WindowPresetEnum


def test_parse_plane_configs_json_string():
    configs = main.parse_plane_configs(
        '[{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗","肺窗"]}]'
    )

    assert configs[0].plane is PlaneTypeEnum.冠状位
    assert configs[0].slice_thickness == 5.0
    assert configs[0].slice_spacing == 10.0
    assert configs[0].window_presets == [WindowPresetEnum.骨窗, WindowPresetEnum.肺窗]


def test_parse_plane_configs_none_uses_request_default():
    assert main.parse_plane_configs(None) is None


def test_parse_plane_configs_rejects_non_array_json():
    with pytest.raises(ValueError, match="plane_configs 必须是 JSON 数组"):
        main.parse_plane_configs('{"plane":"冠状位"}')


def test_parse_arguments_accepts_plane_configs_alias(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "-i",
        "/tmp/dicom",
        "-o",
        "/tmp/output",
        "--plane-configs",
        '[{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗"]}]',
    ])

    args = main.parse_arguments()

    assert args.plane_configs == '[{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗"]}]'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main_plane_configs.py -v`

Expected: FAIL because `main.parse_plane_configs` is not defined and the CLI option does not exist.

- [ ] **Step 3: Implement CLI parsing**

Modify `main.py` imports:

```python
import json
```

```python
from src.models import Request, BodyPartEnum, CheckTypeEnum, ScanThicknessEnum, CutModeEnum, PlaneConfig
```

Add parser argument near other business parameters:

```python
parser.add_argument(
    '--plane_configs',
    '--plane-configs',
    dest='plane_configs',
    type=str,
    default=None,
    help='三平面重建参数 JSON 数组，例如: [{"plane":"冠状位","slice_thickness":5,"slice_spacing":10,"window_presets":["骨窗"]}]'
)
```

Add helper above `main()`:

```python
def parse_plane_configs(value):
    """
    解析命令行传入的三平面配置 JSON。

    返回 None 表示调用方没有显式传参，此时 Request 使用默认三平面骨窗配置。
    """
    if value is None:
        return None

    try:
        raw_configs = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"plane_configs 不是合法 JSON: {e}") from e

    if not isinstance(raw_configs, list):
        raise ValueError("plane_configs 必须是 JSON 数组")

    return [
        PlaneConfig.from_dict(item)
        for item in raw_configs
    ]
```

Update `Request(...)` construction:

```python
plane_configs = parse_plane_configs(args.plane_configs)
request_kwargs = {
    'dicom_dir': args.input,
    'output_dir': args.output,
    'body_part': BodyPartEnum(args.body_part),
    'check_type': CheckTypeEnum(args.check_type),
    'scan_thickness': ScanThicknessEnum(args.scan_thickness),
    'cut_mode': CutModeEnum(args.cut_mode),
}
if plane_configs is not None:
    request_kwargs['plane_configs'] = plane_configs
request = Request(**request_kwargs)
```

Add debug output:

```python
DebugLog(f"三平面参数: {request.to_dict()['plane_configs']}")
```

- [ ] **Step 4: Run CLI parser tests to verify they pass**

Run: `python -m pytest tests/test_main_plane_configs.py -v`

Expected: PASS for all tests in this file.

- [ ] **Step 5: Verify CLI help includes the new option**

Run: `python main.py -h`

Expected: output contains `--plane_configs` and `--plane-configs`.

- [ ] **Step 6: Commit CLI task**

```bash
git add main.py tests/test_main_plane_configs.py
git commit -m "feat: parse plane configs from cli"
```

### Task 3: Report Serialization and Full Verification

**Files:**
- Modify: `src/stages/stage_reporting.py`
- Modify: `tests/test_request_plane_config.py`

**Interfaces:**
- Consumes: `Request.to_dict()` from Task 1.
- Produces: report JSON `request` object including `plane_configs`.

- [ ] **Step 1: Add failing report serialization test**

Append to `tests/test_request_plane_config.py`:

```python
def test_request_to_dict_contains_plane_configs_for_reports():
    request = Request.from_dict({
        "dicom_dir": "/tmp/dicom",
        "output_dir": "/tmp/output",
        "plane_configs": [
            {
                "plane": "冠状位",
                "slice_thickness": 5,
                "slice_spacing": 10,
                "window_presets": ["骨窗", "纵隔窗"],
            }
        ],
    })

    request_data = request.to_dict()

    assert request_data["plane_configs"] == [
        {
            "plane": "冠状位",
            "slice_thickness": 5.0,
            "slice_spacing": 10.0,
            "window_presets": ["骨窗", "纵隔窗"],
        }
    ]
```

- [ ] **Step 2: Run reporting-related test**

Run: `python -m pytest tests/test_request_plane_config.py::test_request_to_dict_contains_plane_configs_for_reports -v`

Expected: PASS if Task 1 already added `to_dict`; if it fails, fix `Request.to_dict` before continuing.

- [ ] **Step 3: Simplify report request serialization**

Modify `_generate_report` in `src/stages/stage_reporting.py`:

```python
'request': context.request.to_dict(),
```

Replace the current manual request dictionary so future request fields are not silently omitted from reports.

- [ ] **Step 4: Run all focused tests**

Run: `python -m pytest tests/test_request_plane_config.py tests/test_main_plane_configs.py -v`

Expected: PASS for both test files.

- [ ] **Step 5: Run syntax and CLI verification**

Run: `python -B -c "import ast, pathlib; files=[pathlib.Path('main.py')]+list(pathlib.Path('src').glob('**/*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'parsed {len(files)} project python files')"`

Expected: output starts with `parsed`.

Run: `python main.py -h`

Expected: exits 0 and displays the new `--plane_configs` help.

- [ ] **Step 6: Commit report task**

```bash
git add src/stages/stage_reporting.py tests/test_request_plane_config.py
git commit -m "feat: include plane configs in reports"
```

## Final Verification

- [ ] Run focused tests:

```bash
python -m pytest tests/test_request_plane_config.py tests/test_main_plane_configs.py -v
```

- [ ] Run project syntax check:

```bash
python -B -c "import ast, pathlib; files=[pathlib.Path('main.py')]+list(pathlib.Path('src').glob('**/*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'parsed {len(files)} project python files')"
```

- [ ] Run CLI help:

```bash
python main.py -h
```

- [ ] Check working tree:

```bash
git status --short
```

Expected: only pre-existing unrelated user changes remain outside this implementation.
