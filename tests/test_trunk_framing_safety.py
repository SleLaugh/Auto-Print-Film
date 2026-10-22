"""肺分割单侧或末端为空时，躯干中央骨不能被误识别为上臂。"""

import numpy as np
import pytest

from src.reconstruction.trunk_framing import extract_trunk_slice


@pytest.mark.parametrize("lung_center_x", (80, 220))
@pytest.mark.parametrize("empty_target_slice", (False, True))
def test_unilateral_lung_reference_preserves_spine_and_posterior_body(
    lung_center_x, empty_target_slice,
):
    y, x = np.ogrid[:240, :300]
    body = ((x - 150) / 120) ** 2 + ((y - 120) / 90) ** 2 <= 1
    lung = (x - lung_center_x) ** 2 + (y - 100) ** 2 <= 25 ** 2
    spine = (x - 150) ** 2 + (y - 170) ** 2 <= 18 ** 2
    raw = np.where(body, 45, -1024)
    raw[lung] = -850
    raw[spine] = 1000
    target = np.zeros_like(lung) if empty_target_slice else lung

    actual = extract_trunk_slice(
        raw, body, target,
        (lung_center_x - 25, lung_center_x + 25, 75, 125),
        np.array([1.0, 1.0]),
    )

    # 这里没有上臂；中央骨虽位于肺 bbox 外，也不能带走后方皮肤。
    np.testing.assert_array_equal(actual, body)
