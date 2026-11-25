"""Unit tests verifying 1:1 Overlay-vs-Click physical alignment across 100%, 125%, 150% DPI & 16:10 letterboxes."""

from unittest.mock import patch

import pytest

from onmyoji_helper.automation.coordinate_mapper import (
    CoordinateMapper,
    project_cropped_to_physical_client,
)
from onmyoji_helper.domain.models import CropOffset
from onmyoji_helper.ui.hud_overlay import BattleOverlayUI
from onmyoji_helper.vision.turn_detector import TurnDetector


@pytest.mark.parametrize("dpi_scale", [1.0, 1.25, 1.50, 2.0])
def test_dwm_dpi_scaled_window_overlay_and_click_1_to_1_alignment(dpi_scale: float) -> None:
    """Regression test: prevents the 1/1.25x top-left shrinkage under Windows 125%/150% DPI scaling.

    When a logical 1144x648 game window is stretched by DWM to `client_w = 1144 * dpi_scale`
    (`1430x810` at 125% DPI, `1716x972` at 150% DPI), BitBlt captures the 1144x648 buffer at
    `(0, 0)` with `x0=0, y0=0`. Both BattleOverlayUI and CoordinateMapper must project the
    Skill Bar (`0.665..0.998`, `0.85..0.995`) and unit slots onto the exact physical bottom-right
    region without shrinking by `1 / dpi_scale`.
    """
    img_w, img_h = 1144, 648
    client_w = int(round(img_w * dpi_scale))
    client_h = int(round(img_h * dpi_scale))
    origin_x, origin_y = 200, 120
    crop = CropOffset(x0=0, y0=0, client_w=client_w, client_h=client_h)

    td = TurnDetector()
    rx1, ry1, rx2, ry2 = td.SKILL_BAR_ROI

    # 1. Verify Overlay physical canvas coordinates land at exact 66.5%..99.8% of physical window
    ov_x1, ov_y1 = BattleOverlayUI._project_point(
        img_w * rx1, img_h * ry1, client_w, client_h, img_w, img_h, crop
    )
    ov_x2, ov_y2 = BattleOverlayUI._project_point(
        img_w * rx2, img_h * ry2, client_w, client_h, img_w, img_h, crop
    )
    assert abs(ov_x1 / client_w - rx1) < 1e-6
    assert abs(ov_x2 / client_w - rx2) < 1e-6
    assert abs(ov_y1 / client_h - ry1) < 1e-6
    assert abs(ov_y2 / client_h - ry2) < 1e-6

    # 2. Verify CoordinateMapper screen click pixel matches Overlay physical coordinate + window origin 1:1
    mapper = CoordinateMapper()
    s3_rx1, s3_ry1, s3_rx2, s3_ry2 = td.PHYSICAL_SLOTS["slot_4"]
    s3_img_x = ((s3_rx1 + s3_rx2) * 0.5) * img_w
    s3_img_y = ((s3_ry1 + s3_ry2) * 0.5) * img_h

    ov_sx, ov_sy = BattleOverlayUI._project_point(
        s3_img_x, s3_img_y, client_w, client_h, img_w, img_h, crop
    )

    with (
        patch("onmyoji_helper.automation.coordinate_mapper.win32gui.IsWindow", return_value=True),
        patch(
            "onmyoji_helper.automation.coordinate_mapper.win32gui.GetClientRect",
            return_value=(0, 0, client_w, client_h),
        ),
        patch(
            "onmyoji_helper.automation.coordinate_mapper.win32gui.ClientToScreen",
            return_value=(origin_x, origin_y),
        ),
        patch(
            "onmyoji_helper.automation.coordinate_mapper.win32gui.GetWindowRect",
            return_value=(origin_x, origin_y, origin_x + client_w, origin_y + client_h),
        ),
        patch("onmyoji_helper.automation.coordinate_mapper.get_root_hwnd", return_value=2002),
    ):
        mapped = mapper.map_image_to_screen(2002, s3_img_x, s3_img_y, img_w, img_h, crop=crop)
        assert mapped is not None
        click_sx, click_sy, _ = mapped
        assert click_sx == origin_x + int(round(ov_sx))
        assert click_sy == origin_y + int(round(ov_sy))
        # S3 button center (~0.96, ~0.9325) must sit in the far bottom-right corner (> 90% X, > 90% Y)
        assert (click_sx - origin_x) > client_w * 0.92
        assert (click_sy - origin_y) > client_h * 0.90


def test_letterbox_16_by_10_crop_offset_and_dpi_combined() -> None:
    mapper = CoordinateMapper()
    # 1920x1200 (16:10) physical fullscreen at 125% DPI: logical buffer 1536x960 with 48px top/bottom bars
    # around a 1536x864 16:9 frame, located on secondary monitor at (1920, -360).
    crop = CropOffset(x0=0, y0=48, client_w=1920, client_h=1200)
    px, py = project_cropped_to_physical_client(768.0, 432.0, 1536, 864, 1920, 1200, crop=crop)
    assert int(round(px)) == 960
    assert int(round(py)) == 600

    with (
        patch("onmyoji_helper.automation.coordinate_mapper.win32gui.IsWindow", return_value=True),
        patch(
            "onmyoji_helper.automation.coordinate_mapper.win32gui.GetClientRect",
            return_value=(0, 0, 1920, 1200),
        ),
        patch(
            "onmyoji_helper.automation.coordinate_mapper.win32gui.ClientToScreen",
            return_value=(1920, -360),
        ),
        patch(
            "onmyoji_helper.automation.coordinate_mapper.win32gui.GetWindowRect",
            return_value=(1920, -360, 3840, 840),
        ),
        patch("onmyoji_helper.automation.coordinate_mapper.get_root_hwnd", return_value=1001),
    ):
        mapped = mapper.map_image_to_screen(
            hwnd=1001,
            img_x=768.0,
            img_y=432.0,
            img_w=1536,
            img_h=864,
            crop=crop,
        )
        assert mapped is not None
        sx, sy, bounds = mapped
        assert sx == 1920 + 960
        assert sy == -360 + 600

        for _ in range(50):
            jx, jy = CoordinateMapper.apply_gaussian_jitter(1921, -359, 12.0, bounds)
            assert bounds[0] + 4 <= jx <= bounds[2] - 4
            assert bounds[1] + 4 <= jy <= bounds[3] - 4
