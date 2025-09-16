# area_calc.py
# Tính diện tích (m²) từ mask và khoảng cách camera→vật thể (m).
# Ưu tiên dùng calibration.json (px_per_meter_at_1m); nếu không có thì dùng HFOV.

import json, math
from typing import Optional
import numpy as np

class AreaCalculator:
    """
    - Nếu có calibration.json với 'px_per_meter_at_1m' = k (px/m tại d=1m):
        m_per_px(d) = d / k
    - Nếu không: dùng HFOV (độ) để ước lượng:
        scene_width = 2 * d * tan(HFOV/2)
        m_per_px(d) = scene_width / frame_width
    """
    def __init__(self, frame_width: int, calib_path: Optional[str] = None, hfov_deg: float = 62.0):
        self.frame_width = int(frame_width)
        self.k_px_per_m_at_1m: Optional[float] = None
        self.hfov_deg = float(hfov_deg)

        if calib_path:
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "px_per_meter_at_1m" in data:
                    self.k_px_per_m_at_1m = float(data["px_per_meter_at_1m"])
                if "hfov_deg" in data:
                    self.hfov_deg = float(data["hfov_deg"])
            except Exception:
                pass

    def meters_per_pixel(self, distance_m: float) -> float:
        if distance_m <= 0:
            raise ValueError("distance_m phải > 0")
        if self.k_px_per_m_at_1m and self.k_px_per_m_at_1m > 0:
            return distance_m / self.k_px_per_m_at_1m
        hfov_rad = math.radians(self.hfov_deg)
        scene_width_m = 2.0 * distance_m * math.tan(hfov_rad / 2.0)
        return scene_width_m / float(self.frame_width)

    def area_m2(self, mask: np.ndarray, distance_m: float) -> float:
        if mask is None:
            return 0.0
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        m_per_px = self.meters_per_pixel(distance_m)
        px_count = int(mask.sum())
        return float(px_count) * (m_per_px ** 2)


def extract_union_mask(result) -> Optional[np.ndarray]:
    """
    Lấy mask hợp nhất (union) từ Ultralytics Results.
    Trả về mask nhị phân uint8 (H, W) hoặc None nếu không có.
    """
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return None

    arr = masks.data
    # Chuyển sang numpy an toàn
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    else:
        arr = np.asarray(arr)

    if arr.ndim == 3:
        union = arr.max(axis=0)
    elif arr.ndim == 2:
        union = arr
    else:
        return None

    return (union > 0.5).astype(np.uint8)
