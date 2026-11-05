from __future__ import annotations

import numpy as np


def segmentation_statistics(
    volume: np.ndarray,
    mask: np.ndarray | None,
    spacing: tuple[float, float, float],
    slice_index: int,
) -> dict[str, float | int | str]:
    if mask is None or not mask.any():
        return {
            "source": "none",
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "volume_cm3": 0.0,
            "slice_area_mm2": 0.0,
            "hu_mean": 0.0,
            "hu_min": 0.0,
            "hu_max": 0.0,
        }

    dz, dy, dx = spacing
    voxel_count = int(mask.sum())
    voxel_volume = float(dz * dy * dx)
    values = volume[mask]
    z = max(0, min(int(slice_index), mask.shape[0] - 1))
    slice_pixels = int(mask[z].sum())
    return {
        "source": "segmentation",
        "voxel_count": voxel_count,
        "volume_mm3": voxel_count * voxel_volume,
        "volume_cm3": voxel_count * voxel_volume / 1000.0,
        "slice_area_mm2": slice_pixels * float(dy * dx),
        "hu_mean": float(values.mean()) if values.size else 0.0,
        "hu_min": float(values.min()) if values.size else 0.0,
        "hu_max": float(values.max()) if values.size else 0.0,
    }


def format_statistics(stats: dict[str, float | int | str]) -> str:
    return "\n".join(
        [
            "分割统计",
            f"体素数量：{int(stats['voxel_count']):,}",
            f"体积：{float(stats['volume_mm3']):.2f} mm^3",
            f"体积：{float(stats['volume_cm3']):.3f} cm^3",
            f"当前切片面积：{float(stats['slice_area_mm2']):.2f} mm^2",
            f"HU 均值：{float(stats['hu_mean']):.2f}",
            f"HU 最小值：{float(stats['hu_min']):.2f}",
            f"HU 最大值：{float(stats['hu_max']):.2f}",
        ]
    )
