from __future__ import annotations

from pathlib import Path

import numpy as np


def resize_nearest(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    scale = min(max_w / w, max_h / h)
    scale = max(scale, 0.1)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    yy = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
    xx = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
    return image[yy[:, None], xx]


def compose_overlay(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    if mask is not None:
        edge = mask.astype(bool)
        rgb[edge, 0] = 255
        rgb[edge, 1] = (rgb[edge, 1] * 0.25).astype(np.uint8)
        rgb[edge, 2] = (rgb[edge, 2] * 0.25).astype(np.uint8)
    return rgb


def compose_label_overlay(
    gray: np.ndarray,
    labels: np.ndarray | None = None,
    colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    if labels is None:
        return rgb
    colors = colors or {
        1: (255, 86, 74),
        2: (84, 190, 255),
        3: (255, 202, 40),
        4: (118, 221, 118),
    }
    labels = labels.astype(np.uint8)
    for label, color in colors.items():
        region = labels == label
        if not region.any():
            continue
        base = rgb[region].astype(np.float32)
        tint = np.array(color, dtype=np.float32)
        rgb[region] = (base * 0.58 + tint * 0.42).clip(0, 255).astype(np.uint8)
        edge = _label_edges(region)
        if not edge.any():
            continue
        rgb[edge, 0] = color[0]
        rgb[edge, 1] = color[1]
        rgb[edge, 2] = color[2]
    return rgb


def _label_edges(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=bool)
    m = mask.astype(bool)
    if m.size == 0:
        return m
    padded = np.pad(m, 1, mode="constant")
    inner = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return m & ~inner


def save_ppm(rgb: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(rgb).tobytes())
    return path


def save_image(rgb: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        try:
            from PIL import Image

            Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)), mode="RGB").save(path)
            return path
        except Exception:
            return save_ppm(rgb, path.with_suffix(".ppm"))
    return save_ppm(rgb, path)


def surface_points(mask: np.ndarray, max_points: int = 4500) -> np.ndarray:
    if mask is None or not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    m = mask.astype(bool)
    padded = np.pad(m, 1, mode="constant")
    inner = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    pts = np.argwhere(m & ~inner).astype(np.float32)
    if pts.shape[0] > max_points:
        step = max(1, pts.shape[0] // max_points)
        pts = pts[::step][:max_points]
    return pts


def project_points(points: np.ndarray, shape: tuple[int, int, int], w: int, h: int) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    zmax, ymax, xmax = [max(v - 1, 1) for v in shape]
    z = points[:, 0] / zmax - 0.5
    y = points[:, 1] / ymax - 0.5
    x = points[:, 2] / xmax - 0.5
    u = (x - y) * 0.95
    v = (x + y) * 0.34 - z * 0.88
    px = (w * 0.5 + u * min(w, h) * 0.58).astype(np.int32)
    py = (h * 0.55 + v * min(w, h) * 0.58).astype(np.int32)
    return np.column_stack([px, py])
