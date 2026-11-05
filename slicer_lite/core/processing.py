from __future__ import annotations

from collections import deque
import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None


def window_to_uint8(image: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(float(width), 1.0)
    lo = float(center) - width / 2.0
    hi = float(center) + width / 2.0
    scaled = (image.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def auto_window(volume: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(volume[np.isfinite(volume)], [1, 99])
    width = max(float(hi - lo), 1.0)
    return float((lo + hi) / 2.0), width


def enhance_uint8(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "histogram":
        return histogram_equalization(image)
    if mode == "denoise":
        return mean_filter(image)
    if mode == "sharpen":
        blurred = mean_filter(image).astype(np.int16)
        sharp = image.astype(np.int16) * 2 - blurred
        return np.clip(sharp, 0, 255).astype(np.uint8)
    return image


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    hist = np.bincount(image.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1)
    table = np.round(cdf * 255).astype(np.uint8)
    return table[image]


def mean_filter(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    total = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    return np.clip(total / 9.0, 0, 255).astype(np.uint8)


def segment_volume(
    volume: np.ndarray,
    mode: str,
    lower: float,
    upper: float,
    seed: tuple[int, int, int] | None = None,
    foreground_seeds: list[tuple[int, int, int]] | None = None,
    background_seeds: list[tuple[int, int, int]] | None = None,
    tolerance: float = 80.0,
) -> np.ndarray:
    if mode == "bone":
        mask = volume > 180
    elif mode == "soft_tissue":
        mask = (volume > -120) & (volume < 220)
        mask = _central_crop(mask, keep_y=0.72, keep_x=0.72)
    elif mode == "body":
        mask = volume > -350
    elif mode == "otsu":
        threshold = otsu_threshold(volume)
        mask = volume >= threshold
    elif mode == "region_grow":
        seeds = foreground_seeds or ([seed] if seed is not None else [])
        if not seeds:
            return np.zeros_like(volume, dtype=bool)
        mask = region_grow(volume, seeds, tolerance, background_seeds or [])
    else:
        mask = (volume >= lower) & (volume <= upper)
    # Keep the default post-processing lightweight enough for live demo on full CT volumes.
    return clean_mask(mask, min_component_size=0, fill_holes=False)


def otsu_threshold(volume: np.ndarray) -> float:
    values = volume[np.isfinite(volume)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=512)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(np.mean(values))
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * centers)
    sum_total = sum_bg[-1]
    valid = (weight_bg > 0) & (weight_fg > 0)
    variance = np.zeros_like(centers, dtype=np.float64)
    mean_bg = np.zeros_like(centers, dtype=np.float64)
    mean_fg = np.zeros_like(centers, dtype=np.float64)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    variance[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    return float(centers[int(np.argmax(variance))])


def region_grow(
    volume: np.ndarray,
    seeds: list[tuple[int, int, int]] | tuple[int, int, int],
    tolerance: float,
    background_seeds: list[tuple[int, int, int]] | None = None,
    max_voxels: int = 1_500_000,
) -> np.ndarray:
    depth, height, width = volume.shape
    if isinstance(seeds, tuple):
        seeds = [seeds]
    seeds = [s for s in seeds if _inside(s, volume.shape)]
    background_seeds = [s for s in (background_seeds or []) if _inside(s, volume.shape)]
    if not seeds:
        return np.zeros_like(volume, dtype=bool)
    fg_values = np.array([volume[z, y, x] for z, y, x in seeds], dtype=np.float32)
    center = float(fg_values.mean())
    lo = center - float(tolerance)
    hi = center + float(tolerance)
    candidate = (volume >= lo) & (volume <= hi)
    if background_seeds:
        bg_values = np.array([volume[z, y, x] for z, y, x in background_seeds], dtype=np.float32)
        bg_center = float(bg_values.mean())
        candidate &= np.abs(volume - center) <= np.abs(volume - bg_center)
    visited = np.zeros_like(candidate, dtype=bool)
    mask = np.zeros_like(candidate, dtype=bool)
    q: deque[tuple[int, int, int]] = deque()
    for z0, y0, x0 in seeds:
        q.append((z0, y0, x0))
        visited[z0, y0, x0] = True
    added = 0
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while q and added < max_voxels:
        z, y, x = q.popleft()
        if not candidate[z, y, x]:
            continue
        mask[z, y, x] = True
        added += 1
        for dz, dy, dx in neighbors:
            nz, ny, nx = z + dz, y + dy, x + dx
            if nz < 0 or nz >= depth or ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if visited[nz, ny, nx]:
                continue
            visited[nz, ny, nx] = True
            if candidate[nz, ny, nx]:
                q.append((nz, ny, nx))
    return mask


def _inside(seed: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    z, y, x = seed
    return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]


def clean_mask(mask: np.ndarray, min_component_size: int = 0, fill_holes: bool = False) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.ndim == 3:
        out = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            slice_mask = _close2d(_open2d(mask[i]))
            if min_component_size > 0:
                slice_mask = _remove_small_components2d(slice_mask, min_component_size)
            if fill_holes:
                slice_mask = _fill_holes2d(slice_mask)
            out[i] = slice_mask
        return out
    out = _close2d(_open2d(mask))
    if min_component_size > 0:
        out = _remove_small_components2d(out, min_component_size)
    if fill_holes:
        out = _fill_holes2d(out)
    return out


def postprocess_mask(mask: np.ndarray, keep_largest: bool = False, fill_holes: bool = False) -> np.ndarray:
    out = mask.astype(bool)
    if ndi is not None:
        return _postprocess_mask_fast(out, keep_largest, fill_holes)
    if keep_largest:
        out = _keep_largest_component_per_slice(out)
    if fill_holes:
        out = _fill_holes_bounded(out)
    return out


def _postprocess_mask_fast(mask: np.ndarray, keep_largest: bool, fill_holes: bool) -> np.ndarray:
    if mask.ndim != 3:
        out = _largest_component2d_fast(mask) if keep_largest else mask.copy()
        return ndi.binary_fill_holes(out).astype(bool) if fill_holes else out
    out = np.zeros_like(mask, dtype=bool) if keep_largest else mask.copy()
    for z in range(mask.shape[0]):
        slice_mask = mask[z]
        if keep_largest:
            slice_mask = _largest_component2d_fast(slice_mask)
        if fill_holes and slice_mask.any():
            slice_mask = ndi.binary_fill_holes(slice_mask)
        out[z] = slice_mask
    return out


def _largest_component2d_fast(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def mask_edges(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d.astype(bool)
    padded = np.pad(m, 1, mode="constant")
    inner = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return m & ~inner


def _central_crop(mask: np.ndarray, keep_y: float, keep_x: float) -> np.ndarray:
    z, y, x = mask.shape
    y0 = int(y * (1 - keep_y) / 2)
    y1 = y - y0
    x0 = int(x * (1 - keep_x) / 2)
    x1 = x - x0
    out = np.zeros_like(mask)
    out[:, y0:y1, x0:x1] = mask[:, y0:y1, x0:x1]
    return out


def _open2d(mask: np.ndarray) -> np.ndarray:
    return _dilate2d(_erode2d(mask))


def _close2d(mask: np.ndarray) -> np.ndarray:
    return _erode2d(_dilate2d(mask))


def _erode2d(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _dilate2d(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _remove_small_components2d(mask: np.ndarray, min_size: int) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = mask.copy()
    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            pts: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))
            if len(pts) < min_size:
                for py, px in pts:
                    out[py, px] = False
    return out


def _fill_holes2d(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    background = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if background[0, x]:
            q.append((0, x)); visited[0, x] = True
        if background[h - 1, x] and not visited[h - 1, x]:
            q.append((h - 1, x)); visited[h - 1, x] = True
    for y in range(h):
        if background[y, 0] and not visited[y, 0]:
            q.append((y, 0)); visited[y, 0] = True
        if background[y, w - 1] and not visited[y, w - 1]:
            q.append((y, w - 1)); visited[y, w - 1] = True
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if visited[ny, nx] or not background[ny, nx]:
                continue
            visited[ny, nx] = True
            q.append((ny, nx))
    holes = background & ~visited
    return mask | holes


def _keep_largest_component_per_slice(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        return _largest_component2d(mask)
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        out[z] = _largest_component2d(mask[z])
    return out


def _fill_holes_bounded(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        return _fill_holes_bounded2d(mask)
    out = mask.copy()
    for z in range(mask.shape[0]):
        out[z] = _fill_holes_bounded2d(mask[z])
    return out


def _fill_holes_bounded2d(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask.astype(bool)
    y0 = max(int(ys.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, mask.shape[0])
    x0 = max(int(xs.min()) - 1, 0)
    x1 = min(int(xs.max()) + 2, mask.shape[1])
    out = mask.astype(bool).copy()
    out[y0:y1, x0:x1] = _fill_holes2d(out[y0:y1, x0:x1])
    return out


def _largest_component2d(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros_like(mask, dtype=bool)
    y0 = max(int(ys.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, mask.shape[0])
    x0 = max(int(xs.min()) - 1, 0)
    x1 = min(int(xs.max()) + 2, mask.shape[1])
    cropped = mask[y0:y1, x0:x1]
    h, w = cropped.shape
    visited = np.zeros_like(cropped, dtype=bool)
    best: list[tuple[int, int]] = []
    for y in range(h):
        row = cropped[y]
        for x in np.flatnonzero(row):
            x = int(x)
            if visited[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            pts: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not cropped[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))
            if len(pts) > len(best):
                best = pts
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y0 + y, x0 + x] = True
    return out


def _largest_component2d_old(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    foreground = list(zip(*np.nonzero(mask)))
    for y, x in foreground:
        if visited[y, x]:
            continue
        q: deque[tuple[int, int]] = deque([(int(y), int(x))])
        visited[y, x] = True
        pts: list[tuple[int, int]] = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                q.append((ny, nx))
        if len(pts) > len(best):
            best = pts
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out
