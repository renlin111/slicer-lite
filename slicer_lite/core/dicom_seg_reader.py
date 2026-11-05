from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

from slicer_lite.core.dicom_reader import Volume


LONG_VR = {"OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"}


@dataclass
class SegImportResult:
    mask: np.ndarray
    label: str = "SEG"
    algorithm: str = "UNKNOWN"
    frames: int = 0
    aligned: bool = False


class DicomSegReadError(RuntimeError):
    pass


def load_dicom_seg(path: str | Path, volume: Volume) -> SegImportResult:
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise DicomSegReadError("SEG 文件过小。")
    pos = 132 if len(raw) >= 132 and raw[128:132] == b"DICM" else 0

    modality = ""
    rows = cols = frames = bits_allocated = 0
    pixel_data: bytes | None = None
    label = "SEG"
    algorithm = "UNKNOWN"

    while pos + 8 <= len(raw):
        tag, vr, value_pos, length, next_pos = _read_element(raw, pos)
        if tag == (0x0008, 0x0060):
            modality = _text(raw[value_pos : value_pos + length])
        elif tag == (0x0028, 0x0010):
            rows = _u16(raw, value_pos)
        elif tag == (0x0028, 0x0011):
            cols = _u16(raw, value_pos)
        elif tag == (0x0028, 0x0008):
            frames = int(float(_text(raw[value_pos : value_pos + length]) or 0))
        elif tag == (0x0028, 0x0100):
            bits_allocated = _u16(raw, value_pos)
        elif tag == (0x0062, 0x0005):
            label = _text(raw[value_pos : value_pos + length]) or label
        elif tag == (0x0062, 0x0008):
            algorithm = _text(raw[value_pos : value_pos + length]) or algorithm
        elif tag == (0x7FE0, 0x0010):
            pixel_data = raw[value_pos : value_pos + length]
            break
        pos = next_pos

    if modality != "SEG":
        raise DicomSegReadError("所选文件不是 DICOM SEG 对象。")
    if rows <= 0 or cols <= 0 or frames <= 0 or bits_allocated != 1 or pixel_data is None:
        raise DicomSegReadError("不支持该 SEG 对象。需要 1-bit 多帧分割对象。")

    frame_masks = _unpack_1bit_frames(pixel_data, frames, rows, cols)
    depth, height, width = volume.data.shape
    if rows != height or cols != width:
        raise DicomSegReadError(f"SEG 尺寸 {cols}x{rows} 与 CT 尺寸 {width}x{height} 不匹配。")

    sop_order = _referenced_sop_order(raw, volume)
    if sop_order:
        mask = np.zeros((depth, height, width), dtype=bool)
        n = min(len(sop_order), frames)
        for frame_index in range(n):
            target_index = sop_order[frame_index]
            if 0 <= target_index < depth:
                mask[target_index] = frame_masks[frame_index]
        aligned = True
    elif frames == depth:
        mask = frame_masks
        aligned = True
    else:
        mask = np.zeros((depth, height, width), dtype=bool)
        n = min(depth, frames)
        start_dst = max((depth - n) // 2, 0)
        start_src = max((frames - n) // 2, 0)
        mask[start_dst : start_dst + n] = frame_masks[start_src : start_src + n]

        aligned = False

    return SegImportResult(mask=mask, label=label, algorithm=algorithm, frames=frames, aligned=aligned)


def _read_element(raw: bytes, pos: int) -> tuple[tuple[int, int], str, int, int, int]:
    if pos + 8 > len(raw):
        raise DicomSegReadError("SEG 文件意外结束。")
    group, element = struct.unpack_from("<HH", raw, pos)
    pos += 4
    vr = raw[pos : pos + 2].decode("ascii", errors="ignore")
    pos += 2
    if vr in LONG_VR:
        pos += 2
        length = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
    else:
        length = struct.unpack_from("<H", raw, pos)[0]
        pos += 2
    if length == 0xFFFFFFFF:
        raise DicomSegReadError("当前轻量导入器不支持未定义长度的 SEG 序列。")
    return (group, element), vr, pos, length, pos + length


def _u16(raw: bytes, pos: int) -> int:
    return struct.unpack_from("<H", raw, pos)[0]


def _text(value: bytes) -> str:
    return value.decode("utf-8", errors="ignore").strip("\x00 ")


def _unpack_1bit_frames(pixel_data: bytes, frames: int, rows: int, cols: int) -> np.ndarray:
    bits_per_frame = rows * cols
    bytes_per_frame = (bits_per_frame + 7) // 8
    masks = np.zeros((frames, rows, cols), dtype=bool)
    for frame in range(frames):
        start = frame * bytes_per_frame
        chunk = pixel_data[start : start + bytes_per_frame]
        if len(chunk) < bytes_per_frame:
            break
        bits = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8), bitorder="little")
        masks[frame] = bits[:bits_per_frame].reshape(rows, cols).astype(bool)
    return masks


def _referenced_sop_order(raw: bytes, volume: Volume) -> list[int]:
    uid_to_index: dict[str, int] = {}
    for i, item in enumerate(volume.slices):
        uid = str(item.metadata.get("SOPInstanceUID", "")).strip("\x00 ")
        if uid:
            uid_to_index[uid] = i
    if not uid_to_index:
        return []

    hits: list[tuple[int, int]] = []
    for uid, index in uid_to_index.items():
        needle = uid.encode("ascii", errors="ignore")
        start = 0
        while needle:
            pos = raw.find(needle, start)
            if pos < 0:
                break
            hits.append((pos, index))
            start = pos + len(needle)
            break
    hits.sort(key=lambda item: item[0])

    order: list[int] = []
    last = None
    for _, index in hits:
        if index == last:
            continue
        order.append(index)
        last = index
    return order
