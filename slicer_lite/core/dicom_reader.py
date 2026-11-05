from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np


TAG_NAMES = {
    (0x0002, 0x0010): "TransferSyntaxUID",
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x103E): "SeriesDescription",
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0018, 0x0050): "SliceThickness",
    (0x0020, 0x0011): "SeriesNumber",
    (0x0020, 0x0013): "InstanceNumber",
    (0x0020, 0x0032): "ImagePositionPatient",
    (0x0028, 0x0002): "SamplesPerPixel",
    (0x0028, 0x0004): "PhotometricInterpretation",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0030): "PixelSpacing",
    (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0101): "BitsStored",
    (0x0028, 0x0103): "PixelRepresentation",
    (0x0028, 0x1050): "WindowCenter",
    (0x0028, 0x1051): "WindowWidth",
    (0x0028, 0x1052): "RescaleIntercept",
    (0x0028, 0x1053): "RescaleSlope",
    (0x7FE0, 0x0010): "PixelData",
}

LONG_VR = {"OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"}
TEXT_VR = {"AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT", "PN", "SH", "ST", "TM", "UI", "UT"}
TAG_VR = {
    (0x0018, 0x0050): "DS",
    (0x0020, 0x0011): "IS",
    (0x0020, 0x0013): "IS",
    (0x0020, 0x0032): "DS",
    (0x0028, 0x0002): "US",
    (0x0028, 0x0010): "US",
    (0x0028, 0x0011): "US",
    (0x0028, 0x0030): "DS",
    (0x0028, 0x0100): "US",
    (0x0028, 0x0101): "US",
    (0x0028, 0x0103): "US",
    (0x0028, 0x1050): "DS",
    (0x0028, 0x1051): "DS",
    (0x0028, 0x1052): "DS",
    (0x0028, 0x1053): "DS",
}


@dataclass
class DicomSlice:
    path: Path
    pixels: np.ndarray
    metadata: dict[str, Any]

    @property
    def instance_number(self) -> int:
        return int(_first_value(self.metadata.get("InstanceNumber"), 0))

    @property
    def z_position(self) -> float:
        ipp = self.metadata.get("ImagePositionPatient")
        if isinstance(ipp, (list, tuple)) and len(ipp) >= 3:
            return float(ipp[2])
        return float(self.instance_number)


@dataclass
class Volume:
    data: np.ndarray
    spacing: tuple[float, float, float]
    slices: list[DicomSlice]
    metadata: dict[str, Any]

    @property
    def shape_text(self) -> str:
        z, y, x = self.data.shape
        return f"{x} x {y} x {z}"


class DicomReadError(RuntimeError):
    pass


def load_dicom_folder(folder: str | Path) -> Volume:
    root = Path(folder)
    if not root.exists():
        raise DicomReadError(f"文件夹不存在：{root}")

    slices: list[DicomSlice] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("._"):
            continue
        try:
            slices.append(read_dicom_slice(path))
        except Exception:
            continue

    if not slices:
        raise DicomReadError("未找到可读取的未压缩 DICOM 图像切片。")

    slices.sort(key=lambda item: (item.z_position, item.instance_number, str(item.path)))
    stack = np.stack([item.pixels for item in slices]).astype(np.float32)

    meta = dict(slices[0].metadata)
    row_spacing, col_spacing = _pixel_spacing(meta.get("PixelSpacing"))
    if len(slices) > 1:
        z_values = [item.z_position for item in slices]
        dz = float(np.median(np.abs(np.diff(z_values)))) or float(meta.get("SliceThickness") or 1.0)
    else:
        dz = float(meta.get("SliceThickness") or 1.0)

    return Volume(data=stack, spacing=(dz, row_spacing, col_spacing), slices=slices, metadata=meta)


def read_dicom_slice(path: str | Path) -> DicomSlice:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 132 or raw[128:132] != b"DICM":
        raise DicomReadError("缺少 DICOM 文件头。")

    pos = 132
    meta: dict[str, Any] = {}

    while pos + 8 <= len(raw):
        group, element = struct.unpack_from("<HH", raw, pos)
        if group != 0x0002:
            break
        tag, vr, value, pos = _read_element(raw, pos, explicit_vr=True)
        name = TAG_NAMES.get(tag)
        if name:
            meta[name] = _decode_value(vr, value)

    transfer_syntax = str(meta.get("TransferSyntaxUID", "1.2.840.10008.1.2.1")).strip("\x00 ")
    if transfer_syntax.startswith("1.2.840.10008.1.2.4"):
        raise DicomReadError("当前轻量读取器不支持压缩格式的 DICOM 传输语法。")
    explicit_vr = transfer_syntax != "1.2.840.10008.1.2"

    pixel_data: bytes | None = None
    while pos + 8 <= len(raw):
        try:
            tag, vr, value, pos = _read_element(raw, pos, explicit_vr=explicit_vr)
        except DicomReadError:
            break
        if not explicit_vr:
            vr = TAG_VR.get(tag, vr)
        name = TAG_NAMES.get(tag)
        if name == "PixelData":
            pixel_data = value
            break
        if name:
            meta[name] = _decode_value(vr, value)

    if pixel_data is None:
        raise DicomReadError("未找到 PixelData 标签。")

    rows = int(meta.get("Rows", 0))
    cols = int(meta.get("Columns", 0))
    bits = int(meta.get("BitsAllocated", 16))
    signed = int(meta.get("PixelRepresentation", 0)) == 1
    samples = int(meta.get("SamplesPerPixel", 1))
    if rows <= 0 or cols <= 0 or samples != 1:
        raise DicomReadError("不支持该 DICOM 图像维度。")
    if bits == 16:
        dtype = np.int16 if signed else np.uint16
    elif bits == 8:
        dtype = np.int8 if signed else np.uint8
    else:
        raise DicomReadError(f"不支持 BitsAllocated={bits}。")

    expected = rows * cols
    pixels = np.frombuffer(pixel_data, dtype=dtype, count=expected)
    if pixels.size != expected:
        raise DicomReadError("PixelData 长度与 Rows x Columns 不匹配。")
    pixels = pixels.reshape(rows, cols).astype(np.float32)

    slope = float(_first_value(meta.get("RescaleSlope"), 1.0) or 1.0)
    intercept = float(_first_value(meta.get("RescaleIntercept"), 0.0) or 0.0)
    pixels = pixels * slope + intercept

    return DicomSlice(path=path, pixels=pixels, metadata=meta)


def _read_element(raw: bytes, pos: int, explicit_vr: bool) -> tuple[tuple[int, int], str, bytes, int]:
    if pos + 8 > len(raw):
        raise DicomReadError("文件意外结束。")
    group, element = struct.unpack_from("<HH", raw, pos)
    tag = (group, element)
    pos += 4
    if explicit_vr:
        vr = raw[pos : pos + 2].decode("ascii", errors="ignore")
        pos += 2
        if vr in LONG_VR:
            pos += 2
            length = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
        else:
            length = struct.unpack_from("<H", raw, pos)[0]
            pos += 2
    else:
        vr = "UN"
        length = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
    if length == 0xFFFFFFFF:
        raise DicomReadError("不支持未定义长度的序列。")
    end = pos + length
    if end > len(raw):
        raise DicomReadError("DICOM 元素长度无效。")
    return tag, vr, raw[pos:end], end


def _decode_value(vr: str, value: bytes) -> Any:
    if vr in TEXT_VR or vr == "UN":
        text = value.decode("utf-8", errors="ignore").strip("\x00 ")
        parts = [item for item in text.split("\\") if item != ""]
        if vr == "DS":
            return [_safe_float(item) for item in parts] if len(parts) > 1 else _safe_float(text)
        if vr == "IS":
            return [_safe_int(item) for item in parts] if len(parts) > 1 else _safe_int(text)
        return parts if len(parts) > 1 else text
    if vr in {"US", "SS", "UL", "SL"}:
        fmt = {"US": "<H", "SS": "<h", "UL": "<I", "SL": "<i"}[vr]
        size = struct.calcsize(fmt)
        vals = [struct.unpack_from(fmt, value, i)[0] for i in range(0, len(value) - size + 1, size)]
        return vals if len(vals) > 1 else (vals[0] if vals else 0)
    return value


def _safe_float(text: Any) -> float:
    try:
        return float(text)
    except Exception:
        return 0.0


def _safe_int(text: Any) -> int:
    try:
        return int(float(text))
    except Exception:
        return 0


def _first_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _pixel_spacing(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]) or 1.0, float(value[1]) or 1.0
    return 1.0, 1.0
