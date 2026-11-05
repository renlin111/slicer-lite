from __future__ import annotations

from pathlib import Path
import tkinter as tk
import numpy as np


class VtkUnavailable(RuntimeError):
    pass


def require_vtk():
    try:
        import vtk  # type: ignore
    except Exception as exc:
        raise VtkUnavailable(str(exc)) from exc
    try:
        vtk.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass
    return vtk


class VtkSegmentationView:
    def __init__(self, parent, width: int = 360, height: int = 360) -> None:
        vtk = require_vtk()
        self.vtk = vtk
        self.widget = tk.Frame(parent, width=width, height=height, bg="#101820", highlightthickness=1, highlightbackground="#52646d")
        self.widget.pack_propagate(False)
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(width, height)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.08, 0.10)
        self.render_window.AddRenderer(self.renderer)
        self.slice_actor = None
        self.plane_actors = {}
        self.model_actor = None
        self.surface_algorithm = None
        self.bounds_actor = None
        self.shape = (1, 1, 1)
        self.spacing = (1.0, 1.0, 1.0)
        self.slice_scale = 1
        self._initialized = False
        self._attached = False
        self._last_mouse: tuple[int, int] | None = None
        self.widget.bind("<Configure>", self._on_widget_configure)
        self.widget.bind("<ButtonPress-1>", self._on_drag_start)
        self.widget.bind("<B1-Motion>", self._on_drag)
        self.widget.bind("<MouseWheel>", self._on_mouse_wheel)

    def pack(self, **kwargs) -> None:
        self.widget.pack(**kwargs)
        self._ensure_embedded()
        if not self._initialized:
            self.interactor.Initialize()
            self._initialized = True

    def reset_scene(self, message: str | None = None) -> None:
        self.renderer.RemoveAllViewProps()
        self.model_actor = None
        self.surface_algorithm = None
        self.slice_actor = None
        self.plane_actors = {}
        self.bounds_actor = None
        self._render()

    def show_mask(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float],
        slice_index: int = 0,
        coronal_index: int | None = None,
        sagittal_index: int | None = None,
        labels: np.ndarray | None = None,
        colors: dict[int, tuple[int, int, int]] | None = None,
    ) -> None:
        vtk = self.vtk
        if mask.size == 0 or not np.any(mask):
            self.reset_scene("当前没有可用于三维重建的分割体素")
            return
        display_mask, display_spacing, slice_scale = _downsample_mask(mask, spacing)
        display_labels = _downsample_labels(labels, slice_scale) if labels is not None else None
        self.shape = tuple(int(v) for v in display_mask.shape)
        self.spacing = tuple(float(v) for v in display_spacing)
        self.slice_scale = max(1, int(slice_scale))
        self.renderer.RemoveAllViewProps()

        vtk_image = self._mask_to_vtk_image(display_mask)
        self.surface_algorithm = None
        self.model_actor = None
        if display_labels is not None and np.any(display_labels):
            first_actor = None
            for label in np.unique(display_labels):
                label = int(label)
                if label <= 0:
                    continue
                actor, smoother = self._surface_actor(
                    display_labels == label,
                    _rgb01((colors or {}).get(label, (255, 86, 74))),
                )
                self.renderer.AddActor(actor)
                if first_actor is None:
                    first_actor = actor
                    self.surface_algorithm = smoother
            self.model_actor = first_actor
        else:
            actor, smoother = self._surface_actor(display_mask, (1.0, 0.30, 0.22))
            self.model_actor = actor
            self.surface_algorithm = smoother
            self.renderer.AddActor(actor)

        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(vtk_image)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self.bounds_actor = vtk.vtkActor()
        self.bounds_actor.SetMapper(outline_mapper)
        self.bounds_actor.GetProperty().SetColor(0.35, 0.55, 0.65)
        self.renderer.AddActor(self.bounds_actor)

        self._add_lights()
        self.update_slice_planes(slice_index, coronal_index, sagittal_index)
        self.renderer.ResetCamera()
        self._render()

    def _surface_actor(self, mask: np.ndarray, color: tuple[float, float, float]):
        vtk = self.vtk
        vtk_image = self._mask_to_vtk_image(mask)
        marching = vtk.vtkMarchingCubes()
        marching.SetInputData(vtk_image)
        marching.SetValue(0, 0.5)
        marching.Update()

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marching.GetOutputPort())
        smoother.SetNumberOfIterations(6)
        smoother.SetRelaxationFactor(0.10)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.86)
        actor.GetProperty().SetSpecular(0.22)
        actor.GetProperty().SetSpecularPower(16)
        return actor, smoother

    def update_slice_plane(self, slice_index: int) -> None:
        self.update_slice_planes(slice_index, None, None)

    def update_slice_planes(
        self,
        axial_index: int,
        coronal_index: int | None = None,
        sagittal_index: int | None = None,
    ) -> None:
        if self.model_actor is None:
            return
        vtk = self.vtk
        for actor in self.plane_actors.values():
            self.renderer.RemoveActor(actor)
        self.plane_actors = {}

        z_count, y_count, x_count = self.shape
        dz, dy, dx = self.spacing
        x_len = max(x_count - 1, 1) * dx
        y_len = max(y_count - 1, 1) * dy
        z_len = max(z_count - 1, 1) * dz

        display_axial = int(axial_index / self.slice_scale)
        z = max(0, min(display_axial, z_count - 1)) * dz
        self._add_plane("axial", (0, 0, z), (x_len, 0, z), (0, y_len, z), (0.25, 0.76, 1.0))

        if coronal_index is not None:
            display_coronal = int(coronal_index / self.slice_scale)
            y = max(0, min(display_coronal, y_count - 1)) * dy
            self._add_plane("coronal", (0, y, 0), (x_len, y, 0), (0, y, z_len), (0.25, 1.0, 0.42))

        if sagittal_index is not None:
            display_sagittal = int(sagittal_index / self.slice_scale)
            x = max(0, min(display_sagittal, x_count - 1)) * dx
            self._add_plane("sagittal", (x, 0, 0), (x, y_len, 0), (x, 0, z_len), (1.0, 0.86, 0.18))
        self._render()

    def _add_plane(self, name: str, origin, point1, point2, color) -> None:
        vtk = self.vtk
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(*origin)
        plane.SetPoint1(*point1)
        plane.SetPoint2(*point2)
        plane.SetResolution(1, 1)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.24)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(*color)
        actor.GetProperty().SetLineWidth(1.2)
        self.renderer.AddActor(actor)
        self.plane_actors[name] = actor
        if name == "axial":
            self.slice_actor = actor

    def save_screenshot(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render_window.Render()
        vtk = self.vtk
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.render_window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        return path

    def save_stl(self, path: str | Path) -> Path:
        if self.surface_algorithm is None:
            raise VtkUnavailable("当前没有可用于 STL 导出的三维重建表面。")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vtk = self.vtk
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(self.surface_algorithm.GetOutputPort())
        writer.Write()
        return path

    def set_model_opacity(self, opacity: float) -> None:
        if self.model_actor is None:
            return
        self.model_actor.GetProperty().SetOpacity(max(0.05, min(float(opacity), 1.0)))
        self._render()

    def _on_canvas_configure(self, event) -> None:
        self._on_widget_configure(event)

    def _on_widget_configure(self, event) -> None:
        self._ensure_embedded()
        self.render_window.SetSize(max(int(event.width), 1), max(int(event.height), 1))
        self._render()

    def _render(self) -> None:
        self._ensure_embedded()
        self.render_window.Render()

    def _ensure_embedded(self) -> None:
        if self._attached:
            return
        self.widget.update_idletasks()
        hwnd = int(self.widget.winfo_id())
        if hwnd <= 0:
            return
        self.render_window.SetWindowInfo(str(hwnd))
        self._attached = True

    def _on_drag_start(self, event) -> None:
        self._last_mouse = (int(event.x), int(event.y))

    def _on_drag(self, event) -> None:
        if self._last_mouse is None:
            self._last_mouse = (int(event.x), int(event.y))
            return
        last_x, last_y = self._last_mouse
        dx = int(event.x) - last_x
        dy = int(event.y) - last_y
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(-dx * 0.5)
        camera.Elevation(dy * 0.5)
        camera.OrthogonalizeViewUp()
        self._last_mouse = (int(event.x), int(event.y))
        self._render()

    def _on_mouse_wheel(self, event) -> None:
        camera = self.renderer.GetActiveCamera()
        camera.Zoom(1.12 if int(event.delta) > 0 else 0.89)
        self._render()

    def _mask_to_vtk_image(self, mask: np.ndarray):
        vtk = self.vtk
        z_count, y_count, x_count = mask.shape
        dz, dy, dx = self.spacing
        image = vtk.vtkImageData()
        image.SetDimensions(x_count, y_count, z_count)
        image.SetSpacing(dx, dy, dz)
        image.SetOrigin(0.0, 0.0, 0.0)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        from vtk.util import numpy_support

        flat = np.ascontiguousarray(mask.astype(np.uint8).ravel(order="C"))
        scalars = numpy_support.numpy_to_vtk(
            num_array=flat,
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        image.GetPointData().SetScalars(scalars)
        return image

    def _add_lights(self) -> None:
        vtk = self.vtk
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(-1, -1, 1)
        light.SetIntensity(0.9)
        self.renderer.AddLight(light)
        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(1, 1, 0.5)
        fill.SetIntensity(0.35)
        self.renderer.AddLight(fill)

    def _add_text(self, text: str) -> None:
        return


def _downsample_mask(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    max_dim: int = 120,
) -> tuple[np.ndarray, tuple[float, float, float], int]:
    z, y, x = mask.shape
    stride = max(1, int(np.ceil(max(z, y, x) / max_dim)))
    if stride == 1:
        return mask.astype(bool), spacing, 1
    reduced = _block_reduce_any(mask.astype(bool), stride)
    dz, dy, dx = spacing
    return reduced, (dz * stride, dy * stride, dx * stride), stride


def _downsample_labels(labels: np.ndarray | None, stride: int) -> np.ndarray | None:
    if labels is None:
        return None
    labels = labels.astype(np.uint8)
    if stride <= 1:
        return labels
    return _block_reduce_max(labels, stride).astype(np.uint8)


def _block_reduce_any(mask: np.ndarray, stride: int) -> np.ndarray:
    padded, shape = _pad_for_blocks(mask, stride)
    z, y, x = shape
    return padded.reshape(z, stride, y, stride, x, stride).any(axis=(1, 3, 5))


def _block_reduce_max(labels: np.ndarray, stride: int) -> np.ndarray:
    padded, shape = _pad_for_blocks(labels, stride)
    z, y, x = shape
    return padded.reshape(z, stride, y, stride, x, stride).max(axis=(1, 3, 5))


def _pad_for_blocks(array: np.ndarray, stride: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    z, y, x = array.shape
    pz = int(np.ceil(z / stride)) * stride
    py = int(np.ceil(y / stride)) * stride
    px = int(np.ceil(x / stride)) * stride
    padded = np.zeros((pz, py, px), dtype=array.dtype)
    padded[:z, :y, :x] = array
    return padded, (pz // stride, py // stride, px // stride)


def _rgb01(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(max(0.0, min(float(c) / 255.0, 1.0)) for c in color)
