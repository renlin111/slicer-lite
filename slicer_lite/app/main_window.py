from __future__ import annotations

from pathlib import Path
import traceback
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from slicer_lite.core.dicom_reader import DicomReadError, Volume, load_dicom_folder
from slicer_lite.core.dicom_seg_reader import DicomSegReadError, load_dicom_seg
from slicer_lite.core.processing import (
    auto_window,
    enhance_uint8,
    mask_edges,
    postprocess_mask,
    segment_volume,
    window_to_uint8,
)
from slicer_lite.core.rendering import (
    compose_label_overlay,
    compose_overlay,
    project_points,
    resize_nearest,
    save_image,
    save_ppm,
    surface_points,
)
from slicer_lite.core.statistics import format_statistics, segmentation_statistics
try:
    from slicer_lite.core.vtk_renderer import VtkSegmentationView, VtkUnavailable
except Exception:
    VtkSegmentationView = None
    VtkUnavailable = RuntimeError


class SlicerLiteApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Slicer Lite - 医学图像处理平台")
        self.geometry("1380x860")
        self.minsize(1120, 720)

        self.project_root = Path(__file__).resolve().parents[2]
        self.runtime_dir = self.project_root / "assets" / "_runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.volume: Volume | None = None
        self.mask: np.ndarray | None = None
        self.label_map: np.ndarray | None = None
        self.surface_cache: np.ndarray | None = None
        self.slice_index = tk.IntVar(value=0)
        self.coronal_index = tk.IntVar(value=0)
        self.sagittal_index = tk.IntVar(value=0)
        self.window_center = tk.DoubleVar(value=40.0)
        self.window_width = tk.DoubleVar(value=400.0)
        self.lower_threshold = tk.DoubleVar(value=-120.0)
        self.upper_threshold = tk.DoubleVar(value=220.0)
        self.region_tolerance = tk.DoubleVar(value=80.0)
        self.vtk_opacity = tk.DoubleVar(value=0.86)
        self.seg_mode = tk.StringVar(value="soft_tissue")
        self.enhance_mode = tk.StringVar(value="none")
        self.interaction_mode = tk.StringVar(value="seed")
        self.active_label = tk.IntVar(value=1)
        self.brush_radius = tk.DoubleVar(value=8.0)
        self.keep_largest = tk.BooleanVar(value=False)
        self.fill_holes = tk.BooleanVar(value=False)
        self.show_mask = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="就绪。请打开 DICOM 文件夹开始。")
        self.info_text = tk.StringVar(value="未加载影像。")
        self.photo: tk.PhotoImage | None = None
        self.mpr_photos: dict[str, tk.PhotoImage] = {}
        self.vtk_view = None
        self.vtk_enabled = False
        self.vtk_mask_rendered = False
        self.seed_point: tuple[int, int, int] | None = None
        self.foreground_seeds: list[tuple[int, int, int]] = []
        self.background_seeds: list[tuple[int, int, int]] = []
        self.measure_points: list[tuple[int, int, int]] = []
        self.segmentation_source = "无"
        self.measure_text = tk.StringVar(value="测量：无")
        self.label_names = {1: "器官", 2: "病灶", 3: "骨骼", 4: "其他"}
        self.label_colors = {
            1: (255, 86, 74),
            2: (84, 190, 255),
            3: (255, 202, 40),
            4: (118, 221, 118),
        }
        self.event_log: list[str] = []
        self._display_origin = (0, 0)
        self._display_size = (1, 1)
        self._painting = False
        self._paint_dirty = False
        self._last_paint_point: tuple[int, int, int] | None = None
        self._pending_3d_job: str | None = None
        self._postprocess_running = False

        self._build_style()
        self._build_menu()
        self._build_layout()

    def _build_style(self) -> None:
        self.configure(bg="#f7f7f7")
        self.option_add("*Font", "{Microsoft YaHei UI} 10")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f7f7f7")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f7f7f7", foreground="#212121")
        style.configure("Panel.TLabel", background="#ffffff", foreground="#4b4b4b")
        style.configure("Header.TLabel", background="#ffffff", foreground="#111111", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("TButton", padding=(10, 5), background="#ffffff", foreground="#111111", bordercolor="#cfcfcf", lightcolor="#ffffff", darkcolor="#cfcfcf", borderwidth=1, relief=tk.SOLID)
        style.configure("Tool.TButton", padding=(12, 6), background="#ffffff", foreground="#111111", bordercolor="#cfcfcf", lightcolor="#ffffff", darkcolor="#cfcfcf", borderwidth=1, relief=tk.SOLID)
        style.configure("Primary.Tool.TButton", padding=(12, 6), background="#111111", foreground="#ffffff", bordercolor="#111111", lightcolor="#111111", darkcolor="#111111", borderwidth=1, relief=tk.SOLID)
        style.map("TButton", background=[("active", "#f2f2f2"), ("pressed", "#e8e8e8")], bordercolor=[("active", "#9e9e9e"), ("pressed", "#777777")], foreground=[("disabled", "#9a9a9a")])
        style.map("Tool.TButton", background=[("active", "#f2f2f2"), ("pressed", "#e8e8e8")], bordercolor=[("active", "#9e9e9e"), ("pressed", "#777777")])
        style.map("Primary.Tool.TButton", background=[("active", "#2b2b2b"), ("pressed", "#000000")], bordercolor=[("active", "#000000"), ("pressed", "#000000")], foreground=[("active", "#ffffff"), ("pressed", "#ffffff")])
        style.configure("TCheckbutton", background="#ffffff", foreground="#212121")
        self._build_checkbutton_style(style)
        style.configure("TRadiobutton", background="#ffffff", foreground="#212121")
        style.configure("TNotebook", background="#ffffff", borderwidth=0)
        style.configure("TNotebook.Tab", background="#f5f5f5", foreground="#4b4b4b", padding=(12, 5))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#111111")])
        style.configure("Horizontal.TScale", background="#ffffff", troughcolor="#eeeeee")

    def _build_checkbutton_style(self, style: ttk.Style) -> None:
        unchecked = tk.PhotoImage(width=14, height=14)
        checked = tk.PhotoImage(width=14, height=14)
        for image in (unchecked, checked):
            image.put("#ffffff", to=(0, 0, 14, 14))
            image.put("#bdbdbd", to=(0, 0, 14, 1))
            image.put("#bdbdbd", to=(0, 13, 14, 14))
            image.put("#bdbdbd", to=(0, 0, 1, 14))
            image.put("#bdbdbd", to=(13, 0, 14, 14))
        for x, y in [(3, 7), (4, 8), (5, 9), (6, 8), (7, 7), (8, 6), (9, 5), (10, 4), (11, 3)]:
            checked.put("#111111", to=(x, y, x + 2, y + 2))
        self._checkbutton_images = (unchecked, checked)
        try:
            style.element_create("Checkmark.indicator", "image", unchecked, ("selected", checked), border=0, sticky="w")
            style.layout(
                "Checkmark.TCheckbutton",
                [
                    (
                        "Checkbutton.padding",
                        {
                            "sticky": "nswe",
                            "children": [
                                ("Checkmark.indicator", {"side": "left", "sticky": ""}),
                                ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [("Checkbutton.label", {"sticky": "nswe"})]}),
                            ],
                        },
                    )
                ],
            )
            style.configure("Checkmark.TCheckbutton", background="#ffffff", foreground="#212121", padding=(0, 2))
        except tk.TclError:
            style.configure("Checkmark.TCheckbutton", background="#ffffff", foreground="#212121")

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="打开 DICOM 文件夹...", command=self.open_dicom_folder)
        file_menu.add_command(label="打开普通图像...", command=self.open_image_file)
        file_menu.add_command(label="导入 DICOM SEG...", command=self.import_dicom_seg)
        file_menu.add_command(label="导出当前视图...", command=self.export_current_view)
        file_menu.add_command(label="导出报告素材包...", command=self.export_report_pack)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.destroy)
        menu.add_cascade(label="文件", menu=file_menu)

        module_menu = tk.Menu(menu, tearoff=False)
        module_menu.add_command(label="图像增强", command=lambda: self.module_tabs.select(1))
        module_menu.add_command(label="图像分割", command=lambda: self.module_tabs.select(2))
        module_menu.add_command(label="三维重建", command=lambda: self.module_tabs.select(3))
        menu.add_cascade(label="模块", menu=module_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="操作说明", command=self.show_operation_guide)
        help_menu.add_command(label="关于", command=self.show_about)
        menu.add_cascade(label="帮助", menu=help_menu)
        self.config(menu=menu)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self, style="Panel.TFrame", padding=(12, 10))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="打开DICOM", style="Primary.Tool.TButton", command=self.open_dicom_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="打开图像", style="Tool.TButton", command=self.open_image_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="导入SEG", style="Tool.TButton", command=self.import_dicom_seg).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="自动窗宽", style="Tool.TButton", command=self.apply_auto_window).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="运行分割", style="Primary.Tool.TButton", command=self.run_segmentation).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="更新3D", style="Tool.TButton", command=self.update_3d_view).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="导出素材包", style="Tool.TButton", command=self.export_report_pack).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(toolbar, text="显示Mask", style="Checkmark.TCheckbutton", variable=self.show_mask, command=self.update_view).pack(side=tk.LEFT, padx=12)
        self.status_label = tk.Label(
            toolbar,
            textvariable=self.status,
            width=44,
            anchor=tk.E,
            bg="#ffffff",
            fg="#4b4b4b",
            font=("Microsoft YaHei UI", 10),
        )
        self.status_label.pack(side=tk.RIGHT, padx=(12, 0))

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 12))

        left = ttk.Frame(main, style="Panel.TFrame", padding=12)
        center = ttk.Frame(main, padding=(12, 0, 12, 0))
        right = ttk.Frame(main, style="Panel.TFrame", padding=12)
        main.add(left, weight=0)
        main.add(center, weight=1)
        main.add(right, weight=0)

        self._build_left_panel(left)
        self._build_center_panel(center)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="模块", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 12))
        self.module_tabs = ttk.Notebook(parent, width=300, height=580)
        self.module_tabs.pack(fill=tk.BOTH, expand=True)

        load_tab = self._scrollable_module_tab("数据")
        enhance_tab = self._scrollable_module_tab("增强")
        segment_tab = self._scrollable_module_tab("分割")
        recon_tab = self._scrollable_module_tab("3D")
        measure_tab = self._scrollable_module_tab("测量")

        ttk.Label(load_tab, text="DICOM 影像", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Button(load_tab, text="打开文件夹", command=self.open_dicom_folder).pack(fill=tk.X, pady=8)
        ttk.Button(load_tab, text="打开 PNG/JPG/BMP", command=self.open_image_file).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(load_tab, textvariable=self.info_text, style="Panel.TLabel", justify=tk.LEFT, wraplength=245).pack(anchor=tk.W, pady=6)

        ttk.Label(enhance_tab, text="窗宽/窗位", style="Header.TLabel").pack(anchor=tk.W)
        self._slider(enhance_tab, "窗位", self.window_center, -1000, 1000, self.update_view)
        self._slider(enhance_tab, "窗宽", self.window_width, 1, 2000, self.update_view)
        ttk.Label(enhance_tab, text="CT 预设", style="Header.TLabel").pack(anchor=tk.W, pady=(16, 4))
        for text, center, width in [
            ("软组织", 40, 400),
            ("骨窗", 300, 1500),
            ("肺窗", -600, 1500),
            ("脑窗", 40, 80),
            ("腹部", 60, 350),
        ]:
            ttk.Button(enhance_tab, text=text, command=lambda c=center, w=width, t=text: self.apply_window_preset(t, c, w)).pack(fill=tk.X, pady=2)
        ttk.Label(enhance_tab, text="图像增强", style="Header.TLabel").pack(anchor=tk.W, pady=(16, 4))
        for text, value in [("原始图像", "none"), ("直方图均衡", "histogram"), ("降噪", "denoise"), ("锐化", "sharpen")]:
            ttk.Radiobutton(enhance_tab, text=text, variable=self.enhance_mode, value=value, command=self.update_view).pack(anchor=tk.W)

        ttk.Label(segment_tab, text="分割模式", style="Header.TLabel").pack(anchor=tk.W)
        for text, value in [
            ("软组织候选", "soft_tissue"),
            ("骨阈值", "bone"),
            ("体表区域", "body"),
            ("Otsu 自动阈值", "otsu"),
            ("种子区域生长", "region_grow"),
            ("自定义 HU 范围", "custom"),
        ]:
            ttk.Radiobutton(segment_tab, text=text, variable=self.seg_mode, value=value).pack(anchor=tk.W)
        ttk.Label(segment_tab, text="自定义 HU 范围", style="Header.TLabel").pack(anchor=tk.W, pady=(16, 4))
        self._slider(segment_tab, "下限", self.lower_threshold, -1000, 1000, None)
        self._slider(segment_tab, "上限", self.upper_threshold, -1000, 2000, None)
        ttk.Label(segment_tab, text="区域生长", style="Header.TLabel").pack(anchor=tk.W, pady=(16, 4))
        self._slider(segment_tab, "容差", self.region_tolerance, 1, 300, None)
        self.seed_label = ttk.Label(segment_tab, text="种子：点击轴位图像", style="Panel.TLabel")
        self.seed_label.pack(anchor=tk.W, pady=4)
        ttk.Label(segment_tab, text="左键添加前景种子；右键添加背景种子。", style="Panel.TLabel", wraplength=245).pack(anchor=tk.W)
        ttk.Button(segment_tab, text="清除种子", command=self.clear_seed).pack(fill=tk.X, pady=4)
        ttk.Button(segment_tab, text="运行分割", command=self.run_segmentation).pack(fill=tk.X, pady=12)
        ttk.Button(segment_tab, text="导入 DICOM SEG", command=self.import_dicom_seg).pack(fill=tk.X, pady=4)
        ttk.Label(segment_tab, text="后处理", style="Header.TLabel").pack(anchor=tk.W, pady=(16, 4))
        ttk.Checkbutton(segment_tab, text="保留每层最大连通域", style="Checkmark.TCheckbutton", variable=self.keep_largest).pack(anchor=tk.W)
        ttk.Checkbutton(segment_tab, text="填充孔洞", style="Checkmark.TCheckbutton", variable=self.fill_holes).pack(anchor=tk.W)
        ttk.Button(segment_tab, text="应用后处理", command=self.apply_postprocess).pack(fill=tk.X, pady=6)
        ttk.Label(segment_tab, text="标签编辑", style="Header.TLabel").pack(anchor=tk.W, pady=(16, 4))
        for label, name in self.label_names.items():
            ttk.Radiobutton(segment_tab, text=f"{label}: {name}", variable=self.active_label, value=label).pack(anchor=tk.W)
        self._slider(segment_tab, "画笔", self.brush_radius, 1, 40, None)
        ttk.Radiobutton(segment_tab, text="画笔添加当前标签（拖动）", variable=self.interaction_mode, value="brush").pack(anchor=tk.W, pady=(4, 0))
        ttk.Radiobutton(segment_tab, text="橡皮擦删除标签（拖动）", variable=self.interaction_mode, value="eraser").pack(anchor=tk.W)
        ttk.Button(segment_tab, text="清空当前标签", command=self.clear_active_label).pack(fill=tk.X, pady=4)

        ttk.Label(recon_tab, text="三维重建", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            recon_tab,
            text="VTK 根据分割 Mask 生成 Marching Cubes 表面。可在 3D 视图中拖动旋转、滚轮缩放；彩色平面表示轴位、冠状位和矢状位切片。",
            style="Panel.TLabel",
            wraplength=245,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=6)
        self._slider(recon_tab, "透明度", self.vtk_opacity, 0.1, 1.0, self.update_vtk_opacity)
        ttk.Button(recon_tab, text="更新 3D 视图", command=self.update_3d_view).pack(fill=tk.X, pady=8)
        ttk.Button(recon_tab, text="导出 STL 模型", command=self.export_stl_model).pack(fill=tk.X, pady=4)

        ttk.Label(measure_tab, text="测量", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Radiobutton(measure_tab, text="种子/分割点击模式", variable=self.interaction_mode, value="seed").pack(anchor=tk.W, pady=(6, 2))
        ttk.Radiobutton(measure_tab, text="探查 HU 值", variable=self.interaction_mode, value="probe").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(measure_tab, text="测量 2D 距离", variable=self.interaction_mode, value="distance").pack(anchor=tk.W, pady=2)
        ttk.Label(measure_tab, textvariable=self.measure_text, style="Panel.TLabel", wraplength=245, justify=tk.LEFT).pack(anchor=tk.W, pady=12)
        ttk.Button(measure_tab, text="清除测量", command=self.clear_measurements).pack(fill=tk.X)

    def _scrollable_module_tab(self, title: str) -> ttk.Frame:
        page = ttk.Frame(self.module_tabs, style="Panel.TFrame")
        canvas = tk.Canvas(page, bg="#ffffff", highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(page, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas, style="Panel.TFrame", padding=10)
        window_id = canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.module_tabs.add(page, text=title)

        def refresh_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def fit_content_width(event) -> None:
            canvas.itemconfigure(window_id, width=max(int(event.width), 1))

        def on_wheel(event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        content.bind("<Configure>", refresh_scroll_region)
        canvas.bind("<Configure>", fit_content_width)
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", on_wheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        return content

    def _build_center_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="多平面切片视图", font=("Microsoft YaHei UI", 13, "bold"), foreground="#111111").pack(anchor=tk.W)
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.BOTH, expand=True, pady=(8, 6))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=3, minsize=330)
        grid.rowconfigure(1, weight=2, minsize=270)

        axial_panel = self._build_mpr_panel(grid, "轴位", "axial")
        coronal_panel = self._build_mpr_panel(grid, "冠状位", "coronal")
        sagittal_panel = self._build_mpr_panel(grid, "矢状位", "sagittal")
        axial_panel.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        coronal_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        sagittal_panel.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

    def _build_mpr_panel(self, parent: ttk.Frame, title: str, key: str) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        ttk.Label(panel, text=title, style="Panel.TLabel", font=("Microsoft YaHei UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        canvas = tk.Canvas(panel, bg="#f5f5f5", highlightthickness=1, highlightbackground="#dedede")
        canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 6))
        canvas.bind("<Configure>", lambda _event: self.update_view())
        canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        if key == "axial":
            canvas.bind("<Button-1>", self._on_image_click)
            canvas.bind("<Motion>", self._on_image_motion)
            canvas.bind("<B1-Motion>", self._on_image_drag)
            canvas.bind("<ButtonRelease-1>", self._on_image_release)
            canvas.bind("<Button-3>", self._on_background_seed_click)
            self.image_canvas = canvas
            var = self.slice_index
        elif key == "coronal":
            self.coronal_canvas = canvas
            var = self.coronal_index
        else:
            self.sagittal_canvas = canvas
            var = self.sagittal_index
        nav = ttk.Frame(panel, style="Panel.TFrame")
        nav.grid(row=2, column=0, sticky="ew")
        nav.columnconfigure(1, weight=1)
        ttk.Label(nav, text="切片", style="Panel.TLabel").pack(side=tk.LEFT)
        scale = ttk.Scale(nav, from_=0, to=0, orient=tk.HORIZONTAL, variable=var, command=lambda _v: self.update_view())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        label = ttk.Label(nav, text="0 / 0", style="Panel.TLabel")
        label.pack(side=tk.RIGHT)
        if key == "axial":
            self.slice_scale = scale
            self.slice_label = label
        elif key == "coronal":
            self.coronal_scale = scale
            self.coronal_label = label
        else:
            self.sagittal_scale = scale
            self.sagittal_label = label
        return panel

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="3D 视图", style="Header.TLabel").pack(anchor=tk.W)
        if VtkSegmentationView is not None:
            try:
                self.vtk_view = VtkSegmentationView(parent, width=360, height=360)
                self.vtk_view.pack(fill=tk.X, pady=(8, 14))
                self.vtk_enabled = True
                self.vtk_view.reset_scene("运行分割后显示 VTK 三维重建")
            except Exception as exc:
                self.vtk_enabled = False
                self.vtk_view = None
                error_log = self.runtime_dir / "vtk_init_error.log"
                error_log.write_text(traceback.format_exc(), encoding="utf-8")
                self.status.set(f"VTK 不可用，已切换到备用 3D 视图：{exc}")
        if not self.vtk_enabled:
            self.view3d = tk.Canvas(parent, width=330, height=330, bg="#f5f5f5", highlightthickness=1, highlightbackground="#dedede")
            self.view3d.pack(fill=tk.X, pady=(8, 14))
            self.view3d.bind("<Configure>", lambda _event: self.update_3d_view())

        ttk.Label(parent, text="影像信息", style="Header.TLabel").pack(anchor=tk.W)
        self.info_box = tk.Text(parent, width=38, height=16, bg="#fafafa", fg="#212121", insertbackground="#212121", relief=tk.FLAT, wrap=tk.WORD, padx=10, pady=10)
        self.info_box.pack(fill=tk.BOTH, expand=True, pady=8)
        self.info_box.insert("1.0", "未加载 DICOM 元数据。")
        self.info_box.configure(state=tk.DISABLED)
        ttk.Button(parent, text="导出报告素材包", command=self.export_report_pack).pack(fill=tk.X, pady=(0, 4))

    def _slider(self, parent: ttk.Frame, label: str, variable: tk.DoubleVar, lo: float, hi: float, command) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=label, style="Panel.TLabel", width=8).pack(side=tk.LEFT)
        scale = ttk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=variable, command=lambda _v: command() if command else None)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(row, textvariable=variable, style="Panel.TLabel", width=8).pack(side=tk.RIGHT)

    def open_dicom_folder(self) -> None:
        initial = self.project_root.parent
        folder = filedialog.askdirectory(title="打开 DICOM 文件夹", initialdir=str(initial))
        if not folder:
            return
        self.status.set("正在加载 DICOM 文件夹...")
        self._log(f"打开 DICOM 文件夹：{folder}")
        threading.Thread(target=self._load_volume_worker, args=(folder,), daemon=True).start()

    def open_image_file(self) -> None:
        path = filedialog.askopenfilename(
            title="打开普通图像",
            initialdir=str(self.project_root.parent),
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.status.set("普通图像加载中，请稍候...")
        self.update_idletasks()
        try:
            from PIL import Image
        except Exception as exc:
            self.status.set("普通图像加载失败。")
            messagebox.showerror("图像加载失败", f"Pillow 不可用：{exc}")
            return
        try:
            image = Image.open(path).convert("L")
            array = np.asarray(image, dtype=np.float32)
        except Exception as exc:
            self.status.set("普通图像加载失败。")
            messagebox.showerror("图像加载失败", str(exc))
            return
        volume = Volume(
            data=array[np.newaxis, :, :],
            spacing=(1.0, 1.0, 1.0),
            slices=[],
            metadata={
                "PatientName": "普通图像",
                "PatientID": Path(path).stem,
                "Modality": "IMAGE",
                "SeriesDescription": Path(path).name,
                "SourcePath": str(path),
            },
        )
        self._log(f"打开普通图像：{path}, shape={array.shape}")
        self._set_volume(volume)

    def _load_volume_worker(self, folder: str) -> None:
        try:
            volume = load_dicom_folder(folder)
        except DicomReadError as exc:
            self.after(0, lambda: messagebox.showerror("DICOM 加载失败", str(exc)))
            self.after(0, lambda: self.status.set("DICOM 加载失败。"))
            return
        self.after(0, lambda: self._set_volume(volume))

    def _set_volume(self, volume: Volume) -> None:
        self.volume = volume
        self.mask = None
        self.label_map = None
        self.surface_cache = None
        self.seed_point = None
        self.foreground_seeds.clear()
        self.background_seeds.clear()
        self.segmentation_source = "无"
        self.vtk_mask_rendered = False
        self._update_seed_label()
        self.slice_index.set(volume.data.shape[0] // 2)
        self.coronal_index.set(volume.data.shape[1] // 2)
        self.sagittal_index.set(volume.data.shape[2] // 2)
        self.slice_scale.configure(to=max(volume.data.shape[0] - 1, 0))
        self.coronal_scale.configure(to=max(volume.data.shape[1] - 1, 0))
        self.sagittal_scale.configure(to=max(volume.data.shape[2] - 1, 0))
        center, width = auto_window(volume.data)
        self.window_center.set(round(center, 1))
        self.window_width.set(round(width, 1))
        self._update_info()
        modality = str(volume.metadata.get("Modality", "DICOM"))
        source_text = f"{len(volume.slices)} 张 DICOM 切片" if volume.slices else "1 张图像切片"
        self.status.set(f"已加载 {source_text}。")
        self._log(f"已加载 {modality} 体数据：shape={volume.data.shape}, spacing={volume.spacing}")
        self.update_view()

    def apply_auto_window(self) -> None:
        if self.volume is None:
            return
        center, width = auto_window(self.volume.data)
        self.window_center.set(round(center, 1))
        self.window_width.set(round(width, 1))
        self.update_view()
        self._update_info()

    def apply_window_preset(self, name: str, center: float, width: float) -> None:
        self.window_center.set(float(center))
        self.window_width.set(float(width))
        self._log(f"应用 CT 窗宽窗位预设：{name}, WL={center}, WW={width}")
        self.status.set(f"窗宽窗位预设：{name}")
        self.update_view()
        self._update_info()

    def run_segmentation(self) -> None:
        if self.volume is None:
            messagebox.showinfo("未加载影像", "请先打开 DICOM 文件夹。")
            return
        if self.seg_mode.get() == "region_grow" and not self.foreground_seeds:
            messagebox.showinfo("需要种子点", "区域生长前，请先在轴位图像上左键点击至少一个前景种子点。")
            return
        self.status.set("分割处理中，请稍候...")
        self.update_idletasks()
        self.mask = segment_volume(
            self.volume.data,
            self.seg_mode.get(),
            self.lower_threshold.get(),
            self.upper_threshold.get(),
            seed=self.seed_point,
            foreground_seeds=self.foreground_seeds,
            background_seeds=self.background_seeds,
            tolerance=self.region_tolerance.get(),
        )
        if self.keep_largest.get() or self.fill_holes.get():
            self.mask = postprocess_mask(self.mask, self.keep_largest.get(), self.fill_holes.get())
        self._set_label_from_mask(self.mask, self.active_label.get())
        self._jump_to_best_mask_slice()
        self.surface_cache = None
        self.vtk_mask_rendered = False
        voxels = int(self.mask.sum())
        self.segmentation_source = self._segmentation_mode_name()
        self.status.set(f"分割完成：{voxels:,} 个体素。正在更新 3D...")
        self._log(f"分割：method={self.segmentation_source}, voxels={voxels}")
        self.update_view()
        self._update_info()
        self._schedule_3d_update()

    def apply_postprocess(self) -> None:
        if self.mask is None:
            messagebox.showinfo("无 Mask", "请先运行分割或导入分割结果。")
            return
        if self._postprocess_running:
            self.status.set("后处理仍在运行，请稍候...")
            return
        self.status.set("后处理加载中，请稍候...")
        self.update_idletasks()
        self._postprocess_running = True
        mask = self.mask.copy()
        keep_largest = self.keep_largest.get()
        fill_holes = self.fill_holes.get()
        label = self.active_label.get()
        source = self.segmentation_source
        threading.Thread(
            target=self._postprocess_worker,
            args=(mask, keep_largest, fill_holes, label, source),
            daemon=True,
        ).start()

    def _postprocess_worker(self, mask: np.ndarray, keep_largest: bool, fill_holes: bool, label: int, source: str) -> None:
        before = int(mask.sum())
        try:
            result = postprocess_mask(mask, keep_largest, fill_holes)
        except Exception:
            error = traceback.format_exc()
            self.after(0, lambda: self._finish_postprocess_error(error))
            return
        after = int(result.sum())
        self.after(0, lambda: self._finish_postprocess(result, before, after, keep_largest, fill_holes, label, source))

    def _finish_postprocess_error(self, error: str) -> None:
        self._postprocess_running = False
        self.status.set("后处理失败。")
        self._log(f"Mask 后处理失败：{error}")
        messagebox.showerror("后处理失败", error)

    def _finish_postprocess(
        self,
        result: np.ndarray,
        before: int,
        after: int,
        keep_largest: bool,
        fill_holes: bool,
        label: int,
        source: str,
    ) -> None:
        self._postprocess_running = False
        self.mask = result
        self._set_label_from_mask(self.mask, label)
        after = int(self.mask.sum())
        self.surface_cache = None
        self.vtk_mask_rendered = False
        self.segmentation_source = f"{source} + 后处理"
        self.status.set(f"后处理完成：{before:,} -> {after:,} 个体素。正在更新 3D...")
        self._log(f"Mask 后处理：keep_largest={keep_largest}, fill_holes={fill_holes}, voxels={before}->{after}")
        self.update_view()
        self._update_info()
        self._schedule_3d_update()

    def _set_label_from_mask(self, mask: np.ndarray, label: int) -> None:
        if self.volume is None:
            return
        self.label_map = np.zeros(self.volume.data.shape, dtype=np.uint8)
        self.label_map[mask.astype(bool)] = int(label)
        self.mask = self.label_map > 0

    def _schedule_3d_update(self) -> None:
        if self._pending_3d_job is not None:
            try:
                self.after_cancel(self._pending_3d_job)
            except Exception:
                pass
        self._pending_3d_job = self.after(180, self._run_scheduled_3d_update)

    def _run_scheduled_3d_update(self) -> None:
        self._pending_3d_job = None
        self.update_3d_view()

    def _refresh_mask_from_labels(self) -> None:
        if self.label_map is None:
            self.mask = None
            self.surface_cache = None
            return
        self.mask = self.label_map > 0
        self.surface_cache = None
        self.vtk_mask_rendered = False

    def clear_active_label(self) -> None:
        if self.label_map is None:
            return
        self.status.set("标签清空处理中，请稍候...")
        self.update_idletasks()
        label = int(self.active_label.get())
        before = int(np.count_nonzero(self.label_map == label))
        self.label_map[self.label_map == label] = 0
        self._refresh_mask_from_labels()
        after = int(np.count_nonzero(self.label_map == label))
        self.status.set(f"已清空标签 {label}：{before:,} -> {after:,} 个体素。")
        self._log(f"清空当前标签：label={label}, voxels={before}->{after}")
        self.update_view()
        self._update_info()

    def _segmentation_mode_name(self) -> str:
        names = {
            "soft_tissue": "软组织候选",
            "bone": "骨阈值",
            "body": "体表区域",
            "otsu": "Otsu 自动阈值",
            "region_grow": "种子区域生长",
            "custom": "自定义 HU 范围",
        }
        return names.get(self.seg_mode.get(), self.seg_mode.get())

    def update_view(self) -> None:
        if self.volume is None or not hasattr(self, "image_canvas"):
            return
        zmax = self.volume.data.shape[0] - 1
        ymax = self.volume.data.shape[1] - 1
        xmax = self.volume.data.shape[2] - 1
        z = int(round(self.slice_index.get()))
        z = max(0, min(z, zmax))
        y = max(0, min(int(round(self.coronal_index.get())), ymax))
        x = max(0, min(int(round(self.sagittal_index.get())), xmax))
        self.slice_index.set(z)
        self.coronal_index.set(y)
        self.sagittal_index.set(x)

        self._render_mpr_canvas(
            "axial",
            self.image_canvas,
            self.volume.data[z],
            self.mask[z] if self.mask is not None and self.show_mask.get() else None,
            self.label_map[z] if self.label_map is not None and self.show_mask.get() else None,
            f"轴位 {z + 1}/{zmax + 1}",
        )
        self._draw_seed_markers(z)
        self.slice_label.configure(text=f"{z + 1} / {zmax + 1}")
        self._render_mpr_canvas(
            "coronal",
            self.coronal_canvas,
            np.flipud(self.volume.data[:, y, :]),
            np.flipud(self.mask[:, y, :]) if self.mask is not None and self.show_mask.get() else None,
            np.flipud(self.label_map[:, y, :]) if self.label_map is not None and self.show_mask.get() else None,
            f"冠状位 {y + 1}/{ymax + 1}",
        )
        self.coronal_label.configure(text=f"{y + 1} / {ymax + 1}")
        self._render_mpr_canvas(
            "sagittal",
            self.sagittal_canvas,
            np.flipud(self.volume.data[:, :, x]),
            np.flipud(self.mask[:, :, x]) if self.mask is not None and self.show_mask.get() else None,
            np.flipud(self.label_map[:, :, x]) if self.label_map is not None and self.show_mask.get() else None,
            f"矢状位 {x + 1}/{xmax + 1}",
        )
        self.sagittal_label.configure(text=f"{x + 1} / {xmax + 1}")

    def _render_mpr_canvas(
        self,
        key: str,
        canvas: tk.Canvas,
        image: np.ndarray,
        mask: np.ndarray | None,
        labels: np.ndarray | None,
        title: str,
    ) -> None:
        gray = window_to_uint8(image, self.window_center.get(), self.window_width.get())
        gray = enhance_uint8(gray, self.enhance_mode.get())
        if labels is not None:
            rgb = compose_label_overlay(gray, labels, self.label_colors)
        else:
            edge = mask_edges(mask) if mask is not None else None
            rgb = compose_overlay(gray, edge)
        canvas_w = max(canvas.winfo_width() - 24, 240)
        canvas_h = max(canvas.winfo_height() - 24, 180)
        rgb = resize_nearest(rgb, canvas_w, canvas_h)
        ppm = save_ppm(rgb, self.runtime_dir / f"{key}_view.ppm")
        photo = tk.PhotoImage(file=str(ppm))
        self.mpr_photos[key] = photo
        canvas.delete("all")
        image_x = canvas_w // 2 + 12
        image_y = canvas_h // 2 + 12
        canvas.create_image(image_x, image_y, image=photo, anchor=tk.CENTER)
        canvas.create_text(16, 16, anchor=tk.NW, fill="#212121", text=title)
        if key == "axial":
            self.photo = photo
            self._display_size = (rgb.shape[1], rgb.shape[0])
            self._display_origin = (image_x - rgb.shape[1] // 2, image_y - rgb.shape[0] // 2)

    def update_3d_view(self) -> None:
        if self.vtk_enabled and self.vtk_view is not None:
            try:
                if self.volume is None:
                    self.vtk_view.reset_scene("请打开 DICOM 数据")
                    return
                if self.mask is None:
                    self.vtk_view.reset_scene("运行分割后显示 VTK 三维重建")
                    return
                if not np.any(self.mask):
                    self.vtk_view.reset_scene("当前没有可用于三维重建的分割体素")
                    self.vtk_mask_rendered = False
                    self.status.set("3D 已清空：分割 Mask 为空。")
                    return
                self.status.set("正在更新 VTK 3D 视图...")
                self.update_idletasks()
                if not self.vtk_mask_rendered:
                    self.vtk_view.show_mask(
                        self.mask,
                        self.volume.spacing,
                        int(round(self.slice_index.get())),
                        int(round(self.coronal_index.get())),
                        int(round(self.sagittal_index.get())),
                        labels=self.label_map,
                        colors=self.label_colors,
                    )
                    self.vtk_mask_rendered = True
                else:
                    self.vtk_view.update_slice_planes(
                        int(round(self.slice_index.get())),
                        int(round(self.coronal_index.get())),
                        int(round(self.sagittal_index.get())),
                    )
                self.status.set("VTK 3D 视图已更新。")
            except Exception as exc:
                self.vtk_enabled = False
                self.vtk_view = None
                error_log = self.runtime_dir / "vtk_runtime_error.log"
                error_log.write_text(traceback.format_exc(), encoding="utf-8")
                self.status.set(f"VTK 3D 更新失败，已启用备用视图：{exc}")
            return

        if not hasattr(self, "view3d"):
            return
        self.view3d.delete("all")
        w = max(self.view3d.winfo_width(), 320)
        h = max(self.view3d.winfo_height(), 300)
        self.view3d.create_rectangle(0, 0, w, h, fill="#f5f5f5", outline="")
        if self.volume is None:
            self.view3d.create_text(w // 2, h // 2, fill="#777777", text="请打开 DICOM 数据")
            return
        if self.surface_cache is None and self.mask is not None:
            self.surface_cache = surface_points(self.mask)
        if self.surface_cache is None or self.surface_cache.size == 0:
            self._draw_volume_box(w, h)
            self.view3d.create_text(w // 2, h - 28, fill="#777777", text="运行分割后重建三维表面")
        else:
            pts2d = project_points(self.surface_cache, self.volume.data.shape, w, h)
            for x, y in pts2d:
                if 0 <= x < w and 0 <= y < h:
                    self.view3d.create_rectangle(x, y, x + 1, y + 1, fill="#2563eb", outline="")
        self._draw_slice_plane(w, h)

    def update_vtk_opacity(self) -> None:
        if self.vtk_enabled and self.vtk_view is not None:
            try:
                self.vtk_view.set_model_opacity(self.vtk_opacity.get())
            except Exception:
                self.status.set("VTK 透明度更新失败；需要时请重新更新 3D 视图。")

    def export_stl_model(self) -> None:
        if not self.vtk_enabled or self.vtk_view is None:
            messagebox.showinfo("VTK 不可用", "导出 STL 需要 VTK 3D 渲染器。")
            return
        if self.mask is None:
            messagebox.showinfo("无模型", "请先运行或导入分割，并更新 3D 视图。")
            return
        self.update_3d_view()
        target = filedialog.asksaveasfilename(
            title="导出 STL 模型",
            defaultextension=".stl",
            filetypes=[("STL 模型", "*.stl"), ("所有文件", "*.*")],
            initialfile="slicer_lite_segmentation.stl",
        )
        if not target:
            return
        try:
            self.vtk_view.save_stl(target)
        except Exception as exc:
            messagebox.showerror("STL 导出失败", str(exc))
            return
        self.status.set(f"已导出 STL 模型：{target}")
        self._log(f"导出 STL 模型：{target}")

    def _draw_volume_box(self, w: int, h: int) -> None:
        margin = 58
        self.view3d.create_rectangle(margin, margin, w - margin, h - margin, outline="#bdbdbd")
        self.view3d.create_line(margin, margin, margin + 42, margin - 30, fill="#bdbdbd")
        self.view3d.create_line(w - margin, margin, w - margin + 42, margin - 30, fill="#bdbdbd")
        self.view3d.create_line(w - margin, h - margin, w - margin + 42, h - margin - 30, fill="#bdbdbd")
        self.view3d.create_rectangle(margin + 42, margin - 30, w - margin + 42, h - margin - 30, outline="#bdbdbd")

    def _draw_slice_plane(self, w: int, h: int) -> None:
        if self.volume is None:
            return
        zmax = max(self.volume.data.shape[0] - 1, 1)
        t = int(round(self.slice_index.get())) / zmax
        y = h * (0.76 - 0.48 * t)
        poly = [w * 0.22, y, w * 0.50, y - 42, w * 0.78, y, w * 0.50, y + 42]
        self.view3d.create_polygon(poly, outline="#2563eb", fill="", width=2)
        self.view3d.create_text(12, 14, anchor=tk.NW, fill="#212121", text="当前轴位切片平面")

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self.volume is None:
            return
        delta = -1 if event.delta < 0 else 1
        self.slice_index.set(self.slice_index.get() + delta)
        self.update_view()

    def _on_image_click(self, event: tk.Event) -> None:
        if self.interaction_mode.get() != "seed":
            self._handle_measure_click(event)
            return
        self._append_seed_from_event(event, foreground=True)

    def _on_image_motion(self, event: tk.Event) -> None:
        if self.interaction_mode.get() not in {"brush", "eraser"}:
            if hasattr(self, "image_canvas"):
                self.image_canvas.delete("brush_preview")
            return
        point = self._voxel_from_event(event)
        if point is None:
            self.image_canvas.delete("brush_preview")
            return
        self._draw_brush_preview(event.x, event.y)

    def _on_image_drag(self, event: tk.Event) -> None:
        if self.interaction_mode.get() not in {"brush", "eraser"}:
            return
        point = self._voxel_from_event(event)
        if point is None:
            return
        self._draw_brush_preview(event.x, event.y)
        erase = self.interaction_mode.get() == "eraser"
        if not self._painting:
            self._painting = True
            self._paint_dirty = False
            self._last_paint_point = point
            self._paint_label(point, erase=erase, refresh=False)
            return
        self._paint_stroke(self._last_paint_point, point, erase=erase)
        self._last_paint_point = point
        if not self._paint_dirty:
            self._paint_dirty = True
            self.after(25, self._refresh_paint_preview)

    def _on_image_release(self, event: tk.Event) -> None:
        if self.interaction_mode.get() not in {"brush", "eraser"}:
            return
        point = self._voxel_from_event(event)
        if point is not None:
            self._paint_stroke(self._last_paint_point, point, erase=self.interaction_mode.get() == "eraser")
        self._painting = False
        self._last_paint_point = None
        self._paint_dirty = False
        self.update_view()
        self._draw_brush_preview(event.x, event.y)
        self._update_info()
        self.status.set("标签已更新，正在更新 3D...")
        self._schedule_3d_update()

    def _on_background_seed_click(self, event: tk.Event) -> None:
        if self.interaction_mode.get() != "seed":
            return
        self._append_seed_from_event(event, foreground=False)

    def _voxel_from_event(self, event: tk.Event) -> tuple[int, int, int] | None:
        if self.volume is None:
            return None
        ox, oy = self._display_origin
        dw, dh = self._display_size
        if event.x < ox or event.y < oy or event.x >= ox + dw or event.y >= oy + dh:
            return None
        z = int(round(self.slice_index.get()))
        y = int((event.y - oy) * self.volume.data.shape[1] / max(dh, 1))
        x = int((event.x - ox) * self.volume.data.shape[2] / max(dw, 1))
        y = max(0, min(y, self.volume.data.shape[1] - 1))
        x = max(0, min(x, self.volume.data.shape[2] - 1))
        return z, y, x

    def _draw_brush_preview(self, canvas_x: int, canvas_y: int) -> None:
        if self.volume is None or not hasattr(self, "image_canvas"):
            return
        self.image_canvas.delete("brush_preview")
        radius = max(1, int(round(self.brush_radius.get())))
        sx = self._display_size[0] / max(self.volume.data.shape[2], 1)
        sy = self._display_size[1] / max(self.volume.data.shape[1], 1)
        screen_radius = max(4, int(round(radius * (sx + sy) * 0.5)))
        if self.interaction_mode.get() == "eraser":
            color = "#ffffff"
        else:
            color = "#ff564a"
            if self.active_label.get() in self.label_colors:
                rgb = self.label_colors[int(self.active_label.get())]
                color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        self.image_canvas.create_oval(
            canvas_x - screen_radius,
            canvas_y - screen_radius,
            canvas_x + screen_radius,
            canvas_y + screen_radius,
            outline=color,
            width=3,
            tags="brush_preview",
        )
        self.image_canvas.create_oval(
            canvas_x - 2,
            canvas_y - 2,
            canvas_x + 2,
            canvas_y + 2,
            fill=color,
            outline="",
            tags="brush_preview",
        )

    def _append_seed_from_event(self, event: tk.Event, foreground: bool) -> None:
        if self.volume is None:
            return
        seed = self._voxel_from_event(event)
        if seed is None:
            return
        z, y, x = seed
        self.seed_point = seed
        if foreground:
            self.foreground_seeds.append(seed)
        else:
            self.background_seeds.append(seed)
        self.seg_mode.set("region_grow")
        self._update_seed_label()
        value = float(self.volume.data[z, y, x])
        kind = "前景" if foreground else "背景"
        self.status.set(f"{kind}种子已设置：z={z + 1}, x={x}, y={y}, HU={value:.1f}")
        self._log(f"区域生长{kind}种子：z={z + 1}, x={x}, y={y}, HU={value:.1f}")
        self.update_view()

    def _handle_measure_click(self, event: tk.Event) -> None:
        if self.volume is None:
            return
        point = self._voxel_from_event(event)
        if point is None:
            return
        if self.interaction_mode.get() in {"brush", "eraser"}:
            self._painting = True
            self._paint_dirty = False
            self._last_paint_point = point
            self._paint_label(point, erase=self.interaction_mode.get() == "eraser")
            return
        z, y, x = point
        value = float(self.volume.data[z, y, x])
        if self.interaction_mode.get() == "probe":
            self.measure_points = [point]
            text = f"探查：z={z + 1}, x={x}, y={y}, HU={value:.1f}"
        else:
            if self.measure_points and self.measure_points[-1][0] != z:
                self.measure_points.clear()
            self.measure_points.append(point)
            self.measure_points = self.measure_points[-2:]
            if len(self.measure_points) == 2:
                p1, p2 = self.measure_points
                dy = (p2[1] - p1[1]) * self.volume.spacing[1]
                dx = (p2[2] - p1[2]) * self.volume.spacing[2]
                distance = float((dx * dx + dy * dy) ** 0.5)
                text = f"距离：{distance:.2f} mm\nP1 z={p1[0] + 1}, x={p1[2]}, y={p1[1]}\nP2 z={p2[0] + 1}, x={p2[2]}, y={p2[1]}"
            else:
                text = f"距离 P1：z={z + 1}, x={x}, y={y}, HU={value:.1f}"
        self.measure_text.set(text)
        self.status.set(text.replace("\n", " | "))
        self._log(f"测量：{text.replace(chr(10), ' | ')}")
        self.update_view()

    def _paint_stroke(
        self,
        start: tuple[int, int, int] | None,
        end: tuple[int, int, int],
        erase: bool = False,
    ) -> None:
        if start is None or start[0] != end[0]:
            self._paint_label(end, erase=erase, refresh=False)
            return
        _, y0, x0 = start
        z, y1, x1 = end
        steps = max(abs(y1 - y0), abs(x1 - x0), 1)
        radius = max(1, int(round(self.brush_radius.get())))
        step_gap = max(1, radius // 2)
        for i in range(0, steps + 1, step_gap):
            t = i / steps
            y = int(round(y0 + (y1 - y0) * t))
            x = int(round(x0 + (x1 - x0) * t))
            self._paint_label((z, y, x), erase=erase, refresh=False)
        self._paint_label(end, erase=erase, refresh=False)

    def _paint_label(self, point: tuple[int, int, int], erase: bool = False, refresh: bool = True) -> None:
        if self.volume is None:
            return
        if self.label_map is None:
            self.label_map = np.zeros(self.volume.data.shape, dtype=np.uint8)
        if self.mask is None:
            self.mask = np.zeros(self.volume.data.shape, dtype=bool)
        z, y, x = point
        radius = max(1, int(round(self.brush_radius.get())))
        y0 = max(0, y - radius)
        y1 = min(self.label_map.shape[1], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(self.label_map.shape[2], x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        brush = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
        if erase:
            self.label_map[z, y0:y1, x0:x1][brush] = 0
        else:
            self.label_map[z, y0:y1, x0:x1][brush] = int(self.active_label.get())
        self.mask[z, y0:y1, x0:x1] = self.label_map[z, y0:y1, x0:x1] > 0
        self.surface_cache = None
        self.vtk_mask_rendered = False
        action = "橡皮擦" if erase else f"绘制标签 {self.active_label.get()}"
        self.segmentation_source = "手动标签编辑"
        self.status.set(f"{action}：z={z + 1}, x={x}, y={y}, 半径={radius}")
        if refresh:
            self._log(f"{action}：z={z + 1}, x={x}, y={y}, radius={radius}")
            self.update_view()
            self._update_info()
            self.status.set("标签已更新，正在更新 3D...")
            self._schedule_3d_update()

    def _refresh_paint_preview(self) -> None:
        if not self._paint_dirty:
            return
        self._paint_dirty = False
        self.update_view()

    def clear_measurements(self) -> None:
        self.measure_points.clear()
        self.measure_text.set("测量：无")
        self.update_view()

    def clear_seed(self) -> None:
        self.seed_point = None
        self.foreground_seeds.clear()
        self.background_seeds.clear()
        self._update_seed_label()
        self.update_view()

    def _update_seed_label(self) -> None:
        if not hasattr(self, "seed_label"):
            return
        if self.seed_point is None:
            self.seed_label.configure(text="种子：无")
        else:
            z, y, x = self.seed_point
            self.seed_label.configure(text=f"种子：前景={len(self.foreground_seeds)}, 背景={len(self.background_seeds)} | 最近 z={z + 1}, x={x}, y={y}")

    def _draw_seed_markers(self, z: int) -> None:
        if self.volume is None:
            return
        for seed, color in [(s, "#00e5ff") for s in self.foreground_seeds] + [(s, "#ffca28") for s in self.background_seeds]:
            if seed[0] != z:
                continue
            sx = self._display_origin[0] + int(seed[2] * self._display_size[0] / self.volume.data.shape[2])
            sy = self._display_origin[1] + int(seed[1] * self._display_size[1] / self.volume.data.shape[1])
            self.image_canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, outline=color, width=2)
        self._draw_measure_markers(z)

    def _draw_measure_markers(self, z: int) -> None:
        if self.volume is None:
            return
        screen_points = []
        for point in self.measure_points:
            if point[0] != z:
                continue
            sx = self._display_origin[0] + int(point[2] * self._display_size[0] / self.volume.data.shape[2])
            sy = self._display_origin[1] + int(point[1] * self._display_size[1] / self.volume.data.shape[1])
            screen_points.append((sx, sy))
            self.image_canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, outline="#b388ff", width=2)
            self.image_canvas.create_line(sx - 7, sy, sx + 7, sy, fill="#b388ff", width=1)
            self.image_canvas.create_line(sx, sy - 7, sx, sy + 7, fill="#b388ff", width=1)
        if len(screen_points) == 2:
            self.image_canvas.create_line(*screen_points[0], *screen_points[1], fill="#b388ff", width=2)

    def import_dicom_seg(self) -> None:
        if self.volume is None:
            messagebox.showinfo("未加载影像", "请先打开与 SEG 匹配的 CT DICOM 文件夹。")
            return
        path = filedialog.askopenfilename(
            title="导入 DICOM SEG",
            initialdir=str(self.project_root.parent),
            filetypes=[("DICOM SEG", "*.dcm *.dicom *.ima"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self.status.set("DICOM SEG 加载中，请稍候...")
        self.update_idletasks()
        try:
            seg = load_dicom_seg(path, self.volume)
        except DicomSegReadError as exc:
            self.status.set("DICOM SEG 加载失败。")
            messagebox.showerror("SEG 导入失败", str(exc))
            return
        self._set_label_from_mask(seg.mask, self.active_label.get())
        self._jump_to_best_mask_slice()
        self.surface_cache = None
        self.vtk_mask_rendered = False
        self.show_mask.set(True)
        self.segmentation_source = f"DICOM SEG ({seg.label}, {seg.algorithm})"
        align_text = "已对齐" if seg.aligned else "居中适配"
        self.status.set(f"已导入 SEG：{seg.label}，帧数={seg.frames}，{align_text}。")
        self._log(f"导入 DICOM SEG：label={seg.label}, frames={seg.frames}, {align_text}")
        self.update_view()
        self._update_info()
        self._schedule_3d_update()

    def _jump_to_best_mask_slice(self) -> None:
        if self.mask is None or not self.mask.any():
            return
        counts = self.mask.reshape(self.mask.shape[0], -1).sum(axis=1)
        best = int(np.argmax(counts))
        self.slice_index.set(best)

    def _update_info(self) -> None:
        if self.volume is None:
            return
        meta = self.volume.metadata
        stats = segmentation_statistics(
            self.volume.data,
            self.mask,
            self.volume.spacing,
            int(round(self.slice_index.get())),
        )
        lines = [
            f"患者：{meta.get('PatientName', '未知')}",
            f"患者ID：{meta.get('PatientID', '未知')}",
            f"模态：{meta.get('Modality', '未知')}",
            f"序列：{meta.get('SeriesDescription', '未知')}",
            f"体数据：{self.volume.shape_text}",
            f"文件数：{len(self.volume.slices)}",
            f"像素间距：z={self.volume.spacing[0]:.3f}, y={self.volume.spacing[1]:.3f}, x={self.volume.spacing[2]:.3f} mm",
            f"窗宽窗位：WW={self.window_width.get():.1f}, WL={self.window_center.get():.1f}",
            f"HU 范围：{float(np.min(self.volume.data)):.1f} 到 {float(np.max(self.volume.data)):.1f}",
            "",
            f"分割来源：{self.segmentation_source}",
            f"分割体素：{int(stats['voxel_count']):,}",
            f"标签体素：{self._label_summary()}",
            f"分割体积：{float(stats['volume_cm3']):.3f} cm^3",
            f"当前层面积：{float(stats['slice_area_mm2']):.2f} mm^2",
            f"HU 均值/最小/最大：{float(stats['hu_mean']):.1f} / {float(stats['hu_min']):.1f} / {float(stats['hu_max']):.1f}",
        ]
        text = "\n".join(lines)
        self.info_text.set(text)
        self.info_box.configure(state=tk.NORMAL)
        self.info_box.delete("1.0", tk.END)
        self.info_box.insert("1.0", text)
        self.info_box.configure(state=tk.DISABLED)

    def export_current_view(self) -> None:
        if self.volume is None:
            return
        target = filedialog.asksaveasfilename(
            title="导出当前视图",
            defaultextension=".ppm",
            filetypes=[("PPM 图像", "*.ppm")],
            initialfile="slicer_lite_view.ppm",
        )
        if target:
            source = self.runtime_dir / "axial_view.ppm"
            Path(target).write_bytes(source.read_bytes())
            self.status.set(f"已导出：{target}")
            self._log(f"导出当前视图：{target}")

    def export_report_pack(self) -> None:
        if self.volume is None:
            messagebox.showinfo("未加载影像", "请先打开 DICOM 文件夹。")
            return
        out = self.project_root / "outputs" / "report_assets"
        out.mkdir(parents=True, exist_ok=True)
        self.status.set("报告素材包导出中，请稍候...")
        self.update_idletasks()
        self.update_view()
        self.update_3d_view()

        axial = self.runtime_dir / "axial_view.ppm"
        if axial.exists():
            (out / "02_axial_overlay.ppm").write_bytes(axial.read_bytes())
        self._export_mpr_images(out)
        self._export_best_slice(out)
        self._export_mask_sequence(out)

        if self.vtk_enabled and self.vtk_view is not None and self.mask is not None:
            try:
                self.vtk_view.save_screenshot(out / "05_vtk_3d_reconstruction.png")
                self.vtk_view.save_stl(out / "11_segmentation_surface.stl")
            except Exception as exc:
                (out / "05_vtk_export_error.txt").write_text(str(exc), encoding="utf-8")
        elif hasattr(self, "view3d"):
            self.view3d.postscript(file=str(out / "05_fallback_3d_view.ps"))

        z = int(round(self.slice_index.get()))
        stats = segmentation_statistics(self.volume.data, self.mask, self.volume.spacing, z)
        if self.mask is not None:
            np.save(out / "06_segmentation_mask.npy", self.mask.astype(np.uint8))
        if self.label_map is not None:
            np.save(out / "12_segmentation_labels.npy", self.label_map.astype(np.uint8))
        lines = [
            "Slicer Lite 报告素材汇总",
            f"学生：蔡静",
            f"学号：2023141670108",
            "",
            self.info_text.get(),
            "",
            format_statistics(stats),
            "",
            f"当前切片：{z + 1}/{self.volume.data.shape[0]}",
            f"分割来源：{self.segmentation_source}",
            f"种子点：{self.seed_point if self.seed_point else '无'}",
            f"前景种子：{self.foreground_seeds}",
            f"背景种子：{self.background_seeds}",
            f"测量：{self.measure_text.get()}",
            f"后处理：保留最大连通域={self.keep_largest.get()}, 填充孔洞={self.fill_holes.get()}",
            f"标签：{self._label_summary()}",
        ]
        (out / "07_statistics.txt").write_text("\n".join(lines), encoding="utf-8")
        (out / "08_processing_log.txt").write_text("\n".join(self.event_log), encoding="utf-8")
        self.status.set(f"报告素材包已导出：{out}")
        self._log(f"导出报告素材包：{out}")
        messagebox.showinfo("导出完成", f"报告素材已导出到：\n{out}")

    def _export_mpr_images(self, out: Path) -> None:
        if self.volume is None:
            return
        z = int(round(self.slice_index.get()))
        y = self.seed_point[1] if self.seed_point else self.volume.data.shape[1] // 2
        x = self.seed_point[2] if self.seed_point else self.volume.data.shape[2] // 2
        z = max(0, min(z, self.volume.data.shape[0] - 1))
        y = max(0, min(y, self.volume.data.shape[1] - 1))
        x = max(0, min(x, self.volume.data.shape[2] - 1))

        axial_gray = window_to_uint8(self.volume.data[z], self.window_center.get(), self.window_width.get())
        save_image(compose_overlay(axial_gray), out / "01_axial_original.png")

        coronal = self.volume.data[:, y, :]
        sagittal = self.volume.data[:, :, x]
        coronal_mask = self.mask[:, y, :] if self.mask is not None and self.show_mask.get() else None
        sagittal_mask = self.mask[:, :, x] if self.mask is not None and self.show_mask.get() else None
        coronal_labels = self.label_map[:, y, :] if self.label_map is not None and self.show_mask.get() else None
        sagittal_labels = self.label_map[:, :, x] if self.label_map is not None and self.show_mask.get() else None

        coronal_gray = np.flipud(window_to_uint8(coronal, self.window_center.get(), self.window_width.get()))
        sagittal_gray = np.flipud(window_to_uint8(sagittal, self.window_center.get(), self.window_width.get()))
        coronal_edge = np.flipud(mask_edges(coronal_mask)) if coronal_mask is not None else None
        sagittal_edge = np.flipud(mask_edges(sagittal_mask)) if sagittal_mask is not None else None

        coronal_rgb = compose_label_overlay(coronal_gray, np.flipud(coronal_labels), self.label_colors) if coronal_labels is not None else compose_overlay(coronal_gray, coronal_edge)
        sagittal_rgb = compose_label_overlay(sagittal_gray, np.flipud(sagittal_labels), self.label_colors) if sagittal_labels is not None else compose_overlay(sagittal_gray, sagittal_edge)
        save_image(resize_nearest(coronal_rgb, 768, 512), out / "03_coronal_overlay.png")
        save_image(resize_nearest(sagittal_rgb, 768, 512), out / "04_sagittal_overlay.png")

    def _export_best_slice(self, out: Path) -> None:
        if self.volume is None or self.mask is None or not self.mask.any():
            return
        counts = self.mask.reshape(self.mask.shape[0], -1).sum(axis=1)
        z = int(np.argmax(counts))
        gray = window_to_uint8(self.volume.data[z], self.window_center.get(), self.window_width.get())
        save_image(compose_overlay(gray, mask_edges(self.mask[z])), out / "09_best_slice_overlay.png")

    def _export_mask_sequence(self, out: Path, max_slices: int = 12) -> None:
        if self.mask is None or not self.mask.any():
            return
        mask_dir = out / "10_mask_sequence"
        mask_dir.mkdir(parents=True, exist_ok=True)
        counts = self.mask.reshape(self.mask.shape[0], -1).sum(axis=1)
        nonzero = np.flatnonzero(counts)
        if nonzero.size == 0:
            return
        indexes = np.linspace(nonzero[0], nonzero[-1], min(max_slices, nonzero.size)).astype(int)
        for z in indexes:
            mask_rgb = np.repeat((self.mask[z].astype(np.uint8) * 255)[..., None], 3, axis=2)
            save_image(mask_rgb, mask_dir / f"mask_z{z + 1:03d}.png")

    def _label_summary(self) -> str:
        if self.label_map is None:
            return "无"
        parts = []
        for label, name in self.label_names.items():
            count = int(np.count_nonzero(self.label_map == label))
            if count:
                parts.append(f"{label}:{name}={count:,}")
        return ", ".join(parts) if parts else "无"

    def _log(self, message: str) -> None:
        from datetime import datetime

        self.event_log.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}")

    def show_about(self) -> None:
        messagebox.showinfo(
            "关于 Slicer Lite",
            "Slicer Lite 是参考 3D Slicer 界面布局设计的医学图像处理课程项目。\n\n"
            "本软件实现了 DICOM 加载、多平面切片浏览、图像增强、分割叠加、DICOM SEG 导入、"
            "手动标签编辑、测量分析和基于 VTK 的三维重建。",
        )

    def show_operation_guide(self) -> None:
        messagebox.showinfo(
            "操作说明",
            "1. 点击“打开DICOM”，选择 PANCREAS_0001 数据文件夹。\n"
            "2. 在“增强”模块调整窗位和窗宽，也可以使用软组织、骨窗、肺窗等 CT 预设。\n"
            "3. 在“分割”模块选择阈值、Otsu 或种子区域生长方法，并点击“运行分割”。\n"
            "4. 需要修正结果时，可使用标签画笔和橡皮擦进行手动编辑。\n"
            "5. 可按需勾选“保留每层最大连通域”或“填充孔洞”，再点击“应用后处理”。\n"
            "6. 在“测量”模块可探查 HU 值或测量二维距离。\n"
            "7. 点击“更新3D”渲染 VTK 三维重建，必要时可导出 STL 模型。\n"
            "8. 点击“导出素材包”保存报告图片、Mask、统计结果、STL 和处理日志。",
        )


def main() -> None:
    app = SlicerLiteApp()
    app.mainloop()
