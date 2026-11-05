# Slicer Lite 医学图像处理平台

这是一个参考 3D Slicer 界面布局制作的轻量医学图像处理课程项目。项目使用 `tkinter + numpy + VTK` 实现，其中 VTK 用于三维分割表面重建和交互式三维显示。

## 功能

- 打开 DICOM 文件夹，读取未压缩 CT/MR 切片。
- 打开 PNG/JPG/BMP/TIF 普通图像，复用增强、分割和导出流程。
- 按轴状位、冠状位、矢状位三向 MPR 浏览医学图像。
- 支持窗宽窗位调节。
- 支持 Soft Tissue、Bone、Lung、Brain、Abdomen 等 CT 窗宽窗位预设。
- 支持直方图均衡化、去噪、锐化等图像增强。
- 支持软组织候选、骨阈值、体部掩膜、自定义 HU 范围分割。
- 支持 Otsu 自动阈值分割。
- 支持左键添加前景种子、右键添加背景种子，并执行多种子区域生长分割。
- 支持分割后处理：逐层保留最大连通域、填洞。
- 支持多标签分割管理：Organ、Lesion、Bone、Other。
- 支持画笔和橡皮擦手工编辑分割标签。
- 支持导入课程附件中的 1-bit 多帧 DICOM SEG 分割对象，并按 referenced SOP Instance UID 对齐 CT 切片。
- 在二维视图叠加显示分割边界。
- 使用 VTK Marching Cubes 从分割掩膜生成三维表面模型。
- 在 VTK 三维视图中交互显示重建模型和轴状位、冠状位、矢状位三个二维切片位置。
- 支持调整三维模型透明度并导出 STL 表面模型。
- 支持轴状位 HU/灰度值探针和二维距离测量。
- 显示分割体素数、体积、当前切片面积和 HU 统计。
- 一键导出报告素材包，包括二维叠加图、VTK 三维图和统计文本。

## 运行

```powershell
cd C:\Users\77692\Desktop\医学图像处理\slicer_lite_project
python run_app.py
```

如果使用 Codex 内置 Python：

```powershell
& "C:\Users\77692\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" run_app.py
```

## 建议演示流程

1. 打开 DICOM 文件夹。
2. 浏览轴状位、冠状位、矢状位三向切片。
3. 调整窗宽窗位，展示图像显示效果。
4. 切换图像增强模式。
5. 使用 CT 窗宽预设，运行软组织、骨阈值、Otsu 或多种子区域生长分割，展示红色分割边界。
6. 可选择导入 DICOM SEG，展示标准分割文件的叠加显示。
7. 应用分割后处理，展示最大连通域和填洞效果。
8. 切换不同标签，用 Brush/Eraser 手工修正分割。
9. 点击 `Update 3D`，展示 VTK 三维表面重建和三向切片平面，并导出 STL。
10. 使用 Measure 模块查询 HU/灰度值或测量二维距离。
11. 点击 `Export Pack`，导出报告用截图、标签数组、STL 和统计文本。

## VTK 说明

项目内已支持把 `vtk` 安装到 `.vendor/python`。如果换到其他电脑运行，推荐执行：

```powershell
python -m pip install -r requirements.txt
```

或安装到项目内：

```powershell
python -m pip install --target .vendor/python -r requirements.txt
```

## 打包建议

安装 PyInstaller 后执行：

```powershell
pyinstaller --onefile --windowed --name SlicerLite run_app.py
```

最终提交文件可命名为：

```text
数字图像处理期末大作业_姓名_学号.zip
```
