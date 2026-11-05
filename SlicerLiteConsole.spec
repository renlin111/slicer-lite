# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path


PYROOT = Path(r'C:\Users\77692\.cache\codex-runtimes\codex-primary-runtime\dependencies\python')
tcl_datas = [
    (str(PYROOT / 'tcl' / 'tcl8.6'), 'tcl/tcl8.6'),
    (str(PYROOT / 'tcl' / 'tk8.6'), 'tcl/tk8.6'),
    (str(PYROOT / 'Lib' / 'tkinter'), 'tkinter'),
]
tk_bins = [
    (str(PYROOT / 'DLLs' / '_tkinter.pyd'), '.'),
    (str(PYROOT / 'DLLs' / 'tcl86t.dll'), '.'),
    (str(PYROOT / 'DLLs' / 'tk86t.dll'), '.'),
]

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=tk_bins,
    datas=tcl_datas,
    hiddenimports=['tkinter', '_tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SlicerLiteConsole',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SlicerLiteConsole',
)
