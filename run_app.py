from pathlib import Path
import os
import sys


runtime = Path(sys.executable).resolve().parent
runtime_tcl = runtime / "tcl"
bundle_tcl = Path(getattr(sys, "_MEIPASS", runtime)) / "tcl"
project_tcl = Path(__file__).resolve().parent / ".vendor" / "python" / "tcl"
for base in (bundle_tcl, runtime_tcl, project_tcl):
    if (base / "tcl8.6" / "init.tcl").exists():
        os.environ.setdefault("TCL_LIBRARY", str(base / "tcl8.6"))
    if (base / "tk8.6" / "tk.tcl").exists():
        os.environ.setdefault("TK_LIBRARY", str(base / "tk8.6"))

vendor = Path(__file__).resolve().parent / ".vendor" / "python"
if vendor.exists():
    sys.path.insert(0, str(vendor))

from slicer_lite.app.main_window import main


if __name__ == "__main__":
    main()
