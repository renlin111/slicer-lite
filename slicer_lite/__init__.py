from pathlib import Path
import sys


_vendor = Path(__file__).resolve().parents[1] / ".vendor" / "python"
if _vendor.exists() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))

__version__ = "0.1.0"
