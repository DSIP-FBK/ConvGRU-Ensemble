import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_ROOT / "convgru-ens"

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
