"""Hace importable `src` y `dashboard` al ejecutar pytest desde la raíz del proyecto."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
