"""Makes the modules living in main_files/ importable.

`acr_tone_fast.finish` (ACR rendering curve + sRGB gamma) is shared with
training: we import it rather than keep a copy, so that the culling tone
mapping is exactly the dataloader's.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN_FILES = REPO / "main_files"


def ensure():
    for p in (str(MAIN_FILES), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


ensure()
