# -*- coding: utf-8 -*-
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

SCRATCH = Path(__file__).parent
nb = json.load(open(r"c:\Users\damie\Documents\ProjectsDL\codingpapers\datasetraw.ipynb", encoding="utf-8"))
src = next(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
           for c in nb["cells"]
           if "show_examples" in (c["source"] if isinstance(c["source"], str) else "".join(c["source"])))
OUT_DIR = SCRATCH / "smoke" / "fivek_data" / "simulated"
out_png = SCRATCH / "show_examples_test.png"
src = src.replace("plt.show()", f"plt.savefig(r'{out_png}', dpi=60)")
exec(src, {"Path": Path, "cv2": cv2, "plt": plt, "OUT_DIR": OUT_DIR})
print("OK ->", out_png)
