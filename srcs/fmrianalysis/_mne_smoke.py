"""Smoke test for the MNE plotting backend.

Run: uv run python srcs/fmrianalysis/_mne_smoke.py
Renders a random fsaverage6 stat map and a parcel map to figs/, then asserts
both PNGs were written and are non-trivially sized.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure srcs/ is on sys.path so `configs.config` resolves regardless of CWD.
_SRCS = Path(__file__).resolve().parent.parent
if str(_SRCS) not in sys.path:
    sys.path.insert(0, str(_SRCS))

import numpy as np

from configs.config import FIGS_DIR
from mne_plotting import (
    _BACKEND_READY,
    _ensure_mne_backend,
    plot_parcel_surface_map,
    plot_surface_stat_map,
)

FSAV6_N_VERTS = 40962


def main() -> None:
    t0 = time.time()
    subjects_dir = _ensure_mne_backend()
    t_init = time.time() - t0
    print(f"backend ready in {t_init:.1f}s, subjects_dir={subjects_dir}")

    rng = np.random.default_rng(0)
    lh = rng.standard_normal(FSAV6_N_VERTS) * 3.0
    rh = rng.standard_normal(FSAV6_N_VERTS) * 3.0

    out1 = FIGS_DIR / "_mne_smoke_vertex.png"
    t1 = time.time()
    plot_surface_stat_map(lh, rh, out1, title="MNE backend smoke test (vertex)")
    t_vertex = time.time() - t1
    sz1 = out1.stat().st_size
    assert sz1 > 50_000, f"smoke PNG suspiciously small: {sz1} bytes ({out1})"
    print(f"vertex map: {out1} ({sz1/1024:.1f} KB) in {t_vertex:.1f}s")

    parcel_vals = rng.standard_normal(400) * 4.0
    out2 = FIGS_DIR / "_mne_smoke_parcel.png"
    t2 = time.time()
    plot_parcel_surface_map(parcel_vals, out2,
                            title="MNE backend smoke test (Schaefer-400)")
    t_parcel = time.time() - t2
    sz2 = out2.stat().st_size
    assert sz2 > 50_000, f"smoke PNG suspiciously small: {sz2} bytes ({out2})"
    print(f"parcel map: {out2} ({sz2/1024:.1f} KB) in {t_parcel:.1f}s")

    print(f"OK: backend=pyvista, subjects_dir={subjects_dir}, "
          f"total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
