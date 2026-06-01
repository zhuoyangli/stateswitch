"""MNE-based brain plotting wrapper.

Opt-in replacement for nilearn surface plots. Renders fsaverage6 vertex-wise
and Schaefer parcel-wise maps to PNG via offscreen PyVista. Visual conventions
(threshold, vmax, cmap, 4-panel L/R x lat/med layout, masked colorbar) match
the existing nilearn helpers — see plotting_config.py.

Public API:
    plot_surface_stat_map(lh, rh, save_path, ...)
    plot_parcel_surface_map(parcel_values, save_path, ...)
    plot_roi_mask(roi_def, save_path, ...)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

# Must set offscreen flags BEFORE any mne.viz / pyvista / vtk / Qt import.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MNE_3D_OPTION_ANTIALIAS", "true")

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from plotting_config import (
    PLOT_PARAMS, FIGURE_PARAMS, COLORBAR_PARAMS, LAYOUT_PARAMS,
)
from localizer_glm_surface import create_masked_colormap

_BACKEND_READY: dict = {}
_SCHAEFER_ANNOT_DIR = Path.home() / "nilearn_data" / "schaefer_2018"


def _ensure_mne_backend(subjects_dir: Path | None = None) -> Path:
    """Lazy, idempotent MNE 3D backend bootstrap. Returns resolved SUBJECTS_DIR."""
    if _BACKEND_READY:
        return _BACKEND_READY["subjects_dir"]

    # Pre-create an offscreen QApplication so MNE skips its display check.
    # Module-level reference keeps it from being garbage-collected.
    from qtpy.QtWidgets import QApplication
    _BACKEND_READY["qapp"] = QApplication.instance() or QApplication(["mne_plotting"])

    import mne

    mne.viz.set_3d_backend("pyvistaqt")
    try:
        mne.viz.set_3d_options(antialias=True)
    except Exception:
        pass

    if subjects_dir is None:
        # Prefer a real FreeSurfer install (has fsaverage3..6 out of the box).
        # Fall back to MNE's own fsaverage cache (only ships fsaverage = fsav7).
        fs_home = os.environ.get("FREESURFER_HOME")
        fs_candidates = [
            Path(fs_home) / "subjects" if fs_home else None,
            Path("/usr/local/freesurfer/8.1.0-1/subjects"),
            Path("/usr/local/freesurfer/subjects"),
        ]
        for cand in fs_candidates:
            if cand and (cand / "fsaverage6").exists():
                subjects_dir = cand
                break
        else:
            subjects_dir = Path.home() / "mne_data" / "MNE-fsaverage-data"
    subjects_dir = Path(subjects_dir)
    if not (subjects_dir / "fsaverage").exists():
        subjects_dir.parent.mkdir(parents=True, exist_ok=True)
        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

    _BACKEND_READY["subjects_dir"] = subjects_dir
    return subjects_dir


@lru_cache(maxsize=4)
def _load_schaefer_annot(
    n_parcels: int = 400, networks: int = 17
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Schaefer .annot files. Returns (lh_labels, rh_labels, names).

    lh_labels[i] / rh_labels[i] are 0 for background or 1..N_hemi for a parcel
    (per-hemisphere indexing — LH and RH labels both restart at 1).
    `names` is the global Schaefer name list (LH first, then RH), length
    n_parcels + 1 with 'Background' at index 0 stripped — wait: we return the
    decoded names as nibabel hands them, which includes a leading background
    name. Callers should index with care; see plot_parcel_surface_map.
    """
    lh_path = _SCHAEFER_ANNOT_DIR / (
        f"lh.Schaefer2018_{n_parcels}Parcels_{networks}Networks_order_fsaverage6.annot"
    )
    rh_path = _SCHAEFER_ANNOT_DIR / (
        f"rh.Schaefer2018_{n_parcels}Parcels_{networks}Networks_order_fsaverage6.annot"
    )
    lh_labels, _, lh_names = nib.freesurfer.read_annot(str(lh_path))
    rh_labels, _, rh_names = nib.freesurfer.read_annot(str(rh_path))
    # Decode bytes -> str
    lh_names = [n.decode() if isinstance(n, bytes) else n for n in lh_names]
    rh_names = [n.decode() if isinstance(n, bytes) else n for n in rh_names]
    # Combined global label list: drop the leading background from each hemi
    # then concatenate LH then RH (matches Schaefer 1..n convention).
    lh_real = [n for n in lh_names if not _is_background(n)]
    rh_real = [n for n in rh_names if not _is_background(n)]
    return lh_labels, rh_labels, lh_real + rh_real


def _is_background(name: str) -> bool:
    return name in ("Background", "Medial_Wall", "???")


def _resolve_plot_params(threshold, vmax, cmap):
    threshold = PLOT_PARAMS["threshold"] if threshold is None else threshold
    vmax = PLOT_PARAMS["vmax"] if vmax is None else vmax
    cmap = PLOT_PARAMS["cmap"] if cmap is None else cmap
    return float(threshold), float(vmax), cmap


def _make_brain(hemi: str, surf: str, fsaverage: str, subjects_dir: Path,
                size=(800, 800)):
    import mne
    return mne.viz.Brain(
        subject=fsaverage,
        hemi=hemi,
        surf=surf,
        subjects_dir=str(subjects_dir),
        background="white",
        size=size,
        show=False,
        cortex="low_contrast",
    )


def plot_surface_stat_map(
    lh_values: np.ndarray,
    rh_values: np.ndarray,
    save_path: Path,
    *,
    title: str = "",
    threshold: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    surf: str = "inflated",
    fsaverage: str = "fsaverage6",
    views: Sequence[str] = ("lateral", "medial", "lateral", "medial"),
    hemis: Sequence[str] = ("lh", "lh", "rh", "rh"),
) -> Path:
    """Render a 4-panel L/R x lateral/medial stat map to PNG via MNE.

    Visual analogue of localizer_glm_surface.plot_surface_results(). Sub-threshold
    vertices render in the gray sulcal background, matching the masked-colormap
    convention used across the project.
    """
    subjects_dir = _ensure_mne_backend()
    threshold, vmax, cmap = _resolve_plot_params(threshold, vmax, cmap)

    hemi_values = {"lh": np.asarray(lh_values).astype(float),
                   "rh": np.asarray(rh_values).astype(float)}

    panel_imgs = []
    for hemi, view in zip(hemis, views):
        brain = _make_brain(hemi, surf, fsaverage, subjects_dir)
        try:
            data = hemi_values[hemi]
            brain.add_data(
                data,
                hemi=hemi,
                fmin=threshold,
                fmid=(threshold + vmax) / 2.0,
                fmax=vmax,
                colormap=cmap,
                colorbar=False,
                transparent=True,  # below-fmin -> transparent -> shows sulcal bg
                center=0,
            )
            brain.show_view(view=view, distance="auto")
            brain.plotter.render()
            img = brain.screenshot(mode="rgba")
            panel_imgs.append(img)
        finally:
            brain.close()

    fig, axes = plt.subplots(
        1, len(panel_imgs),
        figsize=FIGURE_PARAMS["figsize"],
        facecolor=FIGURE_PARAMS["facecolor"],
    )
    if len(panel_imgs) == 1:
        axes = [axes]
    for ax, img in zip(axes, panel_imgs):
        ax.imshow(img)
        ax.set_axis_off()

    plt.subplots_adjust(**LAYOUT_PARAMS)

    masked_cmap = create_masked_colormap(
        cmap, -vmax, vmax, threshold, COLORBAR_PARAMS["gray_color"],
    )
    cbar_ax = fig.add_axes(COLORBAR_PARAMS["position"])
    sm = plt.cm.ScalarMappable(
        cmap=masked_cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(COLORBAR_PARAMS["label"], fontsize=COLORBAR_PARAMS["label_size"])
    cbar.ax.tick_params(labelsize=COLORBAR_PARAMS["tick_size"])
    cbar.ax.axhline(y=threshold, color="black", linewidth=0.5, alpha=0.5)
    cbar.ax.axhline(y=-threshold, color="black", linewidth=0.5, alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=16)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=FIGURE_PARAMS["dpi"],
        bbox_inches="tight",
        facecolor=FIGURE_PARAMS["facecolor"],
        edgecolor=FIGURE_PARAMS["edgecolor"],
    )
    plt.close(fig)
    return save_path


def _parcel_dict_to_vertex_arrays(
    parcel_values, n_parcels: int, networks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert parcel-level values into (lh_vertex_values, rh_vertex_values)."""
    lh_labels, rh_labels, names = _load_schaefer_annot(n_parcels, networks)
    n_lh = n_parcels // 2

    if isinstance(parcel_values, dict):
        # Build a length-n_parcels vector from {label_name or 1-indexed id: value}
        vec = np.full(n_parcels, np.nan)
        name_to_global = {nm: i + 1 for i, nm in enumerate(names)}
        for key, val in parcel_values.items():
            if isinstance(key, str):
                gid = name_to_global.get(key)
                if gid is None:
                    raise KeyError(f"Schaefer label not found: {key!r}")
            else:
                gid = int(key)
            vec[gid - 1] = float(val)
    else:
        vec = np.asarray(parcel_values, dtype=float)
        if vec.shape != (n_parcels,):
            raise ValueError(
                f"parcel_values must be length {n_parcels}, got shape {vec.shape}"
            )

    lh_vec = vec[:n_lh]
    rh_vec = vec[n_lh:]

    lh_out = np.zeros(lh_labels.shape[0], dtype=float)
    rh_out = np.zeros(rh_labels.shape[0], dtype=float)
    # Per-hemi annot label ids are 1..n_lh / 1..n_rh (0 = background/medial wall)
    for local_id in range(1, n_lh + 1):
        v = lh_vec[local_id - 1]
        if np.isnan(v):
            continue
        lh_out[lh_labels == local_id] = v
    n_rh = n_parcels - n_lh
    for local_id in range(1, n_rh + 1):
        v = rh_vec[local_id - 1]
        if np.isnan(v):
            continue
        rh_out[rh_labels == local_id] = v
    return lh_out, rh_out


def plot_parcel_surface_map(
    parcel_values,
    save_path: Path,
    *,
    n_parcels: int = 400,
    networks: int = 17,
    fsaverage: str = "fsaverage6",
    **stat_map_kwargs,
) -> Path:
    """Render Schaefer parcel-level values on fsaverage6 surface.

    `parcel_values` is either a length-n_parcels ndarray (Schaefer global order:
    LH parcels 1..n/2, RH parcels n/2+1..n) or a dict keyed by label string or
    1-indexed global parcel id.
    """
    lh_v, rh_v = _parcel_dict_to_vertex_arrays(parcel_values, n_parcels, networks)
    return plot_surface_stat_map(
        lh_v, rh_v, save_path, fsaverage=fsaverage, **stat_map_kwargs,
    )


def plot_roi_mask(
    roi_def: dict,
    save_path: Path,
    *,
    color: str = "red",
    n_parcels: int = 400,
    networks: int = 17,
    surf: str = "inflated",
    fsaverage: str = "fsaverage6",
    views: Sequence[tuple[str, str]] = (
        ("lh", "lateral"), ("lh", "medial"),
        ("rh", "lateral"), ("rh", "medial"),
    ),
    label_text: str | None = None,
    title: str = "",
) -> Path:
    """Render an ROI parcel mask as a colored patch on inflated brain.

    `roi_def` mirrors the existing ROI_SURFACE shape: {"left": [global_parcel_ids],
    "right": [global_parcel_ids]}. Subcortical-only ROIs that have no surface
    parcels can pass empty lists and provide `label_text` for a text panel.
    """
    subjects_dir = _ensure_mne_backend()
    lh_labels, rh_labels, _ = _load_schaefer_annot(n_parcels, networks)
    n_lh = n_parcels // 2

    left_global = [int(i) for i in roi_def.get("left", [])]
    right_global = [int(i) for i in roi_def.get("right", [])]
    lh_local = [g for g in left_global if 1 <= g <= n_lh]
    rh_local = [g - n_lh for g in right_global if g > n_lh]

    lh_mask_verts = np.where(np.isin(lh_labels, lh_local))[0]
    rh_mask_verts = np.where(np.isin(rh_labels, rh_local))[0]

    if label_text and lh_mask_verts.size == 0 and rh_mask_verts.size == 0:
        fig, ax = plt.subplots(
            1, 1, figsize=FIGURE_PARAMS["figsize"],
            facecolor=FIGURE_PARAMS["facecolor"],
        )
        ax.set_axis_off()
        ax.text(0.5, 0.5, label_text, ha="center", va="center", fontsize=20,
                transform=ax.transAxes)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_PARAMS["dpi"], bbox_inches="tight",
                    facecolor=FIGURE_PARAMS["facecolor"])
        plt.close(fig)
        return save_path

    import mne

    panel_imgs = []
    for hemi, view in views:
        brain = _make_brain(hemi, surf, fsaverage, subjects_dir)
        try:
            verts = lh_mask_verts if hemi == "lh" else rh_mask_verts
            if verts.size > 0:
                lbl = mne.Label(
                    vertices=verts, hemi=hemi, subject=fsaverage,
                    name=f"roi_{hemi}",
                )
                brain.add_label(lbl, color=color, alpha=0.85, borders=False)
            brain.show_view(view=view, distance="auto")
            brain.plotter.render()
            panel_imgs.append(brain.screenshot(mode="rgba"))
        finally:
            brain.close()

    fig, axes = plt.subplots(
        1, len(panel_imgs),
        figsize=FIGURE_PARAMS["figsize"],
        facecolor=FIGURE_PARAMS["facecolor"],
    )
    if len(panel_imgs) == 1:
        axes = [axes]
    for ax, img in zip(axes, panel_imgs):
        ax.imshow(img)
        ax.set_axis_off()
    plt.subplots_adjust(**LAYOUT_PARAMS)
    if title:
        fig.suptitle(title, fontsize=16)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path, dpi=FIGURE_PARAMS["dpi"], bbox_inches="tight",
        facecolor=FIGURE_PARAMS["facecolor"],
        edgecolor=FIGURE_PARAMS["edgecolor"],
    )
    plt.close(fig)
    return save_path
