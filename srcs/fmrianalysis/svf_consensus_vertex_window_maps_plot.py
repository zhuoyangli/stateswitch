"""
Stage 2 (plot) — whole-brain vertex z-score maps of the SVF peri-word response,
split by switching/clustering rater-consensus level.

Reads the .npz caches from svf_consensus_vertex_window_maps.py. Renders (group by
default; --per-subject for individuals):
  1. levels grid   : rows = consensus levels, cols = 4 brain views, one peri-word
                     window (default 3-9 s, the HRF peak for onset-locked words).
  2. contrast      : switching - clustering (class mode), 4 brain views, diverging.
  3. montage       : per level, rows = windows, cols = 4 views (--montage).

Value = mean z-scored BOLD in the window. Diverging RdBu_r, center 0. Because
maps are locked to word onset, every consensus level is on the same clock, so the
grid rows are directly comparable and neighbouring-word activity is not folded
into the picture the way it is in the ROI time courses.

Run under the MNE/PyVista env (NOT uv):
    /home/envs/ssgl/bin/python srcs/fmrianalysis/svf_consensus_vertex_window_maps_plot.py
    /home/envs/ssgl/bin/python srcs/fmrianalysis/svf_consensus_vertex_window_maps_plot.py --by votes
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # mne_plotting / plotting_config / localizer_glm_surface
sys.path.insert(0, str(_HERE.parent))   # configs.config

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.config import FIGS_DIR, ANALYSIS_CACHE_DIR
from plotting_config import FIGURE_PARAMS, COLORBAR_PARAMS
from localizer_glm_surface import create_masked_colormap
from mne_plotting import _ensure_mne_backend, _make_brain

CACHE_DIR = ANALYSIS_CACHE_DIR / 'svf_consensus_vertex_window_maps'
OUTPUT_DIR = FIGS_DIR / 'svf_boundary_consensus_surface'

CLASS_LEVELS = ['clustering', 'ambiguous', 'switching']
VOTE_LEVELS = [f'k{k}' for k in range(8)]
CLASS_LABEL = {'clustering': 'Clustering\n(k≤1)', 'ambiguous': 'Ambiguous\n(k=2–5)',
               'switching': 'Switching\n(k≥6)'}

PANELS = [('lh', 'lateral'), ('lh', 'medial'), ('rh', 'lateral'), ('rh', 'medial')]
PANEL_TITLES = ['L lateral', 'L medial', 'R lateral', 'R medial']
CMAP = 'RdBu_r'

WINDOW_SETS = {
    '4.5s': [(-9.0, -4.5), (-4.5, 0.0), (0.0, 4.5), (4.5, 9.0),
             (9.0, 13.5), (13.5, 18.0)],
}

# PMC ROI outline (Schaefer 17-net DefaultA_pCunPCC), same as the other boundary maps.
SCHAEFER_ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
SCHAEFER_ANNOT = {
    'lh': SCHAEFER_ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot',
    'rh': SCHAEFER_ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot',
}
PMC_MATCH = 'DefaultA_pCunPCC'
PMC_COLOR = 'white'


def levels_for(mode):
    return CLASS_LEVELS if mode == 'class' else VOTE_LEVELS


def level_label(mode, lvl):
    if mode == 'class':
        return CLASS_LABEL[lvl]
    return f'{lvl[1:]}/7\nvotes'


def load_schaefer_roi_vertices(match):
    import nibabel.freesurfer as fs
    out = {}
    for hemi, annot in SCHAEFER_ANNOT.items():
        if not annot.exists():
            out[hemi] = np.array([], dtype=int)
            continue
        labels, _, names = fs.read_annot(str(annot))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        idxs = [i for i, nm in enumerate(names) if match in nm]
        out[hemi] = np.where(np.isin(labels, idxs))[0]
    return out


def _add_border(brain, hemi, vertices, fsaverage, color, name):
    import mne
    label = mne.Label(vertices=np.sort(np.asarray(vertices)), hemi=hemi,
                      subject=fsaverage, name=name)
    brain.add_label(label, color=color, borders=1, alpha=1.0)


def _fmt_sec(x):
    s = f'{abs(x):g}'
    return ('-' + s) if x < 0 else s


def _window_map(epoch, time_sec, a, b):
    """Mean over TRs whose center time falls in [a, b). epoch: (n_window, V)."""
    sel = (time_sec >= a - 1e-6) & (time_sec < b - 1e-6)
    if not sel.any():
        sel = np.zeros_like(time_sec, dtype=bool)
        sel[np.argmin(np.abs(time_sec - 0.5 * (a + b)))] = True
    return epoch[sel].mean(axis=0)


def _panel_img(hemi, view, values, subjects_dir, fsaverage, threshold, vmax,
               pmc_verts=None):
    brain = _make_brain(hemi, 'inflated', fsaverage, subjects_dir)
    try:
        brain.add_data(
            np.asarray(values, float), hemi=hemi,
            fmin=threshold, fmid=(threshold + vmax) / 2.0, fmax=vmax,
            colormap=CMAP, colorbar=False, transparent=True, center=0)
        if pmc_verts is not None and len(pmc_verts.get(hemi, [])):
            _add_border(brain, hemi, pmc_verts[hemi], fsaverage, PMC_COLOR, 'PMC')
        brain.show_view(view=view, distance='auto')
        brain.plotter.render()
        return brain.screenshot(mode='rgba')
    finally:
        brain.close()


def _colorbar(fig, vmax, threshold, x=0.92):
    masked_cmap = create_masked_colormap(
        CMAP, -vmax, vmax, threshold, COLORBAR_PARAMS['gray_color'])
    cax = fig.add_axes([x, 0.3, 0.015, 0.4])
    sm = plt.cm.ScalarMappable(cmap=masked_cmap,
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('BOLD (z-scored)', fontsize=COLORBAR_PARAMS['label_size'])
    cbar.ax.tick_params(labelsize=COLORBAR_PARAMS['tick_size'])


# ---------------------------------------------------------------------------
# Group aggregation across subjects
# ---------------------------------------------------------------------------

def load_level_maps(subjects, mode, align):
    """{level: {'epoch_L','epoch_R','time_sec','n_events','n_subjects'}} group mean."""
    tag = f'by-{mode}_align-{align}'
    group = {}
    for lvl in levels_for(mode):
        eL, eR, tsec, nev = [], [], None, 0
        for sub in subjects:
            p = CACHE_DIR / f'{sub}_{lvl}_{tag}.npz'
            if not p.exists():
                continue
            d = np.load(p)
            eL.append(d['epoch_L']); eR.append(d['epoch_R'])
            tsec = d['time_sec']; nev += int(d['n_events'])
        if eL:
            group[lvl] = dict(epoch_L=np.mean(eL, axis=0), epoch_R=np.mean(eR, axis=0),
                              time_sec=tsec, n_events=nev, n_subjects=len(eL))
    return group


def load_subject_maps(subject, mode, align):
    tag = f'by-{mode}_align-{align}'
    out = {}
    for lvl in levels_for(mode):
        p = CACHE_DIR / f'{subject}_{lvl}_{tag}.npz'
        if p.exists():
            d = np.load(p)
            out[lvl] = dict(epoch_L=d['epoch_L'], epoch_R=d['epoch_R'],
                            time_sec=d['time_sec'], n_events=int(d['n_events']),
                            n_subjects=1)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def render_levels_grid(maps, mode, window, save_path, subjects_dir, fsaverage,
                       threshold, vmax, title, pmc_verts=None):
    """Rows = consensus levels, cols = 4 brain views, one window."""
    levels = [l for l in levels_for(mode) if l in maps]
    a, b = window
    nrows, ncols = len(levels), len(PANELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2 + 1.4, nrows * 1.9),
                             facecolor=FIGURE_PARAMS['facecolor'], squeeze=False)
    for r, lvl in enumerate(levels):
        d = maps[lvl]
        win = {h: _window_map(d['epoch_L'] if h == 'lh' else d['epoch_R'],
                              d['time_sec'], a, b) for h in ('lh', 'rh')}
        for c, (hemi, view) in enumerate(PANELS):
            ax = axes[r, c]
            ax.imshow(_panel_img(hemi, view, win[hemi], subjects_dir, fsaverage,
                                 threshold, vmax, pmc_verts=pmc_verts))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(PANEL_TITLES[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{level_label(mode, lvl)}\n(n={d['n_events']})",
                              fontsize=9, rotation=0, ha='right', va='center', labelpad=26)
    fig.suptitle(f"{title}\nmean z-scored BOLD, {_fmt_sec(a)} to {_fmt_sec(b)} s",
                 fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.12, right=0.9, top=0.9, bottom=0.02, wspace=0.02, hspace=0.05)
    _colorbar(fig, vmax, threshold)
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIGURE_PARAMS['facecolor'])
    plt.close(fig)
    print(f"  saved {save_path}")


def render_contrast(map_a, map_b, window, save_path, subjects_dir, fsaverage,
                    threshold, vmax, title, pmc_verts=None):
    """switching - clustering (or any A-B) contrast, 4 brain views, one window."""
    a, b = window
    fig, axes = plt.subplots(1, len(PANELS), figsize=(len(PANELS) * 2.2 + 1.4, 3.0),
                             facecolor=FIGURE_PARAMS['facecolor'], squeeze=False)
    for c, (hemi, view) in enumerate(PANELS):
        key = 'epoch_L' if hemi == 'lh' else 'epoch_R'
        diff_epoch = map_a[key] - map_b[key]
        vals = _window_map(diff_epoch, map_a['time_sec'], a, b)
        ax = axes[0, c]
        ax.imshow(_panel_img(hemi, view, vals, subjects_dir, fsaverage,
                             threshold, vmax, pmc_verts=pmc_verts))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(PANEL_TITLES[c], fontsize=10)
    fig.suptitle(f"{title}\nΔ z-scored BOLD (red = switching > clustering), "
                 f"{_fmt_sec(a)} to {_fmt_sec(b)} s", fontsize=13, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.68, bottom=0.02, wspace=0.02)
    _colorbar(fig, vmax, threshold)
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIGURE_PARAMS['facecolor'])
    plt.close(fig)
    print(f"  saved {save_path}")


def render_level_montage(d, mode, lvl, windows, save_path, subjects_dir, fsaverage,
                         threshold, vmax, title, pmc_verts=None):
    """One level: rows = windows, cols = 4 brain views."""
    nrows, ncols = len(windows), len(PANELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2 + 1.2, nrows * 1.7),
                             facecolor=FIGURE_PARAMS['facecolor'], squeeze=False)
    for r, (a, b) in enumerate(windows):
        win = {h: _window_map(d['epoch_L'] if h == 'lh' else d['epoch_R'],
                              d['time_sec'], a, b) for h in ('lh', 'rh')}
        for c, (hemi, view) in enumerate(PANELS):
            ax = axes[r, c]
            ax.imshow(_panel_img(hemi, view, win[hemi], subjects_dir, fsaverage,
                                 threshold, vmax, pmc_verts=pmc_verts))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(PANEL_TITLES[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{_fmt_sec(a)} to {_fmt_sec(b)} s', fontsize=9,
                              rotation=90, labelpad=6)
    fig.suptitle(f"{title}  (n={d['n_events']})\nmean z-scored BOLD, time from word onset",
                 fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.01, wspace=0.02, hspace=0.05)
    _colorbar(fig, vmax, threshold)
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIGURE_PARAMS['facecolor'])
    plt.close(fig)
    print(f"  saved {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--by', choices=['class', 'votes'], default='class')
    ap.add_argument('--align', choices=['onset', 'offset'], default='onset')
    ap.add_argument('--subjects', nargs='+', default=None,
                    help='subjects to include in the group mean (default: all cached)')
    ap.add_argument('--per-subject', action='store_true',
                    help='also render per-subject figures')
    ap.add_argument('--montage', action='store_true',
                    help='also render per-level rows=windows temporal montage')
    ap.add_argument('--peak', type=float, nargs=2, default=[3.0, 9.0],
                    help='HRF peak window (s) for the levels grid / contrast')
    ap.add_argument('--vmax', type=float, default=0.4)
    ap.add_argument('--threshold', type=float, default=0.05)
    ap.add_argument('--contrast-vmax', type=float, default=0.15)
    ap.add_argument('--contrast-threshold', type=float, default=0.03)
    args = ap.parse_args()

    subjects_dir = _ensure_mne_backend()
    fsaverage = 'fsaverage6'
    pmc_verts = load_schaefer_roi_vertices(PMC_MATCH)
    peak = tuple(args.peak)
    tag = f'by-{args.by}_align-{args.align}'

    # discover cached subjects
    cached_subs = sorted({p.name.split('_')[0] for p in CACHE_DIR.glob(f'sub-*_{tag}.npz')})
    subjects = args.subjects or cached_subs
    if not subjects:
        print(f"No caches found in {CACHE_DIR} for {tag}. Run the compute stage first.")
        return
    print(f"Group over {len(subjects)} subjects: {subjects}")

    # ---- GROUP ----
    group = load_level_maps(subjects, args.by, args.align)
    if not group:
        print("No group maps.")
        return
    nsub = max(m['n_subjects'] for m in group.values())
    base = OUTPUT_DIR / f'group_{tag}'

    render_levels_grid(
        group, args.by, peak, base / f'GROUP_levels_grid_{tag}.png',
        subjects_dir, fsaverage, args.threshold, args.vmax,
        title=f'SVF peri-word surface maps by consensus level — Group (N={nsub})',
        pmc_verts=pmc_verts)

    if args.by == 'class' and 'switching' in group and 'clustering' in group:
        render_contrast(
            group['switching'], group['clustering'], peak,
            base / f'GROUP_switching_minus_clustering_{tag}.png',
            subjects_dir, fsaverage, args.contrast_threshold, args.contrast_vmax,
            title=f'SVF: switching − clustering — Group (N={nsub})',
            pmc_verts=pmc_verts)

    if args.montage:
        for lvl, d in group.items():
            render_level_montage(
                d, args.by, lvl, WINDOW_SETS['4.5s'],
                base / f'GROUP_montage_{lvl}_{tag}.png',
                subjects_dir, fsaverage, args.threshold, args.vmax,
                title=f'SVF {lvl} — Group (N={d["n_subjects"]})', pmc_verts=pmc_verts)

    # ---- PER SUBJECT ----
    if args.per_subject:
        for sub in subjects:
            smaps = load_subject_maps(sub, args.by, args.align)
            if not smaps:
                continue
            sbase = OUTPUT_DIR / sub
            render_levels_grid(
                smaps, args.by, peak, sbase / f'{sub}_levels_grid_{tag}.png',
                subjects_dir, fsaverage, args.threshold, args.vmax,
                title=f'SVF peri-word surface maps by consensus level — {sub}',
                pmc_verts=pmc_verts)
            if args.by == 'class' and 'switching' in smaps and 'clustering' in smaps:
                render_contrast(
                    smaps['switching'], smaps['clustering'], peak,
                    sbase / f'{sub}_switching_minus_clustering_{tag}.png',
                    subjects_dir, fsaverage, args.contrast_threshold, args.contrast_vmax,
                    title=f'SVF: switching − clustering — {sub}', pmc_verts=pmc_verts)


if __name__ == '__main__':
    main()
