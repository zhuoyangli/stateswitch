"""
Plot dlPFC Schaefer 400 (17-network) parcels on the inflated surface.

Groups shown (lateral views, both hemispheres):
  - ContA_PFCl / ContA_PFClv   (frontoparietal control, network A)
  - ContB_PFCld / ContB_PFClv  (frontoparietal control, network B)
  - DorsAttnB_FEF               (dorsal attention, frontal eye fields)

Note: 'DorsAttn_PFCl' does not exist in the Schaefer 17-network 400 atlas;
the closest dorsal-attention frontal parcels are DorsAttnB_FEF.

Output:
  figs/schaefer_rois/dlpfc_parcels_surface.png
"""
from pathlib import Path
import numpy as np
import nibabel.freesurfer as fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from nilearn import datasets, plotting

from configs.config import FIGS_DIR

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

# Groups to visualise: (group_label, substring_to_match)
# Match is: hemi_prefix in label AND substring in label
GROUPS = [
    ('ContA PFCl',       'ContA_PFCl'),         # also matches ContA_PFClv
    ('ContB PFCl',       'ContB_PFCl'),          # also matches ContB_PFCld / ContB_PFClv
    ('SalVentAttn PFCl', 'SalVentAttnB_PFCl'),  # lateral PFC, salience network
    ('DefaultB PFCl',    'DefaultB_PFCl'),       # lateral PFC, default network B
    ('DorsAttn FEF',     'DorsAttnB_FEF'),
    ('PFCd (all nets)',  '_PFCd_'),              # ContA/ContB/DefaultA/DefaultB PFCd
]

# One base color per group (will be shaded across parcels within a group)
GROUP_COLORS = [
    '#e41a1c',  # red       — ContA PFCl
    '#377eb8',  # blue      — ContB PFCl
    '#984ea3',  # purple    — SalVentAttn PFCl
    '#a65628',  # brown     — DefaultB PFCl
    '#4daf4a',  # green     — DorsAttn FEF
    '#ff7f00',  # orange    — PFCd
]


def load_dlpfc_texture(annot_path, hemi_prefix):
    """Return per-vertex texture (int rank) and parcel list per group.

    Returns
    -------
    texture : np.ndarray shape (n_vertices,)
        0 = background; values 1..N encode individual parcels
    parcel_info : list of (group_idx, rank_within_group, label)
    global_ranks : dict label -> global_rank (1-based)
    """
    labels_arr, _, names = fs.read_annot(str(annot_path))
    names = [n.decode() if hasattr(n, 'decode') else n for n in names]

    texture = np.zeros(len(labels_arr), dtype=float)
    parcel_info = []   # (group_idx, rank_within_group, label, name_idx)
    global_rank = 0

    for g_idx, (_, substr) in enumerate(GROUPS):
        group_parcels = sorted(
            [(ni, nm) for ni, nm in enumerate(names)
             if hemi_prefix in nm and substr in nm],
            key=lambda x: x[1]
        )
        for rank_within, (name_idx, label) in enumerate(group_parcels):
            global_rank += 1
            texture[labels_arr == name_idx] = global_rank
            parcel_info.append((g_idx, rank_within, label, global_rank))

    return texture, parcel_info, global_rank


def make_parcel_colors(parcel_info, n_total):
    """Make a color array (n_total,) with shades per group."""
    colors = np.zeros((n_total, 4))
    # Count how many parcels per group
    group_counts = {}
    for g_idx, rw, _, _ in parcel_info:
        group_counts[g_idx] = group_counts.get(g_idx, 0) + 1

    import matplotlib.colors as mcolors
    for g_idx, rw, _, global_rank in parcel_info:
        base = mcolors.to_rgba(GROUP_COLORS[g_idx])
        n = group_counts[g_idx]
        # shade from 1.0 (full color) to 0.45 (darker)
        t = rw / max(n - 1, 1)
        factor = 1.0 - 0.55 * t
        colors[global_rank - 1] = (
            base[0] * factor,
            base[1] * factor,
            base[2] * factor,
            1.0,
        )
    return colors


def main():
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    tex_lh, info_lh, n_lh = load_dlpfc_texture(LH_ANNOT, 'LH')
    tex_rh, info_rh, n_rh = load_dlpfc_texture(RH_ANNOT, 'RH')

    # Print found parcels
    for hemi, info in [('LH', info_lh), ('RH', info_rh)]:
        print(f'\n{hemi} dlPFC parcels:')
        for g_idx, rw, label, gr in info:
            print(f'  [{GROUPS[g_idx][0]}]  {label}  (rank {gr})')

    # Build color maps — LH and RH get independent color arrays
    colors_lh = make_parcel_colors(info_lh, n_lh) if n_lh > 0 else np.zeros((1, 4))
    colors_rh = make_parcel_colors(info_rh, n_rh) if n_rh > 0 else np.zeros((1, 4))

    cmap_lh = ListedColormap(colors_lh)
    cmap_rh = ListedColormap(colors_rh)

    # -------------------------------------------------------------------------
    # Figure: 2 rows × 2 cols
    #   Row 0: lateral views   (LH left, RH right)
    #   Row 1: medial views    (LH left, RH right)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             subplot_kw={'projection': '3d'})
    fig.suptitle('Schaefer 400 (17-network) — dlPFC Parcels', fontsize=13,
                 fontweight='bold')

    panels = [
        (axes[0, 0], 'left',  'lateral', fsavg.infl_left,  fsavg.sulc_left,  tex_lh, cmap_lh, n_lh, 'LH Lateral'),
        (axes[0, 1], 'right', 'lateral', fsavg.infl_right, fsavg.sulc_right, tex_rh, cmap_rh, n_rh, 'RH Lateral'),
        (axes[1, 0], 'left',  'medial',  fsavg.infl_left,  fsavg.sulc_left,  tex_lh, cmap_lh, n_lh, 'LH Medial'),
        (axes[1, 1], 'right', 'medial',  fsavg.infl_right, fsavg.sulc_right, tex_rh, cmap_rh, n_rh, 'RH Medial'),
    ]

    for ax, hemi, view, mesh, bg, tex, cmap, n, title in panels:
        if n == 0:
            ax.set_title(title + ' (none)', fontsize=10)
            continue
        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=tex,
            hemi=hemi,
            view=view,
            bg_map=bg,
            axes=ax,
            colorbar=False,
            cmap=cmap,
            vmin=0.5,
            vmax=n + 0.5,
            threshold=0.5,
            bg_on_data=True,
            darkness=0.5,
        )
        ax.set_title(title, fontsize=11, fontweight='bold', pad=2)

    # -------------------------------------------------------------------------
    # Legend: one patch per parcel, grouped
    # -------------------------------------------------------------------------
    all_patches = []
    for hemi_label, info, colors in [('LH', info_lh, colors_lh),
                                      ('RH', info_rh, colors_rh)]:
        if not info:
            continue
        all_patches.append(mpatches.Patch(color='none',
                                           label=f'── {hemi_label} ──'))
        prev_group = None
        for g_idx, rw, label, gr in info:
            if g_idx != prev_group:
                all_patches.append(mpatches.Patch(color='none',
                                                   label=f'  {GROUPS[g_idx][0]}'))
                prev_group = g_idx
            short = label.replace(f'17Networks_{hemi_label}_', '')
            all_patches.append(mpatches.Patch(color=colors[gr - 1], label=f'    {short}'))

    fig.legend(
        handles=all_patches,
        loc='lower center',
        ncol=4,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.0,
        handlelength=1.2,
    )

    plt.subplots_adjust(bottom=0.02, hspace=0.05, wspace=0.0)

    out_dir = FIGS_DIR / 'schaefer_rois'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'pfc_parcels_surface.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
