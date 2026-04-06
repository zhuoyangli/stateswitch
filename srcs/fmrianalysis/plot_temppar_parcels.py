"""
Plot Schaefer 400 TempPar parcels on the inflated surface with labels.

Uses the surface-based fsaverage6 .annot files (no vol-to-surf remapping).
Shows lateral views only (LH left, RH right) with a full legend of all parcels.

Output:
  figs/tomloc_schaefer/temppar_parcels_surface.png
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


def load_temppar_texture(annot_path, hemi_prefix):
    """Return (per-vertex texture array, list of (rank, label) for TempPar parcels)."""
    labels, ctab, names = fs.read_annot(str(annot_path))
    names = [n.decode() if hasattr(n, 'decode') else n for n in names]

    temppar = [(name_idx, name) for name_idx, name in enumerate(names)
               if f'{hemi_prefix}_TempPar' in name]
    temppar.sort(key=lambda x: int(x[1].split('_')[-1]))  # sort numerically

    texture = np.zeros(len(labels), dtype=float)
    for rank, (name_idx, _) in enumerate(temppar, 1):
        texture[labels == name_idx] = rank

    return texture, temppar


def main():
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    # Load surface-based textures
    tex_lh, temppar_lh = load_temppar_texture(LH_ANNOT, 'LH')
    tex_rh, temppar_rh = load_temppar_texture(RH_ANNOT, 'RH')

    n_lh = len(temppar_lh)
    n_rh = len(temppar_rh)
    n_total = n_lh + n_rh

    print(f"LH TempPar parcels ({n_lh}):")
    for rank, (_, lbl) in enumerate(temppar_lh, 1):
        print(f"  {rank:2d}. {lbl}")
    print(f"\nRH TempPar parcels ({n_rh}):")
    for rank, (_, lbl) in enumerate(temppar_rh, 1):
        print(f"  {rank:2d}. {lbl}")

    # RH texture values need to be offset so they get distinct colors from LH
    tex_rh_offset = np.where(tex_rh > 0, tex_rh + n_lh, 0)

    # 20 visually distinct colors via tab20
    tab20 = plt.cm.get_cmap('tab20', n_total)
    colors = [tab20(i) for i in range(n_total)]
    cmap = ListedColormap(colors)
    vmin, vmax = 0.5, n_total + 0.5

    # -------------------------------------------------------------------------
    # Figure: 1 row × 2 cols — lateral views only
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={'projection': '3d'})

    for ax, hemi, view, mesh, bg, texture, title in [
        (axes[0], 'left',  'lateral', fsavg.infl_left,  fsavg.sulc_left,  tex_lh,        'LH Lateral'),
        (axes[1], 'right', 'lateral', fsavg.infl_right, fsavg.sulc_right, tex_rh_offset, 'RH Lateral'),
    ]:
        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=texture,
            hemi=hemi,
            view=view,
            bg_map=bg,
            axes=ax,
            colorbar=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=0.5,
            bg_on_data=True,
            darkness=0.5,
        )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=2)

    fig.suptitle('Schaefer 400 (17-network) — TempPar Parcels', fontsize=13, fontweight='bold')

    # -------------------------------------------------------------------------
    # Legend: all LH parcels, then all RH parcels — two labeled groups
    # -------------------------------------------------------------------------
    lh_patches = [
        mpatches.Patch(color=colors[rank - 1],
                       label=lbl.replace('17Networks_LH_TempPar_', 'LH_TempPar_'))
        for rank, (_, lbl) in enumerate(temppar_lh, 1)
    ]
    rh_patches = [
        mpatches.Patch(color=colors[n_lh + rank - 1],
                       label=lbl.replace('17Networks_RH_TempPar_', 'RH_TempPar_'))
        for rank, (_, lbl) in enumerate(temppar_rh, 1)
    ]

    # Pad LH list with blank entries so columns align (LH: 6, RH: 10)
    # Layout: 2 columns per hemisphere → ncol=4, rows = ceil(max(n_lh, n_rh)/2)
    n_cols_per_hemi = 2
    blank = mpatches.Patch(color='none', label='')
    n_pad = (n_cols_per_hemi - n_lh % n_cols_per_hemi) % n_cols_per_hemi
    lh_patches_padded = lh_patches + [blank] * n_pad

    # Interleave: LH col-pair | RH col-pair across rows
    all_patches = lh_patches_padded + rh_patches

    legend = fig.legend(
        handles=all_patches,
        loc='lower center',
        ncol=n_cols_per_hemi * 2,   # 4 columns total: 2 LH + 2 RH
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.12),
        columnspacing=1.0,
        handlelength=1.2,
    )

    # Add hemisphere labels above legend columns
    fig.text(0.28, -0.04, 'Left hemisphere', ha='center', fontsize=9, fontweight='bold')
    fig.text(0.72, -0.04, 'Right hemisphere', ha='center', fontsize=9, fontweight='bold')

    plt.subplots_adjust(bottom=0.05, wspace=0.0)

    out_dir = FIGS_DIR / 'tomloc_schaefer'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'temppar_parcels_surface.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
