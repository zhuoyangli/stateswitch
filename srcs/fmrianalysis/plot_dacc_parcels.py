"""
Visualise dACC parcels from the Schaefer 400 7-network and 17-network atlases.

Two matching strategies:
  - Label matching : find parcels by name (SalVentAttn_Med_1/2/4), then look up
                     the corresponding 17-net parcel by voxel overlap.
  - Coord matching : find whatever parcel sits at the user-provided MNI centroid
                     in each atlas independently.

Uses surface annotation files directly for crisp parcel boundaries.

Figure layout (4 rows × 2 cols, all medial views):
  Row 0  7-net   label-matched      LH | RH
  Row 1  17-net  label-matched      LH | RH
  Row 2  7-net   coord-matched      LH | RH
  Row 3  17-net  coord-matched      LH | RH

Color key (same across all rows):
  red   = Med_1 pair  (user parcels 107 / 311)
  blue  = Med_2 pair  (user parcels 108 / 312)
  green = Med_4       (user parcel  110,  LH only)

Output:
  figs/schaefer_rois/dacc_parcels_surface.png
"""
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.freesurfer as fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting

from configs.config import FIGS_DIR

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')

# ── Color scheme ─────────────────────────────────────────────────────────────
COLORS      = {1: '#e41a1c', 2: '#377eb8', 3: '#4daf4a'}  # red, blue, green
PAIR_LABELS = {1: 'Med_1 pair', 2: 'Med_2 pair', 3: 'Med_4 (LH only)'}

# ── Annot-index matching tables ───────────────────────────────────────────────
# Annot files have 201 entries per hemisphere: index 0 = background, 1-200 = parcels.
# User's global IDs: LH 1-200 = annot idx directly; RH 201-400 → annot idx = global - 200.

# Label matching
# 7-net annot: verified by reading annot names
LABEL_7NET_ANNOT = {
    'LH': {107: 1, 108: 2, 110: 3},  # Med_1, Med_2, Med_4
    'RH': {111: 1, 112: 2},           # Med_1, Med_2  (global 311-200, 312-200)
}
# 17-net annot: annot idx = nilearn vol id - 1 (LH), vol id - 201 (RH)
LABEL_17NET_ANNOT = {
    'LH': {108: 1, 98: 2, 99: 3},    # PFCmp_1, FrMed_1, FrMed_2
    'RH': {112: 1, 96: 2},            # PFCmp_2, FrMed_1
}

# Coord matching: annot indices for parcels found at MNI coords
# (converted from nilearn vol IDs via name matching)
COORD_7NET_ANNOT = {
    'LH': {106: 1, 107: 2, 109: 3},  # vol 107→PFCl_1(annot106), 108→Med_1(annot107), 110→Med_3(annot109)
    'RH': {183: 1, 111: 2},           # vol 384→RH_Default_PFCdPFCm_5(annot183), 312→Med_1(annot111)
}
COORD_17NET_ANNOT = {
    'LH': {107: 1, 97: 2, 98: 3},    # vol 108→OFC_1(annot107), 98→ParMed_3(annot97), 99→FrMed_1(annot98)
    'RH': {172: 1, 95: 2},            # vol 373→PFCm_5(annot172), 296→FrOper_3(annot95)
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def annot_texture(annot_path, idx_to_val):
    """Build a per-vertex texture from an annot file.

    idx_to_val : dict {annot_name_index -> color_value}
    """
    labels_arr, _, _ = fs.read_annot(str(annot_path))
    texture = np.zeros(len(labels_arr), dtype=float)
    for idx, val in idx_to_val.items():
        texture[labels_arr == idx] = val
    return texture


def plot_panel(ax, texture, fsavg, hemi, title):
    surf = fsavg.infl_left  if hemi == 'left'  else fsavg.infl_right
    sulc = fsavg.sulc_left  if hemi == 'left'  else fsavg.sulc_right
    cmap = ListedColormap([COLORS[1], COLORS[2], COLORS[3]])
    plotting.plot_surf_stat_map(
        surf_mesh=surf, stat_map=texture,
        hemi=hemi, view='medial', bg_map=sulc, axes=ax,
        colorbar=False, cmap=cmap, vmin=0.5, vmax=3.5,
        threshold=0.5, bg_on_data=True, darkness=0.5,
    )
    ax.set_title(title, fontsize=9, fontweight='bold', pad=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    lh7  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_7Networks_order.annot'
    rh7  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_7Networks_order.annot'
    lh17 = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
    rh17 = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'

    # Print parcel names for all annot indices used
    def describe_annot(annot_path, idx_map, tag):
        _, _, names = fs.read_annot(str(annot_path))
        names = [n.decode() if hasattr(n, 'decode') else n for n in names]
        print(f'  {tag}')
        for idx, val in idx_map.items():
            print(f'    annot[{idx}] {names[idx]}  (color {val} = {PAIR_LABELS[val]})')

    print('\n=== Label matching ===')
    describe_annot(lh7,  LABEL_7NET_ANNOT['LH'],  '7-net LH')
    describe_annot(rh7,  LABEL_7NET_ANNOT['RH'],  '7-net RH')
    describe_annot(lh17, LABEL_17NET_ANNOT['LH'], '17-net LH')
    describe_annot(rh17, LABEL_17NET_ANNOT['RH'], '17-net RH')
    print('\n=== Coord matching ===')
    describe_annot(lh7,  COORD_7NET_ANNOT['LH'],  '7-net LH')
    describe_annot(rh7,  COORD_7NET_ANNOT['RH'],  '7-net RH')
    describe_annot(lh17, COORD_17NET_ANNOT['LH'], '17-net LH')
    describe_annot(rh17, COORD_17NET_ANNOT['RH'], '17-net RH')

    # Build textures for each row
    rows = [
        ('Label match\n7-network',  lh7,  rh7,  LABEL_7NET_ANNOT),
        ('Label match\n17-network', lh17, rh17, LABEL_17NET_ANNOT),
        ('Coord match\n7-network',  lh7,  rh7,  COORD_7NET_ANNOT),
        ('Coord match\n17-network', lh17, rh17, COORD_17NET_ANNOT),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(10, 18),
                             subplot_kw={'projection': '3d'})
    fig.suptitle('dACC Parcels: label vs. coordinate matching\n'
                 'Schaefer 400 (7-net and 17-net atlases)',
                 fontsize=12, fontweight='bold')

    for r, (row_label, lh_annot, rh_annot, idx_map) in enumerate(rows):
        tex_lh = annot_texture(lh_annot, idx_map['LH'])
        tex_rh = annot_texture(rh_annot, idx_map['RH'])
        plot_panel(axes[r, 0], tex_lh, fsavg, 'left',  'LH medial')
        plot_panel(axes[r, 1], tex_rh, fsavg, 'right', 'RH medial')

        bbox = axes[r, 0].get_position()
        fig.text(0.01, bbox.y0 + bbox.height / 2,
                 row_label, ha='center', va='center',
                 fontsize=9, fontweight='bold', rotation=90)

    patches = [mpatches.Patch(color=COLORS[v], label=lbl)
               for v, lbl in PAIR_LABELS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06,
                        hspace=0.05, wspace=0.02)

    out_dir = FIGS_DIR / 'schaefer_rois'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'dacc_parcels_surface.png'
    plt.savefig(out_path, dpi=200, facecolor='white')
    plt.close()
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
