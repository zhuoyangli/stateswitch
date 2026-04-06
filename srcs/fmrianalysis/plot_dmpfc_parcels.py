"""
Visualise dmPFC parcels from DORSAL_MPFC in schaefer_rois.py.

Each parcel is plotted individually with a distinct color.

LH parcel IDs (Schaefer 400 17-network, annot index == parcel ID):
  Group A: 161–166  (DefaultA_PFCm_1 to _6)
  Group B: 175–180  (DefaultB_PFCd_1 to _6)

RH annot indices found by name-matching:
  Group A: 168–173  (DefaultA_PFCm_1 to _6)
  Group B: 177–181  (DefaultB_PFCd_1 to _5)

Figure: 2 rows (LH, RH) × 2 cols (medial, lateral).

Output:
  figs/schaefer_rois/dmpfc_parcels_surface.png
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

# Each entry: (annot_id, color_index 1-based, short label)
# Color index 1–6: DefaultA_PFCm_1 to _6
# Color index 7–12: DefaultB_PFCd_1 to _6
LH_PARCELS = [
    (161, 1,  'A_PFCm_1'),
    (162, 2,  'A_PFCm_2'),
    (163, 3,  'A_PFCm_3'),
    (164, 4,  'A_PFCm_4'),
    (165, 5,  'A_PFCm_5'),
    (166, 6,  'A_PFCm_6'),
    (175, 7,  'B_PFCd_1'),
    (176, 8,  'B_PFCd_2'),
    (177, 9,  'B_PFCd_3'),
    (178, 10, 'B_PFCd_4'),
    (179, 11, 'B_PFCd_5'),
    (180, 12, 'B_PFCd_6'),
]

RH_PARCELS = [
    (168, 1,  'A_PFCm_1'),
    (169, 2,  'A_PFCm_2'),
    (170, 3,  'A_PFCm_3'),
    (171, 4,  'A_PFCm_4'),
    (172, 5,  'A_PFCm_5'),
    (177, 7,  'B_PFCd_1'),
    (178, 8,  'B_PFCd_2'),
    (179, 9,  'B_PFCd_3'),
    (180, 10, 'B_PFCd_4'),
    (181, 11, 'B_PFCd_5'),
]

N_COLORS = 12

# 12 visually distinct colors: 6 reds/oranges for Group A, 6 blues/purples for Group B
COLORS = [
    '#d73027',  # A_PFCm_1  deep red
    '#f46d43',  # A_PFCm_2  orange-red
    '#fdae61',  # A_PFCm_3  orange
    '#fee090',  # A_PFCm_4  light orange
    '#a50026',  # A_PFCm_5  dark red
    '#fc8d59',  # A_PFCm_6  salmon
    '#4575b4',  # B_PFCd_1  deep blue
    '#74add1',  # B_PFCd_2  medium blue
    '#abd9e9',  # B_PFCd_3  light blue
    '#313695',  # B_PFCd_4  navy
    '#7b2d8b',  # B_PFCd_5  purple
    '#b15cc4',  # B_PFCd_6  light purple (LH only)
]


def build_texture(parcel_list, annot_path, hemi_label):
    labels_arr, _, names = fs.read_annot(str(annot_path))
    names = [n.decode() if hasattr(n, 'decode') else n for n in names]

    texture = np.zeros(len(labels_arr), dtype=float)
    for annot_id, color_idx, label in parcel_list:
        texture[labels_arr == annot_id] = color_idx
        print(f'  {hemi_label} [{annot_id}] {names[annot_id]} → color {color_idx} ({label})')

    return texture


def main():
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    print('LH parcels:')
    tex_lh = build_texture(LH_PARCELS, LH_ANNOT, 'LH')
    print('RH parcels:')
    tex_rh = build_texture(RH_PARCELS, RH_ANNOT, 'RH')

    cmap = ListedColormap(COLORS)  # 12 entries, linearly mapped vmin–vmax

    fig, axes = plt.subplots(2, 2, figsize=(10, 9),
                             subplot_kw={'projection': '3d'})
    fig.suptitle('dmPFC Parcels — Schaefer 400 (17-network)',
                 fontsize=13, fontweight='bold')

    panels = [
        (axes[0, 0], fsavg.infl_left,  fsavg.sulc_left,  tex_lh, 'left',  'medial',  'LH Medial'),
        (axes[0, 1], fsavg.infl_left,  fsavg.sulc_left,  tex_lh, 'left',  'lateral', 'LH Lateral'),
        (axes[1, 0], fsavg.infl_right, fsavg.sulc_right, tex_rh, 'right', 'medial',  'RH Medial'),
        (axes[1, 1], fsavg.infl_right, fsavg.sulc_right, tex_rh, 'right', 'lateral', 'RH Lateral'),
    ]

    for ax, mesh, bg, tex, hemi, view, title in panels:
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
            vmax=N_COLORS + 0.5,
            threshold=0.5,
            bg_on_data=True,
            darkness=0.5,
        )
        ax.set_title(title, fontsize=11, fontweight='bold', pad=2)

    # Legend: one patch per parcel, noting LH-only where applicable
    patches = []
    all_parcels = {(ci, lbl) for _, ci, lbl in LH_PARCELS + RH_PARCELS}
    lh_only = {ci for _, ci, _ in LH_PARCELS} - {ci for _, ci, _ in RH_PARCELS}
    for color_idx in range(1, N_COLORS + 1):
        label = next((lbl for _, ci, lbl in LH_PARCELS if ci == color_idx), None)
        if label is None:
            continue
        suffix = ' (LH only)' if color_idx in lh_only else ''
        patches.append(mpatches.Patch(color=COLORS[color_idx - 1],
                                      label=f'{label}{suffix}'))

    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.subplots_adjust(bottom=0.10, hspace=0.05, wspace=0.0)

    out_dir = FIGS_DIR / 'schaefer_rois'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'dmpfc_parcels_surface.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
