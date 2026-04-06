"""
Plot ROI outlines (black contour) on fsaverage6 inflated surface.

Generates one figure per ROI in schaefer_rois.ALL_ROIS.
Output: figs/schaefer_rois/<key>_outline.png
"""
import warnings
from pathlib import Path

import numpy as np
import nibabel.freesurfer as fs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn import datasets
from nilearn.plotting import plot_surf, plot_surf_contours

from configs.config import FIGS_DIR
from configs.schaefer_rois import ALL_ROIS

ANNOT_DIR = Path('/home/zli230/nilearn_data/schaefer_2018')
LH_ANNOT  = ANNOT_DIR / 'lh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
RH_ANNOT  = ANNOT_DIR / 'rh.Schaefer2018_400Parcels_17Networks_order_fsaverage6.annot'
OUTPUT_DIR = FIGS_DIR / 'schaefer_rois'

TITLE_FS = 13


def _plot_roi(roi_def, key, fsavg, surf_l, surf_r):
    name = roi_def['name']
    lh_ids = roi_def.get('left', [])
    rh_ids = roi_def.get('right', [])

    mask_l = np.isin(surf_l, lh_ids).astype(int)
    mask_r = np.isin(surf_r, rh_ids).astype(int)
    print(f'  {name}: {mask_l.sum()} L + {mask_r.sum()} R vertices')

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(name, fontsize=TITLE_FS, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 2, wspace=0.01, hspace=0.01,
                           top=0.92, bottom=0.05, left=0.02, right=0.98)

    panels = [
        (0, 0, fsavg['infl_left'],  fsavg['sulc_left'],  mask_l, 'left',  'lateral'),
        (0, 1, fsavg['infl_right'], fsavg['sulc_right'], mask_r, 'right', 'lateral'),
        (1, 0, fsavg['infl_left'],  fsavg['sulc_left'],  mask_l, 'left',  'medial'),
        (1, 1, fsavg['infl_right'], fsavg['sulc_right'], mask_r, 'right', 'medial'),
    ]

    for row, col, mesh, sulc, mask, hemi, view in panels:
        ax = fig.add_subplot(gs[row, col], projection='3d')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_surf(
                mesh, surf_map=sulc, hemi=hemi, view=view,
                bg_map=sulc, axes=ax, colorbar=False,
                cmap='gray', bg_on_data=True, darkness=0.5,
            )
            if mask.any():
                plot_surf_contours(
                    mesh, roi_map=mask, hemi=hemi,
                    levels=[1], colors=['black'], axes=ax,
                )

    out = OUTPUT_DIR / f'{key}_outline.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', nargs='+', metavar='KEY',
                        help=f'ROI key(s) to plot (choices: {", ".join(ALL_ROIS)}). '
                             'Defaults to all ROIs.')
    args = parser.parse_args()

    if args.roi:
        unknown = [k for k in args.roi if k not in ALL_ROIS]
        if unknown:
            parser.error(f'Unknown ROI key(s): {unknown}. '
                         f'Valid keys: {list(ALL_ROIS)}')
        rois_to_plot = {k: ALL_ROIS[k] for k in args.roi}
    else:
        rois_to_plot = ALL_ROIS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Fetching fsaverage6...')
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')

    print('Reading Schaefer annot files...')
    lh_labels, _, _ = fs.read_annot(str(LH_ANNOT))
    rh_labels, _, _ = fs.read_annot(str(RH_ANNOT))
    surf_l = lh_labels.astype(int)
    surf_r = np.where(rh_labels > 0, rh_labels + 200, 0).astype(int)

    for key, roi_def in rois_to_plot.items():
        print(f'\n[{key}]')
        _plot_roi(roi_def, key, fsavg, surf_l, surf_r)

    print('\nDone.')


if __name__ == '__main__':
    main()
