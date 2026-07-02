"""
Build a TPJ ROI from the Neurosynth "tpj" association-test z map.

Mirrors build_dacc_roi_neurosynth.py, but for the (bilateral, lateral) TPJ.

Recipe:
  - source map : "tpj" association test (FDR q<0.01), fetched from neurosynth.org
                 (api image 1726) -> data/neurosynth/tpj_association-test_z.nii.gz
  - extraction : WHOLE supra-threshold clusters, NO anatomical cropping
  - symmetry   : keep as-is (no L/R mirroring)

The only curation: keep the lateral temporo-parietal connected-components and
drop the off-target components (medial frontal / precuneus / anterior temporal
pole). A TPJ component = peak |x|>40 mm, y<-30 mm, z>0 mm. Components are kept
whole (no erosion / no mask intersection). This isolates the two dominant
bilateral posterior temporo-parietal clusters (R ~ (58,-50,14); L ~ (-60,-54,20)).

Outputs:
  data/neurosynth/tpj_roi_mask.nii.gz   binary ROI, MNI152 2mm
  data/neurosynth/tpj_roi_z.nii.gz      z-valued ROI, MNI152 2mm
  data/neurosynth/tpj_roi_fsavg6.npz    surface masks: lh, rh (bool, 40962)
  figs/neurosynth/tpj_roi.png           ortho + glass + surface views

Run:  uv run python srcs/fmrianalysis/build_tpj_roi_neurosynth.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt
from nilearn import plotting, surface, datasets

from configs.config import FIGS_DIR

OUT_DATA = FIGS_DIR.parent / 'data' / 'neurosynth'
MAP = OUT_DATA / 'tpj_association-test_z.nii.gz'
Z_THR = 3.1


def is_tpj(peak_mni):
    x, y, z = peak_mni
    return abs(x) > 40 and y < -30 and z > 0


def main():
    img = nib.load(str(MAP))
    data = img.get_fdata()
    affine = img.affine

    mask = data > Z_THR
    bin_struct = generate_binary_structure(3, 1)  # 6-conn, faces
    labels, n = label(mask, structure=bin_struct)

    keep = np.zeros_like(mask)
    kept_info = []
    for lab in range(1, n + 1):
        comp = labels == lab
        nv = int(comp.sum())
        zc = np.where(comp, data, -np.inf)
        pv = np.unravel_index(np.argmax(zc), zc.shape)
        pmni = (affine @ np.array([*pv, 1]))[:3]
        if is_tpj(pmni):
            keep |= comp
            kept_info.append((lab, nv, np.round(pmni).astype(int),
                              float(data[pv])))

    n_vox = int(keep.sum())
    vox_vol = float(np.prod(img.header.get_zooms()[:3]))
    print(f'{n} components total; kept {len(kept_info)} lateral temporo-parietal:')
    for lab, nv, pmni, pz in sorted(kept_info, key=lambda r: -r[1]):
        print(f'  comp {lab:3d}: {nv:4d} vox, peak {tuple(pmni)} z={pz:.2f}')
    print(f'TPJ ROI: {n_vox} voxels = {n_vox * vox_vol:.0f} mm3')

    # save volumetric ROI (binary + z-valued)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    bin_img = nib.Nifti1Image(keep.astype(np.uint8), affine)
    z_img = nib.Nifti1Image(np.where(keep, data, 0).astype(np.float32), affine)
    nib.save(bin_img, str(OUT_DATA / 'tpj_roi_mask.nii.gz'))
    nib.save(z_img, str(OUT_DATA / 'tpj_roi_z.nii.gz'))
    print(f'Saved -> {OUT_DATA / "tpj_roi_mask.nii.gz"}')
    print(f'Saved -> {OUT_DATA / "tpj_roi_z.nii.gz"}')

    # project to fsaverage6 surface
    fsavg = datasets.fetch_surf_fsaverage('fsaverage6')
    surf_masks = {}
    for hemi, pial, white in [('lh', fsavg.pial_left, fsavg.white_left),
                              ('rh', fsavg.pial_right, fsavg.white_right)]:
        tex = surface.vol_to_surf(bin_img, pial, inner_mesh=white,
                                  interpolation='linear', radius=3.0)
        surf_masks[hemi] = np.nan_to_num(tex) > 0.5
    np.savez(OUT_DATA / 'tpj_roi_fsavg6.npz',
             lh=surf_masks['lh'], rh=surf_masks['rh'])
    print(f'Saved -> {OUT_DATA / "tpj_roi_fsavg6.npz"}  '
          f'(lh {surf_masks["lh"].sum()}, rh {surf_masks["rh"].sum()} verts)')

    # figure: ortho + glass + surface (lateral views)
    out_dir = FIGS_DIR / 'neurosynth'
    out_dir.mkdir(parents=True, exist_ok=True)
    peak = tuple(sorted(kept_info, key=lambda r: -r[1])[0][2])

    fig = plt.figure(figsize=(14, 10))
    ax_o = fig.add_subplot(3, 1, 1)
    plotting.plot_stat_map(z_img, threshold=Z_THR, cut_coords=peak,
                           display_mode='ortho', cmap='hot', colorbar=True,
                           figure=fig, axes=ax_o, black_bg=False,
                           title=f'TPJ ROI (Neurosynth "tpj", whole clusters) '
                                 f'— {n_vox * vox_vol:.0f} mm3')
    ax_g = fig.add_subplot(3, 1, 2)
    plotting.plot_glass_brain(z_img, threshold=Z_THR, colorbar=True,
                              cmap='hot_r', plot_abs=False, display_mode='lyrz',
                              figure=fig, axes=ax_g, title='TPJ ROI — glass brain')
    ax_l = fig.add_subplot(3, 2, 5, projection='3d')
    ax_r = fig.add_subplot(3, 2, 6, projection='3d')
    for ax, hemi, infl, sulc in [(ax_l, 'left', fsavg.infl_left, fsavg.sulc_left),
                                 (ax_r, 'right', fsavg.infl_right, fsavg.sulc_right)]:
        m = surf_masks['lh' if hemi == 'left' else 'rh'].astype(float)
        plotting.plot_surf_stat_map(
            surf_mesh=infl, stat_map=m, hemi=hemi, view='lateral', bg_map=sulc,
            axes=ax, colorbar=False, cmap='autumn', threshold=0.5, vmin=0.5,
            vmax=1.5, bg_on_data=True, darkness=0.5, title=f'{hemi} lateral')
    fig.suptitle('Neurosynth "tpj" TPJ ROI (whole clusters, no crop, as-is)',
                 fontsize=13, fontweight='bold')
    plt.subplots_adjust(top=0.93, hspace=0.25, wspace=0.0)
    p = out_dir / 'tpj_roi.png'
    fig.savefig(p, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {p}')


if __name__ == '__main__':
    main()
