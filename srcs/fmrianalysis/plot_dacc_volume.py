"""Visualize dACC ROI parcels in volumetric (MNI) space."""
from pathlib import Path
import numpy as np
from nilearn import datasets, plotting, image

from configs.config import FIGS_DIR
from configs.schaefer_rois import DACC

OUTPUT_DIR = FIGS_DIR / 'schaefer_rois'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
atlas_img = image.load_img(atlas['maps'])
atlas_data = atlas_img.get_fdata()

all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
schaefer_labels = [l for l in all_labels if l != 'Background']

dacc_ids = DACC['left'] + DACC['right']
dacc_labels = DACC['left_labels'] + DACC['right_labels']

# Verify IDs match labels
for pid, lbl in zip(dacc_ids, dacc_labels):
    atlas_label = schaefer_labels[pid - 1]
    status = "OK" if atlas_label == lbl else f"MISMATCH ({atlas_label})"
    print(f"  Parcel {pid:3d}: {lbl}  [{status}]")

mask_data = np.isin(atlas_data, dacc_ids).astype(np.float32)
mask_img = image.new_img_like(atlas_img, mask_data)
print(f"\ndACC voxel count: {int(mask_data.sum())}")

# Glass brain
display = plotting.plot_glass_brain(
    mask_img, display_mode='ortho', colorbar=False,
    title=f'dACC ROI (Schaefer 400, 17-Net)\nParcels: {dacc_ids}',
    plot_abs=False, cmap='Reds', alpha=0.8,
)
out = OUTPUT_DIR / 'dacc_volume_glass_brain.png'
display.savefig(str(out), dpi=200)
print(f"Saved {out}")

# MNI slices
display2 = plotting.plot_roi(
    mask_img, display_mode='ortho', cut_coords=(2, 20, 42),
    title=f'dACC ROI — axial/coronal/sagittal',
    cmap='Reds', alpha=0.7,
)
out2 = OUTPUT_DIR / 'dacc_volume_slices.png'
display2.savefig(str(out2), dpi=200)
print(f"Saved {out2}")
