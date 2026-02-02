#!/usr/bin/env python3
"""Plot inter-run correlations (IRC) for sceneprf data."""

from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.plotting import plot_surf_stat_map
from nilearn.signal import clean
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "srcs"))
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR


def find_sceneprf_runs(subject, hemi):
    """Find all sceneprf runs for a subject and hemisphere."""
    pattern = f"{subject}/**/func/*task-sceneprf*run-*hemi-{hemi}*fsaverage6*.func.gii"
    runs = sorted(DERIVATIVES_DIR.glob(pattern), 
                  key=lambda p: int(re.search(r'run-(\d+)', p.name).group(1)))
    return runs


def load_runs(run_paths):
    """Load and preprocess runs: trim to 248s, high-pass filter, z-score."""
    n_trs = int(248 / TR)  # Keep first 248 seconds
    
    # Load and trim
    runs = []
    for path in run_paths:
        data = np.vstack([da.data for da in nib.load(path).darrays])
        runs.append(data[:n_trs])
    
    # Match lengths
    min_t = min(r.shape[0] for r in runs)
    runs = [r[:min_t] for r in runs]
    
    # Clean each run
    return [clean(r, t_r=TR, detrend=True, standardize='zscore_sample', 
                  high_pass=0.01) for r in runs]


def compute_irc(runs):
    """Compute inter-run correlation: each run vs mean of others."""
    Z = np.stack(runs)  # (n_runs, time, vertices)
    n = len(runs)
    
    # Correlation of each run with mean of all (vectorized)
    mean_all = Z.mean(axis=0)
    dot_mean = np.mean(Z * mean_all, axis=1)  # (n_runs, vertices)
    dot_self = np.mean(Z * Z, axis=1)
    
    corr = (n * dot_mean - dot_self) / (n - 1)
    return corr.mean(axis=0)  # Average across runs


def plot_subject_irc(subject, vmax=0.5):
    """Create IRC surface plots for a subject."""
    fs6 = datasets.fetch_surf_fsaverage('fsaverage6')
    
    # Compute IRC for each hemisphere
    irc_data = {}
    for hemi in ['L', 'R']:
        runs = find_sceneprf_runs(subject, hemi)
        if runs:
            print(f"{subject} hemi-{hemi}: {len(runs)} runs")
            cleaned = load_runs(runs)
            irc_data[hemi] = compute_irc(cleaned)
    
    if not irc_data:
        print(f"No sceneprf data found for {subject}")
        return None
    
    # Plot setup
    views = [
        ("L", fs6.infl_left, fs6.sulc_left, "left", "lateral"),
        ("L", fs6.infl_left, fs6.sulc_left, "left", "medial"),
        ("R", fs6.infl_right, fs6.sulc_right, "right", "lateral"),
        ("R", fs6.infl_right, fs6.sulc_right, "right", "medial"),
    ]
    from matplotlib.colors import LinearSegmentedColormap

    # make a colormap with only the positive (red→white) half of seismic
    seismic_pos = LinearSegmentedColormap.from_list(
        "seismic_pos",
        plt.cm.seismic(np.linspace(0.5, 1, 256))
    )
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={"projection": "3d"})
    
    # Plot each view
    for ax, (hemi, mesh, sulc, hemi_name, view) in zip(axes, views):
        if hemi in irc_data:
            plot_surf_stat_map(mesh, irc_data[hemi], hemi=hemi_name, bg_map=sulc,
                              cmap='inferno', darkness=None, vmin=0, vmax=vmax, threshold=0.05, colorbar=False,
                              axes=ax, view=view)
            ax.set_title(f"{hemi_name} {view}", fontsize=10)
        else:
            ax.set_visible(False)
    
    # Add colorbar and title
    sm = plt.cm.ScalarMappable(cmap='inferno', 
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, 
                 label="Inter-run correlation")
    fig.suptitle(f"{subject} sceneprf IRC", fontsize=12)
    
    # Save figure
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGS_DIR / f"{subject}_sceneprf_irc.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_irc.py <subject_id>")
        sys.exit(1)
    
    subject = sys.argv[1]
    
    fig = plot_subject_irc(subject, vmax=0.5)
    if fig:
        plt.show()