"""
Filmfest Auditory Cortex Lag Correlation

Validates fMRI-stimulus alignment by computing the cross-correlation between
the audio envelope of movie stimuli and the auditory cortex BOLD signal.

Expects a peak at ~4-6s lag (hemodynamic delay) if data are well-aligned.

Usage:
    python filmfest_auditory_lag.py
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, resample
from scipy.stats import zscore as sp_zscore
from scipy.io import wavfile
from nilearn import datasets, surface

# === CONFIG ===
from configs.config import DERIVATIVES_DIR, FIGS_DIR, TR, FILMFEST_SUBJECTS
from fmrianalysis.utils import load_surface_data

ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
OUTPUT_DIR = FIGS_DIR / 'filmfest_auditory_lag'

MP4_FILES = {
    'filmfest1': ANNOTATIONS_DIR / 'FilmFest_part1.mp4',
    'filmfest2': ANNOTATIONS_DIR / 'FilmFest_part2.mp4',
}

MOVIE_INFO = [
    {'id': 1,  'file': 'FilmFest_01_CMIYC_Segments.xlsx',          'task': 'filmfest1', 'name': 'CMIYC'},
    {'id': 2,  'file': 'FilmFest_02_The_Record_Segments.xlsx',     'task': 'filmfest1', 'name': 'The Record'},
    {'id': 3,  'file': 'FilmFest_03_The_Boyfriend_Segments.xlsx',  'task': 'filmfest1', 'name': 'The Boyfriend'},
    {'id': 4,  'file': 'FilmFest_04_The_Shoe_Segments.xlsx',       'task': 'filmfest1', 'name': 'The Shoe'},
    {'id': 5,  'file': 'FilmFest_05_Keith_Reynolds_Segments.xlsx', 'task': 'filmfest1', 'name': 'Keith Reynolds'},
    {'id': 6,  'file': 'FilmFest_06_The_Rock_Segments.xlsx',       'task': 'filmfest2', 'name': 'The Rock'},
    {'id': 7,  'file': 'FilmFest_07_The_Prisoner_Segments.xlsx',   'task': 'filmfest2', 'name': 'The Prisoner'},
    {'id': 8,  'file': 'FilmFest_08_The_Black_Hole_Segments.xlsx', 'task': 'filmfest2', 'name': 'The Black Hole'},
    {'id': 9,  'file': 'FilmFest_09_Post-it_Love_Segments.xlsx',   'task': 'filmfest2', 'name': 'Post-it Love'},
    {'id': 10, 'file': 'FilmFest_10_Bus_Stop_Segments.xlsx',       'task': 'filmfest2', 'name': 'Bus Stop'},
]

HIGH_PASS_HZ = 0.01
ENVELOPE_LP_HZ = 1.0    # low-pass for audio envelope smoothing
AUDIO_SR = 16000         # resample audio to 16 kHz (enough for envelope)
MAX_LAG_SEC = 15         # max lag in seconds
MAX_LAG_TRS = int(np.ceil(MAX_LAG_SEC / TR))


# === ATLAS SETUP ===
print("Loading atlas...")
FSAVERAGE = datasets.fetch_surf_fsaverage('fsaverage6')
SCHAEFER = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
SCHAEFER_LABELS = [l.decode() if isinstance(l, bytes) else str(l) for l in SCHAEFER['labels']]

# Auditory cortex parcels (SomMotB_Aud)
_aud_ids = [i + 1 for i, l in enumerate(SCHAEFER_LABELS) if 'SomMotB_Aud' in l]
_schaefer_L = surface.vol_to_surf(SCHAEFER['maps'], FSAVERAGE['pial_left'],
                                   interpolation='nearest')
_schaefer_R = surface.vol_to_surf(SCHAEFER['maps'], FSAVERAGE['pial_right'],
                                   interpolation='nearest')
AUD_MASK_L = np.isin(np.round(_schaefer_L).astype(int), _aud_ids)
AUD_MASK_R = np.isin(np.round(_schaefer_R).astype(int), _aud_ids)
print(f"Auditory cortex: {AUD_MASK_L.sum()} L + {AUD_MASK_R.sum()} R vertices")


def mss_to_seconds(mss):
    """Convert m.ss timestamp to seconds."""
    minutes = int(mss)
    seconds = round((mss - minutes) * 100)
    return minutes * 60 + seconds


def get_movie_time_range(movie):
    """Return (start_sec, end_sec) for a movie from annotations."""
    df = pd.read_excel(ANNOTATIONS_DIR / movie['file'])
    start_sec = mss_to_seconds(df['SEG-C Start Time (m.ss)'].min())
    end_sec = mss_to_seconds(df['SEG-C End Time (m.ss)'].max())
    return start_sec, end_sec


def extract_audio(mp4_path):
    """Extract audio from mp4 as mono WAV using ffmpeg. Returns (sample_rate, data)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        'ffmpeg', '-y', '-i', str(mp4_path),
        '-ac', '1',           # mono
        '-ar', str(AUDIO_SR), # resample
        '-vn',                # no video
        tmp_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    sr, data = wavfile.read(tmp_path)
    Path(tmp_path).unlink()

    # Normalize to float
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    else:
        data = data.astype(np.float64)

    return sr, data


def compute_envelope(audio, sr, lp_hz=ENVELOPE_LP_HZ):
    """Compute amplitude envelope using Hilbert transform + low-pass filter."""
    # Hilbert envelope
    analytic = hilbert(audio)
    envelope = np.abs(analytic)

    # Low-pass filter the envelope
    nyq = sr / 2.0
    b, a = butter(4, lp_hz / nyq, btype='low')
    envelope = filtfilt(b, a, envelope)

    return envelope


def downsample_to_tr(envelope, sr, tr=TR):
    """Downsample envelope to TR resolution."""
    samples_per_tr = int(sr * tr)
    n_trs = len(envelope) // samples_per_tr
    # Average within each TR bin
    trimmed = envelope[:n_trs * samples_per_tr]
    return trimmed.reshape(n_trs, samples_per_tr).mean(axis=1)


def highpass_filter(data, cutoff=HIGH_PASS_HZ, tr=TR, order=5):
    """Butterworth high-pass filter along time axis (axis=0)."""
    nyq = 1.0 / (2.0 * tr)
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, data, axis=0)


def lag_correlation(bold, envelope, max_lag_trs=MAX_LAG_TRS):
    """Compute correlation between BOLD and envelope at different lags.

    Positive lag = BOLD lags behind envelope (expected due to HRF).

    Returns
    -------
    lags : array of lag values in TRs
    corrs : array of correlation values
    """
    lags = np.arange(-max_lag_trs, max_lag_trs + 1)
    corrs = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            b = bold[lag:]
            e = envelope[:len(b)]
        else:
            e = envelope[-lag:]
            b = bold[:len(e)]

        n = min(len(b), len(e))
        b = b[:n]
        e = e[:n]

        # z-score and correlate
        b = b - b.mean()
        e = e - e.mean()
        denom = np.sqrt((b ** 2).sum() * (e ** 2).sum())
        if denom > 0:
            corrs[i] = (b * e).sum() / denom

    return lags, corrs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Extract audio envelopes ---
    envelopes_tr = {}
    for task, mp4_path in MP4_FILES.items():
        print(f"\nExtracting audio from {mp4_path.name} ...")
        sr, audio = extract_audio(mp4_path)
        print(f"  Audio: {len(audio)} samples, {len(audio)/sr:.1f}s at {sr} Hz")
        envelope = compute_envelope(audio, sr)
        env_tr = downsample_to_tr(envelope, sr)
        envelopes_tr[task] = env_tr
        print(f"  Envelope downsampled to {len(env_tr)} TRs")

    # --- Load fMRI and compute lag correlations ---
    # Per-movie results
    all_movie_results = []

    # Per-run (whole run) results
    all_run_results = []

    for subject, session in FILMFEST_SUBJECTS.items():
        for task in ('filmfest1', 'filmfest2'):
            print(f"\n  {subject} {session} {task} ...")

            # Load surface data
            ts_L = load_surface_data(subject, session, task, 'L',
                                     data_dir=DERIVATIVES_DIR).astype(np.float64).T  # (T, V)
            ts_R = load_surface_data(subject, session, task, 'R',
                                     data_dir=DERIVATIVES_DIR).astype(np.float64).T

            # Extract auditory cortex mean time series
            aud_ts = np.column_stack([
                ts_L[:, AUD_MASK_L],
                ts_R[:, AUD_MASK_R],
            ]).mean(axis=1)  # (T,)

            # High-pass filter and z-score
            aud_ts = highpass_filter(aud_ts.reshape(-1, 1)).ravel()
            aud_ts = sp_zscore(aud_ts, nan_policy='omit')
            aud_ts = np.nan_to_num(aud_ts, nan=0.0)

            n_trs_bold = len(aud_ts)
            env = envelopes_tr[task]

            # Whole-run lag correlation
            n_common = min(n_trs_bold, len(env))
            env_run = sp_zscore(env[:n_common])
            bold_run = aud_ts[:n_common]

            lags, corrs = lag_correlation(bold_run, env_run)
            all_run_results.append({
                'subject': subject,
                'task': task,
                'lags': lags,
                'corrs': corrs,
            })

            # Per-movie lag correlations
            movies_this_task = [m for m in MOVIE_INFO if m['task'] == task]
            for movie in movies_this_task:
                start_sec, end_sec = get_movie_time_range(movie)
                start_tr = int(np.floor(start_sec / TR))
                end_tr = min(int(np.ceil(end_sec / TR)), n_trs_bold, len(env))

                if end_tr - start_tr < 20:  # skip very short segments
                    continue

                bold_seg = aud_ts[start_tr:end_tr]
                env_seg = sp_zscore(env[start_tr:end_tr])

                lags_m, corrs_m = lag_correlation(bold_seg, env_seg)
                all_movie_results.append({
                    'subject': subject,
                    'movie_id': movie['id'],
                    'movie_name': movie['name'],
                    'lags': lags_m,
                    'corrs': corrs_m,
                })

    # === PLOTTING ===

    # --- 1. Whole-run lag correlation per subject ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('Auditory Cortex – Audio Envelope Lag Correlation', fontsize=14)

    for ax, task in zip(axes, ['filmfest1', 'filmfest2']):
        task_results = [r for r in all_run_results if r['task'] == task]
        for r in task_results:
            lag_sec = r['lags'] * TR
            ax.plot(lag_sec, r['corrs'], alpha=0.4, linewidth=1)

        # Group mean
        all_corrs = np.array([r['corrs'] for r in task_results])
        mean_corr = all_corrs.mean(axis=0)
        sem_corr = all_corrs.std(axis=0) / np.sqrt(len(task_results))
        lag_sec = task_results[0]['lags'] * TR

        ax.plot(lag_sec, mean_corr, 'k-', linewidth=2.5, label='Mean')
        ax.fill_between(lag_sec, mean_corr - sem_corr, mean_corr + sem_corr,
                        color='black', alpha=0.15)

        peak_idx = np.argmax(mean_corr)
        peak_lag = lag_sec[peak_idx]
        peak_r = mean_corr[peak_idx]
        ax.axvline(peak_lag, color='red', linestyle='--', alpha=0.7,
                   label=f'Peak: {peak_lag:.1f}s (r={peak_r:.3f})')

        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Lag (seconds)', fontsize=12)
        ax.set_ylabel('Correlation (r)', fontsize=12)
        ax.set_title(task, fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(-MAX_LAG_SEC, MAX_LAG_SEC)

    plt.tight_layout()
    out = OUTPUT_DIR / 'lag_correlation_wholerun.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out}")

    # --- 2. Per-movie lag correlation (group average) ---
    n_movies = len(MOVIE_INFO)
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
    fig.suptitle('Auditory Cortex – Audio Envelope Lag Correlation per Movie',
                 fontsize=14)

    for idx, movie in enumerate(MOVIE_INFO):
        row, col = divmod(idx, 5)
        ax = axes[row, col]

        movie_results = [r for r in all_movie_results if r['movie_id'] == movie['id']]
        if not movie_results:
            ax.set_title(f'{movie["name"]}\n(no data)', fontsize=10)
            continue

        for r in movie_results:
            lag_sec = r['lags'] * TR
            ax.plot(lag_sec, r['corrs'], alpha=0.3, linewidth=0.8)

        all_corrs = np.array([r['corrs'] for r in movie_results])
        mean_corr = all_corrs.mean(axis=0)
        lag_sec = movie_results[0]['lags'] * TR

        ax.plot(lag_sec, mean_corr, 'k-', linewidth=2)
        peak_idx = np.argmax(mean_corr)
        peak_lag = lag_sec[peak_idx]
        peak_r = mean_corr[peak_idx]

        ax.axvline(peak_lag, color='red', linestyle='--', alpha=0.7)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'{movie["name"]}\npeak={peak_lag:.1f}s, r={peak_r:.3f}',
                     fontsize=10)
        ax.set_xlim(-MAX_LAG_SEC, MAX_LAG_SEC)

        if row == 1:
            ax.set_xlabel('Lag (s)', fontsize=10)
        if col == 0:
            ax.set_ylabel('Correlation (r)', fontsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / 'lag_correlation_per_movie.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    print(f"\nDone. Figures in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
