#!/usr/bin/env python3
"""
plot_segb_boundary_alignment.py

Two-panel figure comparing SEG-B segment onsets against viewer-marked
retrospective boundaries in the filmfest task.

Panel 1 (left):  For each movie, what proportion of strong retained boundaries
                 align with a SEG-B segment onset (±1.5 s tolerance)?

Panel 2 (right): For each SEG-B onset (excluding the first of each movie),
                 what is the best-matching boundary strength
                 (priority: strong > moderate > weak > none)?

Usage:
    uv run python srcs/filmfest/plot_segb_boundary_alignment.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.config import MOVIE_INFO, FIGS_DIR

ANNOTATIONS_DIR = Path('/home/datasets/stateswitch/filmfest_annotations')
BS_CSV          = ANNOTATIONS_DIR / 'filmfest_boundary_strength.csv'
OUTPUT_DIR      = FIGS_DIR / 'filmfest'
TOLERANCE       = 1.5   # seconds
BTYPES          = ('strong', 'moderate', 'weak')
COLORS          = {
    'strong':   '#1a237e',
    'moderate': '#3498db',
    'weak':     '#90caf9',
    'none':     '#e0e0e0',
}
FS = 11


def mss_to_seconds(mss):
    """Convert m.ss timestamp to total seconds."""
    minutes = int(mss)
    seconds = round((float(mss) - minutes) * 100)
    return minutes * 60 + seconds


def load_segb_onsets_within_movie(movie_info):
    """Return within-movie SEG-B onset times in seconds.

    The first segment onset is 0 by definition (movie start), so it is
    excluded when used as a transition boundary.
    """
    df = pd.read_excel(ANNOTATIONS_DIR / movie_info['file'])
    segb = df.dropna(subset=['SEG-B_Number'])
    movie_onset_sec = mss_to_seconds(segb['Start Time (m.ss)'].values[0])
    return np.array([
        mss_to_seconds(t) - movie_onset_sec
        for t in segb['Start Time (m.ss)'].values
    ])


# ============================================================================
# ANALYSIS 1: strong boundaries → SEG-B alignment
# ============================================================================

def compute_strong_to_segb(bs):
    strong_ret = bs[(bs['retained_for_fmri'] == 1) & (bs['boundary_type'] == 'strong')]
    movies, n_total, n_match = [], [], []

    for mi in MOVIE_INFO:
        mid = mi['id']
        movie_strong = strong_ret[strong_ret['movie'] == mid]
        if len(movie_strong) == 0:
            continue
        segb_onsets = load_segb_onsets_within_movie(mi)
        matched = sum(
            np.any(np.abs(segb_onsets - ts) <= TOLERANCE)
            for ts in movie_strong['timestamp_sec'].values
        )
        movies.append(f'M{mid}')
        n_total.append(len(movie_strong))
        n_match.append(matched)

    movies.append('All')
    n_total.append(sum(n_total))
    n_match.append(sum(n_match))
    proportions = [m / n for m, n in zip(n_match, n_total)]
    return movies, n_total, n_match, proportions


# ============================================================================
# ANALYSIS 2: SEG-B onsets → boundary type
# ============================================================================

def compute_segb_to_boundary(bs):
    movies, n_segb = [], []
    counts = {bt: [] for bt in BTYPES}
    counts['none'] = []

    for mi in MOVIE_INFO:
        mid = mi['id']
        segb_onsets = load_segb_onsets_within_movie(mi)[1:]   # exclude first (movie start)
        movie_bs = bs[bs['movie'] == mid]
        btype_ts = {
            bt: movie_bs.loc[movie_bs['boundary_type'] == bt, 'timestamp_sec'].values
            for bt in BTYPES
        }
        c = {bt: 0 for bt in BTYPES}
        c['none'] = 0
        for onset in segb_onsets:
            matched = False
            for bt in BTYPES:
                if np.any(np.abs(btype_ts[bt] - onset) <= TOLERANCE):
                    c[bt] += 1
                    matched = True
                    break
            if not matched:
                c['none'] += 1

        movies.append(f'M{mid}')
        n_segb.append(len(segb_onsets))
        for k in c:
            counts[k].append(c[k])

    movies.append('All')
    n_segb.append(sum(n_segb))
    for k in counts:
        counts[k].append(sum(counts[k]))

    return movies, n_segb, counts


# ============================================================================
# FIGURE
# ============================================================================

def make_figure(bs):
    movies1, n_total, n_match, props1 = compute_strong_to_segb(bs)
    movies2, n_segb, counts2          = compute_segb_to_boundary(bs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

    x1 = np.arange(len(movies1))
    x2 = np.arange(len(movies2))

    # ── Panel 1 ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x1, props1, color='#1a237e', edgecolor='white', width=0.6)

    for i, (prop, matched, total) in enumerate(zip(props1, n_match, n_total)):
        ax.text(i, prop + 0.02, f'{matched}/{total}',
                ha='center', va='bottom', fontsize=8)

    ax.axhline(props1[-1], color='k', ls='--', lw=1, alpha=0.5,
               label=f'Overall {props1[-1]:.0%}')

    ax.set_xticks(x1)
    ax.set_xticklabels(movies1, fontsize=FS - 1)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=FS - 1)
    ax.set_ylabel('Proportion of strong boundaries', fontsize=FS)
    ax.set_title(
        'Strong boundaries aligning with a SEG-B onset (±1.5 s tolerance)\n'
        '$\\it{What\\ proportion\\ of\\ strong\\ boundaries\\ align\\ with\\ a\\ SEG\\text{-}B\\ onset?}$',
        fontsize=FS, fontweight='bold',
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=FS - 2, frameon=False)

    # ── Panel 2 ──────────────────────────────────────────────────────────────
    ax = axes[1]
    bottom = np.zeros(len(movies2))
    handles = []
    for bt in ('strong', 'moderate', 'weak', 'none'):
        label = bt.capitalize() if bt != 'none' else 'No boundary'
        props = np.array(counts2[bt]) / np.array(n_segb)
        bar = ax.bar(x2, props, bottom=bottom, color=COLORS[bt],
                     edgecolor='white', width=0.6, label=label)
        handles.append(bar)
        for i, (p, b) in enumerate(zip(props, bottom)):
            if p > 0.08:
                ax.text(i, b + p / 2, f'{p:.0%}',
                        ha='center', va='center', fontsize=7.5, fontweight='bold',
                        color='white' if bt in ('strong', 'none') else 'black')
        bottom += props

    ax.set_xticks(x2)
    ax.set_xticklabels(movies2, fontsize=FS - 1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=FS - 1)
    ax.set_ylabel('Proportion of SEG-B onsets', fontsize=FS)
    ax.set_title(
        'SEG-B onsets by best-matching boundary strength\n'
        '(±1.5 s tolerance, priority: strong > moderate > weak)\n'
        '$\\it{Do\\ viewers\\ mark\\ SEG\\text{-}B\\ onsets\\ as\\ boundaries?}$',
        fontsize=FS, fontweight='bold',
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(ncols=4, fontsize=FS - 2, frameon=False,
              loc='lower center', bbox_to_anchor=(0.5, -0.18))

    fig.tight_layout()
    return fig


# ============================================================================
# TRANSITIONS TABLE FIGURE
# ============================================================================

import textwrap

MAX_CHARS       = 50   # wrap descriptions longer than this (prev/next columns)
MAX_CHARS_WITHIN = 90  # within column spans both description columns

def _wrap(text, max_chars=MAX_CHARS):
    """Return text with a newline inserted if it exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    lines = textwrap.wrap(text, width=max_chars)
    return '\n'.join(lines[:2])   # at most two lines


def _movie_name(filename):
    return filename.replace('.xlsx', '').replace('_Segments', '').split('_', 2)[-1].replace('_', ' ')


def collect_transitions(bs):
    """Return all strong retained boundaries, with SEG-B context.

    For boundaries matching a SEG-B onset: has_match=True, prev and next desc.
    For boundaries not matching: has_match=False, within_desc (containing SEG-B).
    Rows ordered by movie then time.
    """
    strong_ret = bs[(bs['retained_for_fmri'] == 1) & (bs['boundary_type'] == 'strong')]
    rows = []

    for mi in MOVIE_INFO:
        mid = mi['id']
        movie_strong = strong_ret[strong_ret['movie'] == mid]
        if len(movie_strong) == 0:
            continue

        df = pd.read_excel(ANNOTATIONS_DIR / mi['file'])
        segb = df.dropna(subset=['SEG-B_Number']).reset_index(drop=True)
        movie_onset_sec = mss_to_seconds(segb['Start Time (m.ss)'].values[0])
        segb_onsets = np.array([
            mss_to_seconds(t) - movie_onset_sec
            for t in segb['Start Time (m.ss)'].values
        ])
        descriptions = segb['SEG-B Description'].values
        movie_name = _movie_name(mi['file'])

        for ts in sorted(movie_strong['timestamp_sec'].values):
            diffs = np.abs(segb_onsets - ts)
            closest = np.argmin(diffs)
            t_cum = movie_onset_sec + ts
            time_mmss = f'{int(t_cum // 60):02d}:{int(t_cum % 60):02d}'

            if diffs[closest] <= TOLERANCE:
                prev_desc = str(descriptions[closest - 1]) if closest > 0 else '—'
                next_desc = str(descriptions[closest])
                rows.append({
                    'movie_id':   mid,
                    'movie_name': movie_name,
                    'time_s':     ts,
                    'time_mmss':  time_mmss,
                    'has_match':  True,
                    'prev':       _wrap(prev_desc),
                    'next':       _wrap(next_desc),
                    'within':     None,
                })
            else:
                # Find the containing SEG-B segment
                within_idx = max(0, np.searchsorted(segb_onsets, ts, side='right') - 1)
                within_desc = str(descriptions[within_idx])
                rows.append({
                    'movie_id':   mid,
                    'movie_name': movie_name,
                    'time_s':     ts,
                    'time_mmss':  time_mmss,
                    'has_match':  False,
                    'prev':       None,
                    'next':       None,
                    'within':     _wrap(within_desc, MAX_CHARS_WITHIN),
                })
    return rows


def make_transitions_figure(bs):
    """Draw a styled table of all strong retained boundaries with SEG-B context."""
    import textwrap
    from itertools import groupby

    rows = collect_transitions(bs)

    # Layout constants
    ROW_H_1  = 0.38   # single-line row height (inches)
    ROW_H_2  = 0.55   # two-line row height
    HDR_H    = 0.45
    COL_HDR_H = 0.35
    PAD      = 0.15
    FIG_W    = 13.0

    def row_h(row):
        texts = [row.get('prev') or '', row.get('next') or '', row.get('within') or '']
        return ROW_H_2 if any('\n' in t for t in texts) else ROW_H_1

    groups = [(mid, list(g)) for mid, g in groupby(rows, key=lambda r: r['movie_id'])]
    total_h = (PAD + COL_HDR_H
               + sum(HDR_H + sum(row_h(r) for r in g) for _, g in groups)
               + PAD)

    fig, ax = plt.subplots(figsize=(FIG_W, total_h), facecolor='white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    X_TIME_MMSS = 0.03
    X_TIME_S    = 0.12
    X_PREV      = 0.21
    X_ARR       = 0.585   # midpoint of (0.21, 0.97): each col = 0.36 wide
    X_NEXT      = 0.61
    X_WITHIN    = 0.21   # same start as prev; spans both cols for non-matching rows

    HDR_COLOR  = '#1a237e'
    ROW_COLORS = ['#f5f5f5', 'white']
    FS_HDR     = 10
    FS_ROW     = 9

    y = total_h - PAD

    # Column headers
    for x, label in [
        (X_TIME_MMSS, 'Time (mm:ss)'),
        (X_TIME_S,    'Movie time (s)'),
        (X_PREV,      'Previous SEG-B / Containing SEG-B'),
        (X_NEXT,      'Next SEG-B'),
    ]:
        ax.text(x, y, label, fontsize=FS_ROW, fontweight='bold',
                va='top', ha='left', color='#444444')
    y -= COL_HDR_H

    for _, group_rows in groups:
        mid = group_rows[0]['movie_id']
        movie_name = group_rows[0]['movie_name']
        n_match = sum(r['has_match'] for r in group_rows)

        # Movie header bar
        ax.add_patch(plt.Rectangle((0, y - HDR_H), 1, HDR_H,
                                   color=HDR_COLOR, zorder=1))
        ax.text(0.01, y - HDR_H / 2, f'Movie {mid}  —  {movie_name}',
                fontsize=FS_HDR, fontweight='bold', color='white',
                va='center', ha='left', zorder=2)
        ax.text(0.96, y - HDR_H / 2,
                f'{n_match}/{len(group_rows)} match SEG-B onset',
                fontsize=FS_HDR - 1, color='#b0bec5',
                va='center', ha='right', zorder=2)
        y -= HDR_H

        for ri, row in enumerate(group_rows):
            rh = row_h(row)
            bg = ROW_COLORS[ri % 2]
            ax.add_patch(plt.Rectangle((0, y - rh), 1, rh, color=bg, zorder=1))
            cy = y - rh / 2

            t_color = HDR_COLOR if row['has_match'] else '#555555'
            t_weight = 'bold' if row['has_match'] else 'normal'
            ax.text(X_TIME_MMSS, cy, row['time_mmss'],
                    fontsize=FS_ROW, va='center', ha='left',
                    color=t_color, fontweight=t_weight, zorder=2)
            ax.text(X_TIME_S, cy, f'{row["time_s"]:.1f}',
                    fontsize=FS_ROW, va='center', ha='left',
                    color=t_color, fontweight=t_weight, zorder=2)

            if row['has_match']:
                ax.text(X_PREV, cy, row['prev'],
                        fontsize=FS_ROW, va='center', ha='left',
                        color=HDR_COLOR, fontweight='bold', zorder=2, linespacing=1.4)
                ax.text(X_ARR, cy, '→',
                        fontsize=FS_ROW, va='center', ha='center',
                        color='#888888', zorder=2)
                ax.text(X_NEXT, cy, row['next'],
                        fontsize=FS_ROW, va='center', ha='left',
                        color=HDR_COLOR, fontweight='bold', zorder=2, linespacing=1.4)
            else:
                ax.text(X_WITHIN, cy, row['within'],
                        fontsize=FS_ROW, va='center', ha='left',
                        color='#333333', zorder=2, linespacing=1.4)

            y -= rh

    ax.set_title(
        'Strong boundaries aligned with SEG-B onsets: narrative transition context\n'
        '$\\it{What\\ content\\ transition\\ does\\ each\\ strong\\ boundary\\ capture?}$',
        fontsize=FS_HDR + 1, fontweight='bold', pad=10,
    )
    fig.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bs = pd.read_csv(BS_CSV)
    fig = make_figure(bs)
    out = OUTPUT_DIR / 'filmfest_segb_boundary_alignment.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")


    fig2 = make_transitions_figure(bs)
    out2 = OUTPUT_DIR / 'filmfest_segb_strong_boundary_transitions.png'
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved → {out2}")


if __name__ == '__main__':
    main()
