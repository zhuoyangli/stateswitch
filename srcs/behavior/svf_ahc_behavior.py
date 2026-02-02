#!/usr/bin/env python3
"""
Combined Behavioral Analysis for SVF and AHC Tasks

Analyzes response times and semantic distance at boundaries/switches:
1. SVF Task: Inter-response times (IRT) for Switch vs Cluster words
2. AHC Task: Between-sentence intervals for Within vs Between possibilities  
3. SVF Semantic Analysis: GPT-2 static embedding distance
4. AHC Semantic Analysis: Universal Sentence Encoder (USE) embedding distance

All metrics are z-scored within subject. RT data is log-transformed.

Usage:
    python svf_ahc_behavior.py --all                    # Run all analyses
    python svf_ahc_behavior.py --svf-rt                 # SVF IRT analysis only
    python svf_ahc_behavior.py --ahc-rt                 # AHC RT analysis only
    python svf_ahc_behavior.py --svf-semantic           # SVF semantic analysis only
    python svf_ahc_behavior.py --ahc-semantic           # AHC semantic analysis only
    python svf_ahc_behavior.py --merged                 # Create merged RT+semantic plots
    python svf_ahc_behavior.py --summary                # Output summary statistics only
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel, zscore
from scipy.spatial.distance import cosine

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# === PROJECT CONFIG ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR
except ImportError:
    print("Warning: Could not import configs.config, using defaults")
    DATA_DIR = Path("./data")
    DERIVATIVES_DIR = Path("./derivatives")
    FIGS_DIR = Path("./figs")

# === PATH DEFINITIONS ===
SVF_ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
AHC_ANNOTATIONS_DIR = DATA_DIR / "rec/ahc_sentences"

BEHAVIOR_FIGS_DIR = FIGS_DIR / "behavior"
BEHAVIOR_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === STYLE CONSTANTS ===
COLORS = {
    'switch': '#e74c3c',      # Red for switch/boundary
    'cluster': 'gray',        # Gray for cluster/within
    'boundary': '#e74c3c',
    'nonboundary': 'gray',
}

LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
TICK_FONTSIZE = 10

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Subject color palette
SUBJECT_COLORS = {
    'sub-001': '#1f77b4',  # blue
    'sub-003': '#ff7f0e',  # orange
    'sub-004': '#2ca02c',  # green
    'sub-006': '#d62728',  # red
    'sub-007': '#9467bd',  # purple
    'sub-008': '#8c564b',  # brown
    'sub-009': '#e377c2'   # pink
}

def get_subject_label(subject):
    """Convert sub-001 to S01, sub-003 to S03, etc."""
    num = subject.replace('sub-', '')
    return f"S{int(num):02d}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_significance_text(p_value):
    """Return formatted p-value text."""
    if p_value < 0.001:
        return 'p < .001'
    else:
        return f'p = {p_value:.3f}'


def add_jitter(values, jitter_amount=0.15):
    """Add horizontal jitter to values for scatter plot."""
    return np.random.uniform(-jitter_amount, jitter_amount, size=len(values))


def plot_comparison_scatter(ax, data1, data2, label1, label2, p_value, 
                            subjects=None, ylabel='', 
                            color1='gray', color2='#e74c3c', 
                            show_zero_line=True, log_scale_rt=False,
                            show_legend=True):
    """
    Create a scatter plot comparing two conditions with session-level points.
    
    Parameters:
    -----------
    ax : matplotlib axis
    data1, data2 : arrays of session-level values
    label1, label2 : condition labels
    p_value : p-value for annotation
    subjects : list of subject IDs for each session (for color coding)
    ylabel : y-axis label
    color1, color2 : default colors if subjects not provided
    show_zero_line : whether to show horizontal line at y=0
    log_scale_rt : if True, data is log(RT) but display RT values on y-axis
    show_legend : whether to show the legend on this subplot
    """
    # X positions
    x1, x2 = 0, 1
    
    # Add jittered scatter points
    jitter1 = add_jitter(data1)
    jitter2 = add_jitter(data2)
    
    if subjects is not None:
        # Color by subject
        unique_subjects = sorted(set(subjects))
        
        for subj in unique_subjects:
            mask = np.array([s == subj for s in subjects])
            color = SUBJECT_COLORS.get(subj, 'gray')
            
            ax.scatter(x1 + jitter1[mask], data1[mask], c=color, alpha=0.7, s=40, 
                      edgecolors='white', linewidths=0.5, label=get_subject_label(subj))
            ax.scatter(x2 + jitter2[mask], data2[mask], c=color, alpha=0.7, s=40,
                      edgecolors='white', linewidths=0.5)
    else:
        ax.scatter(x1 + jitter1, data1, c=color1, alpha=0.6, s=30, edgecolors='none')
        ax.scatter(x2 + jitter2, data2, c=color2, alpha=0.6, s=30, edgecolors='none')
    
    # Calculate means and SEMs
    mean1, mean2 = np.mean(data1), np.mean(data2)
    sem1 = np.std(data1) / np.sqrt(len(data1))
    sem2 = np.std(data2) / np.sqrt(len(data2))
    
    # Plot mean with error bars (black diamond)
    ax.errorbar(x1, mean1, yerr=sem1, fmt='D', color='black', markersize=4, 
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=10)
    ax.errorbar(x2, mean2, yerr=sem2, fmt='D', color='black', markersize=4,
                capsize=4, capthick=1.5, elinewidth=1.5, zorder=10)
    
    # Add zero line
    if show_zero_line:
        ax.axhline(y=0, color='gray', linewidth=1, zorder=1)
    
    # Set x-axis
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label1, label2], fontsize=LABEL_FONTSIZE)
    
    # Handle y-axis for log RT display
    if log_scale_rt:
        # Data is in log(seconds), but we want to display seconds on y-axis
        y_min_log, y_max_log = np.log(0.1), np.log(5)
        
        # Convert to seconds for tick calculation
        y_min_sec = np.exp(y_min_log)
        y_max_sec = np.exp(y_max_log)
        
        # Choose nice tick values in seconds
        possible_ticks = [0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        tick_values_sec = [t for t in possible_ticks if y_min_sec * 0.7 <= t <= y_max_sec * 1.5]
        
        if len(tick_values_sec) < 3:
            tick_values_sec = [0.5, 1, 2, 5, 10]
        
        tick_values_log = np.log(tick_values_sec)
        
        ax.set_yticks(tick_values_log)
        ax.set_yticklabels([f'{t:g}' for t in tick_values_sec])
        
        # Extend y limits slightly
        y_range_log = y_max_log - y_min_log
        ax.set_ylim(y_min_log - 0.15 * y_range_log, y_max_log + 0.3 * y_range_log)
        
        # P-value annotation position
        y_pos_pval = y_max_log + 0.1 * y_range_log
    else:
        # Standard y-axis handling
        y_max = 0.7
        y_min = -0.3
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(-0.2, 0.8, 0.2))
        y_pos_pval = y_max - 0.1 * y_range

    # Add p-value annotation
    p_text = get_significance_text(p_value)
    ax.text(0.5, y_pos_pval, p_text, ha='center', va='bottom', 
            fontsize=11, style='italic')
    
    # Set ylabel
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend for subjects (outside plot on right) - only if requested
    if subjects is not None and show_legend:
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9, 
                 bbox_to_anchor=(1.02, 0.7), borderaxespad=0)
    
    return ax


# ============================================================================
# SUMMARY STATISTICS FUNCTIONS
# ============================================================================

def compute_svf_summary_statistics():
    """
    Compute summary statistics for SVF task:
    - Duration of session in minutes (mean, SD): based on last word offset time
    - Categories per session (mean, SD): separated by "next" words
    - Words listed per category (mean, SD)
    """
    print("\n" + "=" * 60)
    print("SVF SUMMARY STATISTICS")
    print("=" * 60)
    
    session_durations = []
    categories_per_session = []
    words_per_category_all = []
    
    for fp in sorted(SVF_ANNOTATIONS_DIR.glob("*.csv")):
        parts = fp.stem.split("_")
        subject = parts[0] if len(parts) > 0 else "unknown"
        session = parts[1] if len(parts) > 1 else "unknown"
        session_id = f"{subject}_{session}"
        
        try:
            df = pd.read_csv(fp)
            df = df.sort_values("start").reset_index(drop=True)
            
            if len(df) == 0:
                continue
            
            # Duration: last word offset time (end time of last word) in minutes
            last_offset = df["end"].max()
            duration_minutes = last_offset / 60.0
            session_durations.append(duration_minutes)
            
            # Identify "next" words to determine category boundaries
            df["is_next"] = df["transcription"].astype(str).str.lower() == "next"
            
            # Assign category ID: increment each time we see a "next" word
            df["category_id"] = df["is_next"].cumsum()
            
            # Filter out the "next" words themselves for word counts
            words_df = df[~df["is_next"]].copy()
            
            if len(words_df) == 0:
                continue
            
            # Count categories (unique category IDs that have words)
            words_per_cat = words_df.groupby("category_id").size()
            n_categories = len(words_per_cat)
            categories_per_session.append(n_categories)
            
            # Collect words per category
            words_per_category_all.extend(words_per_cat.values)
            
            print(f"  {session_id}: {duration_minutes:.2f} min, {n_categories} categories, "
                  f"{len(words_df)} words")
            
        except Exception as e:
            print(f"  Error loading {fp.name}: {e}")
    
    if not session_durations:
        print("No SVF data found!")
        return None
    
    # Compute statistics
    duration_mean = np.mean(session_durations)
    duration_sd = np.std(session_durations, ddof=1)
    
    categories_mean = np.mean(categories_per_session)
    categories_sd = np.std(categories_per_session, ddof=1)
    
    words_per_cat_mean = np.mean(words_per_category_all)
    words_per_cat_sd = np.std(words_per_category_all, ddof=1)
    
    print("\n--- SVF Summary Statistics ---")
    print(f"  N sessions: {len(session_durations)}")
    print(f"  Duration of session (min): Mean = {duration_mean:.2f}, SD = {duration_sd:.2f}")
    print(f"  Categories per session: Mean = {categories_mean:.2f}, SD = {categories_sd:.2f}")
    print(f"  Words per category: Mean = {words_per_cat_mean:.2f}, SD = {words_per_cat_sd:.2f}")
    
    return {
        'n_sessions': len(session_durations),
        'duration_mean': duration_mean,
        'duration_sd': duration_sd,
        'categories_mean': categories_mean,
        'categories_sd': categories_sd,
        'words_per_category_mean': words_per_cat_mean,
        'words_per_category_sd': words_per_cat_sd
    }


def compute_ahc_summary_statistics():
    """
    Compute summary statistics for AHC task:
    - Duration of session in minutes (mean, SD): based on last sentence offset time
    - Prompts per session (mean, SD)
    - Number of possibilities generated per prompt (mean, SD)
    """
    print("\n" + "=" * 60)
    print("AHC SUMMARY STATISTICS")
    print("=" * 60)
    
    session_durations = []
    prompts_per_session = []
    possibilities_per_prompt_all = []
    
    for fp in sorted(AHC_ANNOTATIONS_DIR.glob("*.xlsx")):
        parts = fp.stem.split("_")
        subject = parts[0] if len(parts) > 0 else "unknown"
        session = parts[1] if len(parts) > 1 else "unknown"
        session_id = f"{subject}_{session}"
        
        try:
            df = pd.read_excel(fp)
            
            # Check for required columns
            if 'End Time' not in df.columns:
                print(f"  Warning: 'End Time' column not found in {fp.name}")
                continue
            
            if 'Prompt Number' not in df.columns or 'Possibility Number' not in df.columns:
                print(f"  Warning: Required columns missing in {fp.name}")
                continue
            
            # Forward fill prompt number
            df['Prompt Number'] = df['Prompt Number'].ffill()
            
            # Filter to rows with valid data
            df = df[df['Possibility Number'].notna()].copy()
            
            if len(df) == 0:
                continue
            
            # Duration: last sentence offset time (End Time of last sentence) in minutes
            last_offset = df["End Time"].max()
            duration_minutes = last_offset / 60.0
            session_durations.append(duration_minutes)
            
            # Count unique prompts
            unique_prompts = df['Prompt Number'].nunique()
            prompts_per_session.append(unique_prompts)
            
            # Count unique possibilities per prompt
            for prompt_num in df['Prompt Number'].unique():
                prompt_df = df[df['Prompt Number'] == prompt_num]
                n_possibilities = prompt_df['Possibility Number'].nunique()
                possibilities_per_prompt_all.append(n_possibilities)
            
            print(f"  {session_id}: {duration_minutes:.2f} min, {unique_prompts} prompts")
            
        except Exception as e:
            print(f"  Error loading {fp.name}: {e}")
    
    if not session_durations:
        print("No AHC data found!")
        return None
    
    # Compute statistics
    duration_mean = np.mean(session_durations)
    duration_sd = np.std(session_durations, ddof=1)
    
    prompts_mean = np.mean(prompts_per_session)
    prompts_sd = np.std(prompts_per_session, ddof=1)
    
    possibilities_mean = np.mean(possibilities_per_prompt_all)
    possibilities_sd = np.std(possibilities_per_prompt_all, ddof=1)
    
    print("\n--- AHC Summary Statistics ---")
    print(f"  N sessions: {len(session_durations)}")
    print(f"  Duration of session (min): Mean = {duration_mean:.2f}, SD = {duration_sd:.2f}")
    print(f"  Prompts per session: Mean = {prompts_mean:.2f}, SD = {prompts_sd:.2f}")
    print(f"  Possibilities per prompt: Mean = {possibilities_mean:.2f}, SD = {possibilities_sd:.2f}")
    
    return {
        'n_sessions': len(session_durations),
        'duration_mean': duration_mean,
        'duration_sd': duration_sd,
        'prompts_mean': prompts_mean,
        'prompts_sd': prompts_sd,
        'possibilities_per_prompt_mean': possibilities_mean,
        'possibilities_per_prompt_sd': possibilities_sd
    }


def run_summary_statistics():
    """Run summary statistics for both SVF and AHC tasks."""
    svf_stats = compute_svf_summary_statistics()
    ahc_stats = compute_ahc_summary_statistics()
    return svf_stats, ahc_stats


# ============================================================================
# PART 1: SVF IRT ANALYSIS (Log-transformed)
# ============================================================================

def load_svf_session(filepath):
    """Load and parse SVF annotation file for a single session."""
    df = pd.read_csv(filepath)
    df = df.sort_values("start").reset_index(drop=True)
    
    # Parse switch flag
    df["switch_flag"] = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int)
    
    # Compute IRT (inter-response time)
    df["preceding_end"] = df["end"].shift(1)
    df["irt"] = df["start"] - df["preceding_end"]
    
    # Filter out "next" words and invalid entries
    df = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    
    # Remove switches that follow another switch or "next"
    df["preceding_switch_flag"] = df["switch_flag"].shift(1)
    df["preceding_word"] = df["transcription"].shift(1).astype(str).str.lower()
    
    is_switch = df["switch_flag"] == 1
    prev_was_switch = df["preceding_switch_flag"] == 1
    prev_was_next = df["preceding_word"] == "next"
    
    df = df[~(is_switch & (prev_was_switch | prev_was_next))].copy()
    
    return df


def analyze_svf_irt_session(df, session_id):
    """Analyze IRT for a single SVF session. Returns log-transformed values."""
    # Get IRTs and filter positive values for log transform
    cluster_irts = df[df["switch_flag"] == 0]["irt"].dropna().values
    switch_irts = df[df["switch_flag"] == 1]["irt"].dropna().values
    
    cluster_irts = cluster_irts[cluster_irts > 0]
    switch_irts = switch_irts[switch_irts > 0]
    
    if len(cluster_irts) == 0 or len(switch_irts) == 0:
        return None
    
    # Log transform only (no z-scoring)
    log_cluster = np.log(cluster_irts)
    log_switch = np.log(switch_irts)
    
    return {
        'session': session_id,
        'cluster_mean_log': np.mean(log_cluster),
        'switch_mean_log': np.mean(log_switch),
        'cluster_n': len(cluster_irts),
        'switch_n': len(switch_irts),
    }


def run_svf_irt_analysis(return_data=False):
    """Run SVF IRT analysis with log transform."""
    print("\n" + "=" * 60)
    print("SVF IRT ANALYSIS: Switch vs Cluster (Log-transformed)")
    print("=" * 60)
    
    # Collect all sessions grouped by subject
    subject_sessions = defaultdict(list)
    
    for fp in sorted(SVF_ANNOTATIONS_DIR.glob("*.csv")):
        parts = fp.stem.split("_")
        subject = parts[0] if len(parts) > 0 else "unknown"
        session = parts[1] if len(parts) > 1 else "unknown"
        session_id = f"{subject}_{session}"
        
        try:
            df = load_svf_session(fp)
            result = analyze_svf_irt_session(df, session_id)
            if result:
                result['subject'] = subject
                subject_sessions[subject].append(result)
                print(f"  Loaded {session_id}: {result['cluster_n']} cluster, {result['switch_n']} switch")
        except Exception as e:
            print(f"  Error loading {fp.name}: {e}")
    
    if not subject_sessions:
        print("No SVF data found!")
        return None
    
    # Collect all session data for plotting
    all_sessions = []
    for subject, sessions in subject_sessions.items():
        all_sessions.extend(sessions)
    
    # Extract session means for scatter plot
    cluster_means = np.array([s['cluster_mean_log'] for s in all_sessions])
    switch_means = np.array([s['switch_mean_log'] for s in all_sessions])
    subjects = [s['subject'] for s in all_sessions]
    
    # Statistical test (paired by session)
    t_stat, p_val = ttest_rel(switch_means, cluster_means)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    plot_comparison_scatter(
        ax, cluster_means, switch_means,
        'Clustering', 'Switch', p_val,
        subjects=subjects,
        ylabel='Inter-Word Interval (sec)',
        show_zero_line=False,
        log_scale_rt=True
    )
    
    ax.set_title('Semantic Verbal Fluency: Inter-Word Interval', fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    plt.tight_layout()
    out_path = BEHAVIOR_FIGS_DIR / "svf_irt_group.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")
    
    # Print summary (convert back to seconds for interpretability)
    print(f"\n--- SVF IRT Summary ---")
    print(f"  N sessions: {len(all_sessions)}")
    print(f"  Cluster mean: {np.exp(np.mean(cluster_means)):.2f}s (geometric mean)")
    print(f"  Switch mean: {np.exp(np.mean(switch_means)):.2f}s (geometric mean)")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    
    if return_data:
        return all_sessions
    return all_sessions


# ============================================================================
# PART 2: AHC RT ANALYSIS (Log-transformed)
# ============================================================================

def load_ahc_session(filepath):
    """Load and parse AHC sentence file for a single session."""
    df = pd.read_excel(filepath)
    
    required_cols = ['Prompt Number', 'Start Time', 'End Time', 'Possibility Number']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df[df['Possibility Number'].notna()].copy()
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    
    return df


def analyze_ahc_rt_session(df, session_id):
    """Analyze between-sentence intervals for a single AHC session."""
    intervals_within = []
    intervals_between = []
    
    for prompt_num in df['Prompt Number'].unique():
        prompt_df = df[df['Prompt Number'] == prompt_num].reset_index(drop=True)
        
        for i in range(1, len(prompt_df)):
            interval = prompt_df.loc[i, 'Start Time'] - prompt_df.loc[i-1, 'End Time']
            
            if interval <= 0:
                continue
            
            prev_poss = prompt_df.loc[i-1, 'Possibility Number']
            curr_poss = prompt_df.loc[i, 'Possibility Number']
            
            if prev_poss == curr_poss:
                intervals_within.append(interval)
            else:
                intervals_between.append(interval)
    
    if len(intervals_within) == 0 or len(intervals_between) == 0:
        return None
    
    # Log transform only (no z-scoring)
    log_within = np.log(intervals_within)
    log_between = np.log(intervals_between)
    
    return {
        'session': session_id,
        'within_mean_log': np.mean(log_within),
        'between_mean_log': np.mean(log_between),
        'within_n': len(intervals_within),
        'between_n': len(intervals_between),
    }


def run_ahc_rt_analysis(return_data=False):
    """Run AHC RT analysis with log transform."""
    print("\n" + "=" * 60)
    print("AHC RT ANALYSIS: Within vs Between (Log-transformed)")
    print("=" * 60)
    
    subject_sessions = defaultdict(list)
    
    for fp in sorted(AHC_ANNOTATIONS_DIR.glob("*.xlsx")):
        parts = fp.stem.split("_")
        subject = parts[0] if len(parts) > 0 else "unknown"
        session = parts[1] if len(parts) > 1 else "unknown"
        session_id = f"{subject}_{session}"
        
        try:
            df = load_ahc_session(fp)
            result = analyze_ahc_rt_session(df, session_id)
            if result:
                result['subject'] = subject
                subject_sessions[subject].append(result)
                print(f"  Loaded {session_id}: {result['within_n']} within, {result['between_n']} between")
        except Exception as e:
            print(f"  Error loading {fp.name}: {e}")
    
    if not subject_sessions:
        print("No AHC data found!")
        return None
    
    # Collect all session data
    all_sessions = []
    for subject, sessions in subject_sessions.items():
        all_sessions.extend(sessions)
    
    # Extract session means
    within_means = np.array([s['within_mean_log'] for s in all_sessions])
    between_means = np.array([s['between_mean_log'] for s in all_sessions])
    subjects = [s['subject'] for s in all_sessions]
    
    # Statistical test
    t_stat, p_val = ttest_rel(between_means, within_means)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    plot_comparison_scatter(
        ax, within_means, between_means,
        'Within-possibility', 'Between-possibilities', p_val,
        subjects=subjects,
        ylabel='Inter-Sentence Interval (sec)',
        show_zero_line=False,
        log_scale_rt=True
    )
    
    ax.set_title('Possibility Generation: Inter-Sentence Interval', fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    plt.tight_layout()
    out_path = BEHAVIOR_FIGS_DIR / "ahc_rt_group.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")
    
    # Print summary
    print(f"\n--- AHC RT Summary ---")
    print(f"  N sessions: {len(all_sessions)}")
    print(f"  Within mean: {np.exp(np.mean(within_means)):.2f}s (geometric mean)")
    print(f"  Between mean: {np.exp(np.mean(between_means)):.2f}s (geometric mean)")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    
    if return_data:
        return all_sessions
    return all_sessions


# ============================================================================
# PART 3: SVF SEMANTIC ANALYSIS (GPT-2 Static Embeddings)
# ============================================================================

# GPT-2 imports (lazy loading)
GPT2_AVAILABLE = False
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    GPT2_AVAILABLE = True
except ImportError:
    pass


class GPT2WordEmbedder:
    """
    Simple word embedder using GPT-2 static (non-contextualized) embeddings.
    """
    
    def __init__(self, model_name='gpt2-medium'):
        if not GPT2_AVAILABLE:
            raise ImportError("PyTorch and transformers are required for semantic analysis.")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        
        # Get static embeddings (word embedding layer)
        self.static_embeddings = self.model.transformer.wte.weight.detach()
        
        print(f"Loaded {model_name} on {self.device}")
    
    def get_word_embedding(self, word):
        """Get non-contextualized embedding for a word (mean across subtokens)."""
        tokens = self.tokenizer.encode(f" {word}", add_special_tokens=False)
        token_embeddings = self.static_embeddings[tokens]
        return token_embeddings.mean(dim=0).cpu().numpy()
    
    def analyze_word_list(self, words):
        """
        Compute cosine distances between consecutive words.
        Returns list of dicts with 'word', 'position', 'cosine_dist'.
        """
        if len(words) < 2:
            return []
        
        # Get embedding for first word
        prev_emb = self.get_word_embedding(words[0])
        
        results = []
        for i, word in enumerate(words[1:], start=1):
            curr_emb = self.get_word_embedding(word)
            
            # Compute cosine distance
            dist = cosine(prev_emb, curr_emb)
            
            results.append({
                'word': word,
                'position': i,
                'cosine_dist': dist
            })
            
            prev_emb = curr_emb
        
        return results


def run_svf_semantic_analysis(return_data=False):
    """Run SVF semantic analysis using GPT-2 static embeddings with z-scoring."""
    print("\n" + "=" * 60)
    print("SVF SEMANTIC ANALYSIS: GPT-2 Static Embedding Distance (Z-scored)")
    print("=" * 60)
    
    if not GPT2_AVAILABLE:
        print("ERROR: PyTorch/transformers not available.")
        return None
    
    embedder = GPT2WordEmbedder(model_name='gpt2-medium')
    
    # Collect data by session
    session_data = []
    
    for fp in sorted(SVF_ANNOTATIONS_DIR.glob("*.csv")):
        parts = fp.stem.split("_")
        subject = parts[0] if len(parts) > 0 else "unknown"
        session = parts[1] if len(parts) > 1 else "unknown"
        session_id = f"{subject}_{session}"
        
        try:
            df = load_svf_session(fp)
            words = df['transcription'].astype(str).str.lower().tolist()
            switch_flags = df['switch_flag'].values
            
            if len(words) < 3:
                continue
            
            results = embedder.analyze_word_list(words)
            
            # Add switch flags (results start at position 1)
            for r in results:
                r['switch_flag'] = switch_flags[r['position']]
            
            results_df = pd.DataFrame(results)
            
            # Z-score cosine distance within this session
            if len(results_df) > 1:
                results_df['cosine_dist_z'] = zscore(results_df['cosine_dist'].values)
            else:
                continue
            
            # Compute session means for cluster and switch
            cluster_data = results_df[results_df['switch_flag'] == 0]
            switch_data = results_df[results_df['switch_flag'] == 1]
            
            if len(cluster_data) > 0 and len(switch_data) > 0:
                session_result = {
                    'session': session_id,
                    'subject': subject,
                    'cluster_mean_z': cluster_data['cosine_dist_z'].mean(),
                    'switch_mean_z': switch_data['cosine_dist_z'].mean(),
                    'cluster_n': len(cluster_data),
                    'switch_n': len(switch_data),
                }
                
                session_data.append(session_result)
                print(f"  Processed {session_id}: {len(cluster_data)} cluster, {len(switch_data)} switch")
            
        except Exception as e:
            print(f"  Error processing {fp.name}: {e}")
    
    if not session_data:
        print("No data processed!")
        return None
    
    # Extract data for plotting
    cluster_means = np.array([s['cluster_mean_z'] for s in session_data])
    switch_means = np.array([s['switch_mean_z'] for s in session_data])
    subjects = [s['subject'] for s in session_data]
    
    # Statistical test
    t_stat, p_val = ttest_rel(switch_means, cluster_means)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    plot_comparison_scatter(
        ax, cluster_means, switch_means,
        'Cluster', 'Switch', p_val,
        subjects=subjects,
        ylabel='Semantic Distance\n(z-scored)',
        color1=COLORS['cluster'], color2=COLORS['switch']
    )
    
    ax.set_title('SVF: Semantic Distance', fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    plt.tight_layout()
    out_path = BEHAVIOR_FIGS_DIR / "svf_semantic_group.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")
    
    # Print summary
    print(f"\n--- SVF Semantic Summary (N={len(session_data)} sessions) ---")
    print(f"  Cluster mean (z): {np.mean(cluster_means):.3f}")
    print(f"  Switch mean (z): {np.mean(switch_means):.3f}")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    
    if return_data:
        return session_data
    return session_data


# ============================================================================
# PART 4: AHC SEMANTIC ANALYSIS (Universal Sentence Encoder)
# ============================================================================

# TensorFlow Hub imports (lazy loading)
USE_AVAILABLE = False
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    USE_AVAILABLE = True
except ImportError:
    pass


class UniversalSentenceEmbedder:
    """
    Sentence embedder using Universal Sentence Encoder (USE).
    """
    
    def __init__(self, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        if not USE_AVAILABLE:
            raise ImportError("TensorFlow and TensorFlow Hub are required for USE.")
        
        print("Loading Universal Sentence Encoder...")
        self.model = hub.load(model_url)
        print("USE loaded successfully")
    
    def get_embedding(self, sentence):
        """Get embedding for a single sentence."""
        embeddings = self.model([sentence])
        return embeddings[0].numpy()
    
    def get_embeddings_batch(self, sentences):
        """Get embeddings for a batch of sentences."""
        embeddings = self.model(sentences)
        return embeddings.numpy()
    
    def analyze_sentence_list(self, sentences):
        """
        Compute cosine distances between consecutive sentences.
        Returns list of dicts with 'sentence', 'position', 'cosine_dist'.
        """
        if len(sentences) < 2:
            return []
        
        # Clean sentences
        clean_sentences = []
        for s in sentences:
            if not isinstance(s, str):
                s = str(s)
            s = s.strip()
            s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\t')
            clean_sentences.append(s if s else " ")
        
        # Get all embeddings at once (more efficient)
        try:
            embeddings = self.get_embeddings_batch(clean_sentences)
        except Exception as e:
            print(f"    Warning: Error getting batch embeddings: {e}")
            return []
        
        results = []
        for i in range(1, len(embeddings)):
            dist = cosine(embeddings[i-1], embeddings[i])
            
            results.append({
                'sentence': clean_sentences[i],
                'position': i,
                'cosine_dist': dist
            })
        
        return results


def load_ahc_session_for_semantic(filepath):
    """Load and parse AHC sentence file for semantic analysis."""
    df = pd.read_excel(filepath)
    
    sentence_col = None
    for col in ['Sentence', 'sentence', 'Text', 'text', 'Transcription', 'transcription']:
        if col in df.columns:
            sentence_col = col
            break
    
    if sentence_col is None:
        raise ValueError(f"No sentence column found. Available columns: {df.columns.tolist()}")
    
    required_cols = ['Prompt Number', 'Possibility Number']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df['Prompt Number'] = df['Prompt Number'].ffill()
    df = df[df['Possibility Number'].notna()].copy()
    df = df[df[sentence_col].notna()].copy()
    df['Sentence'] = df[sentence_col]
    
    if 'Start Time' in df.columns:
        df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    else:
        df = df.sort_values(['Prompt Number']).reset_index(drop=True)
    
    return df


def run_ahc_semantic_analysis(return_data=False):
    """Run AHC semantic analysis using Universal Sentence Encoder with z-scoring."""
    print("\n" + "=" * 60)
    print("AHC SEMANTIC ANALYSIS: USE Embedding Distance (Z-scored)")
    print("=" * 60)
    
    if not USE_AVAILABLE:
        print("ERROR: TensorFlow/TensorFlow Hub not available.")
        print("Install with: pip install tensorflow tensorflow-hub")
        return None
    
    # Create embedder once
    embedder = UniversalSentenceEmbedder()
    
    session_data = []
    
    for fp in sorted(AHC_ANNOTATIONS_DIR.glob("*.xlsx")):
        parts = fp.stem.split("_")
        subject = parts[0] if len(parts) > 0 else "unknown"
        session = parts[1] if len(parts) > 1 else "unknown"
        session_id = f"{subject}_{session}"
        
        try:
            df = load_ahc_session_for_semantic(fp)
            
            all_results = []
            
            for prompt_num in df['Prompt Number'].unique():
                prompt_df = df[df['Prompt Number'] == prompt_num].reset_index(drop=True)
                
                if len(prompt_df) < 2:
                    continue
                
                sentences = prompt_df['Sentence'].astype(str).tolist()
                possibilities = prompt_df['Possibility Number'].values
                
                results = embedder.analyze_sentence_list(sentences)
                
                # Add boundary flag
                for r in results:
                    pos = r['position']
                    prev_poss = possibilities[pos - 1]
                    curr_poss = possibilities[pos]
                    r['is_boundary'] = 1 if prev_poss != curr_poss else 0
                    r['prompt'] = prompt_num
                
                all_results.extend(results)
            
            if len(all_results) == 0:
                print(f"  {session_id}: No valid results")
                continue
            
            results_df = pd.DataFrame(all_results)
            
            # Z-score cosine distance within this session
            if len(results_df) > 1:
                results_df['cosine_dist_z'] = zscore(results_df['cosine_dist'].values)
            else:
                continue
            
            # Compute session means
            within_data = results_df[results_df['is_boundary'] == 0]
            between_data = results_df[results_df['is_boundary'] == 1]
            
            if len(within_data) > 0 and len(between_data) > 0:
                session_result = {
                    'session': session_id,
                    'subject': subject,
                    'within_mean_z': within_data['cosine_dist_z'].mean(),
                    'between_mean_z': between_data['cosine_dist_z'].mean(),
                    'within_n': len(within_data),
                    'between_n': len(between_data),
                }
                
                session_data.append(session_result)
                print(f"  Processed {session_id}: {len(within_data)} within, {len(between_data)} between")
            
        except Exception as e:
            print(f"  Error processing {fp.name}: {e}")
    
    if not session_data:
        print("No data processed!")
        return None
    
    # Extract data for plotting
    within_means = np.array([s['within_mean_z'] for s in session_data])
    between_means = np.array([s['between_mean_z'] for s in session_data])
    subjects = [s['subject'] for s in session_data]
    
    # Statistical test
    t_stat, p_val = ttest_rel(between_means, within_means)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    plot_comparison_scatter(
        ax, between_means, within_means,
        'Within\npossibility', 'Between\npossibilities', p_val,
        subjects=subjects,
        ylabel='Semantic Distance\n(z-scored)',
        color1=COLORS['nonboundary'], color2=COLORS['boundary']
    )
    
    ax.set_title('AHC: Semantic Distance', fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    plt.tight_layout()
    out_path = BEHAVIOR_FIGS_DIR / "ahc_semantic_group.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")
    
    # Print summary
    print(f"\n--- AHC Semantic Summary (N={len(session_data)} sessions) ---")
    print(f"  Within mean (z): {np.mean(within_means):.3f}")
    print(f"  Between mean (z): {np.mean(between_means):.3f}")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    
    if return_data:
        return session_data
    return session_data


# ============================================================================
# MERGED PLOTS: COMBINE TASKS (SVF + AHC)
# ============================================================================

def create_merged_rt_plot(svf_rt_data, ahc_rt_data):
    """
    Create a merged RT figure combining SVF and AHC tasks side by side.
    Single shared legend for subject colors.
    
    Layout: 1 row x 2 columns (SVF RT, AHC RT)
    """
    print("\n" + "=" * 60)
    print("Creating Merged RT Plot (SVF + AHC)")
    print("=" * 60)
    
    if svf_rt_data is None or ahc_rt_data is None:
        print("  Missing data for merged plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # --- Panel A: SVF IRT ---
    ax = axes[0]
    cluster_means = np.array([s['cluster_mean_log'] for s in svf_rt_data])
    switch_means = np.array([s['switch_mean_log'] for s in svf_rt_data])
    subjects_svf = [s['subject'] for s in svf_rt_data]
    
    t_stat, p_val = ttest_rel(switch_means, cluster_means)
    
    plot_comparison_scatter(
        ax, cluster_means, switch_means,
        'Clustering', 'Switching', p_val,
        subjects=subjects_svf,
        ylabel='Inter-Word Interval (sec)',
        show_zero_line=False,
        log_scale_rt=True,
        show_legend=False
    )
    
    # --- Panel B: AHC RT ---
    ax = axes[1]
    within_means = np.array([s['within_mean_log'] for s in ahc_rt_data])
    between_means = np.array([s['between_mean_log'] for s in ahc_rt_data])
    subjects_ahc = [s['subject'] for s in ahc_rt_data]
    
    t_stat, p_val = ttest_rel(between_means, within_means)
    
    plot_comparison_scatter(
        ax, within_means, between_means,
        'Within-\nexplanations', 'Between-\nexplanations', p_val,
        subjects=subjects_ahc,
        ylabel='Inter-Sentence Interval (sec)',
        show_zero_line=False,
        log_scale_rt=True,
        show_legend=False
    )    
    # --- Create shared legend ---
    all_subjects = sorted(set(subjects_svf) | set(subjects_ahc))
    
    legend_handles = []
    for subj in all_subjects:
        color = SUBJECT_COLORS.get(subj, 'gray')
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                           markersize=8, label=get_subject_label(subj))
        legend_handles.append(handle)
    
    fig.legend(handles=legend_handles, loc='center right', 
               bbox_to_anchor=(1.12, 0.5), fontsize=10, framealpha=0.9,
               title='Subject', title_fontsize=11)
    
    
    plt.tight_layout()
    out_path = BEHAVIOR_FIGS_DIR / "merged_rt_svf_ahc.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    
    return fig


def create_merged_semantic_plot(svf_semantic_data, ahc_semantic_data):
    """
    Create a merged semantic figure combining SVF and AHC tasks side by side.
    Single shared legend for subject colors.
    
    Layout: 1 row x 2 columns (SVF semantic, AHC semantic)
    """
    print("\n" + "=" * 60)
    print("Creating Merged Semantic Plot (SVF + AHC)")
    print("=" * 60)
    
    if svf_semantic_data is None or ahc_semantic_data is None:
        print("  Missing data for merged plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # --- Panel A: SVF Semantic Distance ---
    ax = axes[0]
    cluster_means = np.array([s['cluster_mean_z'] for s in svf_semantic_data])
    switch_means = np.array([s['switch_mean_z'] for s in svf_semantic_data])
    subjects_svf = [s['subject'] for s in svf_semantic_data]
    
    t_stat, p_val = ttest_rel(switch_means, cluster_means)
    
    plot_comparison_scatter(
        ax, cluster_means, switch_means,
        'Clustering', 'Switching', p_val,
        subjects=subjects_svf,
        ylabel='Semantic Distance',
        color1=COLORS['cluster'], color2=COLORS['switch'],
        show_legend=False
    )
    
    # --- Panel B: AHC Semantic Distance ---
    ax = axes[1]
    within_means = np.array([s['within_mean_z'] for s in ahc_semantic_data])
    between_means = np.array([s['between_mean_z'] for s in ahc_semantic_data])
    subjects_ahc = [s['subject'] for s in ahc_semantic_data]
    
    t_stat, p_val = ttest_rel(between_means, within_means)
    
    plot_comparison_scatter(
        ax, between_means, within_means,
        'Within-\nexplanations', 'Between-\nexplanations', p_val,
        subjects=subjects_ahc,
        ylabel='Semantic Distance',
        color1=COLORS['nonboundary'], color2=COLORS['boundary'],
        show_legend=False
    )
    
    # --- Create shared legend ---
    all_subjects = sorted(set(subjects_svf) | set(subjects_ahc))
    
    legend_handles = []
    for subj in all_subjects:
        color = SUBJECT_COLORS.get(subj, 'gray')
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                           markersize=8, label=get_subject_label(subj))
        legend_handles.append(handle)
    
    fig.legend(handles=legend_handles, loc='center right', 
               bbox_to_anchor=(1.12, 0.5), fontsize=10, framealpha=0.9,
               title='Subject', title_fontsize=11)
    
    
    plt.tight_layout()
    out_path = BEHAVIOR_FIGS_DIR / "merged_semantic_svf_ahc.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    
    return fig


def run_merged_plots():
    """Run all analyses and create merged plots (combining tasks)."""
    print("\n" + "=" * 60)
    print("RUNNING MERGED PLOT GENERATION (Combining Tasks)")
    print("=" * 60)
    
    # Run RT analyses
    svf_rt_data = run_svf_irt_analysis(return_data=True)
    ahc_rt_data = run_ahc_rt_analysis(return_data=True)
    
    # Run semantic analyses
    svf_semantic_data = run_svf_semantic_analysis(return_data=True)
    ahc_semantic_data = run_ahc_semantic_analysis(return_data=True)
    
    # Create merged plots
    create_merged_rt_plot(svf_rt_data, ahc_rt_data)
    create_merged_semantic_plot(svf_semantic_data, ahc_semantic_data)
    
    print("\n" + "=" * 60)
    print("MERGED PLOTS COMPLETE")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined Behavioral Analysis for SVF and AHC Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python svf_ahc_behavior.py --all                    # Run all analyses
    python svf_ahc_behavior.py --svf-rt                 # SVF IRT analysis only
    python svf_ahc_behavior.py --ahc-rt                 # AHC RT analysis only
    python svf_ahc_behavior.py --svf-semantic           # SVF semantic analysis only
    python svf_ahc_behavior.py --ahc-semantic           # AHC semantic analysis only
    python svf_ahc_behavior.py --merged                 # Create merged RT+semantic plots
    python svf_ahc_behavior.py --merged-rt              # Create merged RT plot only
    python svf_ahc_behavior.py --summary                # Output summary statistics only
        """
    )
    
    parser.add_argument("--all", action="store_true", 
                        help="Run all analyses")
    parser.add_argument("--svf-rt", action="store_true",
                        help="Run SVF IRT analysis only")
    parser.add_argument("--ahc-rt", action="store_true",
                        help="Run AHC RT analysis only")
    parser.add_argument("--svf-semantic", action="store_true",
                        help="Run SVF semantic analysis only")
    parser.add_argument("--ahc-semantic", action="store_true",
                        help="Run AHC semantic analysis only")
    parser.add_argument("--merged", action="store_true",
                        help="Create merged RT+semantic plots for each task")
    parser.add_argument("--merged-rt", action="store_true",
                        help="Create merged RT plot only (no semantic)")
    parser.add_argument("--summary", action="store_true",
                        help="Output summary statistics for SVF and AHC tasks")
    
    args = parser.parse_args()
    
    if not any([args.all, args.svf_rt, args.ahc_rt, args.svf_semantic, args.ahc_semantic, args.merged, args.merged_rt, args.summary]):
        args.all = True
    
    print("=" * 60)
    print("COMBINED BEHAVIORAL ANALYSIS: SVF & AHC")
    print(f"Output directory: {BEHAVIOR_FIGS_DIR}")
    print("=" * 60)
    
    # If summary is requested, run summary statistics
    if args.summary:
        run_summary_statistics()
        return
    
    # If merged-rt is requested, run only RT analyses and merged RT plot
    if args.merged_rt:
        print("\n" + "=" * 60)
        print("RUNNING MERGED RT PLOT GENERATION")
        print("=" * 60)
        
        svf_rt_data = run_svf_irt_analysis(return_data=True)
        ahc_rt_data = run_ahc_rt_analysis(return_data=True)
        create_merged_rt_plot(svf_rt_data, ahc_rt_data)
        
        print("\n" + "=" * 60)
        print("MERGED RT PLOT COMPLETE")
        print("=" * 60)
        return
    
    # If merged is requested, run the full merged workflow
    if args.merged:
        run_merged_plots()
    else:
        # Run individual analyses
        if args.all or args.svf_rt:
            run_svf_irt_analysis()
        
        if args.all or args.ahc_rt:
            run_ahc_rt_analysis()
        
        if args.all or args.svf_semantic:
            run_svf_semantic_analysis()
        
        if args.all or args.ahc_semantic:
            run_ahc_semantic_analysis()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {BEHAVIOR_FIGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()