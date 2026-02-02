import sys
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind
import argparse
import warnings
from pathlib import Path

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# === 1. CONFIG & PATHS ===
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs.config import DATA_DIR, DERIVATIVES_DIR, FIGS_DIR
except ImportError:
    DATA_DIR, DERIVATIVES_DIR, FIGS_DIR = Path("./data"), Path("./derivatives"), Path("./figs")

ANNOTATIONS_DIR = DATA_DIR / "rec/svf_annotated"
GPT_FIGS_DIR = FIGS_DIR / "gpt_analysis"
GPT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === 2. CONSTANTS ===
DRIFT = 0.05  # Smoothing parameter for transient normalization

# Fixed y-limits for bar plots
YLIM_RAW_BAR = (0, 25)
YLIM_ZSCORE_BAR = (-2, 2)
YLIM_TRANSIENT_BAR = (0, 3)

# Fixed y-limits for line plots (wider range)
YLIM_RAW_LINE = (0, 35)
YLIM_ZSCORE_LINE = (-4, 4)
YLIM_TRANSIENT_LINE = (0, 5)

# Subject colors for individual points (assuming up to 10 subjects)
SUBJECT_COLORS = {
    'sub-001': '#e41a1c',
    'sub-003': '#377eb8',
    'sub-004': '#4daf4a',
    'sub-006': '#984ea3',
    'sub-007': '#ff7f00',
    'sub-008': '#ffff33',
    'sub-009': '#a65628'
}

# Raw metrics (non-transient) - ORDERED
RAW_METRICS = [
    'surprisal_global',
    'surprisal_bigram',
    'entropy_global',
    'bayesian_surprise',
    'cosine_dist_contextualized',
    'cosine_dist_static'
]

# All metrics get normalized
METRICS_TO_NORMALIZE = [
    'surprisal_global',
    'surprisal_bigram',
    'entropy_global',
    'bayesian_surprise',
    'cosine_dist_contextualized',
    'cosine_dist_static'
]

# Transient metrics - ORDERED
TRANSIENT_METRICS = [f"{m}_transient" for m in METRICS_TO_NORMALIZE]

# Z-scored metrics for visualization
ZSCORED_METRICS = [f"{m}_zscore" for m in RAW_METRICS]

# All metrics
ALL_METRICS = RAW_METRICS + TRANSIENT_METRICS + ZSCORED_METRICS

# Explicit order for bar plots
RAW_BAR_ORDER = [
    'Surprisal (Global)',
    'Surprisal (Bigram)',
    'Entropy (Global)',
    'Bayesian Surprise',
    'Cosine Dist (Ctx)',
    'Cosine Dist (Static)'
]

TRANSIENT_BAR_ORDER = [
    'Surprisal (Global, T)',
    'Surprisal (Bigram, T)',
    'Entropy (Global, T)',
    'Bayesian Surprise (T)',
    'Cosine Dist (Ctx, T)',
    'Cosine Dist (Static, T)'
]

ZSCORED_BAR_ORDER = [
    'Surprisal (Global, Z)',
    'Surprisal (Bigram, Z)',
    'Entropy (Global, Z)',
    'Bayesian Surprise (Z)',
    'Cosine Dist (Ctx, Z)',
    'Cosine Dist (Static, Z)'
]

COLORS = {
    'surprisal_global': '#1f77b4',
    'surprisal_global_transient': '#1f77b4',
    'surprisal_global_zscore': '#1f77b4',
    'surprisal_bigram': '#aec7e8',
    'surprisal_bigram_transient': '#aec7e8',
    'surprisal_bigram_zscore': '#aec7e8',
    'entropy_global': '#2ca02c',
    'entropy_global_transient': '#2ca02c',
    'entropy_global_zscore': '#2ca02c',
    'bayesian_surprise': '#d62728',
    'bayesian_surprise_transient': '#d62728',
    'bayesian_surprise_zscore': '#d62728',
    'cosine_dist_contextualized': '#9467bd',
    'cosine_dist_contextualized_transient': '#9467bd',
    'cosine_dist_contextualized_zscore': '#9467bd',
    'cosine_dist_static': '#c5b0d5',
    'cosine_dist_static_transient': '#c5b0d5',
    'cosine_dist_static_zscore': '#c5b0d5'
}

METRIC_LABELS = {
    'surprisal_global': 'Surprisal (Global)',
    'surprisal_global_transient': 'Surprisal (Global, T)',
    'surprisal_global_zscore': 'Surprisal (Global, Z)',
    'surprisal_bigram': 'Surprisal (Bigram)',
    'surprisal_bigram_transient': 'Surprisal (Bigram, T)',
    'surprisal_bigram_zscore': 'Surprisal (Bigram, Z)',
    'entropy_global': 'Entropy (Global)',
    'entropy_global_transient': 'Entropy (Global, T)',
    'entropy_global_zscore': 'Entropy (Global, Z)',
    'bayesian_surprise': 'Bayesian Surprise',
    'bayesian_surprise_transient': 'Bayesian Surprise (T)',
    'bayesian_surprise_zscore': 'Bayesian Surprise (Z)',
    'cosine_dist_contextualized': 'Cosine Dist (Ctx)',
    'cosine_dist_contextualized_transient': 'Cosine Dist (Ctx, T)',
    'cosine_dist_contextualized_zscore': 'Cosine Dist (Ctx, Z)',
    'cosine_dist_static': 'Cosine Dist (Static)',
    'cosine_dist_static_transient': 'Cosine Dist (Static, T)',
    'cosine_dist_static_zscore': 'Cosine Dist (Static, Z)'
}

# Consistent ordering for switch/cluster
LABEL_ORDER = ['Cluster', 'Switch']
LABEL_PALETTE = {'Switch': '#e74c3c', 'Cluster': '#3498db'}


# ==========================================
# 3. NORMALIZATION FUNCTIONS
# ==========================================
def compute_transient_metrics(df, metrics_to_normalize, drift=DRIFT):
    """
    Compute transient (normalized) versions of metrics following Kumar et al. (2023).
    
    Formula (Reynolds et al., 2007):
    1. Initialize running average μ₁ = mean of metric across entire category
    2. Update: μₜ = μₜ₋₁ + (Sₜ - μₜ₋₁) × drift
    3. Transient: Sₜ / μₜ₋₁
    """
    df = df.copy()
    
    for metric in metrics_to_normalize:
        if metric not in df.columns:
            continue
        
        values = df[metric].values
        n = len(values)
        
        if n == 0:
            continue
        
        # Initialize running average with mean of entire category
        mu = np.nanmean(values)
        
        transient_values = np.zeros(n)
        
        for t in range(n):
            if mu > 0:
                transient_values[t] = values[t] / mu
            else:
                transient_values[t] = np.nan
            
            if not np.isnan(values[t]):
                mu = mu + (values[t] - mu) * drift
        
        df[f"{metric}_transient"] = transient_values
    
    return df


def compute_zscore_metrics(df, metrics_to_zscore):
    """
    Compute z-scored versions of metrics for comparable visualization.
    
    Formula: z = (x - mean) / std
    
    Z-scoring is done within the category to preserve relative patterns.
    """
    df = df.copy()
    
    for metric in metrics_to_zscore:
        if metric not in df.columns:
            continue
        
        values = df[metric].values
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        
        if std_val > 0:
            df[f"{metric}_zscore"] = (values - mean_val) / std_val
        else:
            df[f"{metric}_zscore"] = 0.0  # All same value
    
    return df


def compute_ylims(all_category_dfs, metrics, padding=0.1):
    """
    Compute consistent y-axis limits across all categories for given metrics.
    """
    all_values = []
    for tdf in all_category_dfs:
        for metric in metrics:
            if metric in tdf.columns:
                all_values.extend(tdf[metric].dropna().values)
    
    if len(all_values) == 0:
        return (0, 1)
    
    ymin, ymax = np.min(all_values), np.max(all_values)
    yrange = ymax - ymin
    
    if yrange == 0:
        yrange = 1
    
    return (ymin - padding * yrange, ymax + padding * yrange)


# ==========================================
# 4. GPT-2 MULTI-TOKEN ANALYZER
# ==========================================
class GPT2MultiTokenAnalyzer:
    """
    Analyzes word lists using GPT-2, handling multi-token words appropriately.
    """
    
    def __init__(self, model_name='gpt2-medium'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device).eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.static_embeddings = self.model.transformer.wte.weight.detach()
        
        print(f"Loaded {model_name} on {self.device}")
        print(f"Vocab size: {len(self.tokenizer)}, Embedding dim: {self.static_embeddings.shape[1]}")
    
    def _get_static_embedding(self, word):
        """Get non-contextualized embedding (MEAN across subtokens)."""
        tokens = self.tokenizer.encode(f" {word}", add_special_tokens=False)
        token_embeddings = self.static_embeddings[tokens]
        return token_embeddings.mean(dim=0)
    
    def _process_word(self, context_ids, word):
        """Process a single word given context token IDs."""
        word_tokens = self.tokenizer.encode(f" {word}", add_special_tokens=False)
        current_ids = context_ids.clone()
        
        total_surprisal = 0.0
        onset_entropy = None
        
        for i, token_id in enumerate(word_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, output_hidden_states=True)
            
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            p_token = probs[token_id].clamp(min=1e-10).item()
            total_surprisal += -np.log(p_token)
            
            if i == 0:
                log_probs = torch.log(probs.clamp(min=1e-10))
                onset_entropy = -torch.sum(probs * log_probs).item()
            
            current_ids = torch.cat(
                [current_ids, torch.tensor([[token_id]], device=self.device)],
                dim=-1
            )
        
        with torch.no_grad():
            final_outputs = self.model(current_ids, output_hidden_states=True)
        
        final_hidden = final_outputs.hidden_states[-1][0, -1, :]
        final_logits = final_outputs.logits[0, -1, :]
        final_prob_dist = F.softmax(final_logits, dim=-1)
        
        return {
            'surprisal': total_surprisal,
            'entropy': onset_entropy,
            'final_prob_dist': final_prob_dist,
            'final_hidden': final_hidden,
            'updated_ids': current_ids
        }
    
    def _compute_bayesian_surprise(self, prev_dist, curr_dist):
        """Compute KL divergence: KL(P_current || P_previous)"""
        kl = F.kl_div(
            prev_dist.log(),
            curr_dist,
            reduction='sum',
            log_target=False
        ).item()
        return kl
    
    def analyze_category(self, word_list):
        """Analyze a single category (first word excluded from output)."""
        results = []
        word_list = [str(w).strip() for w in word_list if str(w).strip()]
        
        if len(word_list) < 2:
            return results
        
        context_ids_global = self.tokenizer.encode(
            self.tokenizer.eos_token, return_tensors="pt"
        ).to(self.device)
        
        first_word = word_list[0]
        first_result = self._process_word(context_ids_global, first_word)
        
        prev_prob_dist = first_result['final_prob_dist']
        prev_hidden_ctx = first_result['final_hidden']
        prev_static_emb = self._get_static_embedding(first_word)
        context_ids_global = first_result['updated_ids']
        
        for i in range(1, len(word_list)):
            word = word_list[i]
            prev_word = word_list[i - 1]
            
            global_result = self._process_word(context_ids_global, word)
            
            bigram_context_ids = self.tokenizer.encode(
                prev_word, return_tensors="pt"
            ).to(self.device)
            bigram_result = self._process_word(bigram_context_ids, word)
            
            static_emb = self._get_static_embedding(word)
            
            results.append({
                'word': word,
                'position': i,
                'surprisal_global': global_result['surprisal'],
                'surprisal_bigram': bigram_result['surprisal'],
                'entropy_global': global_result['entropy'],
                'bayesian_surprise': self._compute_bayesian_surprise(
                    prev_prob_dist, global_result['final_prob_dist']
                ),
                'cosine_dist_contextualized': cosine(
                    global_result['final_hidden'].cpu().numpy(),
                    prev_hidden_ctx.cpu().numpy()
                ),
                'cosine_dist_static': cosine(
                    static_emb.cpu().numpy(),
                    prev_static_emb.cpu().numpy()
                )
            })
            
            prev_prob_dist = global_result['final_prob_dist']
            prev_hidden_ctx = global_result['final_hidden']
            prev_static_emb = static_emb
            context_ids_global = global_result['updated_ids']
        
        return results


# ==========================================
# 5. PLOTTING FUNCTIONS
# ==========================================
def plot_category(tdf, subject, session, category_id, ax_line, ax_bar,
                  plot_type='zscore', ylim_line=None, ylim_bar=None):
    """
    Plot metrics for a single category with consistent y-axes.
    
    Args:
        plot_type: 'raw', 'zscore', or 'transient'
        ylim_line: y-limits for line plot
        ylim_bar: y-limits for bar plot
    """
    if plot_type == 'transient':
        plot_metrics = TRANSIENT_METRICS
        bar_order = TRANSIENT_BAR_ORDER
        baseline = 1.0
        ylabel = "Transient Value"
    elif plot_type == 'zscore':
        plot_metrics = ZSCORED_METRICS
        bar_order = ZSCORED_BAR_ORDER
        baseline = 0.0
        ylabel = "Z-Score"
    else:  # raw
        plot_metrics = RAW_METRICS
        bar_order = RAW_BAR_ORDER
        baseline = None
        ylabel = "Raw Value"
    
    # --- LINE PLOT ---
    x = np.arange(len(tdf))
    
    for metric in plot_metrics:
        if metric in tdf.columns:
            linestyle = '--' if 'bigram' in metric else '-'
            ax_line.plot(
                x, tdf[metric],
                color=COLORS.get(metric, '#333333'),
                label=METRIC_LABELS.get(metric, metric),
                lw=1.5, marker='o', ms=4, linestyle=linestyle
            )
    
    # Highlight switch words
    for idx, row in tdf.iterrows():
        plot_idx = tdf.index.get_loc(idx)
        if row.get('switch_flag', 0) == 1:
            ax_line.axvspan(plot_idx - 0.4, plot_idx + 0.4, color='red', alpha=0.15)
    
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(tdf['word'], rotation=45, ha='right', fontsize=8)
    ax_line.set_xlabel("Words")
    ax_line.set_ylabel(ylabel)
    
    title_suffix = {'raw': '(Raw)', 'zscore': '(Z-Scored)', 'transient': '(Transient)'}
    ax_line.set_title(f"{subject} | {session} | Category {category_id} {title_suffix[plot_type]}")
    ax_line.legend(loc='upper right', fontsize=6, ncol=2)
    ax_line.grid(True, alpha=0.3)
    
    # Apply y-limits for line plot
    if ylim_line is not None:
        ax_line.set_ylim(ylim_line)
    
    if baseline is not None:
        ax_line.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
    
    # --- BAR PLOT ---
    tdf_plot = tdf.copy()
    tdf_plot['label'] = tdf_plot['switch_flag'].map({1: 'Switch', 0: 'Cluster'})
    
    melted = tdf_plot.melt(
        id_vars=['label'],
        value_vars=[m for m in plot_metrics if m in tdf_plot.columns],
        var_name='Metric',
        value_name='Value'
    )
    melted['Metric'] = melted['Metric'].map(lambda x: METRIC_LABELS.get(x, x))
    melted['Metric'] = pd.Categorical(melted['Metric'], categories=bar_order, ordered=True)
    melted = melted.sort_values('Metric')
    
    sns.barplot(
        data=melted, x='Metric', y='Value', hue='label', ax=ax_bar,
        hue_order=LABEL_ORDER,
        palette=LABEL_PALETTE,
        order=bar_order,
        capsize=0.1, errwidth=1.5
    )
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax_bar.set_xlabel("")
    ax_bar.set_ylabel(ylabel)
    ax_bar.set_title("Switch vs Cluster")
    ax_bar.legend(title='', fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis='y')
    
    # Apply y-limits for bar plot
    if ylim_bar is not None:
        ax_bar.set_ylim(ylim_bar)
    
    if baseline is not None:
        ax_bar.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)


def process_session(subject, session, analyzer):
    """Process a single session and generate plots with consistent y-axes."""
    csv_path = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
    if not csv_path:
        print(f"  No CSV found for {subject} {session}")
        return None
    
    print(f"  Loading: {csv_path[0].name}")
    df = pd.read_csv(csv_path[0]).sort_values("start").reset_index(drop=True)
    
    df["category_id"] = (df["transcription"].astype(str).str.lower() == "next").cumsum()
    df_words = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    df_words["switch_flag"] = pd.to_numeric(df_words["switch_flag"], errors='coerce').fillna(0).astype(int)
    
    grouped = list(df_words.groupby("category_id"))
    
    if len(grouped) == 0:
        print(f"  No categories found")
        return None
    
    # === FIRST PASS: Compute all metrics ===
    all_category_dfs = []
    
    for cat_id, tdf in grouped:
        tdf = tdf.reset_index(drop=True)
        word_list = tdf["transcription"].tolist()
        
        print(f"    Category {cat_id}: {len(word_list)} words → {len(word_list)-1} analyzed")
        
        if len(word_list) < 2:
            continue
        
        metrics = analyzer.analyze_category(word_list)
        
        if len(metrics) == 0:
            continue
        
        metrics_df = pd.DataFrame(metrics)
        tdf_out = tdf.iloc[1:].reset_index(drop=True).copy()
        
        for col in metrics_df.columns:
            tdf_out[col] = metrics_df[col].values
        
        # Compute transient and z-scored metrics
        tdf_out = compute_transient_metrics(tdf_out, METRICS_TO_NORMALIZE, drift=DRIFT)
        tdf_out = compute_zscore_metrics(tdf_out, RAW_METRICS)
        
        tdf_out["subject"] = subject
        tdf_out["session"] = session
        tdf_out["category_id"] = cat_id
        
        all_category_dfs.append(tdf_out)
    
    if len(all_category_dfs) == 0:
        print(f"  No valid categories")
        return None
    
    # === SECOND PASS: Create plots (3 rows per category: raw, zscore, transient) ===
    n_categories = len(all_category_dfs)
    fig, axes = plt.subplots(
        n_categories * 3, 2,
        figsize=(18, 5 * n_categories * 3),
        gridspec_kw={'width_ratios': [2.5, 1]},
        squeeze=False
    )
    
    for idx, tdf_out in enumerate(all_category_dfs):
        cat_id = tdf_out["category_id"].iloc[0]
        
        # Plot raw metrics (fixed y-limits)
        plot_category(
            tdf_out, subject, session, cat_id,
            axes[idx * 3, 0], axes[idx * 3, 1],
            plot_type='raw',
            ylim_line=YLIM_RAW_LINE,
            ylim_bar=YLIM_RAW_BAR
        )
        
        # Plot z-scored metrics (fixed y-limits)
        plot_category(
            tdf_out, subject, session, cat_id,
            axes[idx * 3 + 1, 0], axes[idx * 3 + 1, 1],
            plot_type='zscore',
            ylim_line=YLIM_ZSCORE_LINE,
            ylim_bar=YLIM_ZSCORE_BAR
        )
        
        # Plot transient metrics (fixed y-limits)
        plot_category(
            tdf_out, subject, session, cat_id,
            axes[idx * 3 + 2, 0], axes[idx * 3 + 2, 1],
            plot_type='transient',
            ylim_line=YLIM_TRANSIENT_LINE,
            ylim_bar=YLIM_TRANSIENT_BAR
        )
    
    plt.tight_layout()
    out_path = GPT_FIGS_DIR / f"{subject}_{session}_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")
    
    return pd.concat(all_category_dfs, ignore_index=True)

def create_global_summary(grand_df):
    """Create and save global summary plots with subject-level data points."""
    grand_df["label"] = grand_df["switch_flag"].map({1: "Switch", 0: "Cluster"})
    
    # === Compute subject-level means ===
    subject_means = grand_df.groupby(['subject', 'label'])[
        RAW_METRICS + ZSCORED_METRICS + TRANSIENT_METRICS
    ].mean().reset_index()
    
    # Get unique subjects for coloring
    subjects = sorted(grand_df['subject'].unique())
    
    # === Summary statistics ===
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    all_metrics = RAW_METRICS + TRANSIENT_METRICS
    
    for metric in all_metrics:
        if metric not in grand_df.columns:
            continue
        
        switch_vals = grand_df[grand_df['label'] == 'Switch'][metric].dropna()
        cluster_vals = grand_df[grand_df['label'] == 'Cluster'][metric].dropna()
        
        if len(switch_vals) == 0 or len(cluster_vals) == 0:
            continue
        
        t, p = ttest_ind(switch_vals, cluster_vals, nan_policy='omit')
        
        print(f"\n{METRIC_LABELS.get(metric, metric)}:")
        print(f"  Switch:  M = {switch_vals.mean():.4f}, SD = {switch_vals.std():.4f}, n = {len(switch_vals)}")
        print(f"  Cluster: M = {cluster_vals.mean():.4f}, SD = {cluster_vals.std():.4f}, n = {len(cluster_vals)}")
        print(f"  t = {t:.3f}, p = {p:.4f} {'*' if p < 0.05 else ''}")
    
    # === Helper function to plot with connected subject points ===
    def plot_summary_with_subject_lines(ax, melted_data, metrics_list, bar_order, 
                                         ylabel, ylim, baseline=None, title=""):
        """Plot bar chart with subject points connected by lines."""
        
        # Bar plot (grand mean)
        sns.barplot(
            data=melted_data, x='Metric', y='Value', hue='label', ax=ax,
            hue_order=LABEL_ORDER,
            palette=LABEL_PALETTE,
            order=bar_order,
            capsize=0.1, errwidth=1.5,
            alpha=0.6
        )
        
        # Get x-tick positions for each metric
        metric_positions = {metric: i for i, metric in enumerate(bar_order)}
        
        # Width of each bar group and offset for Cluster/Switch
        bar_width = 0.4
        offsets = {'Cluster': -bar_width/2, 'Switch': bar_width/2}
        
        # Plot subject points and connecting lines
        for subject in subjects:
            subject_data = melted_data[melted_data['subject'] == subject]
            color = SUBJECT_COLORS.get(subject, 'black')
            
            for metric_label in bar_order:
                metric_data = subject_data[subject_data['Metric'] == metric_label]
                
                if len(metric_data) < 2:
                    continue
                
                cluster_val = metric_data[metric_data['label'] == 'Cluster']['Value'].values
                switch_val = metric_data[metric_data['label'] == 'Switch']['Value'].values
                
                if len(cluster_val) == 0 or len(switch_val) == 0:
                    continue
                
                cluster_val = cluster_val[0]
                switch_val = switch_val[0]
                
                x_pos = metric_positions[metric_label]
                x_cluster = x_pos + offsets['Cluster']
                x_switch = x_pos + offsets['Switch']
                
                # Draw connecting line
                ax.plot([x_cluster, x_switch], [cluster_val, switch_val],
                        color=color, linewidth=1, alpha=0.7, zorder=3)
                
                # Draw points
                ax.scatter(x_cluster, cluster_val, color=color, s=25, 
                          edgecolor=None, linewidth=0.5, zorder=4)
                ax.scatter(x_switch, switch_val, color=color, s=25,
                          edgecolor=None, linewidth=0.5, zorder=4)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14)
        
        if baseline is not None:
            ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
        
        # Custom legend: Cluster/Switch bars + subject colors
        handles, labels = ax.get_legend_handles_labels()
        subject_handles = [plt.Line2D([0], [0], marker='o', color=SUBJECT_COLORS.get(s, 'black'),
                                       markersize=5, linestyle='-', linewidth=1, label=s)
                           for s in subjects]
        ax.legend(handles[:2] + subject_handles,
                  labels[:2] + list(subjects),
                  title='', fontsize=7, loc='upper right', ncol=3)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(ylim)
    
    # === PLOT 1: Raw metrics ===
    fig, ax = plt.subplots(figsize=(14, 6))
    
    melted_raw = subject_means.melt(
        id_vars=['subject', 'label'],
        value_vars=[m for m in RAW_METRICS if m in subject_means.columns],
        var_name='Metric',
        value_name='Value'
    )
    melted_raw['Metric'] = melted_raw['Metric'].map(lambda x: METRIC_LABELS.get(x, x))
    melted_raw['Metric'] = pd.Categorical(melted_raw['Metric'], categories=RAW_BAR_ORDER, ordered=True)
    
    plot_summary_with_subject_lines(
        ax, melted_raw, RAW_METRICS, RAW_BAR_ORDER,
        ylabel="Raw Value", ylim=YLIM_RAW_BAR, baseline=None,
        title="Global Summary: Switch vs Cluster - Raw Metrics (Subject Means)"
    )
    
    plt.tight_layout()
    plt.savefig(GPT_FIGS_DIR / "GLOBAL_summary_raw.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === PLOT 2: Z-scored metrics ===
    fig, ax = plt.subplots(figsize=(14, 6))
    
    melted_zscore = subject_means.melt(
        id_vars=['subject', 'label'],
        value_vars=[m for m in ZSCORED_METRICS if m in subject_means.columns],
        var_name='Metric',
        value_name='Value'
    )
    melted_zscore['Metric'] = melted_zscore['Metric'].map(lambda x: METRIC_LABELS.get(x, x))
    melted_zscore['Metric'] = pd.Categorical(melted_zscore['Metric'], categories=ZSCORED_BAR_ORDER, ordered=True)
    
    plot_summary_with_subject_lines(
        ax, melted_zscore, ZSCORED_METRICS, ZSCORED_BAR_ORDER,
        ylabel="Z-Score", ylim=(-0.5, 1), baseline=0.0,
        title="Global Summary: Switch vs Cluster - Z-Scored Metrics (Subject Means)"
    )
    
    plt.tight_layout()
    plt.savefig(GPT_FIGS_DIR / "GLOBAL_summary_zscore.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === PLOT 3: Transient metrics ===
    fig, ax = plt.subplots(figsize=(14, 6))
    
    melted_transient = subject_means.melt(
        id_vars=['subject', 'label'],
        value_vars=[m for m in TRANSIENT_METRICS if m in subject_means.columns],
        var_name='Metric',
        value_name='Value'
    )
    melted_transient['Metric'] = melted_transient['Metric'].map(lambda x: METRIC_LABELS.get(x, x))
    melted_transient['Metric'] = pd.Categorical(melted_transient['Metric'], categories=TRANSIENT_BAR_ORDER, ordered=True)
    
    plot_summary_with_subject_lines(
        ax, melted_transient, TRANSIENT_METRICS, TRANSIENT_BAR_ORDER,
        ylabel="Transient Value", ylim=(0, 2), baseline=1.0,
        title=f"Global Summary: Switch vs Cluster - Transient Metrics (Subject Means, drift={DRIFT})"
    )
    
    plt.tight_layout()
    plt.savefig(GPT_FIGS_DIR / "GLOBAL_summary_transient.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === PLOT 4: Boxplots for raw metrics ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(RAW_METRICS):
        if metric not in grand_df.columns or i >= len(axes):
            continue
        
        ax = axes[i]
        metric_data = grand_df[['label', metric]].dropna()
        
        sns.boxplot(
            data=metric_data, x='label', y=metric, ax=ax,
            order=LABEL_ORDER,
            palette=LABEL_PALETTE
        )
        sns.stripplot(
            data=metric_data, x='label', y=metric, ax=ax,
            order=LABEL_ORDER,
            color='black', alpha=0.3, size=2
        )
        
        ax.set_xlabel("")
        ax.set_ylabel("Raw Value")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(YLIM_RAW_BAR)
    
    for j in range(len(RAW_METRICS), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(GPT_FIGS_DIR / "GLOBAL_boxplots_raw.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === PLOT 5: Boxplots for z-scored metrics ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(ZSCORED_METRICS):
        if metric not in grand_df.columns or i >= len(axes):
            continue
        
        ax = axes[i]
        metric_data = grand_df[['label', metric]].dropna()
        
        sns.boxplot(
            data=metric_data, x='label', y=metric, ax=ax,
            order=LABEL_ORDER,
            palette=LABEL_PALETTE
        )
        sns.stripplot(
            data=metric_data, x='label', y=metric, ax=ax,
            order=LABEL_ORDER,
            color='black', alpha=0.3, size=2
        )
        
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("Z-Score")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(YLIM_ZSCORE_BAR)
    
    for j in range(len(ZSCORED_METRICS), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(GPT_FIGS_DIR / "GLOBAL_boxplots_zscore.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === PLOT 6: Boxplots for transient metrics ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(TRANSIENT_METRICS):
        if metric not in grand_df.columns or i >= len(axes):
            continue
        
        ax = axes[i]
        metric_data = grand_df[['label', metric]].dropna()
        
        sns.boxplot(
            data=metric_data, x='label', y=metric, ax=ax,
            order=LABEL_ORDER,
            palette=LABEL_PALETTE
        )
        sns.stripplot(
            data=metric_data, x='label', y=metric, ax=ax,
            order=LABEL_ORDER,
            color='black', alpha=0.3, size=2
        )
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("Transient Value")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(YLIM_TRANSIENT_BAR)
    
    for j in range(len(TRANSIENT_METRICS), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(GPT_FIGS_DIR / "GLOBAL_boxplots_transient.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === Per-subject summary plots ===
    create_per_subject_summaries(grand_df, subjects)
    
    print(f"\nSaved summary plots to {GPT_FIGS_DIR}")


def create_per_subject_summaries(grand_df, subjects):
    """Create summary bar plots for each subject across their sessions."""
    
    def plot_subject_summary_with_session_lines(ax, melted_data, sessions, session_color_map,
                                                  bar_order, ylabel, ylim, baseline=None, title=""):
        """Plot bar chart with session points connected by lines for a single subject."""
        
        # Bar plot (subject mean across sessions)
        sns.barplot(
            data=melted_data, x='Metric', y='Value', hue='label', ax=ax,
            hue_order=LABEL_ORDER,
            palette=LABEL_PALETTE,
            order=bar_order,
            capsize=0.1, errwidth=1.5,
            alpha=0.6
        )
        
        # Get x-tick positions for each metric
        metric_positions = {metric: i for i, metric in enumerate(bar_order)}
        
        # Width of each bar group and offset for Cluster/Switch
        bar_width = 0.4
        offsets = {'Cluster': -bar_width/2, 'Switch': bar_width/2}
        
        # Plot session points and connecting lines
        for session in sessions:
            session_data = melted_data[melted_data['session'] == session]
            color = session_color_map[session]
            
            for metric_label in bar_order:
                metric_data = session_data[session_data['Metric'] == metric_label]
                
                if len(metric_data) < 2:
                    continue
                
                cluster_val = metric_data[metric_data['label'] == 'Cluster']['Value'].values
                switch_val = metric_data[metric_data['label'] == 'Switch']['Value'].values
                
                if len(cluster_val) == 0 or len(switch_val) == 0:
                    continue
                
                cluster_val = cluster_val[0]
                switch_val = switch_val[0]
                
                x_pos = metric_positions[metric_label]
                x_cluster = x_pos + offsets['Cluster']
                x_switch = x_pos + offsets['Switch']
                
                # Draw connecting line
                ax.plot([x_cluster, x_switch], [cluster_val, switch_val],
                        color=color, linewidth=1, alpha=0.7, zorder=3)
                
                # Draw points
                ax.scatter(x_cluster, cluster_val, color=color, s=25,
                          edgecolor=None, linewidth=0.5, zorder=4)
                ax.scatter(x_switch, switch_val, color=color, s=25,
                          edgecolor=None, linewidth=0.5, zorder=4)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14)
        
        if baseline is not None:
            ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
        
        # Custom legend
        handles, labels = ax.get_legend_handles_labels()
        session_handles = [plt.Line2D([0], [0], marker='o', color=session_color_map[s],
                                       markersize=5, linestyle='-', linewidth=1, label=s)
                           for s in sessions]
        ax.legend(handles[:2] + session_handles,
                  labels[:2] + list(sessions),
                  title='', fontsize=7, loc='upper right', ncol=3)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(ylim)
    
    for subject in subjects:
        subject_df = grand_df[grand_df['subject'] == subject].copy()
        
        if len(subject_df) == 0:
            continue
        
        # Get sessions for this subject
        sessions = sorted(subject_df['session'].unique())
        
        # Compute session-level means
        session_means = subject_df.groupby(['session', 'label'])[
            RAW_METRICS + ZSCORED_METRICS + TRANSIENT_METRICS
        ].mean().reset_index()
        
        # Session colors
        session_colors = plt.cm.tab10(np.linspace(0, 1, max(len(sessions), 1)))
        session_color_map = {s: session_colors[i] for i, s in enumerate(sessions)}
        
        # === Raw metrics for this subject ===
        fig, ax = plt.subplots(figsize=(14, 6))
        
        melted = session_means.melt(
            id_vars=['session', 'label'],
            value_vars=[m for m in RAW_METRICS if m in session_means.columns],
            var_name='Metric',
            value_name='Value'
        )
        melted['Metric'] = melted['Metric'].map(lambda x: METRIC_LABELS.get(x, x))
        melted['Metric'] = pd.Categorical(melted['Metric'], categories=RAW_BAR_ORDER, ordered=True)
        
        plot_subject_summary_with_session_lines(
            ax, melted, sessions, session_color_map, RAW_BAR_ORDER,
            ylabel="Raw Value", ylim=YLIM_RAW_BAR, baseline=None,
            title=f"{subject}: Switch vs Cluster - Raw Metrics (Session Means)"
        )
        
        plt.tight_layout()
        plt.savefig(GPT_FIGS_DIR / f"{subject}_summary_raw.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # === Z-scored metrics for this subject ===
        fig, ax = plt.subplots(figsize=(14, 6))
        
        melted = session_means.melt(
            id_vars=['session', 'label'],
            value_vars=[m for m in ZSCORED_METRICS if m in session_means.columns],
            var_name='Metric',
            value_name='Value'
        )
        melted['Metric'] = melted['Metric'].map(lambda x: METRIC_LABELS.get(x, x))
        melted['Metric'] = pd.Categorical(melted['Metric'], categories=ZSCORED_BAR_ORDER, ordered=True)
        
        plot_subject_summary_with_session_lines(
            ax, melted, sessions, session_color_map, ZSCORED_BAR_ORDER,
            ylabel="Z-Score", ylim=YLIM_ZSCORE_BAR, baseline=0.0,
            title=f"{subject}: Switch vs Cluster - Z-Scored Metrics (Session Means)"
        )
        
        plt.tight_layout()
        plt.savefig(GPT_FIGS_DIR / f"{subject}_summary_zscore.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # === Transient metrics for this subject ===
        fig, ax = plt.subplots(figsize=(14, 6))
        
        melted = session_means.melt(
            id_vars=['session', 'label'],
            value_vars=[m for m in TRANSIENT_METRICS if m in session_means.columns],
            var_name='Metric',
            value_name='Value'
        )
        melted['Metric'] = melted['Metric'].map(lambda x: METRIC_LABELS.get(x, x))
        melted['Metric'] = pd.Categorical(melted['Metric'], categories=TRANSIENT_BAR_ORDER, ordered=True)
        
        plot_subject_summary_with_session_lines(
            ax, melted, sessions, session_color_map, TRANSIENT_BAR_ORDER,
            ylabel="Transient Value", ylim=YLIM_TRANSIENT_BAR, baseline=1.0,
            title=f"{subject}: Switch vs Cluster - Transient Metrics (Session Means, drift={DRIFT})"
        )
        
        plt.tight_layout()
        plt.savefig(GPT_FIGS_DIR / f"{subject}_summary_transient.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved summary plots for {subject}")


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-2 Semantic Fluency Analysis with Transient Normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python script.py --sub sub-01          # Process specific subject
    python script.py --all                 # Process all subjects
    python script.py --test                # Run test with sample data
        """
    )
    parser.add_argument("--sub", type=str, help="Specific subject ID (e.g., sub-01)")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--test", action="store_true", help="Run test with sample data")
    parser.add_argument("--model", type=str, default="gpt2-medium",
                        help="GPT-2 model variant (default: gpt2-medium)")
    parser.add_argument("--drift", type=float, default=DRIFT,
                        help=f"Drift parameter for transient normalization (default: {DRIFT})")
    args = parser.parse_args()
    
    if args.drift != DRIFT:
        DRIFT = args.drift
        print(f"Using drift = {DRIFT}")

    # --- TEST MODE ---
    if args.test:
        print("=" * 60)
        print("RUNNING TEST WITH SAMPLE DATA")
        print("=" * 60)
        
        analyzer = GPT2MultiTokenAnalyzer(model_name=args.model)
        
        test_words = ["dog", "cat", "mouse", "elephant", "apple", "banana", "strawberry"]
        print(f"\nTest words: {test_words}")
        print("(Simulated switch at 'apple')")
        
        results = analyzer.analyze_category(test_words)
        results_df = pd.DataFrame(results)
        
        results_df = compute_transient_metrics(results_df, METRICS_TO_NORMALIZE, drift=DRIFT)
        results_df = compute_zscore_metrics(results_df, RAW_METRICS)
        
        print(f"\nResults (first word '{test_words[0]}' excluded):")
        print("-" * 110)
        
        for _, row in results_df.iterrows():
            print(f"\n{row['word']} (position {row['position']}):")
            print(f"  {'Metric':<30} {'Raw':>12} {'Z-Score':>12} {'Transient':>12}")
            print(f"  {'-'*66}")
            for metric in RAW_METRICS:
                raw_val = row[metric]
                z_val = row.get(f"{metric}_zscore", np.nan)
                trans_val = row.get(f"{metric}_transient", np.nan)
                print(f"  {METRIC_LABELS[metric]:<30} {raw_val:>12.4f} {z_val:>12.4f} {trans_val:>12.4f}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        sys.exit(0)

    # --- MAIN ANALYSIS ---
    print("=" * 60)
    print("GPT-2 SEMANTIC FLUENCY ANALYSIS")
    print(f"Transient normalization drift = {DRIFT}")
    print("=" * 60)
    
    analyzer = GPT2MultiTokenAnalyzer(model_name=args.model)
    
    if args.all:
        all_data = []
        
        for sub_dir in sorted(DERIVATIVES_DIR.glob("sub-*")):
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                print(f"\nProcessing: {sub_dir.name} / {ses_dir.name}")
                result = process_session(sub_dir.name, ses_dir.name, analyzer)
                if result is not None:
                    all_data.append(result)
        
        if all_data:
            grand_df = pd.concat(all_data, ignore_index=True)
            
            csv_path = GPT_FIGS_DIR / "all_sessions_metrics.csv"
            grand_df.to_csv(csv_path, index=False)
            print(f"\nSaved data: {csv_path}")
            
            create_global_summary(grand_df)
            
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print(f"Results saved to: {GPT_FIGS_DIR}")
            print("=" * 60)
        else:
            print("\nNo data found to process.")
    
    elif args.sub:
        for ses_path in sorted(ANNOTATIONS_DIR.glob(f"{args.sub}_ses-*wordtimestamps*.csv")):
            ses_id = ses_path.name.split("_")[1]
            print(f"\nProcessing: {args.sub} / {ses_id}")
            process_session(args.sub, ses_id, analyzer)
    
    else:
        parser.print_help()
        print("\nPlease specify --sub <subject_id>, --all, or --test")