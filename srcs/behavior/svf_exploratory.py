from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define directories
annotations_dir = Path("/Users/gioli/projects/stateswitch/data/rec/svf_annotated/")
outdir = Path("/Users/gioli/projects/stateswitch/figs/svf_figs")
outdir.mkdir(exist_ok=True)

LONG_IRT_THRESHOLD = 20.0

for fp in sorted(annotations_dir.glob("*.csv")):
    # Extract subject and session info from filename
    parts = fp.stem.split("_")
    subject = parts[0] if len(parts) > 0 else "NA"
    session = parts[1] if len(parts) > 1 else "NA"

    # Load CSV and immediately convert to NumPy arrays
    df = pd.read_csv(fp)
    
    # === CONVERT TO NUMPY ARRAYS ===
    # Extract all columns as separate numpy arrays
    transcriptions = df["transcription"].to_numpy()
    starts = pd.to_numeric(df["start"], errors="coerce").to_numpy()
    ends = pd.to_numeric(df["end"], errors="coerce").to_numpy()
    switch_flags = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
    
    # === IDENTIFY CATEGORY BLOCKS ===
    # Find where "next" appears (case-insensitive)
    is_next = np.char.lower(transcriptions.astype(str)) == "next"
    # Create category IDs by cumulative sum of "next" occurrences
    cat_ids = np.cumsum(is_next)
    
    # === COMPUTE IRTs USING NUMPY ===
    n_items = len(transcriptions)
    irts = np.full(n_items, np.nan)  # Initialize IRT array with NaN
    prev_words = np.empty(n_items, dtype=object)  # Store previous words
    
    # Process each category block
    unique_cats = np.unique(cat_ids)
    for cat in unique_cats:
        # Get indices for this category
        cat_mask = (cat_ids == cat)
        cat_indices = np.where(cat_mask)[0]
        
        # Calculate IRTs within this category
        for i in range(1, len(cat_indices)):
            prev_idx = cat_indices[i - 1]
            curr_idx = cat_indices[i]
            
            # Skip if either is "next"
            if is_next[prev_idx] or is_next[curr_idx]:
                continue
            
            # Calculate IRT
            if not np.isnan(starts[curr_idx]) and not np.isnan(ends[prev_idx]):
                irts[curr_idx] = starts[curr_idx] - ends[prev_idx]
                prev_words[curr_idx] = transcriptions[prev_idx]
    
    # === REMOVE "NEXT" ENTRIES ===
    # Create mask for actual words (not "next")
    word_mask = ~is_next
    
    # Filter arrays to keep only actual words
    transcriptions_filtered = transcriptions[word_mask]
    irts_filtered = irts[word_mask]
    switch_flags_filtered = switch_flags[word_mask]
    cat_ids_filtered = cat_ids[word_mask]
    prev_words_filtered = prev_words[word_mask]
    
    # Clip negative IRTs to 0
    irts_filtered = np.clip(irts_filtered, 0, None)
    
    # === FLAG LONG IRTs ===
    long_irt_mask = irts_filtered > LONG_IRT_THRESHOLD
    if np.any(long_irt_mask):
        print(f"\n{subject} {session}: Long IRTs (> {LONG_IRT_THRESHOLD:.1f}s)")
        long_indices = np.where(long_irt_mask)[0]
        for idx in long_indices:
            print(f"  {prev_words_filtered[idx]!r} → {transcriptions_filtered[idx]!r} | "
                  f"cat={cat_ids_filtered[idx]} | switch={switch_flags_filtered[idx]} | "
                  f"IRT={irts_filtered[idx]:.2f}s")
    
    # === SEPARATE IRTs BY SWITCH TYPE ===
    # Use boolean indexing to separate cluster and switch IRTs
    cluster_mask = switch_flags_filtered == 0
    switch_mask = switch_flags_filtered == 1
    
    cluster_irts = irts_filtered[cluster_mask]
    switch_irts = irts_filtered[switch_mask]
    
    # Remove NaN values
    cluster_irts = cluster_irts[~np.isnan(cluster_irts)]
    switch_irts = switch_irts[~np.isnan(switch_irts)]
    
    # === CALCULATE STATISTICS FOR TITLE ===
    n_words = len(transcriptions_filtered)
    n_cluster = len(cluster_irts)
    n_switch = len(switch_irts)
    n_categories = len(np.unique(cat_ids_filtered))
    
    # === CREATE 4-SUBPLOT FIGURE ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{subject} {session}: {n_words} words ({n_cluster} cluster, {n_switch} switch), {n_categories} categories", 
                 fontsize=14)
    
    # === SUBPLOT 1: LINEAR HISTOGRAM ===
    ax1 = axes[0]
    # Create bins with explicit 20+ bin
    bins = np.linspace(0, 20, 21)  # 0-20 in 1s bins
    
    # Clip values > 20 to 20 for visualization
    cluster_irts_clipped = np.clip(cluster_irts, 0, 20)
    switch_irts_clipped = np.clip(switch_irts, 0, 20)
    
    ax1.hist(cluster_irts_clipped, bins=bins, alpha=0.6, label="Cluster")
    ax1.hist(switch_irts_clipped, bins=bins, alpha=0.6, label="Switch")
    ax1.set_xlabel("IRT (s)")
    ax1.set_ylabel("Count")
    ax1.set_title("IRT Distribution (Linear)")
    ax1.set_xlim(0, 20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === SUBPLOT 2: LOG-TRANSFORMED HISTOGRAM ===
    ax2 = axes[1]
    # Log-transform IRTs (add small constant to avoid log(0))
    epsilon = 0.001
    cluster_log_irts = np.log(cluster_irts + epsilon)
    switch_log_irts = np.log(switch_irts + epsilon)
    
    # Create bins for log-transformed data
    log_bins = np.linspace(-3, 3, 21)  # log(0.01) to log(20)
    
    ax2.hist(cluster_log_irts, bins=log_bins, alpha=0.6, label="Cluster")
    ax2.hist(switch_log_irts, bins=log_bins, alpha=0.6, label="Switch")
    
    # Custom x-axis labels
    tick_values = [0.1, 0.5, 1, 2, 5, 10, 20]
    tick_positions = np.log(tick_values)
    tick_labels = [f"{v}" for v in tick_values]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlim(-3, 3)
    
    ax2.set_xlabel("IRT (s) [log scale]")
    ax2.set_ylabel("Count")
    ax2.set_title("IRT Distribution (Log)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === SUBPLOT 3: LINEAR BOXPLOT ===
    ax3 = axes[2]
    # Clip data at 20s for visualization
    data_clipped = [np.clip(cluster_irts, 0, 20), np.clip(switch_irts, 0, 20)]
    bp1 = ax3.boxplot(data_clipped, labels=["Cluster", "Switch"], patch_artist=True,
                      boxprops=dict(facecolor="#A0C4FF"), 
                      medianprops=dict(color="black", linewidth=2))
    ax3.set_ylabel("IRT (s)")
    ax3.set_title("IRT Boxplot (Linear)")
    ax3.set_ylim(-1, 25)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === SUBPLOT 4: LOG-TRANSFORMED BOXPLOT ===
    ax4 = axes[3]
    # Log-transform data for boxplot
    data_log = [cluster_log_irts, switch_log_irts]
    bp2 = ax4.boxplot(data_log, labels=["Cluster", "Switch"], patch_artist=True,
                      boxprops=dict(facecolor="#A0C4FF"), 
                      medianprops=dict(color="black", linewidth=2))
    
    # Set y-axis to log scale with custom labels
    ax4.set_yticks(tick_positions)
    ax4.set_yticklabels(tick_labels)
    ax4.set_ylim(-4, 4)
    
    ax4.set_ylabel("IRT (s) [log scale]")
    ax4.set_title("IRT Boxplot (Log)")
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    
    # Save figure
    fig_path = outdir / f"{subject}_{session}_IRT_analysis.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # === PRINT SUMMARY STATISTICS ===
    cluster_mean = np.mean(cluster_irts) if len(cluster_irts) > 0 else np.nan
    switch_mean = np.mean(switch_irts) if len(switch_irts) > 0 else np.nan
    
    print(f"{subject} {session}: mean IRT — "
          f"switch={switch_mean:.2f}s, "
          f"cluster={cluster_mean:.2f}s, "
          f"n={n_words} words ({n_cluster} cluster, {n_switch} switch), "
          f"{n_categories} categories")
    print(f"Saved → {fig_path.name}")