from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define directories
annotations_dir = Path("/Users/gioli/projects/stateswitch/data/rec/svf_annotated/")
outdir = Path("/Users/gioli/projects/stateswitch/figs/svf_figs")
outdir.mkdir(exist_ok=True)

LONG_IRT_THRESHOLD = 20.0

# === CONFIGURATION ===
PERI_SWITCH_WINDOW = 3  # Look at -6 to +6
MIN_SAMPLES_FOR_PLOT = 1  # Show point even if we only have 1 instance

for fp in sorted(annotations_dir.glob("*.csv")):
    # Extract subject and session info
    parts = fp.stem.split("_")
    subject = parts[0] if len(parts) > 0 else "NA"
    session = parts[1] if len(parts) > 1 else "NA"

    # Load CSV
    df = pd.read_csv(fp)
    
    # === CONVERT TO NUMPY ARRAYS ===
    transcriptions = df["transcription"].to_numpy()
    starts = pd.to_numeric(df["start"], errors="coerce").to_numpy()
    ends = pd.to_numeric(df["end"], errors="coerce").to_numpy()
    switch_flags = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
    
    # === IDENTIFY CATEGORY BLOCKS ===
    is_next = np.char.lower(transcriptions.astype(str)) == "next"
    cat_ids = np.cumsum(is_next)
    
    # === COMPUTE IRTs ===
    n_items = len(transcriptions)
    irts = np.full(n_items, np.nan)
    
    unique_cats = np.unique(cat_ids)
    for cat in unique_cats:
        cat_mask = (cat_ids == cat)
        cat_indices = np.where(cat_mask)[0]
        
        for i in range(1, len(cat_indices)):
            prev_idx = cat_indices[i - 1]
            curr_idx = cat_indices[i]
            
            if is_next[prev_idx] or is_next[curr_idx]:
                continue
            
            if not np.isnan(starts[curr_idx]) and not np.isnan(ends[prev_idx]):
                irts[curr_idx] = starts[curr_idx] - ends[prev_idx]
    
    # === FILTER ARRAYS ===
    word_mask = ~is_next
    transcriptions_filtered = transcriptions[word_mask]
    irts_filtered = irts[word_mask]
    switch_flags_filtered = switch_flags[word_mask]
    cat_ids_filtered = cat_ids[word_mask]
    
    irts_filtered = np.clip(irts_filtered, 0, None)

    # === 1. CLUSTER SIZES ===
    cluster_sizes = []
    for cat in np.unique(cat_ids_filtered):
        cat_mask = cat_ids_filtered == cat
        cat_switches = switch_flags_filtered[cat_mask]
        
        if len(cat_switches) == 0:
            continue
            
        temp_switches = cat_switches.copy()
        temp_switches[0] = 1 
        switch_indices = np.where(temp_switches == 1)[0]
        boundaries = np.concatenate([switch_indices, [len(cat_switches)]])
        lengths = np.diff(boundaries)
        cluster_sizes.extend(lengths)
        
    cluster_sizes = np.array(cluster_sizes)

    # === 2. PERI-SWITCH IRTS ===
    offsets = np.arange(-PERI_SWITCH_WINDOW, PERI_SWITCH_WINDOW + 1)
    peri_switch_data = {k: [] for k in offsets}
    
    global_switch_indices = np.where(switch_flags_filtered == 1)[0]
    
    for idx in global_switch_indices:
        current_cat = cat_ids_filtered[idx]
        
        for k in offsets:
            target_idx = idx + k
            
            if target_idx < 0 or target_idx >= len(irts_filtered):
                continue
            
            if cat_ids_filtered[target_idx] != current_cat:
                continue
            
            is_valid_sequence = True
            if k < 0:
                intervening = switch_flags_filtered[target_idx+1 : idx]
                if np.any(intervening == 1):
                    is_valid_sequence = False
            elif k > 0:
                intervening = switch_flags_filtered[idx+1 : target_idx+1]
                if np.any(intervening == 1):
                    is_valid_sequence = False
            
            if is_valid_sequence:
                val = irts_filtered[target_idx]
                if not np.isnan(val):
                    peri_switch_data[k].append(val)

    # === PREPARE STATS FOR HEADER & PLOTS ===
    cluster_mask = switch_flags_filtered == 0
    switch_mask = switch_flags_filtered == 1
    
    # Get raw counts (including those that might have NaN IRTs, though rare)
    n_total = len(transcriptions_filtered)
    n_cluster_count = np.sum(cluster_mask)
    n_switch_count = np.sum(switch_mask)
    n_cats = len(np.unique(cat_ids_filtered))
    
    # Get IRT data for plotting (removing NaNs)
    cluster_irts = irts_filtered[cluster_mask]
    switch_irts = irts_filtered[switch_mask]
    cluster_irts = cluster_irts[~np.isnan(cluster_irts)]
    switch_irts = switch_irts[~np.isnan(switch_irts)]
    
    # === PLOTTING ===
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    
    # === RESTORED INFORMATIVE HEADER ===
    fig.suptitle(f"{subject} {session}: {n_total} words ({n_cluster_count} cluster, {n_switch_count} switch) | {n_cats} categories", fontsize=16)
    
    # 1. Linear Hist
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 20, 21)
    ax1.hist(np.clip(cluster_irts, 0, 20), bins=bins, alpha=0.6, label="Cluster")
    ax1.hist(np.clip(switch_irts, 0, 20), bins=bins, alpha=0.6, label="Switch")
    ax1.set_title("IRT Distribution (Linear)")
    ax1.legend()
    
    # 2. Log Hist
    ax2 = fig.add_subplot(gs[0, 1])
    epsilon = 0.001
    log_bins = np.linspace(-3, 3, 21)
    ax2.hist(np.log(cluster_irts + epsilon), bins=log_bins, alpha=0.6, label="Cluster")
    ax2.hist(np.log(switch_irts + epsilon), bins=log_bins, alpha=0.6, label="Switch")
    tick_values = [0.1, 0.5, 1, 2, 5, 10, 20]
    ax2.set_xticks(np.log(tick_values))
    ax2.set_xticklabels([str(v) for v in tick_values])
    ax2.set_title("IRT Distribution (Log)")
    
    # 3. Log Boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    bp_data = [np.log(cluster_irts+epsilon), np.log(switch_irts+epsilon)]
    ax3.boxplot(bp_data, tick_labels=["Cluster", "Switch"], patch_artist=True, boxprops=dict(facecolor="#A0C4FF"))
    ax3.set_yticks(np.log(tick_values))
    ax3.set_yticklabels([str(v) for v in tick_values])
    ax3.set_title("IRT Boxplot (Log)")
    
    # 4. Cluster Size
    ax4 = fig.add_subplot(gs[1, 0])
    if len(cluster_sizes) > 0:
        max_size = int(max(cluster_sizes)) if len(cluster_sizes) > 0 else 5
        size_bins = np.arange(0.5, max_size + 1.5, 1)
        ax4.hist(cluster_sizes, bins=size_bins, color="purple", alpha=0.5, edgecolor='black')
        ax4.set_xlabel("Words in Cluster")
        ax4.set_ylabel("Frequency")
        ax4.set_title(f"Cluster Size Dist (Mean: {np.mean(cluster_sizes):.1f})")
        
    # 5. Peri-Switch Dynamics
    ax5 = fig.add_subplot(gs[1, 1:])
    
    means = []
    sems = []
    valid_offsets = []
    counts = []
    
    for k in offsets:
        vals = np.array(peri_switch_data[k])
        if len(vals) >= MIN_SAMPLES_FOR_PLOT:
            valid_offsets.append(k)
            means.append(np.mean(vals))
            if len(vals) > 1:
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                sems.append(0)
            counts.append(len(vals))
    
    if valid_offsets:
        ax5.errorbar(valid_offsets, means, yerr=sems, fmt='-o', capsize=5, color="#D62828", linewidth=2)
        ax5.set_xlabel("Position Relative to Switch Word (0)")
        ax5.set_ylabel("Mean IRT (s)")
        ax5.set_title("IRT Dynamics Before/After Switching")
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-PERI_SWITCH_WINDOW - 0.5, PERI_SWITCH_WINDOW + 0.5)
        ax5.set_xticks(offsets)
        
        # Annotations (Higher up)
        for i, txt in enumerate(counts):
            bar_top = means[i] + sems[i]
            ax5.annotate(f"n={txt}", (valid_offsets[i], bar_top), 
                         xytext=(0, 15), textcoords='offset points', 
                         ha='center', fontsize=8)
            
        ax5.axvline(0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    
    fig_path = outdir / f"{subject}_{session}_IRT_Advanced.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print(f"Processed {subject} {session}: Saved updated analysis.")