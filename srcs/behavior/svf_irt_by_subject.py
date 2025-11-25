from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define directories
annotations_dir = Path("/Users/gioli/projects/stateswitch/data/rec/svf_annotated/")
outdir = Path("/Users/gioli/projects/stateswitch/figs/svf_figs_subjects")
outdir.mkdir(exist_ok=True)

# === CONFIGURATION ===
LONG_IRT_THRESHOLD = 20.0
PERI_SWITCH_WINDOW = 3 
# Range: -3 to 2 (Total 6 positions)
# Pre: -3, -2, -1 (3 words)
# Post: 0, 1, 2 (3 words)
OFFSETS = np.arange(-PERI_SWITCH_WINDOW, PERI_SWITCH_WINDOW) 

# === DATA AGGREGATION STRUCTURE ===
subject_data = {}

print(f"Starting Subject-Level Analysis (Window: {-PERI_SWITCH_WINDOW} to {PERI_SWITCH_WINDOW - 1})...")

for fp in sorted(annotations_dir.glob("*.csv")):
    parts = fp.stem.split("_")
    subject = parts[0] if len(parts) > 0 else "NA"
    session = parts[1] if len(parts) > 1 else "NA"

    if subject not in subject_data:
        subject_data[subject] = {
            "cluster_irts": [],
            "switch_irts": [],
            "cluster_sizes": [],
            "peri_switch": {k: [] for k in OFFSETS},
            "n_total_words": 0,
            "n_cluster_count": 0,
            "n_switch_count": 0,
            "n_categories": 0,
            "session_count": 0,
            "n_valid_pre_seqs": 0,  
            "n_valid_post_seqs": 0  
        }

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

    # === UPDATE BASIC COUNTS ===
    subject_data[subject]["session_count"] += 1
    subject_data[subject]["n_total_words"] += len(transcriptions_filtered)
    subject_data[subject]["n_categories"] += len(np.unique(cat_ids_filtered))

    # === 1. CLUSTER SIZES ===
    session_cluster_sizes = []
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
        session_cluster_sizes.extend(lengths)
    
    subject_data[subject]["cluster_sizes"].extend(session_cluster_sizes)

    # === 2. PERI-SWITCH IRTS (SYMMETRIC 3-WORD CHECK) ===
    global_switch_indices = np.where(switch_flags_filtered == 1)[0]
    
    for idx in global_switch_indices:
        
        # --- CHECK LEFT SIDE (Pre-Switch: -3 to -1) ---
        pre_start = idx - PERI_SWITCH_WINDOW # idx - 3
        left_valid = False
        
        if pre_start >= 0:
            # Must be same category
            if np.all(cat_ids_filtered[pre_start : idx+1] == cat_ids_filtered[idx]):
                # Must be NO switches in the window preceding idx (idx-3 to idx-1)
                pre_window_switches = switch_flags_filtered[pre_start : idx]
                if not np.any(pre_window_switches == 1):
                    left_valid = True
        
        if left_valid:
            subject_data[subject]["n_valid_pre_seqs"] += 1
            for k in range(-PERI_SWITCH_WINDOW, 0): # -3, -2, -1
                val = irts_filtered[idx + k]
                if not np.isnan(val):
                    subject_data[subject]["peri_switch"][k].append(val)
        
        # --- CHECK RIGHT SIDE (Post-Switch: 0 to +2) ---
        # Window: 0, 1, 2 (Total 3 positions)
        post_end = idx + PERI_SWITCH_WINDOW - 1 # idx + 2
        right_valid = False
        
        if post_end < len(irts_filtered):
            # Must be same category
            if np.all(cat_ids_filtered[idx : post_end+1] == cat_ids_filtered[idx]):
                # Must be NO switches in the window following idx (idx+1 to idx+2)
                # idx is the switch (start of cluster). We need 2 more non-switches.
                post_window_switches = switch_flags_filtered[idx+1 : post_end+1]
                if not np.any(post_window_switches == 1):
                    right_valid = True
                    
        if right_valid:
            subject_data[subject]["n_valid_post_seqs"] += 1
            for k in range(0, PERI_SWITCH_WINDOW): # 0, 1, 2
                val = irts_filtered[idx + k]
                if not np.isnan(val):
                    subject_data[subject]["peri_switch"][k].append(val)

    # === SEPARATE IRTs ===
    cluster_mask = switch_flags_filtered == 0
    switch_mask = switch_flags_filtered == 1
    subject_data[subject]["n_cluster_count"] += np.sum(cluster_mask)
    subject_data[subject]["n_switch_count"] += np.sum(switch_mask)
    
    cluster_vals = irts_filtered[cluster_mask]
    switch_vals = irts_filtered[switch_mask]
    subject_data[subject]["cluster_irts"].extend(cluster_vals[~np.isnan(cluster_vals)])
    subject_data[subject]["switch_irts"].extend(switch_vals[~np.isnan(switch_vals)])

print("\n=== DATA QUALITY REPORT ===")

# === GENERATE PLOTS ===
for subj, data in subject_data.items():
    cluster_irts = np.array(data["cluster_irts"])
    switch_irts = np.array(data["switch_irts"])
    cluster_sizes = np.array(data["cluster_sizes"])
    
    n_pre = data["n_valid_pre_seqs"]
    n_post = data["n_valid_post_seqs"]
    
    print(f"Subject {subj}:")
    print(f"  - Total Clusters: {len(cluster_sizes)}")
    print(f"  - Valid Pre-Switch (-3 to -1): {n_pre}")
    print(f"  - Valid Post-Switch (0 to 2): {n_post}")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    
    fig.suptitle(f"{subj} (Aggregated {data['session_count']} sessions): {data['n_total_words']} words", fontsize=16)
    
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
    ax2.set_xticks(np.log([0.1, 0.5, 1, 2, 5, 10, 20]))
    ax2.set_xticklabels(["0.1", "0.5", "1", "2", "5", "10", "20"])
    ax2.set_title("IRT Distribution (Log)")
    
    # 3. Log Boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    bp_data = [np.log(cluster_irts+epsilon), np.log(switch_irts+epsilon)]
    ax3.boxplot(bp_data, tick_labels=["Cluster", "Switch"], patch_artist=True, boxprops=dict(facecolor="#A0C4FF"))
    ax3.set_yticks(np.log([0.1, 0.5, 1, 2, 5, 10, 20]))
    ax3.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20"])
    ax3.set_title("IRT Boxplot (Log)")
    
    # 4. Cluster Size
    ax4 = fig.add_subplot(gs[1, 0])
    if len(cluster_sizes) > 0:
        max_size = int(max(cluster_sizes)) if len(cluster_sizes) > 0 else 5
        size_bins = np.arange(0.5, max_size + 1.5, 1)
        ax4.hist(cluster_sizes, bins=size_bins, color="purple", alpha=0.5, edgecolor='black')
        
        # Draw lines for threshold requirements
        # Both now require size >= 3
        ax4.axvline(PERI_SWITCH_WINDOW, color='blue', linestyle='--', alpha=0.7, linewidth=2, label=f"Req. for Pre/Post (≥{PERI_SWITCH_WINDOW})")
        
        ax4.set_xlabel("Words in Cluster")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.set_title(f"Cluster Sizes (Total n={len(cluster_sizes)})")
        
    # 5. Peri-Switch Dynamics
    ax5 = fig.add_subplot(gs[1, 1:])
    
    means = []
    sems = []
    valid_offsets = []
    
    # We plot all offsets
    for k in OFFSETS:
        vals = np.array(data["peri_switch"][k])
        if len(vals) > 0:
            valid_offsets.append(k)
            means.append(np.mean(vals))
            if len(vals) > 1:
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                sems.append(0)
    
    if valid_offsets:
        ax5.errorbar(valid_offsets, means, yerr=sems, fmt='-o', capsize=5, color="#D62828", linewidth=2)
        ax5.set_xlabel("Position Relative to Switch Word (0)")
        ax5.set_ylabel("Mean IRT (s)")
        
        ax5.set_title(f"IRT Dynamics\nPre n={n_pre} | Post n={n_post}")
        
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-PERI_SWITCH_WINDOW - 0.5, PERI_SWITCH_WINDOW - 0.5) # -3.5 to 2.5
        ax5.set_xticks(OFFSETS)
        ax5.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # Add text annotation
        ax5.text(0.02, 0.95, f"Filtering criteria:\nLeft: Prev Cluster ≥ {PERI_SWITCH_WINDOW}\nRight: Curr Cluster ≥ {PERI_SWITCH_WINDOW}", 
                 transform=ax5.transAxes, fontsize=9, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig_path = outdir / f"{subj}_AllSessions_IRT_Advanced.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
print("All subject plots saved.")