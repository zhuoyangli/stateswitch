from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define directories
annotations_dir = Path("/Users/gioli/projects/stateswitch/data/rec/svf_annotated/")
outdir = Path("/Users/gioli/projects/stateswitch/figs/svf_figs_subjects_multisession")
outdir.mkdir(exist_ok=True)

# === CONFIGURATION ===
LONG_IRT_THRESHOLD = 20.0
PERI_SWITCH_WINDOW = 3 
# Range: -3 to 2 (Total 6 positions)
OFFSETS = np.arange(-PERI_SWITCH_WINDOW, PERI_SWITCH_WINDOW) 

# === DATA STRUCTURE ===
# subject_data[subject_id][session_id] = { ...data... }
subject_data = {}

print(f"Starting Multi-Session Subject Analysis...")

for fp in sorted(annotations_dir.glob("*.csv")):
    parts = fp.stem.split("_")
    subject = parts[0] if len(parts) > 0 else "NA"
    session = parts[1] if len(parts) > 1 else "NA"

    # Initialize subject dict
    if subject not in subject_data:
        subject_data[subject] = {}
        
    # Initialize session dict
    if session not in subject_data[subject]:
        subject_data[subject][session] = {
            "cluster_irts": [],
            "switch_irts": [],
            "peri_switch": {k: [] for k in OFFSETS},
            "n_total": 0
        }

    df = pd.read_csv(fp)
    
    # === CONVERT TO NUMPY ARRAYS ===
    transcriptions = df["transcription"].to_numpy()
    starts = pd.to_numeric(df["start"], errors="coerce").to_numpy()
    ends = pd.to_numeric(df["end"], errors="coerce").to_numpy()
    switch_flags = pd.to_numeric(df["switch_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
    
    # Identify Categories
    is_next = np.char.lower(transcriptions.astype(str)) == "next"
    cat_ids = np.cumsum(is_next)
    
    # Compute IRTs
    n_items = len(transcriptions)
    irts = np.full(n_items, np.nan)
    unique_cats = np.unique(cat_ids)
    for cat in unique_cats:
        cat_mask = (cat_ids == cat)
        cat_indices = np.where(cat_mask)[0]
        for i in range(1, len(cat_indices)):
            prev_idx = cat_indices[i - 1]
            curr_idx = cat_indices[i]
            if is_next[prev_idx] or is_next[curr_idx]: continue
            if not np.isnan(starts[curr_idx]) and not np.isnan(ends[prev_idx]):
                irts[curr_idx] = starts[curr_idx] - ends[prev_idx]
    
    # Filter
    word_mask = ~is_next
    transcriptions_filtered = transcriptions[word_mask]
    irts_filtered = irts[word_mask]
    switch_flags_filtered = switch_flags[word_mask]
    cat_ids_filtered = cat_ids[word_mask]
    irts_filtered = np.clip(irts_filtered, 0, None)
    
    # Store Basic Stats
    subject_data[subject][session]["n_total"] = len(transcriptions_filtered)
    
    # === 1. BOXPLOT DATA (Cluster vs Switch) ===
    cluster_mask = switch_flags_filtered == 0
    switch_mask = switch_flags_filtered == 1
    
    c_vals = irts_filtered[cluster_mask]
    s_vals = irts_filtered[switch_mask]
    
    subject_data[subject][session]["cluster_irts"] = c_vals[~np.isnan(c_vals)]
    subject_data[subject][session]["switch_irts"] = s_vals[~np.isnan(s_vals)]

    # === 2. PERI-SWITCH DYNAMICS (Independent Validation) ===
    global_switch_indices = np.where(switch_flags_filtered == 1)[0]
    
    for idx in global_switch_indices:
        # Check LEFT (-3 to -1)
        pre_start = idx - PERI_SWITCH_WINDOW
        left_valid = False
        if pre_start >= 0:
            if np.all(cat_ids_filtered[pre_start : idx+1] == cat_ids_filtered[idx]):
                if not np.any(switch_flags_filtered[pre_start : idx] == 1):
                    left_valid = True
        
        if left_valid:
            for k in range(-PERI_SWITCH_WINDOW, 0):
                val = irts_filtered[idx + k]
                if not np.isnan(val):
                    subject_data[subject][session]["peri_switch"][k].append(val)
                    
        # Check RIGHT (0 to +2)
        post_end = idx + PERI_SWITCH_WINDOW - 1
        right_valid = False
        if post_end < len(irts_filtered):
            if np.all(cat_ids_filtered[idx : post_end+1] == cat_ids_filtered[idx]):
                if not np.any(switch_flags_filtered[idx+1 : post_end+1] == 1):
                    right_valid = True
                    
        if right_valid:
            for k in range(0, PERI_SWITCH_WINDOW):
                val = irts_filtered[idx + k]
                if not np.isnan(val):
                    subject_data[subject][session]["peri_switch"][k].append(val)

print("Data processing complete. Generating plots...")

# === GENERATE PLOTS ===
for subj, sessions_dict in subject_data.items():
    # Sort sessions to ensure chronological order (ses-01, ses-02...)
    sorted_sessions = sorted(sessions_dict.keys())
    
    if not sorted_sessions:
        continue
        
    print(f"Plotting {subj} with {len(sorted_sessions)} sessions...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Subject: {subj} (Sessions: {', '.join(sorted_sessions)})", fontsize=16)
    
    # === PLOT 1: GROUPED BOXPLOT ===
    ax_box = axes[0]
    
    boxplot_data = []
    labels = []
    colors = []
    positions = []
    
    pos_counter = 1
    
    for ses in sorted_sessions:
        data = sessions_dict[ses]
        
        # Log Transform for Boxplot (visualize better)
        c_log = np.log(data["cluster_irts"] + 0.001)
        s_log = np.log(data["switch_irts"] + 0.001)
        
        boxplot_data.append(c_log)
        boxplot_data.append(s_log)
        
        labels.append(f"{ses}\nClust")
        labels.append(f"{ses}\nSwitch")
        
        colors.append("#A0C4FF") # Cluster Blue
        colors.append("#FFADAD") # Switch Red
        
        positions.append(pos_counter)
        positions.append(pos_counter + 1)
        
        pos_counter += 3 # Add gap between sessions
        
    bp = ax_box.boxplot(boxplot_data, positions=positions, patch_artist=True, 
                        widths=0.8, medianprops=dict(color="black", linewidth=1.5))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(labels, rotation=45, ha='right')
    ax_box.set_ylabel("Log IRT (log seconds)")
    ax_box.set_title("Cluster vs Switch IRTs per Session")
    
    # Log scale ticks on Y-axis for readability
    tick_vals = [0.1, 0.5, 1, 2, 5, 10, 20]
    ax_box.set_yticks(np.log(tick_vals))
    ax_box.set_yticklabels([str(v) for v in tick_vals])
    ax_box.grid(True, axis='y', alpha=0.3)
    
    # Custom Legend for Boxplot
    legend_patches = [
        mpatches.Patch(color='#A0C4FF', label='Cluster IRT'),
        mpatches.Patch(color='#FFADAD', label='Switch IRT')
    ]
    ax_box.legend(handles=legend_patches, loc='upper left')

    # === PLOT 2: OVERLAID DYNAMICS ===
    ax_dyn = axes[1]
    
    # Color map for lines
    cmap = plt.get_cmap("viridis")
    colors_lines = cmap(np.linspace(0, 0.9, len(sorted_sessions)))
    
    for i, ses in enumerate(sorted_sessions):
        data = sessions_dict[ses]
        means = []
        sems = []
        valid_ks = []
        
        for k in OFFSETS:
            vals = np.array(data["peri_switch"][k])
            if len(vals) > 0:
                valid_ks.append(k)
                means.append(np.mean(vals))
                if len(vals) > 1:
                    sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                else:
                    sems.append(0)
                    
        if valid_ks:
            ax_dyn.errorbar(valid_ks, means, yerr=sems, label=ses, 
                            color=colors_lines[i], fmt='-o', capsize=4, linewidth=2, alpha=0.8)

    ax_dyn.set_title("Peri-Switch Dynamics (Overlaid)")
    ax_dyn.set_xlabel("Position Relative to Switch (0)")
    ax_dyn.set_ylabel("Mean IRT (s)")
    ax_dyn.set_xticks(OFFSETS)
    ax_dyn.set_xlim(-PERI_SWITCH_WINDOW - 0.5, PERI_SWITCH_WINDOW - 0.5)
    ax_dyn.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax_dyn.grid(True, alpha=0.3)
    ax_dyn.legend()
    
    plt.tight_layout()
    fig_path = outdir / f"{subj}_MultiSession_Summary.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
print("All multi-session plots saved.")