import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import stats

def load_and_process_file(filepath):
    """Load xlsx file and calculate between-sentence intervals."""
    df = pd.read_excel(filepath)
    
    # Ensure required columns exist
    required_cols = ['Prompt Number', 'Start Time', 'End Time', 'Possibility Number']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Forward-fill the Prompt Number column (since it's only filled for the first sentence of each prompt)
    df['Prompt Number'] = df['Prompt Number'].ffill()
    
    # Remove rows with missing Possibility Number
    df = df[df['Possibility Number'].notna()].copy()
    
    # Sort by Prompt Number and Start Time to ensure correct order
    df = df.sort_values(['Prompt Number', 'Start Time']).reset_index(drop=True)
    
    # Calculate intervals and classify them
    intervals_within = []  # Intervals between sentences within same possibility
    intervals_between = []  # Intervals between sentences of different possibilities
    
    # Process each prompt separately
    for prompt_num in df['Prompt Number'].unique():
        prompt_df = df[df['Prompt Number'] == prompt_num].reset_index(drop=True)
        
        for i in range(1, len(prompt_df)):
            # Calculate interval: Start time of current sentence - End time of previous sentence
            interval = prompt_df.loc[i, 'Start Time'] - prompt_df.loc[i-1, 'End Time']
            
            # Skip negative intervals (might be overlapping speech or errors)
            if interval < 0:
                continue
            
            prev_possibility = prompt_df.loc[i-1, 'Possibility Number']
            curr_possibility = prompt_df.loc[i, 'Possibility Number']
            
            if prev_possibility == curr_possibility:
                intervals_within.append({
                    'interval': interval,
                    'prompt': prompt_num,
                    'from_possibility': prev_possibility,
                    'to_possibility': curr_possibility
                })
            else:
                intervals_between.append({
                    'interval': interval,
                    'prompt': prompt_num,
                    'from_possibility': prev_possibility,
                    'to_possibility': curr_possibility
                })
    
    return intervals_within, intervals_between, df


def plot_single_file(filepath, output_dir=None):
    """Generate descriptive plots for a single file."""
    filename = os.path.basename(filepath)
    file_id = filename.replace('_desc-sentences.xlsx', '')
    
    print(f"\nProcessing: {filename}")
    
    # Load and process data
    intervals_within, intervals_between, df = load_and_process_file(filepath)
    
    within_vals = [d['interval'] for d in intervals_within]
    between_vals = [d['interval'] for d in intervals_between]
    
    print(f"  Within-possibility intervals: n={len(within_vals)}")
    print(f"  Between-possibility intervals: n={len(between_vals)}")
    
    if len(within_vals) == 0 or len(between_vals) == 0:
        print("  ⚠ Not enough data for comparison. Skipping.")
        return None
    
    # Calculate statistics
    within_mean = np.mean(within_vals)
    within_std = np.std(within_vals)
    within_sem = within_std / np.sqrt(len(within_vals))
    
    between_mean = np.mean(between_vals)
    between_std = np.std(between_vals)
    between_sem = between_std / np.sqrt(len(between_vals))
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(within_vals, between_vals)
    
    print(f"  Within-possibility: M={within_mean:.3f}s, SD={within_std:.3f}s")
    print(f"  Between-possibility: M={between_mean:.3f}s, SD={between_std:.3f}s")
    print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f'Between-Sentence Interval Analysis\n{file_id}', fontsize=12, fontweight='bold')
    
    # Plot 1: Bar plot with error bars (SEM)
    ax1 = axes[0]
    conditions = ['Within\nPossibility', 'Between\nPossibilities']
    means = [within_mean, between_mean]
    sems = [within_sem, between_sem]
    colors = ['#4C72B0', '#DD8452']
    
    bars = ax1.bar(conditions, means, yerr=sems, capsize=5, color=colors, 
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    ax1.set_ylabel('Mean Interval (seconds)', fontsize=11)
    ax1.set_title('Mean Intervals (±SEM)', fontsize=11)
    ax1.set_ylim(0, max(means) * 1.4)
    
    # Add significance annotation
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    # Add significance bar
    y_max = max(means) + max(sems)
    ax1.plot([0, 0, 1, 1], [y_max*1.05, y_max*1.1, y_max*1.1, y_max*1.05], 'k-', linewidth=1)
    ax1.text(0.5, y_max*1.12, sig_text, ha='center', va='bottom', fontsize=12)
    
    # Add n values on bars
    for i, (bar, n) in enumerate(zip(bars, [len(within_vals), len(between_vals)])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'n={n}', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Plot 2: Box plot
    ax2 = axes[1]
    box_data = [within_vals, between_vals]
    bp = ax2.boxplot(box_data, labels=conditions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Interval (seconds)', fontsize=11)
    ax2.set_title('Distribution of Intervals', fontsize=11)
    
    # Plot 3: Histogram overlay
    ax3 = axes[2]
    max_val = max(max(within_vals), max(between_vals))
    bins = np.linspace(0, min(max_val, 20), 25)  # Cap at 20s for visualization
    
    ax3.hist(within_vals, bins=bins, alpha=0.6, label=f'Within (n={len(within_vals)})', 
             color=colors[0], edgecolor='black', linewidth=0.5)
    ax3.hist(between_vals, bins=bins, alpha=0.6, label=f'Between (n={len(between_vals)})', 
             color=colors[1], edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Interval (seconds)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Interval Distributions', fontsize=11)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{file_id}_interval_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    plt.show()
    
    # Return summary statistics
    return {
        'file_id': file_id,
        'within_n': len(within_vals),
        'within_mean': within_mean,
        'within_std': within_std,
        'within_sem': within_sem,
        'between_n': len(between_vals),
        'between_mean': between_mean,
        'between_std': between_std,
        'between_sem': between_sem,
        't_statistic': t_stat,
        'p_value': p_value
    }


def analyze_all_files(directory=".", output_dir="output_plots"):
    """Analyze all matching files in a directory."""
    # Find all matching files
    pattern = os.path.join(directory, "*_desc-sentences.xlsx")
    files = glob.glob(pattern)
    
    if not files:
        print("No matching files found!")
        return None
    
    print(f"Found {len(files)} file(s) to analyze")
    print("=" * 60)
    
    # Process each file
    all_results = []
    for filepath in sorted(files):
        result = plot_single_file(filepath, output_dir)
        if result:
            all_results.append(result)
    
    # Create summary dataframe
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, 'summary_statistics.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")
        
        # Print summary table
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(summary_df[['file_id', 'within_mean', 'between_mean', 'p_value']].to_string(index=False))
        
        return summary_df
    
    return None


def create_combined_plot(directory=".", output_dir="output_plots"):
    """Create a combined plot comparing all files."""
    pattern = os.path.join(directory, "*_desc-sentences.xlsx")
    files = glob.glob(pattern)
    
    if not files:
        return
    
    all_within = []
    all_between = []
    file_labels = []
    
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        file_id = filename.replace('_desc-sentences.xlsx', '')
        file_labels.append(file_id)
        
        intervals_within, intervals_between, _ = load_and_process_file(filepath)
        all_within.append([d['interval'] for d in intervals_within])
        all_between.append([d['interval'] for d in intervals_between])
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(file_labels))
    width = 0.35
    
    within_means = [np.mean(w) for w in all_within]
    between_means = [np.mean(b) for b in all_between]
    within_sems = [np.std(w)/np.sqrt(len(w)) for w in all_within]
    between_sems = [np.std(b)/np.sqrt(len(b)) for b in all_between]
    
    bars1 = ax.bar(x - width/2, within_means, width, yerr=within_sems, 
                   label='Within Possibility', color='#4C72B0', capsize=4, alpha=0.8)
    bars2 = ax.bar(x + width/2, between_means, width, yerr=between_sems,
                   label='Between Possibilities', color='#DD8452', capsize=4, alpha=0.8)
    
    ax.set_ylabel('Mean Interval (seconds)', fontsize=12)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_title('Comparison of Between-Sentence Intervals Across Sessions', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace('sub-', 'S').replace('_ses-', '\nSes-').replace('_task-ahc', '') 
                        for label in file_labels], fontsize=9)
    ax.legend()
    ax.set_ylim(0, max(max(within_means), max(between_means)) * 1.4)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'combined_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Combined plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Set your directory path here (default is current directory)
    data_directory = "/Users/gioli/projects/stateswitch/data/rec/ahc_sentences"
    output_directory = "/Users/gioli/projects/stateswitch/figs/ahc_rt"
    
    # Analyze all files and generate individual plots
    summary = analyze_all_files(data_directory, output_directory)
    
    # Generate combined comparison plot
    create_combined_plot(data_directory, output_directory)