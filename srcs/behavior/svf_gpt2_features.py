import sys
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import argparse
import warnings
from pathlib import Path

# Suppress visual and library warnings
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
GPT_FIGS_DIR = FIGS_DIR / "gpt_dual_context"
GPT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === 2. STYLING ===
METRICS = ['surprisal', 'bayesian_surprise', 'entropy', 'cosine_dist']
COLORS = {'surprisal': '#1f77b4', 'bayesian_surprise': '#2ca02c', 
          'entropy': '#000000', 'cosine_dist': '#e377c2'}

# ==========================================
# 3. GPT-2 MULTI-TOKEN ANALYZER
# ==========================================
class GPT2MultiTokenAnalyzer:
    def __init__(self, model_name='gpt2-medium'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device).eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_word_stats(self, context_text, target_word, prev_hidden):
        """Calculates aggregated surprisal and final-token embeddings."""
        # Add leading space to match word-start tokenization
        word_tokens = self.tokenizer.encode(f" {target_word}", add_special_tokens=False)
        current_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(self.device)
        
        total_surp = 0
        first_ent = 0
        first_bs = 0
        last_h = None
        
        for i, token_id in enumerate(word_tokens):
            with torch.no_grad():
                out = self.model(current_ids, output_hidden_states=True)
            
            logits = out.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # 1. Summed Surprisal
            p_token = probs[token_id].clamp(min=1e-10).item()
            total_surp += -np.log(p_token)
            
            # 2. Entropy and Bayesian Surprise (at word onset)
            if i == 0:
                first_ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                # Placeholder for BS (requires posterior after first token)
                temp_ids = torch.cat([current_ids, torch.tensor([[token_id]]).to(self.device)], dim=-1)
                with torch.no_grad():
                    out_post = self.model(temp_ids)
                p_post = F.softmax(out_post.logits[0, -1, :], dim=-1)
                first_bs = F.kl_div(probs.log(), p_post, reduction='sum').item()
            
            # Update for next token
            current_ids = torch.cat([current_ids, torch.tensor([[token_id]]).to(self.device)], dim=-1)
            last_h = out.hidden_states[-1][0, -1, :]

        dist = cosine(last_h.cpu().numpy(), prev_hidden.cpu().numpy()) if prev_hidden is not None else np.nan
        return total_surp, first_bs, first_ent, last_h, dist

    def analyze_trial(self, word_list):
        results = []
        prev_h_g, prev_h_z = None, None

        for i, word in enumerate(word_list):
            # Global Context
            history = word_list[:i]
            ctx_g = self.tokenizer.eos_token if not history else ", ".join(history) + ","
            s_g, bs_g, e_g, h_g, d_g = self._get_word_stats(ctx_g, word, prev_h_g)

            # Zero-Context (for Distance Baseline)
            _, _, _, h_z, d_z = self._get_word_stats(self.tokenizer.eos_token, word, prev_h_z)
            
            # Bigram Context (for Predictive Baseline)
            if i > 0:
                s_b, bs_b, e_b, _, _ = self._get_word_stats(word_list[i-1] + ",", word, None)
            else:
                s_b, bs_b, e_b = np.nan, np.nan, np.nan

            results.append({
                'surprisal_global': s_g, 'surprisal_bigram': s_b,
                'bayesian_surprise_global': bs_g, 'bayesian_surprise_bigram': bs_b,
                'entropy_global': e_g, 'entropy_bigram': e_b,
                'cosine_dist_global': d_g, 'cosine_dist_bigram': d_z
            })
            prev_h_g, prev_h_z = h_g, h_z
        return results

# ==========================================
# 4. PLOTTING
# ==========================================
def process_session(subject, session, analyzer):
    csv_path = list(ANNOTATIONS_DIR.glob(f"{subject}_{session}*wordtimestamps*.csv"))
    if not csv_path: return None
    
    df = pd.read_csv(csv_path[0]).sort_values("start").reset_index(drop=True)
    df["trial_id"] = (df["transcription"].astype(str).str.lower() == "next").cumsum()
    df_words = df[df["transcription"].astype(str).str.lower() != "next"].copy()
    df_words["switch_flag"] = pd.to_numeric(df_words["switch_flag"], errors='coerce').fillna(0)
    
    all_trial_dfs = []
    grouped = list(df_words.groupby("trial_id"))
    fig, axes = plt.subplots(len(grouped), 2, figsize=(26, 8 * len(grouped)), 
                             gridspec_kw={'width_ratios': [3, 1]}, squeeze=False)

    for idx, (tid, tdf) in enumerate(grouped):
        metrics = analyzer.analyze_trial(tdf["transcription"].tolist())
        for key in metrics[0].keys(): tdf[key] = [m[key] for m in metrics]
        
        tdf["label"] = tdf["switch_flag"].map({1: "Switch", 0: "Cluster"})
        tdf["subject"], tdf["session"] = subject, session
        all_trial_dfs.append(tdf)
        
        ax_l, ax_r = axes[idx, 0], axes[idx, 1]
        x = np.arange(len(tdf))

        for m in METRICS:
            ax_l.plot(x, tdf[f'{m}_global'], color=COLORS[m], label=f"{m} (Global)", lw=2.5)
            ax_l.plot(x, tdf[f'{m}_bigram'], color=COLORS[m], ls='--', alpha=0.4)

        for i, sw in enumerate(tdf["switch_flag"]):
            if sw == 1: ax_l.axvspan(i-0.4, i+0.4, color='red', alpha=0.1)
        
        ax_l.set_xticks(x)
        ax_l.set_xticklabels(tdf["transcription"], rotation=45, ha='right')
        ax_l.set_title(f"{subject} {session} | Trial {tid}")
        if idx == 0: ax_l.legend(loc='upper left', ncol=2, fontsize=8)

        # Bars (Exclude first word)
        val_vars = [f"{m}_global" for m in METRICS] + [f"{m}_bigram" for m in METRICS]
        melted = tdf.iloc[1:].melt(id_vars=['label'], value_vars=val_vars)
        melted['variable'] = melted['variable'].str.replace('_global', ' (G)').str.replace('_bigram', ' (B/Z)')
        
        sns.barplot(data=melted, x='variable', y='value', hue='label', ax=ax_r, 
                    palette={"Switch": "red", "Cluster": "black"}, capsize=.1)
        ax_r.set_xticks(range(len(val_vars)))
        ax_r.set_xticklabels(ax_r.get_xticklabels(), rotation=90)

    plt.tight_layout()
    plt.savefig(GPT_FIGS_DIR / f"{subject}_{session}_multitoken_analysis.png")
    plt.close()
    return pd.concat(all_trial_dfs)

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    analyzer = GPT2MultiTokenAnalyzer()
    
    if args.all:
        all_data = []
        # Crawl for all subject folders
        for sub_dir in sorted(list(DERIVATIVES_DIR.glob("sub-*"))):
            for ses_dir in sorted(list(sub_dir.glob("ses-*"))):
                print(f"Processing: {sub_dir.name} {ses_dir.name}")
                res = process_session(sub_dir.name, ses_dir.name, analyzer)
                if res is not None: all_data.append(res)
        
        if all_data:
            grand_df = pd.concat(all_data)
            grand_clean = grand_df.groupby(["subject", "session", "trial_id"], group_keys=False).apply(lambda x: x.iloc[1:])
            grand_clean["label"] = grand_clean["switch_flag"].map({1: "Switch", 0: "Cluster"})
            
            fig, ax = plt.subplots(figsize=(16, 7))
            val_vars = [f"{m}_global" for m in METRICS] + [f"{m}_bigram" for m in METRICS]
            melted_g = grand_clean.melt(id_vars=['label'], value_vars=val_vars)
            melted_g['variable'] = melted_g['variable'].str.replace('_global', ' (G)').str.replace('_bigram', ' (B/Z)')
            sns.barplot(data=melted_g, x='variable', y='value', hue='label', ax=ax, palette={"Switch": "red", "Cluster": "black"})
            plt.xticks(rotation=45)
            plt.title("GLOBAL MULTI-TOKEN COMPARISON")
            plt.savefig(GPT_FIGS_DIR / "GLOBAL_multitoken_summary.png")
            print("Finished. Global summary saved.")
    elif args.sub:
        # Run specific subject sessions
        for ses_p in sorted(ANNOTATIONS_DIR.glob(f"{args.sub}_ses-*wordtimestamps*.csv")):
            ses_id = ses_p.name.split("_")[1]
            process_session(args.sub, ses_id, analyzer)