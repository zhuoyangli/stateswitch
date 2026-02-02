import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Analyzer:
    def __init__(self, model_name='gpt2', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_probs(self, text):
        """Helper to get the next-token probability distribution for a given string."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get logits for the last token and convert to probabilities
        probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
        # Also return hidden states for semantic distance if needed
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return probs, last_hidden_state

    def compute_metrics(self, word_list, window_size=None):
        """
        Processes a list of words where context is built incrementally.
        If the word_list starts fresh, the first word has no context.
        """
        results = []

        for i, word in enumerate(word_list):
            # 1. CONSTRUCT CONTEXT
            if window_size is not None and i > window_size:
                visible_history = word_list[i-window_size : i]
            else:
                visible_history = word_list[:i]
            
            # Context is just the previous words joined by commas
            context_before = ", ".join(visible_history)
            
            # Handle the 'Cold Start' for the very first word in a category
            if not context_before:
                # Use EOS token as a dummy start so the model has *something* to look at
                context_before = self.tokenizer.eos_token
            else:
                context_before += ","

            # 2. GET DISTRIBUTIONS
            probs_before, _ = self._get_probs(context_before)
            
            context_after = f"{context_before} {word}"
            probs_after, hidden_after = self._get_probs(context_after)

            # 3. CALCULATE METRICS
            word_token_id = self.tokenizer.encode(f" {word}")[0]
            word_prob = probs_before[0, word_token_id].clamp(min=1e-10).item()
            surprisal = -torch.log(torch.tensor(word_prob)).item()

            log_probs_before = probs_before.log()
            bayesian_surprise = F.kl_div(log_probs_before, probs_after, reduction='sum').item()

            results.append({
                'word': word,
                'surprisal': surprisal,
                'bayesian_surprise': bayesian_surprise,
                'hidden_state': hidden_after.cpu().numpy()
            })

        return results