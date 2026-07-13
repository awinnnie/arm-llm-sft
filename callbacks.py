import math
import torch
import wandb
from transformers import TrainerCallback
import time

class CustomEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, generation_examples, tatoeba_pairs, wiki_hy_texts, wiki_en_texts, eval_every=20):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_examples = generation_examples
        self.tatoeba_pairs = tatoeba_pairs
        self.wiki_hy_texts = wiki_hy_texts
        self.wiki_en_texts = wiki_en_texts
        self.eval_every = eval_every
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0:
            elapsed = time.time() - self.start_time
            wandb.log({
                "train/seconds_per_step": elapsed / state.global_step
            }, step=state.global_step)
        
        if state.global_step % self.eval_every == 0:
            self.model.eval()
            with torch.no_grad():
                self._log_generations(state.global_step)
                self._log_tatoeba(state.global_step)
                self._log_wiki(state.global_step)
            self.model.train()

    def _generate(self, messages):
        # take all turns except last assistant turn as input
        prompt_messages = list(messages[:-1])
        text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated

    def _log_generations(self, step):
        rows = []
        for ex in self.generation_examples:
            messages = [dict(d) for d in ex['messages']]
            reference = messages[-1]['content']  # last assistant turn
            generated = self._generate(messages)
            rows.append([
                ex['task'],
                messages[-2]['content'],  # user input
                reference,
                generated,
                step
            ])

        table = wandb.Table(
            columns=["task", "input", "reference", "generated", "step"],
            data=rows
        )
        wandb.log({"custom_evaluation/generations": table}, step=step)

    def _get_embedding(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden = outputs.hidden_states[-1] # [1, seq_len, hidden_size]
        embedding = hidden.mean(dim=1) # [1, hidden_size]
        return embedding 
    
    def _log_tatoeba(self, step):
        similarities = []
        for pair in self.tatoeba_pairs:
            # compute cosine similarity
            emb_en =self._get_embedding(pair['en'])
            emb_hy = self._get_embedding(pair['hy'])
            sim = torch.nn.functional.cosine_similarity(emb_en, emb_hy).item()
            similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities)
        wandb.log({"custom_evaluation/tatoeba_similarity": avg_sim}, step=step)

    def _log_wiki(self, step):
        hy_losses = []

        for text in self.wiki_hy_texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.model.device)

            outputs = self.model(**inputs, labels=inputs['input_ids'])
            hy_losses.append(outputs.loss.item())

        avg_hy_loss = sum(hy_losses) / len(hy_losses)
        avg_hy_ppl = math.exp(avg_hy_loss)

        en_losses = []
        for text in self.wiki_en_texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.model.device)
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            en_losses.append(outputs.loss.item())

        avg_en_loss = sum(en_losses) / len(en_losses)
        avg_en_ppl = math.exp(avg_en_loss)
        wandb.log({
            "custom_evaluation/wiki_hy_loss": avg_hy_loss,
            "custom_evaluation/wiki_hy_perplexity": avg_hy_ppl,
            "custom_evaluation/wiki_en_loss": avg_en_loss,
            "custom_evaluation/wiki_en_perplexity": avg_en_ppl,
        }, step=step)
