import math
import torch
import wandb
from sentence_transformers import util
from transformers import TrainerCallback

class CustomEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, generation_examples, tatoeba_pairs, wiki_hy_texts, wiki_en_texts, sim_model, eval_every=20):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_examples = generation_examples
        self.tatoeba_pairs = tatoeba_pairs
        self.wiki_hy_texts = wiki_hy_texts
        self.wiki_en_texts = wiki_en_texts
        self.sim_model = sim_model
        self.eval_every = eval_every

    def on_step_end(self, args, state, control, **kwargs):
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
                max_new_tokens=200,
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
                messages[-2]['content'][:200],  # user input (truncated for display)
                reference[:200],
                generated[:200],
                step
            ])

        table = wandb.Table(
            columns=["task", "input", "reference", "generated", "step"],
            data=rows
        )
        wandb.log({"custom_evaluation/generations": table}, step=step)

    def _log_tatoeba(self, step):
        similarities = []
        for pair in self.tatoeba_pairs:
            # prompt model to translate en -> hy
            messages = [
                {"role": "system", "content": "You are a translator. Translate the given English text to Armenian."},
                {"role": "user", "content": pair['en']}
            ]
            generated_hy = self._generate(messages + [{"role": "assistant", "content": ""}])

            # compute cosine similarity between generated and reference
            emb_gen = self.sim_model.encode(generated_hy, convert_to_tensor=True)
            emb_ref = self.sim_model.encode(pair['hy'], convert_to_tensor=True)
            sim = util.cos_sim(emb_gen, emb_ref).item()
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
