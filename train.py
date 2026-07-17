import wandb
import torch 
# from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import transformers.utils.import_utils as _utils
_utils.check_torch_load_is_safe = lambda: None
from config import config

from eval_data import load_generation_examples, load_tatoeba, load_wiki
from callbacks import CustomEvalCallback

wandb.init(project=config["project"], name=config["run_name"], resume="allow", id=config["run_id"],
	config=config
)

df1 = pd.read_parquet(config["train_parquets"][0])
df2 = pd.read_parquet(config["train_parquets"][1])
df3 = pd.read_parquet(config["train_parquets"][2])

df = pd.concat([df1, df2, df3]).reset_index(drop=True)
print("after concat:", df.columns.tolist())

task_counts = df['task_type'].value_counts()
eligible_tasks = task_counts[task_counts > config["min_task_count"]].index

hy = df[(df['language'] == 'hy') & (df['task_type'].isin(eligible_tasks))]
en = df[(df['language'] == 'en') & (df['task_type'].isin(eligible_tasks))]
en_hy = df[(df['language'] == 'en-hy') & (df['task_type'].isin(eligible_tasks))]

df = pd.concat([
    hy.sample(min(len(hy), config["language_weights"]["hy"]), random_state=config["random_state"]),
    en.sample(min(len(en), config["language_weights"]["en"]), random_state=config["random_state"]),
    en_hy.sample(min(len(en_hy), config["language_weights"]["en-hy"]), random_state=config["random_state"]),
]).sample(frac=1, random_state=config["random_state"]).reset_index(drop=True)

print("after language filter:", df.columns.tolist())

df_eligible = df[df['task_type'].isin(eligible_tasks)]

samples = []
for task in eligible_tasks:
    task_df = df_eligible[df_eligible['task_type'] == task]
    n = min(len(task_df), max(1, round(2000 * len(task_df) / len(df_eligible))))
    samples.append(task_df.sample(n=n, random_state=config["random_state"]))

df = pd.concat(samples).sample(frac=1, random_state=config["random_state"]).reset_index(drop=True).iloc[:config["train_samples"]]


# train 
train_df = df 

# test 
test_raw = pd.read_parquet(config["test_data"])

task_counts_test = test_raw['task_type'].value_counts()
eligible_tasks_test = task_counts_test[task_counts_test > 0].index  # keep all tasks that exist

test_raw = test_raw[test_raw['task_type'].isin(eligible_tasks)]  # same eligible tasks as train
test_raw = test_raw[test_raw['language'].isin(config["languages"])]  # same languages

# proportional sample
test_samples = []
for task in eligible_tasks:
    task_df = test_raw[test_raw['task_type'] == task]
    if len(task_df) == 0:
        continue
    n = min(len(task_df), max(1, round(200 * len(task_df) / len(test_raw[test_raw['task_type'].isin(eligible_tasks)]))))
    test_samples.append(task_df.sample(n=n, random_state=config["random_state"]))

val_df = pd.concat(test_samples).sample(frac=1, random_state=config["random_state"]).reset_index(drop=True).iloc[:config["test_samples"]]

model_name = config["model_name"]

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_mask(example):
    messages = example["messages"]
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
    messages = [dict(d) for d in messages]
    
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    tokenized = tokenizer(
        full_text,
        max_length=1024,
        truncation=True,
        padding=False,
        # ensuring we don't accidentally add BOS/EOS tokens that shift the index
        add_special_tokens=False 
    )
    
    input_ids = tokenized['input_ids']
    labels = [-100] * len(input_ids)  # mask everything by default
    
    # exact ID sequences for Qwen
    assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    
    i = 0
    while i < len(input_ids):
        # check if current position matches the sequence of assistant_header IDs
        if input_ids[i:i+len(assistant_header)] == assistant_header:
            i += len(assistant_header)  # skip past header
            
            # unmask tokens until end token
            while i < len(input_ids):
                if input_ids[i:i+len(end_token)] == end_token:
                    # unmask end token asw
                    for j in range(len(end_token)):
                        if i+j < len(input_ids):
                            labels[i+j] = input_ids[i+j]
                    i += len(end_token)
                    break
                
                labels[i] = input_ids[i]
                i += 1
        else:
            i += 1
    
    tokenized['labels'] = labels
    return tokenized

train_dataset = Dataset.from_pandas(train_df[['messages']]).map(tokenize_and_mask, remove_columns=['messages'])
val_dataset = Dataset.from_pandas(val_df[['messages']]).map(tokenize_and_mask, remove_columns=['messages'])


generation_examples = load_generation_examples(df)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda:0",
)

lora_config = LoraConfig(
    r=config["r"],
    lora_alpha=config["lora_alpha"],
    target_modules=config["target_modules"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
   )

model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="./checkpoints",
    num_train_epochs=config["epochs"],  #d
    per_device_train_batch_size=config["batch_size"], #g
    gradient_accumulation_steps=config["grad_accumulation"], #g
    learning_rate=config["learning_rate"], #g
    lr_scheduler_type=config["lr_scheduler"],
    warmup_ratio=config["warmup_ratio"], #g
    fp16=False, #g
    bf16=False,
    logging_steps=config["logging_steps"], #d
    save_steps=config["save_steps"], #d
    save_total_limit=config["save_total_limit"], #d
    load_best_model_at_end=False,
    eval_strategy="steps",
    eval_steps=config["eval_steps"],
    report_to="wandb",
    max_length=config["max_length"],
)

# loss only on assisstant turns


tatoeba_pairs = load_tatoeba(config["tatoeba"])
wiki_hy, wiki_en = load_wiki(config["wiki"])

eval_callback = CustomEvalCallback(
    model=model,
    tokenizer=tokenizer,
    generation_examples=generation_examples,
    tatoeba_pairs=tatoeba_pairs,
    wiki_hy_texts=wiki_hy,
    wiki_en_texts=wiki_en,
    eval_every=config["eval_every"],
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    callbacks=[eval_callback],
)

# log base model generations at step 0
eval_callback.model.eval()
with torch.no_grad():
    eval_callback._log_generations(0)
eval_callback.model.train()

trainer.train()
wandb.finish()
