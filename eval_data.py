import pandas as pd
from config import config

def load_generation_examples(df):
    preferred_tasks = [
        "Translation",
        "Paraphrase",
        "Grammar fix",
        "Summarize",
        "Simple qa",
    ]
    examples = []
    for task in preferred_tasks:
        subset = df[df['task_type'] == task]
        if len(subset) > 0:
            row = subset.sample(1, random_state=config["random_state"]).iloc[0]
            examples.append({
                "task": task,
                "messages": row['messages']
            })
    return examples

def load_tatoeba(path):
    df = pd.read_parquet(path).sample(config["tatoeba_samples"], random_state=config["random_state"])
    return [
        {"en": row['translation']['en'], "hy": row['translation']['hy']}
        for _, row in df.iterrows()
    ]

def load_wiki(path):
    df = pd.read_parquet(path)
    hy = df[df['lang'] == 'hy'].sample(config["wiki_samples"], random_state=config["random_state"])['text'].tolist()
    en = df[df['lang'] == 'en'].sample(config["wiki_samples"], random_state=config["random_state"])['text'].tolist()
    return hy, en

