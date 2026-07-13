import pandas as pd

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
            row = subset.sample(1, random_state=42).iloc[0]
            examples.append({
                "task": task,
                "messages": row['messages']
            })
    return examples

def load_tatoeba(path):
    df = pd.read_parquet(path).sample(10, random_state=42)
    return [
        {"en": row['translation']['en'], "hy": row['translation']['hy']}
        for _, row in df.iterrows()
    ]

def load_wiki(path):
    df = pd.read_parquet(path)
    hy = df[df['lang'] == 'hy'].sample(10, random_state=42)['text'].tolist()
    en = df[df['lang'] == 'en'].sample(10, random_state=42)['text'].tolist()
    return hy, en

