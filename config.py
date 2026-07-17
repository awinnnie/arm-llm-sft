config = {
    # data
    "train_parquets": [
        "../../data/it_data/data/train-00000-of-00003.parquet",
        "../../data/it_data/data/train-00001-of-00003.parquet",
        "../../data/it_data/data/train-00002-of-00003.parquet",
    ],
    "test_data": "../../data/it_data/data/test-00000-of-00001.parquet",
    "tatoeba": "../../data/tatoeba/data/train-00000-of-00001.parquet",
    "wiki": "../../data/wikipedia_eval/data/train-00000-of-00001.parquet",
    "languages": ["hy", "en", "en-hy"],
    "language_weights": {"hy": 1600, "en": 200, "en-hy": 200},
    "min_task_count": 10000,
    "train_samples": 2000,
    "test_samples": 200,
    "random_state": 42,

    # model
    "model_name": "Qwen/Qwen3-4B-Instruct-2507",

    # lora
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],

    # training
    "epochs": 2,
    "batch_size": 1,
    "grad_accumulation": 16,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.1,
    "max_length": 1024,
    "logging_steps": 20,
    "eval_steps": 20,
    "save_steps": 50,
    "save_total_limit": 3,

    # eval
    "eval_every": 20,
    "tatoeba_samples": 50,
    "wiki_samples": 50,

    # wandb
    "project": "qwen3-finetune",
    "run_name": "exp-006",
    "run_id": "exp-006-run",
}