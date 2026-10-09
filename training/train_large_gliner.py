"""
train_large_gliner.py
=====================
Fine-tunes GLiNER on optimized, high-density internet contract datasets:
- Agile Lab ContractNER (Hugging Face)
- Stanford Atticus CUAD (SEC Edgar)
- In-domain financial auditing clauses
- Boilerplate negative distractors

Bounded to 15-130 tokens to ensure rapid training on CPU (200 steps, ~8 min).
Saves checkpoints every 50 steps.
"""

import json
import os
import sys
import torch
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import SpanDataCollator

# Configure PyTorch CPU threads
torch.set_num_threads(14)


def load_dataset(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "optimized_internet_train.json")
    eval_path = os.path.join(base_dir, "data", "optimized_internet_eval.json")
    output_model_dir = os.path.join(base_dir, "models", "gliner_financial_v2")
    os.makedirs(output_model_dir, exist_ok=True)

    print("=" * 80)
    print(" Training GLiNER on High-Density Internet Contract Datasets")
    print("=" * 80)

    train_data = load_dataset(train_path)
    eval_data = load_dataset(eval_path)
    print(f"Loaded {len(train_data)} training samples from: {train_path}")
    print(f"Loaded {len(eval_data)} evaluation samples from: {eval_path}")

    # Warm-start from existing checkpoint
    warm_checkpoint = os.path.join(base_dir, "models", "gliner_financial")
    if os.path.exists(warm_checkpoint):
        print(f"\n[1] Warm-starting model from fine-tuned checkpoint: '{warm_checkpoint}'...")
        model = GLiNER.from_pretrained(warm_checkpoint)
    else:
        print("\n[1] Initializing base encoder: 'urchade/gliner_small-v2.1'...")
        model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

    # Fast CPU-optimized training arguments (200 total steps)
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        learning_rate=2e-5,               # Backbone LR
        weight_decay=0.01,
        others_lr=2e-4,                  # Head LR
        others_weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=6,    # 100 steps per epoch
        per_device_eval_batch_size=6,
        num_train_epochs=2,               # 2 passes over 600 samples = 200 steps
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=50,                   # Checkpoint every 50 steps
        save_total_limit=2,
        logging_steps=10,
        negatives=0.30,                   # Negative sampling to penalize hallucinations
        report_to="none"
    )

    print("\n[2] Setting up SpanDataCollator and Trainer...")
    data_collator = SpanDataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    print("\n[3] Launching fine-tuning loop (200 steps total)...")
    trainer.train()

    print(f"\n[4] Saving upgraded checkpoint to: '{output_model_dir}'...")
    model.save_pretrained(output_model_dir)
    print("Internet dataset fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
