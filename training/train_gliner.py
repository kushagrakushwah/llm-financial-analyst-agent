"""
Fine-tuning GLiNER on Financial & Contract Auditing Entities.
Specializes the model for exact span extraction of penalty rates, liability caps,
breach conditions, effective dates, and monetary figures.
"""

import json
import os
import sys
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import SpanDataCollator


def load_json_dataset(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "financial_ner_train.json")
    eval_path = os.path.join(base_dir, "data", "financial_ner_eval.json")
    output_model_dir = os.path.join(base_dir, "models", "gliner_financial")
    os.makedirs(output_model_dir, exist_ok=True)

    print("=" * 70)
    print(" Fine-Tuning GLiNER for Financial Contract Entity Extraction")
    print("=" * 70)

    train_data = load_json_dataset(train_path)
    eval_data = load_json_dataset(eval_path)
    print(f"Loaded {len(train_data)} training samples from {train_path}")
    print(f"Loaded {len(eval_data)} evaluation samples from {eval_path}")

    # 1. Base model to specialize
    base_model_id = os.getenv("BASE_GLINER_MODEL", "urchade/gliner_small-v2.1")
    print(f"\n[1] Initializing base encoder: '{base_model_id}'...")
    model = GLiNER.from_pretrained(base_model_id)

    # 2. Training configuration with differential learning rates & negative sampling
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        learning_rate=1e-5,               # Backbone (DeBERTa) LR to avoid catastrophic forgetting
        weight_decay=0.01,
        others_lr=1e-4,                   # Classification/Projection head LR for rapid adaptation
        others_weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1,
        negatives=1.0,                    # Samples negative entity types to maintain zero-shot precision
        report_to="none"
    )

    # 3. Setup Collator and Trainer
    print("\n[2] Configuring SpanDataCollator and Trainer...")
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

    # 4. Train
    print("\n[3] Launching fine-tuning loop...")
    trainer.train()

    # 5. Save Model
    print(f"\n[4] Saving fine-tuned financial checkpoint to '{output_model_dir}'...")
    model.save_pretrained(output_model_dir)
    print("Fine-tuning completed successfully!")

    # 6. Test on Unseen Financial Clause
    print("\n" + "=" * 70)
    print(" [5] Testing Fine-Tuned Checkpoint on Unseen Contract Clause")
    print("=" * 70)

    test_clause = (
        "In the event of unexcused milestone delay, Contractor shall incur a liquidated damages "
        "penalty of 2.5% per week of delay, provided aggregate liability is capped at $750,000 USD."
    )
    test_labels = [
        "contracting_party",
        "penalty_rate",
        "penalty_condition",
        "liability_cap",
        "monetary_amount"
    ]

    finetuned_model = GLiNER.from_pretrained(output_model_dir)
    entities = finetuned_model.predict_entities(test_clause, test_labels, threshold=0.35)

    print(f"\nTest Contract Clause:\n  \"{test_clause}\"\n")
    print("Extracted Spans:")
    for ent in entities:
        print(f"  [{ent['label'].upper():<18}] -> \"{ent['text']}\" (score: {ent['score']:.2f}, span: [{ent['start']}:{ent['end']}])")


if __name__ == "__main__":
    main()
