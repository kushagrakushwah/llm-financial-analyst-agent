"""
GRPO (Group Relative Policy Optimization) fine-tuning for Qwen2.5-7B
on financial auditing tasks.
"""
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from agent.environment import FinancialAuditEnv

MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "outputs/qwen2.5-7b-financial-grpo"


def build_dataset(n_samples: int = 500) -> Dataset:
    env = FinancialAuditEnv()
    prompts = [env.reset() for _ in range(n_samples)]
    return Dataset.from_dict({"prompt": prompts})


def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    env = FinancialAuditEnv()
    rewards = []
    for prompt, completion in zip(prompts, completions):
        env.reset()
        env.current_task = env.reset()  # re-sample for scoring context
        rewards.append(env.compute_reward(completion))
    return rewards


def main():
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    dataset = build_dataset(500)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        group_size=4,           # G in GRPO
        num_generations=4,
        max_new_tokens=256,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
