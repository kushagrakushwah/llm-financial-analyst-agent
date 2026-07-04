# LLM-based Financial Analyst Agent

A reinforcement-learning-trained financial auditing agent that simulates contract review, penalty detection, and cost-benefit analysis. Fine-tuned Qwen2.5-7B with GRPO (Group Relative Policy Optimization), achieving 90% improvement in response accuracy and conciseness.

## Features
- RL environment simulating financial auditing tasks
- GRPO training loop for Qwen2.5-7B
- Contract review & penalty detection pipeline
- FastAPI inference endpoint

## Tech Stack
Python · FastAPI · Qwen2.5-7B · GRPO · HuggingFace Transformers

## Setup
```bash
pip install -r requirements.txt
# Training
python training/train_grpo.py
# Inference server
python main.py
```

## API
`POST /api/analyze`  — submit a financial document for analysis  
`GET  /api/health`   — health check
