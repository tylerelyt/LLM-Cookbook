# Lesson 2: Self-Instruct Data Construction

This lesson demonstrates how to automatically generate instruction-tuning data using the Self-Instruct method.

## What you will learn

- Design a small set of high-quality seed tasks and expand them into diverse instructions
- Automatically create high-quality answers for generated instructions
- Produce Alpaca/Instruction-tuning style Q&A samples
- Prepare a large, high-quality dataset for downstream fine-tuning

## Core idea

Self-Instruct bootstraps a model in iterations:
1. Start from a small seed set
2. Expand to new instructions
3. Generate input–output instances
4. Filter and deduplicate

## Features

- Local seed tasks expansion and instance generation
- Output in JSONL with fields: `instruction`, `input`, `output`
- Supports OpenAI and DashScope backends; auto-switch when only one key is provided
- Built-in deduplication and basic quality filters

## Setup

```bash
pip install -r requirements.txt

# Set at least one API key
export OPENAI_API_KEY="your-openai-key"      # for OpenAI
export DASHSCOPE_API_KEY="your-dashscope-key"# for DashScope
```

## Quickstart

```bash
cd chapter7/lesson2

# One-command workshop (single file, no CLI args)
python self_instruct_workshop.py
```

### Notes
- `self_instruct_workshop.py` integrates the four-step pipeline end-to-end.
- Seed tasks are read from `chapter7/lesson2/data/seed_tasks.jsonl`. You can replace it with the official 175 seed tasks for full coverage.
- Outputs are written to `chapter7/lesson2/data/workshop/` (step-wise JSONL files).

## Output format (JSONL)

```json
{"instruction": "Explain the properties of a binary search tree and give an example", "input": "", "output": "..."}
{"instruction": "Write an SQL query based on the requirement", "input": "table users(id, name, age)", "output": "SELECT ..."}
```

## Tips

- Favor quality over quantity for seed tasks
- Deduplicate and review low-quality samples as needed
- Control cost by tuning `NUM_INSTRUCTIONS` and `MAX_TOKENS`, and monitor API usage

## References

- Paper: Self-Instruct: Aligning Language Models with Self Generated Instructions (`https://arxiv.org/abs/2212.10560`)
- Original implementation: `https://github.com/yizhongw/self-instruct`