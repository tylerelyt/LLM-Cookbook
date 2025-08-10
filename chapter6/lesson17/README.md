# Lesson 17: Omission-Aware Reflection System (Single-file Workshop, Real AutoGen)

This directory provides a concise single-file script `workshop.py` that demonstrates an AutoGen-based multi-agent workflow for content generation with omission-aware reflection. A DashScope-compatible API is required.

## Quick start

```bash
export DASHSCOPE_API_KEY='your-api-key-here'
export LLM_MODEL='qwen-max'   # optional, defaults to qwen-max
pip install -r requirements.txt
python workshop.py --task "Write a short piece about AI in education with examples and challenges"
```

## What the script does

- Persistent reflection memory stored as a flat list in `chapter6/lesson17/reflection_memory.json`
- Semantic retrieval only (DashScope-compatible OpenAI embeddings). If embeddings are unavailable, retrieval returns empty (no keyword fallback)
- Two agents via nested chats:
  - Generator: produces content and is primed with relevant past reflections
  - Reflector: analyzes omissions and outputs JSON; omissions are parsed and saved

## Requirements

Install from `requirements.txt` (autogen-agentchat, openai, dashscope, numpy, etc.).

## Notes

- The memory file path is printed at the end of each run for convenience.
- The storage format is flat (no hierarchical grouping by domain) to simplify reuse.