# Chapter 7 Lesson 1: Few-shot Instruction Data Construction Demo

This demonstration shows how to convert structured datasets into Few-shot format instruction fine-tuning data.

## Features

- Supports multiple task types: sentiment analysis, text classification, QA, translation
- Auto-generates diverse instruction templates
- Constructs Few-shot examples (task description + examples + specific input)
- Outputs standard JSONL format

## Quick Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Demo: Construct Few-shot instruction format from sentiment classification dataset
python construct_fewshot.py \
  --input-file data/sentiment_demo.json \
  --task-type sentiment \
  --output-path data/output.jsonl
```

## Supported Task Types

- `sentiment`: Sentiment analysis
- `classification`: Text classification  
- `qa`: Question answering
- `translate`: Translation

## Input Data Format

Supports CSV, JSON, JSONL, TSV formats with automatic field recognition.

## Output Format

Generated JSONL file contains:
- `instruction`: Few-shot instruction (task description + examples + specific input)
- `input`: Empty string
- `output`: Target answer

## Dataset Features

- **Diverse Expressions**: English, Chinese, and mixed language comments
- **Rich Emotions**: Various positive and negative expressions
- **Real Scenarios**: Realistic movie review language
- **Complex Structures**: Simple and complex emotional expressions

## Sample Output

```json
{
  "instruction": "Determine whether the following sentence expresses positive or negative emotion?\nExample:\nSentence: \"A masterpiece, a profoundly moving film.\" Emotion: Positive\nNow please judge this sentence:\nSentence: \"The plot is predictable and the acting is dull.\" Emotion:",
  "input": "",
  "output": "Negative"
}
```
