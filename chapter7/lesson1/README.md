# Lesson 1: Automated Few-shot Instruction Data Construction

This lesson demonstrates how to automatically construct few-shot instruction fine-tuning data from structured datasets. Learn the systematic approach to transform raw data into high-quality training examples for instruction-following models.

## Key Automation Features

- **Multi-task Support**: Automatic handling of sentiment analysis, classification, QA, and translation
- **Template Generation**: Automatically creates diverse instruction templates for the same task
- **Few-shot Construction**: Systematically builds instruction-example-input patterns
- **Format Standardization**: Outputs ready-to-use JSONL format for instruction fine-tuning

## Quick Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Demo: Construct few-shot instruction data from sentiment dataset
python construct_fewshot.py \
  --input-file data/sentiment_demo.json \
  --task-type sentiment \
  --output-path data/output.jsonl
```

## Automation Process

### 1. Task Type Detection
- `sentiment`: Sentiment analysis tasks
- `classification`: General text classification
- `qa`: Question-answering tasks
- `translate`: Translation tasks

### 2. Data Format Handling
Automatically processes CSV, JSON, JSONL, TSV formats with intelligent field recognition.

### 3. Instruction Generation Pipeline
1. **Template Selection**: Randomly picks from diverse instruction templates
2. **Example Pairing**: Automatically selects relevant examples for few-shot learning
3. **Format Construction**: Builds instruction-input-output triplets
4. **Quality Assurance**: Validates data completeness and format consistency

## Example Data Features

- **Diverse Expressions**: English, Chinese, and mixed-language content
- **Rich Emotions**: Various positive and negative sentiment expressions
- **Real Scenarios**: Authentic movie review language patterns
- **Complex Structures**: Simple to complex emotional expressions

## Automated Output Example

**Input Data**: `{"sentence": "The plot is predictable and the acting is dull.", "label": 0}`

**Generated Instruction**:
```json
{
  "instruction": "判断下面这句话表达的是积极情绪还是消极情绪？\n例子：\n句子：\"A masterpiece, a profoundly moving film.\" 情绪：正面\n现在请判断下面这个句子：\n句子：\"The plot is predictable and the acting is dull.\" 情绪：",
  "input": "",
  "output": "负面"
}
```

## Why Automation Matters

1. **Scale**: Process thousands of examples automatically
2. **Consistency**: Maintain uniform instruction format across datasets
3. **Diversity**: Generate multiple template variations to improve robustness
4. **Efficiency**: Reduce manual data preparation time from days to minutes
5. **Quality**: Systematic approach ensures complete and valid instruction data


