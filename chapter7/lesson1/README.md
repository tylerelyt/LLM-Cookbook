# Lesson 1: Few-shot 格式指令微调数据构造演示

本演示展示如何将结构化数据集转换为 Few-shot 格式的指令微调数据。

## 功能特点

- 支持多种任务类型：情感分析、文本分类、问答、翻译
- 自动生成多样化的指令模板
- 构造 Few-shot 示例（任务描述 + 示例 + 具体输入）
- 输出标准 JSONL 格式

## 快速演示

```bash
# 安装依赖
pip install -r requirements.txt

# 演示：从情感分类数据集构造 Few-shot 指令格式
python construct_fewshot.py \
  --input-file data/sentiment_demo.json \
  --task-type sentiment \
  --output-path data/output.jsonl
```

## 支持的任务类型

- `sentiment`: 情感分析
- `classification`: 文本分类  
- `qa`: 问答
- `translate`: 翻译

## 输入数据格式

支持 CSV、JSON、JSONL、TSV 格式，自动识别字段名。

## 输出格式

生成的 JSONL 文件包含：
- `instruction`: Few-shot 指令（任务描述 + 示例 + 具体输入）
- `input`: 空字符串
- `output`: 目标答案

## 示例数据特点

- **多样化表达**：包含英文、中文、中英文混合的评论
- **丰富情感**：正面、负面情感的不同表达方式
- **真实场景**：电影评论的真实语言表达
- **复杂结构**：包含简单和复杂的情感表达

## 示例输出

```json
{
  "instruction": "判断下面这句话表达的是积极情绪还是消极情绪？\n例子：\n句子：\"A masterpiece, a profoundly moving film.\" 情绪：正面\n现在请判断下面这个句子：\n句子：\"The plot is predictable and the acting is dull.\" 情绪：",
  "input": "",
  "output": "负面"
}
```


