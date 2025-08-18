# Lesson 3: Alpaca 训练数据构造

本课程演示如何构造 Alpaca 风格的训练数据，这是 Stanford Alpaca 项目使用的标准格式。你将学会：

- 理解 Alpaca 数据格式的标准化结构
- 从各种源数据转换为 Alpaca 格式
- 构造适合监督微调 (SFT) 的对话数据
- 为指令微调准备高质量的训练集

## Alpaca 格式特点

Alpaca 格式是指令微调的经典格式，包含三个字段：
- `instruction`：任务指令
- `input`：可选的输入上下文
- `output`：期望的输出

## 功能概览

- 支持多种源数据格式转换为 Alpaca 标准格式
- 内置 Alpaca 风格的指令模板库
- 自动处理有/无输入的两种情况
- 支持批量数据转换和质量过滤
- 输出标准 JSONL 格式，可直接用于训练

## 环境准备

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
cd chapter7/lesson3

# 方式一：从 Few-shot 数据转换为 Alpaca 格式
python alpaca_constructor.py \
  --input-file ../lesson1/data/fewshot_sentiment.jsonl \
  --output-path data/alpaca_sentiment.jsonl \
  --conversion-type fewshot_to_alpaca

# 方式二：从问答对构造 Alpaca 格式
python alpaca_constructor.py \
  --input-file data/qa_pairs.json \
  --output-path data/alpaca_qa.jsonl \
  --conversion-type qa_to_alpaca

# 方式三：从对话数据构造 Alpaca 格式
python alpaca_constructor.py \
  --input-file data/conversations.jsonl \
  --output-path data/alpaca_chat.jsonl \
  --conversion-type chat_to_alpaca
```

## Alpaca 格式示例

### 无输入示例
```json
{
  "instruction": "解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中自动学习和改进，而无需被明确编程..."
}
```

### 有输入示例
```json
{
  "instruction": "将以下文本翻译成英文",
  "input": "人工智能正在改变我们的世界",
  "output": "Artificial intelligence is changing our world"
}
```

## 转换类型

- `fewshot_to_alpaca`：将 Few-shot 指令转换为简洁的 Alpaca 格式
- `qa_to_alpaca`：将问答对转换为 Alpaca 格式
- `chat_to_alpaca`：将对话数据转换为 Alpaca 格式
- `classification_to_alpaca`：将分类数据转换为 Alpaca 格式

## 数据质量控制

- 自动过滤过短或过长的指令/回答
- 去重复指令，保持数据集多样性
- 标准化格式，确保训练兼容性
- 支持人工审核和批量编辑

## 下一步

完成 Alpaca 数据构造后，可以：
1. 直接用于 LoRA 或全参数微调
2. 进入 `chapter7/lesson4` 构造 RLHF 训练数据
3. 与其他数据集混合，增加训练多样性

## 参考

- Stanford Alpaca 项目：[tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- Alpaca 数据集：[yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
