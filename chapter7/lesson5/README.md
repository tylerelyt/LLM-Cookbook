# Lesson 4: RLHF 训练数据构造

本课程演示如何构造 RLHF (Reinforcement Learning from Human Feedback) 训练数据。你将学会：

- 理解 RLHF 的数据需求和格式
- 构造偏好数据对 (chosen vs rejected)
- 生成用于 reward model 训练的比较数据
- 为 PPO 训练准备高质量的偏好数据集

## RLHF 数据特点

RLHF 需要特殊的偏好数据格式：
- `prompt`：输入提示
- `chosen`：人类偏好的回答
- `rejected`：人类不偏好的回答
- `score_chosen`：偏好回答的评分（可选）
- `score_rejected`：非偏好回答的评分（可选）

## 功能概览

- 从现有对话数据构造偏好对
- 使用模型生成多个候选回答并排序
- 自动构造 chosen/rejected 对比数据
- 支持多种偏好信号源（评分、排序、二元选择）
- 输出符合 RLHF 训练要求的格式

## 环境准备

```bash
pip install -r requirements.txt

# 设置 API Key 用于生成候选回答
export OPENAI_API_KEY="your-openai-key"          # 如使用 OpenAI
export DASHSCOPE_API_KEY="your-dashscope-key"    # 如使用 DashScope
```

## 快速开始

```bash
cd chapter7/lesson4

# 主要方法：使用高低级模型差异生成偏好对
python rlhf_constructor.py \
  --input-file ../lesson3/data/alpaca_demo.jsonl \
  --method model_comparison \
  --provider dashscope \
  --high-model qwen-max \
  --low-model qwen-plus \
  --output-path data/rlhf_preference.jsonl

# 从 Few-shot 数据构造偏好对
python rlhf_constructor.py \
  --input-file ../lesson1/data/fewshot_sentiment.jsonl \
  --method model_comparison \
  --provider dashscope \
  --high-model qwen-max \
  --low-model qwen-turbo \
  --output-path data/rlhf_sentiment.jsonl

# 使用 OpenAI 模型构造偏好对
python rlhf_constructor.py \
  --input-file ../lesson2/data/self_instruct.jsonl \
  --method model_comparison \
  --provider openai \
  --high-model gpt-4o \
  --low-model gpt-4o-mini \
  --output-path data/rlhf_openai.jsonl
```

## RLHF 格式示例

### 基本偏好对格式
```json
{
  "prompt": "请解释什么是机器学习",
  "chosen": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中自动学习和改进，而无需被明确编程。通过算法分析大量数据，机器学习模型可以识别模式、做出预测和决策。",
  "rejected": "机器学习就是让机器变得很聪明的技术。"
}
```

### 带评分的格式
```json
{
  "prompt": "请解释什么是机器学习",
  "chosen": "机器学习是人工智能的一个分支...",
  "rejected": "机器学习就是让机器变得很聪明的技术。",
  "score_chosen": 8.5,
  "score_rejected": 3.2
}
```

## 构造方法

### 1. model_comparison (主要方法)
- **核心思路**：针对同一个问题，分别用高级模型和低级模型生成回答
- **高级模型** (如 qwen-max)：生成高质量回答作为 `chosen`
- **低级模型** (如 qwen-plus)：生成较低质量回答作为 `rejected` 
- **优势**：模型能力差异天然保证了偏好对的质量梯度

### 2. generate_pairs
- 为每个 prompt 生成多个候选回答
- 使用质量评估模型对回答排序
- 选择最佳和最差回答构成偏好对

### 3. rank_responses
- 对已有的多个回答进行排序
- 基于排序构造偏好对
- 支持多种排序策略

## 质量控制

- 自动过滤质量差异不明显的对
- 确保 chosen 明显优于 rejected
- 支持人工审核和标注
- 提供数据质量分析报告

## 评估指标

- 偏好对质量分布
- chosen/rejected 长度比较
- 多样性分析
- 潜在偏见检测

## 下一步

完成 RLHF 数据构造后，可以：
1. 训练 reward model
2. 使用 PPO 进行 RLHF 训练
3. 评估模型的偏好对齐效果

## 参考

- InstructGPT 论文：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Anthropic Constitutional AI：[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- HuggingFace TRL：[TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl)
