# Lesson 2: Self-Instruct 数据构造

本课程演示如何使用 Self-Instruct 方法自动生成指令微调数据。你将学会：

- 设计少量种子任务，利用大模型扩展生成多样化指令
- 自动为生成的指令创建高质量的回答
- 生成符合 Alpaca/Instruction Tuning 风格的问答样本
- 为后续微调准备大规模高质量数据集

## 核心思想

Self-Instruct 通过迭代引导语言模型：
1. 从少量种子任务开始
2. 让模型生成新的指令变体
3. 为每个指令生成对应的回答
4. 过滤和去重，构建大规模数据集

## 功能概览

- 支持提供种子任务列表，自动扩展为更大规模的指令集合
- 自动为生成的指令创建回答
- 输出 `JSONL`（每行一个样本），字段包含：`instruction`、`input`、`output`
- 支持 `OpenAI` 与 `DashScope` API 后端
- 内置去重和质量过滤机制

## 环境准备

```bash
pip install -r requirements.txt

# 设置任一可用 API Key（至少一个）
export OPENAI_API_KEY="your-openai-key"          # 如使用 OpenAI
export DASHSCOPE_API_KEY="your-dashscope-key"    # 如使用 DashScope
```

## 快速开始

```bash
cd chapter7/lesson2

# 一键演示（单文件，无需参数）
python workshop.py

# 方式一：最小示例，使用内置种子任务
python self_instruct.py \
  --num-instructions 100 \
  --output-path data/self_instruct.jsonl

# 方式二：指定提供商与模型（DashScope 示例）
python self_instruct.py \
  --provider dashscope \
  --model qwen-plus \
  --num-instructions 200 \
  --output-path data/self_instruct_qwen.jsonl

# 方式三：自定义种子任务文件
python self_instruct.py \
  --seed-file data/custom_seeds.txt \
  --num-instructions 500 \
  --output-path data/self_instruct_custom.jsonl
```

### 说明
- `workshop.py` 将 Self-Instruct 四步流程（生成指令 → 分类识别 → 实例生成 → 过滤去重）整合在一个脚本内，直接运行即可。
- 种子任务默认从本地 `chapter7/lesson2/data/seed_tasks.jsonl` 读取，可自行替换为上游完整 175 条以保持一致性。
- 中间产物与最终数据默认输出到 `chapter7/lesson2/data/workshop/`。

## Self-Instruct 流程

1. **种子任务设计**：提供 5-10 个高质量种子指令
2. **指令扩展**：模型根据种子生成新指令
3. **回答生成**：为每个指令生成对应回答
4. **质量过滤**：去重复、过滤低质量样本
5. **迭代优化**：重复以上过程直到达到目标数量

## 输出格式示例（JSONL）

```json
{"instruction": "解释二叉搜索树的性质，并给出示例", "input": "", "output": "二叉搜索树（BST）..."}
{"instruction": "根据需求编写 SQL 查询", "input": "表 users(id, name, age)", "output": "SELECT ..."}
```

## 参数说明

- `--provider`：`openai` 或 `dashscope`，默认 `openai`
- `--model`：模型名称，示例：`gpt-4o-mini`、`qwen-plus`
- `--num-instructions`：目标指令数量
- `--seed-file`：自定义种子任务文本文件路径（可选）
- `--output-path`：输出 JSONL 文件路径
- `--temperature`：采样温度，默认 0.7
- `--max-tokens`：单次生成最大 tokens，默认 512

## 建议与注意

- 质量优先：少量高质量种子任务 > 海量低质量扩展
- 去重与多样性：脚本内置简单去重，必要时可二次清洗
- 成本控制：请合理设置 `num-instructions` 与 `max-tokens`，并监控 API 费用

## 下一步

完成数据生成后，前往 `chapter7/lesson3` 使用 Alpaca 风格格式化训练数据。

## 参考

- Self-Instruct 论文：[Self-Instruct: Aligning Language Models with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
- 原始实现：[yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)