# Chapter 8 Lesson 1: DashScope Embedding + 逻辑回归情感分类

本课程演示如何使用 DashScope Embedding API 生成文本向量，然后训练逻辑回归模型进行情感分类。

## 功能特点

- **DashScope Embedding**: 使用阿里云 DashScope 的 text-embedding-v1 模型
- **逻辑回归分类**: 简单高效的线性分类器
- **多语言支持**: 支持中英文混合文本
- **完整流程**: 从数据加载到模型训练、评估、预测的完整流程
- **可视化结果**: 自动生成混淆矩阵和性能对比图表

## 技术架构

```
文本数据 → DashScope Embedding → 特征标准化 → 逻辑回归 → 情感分类
```

## 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

## 快速开始

```bash
cd chapter8/lesson1

# 基础演示
python dashscope_lr_sentiment.py

# 自定义参数
python dashscope_lr_sentiment.py \
  --input-file data/sentiment_demo.json \
  --test-size 0.2 \
  --batch-size 5 \
  --model-path models/my_sentiment_model.pkl
```

## 参数说明

- `--input-file`: 输入数据文件路径 (JSON格式)
- `--test-size`: 测试集比例，默认 0.2
- `--random-state`: 随机种子，默认 42
- `--batch-size`: Embedding 生成批次大小，默认 32
- `--model-path`: 模型保存路径，默认 models/dashscope_lr_sentiment.pkl
- `--results-path`: 结果保存目录，默认 results/
- `--plot-results`: 是否绘制结果图表，默认 True

## 输入数据格式

JSON 格式，包含 sentence 和 label 字段：

```json
[
  {"sentence": "这部电影真的很棒！", "label": 1},
  {"sentence": "剧情太无聊了。", "label": 0},
  {"sentence": "The movie was fantastic!", "label": 1},
  {"sentence": "This film is terrible.", "label": 0}
]
```

## 输出结果

### 1. 模型文件
- `models/dashscope_lr_sentiment.pkl`: 训练好的模型

### 2. 评估结果
- `results/classification_results.json`: 详细的分类报告
- `results/classification_results.png`: 可视化图表

### 3. 控制台输出
- 训练过程信息
- 准确率、精确率、召回率、F1分数
- 演示预测结果

## 性能指标

模型会输出以下性能指标：
- **准确率 (Accuracy)**: 整体分类正确率
- **精确率 (Precision)**: 预测为正面的样本中真正为正面的比例
- **召回率 (Recall)**: 真实正面样本中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均

## 使用示例

### 训练模型
```python
from dashscope_lr_sentiment import DashScopeSentimentClassifier

# 初始化分类器
classifier = DashScopeSentimentClassifier()

# 加载数据
texts, labels = classifier.load_data("data/sentiment_demo.json")

# 生成 Embedding
embeddings = classifier.generate_embeddings(texts)

# 训练模型
# ... (自动处理数据划分和训练)
```

### 预测新文本
```python
# 预测单个文本
texts = ["这部电影真的很棒！", "剧情太无聊了"]
predictions, probabilities = classifier.predict(texts)

for text, pred, prob in zip(texts, predictions, probabilities):
    sentiment = "正面" if pred == 1 else "负面"
    confidence = max(prob)
    print(f"{text} -> {sentiment} ({confidence:.3f})")
```

### 加载已训练模型
```python
# 加载模型
classifier = DashScopeSentimentClassifier()
classifier.load_model("models/dashscope_lr_sentiment.pkl")

# 直接预测
predictions, probabilities = classifier.predict(["新文本"])
```

## 优势特点

1. **高效性**: DashScope 提供高质量的文本表示
2. **轻量级**: 逻辑回归模型训练快速，推理高效
3. **可解释性**: 线性模型易于理解和调试
4. **多语言**: 支持中英文混合文本
5. **完整流程**: 从数据处理到模型部署的完整解决方案

## 注意事项

- 需要设置 DASHSCOPE_API_KEY 环境变量
- API 调用会产生费用，请注意控制批次大小
- 数据量较少时可能需要调整模型参数
- 逻辑回归适合线性可分的数据

## 扩展应用

- 可以替换为其他分类器 (SVM, Random Forest 等)
- 支持多分类任务扩展
- 可以集成到 Web 服务中
- 支持增量学习和模型更新
