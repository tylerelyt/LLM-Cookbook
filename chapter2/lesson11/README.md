# Knowledge Graph Construction with Chain-of-Thought

An advanced pipeline for building knowledge graphs from unstructured text using Large Language Models with Chain-of-Thought reasoning.

## ✨ Features

- **Entity-Relation-Attribute Separation**: Correctly distinguishes between entities, relationships, and attributes
- **Chain-of-Thought Reasoning**: Uses multi-step reasoning to improve entity linking accuracy
- **Interactive Visualization**: Generates beautiful, interactive HTML visualizations using Pyvis
- **Schema Optimization**: Automatically refines and normalizes knowledge graph schemas
- **Comprehensive Logging**: Detailed pipeline execution logs for debugging and analysis

## 🚀 Quick Start

### Installation

```bash
pip install openai networkx pyvis
```

### Setup

Set your LLM API key as an environment variable:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
# or
export OPENAI_API_KEY="your-api-key-here"
```

### Run

```bash
python knowledge_pipeline.py
```

This will process the sample text and generate:
- `examples/demo_outputs/knowledge_graph.html` - Interactive visualization
- `examples/demo_outputs/knowledge_graph.json` - Structured graph data
- `examples/demo_outputs/pipeline.log` - Detailed execution log

## 📋 Pipeline Steps

1. **Candidate Extraction**: Extract entities, relations, and attributes from text
2. **Schema Optimization**: Refine and normalize entity/relation/attribute types  
3. **Refinement & Relabeling**: Apply optimized schema to candidates
4. **Role Analysis**: Use Chain-of-Thought to identify unique entities
5. **Context-Aware Entity Linking**: Merge aliases based on role analysis

## 🎯 Example Output

**Input Text:**
> "一只敏捷的棕色狐狸跳过了一只懒狗。这只又快又聪明的狐狸戏弄了一条昏昏欲睡的老狗。这只名叫弗莱迪的懒狗很不高兴。这只名叫乐乐的狐狸非常得意。"

**Generated Knowledge Graph:**
- **Entities**: 乐乐 (狐狸), 弗莱迪 (狗)
- **Attributes**: 性格特征, 情感状态
- **Relations**: 跳过, 戏弄
- **Structure**: 2 nodes, 2 edges, density 1.0

## 🏗️ Architecture

The system uses a multi-step pipeline with LLM-based processing:

- **Flexible LLM Support**: Compatible with OpenAI GPT and Alibaba Qwen models
- **Robust Error Handling**: Comprehensive validation and fallback mechanisms
- **Modular Design**: Easy to extend and customize for different domains
- **Performance Optimized**: Efficient processing with minimal API calls

## 📊 Visualization Features

- **Interactive Graph**: Drag, zoom, and hover interactions
- **Entity Attributes**: Rich hover tooltips showing entity properties
- **Relationship Labels**: Clear edge labeling with relationship types
- **Automatic Layout**: Physics-based node positioning
- **Export Options**: Save as HTML or JSON formats

## 🔧 Configuration

The pipeline supports various configuration options:

- **Model Selection**: Choose between different LLM models
- **Schema Types**: Customize entity, relation, and attribute types
- **Visualization**: Adjust colors, layouts, and interaction settings
- **Logging**: Control detail level and output formats

## 📁 Project Structure

```
├── knowledge_pipeline.py    # Main pipeline implementation
├── examples/
│   └── demo_outputs/       # Example outputs
│       ├── knowledge_graph.html
│       ├── knowledge_graph.json
│       ├── pipeline.log
│       └── sample_text.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License. 

## 环境配置

### API 密钥设置

本项目需要设置API密钥才能运行：

```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # 可选，如果使用OpenAI模型
```

您可以在终端中临时设置，或者添加到您的shell配置文件中（如 `~/.bashrc` 或 `~/.zshrc`）以便永久生效。 