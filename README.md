# LLM-Workshop

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/tylerelyt/LLM-Workshop.svg?style=social&label=Star)](https://github.com/tylerelyt/LLM-Workshop)
[![GitHub forks](https://img.shields.io/github/forks/tylerelyt/LLM-Workshop.svg?style=social&label=Fork)](https://github.com/tylerelyt/LLM-Workshop)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

> **Learn Large Language Model development through hands-on projects and real-world implementations.**

A practical workshop for building LLM applications from scratch. Learn by doing - each project guides you through essential concepts while building production-ready systems, from conversational AI to multi-agent architectures.

## 🏗️ Workshop Structure

**LLM-Workshop** provides hands-on learning experiences through practical projects, featuring:

- **🧠 Conversational AI** - Advanced dialog systems with reasoning capabilities
- **🔍 Retrieval-Augmented Generation** - Knowledge-grounded QA systems and vector databases
- **🕸️ Knowledge Graph Engineering** - Automated entity extraction and graph visualization
- **🤖 Multi-Agent Systems** - Distributed AI coordination and task orchestration
- **⚡ Production Infrastructure** - Scalable deployment patterns and monitoring

## 🛠️ Learning Modules

### 🧠 Chapter 1: Conversational Intelligence
**Core dialog systems and reasoning frameworks**
- Foundation models with long-context processing capabilities
- Instruction following with chain-of-thought reasoning
- Tool-augmented agents for mathematical and logical tasks
- In-context learning and multi-turn conversation modeling

### 🔍 Chapter 2: Retrieval & Knowledge Engineering  
**Advanced RAG architectures and knowledge graphs**
- Semantic retrieval with vector databases and embedding models
- Automated entity extraction and knowledge graph construction
- Interactive graph visualization with Pyvis and NetworkX
- Multi-hop reasoning with feedback optimization loops

### 🤖 Chapter 3: Multi-Agent Orchestration
**Distributed AI coordination and collaboration patterns**
- Inter-agent communication protocols and message passing
- Task decomposition and hierarchical planning strategies
- Consensus mechanisms and conflict resolution algorithms
- Distributed reasoning and collaborative problem solving


## ⚡ Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **API Keys**: OpenAI or Alibaba DashScope

### Installation
```bash
git clone https://github.com/tylerelyt/LLM-Workshop.git
cd LLM-Workshop
pip install -r requirements.txt
```

### Environment Setup
```bash
# Required: Set your LLM API key
export DASHSCOPE_API_KEY="your-dashscope-key"
# or
export OPENAI_API_KEY="your-openai-key"

# Recommended: Use virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Start Building
```bash
# Build an interactive knowledge graph from scratch
cd chapter2/lesson11
python knowledge_pipeline.py

# Create multi-agent collaboration systems
cd chapter3/lesson17  
python agent_manager.py

# Develop conversational AI with tool integration
cd chapter1/lesson3
python demo.py
```

## 🏛️ Project Structure

```
LLM-Workshop/
├── chapter1/                    # Conversational Intelligence
│   ├── lesson1/                 # Agent architectures & tool integration
│   ├── lesson2/                 # Mathematical reasoning & expression parsing  
│   ├── lesson3/                 # Multi-modal dialog systems
│   └── lesson*/                 # Advanced reasoning patterns
├── chapter2/                    # Knowledge Engineering
│   ├── lesson11/                # Knowledge graph construction pipeline
│   │   ├── knowledge_pipeline.py
│   │   └── examples/demo_outputs/
│   └── lesson*/                 # RAG & retrieval systems
├── chapter3/                    # Multi-Agent Systems  
│   ├── lesson17/                # Agent coordination frameworks
│   └── lesson18/                # Multi-modal knowledge graphs
└── requirements.txt             # Production dependencies
```

## 🆘 Troubleshooting

**Getting Updates**
```bash
git pull origin main  # Get latest code and resources
```

**Common Issues**
- **Environment Setup**: Ensure all dependencies are installed per documentation
- **API Keys**: Verify your LLM API keys are properly configured
- **Python Version**: Requires Python 3.8+ with compatible packages

For persistent issues, check lesson-specific README files or open an issue.

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** this repository
2. **Create** a feature branch (`git checkout -b feature-amazing`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature-amazing`)
5. **Open** a Pull Request

## 👥 Contributors

<a href="https://github.com/tylerelyt/LLM-Workshop/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tylerelyt/LLM-Workshop" />
</a>

## 📈 Stargazers Over Time

[![Stargazers over time](https://starchart.cc/tylerelyt/LLM-Workshop.svg?variant=adaptive)](https://starchart.cc/tylerelyt/LLM-Workshop)

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this code for any purpose, including commercial use.

## 🎯 Who Should Join This Workshop

**LLM-Workshop** is designed for developers who learn best by building real projects:

- **🚀 Aspiring AI Engineers** - Build your first production LLM applications
- **💻 Full-Stack Developers** - Add AI capabilities to your existing skills
- **🔬 Research Engineers** - Bridge the gap between papers and production code  
- **🏗️ System Builders** - Learn scalable AI architecture patterns through practice

> Learn by doing: Each module takes you from concept to working implementation, building real systems you can deploy and extend.
