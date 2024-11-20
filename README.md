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

### 🔧 Chapter 2: Intermediate Processing
**Document processing and text analysis**
- Advanced text processing and semantic analysis
- Multi-format document handling and extraction
- Content understanding and transformation patterns

### 🔍 Chapter 3: Retrieval & Knowledge Engineering  
**Advanced RAG architectures and knowledge graphs**
- Production-grade RAG systems with BGE-m3 and BGE-reranker
- Automated entity extraction and knowledge graph construction
- Chain-of-Thought reasoning for knowledge extraction
- Enterprise NL2SQL with intelligent rejection mechanisms
- Interactive graph visualization with Pyvis and NetworkX

### 🤖 Chapter 4: Multi-Agent Orchestration
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

# Each lesson has its own dependencies - install what you need:
# Chapter 1: Foundation
pip install -r chapter1/lesson1/requirements.txt  # Agent Architectures
pip install -r chapter1/lesson2/requirements.txt  # Mathematical Reasoning
pip install -r chapter1/lesson3/requirements.txt  # Multi-modal Dialogs

# Chapter 2: Intermediate Processing  
pip install -r chapter2/lesson6/requirements.txt  # Text Processing
pip install -r chapter2/lesson7/requirements.txt  # Document Understanding
pip install -r chapter2/lesson9/requirements.txt  # Content Transformation

# Chapter 3: Advanced Knowledge Engineering
pip install -r chapter3/lesson10/requirements.txt  # RAG System
pip install -r chapter3/lesson11/requirements.txt  # Knowledge Graph
pip install -r chapter3/lesson12/requirements.txt  # NL2SQL

# Chapter 4: Expert Multi-Agent Systems
pip install -r chapter4/lesson17/requirements.txt  # Agent Coordination
pip install -r chapter4/lesson18/requirements.txt  # Advanced Collaboration
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
cd chapter3/lesson11
python knowledge_pipeline.py

# Create production-grade RAG systems
cd chapter3/lesson10
python rag_pipeline.py

# Develop enterprise NL2SQL systems
cd chapter3/lesson12
python nl2sql_engine.py

# Create multi-agent collaboration systems
cd chapter4/lesson17  
python agent_manager.py

# Develop conversational AI with tool integration
cd chapter1/lesson3
python multimodal_chat.py
```

## 🏛️ Project Structure

```
LLM-Workshop/
├── chapter1/                    # Conversational Intelligence (Foundation)
│   ├── lesson1/                 # Agent architectures & tool integration
│   │   └── requirements.txt     # Flask, transformers, torch
│   ├── lesson2/                 # Mathematical reasoning & expression parsing  
│   │   └── requirements.txt     # sympy, scipy, mathematical libs
│   ├── lesson3/                 # Multi-modal dialog systems
│   │   └── requirements.txt     # Multi-modal processing
│   ├── lesson4/                 # Advanced dialog patterns
│   │   └── requirements.txt     # Conversation management
│   └── lesson5/                 # Reasoning optimization
│       └── requirements.txt     # Performance optimization
├── chapter2/                    # Intermediate Processing  
│   ├── lesson6/                 # Advanced text processing
│   │   └── requirements.txt     # ollama, NLP libraries
│   ├── lesson7/                 # Document understanding
│   │   └── requirements.txt     # langchain, document processing
│   └── lesson9/                 # Content transformation
│       └── requirements.txt     # Format conversion tools
├── chapter3/                    # Advanced Knowledge Engineering
│   ├── lesson10/                # Production-grade RAG with BGE-m3
│   │   ├── rag_pipeline.py
│   │   ├── requirements.txt     # BGE models, RAG dependencies
│   │   └── README.md
│   ├── lesson11/                # Knowledge graph construction
│   │   ├── knowledge_pipeline.py
│   │   ├── requirements.txt     # networkx, pyvis, CoT reasoning
│   │   └── examples/outputs/
│   └── lesson12/                # Enterprise NL2SQL systems
│       ├── nl2sql_engine.py
│       ├── requirements.txt     # SQL, vector search, analytics
│       └── README.md
├── chapter4/                    # Expert-Level Multi-Agent Systems  
│   ├── lesson17/                # Agent coordination frameworks
│   │   └── requirements.txt     # Multi-agent dependencies
│   └── lesson18/                # Advanced collaboration patterns
│       └── requirements.txt     # Advanced collaboration dependencies
└── README.md                    # Project documentation
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
