# Lesson 18: AutoGen + MemGPT 智能体示例

本课程展示了如何将 MemGPT 与 AutoGen 框架集成，创建一个具有长期记忆能力的多智能体系统。通过这个示例，您将学习如何构建能够记住对话历史、进行群聊协作的智能体。

## 主要学习目标

- **AutoGen 框架集成**: 了解如何将 MemGPT 与 AutoGen 多智能体框架结合
- **记忆管理**: 学习如何使用 MemGPT 为智能体提供长期记忆能力
- **多智能体协作**: 探索不同专业智能体之间的协作模式
- **群聊对话**: 实现智能体之间的自然对话和问题解决

## 文件说明

- `README.md`: 本文件，提供课程概述
- `requirements.txt`: 列出所有必要的 Python 依赖
- `autogen_memgpt_agent.py`: 主要的智能体示例代码

## 功能特性

### 🤖 智能体类型
- **MemGPT 智能体**: 具有长期记忆能力的助手
- **专家智能体**: 技术专家，提供专业建议
- **创意智能体**: 创意专家，提供创新想法
- **用户代理**: 管理对话流程

### 🧠 核心功能
- **长期记忆**: MemGPT 智能体能够记住对话历史
- **群聊协作**: 多个智能体协同工作解决问题
- **状态保存**: 可以保存和加载智能体状态
- **自动对话**: 支持无需人工干预的自动对话

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 设置环境变量
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 3. 运行示例
```bash
python autogen_memgpt_agent.py
```

## 使用示例

### 基本用法
```python
from autogen_memgpt_agent import AutoGenMemGPTAgent

# 创建智能体管理器
agent_manager = AutoGenMemGPTAgent()

# 开始对话
agent_manager.start_conversation("请帮我设计一个智能客服系统")
```

### 保存和加载智能体
```python
# 保存 MemGPT 智能体状态
agent_manager.save_memgpt_agent()

# 加载已保存的智能体
agent_manager.load_memgpt_agent("MemGPT_Assistant")
```

## 示例对话场景

1. **技术设计**: 智能客服系统设计
2. **AI 应用开发**: 个性化 AI 助手开发
3. **趋势分析**: AI 技术发展趋势分析
4. **创意营销**: 新产品营销方案设计

## 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MemGPT Agent  │    │  Expert Agent   │    │ Creative Agent  │
│   (记忆管理)     │    │   (技术专家)     │    │   (创意专家)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  GroupChat      │
                    │  (群聊管理)      │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  User Proxy     │
                    │  (用户代理)      │
                    └─────────────────┘
```

## 注意事项

1. **API 密钥**: 需要有效的 OpenAI API 密钥
2. **MemGPT 安装**: 如果未安装 MemGPT，将使用模拟版本
3. **网络连接**: 需要稳定的网络连接访问 OpenAI API
4. **资源消耗**: 长时间对话可能消耗较多 API 调用

## 扩展功能

- **文档处理**: 可以添加文档加载和处理功能
- **本地模型**: 支持使用本地 LLM 模型
- **自定义智能体**: 可以根据需要添加更多专业智能体
- **持久化存储**: 支持更复杂的记忆存储方案

## 参考资源

- [MemGPT 官方文档](https://memgpt.readme.io/docs/autogen)
- [AutoGen 框架文档](https://microsoft.github.io/autogen/)
- [OpenAI API 文档](https://platform.openai.com/docs) 