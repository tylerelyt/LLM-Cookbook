# 📦 Lesson 1: KV-Cache Optimization & Prompt Engineering

> **Build cache-efficient LLM agents via prompt & context design best practices.**

本课程通过真实可运行的单文件示例，逐一讲解 KV-Cache 的工作机制，并从上下文工程角度提出三项关键优化策略：**提示前缀稳定性**、**上下文追加写入**、**缓存边界标记**。所有示例可直接运行，无需依赖公共模块。

---

## 🎯 学习目标

* 理解 KV-Cache 对性能和成本的影响机制
* 掌握 Prompt 稳定性与上下文设计的工程原则
* 学会判断和修复导致缓存失效的设计问题
* 使用手动边界与会话一致性控制缓存命中

---

## 📁 目录结构

```
chapter4/lesson1/
├── stable_prefix_vs_timestamp.py         # 提示前缀稳定性
├── append_only_vs_inplace_edit.py       # 上下文追加 vs 修改
├── manual_cache_boundary.py             # 显式缓存断点
├── session_consistency.py               # 分布式缓存一致性
├── run_all_examples.py                  # 综合运行脚本
├── requirements.txt                     # 依赖文件
└── README.md                           # 说明文档
```

---

## 🧪 示例说明（可单独运行）

### 1️⃣ `stable_prefix_vs_timestamp.py`

> 比较动态时间戳 vs 静态提示前缀对缓存命中率的影响

* ❌ 错误示范：每轮加入时间戳，导致完全缓存失效
* ✅ 正确示范：使用固定前缀，TTFT 显著下降

---

### 2️⃣ `append_only_vs_inplace_edit.py`

> 比较修改历史内容 vs 追加日志的缓存表现

* ❌ 修改 JSON 中已有字段，导致序列化变更破坏缓存
* ✅ 追加新日志项，保持上下文一致性和缓存命中

---

### 3️⃣ `manual_cache_boundary.py`

> 在不自动缓存前缀的环境中使用手动断点

* 使用特殊 token（如 `// __CACHE_END__`）手动划分缓存前缀
* 用于 vLLM/TGI 或自定义推理系统中的缓存策略实现

---

### 4️⃣ `session_consistency.py`

> 使用 `session_id` 控制请求路由，实现分布式缓存一致性

* 模拟在多 Worker 环境中引入 session 粘连
* 展示开启 session ID 后缓存命中率提升显著

---

## 🚀 快速运行

### 安装依赖

```bash
# 方式1: 使用环境配置脚本（推荐）
python setup_and_test.py

# 方式2: 手动安装
pip install vllm torch

# 方式3: 使用项目提供的 requirements.txt
pip install -r requirements.txt
```

### 环境验证

```bash
# 运行环境配置和测试脚本
python setup_and_test.py
```

### 运行示例

```bash
# 方式1: 运行综合脚本（推荐）
python run_all_examples.py

# 方式2: 运行单个示例
python stable_prefix_vs_timestamp.py      # 提示前缀稳定性
python append_only_vs_inplace_edit.py     # 上下文追加 vs 修改
python manual_cache_boundary.py           # 手动缓存边界
python session_consistency.py             # 会话一致性

# 方式3: 快速运行指定示例
python run_all_examples.py 1 3            # 只运行示例1和3
```

---

## 💡 核心启示

> "只要有一个 token 不同，缓存从该点起就全部作废。"

因此，无论是系统提示、历史上下文，还是会话标识，**每个设计细节都决定了缓存是否命中**。Prompt 工程的首要目标，就是保证 KV-Cache 的最大化复用。

### 🎯 vLLM 实现要点

1. **启用前缀缓存**: 使用 `enable_prefix_caching=True` 参数
2. **会话管理**: 通过 `session_id` 实现请求路由一致性
3. **手动边界**: 使用特殊标记分离静态和动态内容
4. **性能监控**: 实时跟踪缓存命中率和响应时间

### 🔧 技术特性

- **异步处理**: 使用 `asyncio` 实现并发请求
- **分布式模拟**: 模拟多工作节点环境
- **智能分割**: 自动识别可缓存的前缀内容
- **会话粘连**: 确保相同会话的请求路由到固定节点

---

## 📚 进一步阅读

* vLLM: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
* TGI: [https://github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
* OpenAI Prompt Guidelines: [https://platform.openai.com/docs](https://platform.openai.com/docs)

