#!/usr/bin/env python3
"""
Lesson 17 Workshop（单文件，仅真实 AutoGen 版本）

功能：基于 AutoGen 的“遗漏点发现 + 反思改进”系统，要求可用 API。

用法：
    export DASHSCOPE_API_KEY='your-api-key-here'
    export LLM_MODEL='qwen-max'   # 可选，默认 qwen-max
    pip install -r requirements.txt
    python workshop.py --task "你的任务文本"

依赖：见同目录 `requirements.txt`
"""

import argparse
import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============== 记忆系统：遗漏点长期记忆（语义检索，扁平存储） ===============
class OmissionMemory:
    """遗漏点长期记忆系统（扁平存储）。

    - 持久化到脚本同目录 JSON 文件 `reflection_memory.json`
    - 仅语义检索（DashScope 兼容 OpenAI 接口）。不可用则不检索（返回空）
    - 存储结构为扁平列表：每条为一个反思条目（不再按领域分层）。
    """

    def __init__(self, memory_file: str = None) -> None:
        # 将持久化文件固定在脚本目录，避免工作目录变化导致丢失
        default_path = os.path.join(os.path.dirname(__file__), "reflection_memory.json")
        self.memory_file = memory_file or default_path
        self.entries: List[Dict] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self._embedding_client = self._init_embedding_client()
        self.load_memory()

    def _init_embedding_client(self):
        try:
            import openai  # type: ignore

            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                return None

            # 使用 DashScope 兼容模式
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            return client
        except Exception:
            return None

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not self._embedding_client:
            return None
        try:
            resp = self._embedding_client.embeddings.create(
                model="text-embedding-v1", input=text
            )
            return np.array(resp.data[0].embedding)
        except Exception:
            return None

    def _text_for_embedding(self, item: Dict) -> str:
        parts = [
            item.get("domain", ""),
            item.get("type", ""),
            item.get("description", ""),
            item.get("suggestion", ""),
        ]
        return " ".join([p for p in parts if p])

    def add_omissions(self, domain: str, omissions: List[Dict]) -> None:
        for omission in omissions:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "domain": domain,
                "type": omission.get("type", ""),
                "description": omission.get("description", ""),
                "suggestion": omission.get("suggestion", ""),
            }
            self.entries.append(entry)

            emb_text = self._text_for_embedding(entry)
            emb = self._get_embedding(emb_text)
            if emb is not None:
                self.embeddings[entry["id"]] = emb

        self.save_memory()
        print(f"💾 已保存 {len(omissions)} 个遗漏点到领域：{domain}")

    def get_relevant_omissions(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.entries:
            return []

        # 仅语义召回（若 embeddings 缺失则懒加载计算）
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return []

        scored: List[Tuple[float, Dict]] = []
        for item in self.entries:
            emb = self.embeddings.get(item["id"])  # type: ignore[index]
            if emb is None:
                # 懒加载为该条记忆生成嵌入
                text = self._text_for_embedding(item)
                emb = self._get_embedding(text)
                if emb is None:
                    continue
                self.embeddings[item["id"]] = emb  # type: ignore[index]
            sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def save_memory(self) -> None:
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_memory(self) -> None:
        if not os.path.exists(self.memory_file):
            return
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 新版仅支持扁平列表
            self.entries = data if isinstance(data, list) else []
        except Exception:
            self.entries = []


memory = OmissionMemory()


# ========================= 真实模式（AutoGen 多智能体） =========================
def _autogen_available() -> bool:
    try:
        import autogen  # noqa: F401
        return True
    except Exception:
        return False


def _create_llm_config() -> Optional[Dict]:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None

    model = os.getenv("LLM_MODEL", "qwen-max")
    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
        ],
        "temperature": 0.7,
    }


def _reflection_message(recipient, messages, sender, config):
    content = recipient.chat_messages_for_summary(sender)[-1]["content"]

    # 在反思前检索相关历史反思，作为上下文提示
    relevant = memory.get_relevant_omissions(content)
    hints = ""
    if relevant:
        lines = [f"• {it.get('description', '')}（建议：{it.get('suggestion', '')}）" for it in relevant]
        hints = "\n历史遗漏点参考:\n" + "\n".join(lines)

    return f"""请分析以下内容的遗漏点：

{content}
{hints}

请以JSON格式输出遗漏点分析：
```json
{{
  "domain": "内容主题领域（自动识别）",
  "omissions": [
    {{
      "type": "遗漏类型",
      "description": "具体遗漏描述",
      "suggestion": "改进建议"
    }}
  ]
}}
```"""


def _extract_omissions_from_reflection(reflection_text: str) -> Tuple[str, List[Dict]]:
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", reflection_text, re.DOTALL)
        if not match:
            return "通用", []
        data = json.loads(match.group(1))
        domain = data.get("domain", "通用")
        omissions = data.get("omissions", [])
        if isinstance(omissions, list):
            return domain, omissions
        return domain, []
    except Exception:
        return "通用", []


def run_real_autogen_flow(task: str) -> None:
    from autogen import AssistantAgent, UserProxyAgent  # type: ignore

    llm_config = _create_llm_config()
    if not llm_config:
        raise SystemExit("❌ 未检测到 DASHSCOPE_API_KEY，请先设置后再运行。")

    print("🔧 使用真实 AutoGen 反思流程（DashScope 兼容）")

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config={"use_docker": False},
    )

    # 在生成前检索相关历史反思，给生成代理作为系统提示
    prior_reflections = memory.get_relevant_omissions(task)
    prior_text = ""
    if prior_reflections:
        prior_text = "历史遗漏点参考：\n" + "\n".join(
            [f"- {it.get('description', '')}（建议：{it.get('suggestion', '')}）" for it in prior_reflections]
        )

    generator = AssistantAgent(
        name="generator",
        system_message=(
            "你是内容生成专家。根据要求生成内容，严格避免历史反思中指出的遗漏点。\n" + prior_text
        ).strip(),
        llm_config=llm_config,
    )

    reflector = AssistantAgent(
        name="reflector",
        system_message="你是遗漏点发现专家。分析内容遗漏点并输出JSON格式结果。",
        llm_config=llm_config,
    )

    user_proxy.register_nested_chats(
        [
            {
                "recipient": reflector,
                "message": _reflection_message,
                "max_turns": 1,
            }
        ],
        trigger=generator,
    )

    result = user_proxy.initiate_chat(generator, message=task, max_turns=2)

    # 提取遗漏点并入库（优先从反思代理的消息中抽取；若失败则全局扫描一次）
    saved = False
    if reflector.name in user_proxy.chat_messages:
        msgs = user_proxy.chat_messages[reflector.name]
        if msgs:
            reflection_text = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else str(msgs[-1])
            domain, omissions = _extract_omissions_from_reflection(reflection_text)
            if omissions:
                memory.add_omissions(domain, omissions)
                print(f"✅ 已保存 {len(omissions)} 个遗漏点到领域：{domain}")
                saved = True

    if not saved:
        # 全量扫描所有对话消息，寻找符合结构的 JSON 代码块
        try:
            for agent_name, msgs in user_proxy.chat_messages.items():
                for msg in msgs[::-1]:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    domain, omissions = _extract_omissions_from_reflection(content)
                    if omissions:
                        memory.add_omissions(domain, omissions)
                        print(f"✅（全局扫描）已保存 {len(omissions)} 个遗漏点到领域：{domain}")
                        saved = True
                        break
                if saved:
                    break
        except Exception:
            pass

    print(f"📊 当前反思条目数：{len(memory.entries)}")
    # 运行结束时明确提示存储位置，方便用户后续加载复用
    print(f"🗂️ 反思记忆持久化文件：{memory.memory_file}")


# ================================== CLI ==================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 17 单文件 Workshop（真实 AutoGen 版）")
    parser.add_argument(
        "--task",
        type=str,
        default=(
            "写一篇关于人工智能在医疗领域应用的文章，要求：\n- 500字左右\n- 包含具体应用案例\n- 讨论挑战和前景"
        ),
        help="自定义任务文本",
    )
    args = parser.parse_args()

    print("🚀 启动 Lesson 17 Workshop（真实 AutoGen 版）")

    if not _autogen_available():
        raise SystemExit(
            "❌ 未安装 autogen-agentchat，请先执行: pip install -r requirements.txt"
        )

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit(
            "❌ 未检测到 DASHSCOPE_API_KEY，请先设置环境变量后再运行。"
        )

    run_real_autogen_flow(args.task)


if __name__ == "__main__":
    main()


