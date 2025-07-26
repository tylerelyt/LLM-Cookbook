#!/usr/bin/env python3
"""
简化版Reflexion Agent示例
展示AutoGen中Reflexion机制的核心概念
"""

import os
from autogen import AssistantAgent, UserProxyAgent
from autogen.cache import Cache


def simple_reflexion_demo():
    """简化的Reflexion演示"""
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请设置DASHSCOPE_API_KEY环境变量")
        return
    
    # 读取当前 LLM 配置（与主项目保持一致）
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-max")
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
    
    if not DASHSCOPE_API_KEY:
        raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

    llm_config = {
        "model": LLM_MODEL,
        "api_key": DASHSCOPE_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    
    # 创建代理
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=5,
        code_execution_config={"use_docker": False}
    )
    
    # 写作助手
    writer = AssistantAgent(
        name="writer",
        system_message="你是一个写作助手，负责创作文章。",
        llm_config=llm_config
    )
    
    # 反思助手
    critic = AssistantAgent(
        name="critic", 
        system_message="你是一个评论家，对文章进行批评和改进建议。",
        llm_config=llm_config
    )
    
    # 反思消息函数
    def reflection_msg(recipient, messages, sender, config):
        print("🤔 反思中...")
        last_content = recipient.chat_messages_for_summary(sender)[-1]['content']
        return f"请对以下内容进行批评分析：\n\n{last_content}"
    
    # 设置嵌套聊天
    user_proxy.register_nested_chats([
        {
            "recipient": critic,
            "message": reflection_msg,
            "max_turns": 1,
        }
    ], trigger=writer)
    
    # 运行演示
    with Cache.disk(cache_seed=42) as cache:
        user_proxy.initiate_chat(
            writer,
            message="写一篇关于人工智能的短文，不超过200字。",
            max_turns=2,
            cache=cache
        )


if __name__ == "__main__":
    simple_reflexion_demo() 