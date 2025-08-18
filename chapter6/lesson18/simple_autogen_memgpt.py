#!/usr/bin/env python3
"""
简化版 AutoGen + MemGPT 示例

这是一个简化版本，用于快速测试 AutoGen 和 MemGPT 的集成。
如果 MemGPT 未安装，将使用标准的 AutoGen 智能体。
"""

import os
import autogen

def create_simple_agents():
    """创建简化的智能体"""
    
    # 获取 API 密钥 (优先使用 DashScope)
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if dashscope_key:
        # 使用 DashScope (默认)
        config_list = [
            {
                "model": "qwen-plus",
                "api_key": dashscope_key,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
        ]
    elif openai_key:
        # 备用 OpenAI
        config_list = [
            {
                "model": "gpt-4",
                "api_key": openai_key,
            }
        ]
    else:
        print("❌ 请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        return None
    
    # 创建智能体
    agents = {}
    
    # 用户代理
    agents["user_proxy"] = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        llm_config={"config_list": config_list},
    )
    
    # 助手智能体（模拟 MemGPT）
    agents["assistant"] = autogen.AssistantAgent(
        name="Assistant",
        system_message="""你是一个智能助手，具有以下特点：
        1. 能够记住对话中的重要信息
        2. 提供专业、有用的建议
        3. 在对话中保持连贯性
        4. 擅长技术分析和创意生成
        
        请始终保持友好和专业。""",
        llm_config={"config_list": config_list},
    )
    
    # 专家智能体
    agents["expert"] = autogen.AssistantAgent(
        name="Expert",
        system_message="""你是一个技术专家，擅长：
        1. 技术问题分析和解决方案
        2. 系统架构设计
        3. 代码审查和优化
        4. 技术趋势分析
        
        请提供专业、准确的技术建议。""",
        llm_config={"config_list": config_list},
    )
    
    return agents

def run_simple_conversation():
    """运行简化的对话示例"""
    
    print("🚀 启动简化版 AutoGen + MemGPT 示例")
    print("=" * 40)
    
    # 创建智能体
    agents = create_simple_agents()
    if not agents:
        return
    
    # 创建群聊
    groupchat = autogen.GroupChat(
        agents=list(agents.values()),
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin"
    )
    
    # 创建群聊管理器
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]}
    )
    
    # 示例对话
    example_messages = [
        "请帮我分析一下当前AI技术的发展趋势。",
        "我想开发一个智能聊天机器人，有什么建议吗？",
        "如何设计一个用户友好的移动应用界面？"
    ]
    
    for i, message in enumerate(example_messages, 1):
        print(f"\n🎯 对话 {i}: {message}")
        print("-" * 40)
        
        # 启动对话
        agents["user_proxy"].initiate_chat(
            manager,
            message=message
        )
        
        print("\n" + "=" * 40)
        
        if i < len(example_messages):
            user_input = input("按回车继续，或输入 'q' 退出: ")
            if user_input.lower() == 'q':
                break
    
    print("\n✅ 对话完成！")

if __name__ == "__main__":
    run_simple_conversation() 