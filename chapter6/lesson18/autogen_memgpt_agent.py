#!/usr/bin/env python3
"""
AutoGen + MemGPT 智能体示例

这个示例展示了如何将 MemGPT 与 AutoGen 框架集成，创建一个具有记忆能力的多智能体系统。
智能体可以进行群聊对话，并且 MemGPT 智能体具有长期记忆和文档检索能力。

参考文档: https://memgpt.readme.io/docs/autogen
"""

import os
import autogen
from typing import List, Dict, Any
import json

# 尝试导入 MemGPT 相关模块
try:
    from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config, load_autogen_memgpt_agent
    MEMGPT_AVAILABLE = True
except ImportError:
    print("警告: MemGPT 未安装，将使用模拟的 MemGPT 智能体")
    MEMGPT_AVAILABLE = False

class AutoGenMemGPTAgent:
    """AutoGen + MemGPT 智能体管理器"""
    
    def __init__(self, api_key: str = None, use_dashscope: bool = True):
        """
        初始化智能体管理器
        
        Args:
            api_key: API 密钥，如果为 None 则从环境变量获取
            use_dashscope: 是否使用 DashScope (默认 True)
        """
        self.use_dashscope = use_dashscope
        
        if use_dashscope:
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            self.model = "qwen-plus"
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if not self.api_key:
                print("警告: 未设置 DASHSCOPE_API_KEY，尝试使用 OpenAI")
                self.use_dashscope = False
                
        if not self.use_dashscope:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4"
            self.base_url = None
            if not self.api_key:
                print("警告: 未设置 API 密钥，某些功能可能无法正常工作")
        
        self.agents = {}
        self.groupchat = None
        self.chat_manager = None
        
    def create_agents(self):
        """创建各种智能体"""
        
        # 配置列表 - 用于非 MemGPT 智能体
        config_list = [
            {
                "model": self.model,
                "api_key": self.api_key,
                "base_url": self.base_url,
            }
        ]
        # 移除 None 值
        config_list = [{k: v for k, v in config.items() if v is not None} for config in config_list]
        
        # MemGPT 智能体配置
        if MEMGPT_AVAILABLE:
            if self.use_dashscope:
                config_list_memgpt = [
                    {
                        "model": self.model,
                        "context_window": 8192,
                        "preset": "memgpt_chat",
                        "model_endpoint_type": "openai",  # DashScope 使用 OpenAI 兼容接口
                        "openai_key": self.api_key,
                        "base_url": self.base_url,
                    }
                ]
            else:
                config_list_memgpt = [
                    {
                        "model": self.model,
                        "context_window": 8192,
                        "preset": "memgpt_chat",
                        "model_endpoint_type": "openai",
                        "openai_key": self.api_key,
                    }
                ]
        else:
            # 模拟配置
            config_list_memgpt = config_list
            
        llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}
        
        # MemGPT 智能体接口配置
        interface_kwargs = {
            "debug": False,
            "show_inner_thoughts": True,
            "show_function_outputs": False,
        }
        
        # 创建 MemGPT 智能体
        if MEMGPT_AVAILABLE:
            self.agents["memgpt_agent"] = create_memgpt_autogen_agent_from_config(
                "MemGPT_Assistant",
                llm_config=llm_config_memgpt,
                system_message="""你是一个具有长期记忆能力的智能助手。你擅长：
                1. 记住对话历史中的重要信息
                2. 基于历史对话提供连贯的建议
                3. 分析复杂问题并提供深入见解
                4. 在群聊中与其他智能体协作
                
                请始终保持友好、专业，并充分利用你的记忆能力来提供有价值的帮助。""",
                interface_kwargs=interface_kwargs,
                default_auto_reply="我正在思考这个问题，让我基于我的记忆来回答...",
                skip_verify=True,
                auto_save=True,
            )
        else:
            # 创建模拟的 MemGPT 智能体
            self.agents["memgpt_agent"] = autogen.AssistantAgent(
                name="MemGPT_Assistant",
                system_message="""我是一个模拟的 MemGPT 智能体（实际 MemGPT 未安装）。
                我具有记忆能力，能够记住对话历史并提供连贯的建议。
                我擅长分析复杂问题并在群聊中与其他智能体协作。""",
                llm_config={"config_list": config_list},
            )
        
        # 创建用户代理
        self.agents["user_proxy"] = autogen.UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",  # 自动回复，不需要人工输入
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"work_dir": "workspace"},
            llm_config={"config_list": config_list},
        )
        
        # 创建专家智能体
        self.agents["expert_agent"] = autogen.AssistantAgent(
            name="Expert_Agent",
            system_message="""你是一个专业的技术专家，擅长：
            1. 技术问题分析和解决方案设计
            2. 代码审查和优化建议
            3. 系统架构设计
            4. 与 MemGPT 智能体协作解决复杂问题
            
            请提供专业、准确的技术建议。""",
            llm_config={"config_list": config_list},
        )
        
        # 创建创意智能体
        self.agents["creative_agent"] = autogen.AssistantAgent(
            name="Creative_Agent",
            system_message="""你是一个创意专家，擅长：
            1. 创新思维和创意生成
            2. 用户体验设计
            3. 产品概念开发
            4. 与 MemGPT 智能体协作产生创新想法
            
            请提供富有创意和想象力的建议。""",
            llm_config={"config_list": config_list},
        )
        
    def setup_groupchat(self):
        """设置群聊"""
        if not self.agents:
            self.create_agents()
            
        # 创建群聊
        self.groupchat = autogen.GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=15,
            speaker_selection_method="round_robin"
        )
        
        # 创建群聊管理器
        self.chat_manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config={"config_list": [{"model": "gpt-4", "api_key": self.openai_api_key}]}
        )
        
    def start_conversation(self, initial_message: str):
        """
        开始群聊对话
        
        Args:
            initial_message: 初始消息
        """
        if not self.chat_manager:
            self.setup_groupchat()
            
        print("🤖 启动 AutoGen + MemGPT 群聊...")
        print(f"📝 初始消息: {initial_message}")
        print("-" * 50)
        
        # 启动对话
        self.agents["user_proxy"].initiate_chat(
            self.chat_manager,
            message=initial_message
        )
        
    def save_memgpt_agent(self):
        """保存 MemGPT 智能体状态"""
        if MEMGPT_AVAILABLE and "memgpt_agent" in self.agents:
            try:
                self.agents["memgpt_agent"].save()
                print("✅ MemGPT 智能体状态已保存")
            except Exception as e:
                print(f"❌ 保存 MemGPT 智能体状态失败: {e}")
                
    def load_memgpt_agent(self, agent_name: str = "MemGPT_Assistant"):
        """加载已保存的 MemGPT 智能体"""
        if MEMGPT_AVAILABLE:
            try:
                self.agents["memgpt_agent"] = load_autogen_memgpt_agent(
                    agent_config={"name": agent_name}
                )
                print("✅ MemGPT 智能体已加载")
            except Exception as e:
                print(f"❌ 加载 MemGPT 智能体失败: {e}")
                print("将创建新的 MemGPT 智能体...")
                self.create_agents()
        else:
            print("❌ MemGPT 未安装，无法加载智能体")
            
    def get_agent_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        info = {
            "total_agents": len(self.agents),
            "agent_names": list(self.agents.keys()),
            "memgpt_available": MEMGPT_AVAILABLE,
            "groupchat_ready": self.chat_manager is not None
        }
        return info

def main():
    """主函数 - 演示 AutoGen + MemGPT 智能体"""
    
    print("🚀 AutoGen + MemGPT 智能体演示")
    print("=" * 50)
    
    # 创建智能体管理器
    agent_manager = AutoGenMemGPTAgent()
    
    # 显示智能体信息
    info = agent_manager.get_agent_info()
    print(f"📊 智能体信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
    print()
    
    # 示例对话
    example_conversations = [
        "请帮我设计一个智能客服系统，需要考虑用户体验和技术实现。",
        "我想开发一个AI助手应用，能够记住用户的偏好并提供个性化服务。",
        "请分析当前AI技术的发展趋势，并预测未来5年的发展方向。",
        "我需要一个创意营销方案来推广我们的新产品。"
    ]
    
    for i, message in enumerate(example_conversations, 1):
        print(f"\n🎯 示例对话 {i}:")
        agent_manager.start_conversation(message)
        
        # 保存 MemGPT 智能体状态
        agent_manager.save_memgpt_agent()
        
        print("\n" + "="*50)
        
        # 询问是否继续
        if i < len(example_conversations):
            user_input = input("按回车键继续下一个对话，或输入 'q' 退出: ")
            if user_input.lower() == 'q':
                break

if __name__ == "__main__":
    main() 