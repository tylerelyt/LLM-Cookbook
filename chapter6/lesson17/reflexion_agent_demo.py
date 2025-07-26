#!/usr/bin/env python3
"""
Reflexion Agent Demo with AutoGen
基于Microsoft AutoGen文档的Reflexion实现
演示如何使用嵌套聊天实现LLM反思机制
"""

import os
import sys
from typing import Dict, Any

# 检查并安装必要的依赖
try:
    from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
    from autogen.cache import Cache
    from autogen.coding import LocalCommandLineCodeExecutor
except ImportError:
    print("正在安装autogen-agentchat...")
    os.system("pip install autogen-agentchat")
    from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
    from autogen.cache import Cache
    from autogen.coding import LocalCommandLineCodeExecutor


def setup_environment():
    """设置环境变量和配置"""
    # 检查DashScope API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告: 未找到DASHSCOPE_API_KEY环境变量")
        print("请设置您的DashScope API密钥:")
        print("export DASHSCOPE_API_KEY='your-api-key-here'")
        return False
    
    # 创建代码执行目录
    os.makedirs("coding", exist_ok=True)
    return True


def create_llm_config():
    """创建LLM配置"""
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
    
    return llm_config


def create_agents(llm_config):
    """创建所需的代理"""
    # 代码执行器
    code_executor = LocalCommandLineCodeExecutor(work_dir="coding")
    
    # 用户代理
    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and 
                                    x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,
        code_execution_config={"executor": code_executor},
    )
    
    # 写作助手代理
    writing_assistant = AssistantAgent(
        name="writing_assistant",
        system_message="""你是一个专业的写作助手，负责创作高质量的博客文章。
你的任务是：
1. 根据用户要求生成引人入胜的博客文章
2. 如果收到批评意见，基于反馈改进之前的版本
3. 确保文章结构清晰、内容丰富、语言生动
4. 针对目标受众调整写作风格和深度""",
        llm_config=llm_config,
    )
    
    # 反思助手代理
    reflection_assistant = AssistantAgent(
        name="reflection_assistant",
        system_message="""你是一个专业的写作评论家，负责对文章进行深入分析和批评。
你的任务包括：
1. 分析文章的结构、内容和风格
2. 提供具体的改进建议
3. 指出文章的优点和不足
4. 建议如何增强文章的影响力和可读性
5. 确保建议具体、可操作且建设性""",
        llm_config=llm_config,
    )
    
    return user_proxy, writing_assistant, reflection_assistant


def reflection_message(recipient, messages, sender, config):
    """生成反思消息的函数"""
    print("🤔 正在进行反思分析...")
    last_message = recipient.chat_messages_for_summary(sender)[-1]['content']
    return f"""请对以下文章进行深入反思和批评分析：

{last_message}

请从以下角度进行分析：
1. 内容深度和准确性
2. 结构和组织
3. 语言风格和可读性
4. 目标受众的适配性
5. 具体改进建议

请提供详细、建设性的反馈。"""


def setup_reflection_mechanism(user_proxy, reflection_assistant, writing_assistant):
    """设置反思机制"""
    nested_chat_queue = [
        {
            "recipient": reflection_assistant,
            "message": reflection_message,
            "max_turns": 1,
        },
    ]
    
    user_proxy.register_nested_chats(
        nested_chat_queue,
        trigger=writing_assistant,
    )


def run_demo():
    """运行Reflexion agent演示"""
    print("🚀 启动AutoGen Reflexion Agent演示")
    print("=" * 50)
    
    # 设置环境
    if not setup_environment():
        return
    
    # 创建配置
    llm_config = create_llm_config()
    
    # 创建代理
    user_proxy, writing_assistant, reflection_assistant = create_agents(llm_config)
    
    # 设置反思机制
    setup_reflection_mechanism(user_proxy, reflection_assistant, writing_assistant)
    
    # 演示任务
    demo_tasks = [
        {
            "title": "AI技术发展博客文章",
            "prompt": """写一篇关于人工智能最新发展的博客文章。
要求：
- 文章应该引人入胜，适合普通读者理解
- 包含3个以上段落，但不超过1000字
- 涵盖AI在医疗、教育、娱乐等领域的应用
- 讨论AI的伦理考虑和未来展望"""
        },
        {
            "title": "可持续发展主题文章", 
            "prompt": """创作一篇关于可持续发展和绿色技术的博客文章。
要求：
- 面向企业决策者和环保意识强的读者
- 包含具体的数据和案例
- 讨论技术解决方案和商业模式创新
- 提供实用的行动建议"""
        }
    ]
    
    # 使用缓存来提高效率
    with Cache.disk(cache_seed=42) as cache:
        for i, task in enumerate(demo_tasks, 1):
            print(f"\n📝 演示 {i}: {task['title']}")
            print("-" * 30)
            
            try:
                user_proxy.initiate_chat(
                    writing_assistant,
                    message=task['prompt'],
                    max_turns=2,
                    cache=cache,
                )
                
                print(f"\n✅ 演示 {i} 完成")
                
            except Exception as e:
                print(f"❌ 演示 {i} 出错: {e}")
            
            if i < len(demo_tasks):
                input("\n按回车键继续下一个演示...")
    
    print("\n🎉 所有演示完成！")
    print("\n💡 这个演示展示了：")
    print("1. 如何使用AutoGen创建多代理系统")
    print("2. 如何实现Reflexion机制进行自我反思")
    print("3. 如何通过嵌套聊天实现代理间的协作")
    print("4. 如何使用缓存提高效率")


def main():
    """主函数"""
    print("AutoGen Reflexion Agent 演示程序")
    print("=" * 40)
    
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        print("请检查您的DashScope API密钥设置")


if __name__ == "__main__":
    main() 