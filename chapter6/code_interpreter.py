import os
import autogen
from autogen import AssistantAgent, UserProxyAgent

# LLM 配置
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

llm_config = {
    "config_list": [{
        "model": LLM_MODEL,
        "api_key": DASHSCOPE_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }],
    "temperature": 0.2,
}

# 创建 Code Interpreter 智能体
assistant = AssistantAgent(
    name="CodeInterpreter",
    llm_config=llm_config,
    system_message="""你是一个专业的数据分析师和代码解释器。你能够编写Python代码来分析数据、生成可视化图表，并提供详细的解释。

你的能力包括：
1. 数据生成和统计分析
2. 数据可视化（直方图、箱线图、Q-Q图等）
3. 统计检验（正态性检验、描述性统计等）
4. 生成综合的分析报告

请确保：
- 代码能够正确执行
- 生成有意义的输出
- 提供详细的解释和分析
- 处理可能的错误情况
- 生成高质量的可视化图表""",
    code_execution_config={
        "work_dir": "./code_interpreter_workspace", 
        "use_docker": False
    }
)

# 创建用户代理
user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",  # 自动对话，无需人工输入
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "./code_interpreter_workspace", 
        "use_docker": False
    }
)

task = """
我需要分析一组数据的统计特征并生成可视化报告。

具体需求：
1. 生成1000个符合正态分布的随机数据点，均值为50，标准差为10
2. 计算这组数据的均值、标准差、中位数、最大值、最小值等统计指标
3. 创建一个直方图来展示数据分布，包含正态分布曲线
4. 将图表保存为PNG文件，并告诉我统计结果和文件名
5. 提供对数据分布的分析和解释
6. 额外添加Q-Q图来检验正态性
7. 计算偏度和峰度
8. 生成一个综合的分析报告

请帮我完成这个数据分析任务，确保代码能够正确执行，并提供详细的解释。
"""

if __name__ == "__main__":
    print("=== Code Interpreter 数据分析演示 ===")
    print("开始执行数据分析任务...")
    print("=" * 50)
    
    try:
        user.initiate_chat(
            assistant,
            message=task,
            max_turns=10
        )
        print("\n" + "=" * 50)
        print("数据分析任务完成！")
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}") 