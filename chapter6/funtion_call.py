import requests
import json
import datetime
import os
import re
import random
import math
from collections import defaultdict, deque
from openai import OpenAI
import inspect

# 尝试导入 matplotlib 用于绘图
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib 未安装，将使用文本形式的数据可视化")

# 通义千问配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

if not DASHSCOPE_API_KEY:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

# 通义千问 OpenAI 兼容客户端
def get_qwen_client():
    """获取通义千问 OpenAI 兼容客户端"""
    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

# 通用 LLM chat 接口
def call_llm(messages, model=None, stream=False, temperature=0.2, extra_params=None):
    """
    通用 LLM chat/completions 接口，使用通义千问 OpenAI 兼容接口。
    messages: [{"role": "user"/"system"/"assistant", "content": "..."}]
    model: 模型名
    stream: 是否流式
    extra_params: 其他参数
    """
    if model is None:
        model = LLM_MODEL
    try:
        client = get_qwen_client()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream,
            **(extra_params or {})
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] LLM调用失败: {str(e)}")
        return f"[ERROR] LLM调用失败: {str(e)}"

        # ========== 1. 本地函数实现 ==========

def fetch_data(source, query):
    """数据获取工具"""
    try:
        # 模拟从不同数据源获取数据
        source_lower = source.lower() if isinstance(source, str) else str(source).lower()
        if source_lower == "csv":
            # 模拟CSV数据
            data = "姓名,年龄,城市,收入\n张三,25,北京,8000\n李四,30,上海,12000\n王五,28,广州,10000\n赵六,35,深圳,15000"
            result = f"从CSV获取的数据:\n{data}"
            return result
        elif source_lower == "api":
            # 模拟API数据
            data = "用户ID,购买金额,购买时间\n001,299,2024-01-15\n002,599,2024-01-16\n003,199,2024-01-17\n004,899,2024-01-18"
            result = f"从API获取的数据:\n{data}"
            return result
        else:
            return f"不支持的数据源: {source}"
    except Exception as e:
        return f"数据获取失败: {str(e)}"

def process_data(data, operation):
    """数据处理工具"""
    try:
        # 解析数据
        lines = data.split('\n')
        if len(lines) < 2:
            return "数据格式错误"
        
        header = lines[0]
        rows = lines[1:]
        
        operation_lower = operation.lower() if isinstance(operation, str) else str(operation).lower()
        if operation_lower == "filter":
            # 过滤操作：只保留收入大于10000的记录
            filtered_rows = []
            for row in rows:
                if ',' in row:
                    fields = row.split(',')
                    if len(fields) >= 4:
                        try:
                            income = int(fields[3])
                            if income > 10000:
                                filtered_rows.append(row)
                        except ValueError:
                            continue
            result = header + '\n' + '\n'.join(filtered_rows)
            return f"过滤后的数据:\n{result}"
        
        elif operation_lower == "sort" or operation_lower == "排序":
            # 排序操作：按收入排序
            data_rows = []
            for row in rows:
                if ',' in row:
                    fields = row.split(',')
                    if len(fields) >= 4:
                        try:
                            income = int(fields[3])
                            data_rows.append((income, row))
                        except ValueError:
                            continue
            
            sorted_rows = sorted(data_rows, key=lambda x: x[0], reverse=True)
            result = header + '\n' + '\n'.join([row[1] for row in sorted_rows])
            return f"排序后的数据:\n{result}"
        
        elif operation_lower == "aggregate":
            # 聚合操作：计算平均收入
            total_income = 0
            count = 0
            for row in rows:
                if ',' in row:
                    fields = row.split(',')
                    if len(fields) >= 4:
                        try:
                            income = int(fields[3])
                            total_income += income
                            count += 1
                        except ValueError:
                            continue
            
            if count > 0:
                avg_income = total_income / count
                return f"聚合统计结果:\n总记录数: {count}\n总收入: {total_income}\n平均收入: {avg_income:.2f}"
            else:
                return "没有有效数据可聚合"
        
        else:
            return f"不支持的操作: {operation}"
    except Exception as e:
        return f"数据处理失败: {str(e)}"

def create_table_and_chart(data, chart_type):
    """创建表格和图表工具"""
    try:
        # 解析数据
        lines = data.split('\n')
        if len(lines) < 2:
            return "数据格式错误"
        
        header = lines[0]
        rows = lines[1:]
        
        # 创建表格
        table = "数据表格:\n"
        table += "=" * 50 + "\n"
        table += header + "\n"
        table += "-" * 50 + "\n"
        for row in rows:
            if row.strip():
                table += row + "\n"
        table += "=" * 50 + "\n"
        
        # 创建图表
        chart_type_lower = chart_type.lower() if isinstance(chart_type, str) else str(chart_type).lower()
        if chart_type_lower == "bar" or chart_type_lower == "柱状图":
            # 柱状图
            chart = "柱状图:\n"
            names = []
            incomes = []
            
            for row in rows:
                if ',' in row:
                    fields = row.split(',')
                    if len(fields) >= 4:
                        try:
                            name = fields[0]
                            income = int(fields[3])
                            names.append(name)
                            incomes.append(income)
                        except ValueError:
                            continue
            
            if names and incomes:
                max_income = max(incomes)
                for i, (name, income) in enumerate(zip(names, incomes)):
                    bar_length = int((income / max_income) * 20)
                    bar = "█" * bar_length
                    chart += f"{name:4s} | {bar} {income:6d}\n"
        
        elif chart_type_lower == "pie" or chart_type_lower == "饼图":
            # 饼图（简化版）
            chart = "饼图分析:\n"
            total_income = 0
            income_data = []
            
            for row in rows:
                if ',' in row:
                    fields = row.split(',')
                    if len(fields) >= 4:
                        try:
                            name = fields[0]
                            income = int(fields[3])
                            total_income += income
                            income_data.append((name, income))
                        except ValueError:
                            continue
            
            if income_data and total_income > 0:
                for name, income in income_data:
                    percentage = (income / total_income) * 100
                    chart += f"{name}: {income} ({percentage:.1f}%)\n"
        
        else:
            chart = f"不支持图表类型: {chart_type}"
        
        result = table + "\n" + chart
        
        # 尝试在Jupyter环境中显示
        try:
            from IPython.display import display, HTML
            # 创建HTML表格
            html_table = "<table border='1' style='border-collapse: collapse;'>"
            html_table += "<tr><th>" + "</th><th>".join(header.split(',')) + "</th></tr>"
            for row in rows:
                if row.strip():
                    html_table += "<tr><td>" + "</td><td>".join(row.split(',')) + "</td></tr>"
            html_table += "</table>"
            
            display(HTML(html_table))
            
        except ImportError:
            pass
        
        return result
    except Exception as e:
        return f"表格和图表创建失败: {str(e)}"

# 函数注册表
AVAILABLE_FUNCTIONS = {
    "fetch_data": fetch_data,
    "process_data": process_data,
    "create_table_and_chart": create_table_and_chart
}

def call_function(name, args):
    """调用指定的函数"""
    try:
        if name in AVAILABLE_FUNCTIONS:
            # 过滤掉额外的参数，只保留函数需要的参数
            func = AVAILABLE_FUNCTIONS[name]
            sig = inspect.signature(func)
            filtered_args = {}
            
            for param_name, param_value in args.items():
                if param_name in sig.parameters:
                    filtered_args[param_name] = param_value
            
            result = func(**filtered_args)
            return result
        else:
            return f"[ERROR] 未知函数: {name}"
            
    except Exception as e:
        return f"[ERROR] 函数执行失败: {e}"

def generate_function_descriptions():
    """通过函数注入生成函数描述"""
    descriptions = []
    
    for func_name, func in AVAILABLE_FUNCTIONS.items():
        # 获取函数签名
        sig = inspect.signature(func)
        params = []
        
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                params.append(f"{param_name}={param.default}")
            else:
                params.append(param_name)
        
        # 获取函数文档
        doc = func.__doc__ or "无描述"
        
        # 生成函数描述
        param_str = ", ".join(params)
        description = f"- {func_name}({param_str}): {doc}"
        descriptions.append(description)
    
    return "\n".join(descriptions)



# ========== 2. 构造优化的 function call 提示词 ==========
def get_system_prompt():
    """动态生成系统提示词"""
    function_descriptions = generate_function_descriptions()
    
    return f"""
你是一个具备强大需求理解拆解能力和链式调用组合能力的AI助手。

**核心能力：**
1. **需求理解拆解**：能够将复杂任务自动拆解为多个子步骤
2. **链式调用组合**：能够智能组合多个函数，形成完整的解决方案

**可用函数：**
{function_descriptions}

**链式调用格式（每行一个JSON）：**
{{"function": "函数名1", "arguments": {{参数字典1}}}}
{{"function": "函数名2", "arguments": {{参数字典2}}}}
{{"function": "函数名3", "arguments": {{参数字典3}}}}

**参数格式要求：**
- source参数：使用小写，如 "csv", "api"
- operation参数：使用小写，如 "sort", "filter", "aggregate"
- chart_type参数：使用小写，如 "bar", "pie"
- data参数：对于链式调用，使用占位符如 "用户数据", "排序后的用户数据"

**重要要求：**
1. 对于复杂任务，必须拆解为多个步骤并链式调用
2. 每个JSON必须单独一行，不要在同一行输出多个JSON
3. 不要在JSON前后添加其他文字说明
4. 确保函数调用的顺序逻辑正确
5. 参数名称和值要严格按照函数定义
6. 如果不需要调用函数，直接自然语言回复

**示例链式调用思路：**
- 获取数据 → 处理数据 → 创建表格和图表
"""

# ========== 3. 与 LLM API 交互 ==========

def ask_llm(user_input, history=None):
    if history is None:
        history = []
    messages = [
        {"role": "system", "content": get_system_prompt()},
        *history,
        {"role": "user", "content": user_input}
    ]
    return call_llm(messages, model=LLM_MODEL)

# ========== 4. 主流程：解析 function call 并执行 ==========
def combo_demo():
    """真正的 LLM 驱动链式工具调用演示"""
    print("\n=== LLM 驱动链式工具调用演示 ===")
    print("展示大模型的需求理解拆解能力和链式调用组合能力")
    print("核心特点：上一个工具的输出作为下一个工具的输入")
    
    # 一个实用的链式调用例子
    task = "请从CSV数据源获取用户数据，对数据进行排序处理，然后创建表格和柱状图"
    
    print(f"\n[INFO] 开始执行任务: {task}")
    print("=" * 80)
    
    # ========== 步骤1：AI模型需求拆解 ==========
    print(f"\n[LLM] 阶段1: AI模型需求拆解")
    print("-" * 40)
    reply = ask_llm(task, [])
    print(f"[LLM] AI拆解结果: {reply}")
    
    # ========== 步骤2：解析并执行链式调用 ==========
    try:
        # 按行分割，每行尝试解析一个 JSON
        lines = reply.strip().split('\n')
        json_objects = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 尝试从行中提取 JSON
            json_start = line.find('{')
            json_end = line.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                try:
                    json_str = line[json_start:json_end]
                    data = json.loads(json_str)
                    if "function" in data and "arguments" in data:
                        json_objects.append(data)
                except json.JSONDecodeError:
                    continue
        
        if json_objects:
            print(f"\n[FUNCTION] 阶段2: Function工具链式执行")
            print("-" * 40)
            print(f"[FUNCTION] 检测到 {len(json_objects)} 个函数调用，开始链式执行")
            
            # 链式调用：上一个工具的输出作为下一个工具的输入
            previous_result = None
            chain_results = []
            
            for j, func_call in enumerate(json_objects, 1):
                func_name = func_call["function"]
                args = func_call["arguments"]
                
                print(f"\n[FUNCTION] 步骤 {j}: 调用函数 {func_name}")
                print(f"[FUNCTION] 参数: {args}")
                
                # 如果是链式调用，将上一个结果作为输入
                if previous_result is not None and j > 1:
                    print(f"[FUNCTION] 使用上一步结果作为输入")
                    
                    # 改进的占位符替换逻辑
                    if "data" in args:
                        # 检查参数中是否包含占位符
                        data_param = args["data"]
                        if isinstance(data_param, str):
                            # 检查各种占位符格式
                            placeholder_patterns = [
                                "result_of_", "fetched_data", "processed_data", 
                                "fetch_data_result", "process_data_result",
                                "result_of_previous_function", "previous_result",
                                "用户数据", "排序后的用户数据", "销售数据", "过滤后的销售数据"
                            ]
                            
                            is_placeholder = any(pattern in data_param for pattern in placeholder_patterns)
                            
                            if is_placeholder:
                                # 智能提取实际数据
                                if "从CSV获取的数据" in str(previous_result) or "从API获取的数据" in str(previous_result):
                                    # 提取数据部分，去掉标题行
                                    data_lines = str(previous_result).split('\n', 1)
                                    if len(data_lines) > 1:
                                        args["data"] = data_lines[1]
                                        print(f"[FUNCTION] 自动提取数据内容用于下一步处理")
                                elif "处理后的数据" in str(previous_result) or "排序后的数据" in str(previous_result):
                                    # 提取处理后的数据
                                    data_lines = str(previous_result).split('\n', 1)
                                    if len(data_lines) > 1:
                                        args["data"] = data_lines[1]
                                        print(f"[FUNCTION] 自动提取处理后的数据用于可视化")
                                else:
                                    # 直接使用上一步结果
                                    args["data"] = previous_result
                                    print(f"[FUNCTION] 使用上一步完整结果")
                
                # 执行函数调用
                print(f"[FUNCTION] 执行函数 {func_name}...")
                result = call_function(func_name, args)
                print(f"[FUNCTION] 函数执行完成，结果: {result}")
                
                # 保存结果用于下一步
                previous_result = result
                chain_results.append(f"步骤{j}({func_name}): {result}")
            
        else:
            print("[WARNING] 未检测到有效的函数调用")
            print("[INFO] LLM可能返回了自然语言回复，而不是函数调用")
            
    except Exception as e:
        print(f"[ERROR] 处理异常: {e}")
        import traceback
        print(f"[ERROR] 详细错误信息: {traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("[INFO] LLM 驱动链式工具调用演示完成")
    print("[INFO] 演示内容总结:")
    print("  - 理解复杂需求并自动拆解为多个步骤")
    print("  - 智能组合多个工具形成链式调用")
    print("  - 让上一个工具的输出作为下一个工具的输入")
    print("[INFO] 在实际应用中，这种能力让 AI 能够解决复杂的多步骤问题")

if __name__ == "__main__":
    combo_demo() 