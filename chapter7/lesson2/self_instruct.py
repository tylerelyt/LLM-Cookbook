import os
import json
from typing import List, Dict, Optional, Tuple

import orjson
import typer
from tqdm import tqdm


app = typer.Typer(add_completion=False, no_args_is_help=True)


DEFAULT_SEED_TASKS: List[str] = [
    "给出学习某项新技术的分步计划",
    "解释一个常见的数据结构并举例",
    "根据产品需求生成测试用例",
    "将一段英文文本总结为要点",
    "根据表结构编写 SQL 查询",
]


def _read_lines_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _read_seed_jsonl(path: str) -> List[str]:
    """兼容原 Self-Instruct 的 seed_tasks.jsonl，每行包含 instruction/instances。"""
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            instr = (
                obj.get("instruction")
                or obj.get("Instruction")
                or obj.get("task")
                or obj.get("Task")
            )
            if instr:
                items.append(str(instr).strip())
    return items


def _read_seed_json(path: str) -> List[str]:
    """若为 JSON 列表结构，提取其中的 instruction 字段。"""
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            instr = (
                obj.get("instruction")
                or obj.get("Instruction")
                or obj.get("task")
                or obj.get("Task")
            )
            if instr:
                items.append(str(instr).strip())
    return items


def load_seed_tasks(seed_file: Optional[str]) -> List[str]:
    """加载种子任务，优先使用原 Self-Instruct 的 seed_tasks.jsonl。
    支持 .jsonl / .json / .txt 三种格式。未找到时退回内置样例。
    """
    # 候选路径：显式参数，其次 lesson2/data 目录中的 seed_tasks 文件
    base_dir = os.path.dirname(__file__)
    candidates: List[str] = []
    if seed_file:
        candidates.append(seed_file)
    candidates.extend([
        os.path.join(base_dir, "data", "seed_tasks.jsonl"),
        os.path.join(base_dir, "data", "seed_tasks.json"),
        os.path.join(base_dir, "data", "seed_tasks.txt"),
    ])

    for path in candidates:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".jsonl":
                items = _read_seed_jsonl(path)
            elif ext == ".json":
                items = _read_seed_json(path)
            else:
                items = _read_lines_txt(path)
        except Exception:
            items = []
        if items:
            return items

    return DEFAULT_SEED_TASKS


def call_model(provider: str, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    else:
        import dashscope
        from http import HTTPStatus

        resp = dashscope.Generation.call(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output.text.strip()
        else:
            raise RuntimeError(f"DashScope error: {resp.code} {resp.message}")


# ------------------------------ Step 1. 指令生成 ------------------------------
INSTRUCTION_EXPAND_PROMPT = (
    """
你是数据合成专家。根据以下"种子任务"生成更多多样化的中文"指令任务"，用于指令微调数据集构建：

要求：
- 每条指令应简洁明确，避免含糊其辞
- 覆盖多领域（编程、数据分析、教育、法律、产品、写作等）
- 多样化指令类型（解释、生成、改写、归纳、评价、规划、推理等）
- 使用第二人称或直接式命令（如"请…"，"帮我…"）
- 仅输出指令，每行一条，不要编号或多余解释

种子任务：
{seed_list}

请直接给出生成的指令：
"""
).strip()


# ---------------------- Step 2. 分类任务识别（论文一致） ----------------------
CLASSIFICATION_DETECT_PROMPT = (
    """
判断下面的"指令"是否是一个"分类任务"（classification），例如：情感分类、话题分类、文本是否属于某类别等。

仅回复一个词：Yes 或 No。

指令：{instruction}
"""
).strip()


def is_classification_task(provider: str, model: str, instruction: str, max_tokens: int, temperature: float) -> bool:
    try:
        reply = call_model(provider, model, CLASSIFICATION_DETECT_PROMPT.format(instruction=instruction), max_tokens=max_tokens, temperature=temperature)
        reply = reply.strip().lower()
        return reply.startswith("y")
    except Exception:
        return False


# -------------------------- Step 3. 实例生成（两路） --------------------------
ANSWER_PROMPT_TEMPLATE_INPUT_FIRST = (
    """
请根据以下"指令"构造一个"输入-输出"示例用于指令微调。

要求：
- 先给出"Input:"一行，再给出"Output:"一行
- 用中文给出自然、合理且不含敏感内容的示例
- 输出应为最终答案，不要解释过程

指令：{instruction}
"""
).strip()

ANSWER_PROMPT_TEMPLATE_OUTPUT_FIRST = (
    """
请根据以下"分类指令"构造一个"输出-输入"示例（Output-first）。

要求：
- 先给出该分类任务的一个可能"Output:"（如类别标签），再给出与该输出一致的"Input:"文本
- 两行输出，格式严格为：
Output: <简短标签或结论>
Input: <与标签一致的文本或内容>
- 用中文给出自然、合理且不含敏感内容的示例

指令：{instruction}
"""
).strip()


def parse_io_from_text(text: str) -> Tuple[str, str]:
    """从模型返回文本中解析 Input/Output，两种排列均兼容。
    返回 (input_text, output_text)。解析失败返回空串。
    """
    input_text = ""
    output_text = ""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("input:"):
            input_text = line.split(":", 1)[1].strip()
        elif line.lower().startswith("output:"):
            output_text = line.split(":", 1)[1].strip()
    return input_text, output_text


# ------------------------------ Step 4. 过滤规则 ------------------------------
DISALLOWED_SUBSTRINGS = [
    "自杀", "爆炸物", "武器制作", "仇恨言论", "色情", "未成年人", "恐怖主义",
]


def passes_basic_filter(instruction: str, input_text: str, output_text: str) -> bool:
    if not instruction or len(instruction) < 4:
        return False
    if not output_text or len(output_text) < 2:
        return False
    if len(input_text) > 2000 or len(output_text) > 2000:
        return False
    text_all = f"{instruction}\n{input_text}\n{output_text}"
    for bad in DISALLOWED_SUBSTRINGS:
        if bad in text_all:
            return False
    # 粗略去除高度重复
    if input_text.strip() == output_text.strip():
        return False
    if instruction.strip() == input_text.strip() or instruction.strip() == output_text.strip():
        return False
    return True


def save_jsonl(records: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")


def deduplicate_strings(items: List[str]) -> List[str]:
    seen = set()
    unique_list = []
    for item in items:
        key = item.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_list.append(key)
    return unique_list


@app.command()
def main(
    provider: str = typer.Option("openai", help="openai 或 dashscope"),
    model: str = typer.Option("gpt-4o-mini", help="模型名称，例如 gpt-4o-mini 或 qwen-plus"),
    num_instructions: int = typer.Option(100, help="目标指令数量"),
    seed_file: Optional[str] = typer.Option(None, help="自定义种子任务文件路径"),
    output_path: str = typer.Option("data/self_instruct.jsonl", help="输出 JSONL 路径"),
    temperature: float = typer.Option(0.7, help="采样温度"),
    max_tokens: int = typer.Option(512, help="单次生成最大 tokens"),
    instances_per_instruction: int = typer.Option(1, help="每条指令生成的实例数量"),
    enable_filtering: bool = typer.Option(True, help="是否启用质量过滤"),
):
    """与 Self-Instruct 论文一致的四步流程：
    1) 指令生成 2) 分类任务识别 3) 实例生成（输出先/输入先） 4) 过滤与去重
    """

    # Step 0: 载入种子
    seed_tasks = load_seed_tasks(seed_file)
    seed_text = "\n".join(f"- {s}" for s in seed_tasks)

    # Step 1: 扩展指令集合
    expanded_instructions: List[str] = []
    while len(expanded_instructions) < num_instructions:
        prompt = INSTRUCTION_EXPAND_PROMPT.format(seed_list=seed_text)
        text = call_model(provider.lower(), model, prompt, max_tokens=max_tokens, temperature=temperature)
        candidates = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        expanded_instructions.extend(candidates)
        expanded_instructions = deduplicate_strings(expanded_instructions)
        typer.echo(f"当前已生成 {len(expanded_instructions)} 条指令...")

    expanded_instructions = expanded_instructions[:num_instructions]
    typer.echo(f"指令扩展完成，共 {len(expanded_instructions)} 条")

    # Step 2 & 3: 分类识别 + 实例生成
    records: List[Dict[str, str]] = []
    for instr in tqdm(expanded_instructions, desc="实例生成与过滤"):
        cls_flag = is_classification_task(provider.lower(), model, instr, max_tokens=64, temperature=0)
        for _ in range(max(1, instances_per_instruction)):
            try:
                if cls_flag:
                    raw = call_model(
                        provider.lower(),
                        model,
                        ANSWER_PROMPT_TEMPLATE_OUTPUT_FIRST.format(instruction=instr),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    raw = call_model(
                        provider.lower(),
                        model,
                        ANSWER_PROMPT_TEMPLATE_INPUT_FIRST.format(instruction=instr),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
            except Exception as e:
                raw = f"Output: (生成失败) {e}\nInput: "

            input_text, output_text = parse_io_from_text(raw)
            # 若模型未严格遵守顺序，尝试反向解析
            if not input_text and not output_text:
                # 兜底：将整段作为输出
                output_text = raw.strip()
            rec = {"instruction": instr.strip(), "input": input_text.strip(), "output": output_text.strip()}
            if not enable_filtering or passes_basic_filter(instr, rec["input"], rec["output"]):
                records.append(rec)

    # 去重（按 instruction+input 维度）
    seen_pairs = set()
    deduped: List[Dict[str, str]] = []
    for r in records:
        key = (r["instruction"].strip(), r["input"].strip())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append(r)

    save_jsonl(deduped, output_path)
    typer.echo(f"Self-Instruct 完成！保存 {len(deduped)} 条数据至: {output_path}")


if __name__ == "__main__":
    app()
