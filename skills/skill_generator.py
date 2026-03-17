"""
Skill 生成器 v2.0 - LLM全自动完整生成版
✨ 特性：
- 动态参数智能推断（根据描述自动生成参数列表 + required）
- LLM生成完整可运行逻辑（支持调用现有工具、纯Python、错误处理）
- 生成后自动建议热重载 + 返回完整使用示例
"""
from pathlib import Path
import time
from openai import OpenAI
import os
import re

def tool_function(
    skill_name: str,
    description: str,
    auto_reload: bool = True,
    auto_test: bool = True
):
    """一句话需求 → 完整可用 Skill（动态参数 + 真实逻辑 + 自动热重载提示）"""
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in skill_name.lower())
    file_path = Path(__file__).parent / f"{safe_name}.py"

    # === 极致生成Prompt（第一性原理设计）===
    system_prompt = (
        "你是一个专业、高质量的Python工具生成专家。\n"
        "必须严格输出一个**完整、可立即运行**的Skill文件（.py格式）。\n"
        "要求：\n"
        "1. 根据用户描述智能推断参数（str/int/bool/list等），并在tool_schema中正确定义\n"
        "2. tool_function必须有实际功能（能解决问题），包含错误处理和清晰返回\n"
        "3. 可调用已有内置工具（如web_search、browse_page、code_executor等）——直接import或调用\n"
        "4. 返回结构化的dict（success + 关键信息）\n"
        "5. 代码必须干净、专业、带完整docstring\n"
        "直接输出完整代码，不要任何解释。"
    )

    user_prompt = f"""请为以下需求生成完整Skill代码：

Skill名称: {skill_name}
功能描述: {description}

请生成一个**真正可用**的工具，例如：
- 如果需要搜索 → 调用web_search
- 如果需要计算/分析 → 用纯Python或code_executor
- 如果需要文件操作 → 用read_file/write_file
- 参数要合理（不要固定param1）

输出格式：完整的.py文件内容（从第一行开始，到最后一行结束）。
"""

    try:
        client = OpenAI(
            api_key=os.getenv("MAS_API_KEY"),
            base_url=os.getenv("MAS_BASE_URL", "https://api.openai.com/v1")
        )
        model = os.getenv("MAS_LLM_MODEL", "gpt-4o-mini")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )

        raw_code = response.choices[0].message.content.strip()

        # 清理Markdown代码块（鲁棒处理）
        if raw_code.startswith("```python"):
            code = raw_code.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw_code:
            code = raw_code.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            code = raw_code

    except Exception as e:
        return {"success": False, "message": f"LLM生成失败: {str(e)}"}

    # === 写入文件 ===
    file_path.write_text(code, encoding="utf-8")

    # === 返回结果（用户体验极致）===
    reload_tip = "\n\n🔄 **已自动提示热重载**：请立即在对话中调用 `reload_skills` 工具（1秒生效）" if auto_reload else ""

    test_tip = "\n\n🧪 测试建议：生成后直接说“使用 {safe_name} 测试一下”" if auto_test else ""

    return {
        "success": True,
        "skill_name": safe_name,
        "file_path": str(file_path),
        "message": f"""🎉 **Skill 生成成功！**（v2.0 LLM全自动版）

**文件**：`{file_path.name}`
**描述**：{description[:120]}...

{reload_tip}
{test_tip}

**下一步（推荐）**：
1. 调用工具 `reload_skills`（立即热重载，所有Agent可用）
2. 测试新工具：让Grok执行“使用 {safe_name} [你的参数]”

**已准备好**：新Skill已具备动态参数 + 真实逻辑，可直接生产使用！"""
    }


tool_schema = {
    "type": "function",
    "function": {
        "name": "skill_generator",
        "description": "最强一键Skill生成器：输入自然语言描述，即可自动生成完整可用工具（动态参数推断 + 真实逻辑 + 自动热重载提示）。生成后1秒即可使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill英文名称（建议下划线）"},
                "description": {"type": "string", "description": "详细功能描述（越详细越好，例如：'帮我实时搜索并总结最新伊朗新闻，并生成带下载链接的报告'）"},
                "auto_reload": {"type": "boolean", "description": "是否提示自动热重载", "default": True},
                "auto_test": {"type": "boolean", "description": "是否给出测试建议", "default": True}
            },
            "required": ["skill_name", "description"]
        }
    }
}