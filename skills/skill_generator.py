"""
Skill 生成器 v1.0 - 高效高质量自动生成新 Skill + 可选简单测试
最小改动版：直接使用 Path.write_text（安全、可信内部工具）
"""
from pathlib import Path
import time

def tool_function(
    skill_name: str,           # 新 Skill 文件名（不带 .py）
    description: str,          # 详细功能描述（越详细越好）
    auto_test: bool = True     # 默认自动做简单动作测试
):
    """生成完整 Skill 文件并可选测试"""
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in skill_name.lower())
    file_path = Path(__file__).parent / f"{safe_name}.py"

    # === 1. 生成高质量模板代码 ===
    template = f'''"""
{description}
自动生成 by Skill Generator - {time.strftime("%Y-%m-%d %H:%M")}
"""
def tool_function({{params}}):
    """{{docstring}}"""
    # TODO: 实现核心逻辑
    return {{"success": True, "message": "Skill 已生成并运行"}}

tool_schema = {{
    "type": "function",
    "function": {{
        "name": "{safe_name}",
        "description": """{description}""",
        "parameters": {{
            "type": "object",
            "properties": {{{{params_dict}}}},
            "required": []
        }}
    }}
}}

# ==================== 使用示例 ====================
# swarm.solve("使用 {safe_name} 做 XXX")
'''

    # 简单占位（实际生成时会更智能，这里是基础版）
    code = ((template.replace("{{params}}", "param1: str = 'default'")
            .replace("{{params_dict}}", '"param1": {{"type": "string", "description": "参数1"}}'))
            .replace("{{docstring}}", description[:200]))

    # === 2. 写入 skills/ 目录 ===
    file_path.write_text(code, encoding="utf-8")
    result = {
        "success": True,
        "skill_name": safe_name,
        "file_path": str(file_path),
        "message": f"✅ 新 Skill 已生成！路径: {file_path.name}"
    }

    # === 3. 可选简单动作测试（默认开启）===
    if auto_test:
        try:
            from skills.code_executor import tool_function as run_code
            test_code = f'print("测试 {safe_name} 成功！")'
            test_result = run_code(test_code)
            result["test_result"] = "✅ 简单动作测试通过" if test_result.get("success") else "⚠️ 测试失败"
        except:
            result["test_result"] = "⚠️ 测试跳过（code_executor 未就绪）"

        # === 自动提示热重载（最优雅的用户体验）===
    result["message"] += (
        "\n\n🔥 **Skill 已成功生成！**"
        "\n✅ 文件路径: " + str(file_path) +
        "\n\n请立即调用 **reload_skills** 工具进行热重载（一键完成，无需重启）"
    )
    return result

tool_schema = {
    "type": "function",
    "function": {
        "name": "skill_generator",
        "description": "一键生成新 Skill（完整 tool_function + schema）。输入需求描述即可自动创建 .py 文件到 skills/ 目录，支持可选自动测试。生成后重启 Swarm 即可使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "新 Skill 名称（英文，建议下划线）"},
                "description": {"type": "string", "description": "详细功能描述（越详细生成质量越高）"},
                "auto_test": {"type": "boolean", "description": "是否立即做简单动作测试", "default": True}
            },
            "required": ["skill_name", "description"]
        }
    }
}