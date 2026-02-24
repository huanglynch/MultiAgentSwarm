"""
文件写入工具

功能：将内容写入指定文件（相对于脚本目录）

安全限制：
- 只允许写入脚本目录及其子目录
- 自动创建不存在的目录
- 支持覆盖 (w) 和追加 (a) 模式
"""

import os
from pathlib import Path


def tool_function(file_path: str, content: str, mode: str = "w"):
    """
    写入文件内容（安全版本）

    Args:
        file_path: 文件路径（相对于脚本根目录或绝对路径）
        content: 要写入的内容
        mode: 写入模式，"w" 覆盖，"a" 追加

    Returns:
        Dict: {"success": bool, "file_path": str, "bytes_written": int} 或错误信息
    """
    try:
        # ===== 获取脚本根目录（skills/ 的父目录） =====
        # 当前文件: D:/huang/data/working/python/openagent/skills/file/write_file.py
        # 脚本根目录: D:/huang/data/working/python/openagent/
        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()

        # ===== 路径规范化 =====
        target_path = Path(file_path)

        # 如果是相对路径，转换为相对于脚本根目录的绝对路径
        if not target_path.is_absolute():
            target_path = SCRIPT_ROOT / target_path

        # 解析路径（处理 .. 和 . 等）
        target_path = target_path.resolve()

        # ===== 安全检查：确保文件在脚本根目录内 =====
        try:
            relative_path = target_path.relative_to(SCRIPT_ROOT)
        except ValueError:
            return {
                "success": False,
                "error": f"安全错误：不允许写入脚本目录外的文件",
                "attempted_path": str(target_path),
                "script_root": str(SCRIPT_ROOT)
            }

        # ===== 文件扩展名白名单检查（可选，更严格） =====
        ALLOWED_EXTENSIONS = {
            '.txt', '.md', '.json', '.csv', '.yaml', '.yml',
            '.log', '.html', '.xml', '.py', '.sh', '.sql', '.rst'
        }

        if target_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return {
                "success": False,
                "error": f"不允许的文件类型: {target_path.suffix}",
                "allowed_types": list(ALLOWED_EXTENSIONS)
            }

        # ===== 创建目录 =====
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # ===== 写入模式验证 =====
        if mode not in ("w", "a"):
            return {
                "success": False,
                "error": f"无效的写入模式: {mode}，只支持 'w' 或 'a'"
            }

        # ===== 写入文件 =====
        with open(target_path, mode, encoding="utf-8") as f:
            f.write(content)

        # ===== 获取文件大小 =====
        file_size = target_path.stat().st_size

        return {
            "success": True,
            "file_path": str(target_path),
            "relative_path": str(relative_path),
            "bytes_written": len(content.encode("utf-8")),
            "file_size": file_size,
            "mode": "覆盖写入" if mode == "w" else "追加写入"
        }

    except PermissionError:
        return {
            "success": False,
            "error": f"权限错误：无法写入文件 {file_path}"
        }
    except OSError as e:
        return {
            "success": False,
            "error": f"系统错误：{e}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"未知错误：{type(e).__name__} - {str(e)}"
        }


tool_schema = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "将内容写入指定文件（相对于脚本根目录）。\n"
            "支持的文件类型: .txt, .md, .json, .csv, .yaml, .log, .html, .xml, .py 等\n"
            "安全限制: 只能写入脚本目录及其子目录"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "目标文件路径（相对于脚本根目录）。\n"
                        "示例: 'reports/ai_report.md' 或 './output/data.json'"
                    )
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文件内容"
                },
                "mode": {
                    "type": "string",
                    "description": "写入模式：'w' 覆盖写入，'a' 追加写入",
                    "enum": ["w", "a"],
                    "default": "w"
                }
            },
            "required": ["file_path", "content"]
        }
    }
}