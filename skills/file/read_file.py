"""
文件读取工具

功能：读取指定文件的内容（相对于脚本目录）

安全限制：
- 只允许读取脚本目录及其子目录的文件
"""

import os
from pathlib import Path


def tool_function(file_path: str):
    """
    读取文件内容（安全版本）

    Args:
        file_path: 文件路径（相对于脚本根目录或绝对路径）

    Returns:
        Dict: {"success": bool, "content": str, "length": int} 或错误信息
    """
    try:
        # ===== 获取脚本根目录 =====
        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()

        # ===== 路径规范化 =====
        target_path = Path(file_path)

        if not target_path.is_absolute():
            target_path = SCRIPT_ROOT / target_path

        target_path = target_path.resolve()

        # ===== 安全检查 =====
        try:
            relative_path = target_path.relative_to(SCRIPT_ROOT)
        except ValueError:
            return {
                "success": False,
                "error": f"安全错误：不允许读取脚本目录外的文件",
                "attempted_path": str(target_path),
                "script_root": str(SCRIPT_ROOT)
            }

        # ===== 文件存在性检查 =====
        if not target_path.exists():
            return {
                "success": False,
                "error": f"文件不存在: {relative_path}"
            }

        if not target_path.is_file():
            return {
                "success": False,
                "error": f"不是文件（可能是目录）: {relative_path}"
            }

        # ===== 读取文件 =====
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "success": True,
            "content": content,
            "length": len(content),
            "file_path": str(target_path),
            "relative_path": str(relative_path)
        }

    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"文件不是 UTF-8 文本格式或包含二进制内容"
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"权限错误：无法读取文件 {file_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"未知错误：{type(e).__name__} - {str(e)}"
        }


tool_schema = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "读取指定文件的内容（相对于脚本根目录）。\n"
            "安全限制: 只能读取脚本目录及其子目录的文件"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "要读取的文件路径（相对于脚本根目录）。\n"
                        "示例: 'data/input.txt' 或 './config.yaml'"
                    )
                }
            },
            "required": ["file_path"]
        }
    }
}