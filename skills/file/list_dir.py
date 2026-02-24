"""
目录列表工具

功能：列出指定目录下的文件和子目录
"""

from pathlib import Path


def tool_function(directory: str = "."):
    """
    列出目录内容（安全版本）

    Args:
        directory: 目录路径（相对于脚本根目录，默认为根目录）

    Returns:
        Dict: {"success": bool, "files": list, "directories": list}
    """
    try:
        # ===== 获取脚本根目录 =====
        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()

        # ===== 路径规范化 =====
        target_path = Path(directory)

        if not target_path.is_absolute():
            target_path = SCRIPT_ROOT / target_path

        target_path = target_path.resolve()

        # ===== 安全检查 =====
        try:
            relative_path = target_path.relative_to(SCRIPT_ROOT)
        except ValueError:
            return {
                "success": False,
                "error": f"安全错误：不允许访问脚本目录外的路径",
                "attempted_path": str(target_path)
            }

        # ===== 检查目录是否存在 =====
        if not target_path.exists():
            return {
                "success": False,
                "error": f"目录不存在: {relative_path}"
            }

        if not target_path.is_dir():
            return {
                "success": False,
                "error": f"不是目录: {relative_path}"
            }

        # ===== 列出内容 =====
        files = []
        directories = []

        for item in sorted(target_path.iterdir()):
            item_rel = item.relative_to(SCRIPT_ROOT)

            if item.is_dir():
                directories.append({
                    "name": item.name,
                    "path": str(item_rel),
                    "type": "directory"
                })
            else:
                size = item.stat().st_size
                files.append({
                    "name": item.name,
                    "path": str(item_rel),
                    "size": size,
                    "type": "file"
                })

        return {
            "success": True,
            "directory": str(relative_path or "."),
            "files": files,
            "directories": directories,
            "total_files": len(files),
            "total_directories": len(directories)
        }

    except PermissionError:
        return {
            "success": False,
            "error": f"权限错误：无法访问目录 {directory}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"未知错误：{type(e).__name__} - {str(e)}"
        }


tool_schema = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": (
            "列出指定目录下的文件和子目录（相对于脚本根目录）。\n"
            "安全限制: 只能访问脚本目录及其子目录"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "要列出的目录路径（相对于脚本根目录）。\n"
                        "示例: 'reports' 或 './data'\n"
                        "默认值: '.' (脚本根目录)"
                    ),
                    "default": "."
                }
            }
        }
    }
}