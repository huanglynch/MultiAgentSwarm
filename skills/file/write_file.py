"""
文件写入工具（已强制 uploads/ 输出版）
功能：将内容写入指定文件，**所有生成文件自动放入 uploads/**（支持子目录）
安全限制：只能写入脚本目录及其子目录
"""

import re
import time
from pathlib import Path


def tool_function(file_path: str, content: str, mode: str = "w"):
    """
    写入文件内容（安全版本 + 强制 uploads/ 输出）
    """
    try:
        # ================ 【核心：强制所有生成文件进入 uploads/】================
        original_path = file_path
        # 清理文件名（防止中文、空格导致下载失败）
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', Path(file_path).name)
        if len(safe_name) < 3:
            safe_name = f"generated_file_{int(time.time())}"

        # 强制放入 uploads/（无论用户输入什么路径）
        if not str(file_path).lower().startswith('uploads'):
            file_path = f"uploads/{safe_name}"
            print(f"📤 write_file 已强制规范化: '{original_path}' → '{file_path}'")
        else:
            file_path = str(Path(file_path))  # 用户已指定 uploads/，保留结构

        # ====================== 原有安全逻辑（完全不动） ======================
        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()

        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = SCRIPT_ROOT / target_path
        target_path = target_path.resolve()

        try:
            relative_path = target_path.relative_to(SCRIPT_ROOT)
        except ValueError:
            return {"success": False, "error": "安全错误：不允许写入脚本目录外的文件"}

        ALLOWED_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.yaml', '.yml', '.log', '.html', '.xml', '.py', '.sh', '.sql', '.rst'}
        if target_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return {"success": False, "error": f"不允许的文件类型: {target_path.suffix}"}

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if mode not in ("w", "a"):
            return {"success": False, "error": f"无效的写入模式: {mode}"}

        with open(target_path, mode, encoding="utf-8") as f:
            f.write(content)

        file_size = target_path.stat().st_size
        return {
            "success": True,
            "file_path": str(target_path),
            "relative_path": str(relative_path),
            "bytes_written": len(content.encode("utf-8")),
            "file_size": file_size,
            "mode": "覆盖写入" if mode == "w" else "追加写入",
            "download_url": f"/uploads/{Path(file_path).name}"   # 供前端/飞书使用
        }
    except Exception as e:
        return {"success": False, "error": f"未知错误：{type(e).__name__} - {str(e)}"}


tool_schema = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "将内容写入文件。**所有生成文件会自动放入 uploads/ 目录**（支持子目录），返回可直接点击的下载链接。支持 .txt .md .json .csv 等文本格式。",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "目标文件名（可带路径，如 reports/report.md）"},
                "content": {"type": "string", "description": "要写入的文件内容"},
                "mode": {"type": "string", "enum": ["w", "a"], "default": "w"}
            },
            "required": ["file_path", "content"]
        }
    }
}