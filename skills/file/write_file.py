"""
文件写入工具 v2.1（中文/日文友好 + 生产强化版）
- 支持中文/日文文件名（只过滤危险字符）
- 启动时强制创建 uploads/
- 详细错误 + 日志 + 友好返回
"""
import re
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def tool_function(file_path: str, content: str, mode: str = "w"):
    try:
        if not content or not file_path:
            return {"success": False, "error": "file_path 和 content 不能为空"}

        original_path = file_path.strip()
        original_name = Path(original_path).name

        # === 1. 温和清洗（保留中文/日文，只去 Windows 非法字符）===
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', original_name)
        if len(safe_name) < 2 or safe_name in ('.', '..'):
            safe_name = f"report_{int(time.time())}.md"
        # 自动加 .md
        if not safe_name.lower().endswith(('.md', '.txt', '.json')):
            safe_name += '.md'

        # 限制文件名长度（防止超长路径）
        safe_name = safe_name[:200]

        # === 2. 强制 uploads/ + 日期前缀（防重名）===
        today = datetime.now().strftime("%Y%m%d")
        if not str(original_path).lower().startswith("uploads"):
            file_path = f"uploads/{today}_{safe_name}"
        else:
            parent = Path(original_path).parent
            file_path = str(parent / f"{today}_{safe_name}" if str(parent) != "." else f"uploads/{today}_{safe_name}")

        # === 3. 安全解析 + 确保目录存在 ===
        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()
        uploads_dir = SCRIPT_ROOT / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        target_path = (SCRIPT_ROOT / file_path).resolve()

        try:
            relative_path = target_path.relative_to(SCRIPT_ROOT)
        except ValueError:
            return {"success": False, "error": "安全限制：只能写入项目目录内", "attempted": str(target_path)}

        ALLOWED_EXT = {'.txt', '.md', '.json', '.csv', '.yaml', '.yml', '.html', '.log'}
        if target_path.suffix.lower() not in ALLOWED_EXT:
            return {"success": False, "error": f"不支持的文件类型: {target_path.suffix}（推荐 .md）"}

        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, mode, encoding="utf-8") as f:
            f.write(content)

        # 【修复】使用相对路径生成正确的 URL
        download_url = f"/{relative_path.as_posix()}"
        logger.info(f"✅ write_file 成功: {relative_path} ({len(content):,} 字符)")

        return {
            "success": True,
            "message": f"✅ 文件写入成功！\n路径: {relative_path}\n下载: {download_url}",
            "file_path": str(target_path),
            "relative_path": str(relative_path),
            "download_url": download_url,
            "size": len(content)
        }

    except PermissionError:
        logger.error("write_file 权限错误")
        return {"success": False, "error": "权限错误：无法写入 uploads/ 目录。请检查文件夹权限或以管理员运行"}
    except Exception as e:
        logger.error(f"write_file 异常: {str(e)}", exc_info=True)
        return {"success": False, "error": f"写入失败: {type(e).__name__} - {str(e)[:200]}"}


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