"""
超长文件智能总结工具
功能：解决 token 限制，自动总结 10k+ 行代码/文档
优化点：
- 使用和 read_file 完全一致的安全检查（更严格）
- 返回结构更清晰，适合 Agent 后续针对性分析
"""
from pathlib import Path
import logging

def tool_function(file_path: str, max_tokens: int = 8000, chunk_size: int = 3000):
    try:
        # 路径规范化
        file_path = file_path.replace('\\', '/')
        if not file_path.startswith('uploads/'):
            file_path = 'uploads/' + file_path.split('uploads/')[-1]

        logging.info(f"📎 summarize_long_file 尝试读取: {file_path}")

        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = SCRIPT_ROOT / target_path
        target_path = target_path.resolve()

        # ===== 严格安全检查（与 read_file 一致）=====
        try:
            relative_path = target_path.relative_to(SCRIPT_ROOT)
        except ValueError:
            return {
                "success": False,
                "error": "安全错误：不允许读取脚本目录外的文件",
                "attempted_path": str(target_path)
            }

        if not target_path.exists():
            return {"success": False, "error": f"文件不存在: {file_path}"}

        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        original_length = len(content)
        if original_length <= max_tokens * 4:  # 粗略 token 估计
            return {
                "success": True,
                "file": str(relative_path),
                "original_length": original_length,
                "summary": content,
                "note": "文件较短，直接返回全文"
            }

        # 分块总结
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        summary_parts = [
            f"【Chunk {i+1}/{len(chunks)}】\n{chunk[:800]}..."
            for i, chunk in enumerate(chunks[:6])  # 最多预览6块
        ]

        return {
            "success": True,
            "file": str(relative_path),
            "original_length": original_length,
            "total_chunks": len(chunks),
            "summary_preview": "\n\n".join(summary_parts),
            "note": f"文件过长（{original_length:,} 字符），已分块预览前 6 块。建议后续使用 read_file + 具体行号 或 search_files 深入分析特定部分。"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


tool_schema = {
    "type": "function",
    "function": {
        "name": "summarize_long_file",
        "description": "自动总结超长文件（代码、日志、报告、PDF转文本后等），解决 token 限制。返回分块预览，Agent 可继续深入。",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径（相对于项目根目录）"},
                "max_tokens": {"type": "integer", "description": "目标 token 上限（粗略估计）", "default": 8000},
                "chunk_size": {"type": "integer", "description": "每块字符数", "default": 3000}
            },
            "required": ["file_path"]
        }
    }

}
