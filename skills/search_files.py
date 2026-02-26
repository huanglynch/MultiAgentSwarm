"""
项目内全文搜索工具
功能：瞬间找到函数、变量、类、注释等，取代 list_dir + 多次 read_file
"""
from pathlib import Path
import re

def tool_function(keyword: str, file_extensions: list = None, max_results: int = 20, context_lines: int = 5):
    """
    在项目内搜索关键词
    """
    try:
        SCRIPT_ROOT = Path(__file__).parent.parent.parent.absolute()
        if file_extensions is None:
            file_extensions = [".py", ".md", ".yaml", ".yml", ".json", ".txt", ".html", ".js", ".css"]

        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for file_path in SCRIPT_ROOT.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines):
                        if pattern.search(line):
                            # 提取上下文
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            context = "".join(lines[start:end]).strip()

                            results.append({
                                "file": str(file_path.relative_to(SCRIPT_ROOT)),
                                "line": i + 1,
                                "match": line.strip(),
                                "context": context
                            })

                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
                except:
                    continue  # 跳过无法读取的文件（二进制等）

        return {
            "success": True,
            "keyword": keyword,
            "matches": results,
            "total": len(results),
            "searched_files": "全部项目文件（受扩展名限制）"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


tool_schema = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "在整个项目内全文搜索关键词（函数名、变量、注释、类名等）。速度极快，强烈建议作为文件探索的第二步（第一步是 get_project_structure）。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "要搜索的关键词"},
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "文件扩展名列表，例如 ['.py','.md']，默认搜索常见文本文件",
                    "default": [".py",".md",".yaml",".yml",".json",".txt"]
                },
                "max_results": {"type": "integer", "description": "最多返回结果数", "default": 20},
                "context_lines": {"type": "integer", "description": "每处匹配显示的上下文行数", "default": 5}
            },
            "required": ["keyword"]
        }
    }
}