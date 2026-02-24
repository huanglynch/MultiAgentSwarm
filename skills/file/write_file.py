"""
文件写入工具
功能：将内容写入指定文件
"""

import os

def tool_function(file_path: str, content: str, mode: str = "w"):
    """写入文件内容"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        with open(file_path, mode, encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True, 
            "file_path": file_path, 
            "bytes_written": len(content.encode("utf-8"))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "将内容写入指定文件，自动创建目录",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "目标文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容"
                },
                "mode": {
                    "type": "string",
                    "description": "写入模式：w(覆盖) 或 a(追加)",
                    "enum": ["w", "a"],
                    "default": "w"
                }
            },
            "required": ["file_path", "content"]
        }
    }
}
