"""
JSON 解析工具
功能：读取和解析 JSON 文件或字符串
"""

import json

def tool_function(source: str, source_type: str = "file"):
    """解析 JSON 数据"""
    try:
        if source_type == "file":
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            source_info = f"文件: {source}"
        elif source_type == "string":
            data = json.loads(source)
            source_info = "JSON 字符串"
        else:
            return {"success": False, "error": "source_type 必须是 'file' 或 'string'"}

        return {
            "success": True,
            "source": source_info,
            "data": data,
            "type": type(data).__name__
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON 格式错误: {e}"}
    except FileNotFoundError:
        return {"success": False, "error": f"文件不存在: {source}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "json_parser",
        "description": "解析 JSON 文件或字符串",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "文件路径或 JSON 字符串"
                },
                "source_type": {
                    "type": "string",
                    "description": "数据源类型",
                    "enum": ["file", "string"],
                    "default": "file"
                }
            },
            "required": ["source"]
        }
    }
}
