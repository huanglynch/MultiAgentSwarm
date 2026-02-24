"""
CSV 读取工具
功能：读取和解析 CSV 文件
"""

def tool_function(file_path: str, max_rows: int = 100):
    """读取 CSV 文件"""
    try:
        import csv

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                data.append(dict(row))

        return {
            "success": True,
            "file_path": file_path,
            "rows": len(data),
            "columns": list(data[0].keys()) if data else [],
            "data": data
        }
    except FileNotFoundError:
        return {"success": False, "error": f"文件不存在: {file_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "csv_reader",
        "description": "读取 CSV 文件并解析为结构化数据",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "CSV 文件路径"
                },
                "max_rows": {
                    "type": "integer",
                    "description": "最大读取行数",
                    "default": 100
                }
            },
            "required": ["file_path"]
        }
    }
}
