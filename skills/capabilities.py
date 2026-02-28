# skills/capabilities.py
def tool_function():
    """返回当前系统完整能力描述（结构化）"""
    return {
        "main_tasks": "文件读写、代码执行、网页搜索、实时分析、并行子任务、定时提醒、主动推送消息",
        "loaded_skills": ["read_file", "write_file", "edit_file", "list_dir", "exec", "web_search", "web_fetch", "message", "spawn", "cron"],
        "missing_before": ["message", "spawn", "cron"],
        "now_available": "已全部补齐！",
        "how_to_use": "直接调用对应工具，或告诉我具体任务，我会自动选择"
    }

tool_schema = {
    "type": "function",
    "function": {
        "name": "capabilities",
        "description": "查询当前 Agent 已加载的全部技能和能力边界",
        "parameters": {"type": "object", "properties": {}}
    }
}