# skills/message.py
import requests
import json

def tool_function(webhook_url: str, content: str, title: str = "MultiAgentSwarm 通知"):
    """主动向 Feishu/Webhook 发送消息"""
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [[{"tag": "text", "text": content}]]
                }
            }
        }
    }
    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
        return {"success": True, "status": r.status_code, "response": r.text[:200]}
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "message",
        "description": "主动向 Feishu/Webhook 发送消息或通知（支持标题）",
        "parameters": {
            "type": "object",
            "properties": {
                "webhook_url": {"type": "string", "description": "Feishu 等 Webhook 地址"},
                "content": {"type": "string", "description": "消息正文"},
                "title": {"type": "string", "description": "消息标题（可选）"}
            },
            "required": ["webhook_url", "content"]
        }
    }
}