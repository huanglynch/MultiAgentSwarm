"""
网页搜索工具
功能：使用 DuckDuckGo 搜索实时信息
"""

def tool_function(query: str, num_results: int = 5):
    """实时网页搜索"""
    try:
        from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=num_results)
        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            }
            for r in results
        ]

        return {
            "success": True,
            "query": query,
            "results": formatted,
            "count": len(formatted)
        }
    except ImportError:
        return {
            "success": False,
            "error": "请安装依赖: pip install duckduckgo-search"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "使用 DuckDuckGo 搜索最新网络信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量 (1-10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}
