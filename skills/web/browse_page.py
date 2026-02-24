"""
网页浏览工具
功能：获取网页内容并提取文本
"""

def tool_function(url: str, max_length: int = 5000):
    """获取网页内容"""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # 限制长度
        if len(text) > max_length:
            text = text[:max_length] + "\n... (内容已截断)"

        return {
            "success": True,
            "url": url,
            "title": soup.title.string if soup.title else "无标题",
            "content": text,
            "length": len(text)
        }
    except ImportError:
        return {
            "success": False,
            "error": "请安装依赖: pip install requests beautifulsoup4"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

tool_schema = {
    "type": "function",
    "function": {
        "name": "browse_page",
        "description": "获取网页内容并提取纯文本",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "目标网页 URL"
                },
                "max_length": {
                    "type": "integer",
                    "description": "最大返回文本长度",
                    "default": 5000
                }
            },
            "required": ["url"]
        }
    }
}
