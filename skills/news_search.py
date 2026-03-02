#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill: news_search
支持中文、日文、英文查询的 Google News RSS 搜索（已彻底修复 URL control characters 错误）
"""
from urllib.parse import urlencode
import requests
from typing import Dict, Any


def tool_function(
        query: str = "Iran",
        hours: int = 24,
        lang: str = "zh"  # ← 新增：支持中/日/英
) -> str:
    """
    Google News RSS 实时搜索（支持中日英）

    Args:
        query: 搜索关键词（支持中文、日文、英文）
        hours: 最近多少小时（默认24h）
        lang: 语言（zh / ja / en）
    """
    # ====================== 语言映射（中日英） ======================
    lang_map = {
        "zh": {"hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"},
        "ja": {"hl": "ja-JP", "gl": "JP", "ceid": "JP:ja"},
        "en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    }
    cfg = lang_map.get(lang.lower(), lang_map["zh"])  # 默认中文

    # ====================== 方法二：urlencode（彻底解决 control characters） ======================
    params = {
        "q": f"{query} when:{hours}h",  # 支持任意语言（包括空格、中文、日文）
        "hl": cfg["hl"],
        "gl": cfg["gl"],
        "ceid": cfg["ceid"]
    }

    base_url = "https://news.google.com/rss/search"
    url = base_url + "?" + urlencode(params)  # 自动处理所有特殊字符和多语言

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        # 返回前 10 条新闻（可根据需要调整）
        return f"✅ 新闻搜索成功（{lang.upper()}）\n查询: {query} (最近{hours}小时)\n\n{resp.text[:8000]}"

    except Exception as e:
        return f"⚠️ 新闻搜索失败: {str(e)}\nURL: {url}"


# ====================== Tool Schema（必须保留） ======================
tool_schema = {
    "type": "function",
    "function": {
        "name": "news_search",
        "description": "实时搜索 Google News（支持中文、日文、英文查询，自动处理特殊字符）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词（支持中文、日文、英文）"
                },
                "hours": {
                    "type": "integer",
                    "description": "最近多少小时（默认24）",
                    "default": 24
                },
                "lang": {
                    "type": "string",
                    "enum": ["zh", "ja", "en"],
                    "description": "语言：zh=中文, ja=日文, en=英文（默认zh）",
                    "default": "zh"
                }
            },
            "required": ["query"]
        }
    }
}