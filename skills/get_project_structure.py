#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

def tool_function(max_depth: int = 3, max_files_per_dir: int = 20):
    """获取项目完整结构树 - 永远作为文件探索的第一步"""
    root = Path(".").resolve()
    result = [f"📁 项目根目录: {root.name} ({root})", "="*60]

    def walk(path: Path, depth: int):
        if depth > max_depth:
            return
        indent = "  " * depth
        try:
            # 只显示目录
            for item in sorted(path.iterdir()):
                rel = item.relative_to(root)
                if item.is_dir():
                    result.append(f"{indent}📁 {rel}/")
                    walk(item, depth + 1)
                else:
                    # 文件只显示前N个，避免输出爆炸
                    if len([x for x in result if str(rel.parent) in x]) < max_files_per_dir:
                        size = f" ({item.stat().st_size//1024}KB)" if item.stat().st_size > 1024 else ""
                        result.append(f"{indent}  📄 {rel.name}{size}")
        except PermissionError:
            result.append(f"{indent}🚫 权限拒绝: {path}")

    walk(root, 0)
    return "\n".join(result[:300])  # 严格限长，防止token爆炸

tool_schema = {
    "type": "function",
    "function": {
        "name": "get_project_structure",
        "description": "获取项目完整目录结构树 - 永远作为文件探索的第一步！任何需要了解项目布局的任务都必须先调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "max_depth": {
                    "type": "integer",
                    "description": "最大遍历深度（建议 2-4）",
                    "default": 3
                },
                "max_files_per_dir": {
                    "type": "integer",
                    "description": "每个目录最多显示的文件数量（防止输出过长）",
                    "default": 15
                }
            }
        }
    }
}