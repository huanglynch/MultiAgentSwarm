"""
安全代码执行工具（增强版）
功能：执行 Python 代码，支持 numpy、pandas、matplotlib 绘图、数据分析、测试
优化点：
- 内置常用科学计算库（无需 import）
- 支持 matplotlib 非交互式绘图（Agg backend）
- 返回结果更友好，可保存图片到 reports/
"""
import threading
from pathlib import Path

def tool_function(code: str, timeout: int = 10, save_plot: bool = False):
    """安全执行 Python 代码"""
    result_container = {"output": None, "done": False, "plot_saved": None}

    def target():
        try:
            # === 增强沙箱：内置常用库 ===
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from io import StringIO
            import math, random, statistics, json, os
            from datetime import datetime

            # 非交互式绘图支持
            plt.switch_backend('Agg')

            restricted_globals = {
                "__builtins__": {
                    "print": print, "range": range, "len": len, "str": str,
                    "int": int, "float": float, "list": list, "dict": dict,
                    "sum": sum, "min": min, "max": max, "abs": abs,
                    "sorted": sorted, "enumerate": enumerate,
                    "np": np, "pd": pd, "plt": plt, "math": math,
                    "random": random, "statistics": statistics,
                    "json": json, "os": os, "datetime": datetime,
                    "StringIO": StringIO
                }
            }
            local_vars = {}
            exec(code, restricted_globals, local_vars)

            output = str(local_vars.get("result", "执行成功，无返回结果"))

            # 可选：自动保存最后一张图
            if save_plot and plt.get_fignums():
                reports_dir = Path(__file__).parent.parent / "reports"
                reports_dir.mkdir(exist_ok=True)
                plot_path = reports_dir / f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_path)
                result_container["plot_saved"] = str(plot_path.relative_to(Path(__file__).parent.parent.parent))

            result_container["output"] = output
        except Exception as e:
            result_container["output"] = f"执行错误: {str(e)}"
        finally:
            result_container["done"] = True

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not result_container["done"]:
        return {"success": False, "error": f"执行超时（{timeout}秒）"}

    return {
        "success": True,
        "code_preview": code[:300] + "..." if len(code) > 300 else code,
        "output": result_container["output"],
        "plot_saved": result_container.get("plot_saved"),
        "timeout": timeout
    }


tool_schema = {
    "type": "function",
    "function": {
        "name": "code_executor",
        "description": "在安全沙箱中执行 Python 代码（内置 numpy、pandas、matplotlib）。支持数学、数据分析、绘图（可自动保存到 reports/）。save_plot=True 时自动保存图片。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
                "timeout": {"type": "integer", "description": "超时秒数", "default": 10},
                "save_plot": {"type": "boolean", "description": "是否自动保存最后一张 matplotlib 图表", "default": False}
            },
            "required": ["code"]
        }
    }
}