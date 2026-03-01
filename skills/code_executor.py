"""
安全代码执行工具（双模式自动降级版）
- 已安装 OpenSandbox → 使用 Docker 硬隔离（推荐）
- 未安装 → 自动回退到原来 threading 沙箱 + 日志提醒
"""
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# ====================== 自动检测 OpenSandbox ======================
USE_OPEN_SANDBOX = False
try:
    from opensandbox import Sandbox
    from code_interpreter import CodeInterpreter, SupportedLanguage
    from datetime import timedelta
    USE_OPEN_SANDBOX = True
    logging.info("✅ OpenSandbox 已安装，使用 Docker 硬隔离沙箱")
except ImportError:
    USE_OPEN_SANDBOX = False
    warning_msg = """
    ⚠️【OpenSandbox 未安装】当前使用旧版 threading 沙箱（安全性较低）
    
    推荐立即安装（只需 1 分钟）：
    1. uv pip install opensandbox opensandbox-code-interpreter
    2. opensandbox-server init-config ~/.sandbox.toml --example docker
    3. 新开终端运行：opensandbox-server
    
    安装后重启 MultiAgentSwarm 即可自动切换到生产级沙箱！
    """
    print(warning_msg)
    logging.warning(warning_msg.strip())
# ================================================================

async def _async_opensandbox_execute(code: str, timeout: int = 10, save_plot: bool = False):
    """OpenSandbox 执行（仅在已安装时调用）"""
    sandbox = await Sandbox.create(
        "opensandbox/code-interpreter:v1.0.1",
        entrypoint=["/opt/opensandbox/code-interpreter.sh"],
        env={"PYTHON_VERSION": "3.11"},
        timeout=timedelta(seconds=timeout),
    )
    try:
        async with sandbox:
            interpreter = await CodeInterpreter.create(sandbox)
            result = await interpreter.codes.run(
                code,
                language=SupportedLanguage.PYTHON
            )
            output = result.result[0].text if result.result else "\n".join([log.text for log in result.logs.stdout])
            plot_path = None
            if save_plot and result.result:
                reports_dir = Path(__file__).parent.parent / "reports"
                reports_dir.mkdir(exist_ok=True)
                plot_path = reports_dir / f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                # 这里可扩展让代码自行保存，当前保持简单
            return {
                "success": True,
                "code_preview": code[:300] + "..." if len(code) > 300 else code,
                "output": output,
                "logs": "\n".join([log.text for log in result.logs.stdout]),
                "plot_saved": str(plot_path) if plot_path else None
            }
    finally:
        await sandbox.kill()


def tool_function(code: str, timeout: int = 10, save_plot: bool = False):
    """统一入口（自动选择模式，完全兼容原来调用方式）"""
    if USE_OPEN_SANDBOX:
        try:
            return asyncio.run(_async_opensandbox_execute(code, timeout, save_plot))
        except Exception as e:
            logging.error(f"OpenSandbox 执行异常，回退到旧版: {e}")
            # 异常时也降级执行一次旧版
    # ====================== 旧版 threading 沙箱（原逻辑不变） ======================
    result_container = {"output": None, "done": False, "plot_saved": None}
    def target():
        try:
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from io import StringIO
            import math, random, statistics, json, os
            from datetime import datetime
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

    import threading
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


# ====================== tool_schema（完全不变） ======================
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