#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重置记忆工具 - 让 Agent 在对话中就能安全重置 VectorMemory
"""

def tool_function(confirm: bool = False):
    """重置 VectorMemory（需要 confirm=True 才能执行）"""
    # 这里需要访问 swarm 实例（在实际调用时由上层传入）
    # 由于 Skill 是独立文件，我们用全局或从主模块导入（最简单方式）
    try:
        from multi_agent_swarm_v4 import swarm  # 直接导入你的主实例（生产常用方式）
        if swarm.reset_vector_memory(confirm=confirm):
            return {
                "success": True,
                "message": "✅ VectorMemory 已完全重置！所有历史记忆已清空，系统恢复初始状态。"
            }
        else:
            return {
                "success": False,
                "message": "⚠️ 重置被拒绝。请确认你真的要清空所有长期记忆（添加 confirm=true）。"
            }
    except Exception as e:
        return {"success": False, "message": f"重置失败: {str(e)}"}


tool_schema = {
    "type": "function",
    "function": {
        "name": "reset_memory",
        "description": "彻底重置 VectorMemory（长期记忆）。**危险操作**！必须带 confirm=true 参数才能执行。用于切换项目、清理历史或测试时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "安全确认，必须设置为 true 才能真正执行重置",
                    "default": False
                }
            }
        }
    }
}

# 使用方法（WebUI 对话中直接说）：
#
# 用户/Agent：请重置记忆 confirm=true
# Agent 就会调用 reset_memory 工具，执行清空并返回确认消息。