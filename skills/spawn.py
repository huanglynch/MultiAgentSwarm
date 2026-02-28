# skills/spawn.py
import threading
from multi_agent_swarm_v3 import MultiAgentSwarm  # 复用主系统

def tool_function(sub_task: str, memory_key: str = "subtask"):
    """启动一个独立的子Agent异步执行任务"""
    def run_sub_swarm():
        try:
            sub_swarm = MultiAgentSwarm()  # 新实例
            result = sub_swarm.solve(sub_task, use_memory=True, memory_key=memory_key)
            print(f"✅ 子任务完成: {result[:200]}...")
        except Exception as e:
            print(f"❌ 子任务失败: {e}")

    thread = threading.Thread(target=run_sub_swarm, daemon=True)
    thread.start()
    return {"success": True, "message": f"子代理已启动（线程ID: {thread.ident}），任务: {sub_task[:80]}..."}

tool_schema = {
    "type": "function",
    "function": {
        "name": "spawn",
        "description": "启动子代理异步执行耗时或并行任务（不阻塞主流程）",
        "parameters": {
            "type": "object",
            "properties": {
                "sub_task": {"type": "string", "description": "子任务完整描述"},
                "memory_key": {"type": "string", "description": "记忆键名（可选）"}
            },
            "required": ["sub_task"]
        }
    }
}