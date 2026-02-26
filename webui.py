#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiAgentSwarm WebUI - FastAPI 实现
美观、简洁、功能完整的 Web 界面（支持真实流式输出）
"""

import asyncio
import json
import os
import uuid
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 导入你的 Swarm 系统
from multi_agent_swarm_v3 import MultiAgentSwarm

# ====================== FastAPI 应用 ======================
app = FastAPI(title="MultiAgentSwarm WebUI", version="3.1.0")

# 全局 Swarm 实例
swarm: Optional[MultiAgentSwarm] = None

# 对话历史存储 {session_id: [messages]}
conversations: Dict[str, List[Dict]] = {}


# ====================== Pydantic 模型 ======================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_memory: bool = False
    memory_key: str = "default"
    force_complexity: Optional[str] = None


class ConfigUpdate(BaseModel):
    adversarial_debate: bool = True
    meta_critic: bool = True
    task_decomposition: bool = True
    knowledge_graph: bool = True
    adaptive_reflection: bool = True
    intelligent_routing: bool = True
    max_rounds: int = 3
    quality_threshold: int = 85
    stop_threshold: int = 80
    convergence_delta: int = 3
    force_complexity: Optional[str] = None


# ====================== 初始化 Swarm ======================
def init_swarm():
    """初始化 Swarm 实例"""
    global swarm
    try:
        swarm = MultiAgentSwarm(config_path="swarm_config.yaml")
        print("✅ Swarm 初始化成功")
    except Exception as e:
        print(f"❌ Swarm 初始化失败: {e}")
        raise


# ====================== 工具函数 ======================
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """获取或创建会话 ID"""
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in conversations:
        conversations[session_id] = []
    return session_id


def update_config(config: ConfigUpdate):
    """动态更新 Swarm 配置"""
    if not swarm:
        raise HTTPException(status_code=500, detail="Swarm 未初始化")

    # 更新增强功能
    swarm.enable_adversarial_debate = config.adversarial_debate
    swarm.enable_meta_critic = config.meta_critic
    swarm.enable_task_decomposition = config.task_decomposition
    swarm.enable_knowledge_graph = config.knowledge_graph
    swarm.enable_adaptive_depth = config.adaptive_reflection
    swarm.intelligent_routing_enabled = config.intelligent_routing

    # 更新自适应反思参数
    swarm.max_reflection_rounds = config.max_rounds
    swarm.reflection_quality_threshold = config.quality_threshold
    swarm.stop_quality_threshold = config.stop_threshold
    swarm.quality_convergence_delta = config.convergence_delta

    # 更新智能路由
    swarm.force_complexity = config.force_complexity

    print(f"✅ 配置已更新: {config.dict()}")


# ====================== API 端点 ======================
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    init_swarm()


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回 Web 界面"""
    return get_html_template()


@app.get("/api/sessions")
async def list_sessions():
    """获取所有会话"""
    return {
        "sessions": [
            {
                "id": sid,
                "message_count": len(msgs),
                "last_message": msgs[-1]["content"][:50] if msgs else ""
            }
            for sid, msgs in conversations.items()
        ]
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话历史"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"messages": conversations[session_id]}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in conversations:
        del conversations[session_id]
    return {"status": "ok"}


@app.delete("/api/session/{session_id}/message/{message_index}")
async def delete_message(session_id: str, message_index: int):
    """删除指定消息"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    if 0 <= message_index < len(conversations[session_id]):
        conversations[session_id].pop(message_index)
    return {"status": "ok"}


@app.post("/api/config")
async def update_swarm_config(config: ConfigUpdate):
    """更新 Swarm 配置"""
    try:
        update_config(config)
        return {"status": "ok", "config": config.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_swarm_config():
    """获取当前 Swarm 配置"""
    if not swarm:
        raise HTTPException(status_code=500, detail="Swarm 未初始化")

    return {
        "adversarial_debate": swarm.enable_adversarial_debate,
        "meta_critic": swarm.enable_meta_critic,
        "task_decomposition": swarm.enable_task_decomposition,
        "knowledge_graph": swarm.enable_knowledge_graph,
        "adaptive_reflection": swarm.enable_adaptive_depth,
        "intelligent_routing": swarm.intelligent_routing_enabled,
        "max_rounds": swarm.max_reflection_rounds,
        "quality_threshold": swarm.reflection_quality_threshold,
        "stop_threshold": swarm.stop_quality_threshold,
        "convergence_delta": swarm.quality_convergence_delta,
        "force_complexity": swarm.force_complexity,
    }


@app.get("/api/export/{session_id}")
async def export_session(session_id: str):
    """导出会话历史为 Markdown 格式"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = conversations[session_id]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"conversation_{timestamp}.md"

    # ✅ 使用 tempfile 创建跨平台临时文件
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# MultiAgentSwarm 对话记录\n\n")
            f.write(f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**会话 ID**: {session_id}\n\n")
            f.write("---\n\n")

            for msg in messages:
                role_name = "👤 用户" if msg['role'] == 'user' else "🤖 助手"
                f.write(f"## {role_name}\n\n")
                f.write(f"**时间**: {msg['timestamp']}\n\n")
                f.write(f"{msg['content']}\n\n")
                f.write("---\n\n")

        return FileResponse(
            filepath,
            filename=filename,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点（支持真实流式输出）"""
    await websocket.accept()

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = get_or_create_session(data.get("session_id"))
            use_memory = data.get("use_memory", False)
            memory_key = data.get("memory_key", "default")
            force_complexity = data.get("force_complexity")

            # 保存用户消息
            user_msg = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            conversations[session_id].append(user_msg)

            # 发送会话 ID
            await websocket.send_json({
                "type": "session_id",
                "session_id": session_id
            })

            # ✅ 创建流式和日志队列
            stream_queue = asyncio.Queue()
            log_queue = asyncio.Queue()

            # ✅ 后台任务：持续发送流式数据
            async def stream_sender():
                """持续发送流式数据到前端"""
                while True:
                    try:
                        data = await asyncio.wait_for(stream_queue.get(), timeout=0.1)
                        if data is None:  # 结束信号
                            break
                        await websocket.send_json(data)
                    except asyncio.TimeoutError:
                        continue

            # ✅ 后台任务：持续发送日志
            async def log_sender():
                """持续发送日志到前端"""
                while True:
                    try:
                        log_msg = await asyncio.wait_for(log_queue.get(), timeout=0.1)
                        if log_msg is None:  # 结束信号
                            break
                        # 简化日志显示
                        simplified = log_msg[:60] + "..." if len(log_msg) > 60 else log_msg
                        await websocket.send_json({
                            "type": "log",
                            "content": simplified
                        })
                    except asyncio.TimeoutError:
                        continue

            # 启动后台任务
            sender_task = asyncio.create_task(stream_sender())
            log_task = asyncio.create_task(log_sender())

            try:
                loop = asyncio.get_event_loop()

                # ✅ 定义流式回调
                def stream_callback(agent_name: str, content: str):
                    """流式内容回调 - 将内容发送到队列"""
                    asyncio.run_coroutine_threadsafe(
                        stream_queue.put({
                            "type": "stream",
                            "agent": agent_name,
                            "content": content
                        }),
                        loop
                    )

                # ✅ 定义日志回调
                def log_callback(message: str):
                    """日志回调 - 将日志发送到队列"""
                    asyncio.run_coroutine_threadsafe(
                        log_queue.put(message),
                        loop
                    )

                # ✅ 执行 Swarm（带回调）
                answer = await loop.run_in_executor(
                    None,
                    lambda: swarm.solve(
                        message,
                        use_memory,
                        memory_key,
                        None,  # image_paths
                        force_complexity,
                        stream_callback=stream_callback,  # ✅ 传递流式回调
                        log_callback=log_callback  # ✅ 传递日志回调
                    )
                )

                # 保存 AI 回复
                ai_msg = {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat()
                }
                conversations[session_id].append(ai_msg)

                # 发送结束信号
                await stream_queue.put(None)
                await log_queue.put(None)
                await sender_task
                await log_task

                await websocket.send_json({
                    "type": "end"
                })

            except Exception as e:
                # 异常处理
                await stream_queue.put(None)
                await log_queue.put(None)
                await sender_task
                await log_task

                error_msg = f"❌ 执行失败: {str(e)}"

                # ✅ 检查连接状态再发送
                try:
                    await websocket.send_json({
                        "type": "error",
                        "content": error_msg
                    })
                except:
                    print(f"⚠️ WebSocket 已关闭，无法发送错误消息")
                    break

                # 保存错误消息
                conversations[session_id].append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        print("WebSocket 断开连接")
    except Exception as e:
        print(f"WebSocket 错误: {e}")


# ====================== HTML 模板（完整版）======================
def get_html_template():
    """返回 HTML 模板"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MultiAgentSwarm WebUI</title>
    <!-- Marked.js for Markdown rendering -->
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 100%;
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        .sidebar {
            width: 300px;
            background: #f8f9fa;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
        }

        .sidebar-header {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .sidebar-header h2 {
            font-size: 18px;
            margin-bottom: 5px;
        }

        .sidebar-header p {
            font-size: 12px;
            opacity: 0.9;
        }

        .session-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }

        .session-item {
            padding: 12px;
            margin-bottom: 8px;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid transparent;
        }

        .session-item:hover {
            border-color: #667eea;
            transform: translateX(5px);
        }

        .session-item.active {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .new-session-btn {
            margin: 10px;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }

        .new-session-btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            padding: 20px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chat-header h1 {
            font-size: 24px;
            color: #667eea;
        }

        .header-buttons button {
            margin-left: 10px;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }

        .btn-settings {
            background: #f0f4ff;
            color: #667eea;
        }

        .btn-settings:hover {
            background: #e0e8ff;
        }

        .btn-export {
            background: #4caf50;
            color: white;
        }

        .btn-export:hover {
            background: #45a049;
        }

        .btn-clear {
            background: #f44336;
            color: white;
        }

        .btn-clear:hover {
            background: #da190b;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #fafafa;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 18px;
            position: relative;
            word-wrap: break-word;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .message.assistant .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        /* Markdown 样式 */
        .message-content h1, .message-content h2, .message-content h3 {
            margin-top: 16px;
            margin-bottom: 8px;
        }

        .message-content h1 { font-size: 1.5em; }
        .message-content h2 { font-size: 1.3em; }
        .message-content h3 { font-size: 1.1em; }

        .message-content p {
            margin-bottom: 12px;
            line-height: 1.6;
        }

        .message-content ul, .message-content ol {
            margin-left: 20px;
            margin-bottom: 12px;
        }

        .message-content li {
            margin-bottom: 6px;
            line-height: 1.6;
        }

        .message-content code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .message-content pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin-bottom: 12px;
        }

        .message-content pre code {
            background: none;
            padding: 0;
            color: inherit;
        }

        .message-content blockquote {
            border-left: 4px solid #667eea;
            padding-left: 12px;
            margin: 12px 0;
            color: #666;
            font-style: italic;
        }

        .message-content table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 12px;
        }

        .message-content th, .message-content td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        .message-content th {
            background: #f4f4f4;
            font-weight: bold;
        }

        /* ✅ 思考过程样式 */
        .thinking-details {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .thinking-details summary {
            cursor: pointer;
            font-weight: 600;
            color: white;
            user-select: none;
            list-style: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .thinking-details summary::-webkit-details-marker {
            display: none;
        }

        .thinking-details summary::before {
            content: '▼';
            display: inline-block;
            transition: transform 0.3s;
        }

        .thinking-details:not([open]) summary::before {
            transform: rotate(-90deg);
        }

        .thinking-logs {
            margin-top: 12px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 8px;
            padding: 12px;
            max-height: 300px;
            overflow-y: auto;
        }

        .log-entry {
            padding: 6px 10px;
            margin: 4px 0;
            background: rgba(102, 126, 234, 0.1);
            border-left: 3px solid #667eea;
            border-radius: 4px;
            font-size: 13px;
            font-family: 'Consolas', 'Monaco', monospace;
            color: #2d3748;
            word-break: break-word;
        }

        /* ✅ 流式消息样式 */
        .message.streaming {
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }

        .agent-label {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .streaming-content {
            min-height: 20px;
        }

        /* ✅ 日志滚动条美化 */
        .thinking-logs::-webkit-scrollbar {
            width: 6px;
        }

        .thinking-logs::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.05);
            border-radius: 3px;
        }

        .thinking-logs::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 3px;
        }

        .thinking-logs::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
        }

        .message-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .message:hover .message-actions {
            opacity: 1;
        }

        .message-actions button {
            padding: 4px 12px;
            font-size: 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: #f0f0f0;
            color: #666;
            transition: all 0.3s;
        }

        .message-actions button:hover {
            background: #e0e0e0;
            color: #333;
        }

        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        #messageInput {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            font-size: 14px;
            outline: none;
            transition: all 0.3s;
            resize: none;
            min-height: 72px;
            max-height: 200px;
            overflow-y: auto;
            font-family: inherit;
            line-height: 1.5;
        }

        #messageInput:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        #sendBtn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
            height: 50px;
        }

        #sendBtn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        #sendBtn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .settings-panel {
            position: fixed;
            top: 0;
            right: -400px;
            width: 400px;
            height: 100vh;
            background: white;
            box-shadow: -5px 0 20px rgba(0,0,0,0.2);
            transition: right 0.3s;
            z-index: 1000;
            overflow-y: auto;
        }

        .settings-panel.active {
            right: 0;
        }

        .settings-header {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .settings-content {
            padding: 20px;
        }

        .setting-group {
            margin-bottom: 25px;
        }

        .setting-group h3 {
            font-size: 16px;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }

        .setting-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
        }

        .setting-item label {
            font-size: 14px;
            color: #666;
            flex: 1;
        }

        .toggle-switch {
            width: 50px;
            height: 26px;
            background: #ccc;
            border-radius: 13px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }

        .toggle-switch.active {
            background: #667eea;
        }

        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: left 0.3s;
        }

        .toggle-switch.active::after {
            left: 26px;
        }

        .setting-item input[type="number"],
        .setting-item select {
            width: 80px;
            padding: 6px 10px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
        }

        .setting-item select {
            width: 120px;
        }

        .close-settings {
            background: none;
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
        }

        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                left: -300px;
                height: 100vh;
                z-index: 999;
                transition: left 0.3s;
            }

            .sidebar.active {
                left: 0;
            }

            .container {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h2>🤖 会话列表</h2>
                <p>MultiAgentSwarm v3.1.0</p>
            </div>
            <div class="session-list" id="sessionList"></div>
            <button class="new-session-btn" onclick="createNewSession()">➕ 新建会话</button>
        </div>

        <div class="chat-container">
            <div class="chat-header">
                <h1>💬 MultiAgentSwarm</h1>
                <div class="header-buttons">
                    <button class="btn-settings" onclick="toggleSettings()">⚙️ 设置</button>
                    <button class="btn-export" onclick="exportChat()">💾 导出</button>
                    <button class="btn-clear" onclick="clearChat()">🗑️ 清空</button>
                </div>
            </div>

            <div class="messages" id="messages"></div>

            <div class="input-area">
                <div class="input-wrapper">
                    <textarea 
                        id="messageInput" 
                        placeholder="输入你的问题...（Enter 换行，Ctrl+Enter 发送）" 
                        onkeydown="handleKeyDown(event)"
                        rows="3"
                    ></textarea>
                    <button id="sendBtn" onclick="sendMessage()">发送 🚀</button>
                </div>
            </div>
        </div>
    </div>

    <div class="settings-panel" id="settingsPanel">
        <div class="settings-header">
            <h2>⚙️ 高级设置</h2>
            <button class="close-settings" onclick="toggleSettings()">✕</button>
        </div>
        <div class="settings-content">
            <div class="setting-group">
                <h3>🚀 增强功能</h3>
                <div class="setting-item">
                    <label>对抗式辩论</label>
                    <div class="toggle-switch active" data-config="adversarial_debate"></div>
                </div>
                <div class="setting-item">
                    <label>Meta-Critic</label>
                    <div class="toggle-switch active" data-config="meta_critic"></div>
                </div>
                <div class="setting-item">
                    <label>任务分解</label>
                    <div class="toggle-switch active" data-config="task_decomposition"></div>
                </div>
                <div class="setting-item">
                    <label>知识图谱</label>
                    <div class="toggle-switch active" data-config="knowledge_graph"></div>
                </div>
                <div class="setting-item">
                    <label>自适应反思</label>
                    <div class="toggle-switch active" data-config="adaptive_reflection"></div>
                </div>
                <div class="setting-item">
                    <label>智能路由</label>
                    <div class="toggle-switch active" data-config="intelligent_routing"></div>
                </div>
            </div>

            <div class="setting-group">
                <h3>📊 反思参数</h3>
                <div class="setting-item">
                    <label>最大轮次</label>
                    <input type="number" id="max_rounds" value="3" min="1" max="10">
                </div>
                <div class="setting-item">
                    <label>质量阈值</label>
                    <input type="number" id="quality_threshold" value="85" min="0" max="100">
                </div>
                <div class="setting-item">
                    <label>停止阈值</label>
                    <input type="number" id="stop_threshold" value="80" min="0" max="100">
                </div>
                <div class="setting-item">
                    <label>收敛阈值</label>
                    <input type="number" id="convergence_delta" value="3" min="1" max="10">
                </div>
            </div>

            <div class="setting-group">
                <h3>🧭 智能路由</h3>
                <div class="setting-item">
                    <label>强制模式</label>
                    <select id="force_complexity">
                        <option value="">自动判断</option>
                        <option value="simple">Simple</option>
                        <option value="medium">Medium</option>
                        <option value="complex">Complex</option>
                    </select>
                </div>
            </div>

            <button 
                style="width: 100%; padding: 12px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; margin-top: 20px;"
                onclick="saveSettings()"
            >
                💾 保存设置
            </button>
        </div>
    </div>

    <script>
        let ws = null;
        let currentSessionId = null;
        let isProcessing = false;
        let currentStreamingDiv = null;  // ✅ 新增
        let currentStreamingAgent = null;  // ✅ 新增
        let thinkingDetailsElement = null;  // ✅ 新增

        // 配置 Marked.js
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });

        document.addEventListener('DOMContentLoaded', async function() {
            await loadConfig();
            await loadSessions();
            createNewSession();
            initToggleSwitches();
        });

        function initToggleSwitches() {
            document.querySelectorAll('.toggle-switch').forEach(function(toggle) {
                toggle.addEventListener('click', function() {
                    this.classList.toggle('active');
                });
            });
        }

        function connectWebSocket() {
            if (ws) ws.close();

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws';
            ws = new WebSocket(wsUrl);

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.type === 'session_id') {
                    currentSessionId = data.session_id;
                } 
                else if (data.type === 'log') {
                    // ✅ 处理思考日志
                    addThinkingLog(data.content);
                }
                else if (data.type === 'stream') {
                    // ✅ 处理流式输出
                    updateStreamingMessage(data.agent, data.content);
                }
                else if (data.type === 'error') {
                    if (thinkingDetailsElement) {
                        thinkingDetailsElement.remove();
                        thinkingDetailsElement = null;
                    }
                    addMessage('assistant', data.content);
                } 
                // ✅ 修改 end 事件处理
                else if (data.type === 'end') {
                    // 确保流式消息已完成
                    finalizeStreamingMessage();
                    
                    // 关闭思考过程
                    if (thinkingDetailsElement) {
                        thinkingDetailsElement.removeAttribute('open');
                        thinkingDetailsElement = null;
                    }
                    
                    isProcessing = false;
                    document.getElementById('sendBtn').disabled = false;
                    
                    // ✅ 刷新会话列表（显示最新消息预览）
                    loadSessions();
                }
            };

            ws.onerror = function() {
                console.error('WebSocket 错误');
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
            };
        }

        // ✅ 新增：添加思考日志
        function addThinkingLog(logContent) {
            if (!thinkingDetailsElement) {
                // 创建思考过程容器
                const messagesDiv = document.getElementById('messages');
                thinkingDetailsElement = document.createElement('details');
                thinkingDetailsElement.className = 'thinking-details';
                thinkingDetailsElement.open = true;

                thinkingDetailsElement.innerHTML = `
                    <summary>🤔 思考过程</summary>
                    <div class="thinking-logs"></div>
                `;

                messagesDiv.appendChild(thinkingDetailsElement);
            }

            // 添加日志条目
            const logsDiv = thinkingDetailsElement.querySelector('.thinking-logs');
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.textContent = logContent;
            logsDiv.appendChild(logEntry);

            // 自动滚动
            const messagesDiv = document.getElementById('messages');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // ✅ 新增：更新流式消息
        function updateStreamingMessage(agent, content) {
            if (!currentStreamingDiv || currentStreamingAgent !== agent) {
                // 创建新的流式消息容器
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant streaming';

                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = `
                    <div class="agent-label">[${agent}]</div>
                    <div class="streaming-content"></div>
                `;

                messageDiv.appendChild(contentDiv);
                messagesDiv.appendChild(messageDiv);

                currentStreamingDiv = contentDiv.querySelector('.streaming-content');
                currentStreamingAgent = agent;
            }

            // 追加内容（先累积纯文本，然后渲染 Markdown）
            if (!currentStreamingDiv.dataset.rawText) {
                currentStreamingDiv.dataset.rawText = '';
            }
            currentStreamingDiv.dataset.rawText += content;

            // 渲染 Markdown
            currentStreamingDiv.innerHTML = marked.parse(currentStreamingDiv.dataset.rawText);

            // 自动滚动
            const messagesDiv = document.getElementById('messages');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // ✅ 新增：完成流式消息
        function finalizeStreamingMessage() {
            if (currentStreamingDiv) {
                // 添加操作按钮
                const actionsDiv = document.createElement('div');
                actionsDiv.className = 'message-actions';
                actionsDiv.innerHTML = `
                    <button onclick="copyMessage(this)">📋 复制</button>
                    <button onclick="deleteMessage(this)">🗑️ 删除</button>
                `;
                currentStreamingDiv.parentElement.appendChild(actionsDiv);
        
                // 移除 streaming 类
                const messageDiv = currentStreamingDiv.closest('.message');
                messageDiv.classList.remove('streaming');
        
                // ✅ 新增：标记为已完成（用于后续识别）
                messageDiv.dataset.finalized = 'true';
        
                // 清除引用
                delete currentStreamingDiv.dataset.rawText;
                currentStreamingDiv = null;
                currentStreamingAgent = null;
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || isProcessing) return;

            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;

            addMessage('user', message);
            input.value = '';

            connectWebSocket();

            await new Promise(function(resolve) {
                const checkConnection = setInterval(function() {
                    if (ws.readyState === WebSocket.OPEN) {
                        clearInterval(checkConnection);
                        resolve();
                    }
                }, 100);
            });

            const forceComplexity = document.getElementById('force_complexity').value || null;

            ws.send(JSON.stringify({
                message: message,
                session_id: currentSessionId,
                use_memory: false,
                memory_key: 'default',
                force_complexity: forceComplexity
            }));
        }

        function addMessage(role, content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            if (role === 'assistant') {
                // 使用 Marked.js 渲染 Markdown
                contentDiv.innerHTML = marked.parse(content);

                const actionsDiv = document.createElement('div');
                actionsDiv.className = 'message-actions';
                actionsDiv.innerHTML = '<button onclick="copyMessage(this)">📋 复制</button><button onclick="deleteMessage(this)">🗑️ 删除</button>';
                contentDiv.appendChild(actionsDiv);
            } else {
                contentDiv.textContent = content;
            }

            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function copyMessage(btn) {
            const content = btn.closest('.message-content').cloneNode(true);
            content.querySelector('.message-actions').remove();
            const text = content.textContent || content.innerText;
            navigator.clipboard.writeText(text);
            btn.textContent = '✅ 已复制';
            setTimeout(function() { btn.textContent = '📋 复制'; }, 2000);
        }

        function deleteMessage(btn) {
            if (confirm('确定删除这条消息吗?')) {
                btn.closest('.message').remove();
            }
        }

        function handleKeyDown(event) {
            // Ctrl+Enter 或 Cmd+Enter 发送消息
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                event.preventDefault();
                sendMessage();
            }
        }

        function clearChat() {
            if (confirm('确定清空当前对话吗?')) {
                document.getElementById('messages').innerHTML = '';
                if (currentSessionId) {
                    fetch('/api/session/' + currentSessionId, { method: 'DELETE' });
                }
                createNewSession();
            }
        }

        function exportChat() {
            if (!currentSessionId) {
                alert('当前没有对话');
                return;
            }
            window.open('/api/export/' + currentSessionId, '_blank');
        }

        function createNewSession() {
            currentSessionId = null;
            currentStreamingDiv = null;
            currentStreamingAgent = null;
            thinkingDetailsElement = null;
            document.getElementById('messages').innerHTML = '';
            addMessage('assistant', '👋 你好！我是 MultiAgentSwarm，一个多智能体协作系统。有什么可以帮你的吗？');
        }

        async function loadSessions() {
            try {
                const response = await fetch('/api/sessions');
                const data = await response.json();

                const listDiv = document.getElementById('sessionList');
                listDiv.innerHTML = '';

                data.sessions.forEach(function(session) {
                    const div = document.createElement('div');
                    div.className = 'session-item';
                    if (session.id === currentSessionId) div.classList.add('active');
                    div.innerHTML = '<div style="font-weight: bold; margin-bottom: 5px;">💬 会话 ' + session.id.slice(0, 8) + '</div><div style="font-size: 12px; color: #999;">' + session.last_message + '...</div>';
                    div.onclick = function() { loadSession(session.id); };
                    listDiv.appendChild(div);
                });
            } catch (error) {
                console.error('加载会话失败:', error);
            }
        }

        async function loadSession(sessionId) {
            try {
                const response = await fetch('/api/session/' + sessionId);
                const data = await response.json();

                currentSessionId = sessionId;
                currentStreamingDiv = null;
                currentStreamingAgent = null;
                thinkingDetailsElement = null;
                document.getElementById('messages').innerHTML = '';

                data.messages.forEach(function(msg) {
                    addMessage(msg.role, msg.content);
                });

                document.querySelectorAll('.session-item').forEach(function(item) {
                    item.classList.remove('active');
                });
                event.target.closest('.session-item').classList.add('active');
            } catch (error) {
                console.error('加载会话失败:', error);
            }
        }

        function toggleSettings() {
            document.getElementById('settingsPanel').classList.toggle('active');
        }

        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                const config = await response.json();

                document.querySelectorAll('.toggle-switch').forEach(function(toggle) {
                    const key = toggle.dataset.config;
                    if (config[key]) {
                        toggle.classList.add('active');
                    } else {
                        toggle.classList.remove('active');
                    }
                });

                document.getElementById('max_rounds').value = config.max_rounds;
                document.getElementById('quality_threshold').value = config.quality_threshold;
                document.getElementById('stop_threshold').value = config.stop_threshold;
                document.getElementById('convergence_delta').value = config.convergence_delta;
                document.getElementById('force_complexity').value = config.force_complexity || '';
            } catch (error) {
                console.error('加载配置失败:', error);
            }
        }

        async function saveSettings() {
            const config = {
                adversarial_debate: document.querySelector('[data-config="adversarial_debate"]').classList.contains('active'),
                meta_critic: document.querySelector('[data-config="meta_critic"]').classList.contains('active'),
                task_decomposition: document.querySelector('[data-config="task_decomposition"]').classList.contains('active'),
                knowledge_graph: document.querySelector('[data-config="knowledge_graph"]').classList.contains('active'),
                adaptive_reflection: document.querySelector('[data-config="adaptive_reflection"]').classList.contains('active'),
                intelligent_routing: document.querySelector('[data-config="intelligent_routing"]').classList.contains('active'),
                max_rounds: parseInt(document.getElementById('max_rounds').value),
                quality_threshold: parseInt(document.getElementById('quality_threshold').value),
                stop_threshold: parseInt(document.getElementById('stop_threshold').value),
                convergence_delta: parseInt(document.getElementById('convergence_delta').value),
                force_complexity: document.getElementById('force_complexity').value || null
            };

            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });

                if (response.ok) {
                    alert('✅ 设置已保存');
                    toggleSettings();
                } else {
                    alert('❌ 保存失败');
                }
            } catch (error) {
                console.error('保存配置失败:', error);
                alert('❌ 保存失败: ' + error.message);
            }
        }
    </script>
</body>
</html>"""


# ====================== 启动服务器 ======================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 80)
    print("🚀 MultiAgentSwarm WebUI 启动中...")
    print("=" * 80)
    print("📍 访问地址: http://localhost:8060")
    print("📖 API 文档: http://localhost:8060/docs")
    print("=" * 80 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8060,
        log_level="info"
    )