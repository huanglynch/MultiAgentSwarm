#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiAgentSwarm WebUI - FastAPI 后端（精简版）
"""

import asyncio
import json
import os
import uuid
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# 在文件顶部添加这些 import（如果没有）
import re
import unicodedata
from pathlib import Path

# 导入你的 Swarm 系统
from multi_agent_swarm_v3 import MultiAgentSwarm

# ====================== FastAPI 应用 ======================
app = FastAPI(title="MultiAgentSwarm WebUI", version="3.1.0")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局变量
swarm: Optional[MultiAgentSwarm] = None
conversations: Dict[str, List[Dict]] = {}


# ====================== Pydantic 模型 ======================
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

    swarm.enable_adversarial_debate = config.adversarial_debate
    swarm.enable_meta_critic = config.meta_critic
    swarm.enable_task_decomposition = config.task_decomposition
    swarm.enable_knowledge_graph = config.knowledge_graph
    swarm.enable_adaptive_depth = config.adaptive_reflection
    swarm.intelligent_routing_enabled = config.intelligent_routing
    swarm.max_reflection_rounds = config.max_rounds
    swarm.reflection_quality_threshold = config.quality_threshold
    swarm.stop_quality_threshold = config.stop_threshold
    swarm.quality_convergence_delta = config.convergence_delta
    swarm.force_complexity = config.force_complexity

    print(f"✅ 配置已更新: {config.dict()}")


# ====================== API 端点 ======================
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    init_swarm()


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回主页（重定向到 static/index.html）"""
    return FileResponse("static/index.html")


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


# ==================== 新增：文件名净化函数 ====================
def sanitize_filename(original_name: str) -> str:
    """把中文、空格、特殊符号全部转为安全英文"""
    # 1. 中文转拼音（可选，更友好）或直接转 ASCII
    name = unicodedata.normalize('NFKD', original_name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # 2. 只保留安全字符
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)  # 空格、中文等 → _
    name = re.sub(r'_{2,}', '_', name)  # 多个下划线合并
    name = name.strip('_')

    # 3. 如果全被清理掉了，就给个默认名
    if not name:
        name = "file"
    return name.lower()  # 统一小写，更美观


# ==================== 修改 upload_file 函数 ====================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        MAX_FILE_SIZE = 10 * 1024 * 1024
        ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.gif', '.bmp'}

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

        # === 关键修改：自动净化文件名 ===
        stem = Path(file.filename).stem
        safe_stem = sanitize_filename(stem)
        safe_filename = f"{uuid.uuid4().hex[:8]}_{safe_stem}{file_ext}"

        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / safe_filename

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"文件过大（最大10MB）")

        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "status": "ok",
            "filename": safe_filename,  # 返回净化后的名字
            "path": str(file_path),
            "type": file_ext,
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


# ====================== WebSocket 端点（增强稳定性）======================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点（支持真实流式输出 + 心跳保活）"""
    await websocket.accept()
    import time
    start_time = time.time()

    # ✅ 心跳任务（每30秒发送一次 ping）
    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(30)
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({"type": "ping"})
        except Exception as e:
            print(f"⚠️ 心跳任务异常: {e}")

    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            data = await websocket.receive_json()

            # 🔥🔥🔥 关键修复：忽略心跳 pong（只需加这 3 行）🔥🔥🔥
            if data.get("type") in ("pong", "ping"):
                continue  # ← 跳过心跳，不当成用户消息

            message = data.get("message", "").strip()
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "content": "❌ 消息不能为空"
                })
                continue

            session_id = get_or_create_session(data.get("session_id"))
            use_memory = data.get("use_memory", False)
            memory_key = data.get("memory_key", "default")
            force_complexity = data.get("force_complexity")

            user_msg = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            conversations[session_id].append(user_msg)

            await websocket.send_json({
                "type": "session_id",
                "session_id": session_id
            })

            # 构建历史上下文（保持原有逻辑）
            history_context = ""
            history_lines = []

            if len(conversations[session_id]) > 1:
                recent_messages = conversations[session_id][:-1]
                if len(recent_messages) > 10:
                    recent_messages = recent_messages[-10:]

                MAX_HISTORY_TOKENS = 2000
                accumulated_text = ""
                selected_messages = []

                for msg in reversed(recent_messages):
                    candidate = f"{msg['content']}\n\n{accumulated_text}"
                    estimated_tokens = len(candidate) * 0.75

                    if estimated_tokens > MAX_HISTORY_TOKENS:
                        break

                    accumulated_text = candidate
                    selected_messages.insert(0, msg)

                if selected_messages:
                    for msg in selected_messages:
                        if msg["role"] == "system":
                            history_lines.append(msg["content"])
                        else:
                            role_name = "User" if msg["role"] == "user" else "Assistant"
                            content = msg["content"][:500]
                            if len(msg["content"]) > 500:
                                content += "..."
                            history_lines.append(f"{role_name}: {content}")

                    history_context = "\n\n".join(history_lines)

            if history_context:
                full_message = f"""=== 📚 对话历史（最近 {len(history_lines)} 轮）===
{history_context}

=== 💬 当前问题 ===
User: {message}"""
            else:
                full_message = message

            # 附件处理（保持原有逻辑）
            if "📎 附件:" in message:
                try:
                    file_paths = [
                        line.strip("- ").strip()
                        for line in message.split("📎 附件:")[-1].split("\n")
                        if line.strip().startswith("- ")
                    ]

                    if file_paths:
                        await websocket.send_json({
                            "type": "log",
                            "content": f"📂 检测到 {len(file_paths)} 个附件，正在解析..."
                        })

                        file_contents = []
                        MAX_PREVIEW_LENGTH = 10000

                        for path in file_paths:
                            try:
                                path = path.strip()

                                if path.endswith('.pdf'):
                                    result = swarm.tool_registry['pdf_reader']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')
                                        truncated = False

                                        if len(content) > MAX_PREVIEW_LENGTH:
                                            content = content[:MAX_PREVIEW_LENGTH]
                                            truncated = True

                                        file_contents.append(
                                            f"### 📄 {Path(path).name} (PDF)\n"
                                            f"【系统指令：以下是附件完整解析内容（已截断），请直接基于此内容进行分析评价，无需再次调用 pdf_reader、summarize_long_file 或任何读取工具】\n"
                                            f"页数: {result.get('pages', '未知')}\n"
                                            f"预览长度: {len(content)} 字符{'（已截断）' if truncated else ''}\n"
                                            f"内容:\n{content}"
                                            + (
                                                "\n\n💡 **提示**: 文件过长已截断，如需完整分析请明确要求使用 `summarize_long_file` 工具。" if truncated else "")
                                        )
                                    else:
                                        file_contents.append(
                                            f"### ❌ {Path(path).name} 解析失败: {result.get('error', '未知错误')}")

                                elif path.endswith(('.txt', '.md')):
                                    result = swarm.tool_registry['read_file']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')
                                        truncated = False

                                        if len(content) > MAX_PREVIEW_LENGTH:
                                            content = content[:MAX_PREVIEW_LENGTH]
                                            truncated = True

                                        file_contents.append(
                                            f"### 📄 {Path(path).name}\n"
                                            f"【系统指令：以下是附件完整解析内容（已截断），请直接基于此内容进行分析评价，无需再次调用 pdf_reader、summarize_long_file 或任何读取工具】\n"
                                            f"大小: {result.get('length', 0)} 字符\n"
                                            f"预览长度: {len(content)} 字符{'（已截断）' if truncated else ''}\n"
                                            f"内容:\n{content}"
                                            + (
                                                "\n\n💡 **提示**: 文件过长已截断，如需完整分析请明确要求使用 `summarize_long_file` 工具。" if truncated else "")
                                        )
                                    else:
                                        file_contents.append(
                                            f"### ❌ {Path(path).name} 读取失败: {result.get('error', '未知错误')}")

                                elif path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                    file_contents.append(f"### 🖼️ {Path(path).name} (图片)\n路径: {path}")

                            except Exception as e:
                                file_contents.append(f"### ❌ {Path(path).name} 处理失败: {str(e)}")

                        if file_contents:
                            file_section = "\n\n=== 📄 附件内容 ===\n" + "\n\n".join(file_contents)
                            full_message = full_message + file_section

                            await websocket.send_json({
                                "type": "log",
                                "content": f"✅ 附件解析完成，总计 {len(file_contents)} 个文件"
                            })

                except Exception as e:
                    print(f"⚠️ 附件解析失败: {e}")
                    await websocket.send_json({
                        "type": "log",
                        "content": f"⚠️ 附件解析失败: {str(e)[:50]}"
                    })

            # 创建异步队列
            stream_queue = asyncio.Queue()
            log_queue = asyncio.Queue()

            async def stream_sender():
                """持续发送流式数据到前端"""
                while True:
                    try:
                        data = await asyncio.wait_for(stream_queue.get(), timeout=0.1)
                        if data is None:
                            break
                        # ✅ 检查连接状态
                        if websocket.client_state.name == "CONNECTED":
                            await websocket.send_json(data)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"⚠️ 流式发送失败: {e}")
                        break

            async def log_sender():
                """持续发送日志到前端"""
                while True:
                    try:
                        log_msg = await asyncio.wait_for(log_queue.get(), timeout=0.1)
                        if log_msg is None:
                            break
                        simplified = log_msg[:60] + "..." if len(log_msg) > 60 else log_msg
                        # ✅ 检查连接状态
                        if websocket.client_state.name == "CONNECTED":
                            await websocket.send_json({
                                "type": "log",
                                "content": simplified
                            })
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"⚠️ 日志发送失败: {e}")
                        break

            sender_task = asyncio.create_task(stream_sender())
            log_task = asyncio.create_task(log_sender())

            try:
                loop = asyncio.get_event_loop()

                def stream_callback(agent_name: str, content: str):
                    """流式内容回调"""
                    asyncio.run_coroutine_threadsafe(
                        stream_queue.put({
                            "type": "stream",
                            "agent": agent_name,
                            "content": content
                        }),
                        loop
                    )

                def log_callback(message: str):
                    """日志回调"""
                    asyncio.run_coroutine_threadsafe(
                        log_queue.put(message),
                        loop
                    )

                answer = await loop.run_in_executor(
                    None,
                    lambda: swarm.solve(
                        full_message,
                        use_memory,
                        memory_key,
                        None,
                        force_complexity,
                        stream_callback=stream_callback,
                        log_callback=log_callback
                    )
                )

                ai_msg = {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat()
                }
                conversations[session_id].append(ai_msg)

                await stream_queue.put(None)
                await log_queue.put(None)
                await sender_task
                await log_task

                elapsed = time.time() - start_time
                # ✅ 检查连接状态
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({
                        "type": "log",
                        "content": f"⏱️ 总耗时: {elapsed:.2f}秒"
                    })
                    await websocket.send_json({
                        "type": "end"
                    })

                start_time = time.time()

            except Exception as e:
                print(f"❌ Swarm 执行失败: {e}")
                import traceback
                traceback.print_exc()

                await stream_queue.put(None)
                await log_queue.put(None)

                try:
                    await sender_task
                    await log_task
                except:
                    pass

                error_msg = f"❌ 执行失败: {str(e)[:200]}"

                try:
                    # ✅ 检查连接状态
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_json({
                            "type": "error",
                            "content": error_msg
                        })
                        await websocket.send_json({
                            "type": "end"
                        })
                except Exception as send_error:
                    print(f"⚠️ WebSocket 已关闭，无法发送错误消息: {send_error}")
                    break

                conversations[session_id].append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        print(f"🔌 WebSocket 断开连接")
    except Exception as e:
        print(f"💥 WebSocket 致命错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ✅ 清理心跳任务
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


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