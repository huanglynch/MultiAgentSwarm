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
from fastapi import UploadFile, File
from pathlib import Path

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
    """
    WebSocket 端点（支持真实流式输出 + 多轮对话历史）
    ✅ 优化点：
    1. 智能历史管理（最近 10 轮 + Token 限制）
    2. 历史压缩（每 5 轮自动总结）
    3. 异常恢复机制
    4. 性能监控
    """
    await websocket.accept()

    # 性能监控
    import time
    start_time = time.time()

    try:
        while True:
            # ==================== 1. 接收并解析消息 ====================
            data = await websocket.receive_json()
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

            # ==================== 2. 保存用户消息 ====================
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

            # ==================== 3. 构建智能对话历史 ====================
            history_context = ""
            history_lines = []  # ✅ 提前初始化

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

                # 自动总结逻辑（保持不变）
                total_messages = len(conversations[session_id])
                if total_messages > 5 and total_messages % 5 == 0:
                    last_msg = conversations[session_id][-2]
                    if last_msg.get("role") != "system":
                        try:
                            summary_prompt = "请用 100 字以内总结前 5 轮对话的关键信息和上下文。"
                            summary_history = [
                                {"speaker": "User" if m["role"] == "user" else "Assistant",
                                 "content": m["content"]}
                                for m in selected_messages[-5:]
                            ]
                            summary_history.append({"speaker": "System", "content": summary_prompt})

                            summary = swarm.leader.generate_response(
                                summary_history,
                                round_num=0,
                                force_non_stream=True
                            )

                            summary_msg = {
                                "role": "system",
                                "content": f"📝 [历史总结] {summary}",
                                "timestamp": datetime.now().isoformat()
                            }
                            conversations[session_id].insert(-1, summary_msg)
                            selected_messages.append(summary_msg)

                            await websocket.send_json({
                                "type": "log",
                                "content": "📝 已生成历史总结"
                            })
                        except Exception as e:
                            print(f"⚠️ 生成历史总结失败: {e}")

                # 构建历史文本
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

            # ✅ 现在 history_lines 一定有定义
            if history_context:
                full_message = f"""=== 📚 对话历史（最近 {len(history_lines)} 轮）===
            {history_context}

            === 💬 当前问题 ===
            User: {message}"""
            else:
                full_message = message

            # ✅✅✅ 新增：自动解析附件内容 ✅✅✅
            if "📎 附件:" in message:
                try:
                    # 提取文件路径
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

                        # 自动读取文件内容
                        file_contents = []
                        MAX_PREVIEW_LENGTH = 10000  # ✅ 统一定义最大预览长度

                        for path in file_paths:
                            try:
                                path = path.strip()

                                # ===== PDF 处理 =====
                                if path.endswith('.pdf'):
                                    result = swarm.tool_registry['pdf_reader']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')
                                        truncated = False

                                        # ✅ 截断逻辑
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

                                # ===== TXT/MD 处理 =====
                                elif path.endswith(('.txt', '.md')):
                                    result = swarm.tool_registry['read_file']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')
                                        truncated = False

                                        # ✅ 截断逻辑
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

                                # ===== 图片处理 =====
                                elif path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                    file_contents.append(f"### 🖼️ {Path(path).name} (图片)\n路径: {path}")

                            except Exception as e:
                                file_contents.append(f"### ❌ {Path(path).name} 处理失败: {str(e)}")

                        # 将文件内容附加到完整消息（在历史上下文之后）
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

            # ==================== 4. 创建异步队列 ====================
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
                    except Exception as e:
                        print(f"⚠️ 流式发送失败: {e}")
                        break

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
                    except Exception as e:
                        print(f"⚠️ 日志发送失败: {e}")
                        break

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

                # ==================== 5. 执行 Swarm（带回调）====================
                answer = await loop.run_in_executor(
                    None,
                    lambda: swarm.solve(
                        full_message,  # ✅ 包含历史的完整消息
                        use_memory,
                        memory_key,
                        None,  # image_paths
                        force_complexity,
                        stream_callback=stream_callback,  # ✅ 传递流式回调
                        log_callback=log_callback  # ✅ 传递日志回调
                    )
                )

                # ==================== 6. 保存 AI 回复 ====================
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

                # ==================== 7. 性能监控 ====================
                elapsed = time.time() - start_time
                await websocket.send_json({
                    "type": "log",
                    "content": f"⏱️ 总耗时: {elapsed:.2f}秒"
                })

                await websocket.send_json({
                    "type": "end"
                })

                # 重置计时器
                start_time = time.time()

            except Exception as e:
                # ==================== 8. 异常处理 ====================
                print(f"❌ Swarm 执行失败: {e}")
                import traceback
                traceback.print_exc()

                # 停止后台任务
                await stream_queue.put(None)
                await log_queue.put(None)

                try:
                    await sender_task
                    await log_task
                except:
                    pass

                error_msg = f"❌ 执行失败: {str(e)[:200]}"

                # ✅ 检查连接状态再发送
                try:
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

                # 保存错误消息
                conversations[session_id].append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        print(f"🔌 WebSocket 断开连接 (session: {session_id if 'session_id' in locals() else 'unknown'})")
    except Exception as e:
        print(f"💥 WebSocket 致命错误: {e}")
        import traceback
        traceback.print_exc()

        try:
            await websocket.send_json({
                "type": "error",
                "content": f"❌ 连接错误: {str(e)[:100]}"
            })
        except:
            pass


# ====================== 新增：文件上传端点 ======================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    接收文件上传并保存到临时目录
    支持：PDF、TXT、MD、图片
    """
    try:
        # ✅ 新增：文件大小限制
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

        # 验证文件类型
        ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file_ext}"
            )

        # 保存到临时目录（确保路径安全）
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        # 生成安全文件名（防止路径注入）
        safe_filename = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
        file_path = upload_dir / safe_filename

        # ✅ 读取文件并检查大小
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（最大 {MAX_FILE_SIZE / (1024 * 1024):.0f}MB）"
            )

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(content)

        # 返回相对路径（供 Swarm 使用）
        return {
            "status": "ok",
            "filename": safe_filename,
            "path": str(file_path),
            "type": file_ext,
            "size": len(content)
        }

    except HTTPException:
        raise  # ✅ 直接抛出 HTTP 异常
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


# ====================== HTML 模板（完整版）======================
def get_html_template():
    """返回 HTML 模板（极简风格）"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MultiAgentSwarm WebUI</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(15,23,42,0.08);
            width: 100%;
            max-width: 100%;
            height: 100vh;
            display: flex;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }

        /* Sidebar */
        .sidebar {
            width: 300px;
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 24px;
            background: #ffffff;
            border-bottom: 1px solid #e2e8f0;
        }
        .sidebar-header h2 { font-size: 18px; font-weight: 600; color: #0f172a; }
        .sidebar-header p { font-size: 13px; color: #64748b; margin-top: 4px; }

        .session-list { flex: 1; overflow-y: auto; padding: 12px; }
        .session-item {
            padding: 14px 16px;
            margin-bottom: 8px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .session-item:hover, .session-item.active {
            border-color: #0ea5e9;
            background: #f0f9ff;
            transform: translateX(4px);
        }

        .new-session-btn {
            margin: 12px;
            padding: 14px;
            background: #0ea5e9;
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .new-session-btn:hover { background: #0284c8; transform: translateY(-1px); }

        /* Chat area */
        .chat-container { flex: 1; display: flex; flex-direction: column; }
        .chat-header {
            padding: 20px 24px;
            background: #ffffff;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-header h1 { font-size: 22px; font-weight: 600; color: #0f172a; }

        .header-buttons button {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-settings { background: #f1f5f9; color: #475569; }
        .btn-settings:hover { background: #e2e8f0; }
        .btn-export { background: #10b981; color: white; }
        .btn-export:hover { background: #059669; }
        .btn-clear { background: #ef4444; color: white; }
        .btn-clear:hover { background: #dc2626; }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            background: #fafafa;
            scroll-behavior: smooth;
        }

        .message {
            margin-bottom: 28px;
            display: flex;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 72%;
            padding: 16px 20px;
            border-radius: 18px;
            position: relative;
            word-wrap: break-word;
            line-height: 1.6;
        }
        .message.user .message-content {
            background: #0ea5e9;
            color: white;
        }
        .message.assistant .message-content {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 12px rgba(15,23,42,0.06);
        }

        /* Thinking box */
        .thinking-details {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(15,23,42,0.05);
        }
        .thinking-details summary {
            font-weight: 600;
            color: #334155;
            cursor: pointer;
            list-style: none;
        }

        /* Buttons & inputs */
        .action-btn {
            padding: 16px 24px;
            background: #0ea5e9;
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
        }
        .action-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(14,165,233,0.3); }

        .upload-btn { background: #10b981; }
        .upload-btn:hover:not(:disabled) { box-shadow: 0 6px 16px rgba(16,185,129,0.3); }

        #messageInput {
            flex: 1;
            padding: 16px 20px;
            border: 1.5px solid #cbd5e1;
            border-radius: 14px;
            font-size: 15px;
            resize: none;
            min-height: 72px;
            max-height: 200px;
        }
        #messageInput:focus {
            border-color: #0ea5e9;
            outline: none;
            box-shadow: 0 0 0 3px rgba(14,165,233,0.1);
        }

        /* Other styles remain the same (markdown, logs, etc.) */
        .agent-label {
            background: #0ea5e9;
            color: white;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            display: inline-block;
        }

        .message-actions button {
            padding: 6px 14px;
            font-size: 13px;
            background: #f1f5f9;
            border: none;
            border-radius: 6px;
            color: #64748b;
        }
        .message-actions button:hover { background: #e2e8f0; color: #334155; }

        /* ✅ 修改输入区域样式 */
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: stretch;  /* ✅ 改为 stretch，确保按钮等高 */
        }

        /* ✅ 新增：文件列表容器 */
        .file-list-container {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            min-height: 0;  /* ✅ 没有文件时不占空间 */
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

        /* ✅ 统一按钮样式（上传 + 发送）*/
        .action-btn {
            padding: 15px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 15px;  /* ✅ 改为 15px 统一风格 */
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;  /* ✅ 改为 600 更统一 */
            transition: all 0.3s;
            white-space: nowrap;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            min-width: 120px;  /* ✅ 确保按钮宽度一致 */
        }

        .action-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .action-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* ✅ 上传按钮特殊样式（可选：区分颜色）*/
        .upload-btn {
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        }

        .upload-btn:hover:not(:disabled) {
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
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
                <!-- ✅ 已上传文件列表 -->
                <div class="file-list-container" id="uploadedFiles"></div>
                
                <!-- ✅ 输入框 + 按钮（并排布局）-->
                <div class="input-wrapper">
                    <textarea 
                        id="messageInput" 
                        placeholder="输入你的问题...（Enter 换行, Ctrl+Enter 发送）" 
                        onkeydown="handleKeyDown(event)"
                        rows="3"
                    ></textarea>
                    
                    <!-- ✅ 上传按钮 -->
                    <button 
                        class="action-btn upload-btn" 
                        onclick="document.getElementById('fileInput').click()"
                        title="支持 PDF、TXT、MD、图片（最大 10MB）"
                    >
                        📎 上传附件
                    </button>
                    <input 
                        type="file" 
                        id="fileInput" 
                        accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.gif,.bmp" 
                        multiple 
                        style="display: none;" 
                        onchange="handleFileUpload(event)"
                    >
                    
                    <!-- ✅ 发送按钮 -->
                    <button id="sendBtn" class="action-btn" onclick="sendMessage()">
                        发送 🚀
                    </button>
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
        let currentStreamingDiv = null;
        let currentStreamingAgent = null;
        let thinkingDetailsElement = null;
        let uploadedFilePaths = [];  // ✅ 移到全局作用域

        // 配置 Marked.js
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
        
        // ==================== 新增：文件上传逻辑 ====================
        async function handleFileUpload(event) {
            const files = event.target.files;
            const uploadedFilesDiv = document.getElementById('uploadedFiles');
            
            for (let file of files) {
                // 1. 显示上传中状态
                const fileTag = document.createElement('div');
                fileTag.style.cssText = 'padding: 8px 12px; background: #e0e0e0; border-radius: 8px; display: flex; align-items: center; gap: 8px;';
                fileTag.innerHTML = '⏳ ' + file.name + ' (上传中...)';
                uploadedFilesDiv.appendChild(fileTag);
                
                try {
                    // 2. 上传文件
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'ok') {
                        // 3. 更新显示为成功状态
                        fileTag.innerHTML = '✅ ' + file.name + ' (' + formatBytes(data.size) + ')' +
                            '<button onclick="removeUploadedFile(\' + data.path + \', this.parentElement)" ' +
                            'style="background: none; border: none; cursor: pointer; font-size: 16px; margin-left: 8px;">❌</button>';
                        fileTag.style.background = '#d4edda';
                        
                        // 4. 保存路径
                        uploadedFilePaths.push(data.path);
                    } else {
                        throw new Error(data.detail || '上传失败');
                    }
                } catch (error) {
                    fileTag.innerHTML = '❌ ' + file.name + ' (失败)';
                    fileTag.style.background = '#f8d7da';
                    console.error('上传失败:', error);
                }
            }
            
            // 清空 input（允许重复上传同名文件）
            event.target.value = '';
        }
        
        function removeUploadedFile(path, element) {
            uploadedFilePaths = uploadedFilePaths.filter(function(p) { return p !== path; });
            element.remove();
        }
        
        function formatBytes(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
        
        // ==================== 修改：发送消息时附带文件 ====================
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
        
            if (!message && uploadedFilePaths.length === 0) return;
            if (isProcessing) return;
        
            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;
        
            // ✅ 修复：移除多余的转义符
            let fullMessage = message;
            if (uploadedFilePaths.length > 0) {
                const fileList = uploadedFilePaths.map(function(p) { return '- ' + p; }).join('\\n');
                fullMessage = message + '\\n\\n📎 附件:\\n' + fileList;
            }
        
            addMessage('user', fullMessage);
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
                message: fullMessage,
                session_id: currentSessionId,
                use_memory: false,
                memory_key: 'default',
                force_complexity: forceComplexity
            }));
        
            // 清空已上传文件列表
            uploadedFilePaths = [];
            document.getElementById('uploadedFiles').innerHTML = '';
        }
        
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
        
            // ✅ 双重滚动：先滚动思考日志容器，再滚动外部消息容器
            // 1. 滚动思考日志容器到底部
            logsDiv.scrollTop = logsDiv.scrollHeight;
            
            // 2. 滚动外部消息容器到底部（使用 setTimeout 确保 DOM 更新完成）
            setTimeout(function() {
                const messagesDiv = document.getElementById('messages');
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }, 0);
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

        // 前端 HTML 模板中的 addMessage 函数（增强版）
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;
        
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
        
            // ✅ 新增：解析附件并显示卡片
            if (role === 'user' && content.includes('📎 附件:')) {
                const parts = content.split('📎 附件:');
                const mainText = parts[0].trim();
                const attachmentSection = parts[1] || '';
                
                // 提取附件列表
                const attachmentLines = attachmentSection.split('\\n').filter(line => line.trim().startsWith('- '));
                
                // 构建消息内容
                let htmlContent = '';
                
                // 主文本
                if (mainText) {
                    htmlContent += '<div style="margin-bottom: 12px;">' + mainText + '</div>';
                }
                
                // 附件卡片（✅ 使用字符串拼接代替模板字符串）
                if (attachmentLines.length > 0) {
                    htmlContent += '<div style="margin-top: 12px;">';
                    attachmentLines.forEach(function(line) {
                        const path = line.replace('- ', '').trim();
                        const filename = path.split('/').pop();
                        const fileExt = filename.split('.').pop().toLowerCase();
                        
                        // 根据文件类型显示不同图标
                        let icon = '📎';
                        if (fileExt === 'pdf') icon = '📄';
                        else if (['txt', 'md'].includes(fileExt)) icon = '📝';
                        else if (['png', 'jpg', 'jpeg', 'gif', 'bmp'].includes(fileExt)) icon = '🖼️';
                        
                        htmlContent += '<div class="attachment-card" style="' +
                            'background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);' +
                            'padding: 10px 14px;' +
                            'border-radius: 10px;' +
                            'margin: 6px 0;' +
                            'border-left: 3px solid #667eea;' +
                            'display: flex;' +
                            'align-items: center;' +
                            'gap: 8px;' +
                            '">' +
                            '<span style="font-size: 20px;">' + icon + '</span>' +
                            '<strong style="color: #667eea;">' + filename + '</strong>' +
                            '</div>';
                    });
                    htmlContent += '</div>';
                }
                
                contentDiv.innerHTML = htmlContent;
            } 
            // ✅ 原有逻辑（非附件消息）
            else if (role === 'assistant') {
                contentDiv.innerHTML = marked.parse(content);
            } else {
                contentDiv.textContent = content;
            }
        
            // ✅ 统一添加操作按钮（用户和助手都有）
            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'message-actions';
            actionsDiv.innerHTML = '<button onclick="copyMessage(this)">📋 复制</button>' +
                '<button onclick="deleteMessage(this)">🗑️ 删除</button>';
            contentDiv.appendChild(actionsDiv);
        
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // ==================== 改进版复制函数（自动选中 + 干净复制）====================
        function copyMessage(btn) {
            const messageContent = btn.closest('.message-content');
            if (!messageContent) return;
        
            // 1. 准备干净文本（排除按钮、Agent标签，优先使用流式原始文本）
            const tempClone = messageContent.cloneNode(true);
            tempClone.querySelectorAll('.message-actions, .agent-label').forEach(el => el.remove());
            
            let textToCopy = '';
            const streamingDiv = messageContent.querySelector('.streaming-content');
            if (streamingDiv && streamingDiv.dataset.rawText) {
                textToCopy = streamingDiv.dataset.rawText;   // 流式时用原始累积文本（更准确）
            } else {
                textToCopy = tempClone.textContent.trim() || tempClone.innerText.trim();
            }
        
            // 2. 执行复制
            navigator.clipboard.writeText(textToCopy).then(() => {
                // 3. 自动选中消息内容（视觉高亮）
                selectAllMessageContent(messageContent);
        
                // 4. 按钮反馈
                const originalText = btn.textContent;
                btn.textContent = '✅ 已复制';
                btn.style.backgroundColor = '#4ade80';
                btn.style.color = '#fff';
        
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.style.backgroundColor = '';
                    btn.style.color = '';
                    window.getSelection().removeAllRanges();   // 自动取消选中
                }, 1800);
            }).catch(err => {
                console.error('复制失败:', err);
                const originalText = btn.textContent;
                btn.textContent = '❌ 失败';
                setTimeout(() => { btn.textContent = originalText; }, 1500);
            });
        }
        
        // ==================== 新增：视觉选中函数 ====================
        function selectAllMessageContent(contentElement) {
            const selection = window.getSelection();
            selection.removeAllRanges();
        
            const range = document.createRange();
            let target = contentElement.querySelector('.streaming-content') || contentElement;
        
            // 跳过 Agent 标签，只选中真正的内容部分
            const agentLabel = contentElement.querySelector('.agent-label');
            if (agentLabel && target === contentElement) {
                range.setStartAfter(agentLabel);
                range.setEnd(contentElement, contentElement.childNodes.length);
            } else {
                range.selectNodeContents(target);
            }
        
            selection.addRange(range);
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
                    div.onclick = function(e) { loadSession(session.id, e); };
                    listDiv.appendChild(div);
                });
            } catch (error) {
                console.error('加载会话失败:', error);
            }
        }

        async function loadSession(sessionId, e = null) {
            try {
                const response = await fetch('/api/session/' + sessionId);
                const data = await response.json();
        
                currentSessionId = sessionId;
                currentStreamingDiv = null;
                currentStreamingAgent = null;
                thinkingDetailsElement = null;
                document.getElementById('messages').innerHTML = '';
        
                // 渲染历史消息
                data.messages.forEach(function(msg) {
                    addMessage(msg.role, msg.content);
                });
        
                // 清除所有 active 状态
                document.querySelectorAll('.session-item').forEach(function(item) {
                    item.classList.remove('active');
                });
        
                // 安全设置当前会话为 active（兼容直接调用和点击调用）
                if (e && e.target) {
                    const clickedItem = e.target.closest('.session-item');
                    if (clickedItem) {
                        clickedItem.classList.add('active');
                    }
                } else if (currentSessionId) {
                    // 兜底：通过 ID 查找并激活（防止点击事件丢失）
                    document.querySelectorAll('.session-item').forEach(function(item) {
                        if (item.textContent.includes(currentSessionId.slice(0, 8))) {
                            item.classList.add('active');
                        }
                    });
                }
        
            } catch (error) {
                console.error('加载会话失败:', error);
                alert('加载会话失败，请刷新页面重试');
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
        
        document.addEventListener('DOMContentLoaded', async function() {
            await loadConfig();
            await loadSessions();
            createNewSession();
            initToggleSwitches();
        });
        
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