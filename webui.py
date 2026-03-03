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
import time
import re
import unicodedata
import hmac
import hashlib
import base64
import yaml
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 飞书 SDK 延迟加载（未安装时不崩溃）
lark_oapi = None
CreateMessageRequest = None
CreateMessageRequestBody = None
GetMessageResourceRequest = None

# 导入你的 Swarm 系统
from multi_agent_swarm_v3 import MultiAgentSwarm

# 全局配置（启动时加载）
feishu_config = {}
feishu_client = None

# ====================== FastAPI 应用 ======================
app = FastAPI(title="MultiAgentSwarm WebUI", version="3.1.0")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")
# ====================== ✨ 输出文件下载支持（用户需求核心） ======================
# 让 uploads/ 目录下的修改后文件可直接下载（兼容 WebUI + 飞书）
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
print("✅ /uploads 下载服务已启用（修改后的 Excel/报告可直接点击下载）")

# ====================== 全局变量（必须在这里定义！） ======================
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
    global conversations  # ← 新增这一行（关键！）
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
    global feishu_config
    with open("swarm_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    feishu_config = cfg.get("feishu", {})

    init_swarm()

    # === 严格按要求：appid 和 app_secret 都非空才启用飞书 ===
    app_id = feishu_config.get("app_id", "").strip()
    app_secret = feishu_config.get("app_secret", "").strip()

    if app_id and app_secret:
        print("🚀 飞书配置有效，正在启动长连接服务...")
        threading.Thread(
            target=start_feishu_long_connection,
            daemon=True,
            name="Feishu-Long-Connection"
        ).start()
    else:
        print("ℹ️  飞书功能已关闭（swarm_config.yaml 中 app_id 或 app_secret 未配置）")


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
    """WebSocket 端点（支持真实流式输出 + 心跳保活 + 取消功能）"""
    global conversations  # ← 新增这一行（关键！）
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

            # ✅ 新增：处理取消请求
            if data.get("type") == "cancel":
                if swarm:
                    swarm.cancel_current_task()
                    await websocket.send_json({
                        "type": "log",
                        "content": "🛑 正在尝试取消任务..."
                    })
                    await websocket.send_json({
                        "type": "stream",
                        "agent": "System",
                        "content": "\n\n⏸️ **正在取消任务，请稍候...**\n\n"
                    })
                continue

            # 🔥 忽略心跳 pong
            if data.get("type") in ("pong", "ping"):
                continue

            message = data.get("message", "").strip()
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "content": "❌ 消息不能为空"
                })
                continue

            # ====================== 【关键修复：每条新消息都重置取消标志】 ======================
            if swarm:
                swarm._reset_cancel_flag()
            # ==================================================================================

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

            # 构建历史上下文
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

            # ====================== 【附件处理 - 已修正，独立处理】 ======================
            image_paths = []  # ← 必须在这里初始化（关键修复）
            file_contents = []
            if "📎 附件:" in message:  # 用原始 message 判断
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
                        MAX_PREVIEW_LENGTH = 10000
                        for path in file_paths:
                            try:
                                path = path.strip()
                                # === 图片走多模态通道 ===
                                if path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                    if os.path.exists(path):
                                        image_paths.append(path)
                                        file_contents.append(f"### 🖼️ {Path(path).name} (已作为多模态视觉输入)")
                                        await websocket.send_json({
                                            "type": "log",
                                            "content": f"📸 图片已加载为多模态输入: {Path(path).name}"
                                        })
                                    continue

                                # PDF / TXT / MD 处理（保持不变）
                                if path.endswith('.pdf'):
                                    result = swarm.tool_registry['pdf_reader']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')[:MAX_PREVIEW_LENGTH]
                                        file_contents.append(
                                            f"### 📄 {Path(path).name} (PDF)\n【系统指令：以下是附件完整解析内容，请直接基于此内容分析】\n内容:\n{content}"
                                        )
                                elif path.endswith(('.txt', '.md')):
                                    result = swarm.tool_registry['read_file']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')[:MAX_PREVIEW_LENGTH]
                                        file_contents.append(
                                            f"### 📄 {Path(path).name}\n【系统指令：以下是附件完整解析内容，请直接基于此内容分析】\n内容:\n{content}"
                                        )
                            except Exception as e:
                                file_contents.append(f"### ❌ {Path(path).name} 处理失败: {str(e)}")

                        if file_contents:
                            file_section = "\n\n=== 📄 附件内容 ===\n" + "\n\n".join(file_contents)
                            full_message = full_message + file_section
                            await websocket.send_json({
                                "type": "log",
                                "content": f"✅ 附件解析完成（{len(image_paths)} 张图片 + {len(file_contents) - len(image_paths)} 个文本文件）"
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

                # 新增：
                if swarm and swarm._check_cancellation():
                    await websocket.send_json({
                        "type": "error",
                        "content": "⏸️ 任务已被用户取消"
                    })
                    await websocket.send_json({"type": "end"})
                    return  # ← 直接返回，不执行 solve
                answer = await loop.run_in_executor(
                    None,
                    lambda: swarm.solve(
                        full_message,
                        use_memory,
                        memory_key,
                        image_paths,  # ← 关键修改（原来是 None）
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
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({
                        "type": "log",
                        "content": f"⏱️ 总耗时: {elapsed:.2f}秒"
                    })
                    await websocket.send_json({
                        "type": "end"
                    })

                start_time = time.time()

            except asyncio.CancelledError:
                await websocket.send_json({
                    "type": "error",
                    "content": "⏸️ 任务已被用户取消"
                })
                await websocket.send_json({"type": "end"})
                return

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
        # ✅ 连接断开时尝试取消任务
        if swarm:
            swarm.cancel_current_task()
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


# ====================== ✨ 飞书官方 SDK 长连接模式 ✨ ======================
def start_feishu_long_connection():
    """启动飞书长连接（仅在配置有效时被调用）"""
    # nest_asyncio 保护
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass

    global feishu_client, lark_oapi, CreateMessageRequest, CreateMessageRequestBody, GetMessageResourceRequest

    # 2. SDK 导入检查（使用正确类名）
    try:
        import lark_oapi as lark_module
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest as Req,
            CreateMessageRequestBody as ReqBody,
            GetMessageResourceRequest as ResourceReq   # ← 关键修复：正确类名
        )
        lark_oapi = lark_module
        CreateMessageRequest = Req
        CreateMessageRequestBody = ReqBody
        GetMessageResourceRequest = ResourceReq
        print("✅ lark-oapi 导入成功")
    except Exception as e:
        print(f"❌ lark-oapi 导入失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 启动长连接
    try:
        feishu_client = lark_oapi.Client.builder().app_id(feishu_config.get("app_id")).app_secret(feishu_config.get("app_secret")).build()

        def handle_message(data: lark_oapi.im.v1.P2ImMessageReceiveV1):
            try:
                event = data.event
                msg = event.message
                chat_id = msg.chat_id
                message_id = msg.message_id
                if not message_id or not chat_id:
                    return

                # ====================== 【新增】飞书收到消息后立即在原消息下方打 👍 ======================
                # 目的：实现图片中「Nanobot-MiniPC 👍」效果，非回复消息，立即反馈
                try:
                    reaction_req = lark_oapi.BaseRequest.builder() \
                        .http_method(lark_oapi.HttpMethod.POST) \
                        .uri(f"/open-apis/im/v1/messages/{message_id}/reactions") \
                        .token_types({lark_oapi.AccessTokenType.TENANT}) \
                        .body({
                        "reaction_type": {
                            "emoji_type": "THUMBSUP"  # 👍 官方正确代码（THUMBSUP = 点赞）
                        }
                    }) \
                        .build()

                    reaction_resp = feishu_client.request(reaction_req)
                    if reaction_resp.success():
                        print(f"✅ 👍 已为消息 {message_id[:12]}... 添加收到确认（原消息下方可见）")
                    else:
                        print(f"⚠️ 👍 添加失败（不影响处理）: {reaction_resp.msg}")
                except Exception as e:
                    print(f"⚠️ 添加 👍 反应异常（已隔离，不阻塞主流程）: {str(e)[:80]}")
                # ====================== 新增结束 ======================

                content_str = msg.content or "{}"
                content_json = json.loads(content_str)

                full_message = ""
                content = ""

                # ==================== 处理附件（已修复 field validation） ====================
                if msg.message_type in ("file", "image"):
                    file_key = None
                    original_name = "unknown_file"
                    file_type = "file"  # 默认

                    if msg.message_type == "image":
                        file_key = content_json.get("image_key")
                        original_name = f"image_{int(time.time())}.jpg"
                        file_type = "image"
                    else:
                        file_key = content_json.get("file_key")
                        original_name = content_json.get("file_name", "file")
                        # file_type = "file"
                        file_type = msg.message_type if msg.message_type in ("image", "file") else "file"

                    if not file_key:
                        return

                    print(f"📥 Feishu 收到附件: {original_name} (type={file_type})")

                    # === 关键修复：必须传入 type 参数 ===
                    request = GetMessageResourceRequest.builder() \
                        .message_id(message_id) \
                        .file_key(file_key) \
                        .type(file_type) \
                        .build()

                    response = feishu_client.im.v1.message_resource.get(request)

                    if not response.success():
                        print(f"❌ 下载附件失败: {response.msg}")
                        # 下载失败也友好回复用户
                        reminder = CreateMessageRequest.builder() \
                            .receive_id_type("chat_id") \
                            .request_body(
                            CreateMessageRequestBody.builder()
                            .receive_id(chat_id)
                            .msg_type("text")
                            .content(
                                json.dumps({"text": f"⚠️ 已收到您的附件 {original_name}，但下载失败，请尝试重新发送"}))
                            .build()
                        ).build()
                        feishu_client.im.v1.message.create(reminder)
                        return

                    # 保存文件（复用你的净化函数）
                    safe_stem = sanitize_filename(Path(original_name).stem)
                    safe_filename = f"{uuid.uuid4().hex[:8]}_{safe_stem}{Path(original_name).suffix.lower()}"
                    upload_dir = Path("uploads")
                    upload_dir.mkdir(exist_ok=True)
                    file_path = upload_dir / safe_filename

                    with open(file_path, "wb") as f:
                        f.write(response.raw.content)

                    print(f"✅ 附件已保存: {file_path} ({len(response.raw.content) / 1024:.1f} KB)")
                    full_message = f"[来自飞书] 用户发送了附件\n\n📎 附件:\n- {file_path}"

                    # 成功提醒
                    reminder = CreateMessageRequest.builder() \
                        .receive_id_type("chat_id") \
                        .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .msg_type("text")
                        .content(json.dumps({"text": f"🤖 已成功收到附件《{original_name}》，正在分析处理中..."}))
                        .build()
                    ).build()
                    feishu_client.im.v1.message.create(reminder)

                # ==================== 处理纯文本 ====================
                elif msg.message_type == "text":
                    content = content_json.get("text", "").strip()
                    if not content:
                        return
                    full_message = f"[来自飞书] {content}"
                else:
                    return

                # ==================== 是否需要回复 ====================
                should_reply = msg.chat_type == "p2p"
                if msg.chat_type == "group" and hasattr(event, "mentions") and event.mentions:
                    for m in event.mentions:
                        if getattr(m.id, "open_id", None):
                            should_reply = True
                            if msg.message_type == "text":
                                content = re.sub(r'@\S+\s*', '', content).strip()
                                full_message = f"[来自飞书] {content}"
                            break
                if not should_reply:
                    return

                print(f"📥 Feishu 长连接收到消息: {full_message[:70]}...")

                # Swarm 处理（现在附件路径会正确传入）
                answer = swarm.solve(
                    full_message,
                    use_memory=True,
                    memory_key="feishu_long"
                )

                request = CreateMessageRequest.builder() \
                    .receive_id_type("chat_id") \
                    .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(json.dumps({"text": answer}))
                    .build()
                ).build()

                response = feishu_client.im.v1.message.create(request)
                if response.success():
                    print("✅ 已自动回复处理结果")
                else:
                    print(f"⚠️ 回复失败: {response.msg}")

            except Exception as e:
                print(f"❌ 处理飞书消息异常: {e}")
                import traceback
                traceback.print_exc()

        # 注册并启动
        event_handler = lark_oapi.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(handle_message) \
            .build()

        cli = lark_oapi.ws.Client(
            app_id=feishu_config.get("app_id"),
            app_secret=feishu_config.get("app_secret"),
            event_handler=event_handler,
            log_level=lark_oapi.LogLevel.INFO
        )

        print("🚀 飞书官方长连接客户端已启动（WebSocket 已建立）")
        cli.start()

    except Exception as e:
        print(f"⚠️ 飞书长连接启动失败: {e}")


async def send_log(msg: str):
    """实时推送到 WebUI 日志区"""
    print(f"🪝 {msg}")
    # 如果需要推送到前端，可在此扩展

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