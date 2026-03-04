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
import yaml
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import smtplib
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header, decode_header
from email.utils import formataddr

# 导入你的 Swarm 系统
from multi_agent_swarm_v3 import MultiAgentSwarm

# 全局配置（启动时加载）
feishu_config = {}
feishu_client = None
lark_oapi = None
CreateMessageRequest = None
CreateMessageRequestBody = None
GetMessageResourceRequest = None


# ====================== 邮件编码辅助函数 ======================
def decode_email_payload(part):
    """增强版邮件内容解码（严格编码检测，支持中日韩）"""
    try:
        payload = part.get_payload(decode=True)
        if not payload:
            return ""

        # 🔥 优先尝试 charset 参数
        charset = part.get_content_charset()
        if charset:
            try:
                decoded = payload.decode(charset, errors='strict')
                # 验证是否包含有效字符
                if decoded and sum(c.isprintable() for c in decoded) / max(len(decoded), 1) > 0.5:
                    print(f"  ✅ 使用 Content-Type charset={charset} 解码成功")
                    return decoded
            except Exception as e:
                print(f"  ⚠️ charset={charset} 解码失败: {e}")

        # 🔥 智能编码检测（严格验证，新增日语支持）
        encodings = [
            # 中文编码
            ('gbk', lambda s: sum(1 for c in s if '\u4e00' <= c <= '\u9fff')),  # 中文字符数
            ('gb2312', lambda s: sum(1 for c in s if '\u4e00' <= c <= '\u9fff')),
            ('big5', lambda s: sum(1 for c in s if '\u4e00' <= c <= '\u9fff')),
            # 日语编码
            ('shift_jis', lambda s: sum(1 for c in s if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')),  # 平假名+片假名+汉字
            ('euc-jp', lambda s: sum(1 for c in s if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')),
            ('iso-2022-jp', lambda s: sum(1 for c in s if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')),
            # 通用编码
            ('utf-8', lambda s: sum(1 for c in s if c.isprintable())),
            ('iso-8859-1', lambda s: sum(1 for c in s if 32 <= ord(c) <= 126)),
            ('windows-1252', lambda s: sum(1 for c in s if c.isalpha()))
        ]

        best_result = None
        best_score = 0

        for encoding, scorer in encodings:
            try:
                decoded = payload.decode(encoding, errors='strict')

                # 计算质量得分
                if len(decoded) == 0:
                    continue

                # 1. 可打印字符占比
                printable_ratio = sum(c.isprintable() for c in decoded) / len(decoded)
                if printable_ratio < 0.3:
                    continue

                # 2. 特定语言字符得分
                special_score = scorer(decoded)

                # 3. 综合得分（特定字符权重更高）
                total_score = printable_ratio * 100 + special_score * 10

                if total_score > best_score:
                    best_score = total_score
                    best_result = (encoding, decoded)

            except Exception:
                continue

        if best_result:
            encoding, decoded = best_result
            print(f"  ✅ 智能检测使用 {encoding} 解码（得分: {best_score:.1f}）")
            return decoded

        # 🔥 最后尝试：忽略错误的 UTF-8
        print(f"  ⚠️ 所有严格解码失败，使用 UTF-8 忽略错误")
        return payload.decode('utf-8', errors='ignore')

    except Exception as e:
        print(f"  ❌ 解码异常: {e}")
        return ""

# ====================== FastAPI 应用 ======================
app = FastAPI(title="MultiAgentSwarm WebUI", version="3.1.0")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
print("✅ /uploads 下载服务已启用")

# ====================== 全局变量 ======================
# swarm: Optional[MultiAgentSwarm] = None # 单session
swarms: Dict[str, MultiAgentSwarm] = {}   # ← 每个 session 一个独立实例
global_swarm: Optional[MultiAgentSwarm] = None    # ← 新增：飞书、邮箱、配置API共用一个（最简方案）
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


# ====================== 工具函数 ======================
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """获取或创建会话 ID"""
    global conversations
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in conversations:
        conversations[session_id] = []
    return session_id


def get_or_create_swarm(session_id: str) -> MultiAgentSwarm:
    """懒加载 + 每个 session_id 一个完全独立的 Swarm 实例（核心 8 行）"""
    global swarms
    if session_id not in swarms:
        print(f"🚀 为会话 {session_id[:8]}... 创建独立 Swarm 实例")
        swarms[session_id] = MultiAgentSwarm(config_path="swarm_config.yaml")
    return swarms[session_id]


def update_config(config: ConfigUpdate):
    """动态更新 Swarm 配置（作用于全局实例，飞书/邮箱/配置页共用）"""
    global global_swarm
    if not global_swarm:
        raise HTTPException(status_code=500, detail="Swarm 未初始化（全局实例）")

    # ✅ 批量更新所有高级特性开关
    global_swarm.enable_adversarial_debate = config.adversarial_debate
    global_swarm.enable_meta_critic = config.meta_critic
    global_swarm.enable_task_decomposition = config.task_decomposition
    global_swarm.enable_knowledge_graph = config.knowledge_graph
    global_swarm.enable_adaptive_depth = config.adaptive_reflection
    global_swarm.intelligent_routing_enabled = config.intelligent_routing

    # ✅ 反思与路由参数
    global_swarm.max_reflection_rounds = config.max_rounds
    global_swarm.reflection_quality_threshold = config.quality_threshold
    global_swarm.stop_quality_threshold = config.stop_threshold
    global_swarm.quality_convergence_delta = config.convergence_delta
    global_swarm.force_complexity = config.force_complexity

    print(f"✅ 配置已全局更新: {config.dict()}")
    # 可选：同步更新所有活跃 session 的实例（极致一致性）
    for sid, swarm_inst in list(swarms.items()):
        swarm_inst.enable_adversarial_debate = config.adversarial_debate
        # ...（如果需要也可以同步其他参数，这里保持最小改动，只同步核心开关）


def sanitize_filename(original_name: str) -> str:
    """把中文、空格、特殊符号全部转为安全英文"""
    name = unicodedata.normalize('NFKD', original_name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    name = re.sub(r'_{2,}', '_', name)
    name = name.strip('_')
    if not name:
        name = "file"
    return name.lower()


def save_uploaded_content(content: bytes, original_name: str) -> Optional[str]:
    """保存附件并返回路径"""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    safe_stem = sanitize_filename(Path(original_name).stem)
    safe_filename = f"{uuid.uuid4().hex[:8]}_{safe_stem}{Path(original_name).suffix.lower()}"
    file_path = upload_dir / safe_filename
    try:
        with open(file_path, "wb") as f:
            f.write(content)
        return str(file_path)
    except Exception as e:
        print(f"❌ 附件保存失败: {e}")
        return None


# ====================== API 端点 ======================
@app.on_event("startup")
async def startup_event():
    global feishu_config
    with open("swarm_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    feishu_config = cfg.get("feishu", {})

    # 删除或注释掉这行（因为现在懒加载）：
    # init_swarm()

    # 飞书启动
    app_id = feishu_config.get("app_id", "").strip()
    app_secret = feishu_config.get("app_secret", "").strip()
    feishu_enable = feishu_config.get("enable", False)
    if app_id and app_secret and feishu_enable:
        print("🚀 飞书配置有效，正在启动长连接服务...")
        threading.Thread(
            target=start_feishu_long_connection,
            daemon=True,
            name="Feishu-Long-Connection"
        ).start()
    else:
        print("ℹ️  飞书功能已关闭")

    # 邮箱启动
    email_config = cfg.get("email", {})
    email_enable = email_config.get("enable", False)
    if email_config.get("imap_user") and email_enable:
        threading.Thread(
            target=start_email_poller,
            args=(email_config,),
            daemon=True,
            name="Email-Poller"
        ).start()
        print("🚀 邮件配置有效，正在启动邮箱连接服务...")

    # 初始化全局 Swarm（供飞书、邮箱、配置API使用）
    global global_swarm
    if not global_swarm:
        print("🚀 初始化全局 Swarm（飞书/邮箱专用）")
        global_swarm = MultiAgentSwarm(config_path="swarm_config.yaml")


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回主页"""
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
    """删除会话 + 清理独立 Swarm 实例"""
    if session_id in conversations:
        del conversations[session_id]
    if session_id in swarms:          # ← 新增
        del swarms[session_id]
        print(f"🗑️ 已释放会话 {session_id[:8]} 的 Swarm 实例")
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
    """获取当前 Swarm 配置（从全局实例读取）"""
    global global_swarm
    if not global_swarm:
        raise HTTPException(status_code=500, detail="Swarm 未初始化（全局实例）")

    return {
        "adversarial_debate": global_swarm.enable_adversarial_debate,
        "meta_critic": global_swarm.enable_meta_critic,
        "task_decomposition": global_swarm.enable_task_decomposition,
        "knowledge_graph": global_swarm.enable_knowledge_graph,
        "adaptive_reflection": global_swarm.enable_adaptive_depth,
        "intelligent_routing": global_swarm.intelligent_routing_enabled,
        "max_rounds": global_swarm.max_reflection_rounds,
        "quality_threshold": global_swarm.reflection_quality_threshold,
        "stop_threshold": global_swarm.stop_quality_threshold,
        "convergence_delta": global_swarm.quality_convergence_delta,
        "force_complexity": global_swarm.force_complexity,
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
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        MAX_FILE_SIZE = 10 * 1024 * 1024
        ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.gif', '.bmp'}

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

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
            "filename": safe_filename,
            "path": str(file_path),
            "type": file_ext,
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


# ====================== WebSocket 端点 ======================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点（支持真实流式输出 + 心跳保活 + 取消功能）"""
    global conversations
    await websocket.accept()
    start_time = time.time()

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

            if data.get("type") == "cancel":
                session_id = data.get("session_id")
                if session_id and session_id in swarms:
                    swarms[session_id].cancel_current_task()
                elif global_swarm:
                    global_swarm.cancel_current_task()
                await websocket.send_json({
                    "type": "log",
                    "content": "🛑 正在尝试取消任务..."
                })
                continue

            if data.get("type") in ("pong", "ping"):
                continue

            message = data.get("message", "").strip()
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "content": "❌ 消息不能为空"
                })
                continue

            # ==================== 关键修复：顺序 + 定义正确 ====================
            session_id = get_or_create_session(data.get("session_id"))
            use_memory = data.get("use_memory", False)
            memory_key = data.get("memory_key", "default")
            force_complexity = data.get("force_complexity")

            # 当前会话的独立 Swarm 实例（必须先定义！）
            current_swarm = get_or_create_swarm(session_id)
            current_swarm._reset_cancel_flag()
            # ============================================================

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

            # 构建历史上下文（简化版，可根据需要扩展）
            full_message = message

            # ============ 🔥 修复：补全附件处理代码 ============
            image_paths = []
            file_contents = []
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
                        MAX_PREVIEW_LENGTH = 10000
                        for path in file_paths:
                            try:
                                path = path.strip()
                                # 图片走多模态通道
                                if path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                    if os.path.exists(path):
                                        image_paths.append(path)
                                        file_contents.append(f"### 🖼️ {Path(path).name} (已作为多模态视觉输入)")
                                        await websocket.send_json({
                                            "type": "log",
                                            "content": f"📸 图片已加载: {Path(path).name}"
                                        })
                                    continue

                                # PDF / TXT / MD 处理
                                if path.endswith('.pdf'):
                                    result = current_swarm.tool_registry['pdf_reader']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')[:MAX_PREVIEW_LENGTH]
                                        file_contents.append(
                                            f"### 📄 {Path(path).name} (PDF)\n内容:\n{content}"
                                        )
                                elif path.endswith(('.txt', '.md')):
                                    result = current_swarm.tool_registry['read_file']['func'](file_path=path)
                                    if result.get('success'):
                                        content = result.get('content', '')[:MAX_PREVIEW_LENGTH]
                                        file_contents.append(
                                            f"### 📄 {Path(path).name}\n内容:\n{content}"
                                        )
                            except Exception as e:
                                file_contents.append(f"### ❌ {Path(path).name} 处理失败: {str(e)}")

                        if file_contents:
                            file_section = "\n\n=== 📄 附件内容 ===\n" + "\n\n".join(file_contents)
                            full_message = full_message + file_section
                            await websocket.send_json({
                                "type": "log",
                                "content": f"✅ 附件解析完成"
                            })
                except Exception as e:
                    print(f"⚠️ 附件解析失败: {e}")
                    await websocket.send_json({
                        "type": "log",
                        "content": f"⚠️ 附件解析失败"
                    })

            # 创建异步队列
            stream_queue = asyncio.Queue()
            log_queue = asyncio.Queue()

            async def stream_sender():
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
                    asyncio.run_coroutine_threadsafe(
                        stream_queue.put({
                            "type": "stream",
                            "agent": agent_name,
                            "content": content
                        }),
                        loop
                    )

                def log_callback(message: str):
                    asyncio.run_coroutine_threadsafe(
                        log_queue.put(message),
                        loop
                    )

                if current_swarm and current_swarm._check_cancellation():
                    await websocket.send_json({
                        "type": "error",
                        "content": "⏸️ 任务已被用户取消"
                    })
                    await websocket.send_json({"type": "end"})
                    return

                answer = await loop.run_in_executor(
                    None,
                    lambda: current_swarm.solve(  # ← 把 swarm 改成 current_swarm
                        full_message,
                        use_memory,
                        memory_key,
                        image_paths,
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
                    await websocket.send_json({"type": "end"})

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
                        await websocket.send_json({"type": "end"})
                except Exception as send_error:
                    print(f"⚠️ WebSocket 已关闭: {send_error}")
                    break

                conversations[session_id].append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        print(f"🔌 WebSocket 断开连接")
        if global_swarm:
            global_swarm.cancel_current_task()
    except Exception as e:
        print(f"💥 WebSocket 致命错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


# ====================== 飞书长连接 ======================
def start_feishu_long_connection():
    """启动飞书长连接"""
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass

    global feishu_client, lark_oapi, CreateMessageRequest, CreateMessageRequestBody, GetMessageResourceRequest

    try:
        import lark_oapi as lark_module
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest as Req,
            CreateMessageRequestBody as ReqBody,
            GetMessageResourceRequest as ResourceReq
        )
        lark_oapi = lark_module
        CreateMessageRequest = Req
        CreateMessageRequestBody = ReqBody
        GetMessageResourceRequest = ResourceReq
        print("✅ lark-oapi 导入成功")
    except Exception as e:
        print(f"❌ lark-oapi 导入失败: {e}")
        return

    try:
        feishu_client = lark_oapi.Client.builder().app_id(feishu_config.get("app_id")).app_secret(
            feishu_config.get("app_secret")).build()

        def handle_message(data: lark_oapi.im.v1.P2ImMessageReceiveV1):
            try:
                event = data.event
                msg = event.message
                chat_id = msg.chat_id
                message_id = msg.message_id
                if not message_id or not chat_id:
                    return

                # 添加👍反应
                try:
                    reaction_req = lark_oapi.BaseRequest.builder() \
                        .http_method(lark_oapi.HttpMethod.POST) \
                        .uri(f"/open-apis/im/v1/messages/{message_id}/reactions") \
                        .token_types({lark_oapi.AccessTokenType.TENANT}) \
                        .body({"reaction_type": {"emoji_type": "THUMBSUP"}}) \
                        .build()

                    reaction_resp = feishu_client.request(reaction_req)
                    if reaction_resp.success():
                        print(f"✅ 👍 已添加")
                except Exception as e:
                    print(f"⚠️ 👍 添加失败: {str(e)[:80]}")

                content_str = msg.content or "{}"
                content_json = json.loads(content_str)
                full_message = ""

                # 处理附件
                if msg.message_type in ("file", "image"):
                    file_key = None
                    original_name = "unknown_file"
                    file_type = msg.message_type if msg.message_type in ("image", "file") else "file"

                    if msg.message_type == "image":
                        file_key = content_json.get("image_key")
                        original_name = f"image_{int(time.time())}.jpg"
                    else:
                        file_key = content_json.get("file_key")
                        original_name = content_json.get("file_name", "file")

                    if not file_key:
                        return

                    print(f"📥 Feishu 收到附件: {original_name}")

                    request = GetMessageResourceRequest.builder() \
                        .message_id(message_id) \
                        .file_key(file_key) \
                        .type(file_type) \
                        .build()

                    response = feishu_client.im.v1.message_resource.get(request)

                    if not response.success():
                        print(f"❌ 下载附件失败: {response.msg}")
                        return

                    safe_stem = sanitize_filename(Path(original_name).stem)
                    safe_filename = f"{uuid.uuid4().hex[:8]}_{safe_stem}{Path(original_name).suffix.lower()}"
                    upload_dir = Path("uploads")
                    upload_dir.mkdir(exist_ok=True)
                    file_path = upload_dir / safe_filename

                    with open(file_path, "wb") as f:
                        f.write(response.raw.content)

                    print(f"✅ 附件已保存: {file_path}")
                    full_message = f"[来自飞书] 用户发送了附件\n\n📎 附件:\n- {file_path}"

                    reminder = CreateMessageRequest.builder() \
                        .receive_id_type("chat_id") \
                        .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .msg_type("text")
                        .content(json.dumps({"text": f"🤖 已收到附件《{original_name}》，正在处理..."}))
                        .build()
                    ).build()
                    feishu_client.im.v1.message.create(reminder)

                elif msg.message_type == "text":
                    content = content_json.get("text", "").strip()
                    if not content:
                        return
                    full_message = f"[来自飞书] {content}"
                else:
                    return

                # 判断是否需要回复
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

                print(f"📥 Feishu 收到消息: {full_message[:70]}...")

                # ✅ 调用 Swarm
                answer = global_swarm.solve(
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
                    print("✅ 已回复")
                else:
                    print(f"⚠️ 回复失败: {response.msg}")

            except Exception as e:
                print(f"❌ 处理飞书消息异常: {e}")
                import traceback
                traceback.print_exc()

        event_handler = lark_oapi.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(handle_message) \
            .build()

        cli = lark_oapi.ws.Client(
            app_id=feishu_config.get("app_id"),
            app_secret=feishu_config.get("app_secret"),
            event_handler=event_handler,
            log_level=lark_oapi.LogLevel.INFO
        )

        print("🚀 飞书长连接已启动")
        cli.start()

    except Exception as e:
        print(f"⚠️ 飞书长连接启动失败: {e}")


# ====================== 邮箱轮询（完整版）======================
def start_email_poller(config: dict):
    """邮箱轮询主函数"""
    interval = config.get("check_interval", 60)

    print(f"📧 Email Poller 已启动（间隔 {interval} 秒）")
    print(f"🎯 关键词: {config.get('email_keywords', [])}")

    def fetch_recent_unseen():
        """获取最近24小时未读邮件（终极增强版）"""
        try:
            mail = imaplib.IMAP4_SSL(config["imap_server"], 993)

            # IMAP ID
            try:
                tag = mail._new_tag()
                mail.send(f'{tag} ID ("name" "MultiAgentSwarm" "version" "3.1" "vendor" "xAI")\r\n'.encode())
                while True:
                    line = mail.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith(tag.decode()):
                        if 'OK' in line:
                            print("✅ IMAP ID 已发送")
                        break
                    elif line.startswith('*'):
                        continue
                    else:
                        break
            except Exception as e:
                print(f"⚠️ IMAP ID 失败: {e}")

            mail.login(config["imap_user"], config["imap_pass"])

            status, message_count = mail.select('INBOX')
            if status != 'OK':
                mail.logout()
                return []

            since_date = (datetime.now() - timedelta(days=1)).strftime("%d-%b-%Y")
            search_criteria = f'(UNSEEN SINCE {since_date})'

            status, messages = mail.search(None, search_criteria)
            if status != 'OK':
                mail.close()
                mail.logout()
                return []

            mail_ids = messages[0].split()
            if not mail_ids:
                mail.close()
                mail.logout()
                return []

            print(f"📬 找到 {len(mail_ids)} 封未读邮件")

            results = []
            keywords = config.get('email_keywords', [])

            for mail_id in mail_ids:
                try:
                    status, msg_data = mail.fetch(mail_id, '(RFC822)')
                    if status != 'OK':
                        continue

                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)

                    # 解析主题
                    subject = ''
                    subject_header = msg.get('Subject', '')
                    for part, encoding in decode_header(subject_header):
                        if isinstance(part, bytes):
                            subject += part.decode(encoding or 'utf-8', errors='ignore')
                        else:
                            subject += str(part)

                    from_header = msg.get('From', '')

                    # 关键词过滤
                    if keywords and not any(kw in subject or kw in from_header for kw in keywords):
                        print(f"⏭️  跳过: {subject[:30]}")
                        continue

                    # ============ 🔥 终极增强：正文提取 ============
                    body = ''
                    html_body = ''
                    attachments = []

                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition", ""))

                            # ✅ 提取纯文本（使用增强解码）
                            if content_type == "text/plain" and "attachment" not in content_disposition:
                                decoded = decode_email_payload(part)
                                if decoded:
                                    body += decoded

                            # ✅ 提取 HTML（使用增强解码）
                            elif content_type == "text/html" and "attachment" not in content_disposition:
                                decoded = decode_email_payload(part)
                                if decoded:
                                    html_body += decoded

                            # 提取附件
                            elif "attachment" in content_disposition:
                                filename = part.get_filename()
                                if filename:
                                    decoded_parts = decode_header(filename)
                                    filename = ''.join([
                                        p.decode(enc or 'utf-8', errors='ignore') if isinstance(p, bytes) else str(p)
                                        for p, enc in decoded_parts
                                    ])
                                    payload = part.get_payload(decode=True)
                                    if payload:
                                        path = save_uploaded_content(payload, filename)
                                        if path:
                                            attachments.append(path)
                    else:
                        # ✅ 单部分邮件（使用增强解码）
                        content_type = msg.get_content_type()
                        if content_type == "text/html":
                            html_body = decode_email_payload(msg)
                        else:
                            body = decode_email_payload(msg)

                    # ✅ 如果纯文本为空，使用 HTML 转文本
                    if not body.strip() or len(body) < 10:
                        if html_body:
                            print(f"  ⚠️  纯文本缺失，转换 HTML")

                            import html
                            html_body = html.unescape(html_body)

                            # 移除 <script> 和 <style>
                            html_body = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_body,
                                               flags=re.DOTALL | re.IGNORECASE)

                            # 移除所有 HTML 标签
                            body = re.sub(r'<[^>]+>', '', html_body)

                            # 清理多余空白
                            body = re.sub(r'\s+', ' ', body).strip()

                            # 智能判断是否为有效内容
                            if body:
                                visible_chars = sum(1 for c in body if c.isprintable() and not c.isspace())

                                if visible_chars < 5:
                                    body = "[邮件正文无法解析 - 可能是纯图片或格式化邮件]"
                                    print(f"  ⚠️  正文无效（可见字符: {visible_chars}）")
                                else:
                                    meaningful_chars = sum(1 for c in body if c.isalnum() or c in '，。！？、；：""''（）《》【】…—')
                                    if meaningful_chars < 3:
                                        body = "[邮件正文无法解析 - 可能是纯图片或格式化邮件]"
                                        print(f"  ⚠️  正文无效（有意义字符: {meaningful_chars}）")
                                    else:
                                        print(f"  ✅ HTML 转文本成功（{len(body)} 字符，{meaningful_chars} 有意义字符）")

                    # 标记已读
                    mail.store(mail_id, '+FLAGS', '\\Seen')

                    results.append({
                        'subject': subject,
                        'from': from_header,
                        'date': msg.get('Date', ''),
                        'body': body.strip() if body.strip() else "[空邮件]",
                        'attachments': attachments
                    })

                    print(f"✅ 获取: {subject[:50]}")
                    print(f"   正文长度: {len(body)} 字符")

                except Exception as e:
                    print(f"⚠️ 处理失败: {e}")
                    continue

            mail.close()
            mail.logout()
            return results

        except Exception as e:
            print(f"❌ IMAP 失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def send_reply(to: str, subject: str, body: str, original_date: str = "", attachments: List[str] = None):
        """发送回复（支持附件）"""
        try:
            from email.mime.base import MIMEBase
            from email import encoders

            msg = MIMEMultipart()
            msg['From'] = formataddr((
                Header("MultiAgentSwarm AI", 'utf-8').encode(),
                config['imap_user']
            ))
            msg['To'] = to
            msg['Subject'] = Header(subject, 'utf-8')

            full_body = f"""您好！

    以下是 AI 助手的回复：

    {body}

    ---
    本邮件由 MultiAgentSwarm AI 系统自动生成
    原始邮件时间: {original_date}
    """

            msg.attach(MIMEText(full_body, 'plain', 'utf-8'))

            if attachments:
                for file_path in attachments:
                    try:
                        if not os.path.exists(file_path):
                            continue

                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)

                            filename = os.path.basename(file_path)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename="{Header(filename, "utf-8").encode()}"'
                            )
                            msg.attach(part)
                            print(f"  📎 已附加: {filename}")

                    except Exception as e:
                        print(f"  ⚠️ 附件失败 {file_path}: {e}")

            port = config.get("smtp_port", 465)
            if port == 465:
                server = smtplib.SMTP_SSL(config["smtp_server"], port)
            else:
                server = smtplib.SMTP(config["smtp_server"], port)
                server.starttls()

            server.login(config["imap_user"], config["imap_pass"])
            server.send_message(msg)
            server.quit()

            print(f"✅ 回复已发送至: {to}")
            return True

        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

    # 主循环
    while True:
        try:
            print(f"\n{'=' * 60}")
            print(f"🔄 检查邮件 ({datetime.now().strftime('%H:%M:%S')})")
            print(f"{'=' * 60}\n")

            emails = fetch_recent_unseen()

            if not emails:
                print("📭 无新邮件")
            else:
                print(f"📬 处理 {len(emails)} 封邮件\n")

                for idx, email_data in enumerate(emails, 1):
                    try:
                        print(f"{'─' * 60}")
                        print(f"📧 [{idx}/{len(emails)}] {email_data['subject'][:40]}")
                        print(f"👤 {email_data['from']}")

                        question = f"[邮件主题]\n{email_data['subject']}\n\n[邮件内容]\n{email_data['body']}"

                        if email_data['attachments']:
                            question += "\n\n[附件]\n" + "\n".join(f"- {a}" for a in email_data['attachments'])
                            print(f"📎 {len(email_data['attachments'])} 个附件")

                        print("🤖 AI 处理中...")

                        # ✅ 调用 Swarm
                        if global_swarm:
                            answer = global_swarm.solve(
                                question,
                                use_memory=True,
                                memory_key="email_auto"
                            )
                        else:
                            answer = "❌ Swarm 未初始化"

                        print(f"✅ AI 完成（{len(answer)} 字符）")

                        # 提取发件人邮箱
                        from_email = re.findall(r'[\w\.-]+@[\w\.-]+', email_data['from'])

                        if from_email:
                            # 🔥 检查 AI 回复中是否提到需要发送附件
                            attachments_to_send = []

                            # 如果 AI 生成了新文件（如报告、图表等），将路径添加到这里
                            # 例如：从 answer 中提取 /uploads/xxx.pdf 这样的路径
                            upload_pattern = r'uploads/[\w\-\.]+\.\w+'
                            found_files = re.findall(upload_pattern, answer)

                            seen = set()
                            MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB

                            for file_path in found_files:
                                if os.path.exists(file_path):
                                    file_size = os.path.getsize(file_path)
                                    if file_size > MAX_ATTACHMENT_SIZE:
                                        print(f"  ⚠️ 附件过大，跳过: {file_path} ({file_size / 1024 / 1024:.1f}MB)")
                                        continue
                                    if file_path not in seen:
                                        attachments_to_send.append(file_path)
                                        seen.add(file_path)

                            send_reply(
                                to=from_email[0],
                                subject=f"Re: {email_data['subject']}",
                                body=answer,
                                original_date=email_data['date'],
                                attachments=attachments_to_send  # 🔥 新增参数
                            )

                        if idx < len(emails):
                            time.sleep(2)

                    except Exception as e:
                        print(f"❌ 处理失败: {e}")
                        continue

        except Exception as e:
            print(f"⚠️ 轮询异常: {e}")

        time.sleep(interval)


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