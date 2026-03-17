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

import markdown   # ← 新增这一行

from multi_agent_swarm_v5_1 import MultiAgentSwarm   # ← 推荐改成这个名字
print("🚀 已加载 v5 Swarm（含 Verifier + KG主动进化）")

# 全局配置（启动时加载）
feishu_config = {}
feishu_client = None
lark_oapi = None
CreateMessageRequest = None
CreateMessageRequestBody = None
GetMessageResourceRequest = None

feishu_swarm: Optional[MultiAgentSwarm] = None   # ← 新增：飞书专用
email_swarm: Optional[MultiAgentSwarm] = None    # ← 新增：邮件专用
email_enabled: bool = False
feishu_enabled: bool = False


# ====================== 邮件编码辅助函数 ======================
def decode_email_payload(part):
    """增强版邮件内容解码（严格编码检测，支持中日韩）"""
    try:
        payload = part.get_payload(decode=True)
        if not payload:
            return ""

        # 🔥 优先尝试 charset 参数（方法3：直接信任邮件头）
        charset = part.get_content_charset()
        if charset:
            try:
                decoded = payload.decode(charset, errors='strict')
                # ✅ 只要有非空内容就信任（关键修改）
                if decoded and len(decoded.strip()) > 0:
                    print(f"  ✅ 使用 Content-Type charset={charset} 解码成功")
                    return decoded  # 直接返回，不再做后续验证
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
# ====================== 管理员指令系统 ======================
SWARM_ADMIN_PASSWORD = os.getenv("SWARM_ADMIN_PASSWORD", "1234567890")
# email_enabled = True      # 邮件轮询开关
# feishu_enabled = True     # 飞书长连接开关


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
    """每个会话创建一个独立的 v5 Swarm 实例（v4 已移除）"""
    global swarms
    if session_id not in swarms:
        print(f"🚀 会话 {session_id[:8]}... 创建 v5 Swarm 实例")
        swarms[session_id] = MultiAgentSwarm(config_path="swarm_config.yaml")
        swarms[session_id]._swarm_version = "v5"  # 调试标记，可保留
    return swarms[session_id]


def update_config(config: ConfigUpdate):
    """动态更新 Swarm 配置（作用于全局实例，飞书/邮箱/配置页共用）"""
    global global_swarm
    if not global_swarm:
        raise HTTPException(status_code=500, detail="Swarm 未初始化（全局实例）")

    # ✅ 批量更新全局实例
    global_swarm.enable_adversarial_debate = config.adversarial_debate
    global_swarm.enable_meta_critic = config.meta_critic
    global_swarm.enable_task_decomposition = config.task_decomposition
    global_swarm.enable_knowledge_graph = config.knowledge_graph
    global_swarm.enable_adaptive_depth = config.adaptive_reflection
    global_swarm.intelligent_routing_enabled = config.intelligent_routing

    global_swarm.max_reflection_rounds = config.max_rounds
    global_swarm.reflection_quality_threshold = config.quality_threshold
    global_swarm.stop_quality_threshold = config.stop_threshold
    global_swarm.quality_convergence_delta = config.convergence_delta
    global_swarm.force_complexity = config.force_complexity

    print(f"✅ 配置已全局更新: {config.dict()}")

    # ✅ 同步所有 Web 会话实例（你已经加了，非常好）
    for sid, swarm_inst in list(swarms.items()):
        swarm_inst.enable_adversarial_debate = config.adversarial_debate
        swarm_inst.enable_meta_critic = config.meta_critic
        swarm_inst.enable_task_decomposition = config.task_decomposition
        swarm_inst.enable_knowledge_graph = config.knowledge_graph
        swarm_inst.enable_adaptive_depth = config.adaptive_reflection
        swarm_inst.intelligent_routing_enabled = config.intelligent_routing
        swarm_inst.max_reflection_rounds = config.max_rounds
        swarm_inst.reflection_quality_threshold = config.quality_threshold
        swarm_inst.stop_quality_threshold = config.stop_threshold
        swarm_inst.quality_convergence_delta = config.convergence_delta
        swarm_inst.force_complexity = config.force_complexity

    # 🔥 新增：同步飞书和邮件专用实例（关键！）
    for swarm_inst in (feishu_swarm, email_swarm):
        if swarm_inst:
            swarm_inst.enable_adversarial_debate = config.adversarial_debate
            swarm_inst.enable_meta_critic = config.meta_critic
            swarm_inst.enable_task_decomposition = config.task_decomposition
            swarm_inst.enable_knowledge_graph = config.knowledge_graph
            swarm_inst.enable_adaptive_depth = config.adaptive_reflection
            swarm_inst.intelligent_routing_enabled = config.intelligent_routing
            swarm_inst.max_reflection_rounds = config.max_rounds
            swarm_inst.reflection_quality_threshold = config.quality_threshold
            swarm_inst.stop_quality_threshold = config.stop_threshold
            swarm_inst.quality_convergence_delta = config.convergence_delta
            swarm_inst.force_complexity = config.force_complexity


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

def handle_admin_command(message: str, session_id: Optional[str] = None) -> Optional[str]:
    """对话内管理员指令系统（完整中英双语支持）"""
    if not message:
        return None

    # 🔥 必须放在最开头
    global email_enabled
    global feishu_enabled

    msg = message.lower().strip()

    # 提取口令
    pw_match = re.search(r'(?:口令|password)[:=]\s*(\S+)', message, re.IGNORECASE)
    provided_pw = pw_match.group(1) if pw_match else None

    # ==================== /help（中英双语 + 丰富英文触发词） ====================
    if any(k in msg for k in [
        "/help", "help", "commands", "admin help", "指令帮助",
        "管理员命令", "命令列表", "help me", "what commands"
    ]):
        pw_status = "**❌ 已禁用（空密码）**" if not SWARM_ADMIN_PASSWORD else "✅ 已启用"
        help_text = f"""**🛠 MultiAgentSwarm Admin Commands（管理员指令）**

**Password Status**: {pw_status}

**Basic Commands（基础指令 - 无需口令）**
• `/help` 或 `help` 或 `commands`          → 显示此帮助
• `系统状态` / `status` / `system status`  → 查看实时状态
• `开启邮件` / `enable email`              → 开启邮件轮询
• `关闭邮件` / `disable email`             → 关闭邮件轮询
• `开启飞书` / `enable feishu`             → 开启飞书
• `关闭飞书` / `disable feishu`            → 关闭飞书
• `清理上传文件` / `clean uploads`         → 清理附件文件夹

**Dangerous Commands（危险指令 - 需要口令）**
• `重置记忆 口令:xxx` / `reset memory password:xxx` → 清空所有记忆
• `关闭程序 口令:xxx` / `shutdown password:xxx`     → 安全关闭整个服务

**Password Setting**: Environment variable `SWARM_ADMIN_PASSWORD`（留空 = 禁用口令）

Just type any command — system auto-detects 中英！"""
        return help_text

    # 关闭程序（已修复为 os._exit）
    if any(k in msg for k in ["关闭程序", "关机", "shutdown", "退出程序", "exit", "quit"]):
        if provided_pw == SWARM_ADMIN_PASSWORD or not SWARM_ADMIN_PASSWORD:
            print("🛑 收到管理员关闭指令 → 正在安全退出...")
            import os
            os._exit(0)
        else:
            return "❌ 口令错误！关闭程序需要正确口令。"

    # 重置记忆（支持英文）
    if any(k in msg for k in ["重置记忆", "清空记忆", "reset memory", "clear memory", "reset", "clear all"]):
        if (provided_pw == SWARM_ADMIN_PASSWORD or not SWARM_ADMIN_PASSWORD) and session_id:
            if session_id in conversations:
                conversations[session_id].clear()
            if session_id in swarms:
                swarm = swarms[session_id]
                if hasattr(swarm, "reset_memory"):
                    swarm.reset_memory()
                try:
                    if hasattr(swarm, 'primal_memory'): swarm.primal_memory.clear()
                    if hasattr(swarm, 'vector_memory'): swarm.vector_memory.clear()
                    if hasattr(swarm, 'knowledge_graph'): swarm.knowledge_graph.clear()
                except:
                    pass
            return "✅ 当前会话 + 所有记忆层已完全重置！（All memory cleared!）"

    # 邮件开关（支持英文）
    if any(k in msg for k in ["开启邮件", "打开邮件", "enable email", "start email"]):
        email_enabled = True
        return "✅ 邮件功能已开启（Email polling enabled）"
    if any(k in msg for k in ["关闭邮件", "停止邮件", "disable email", "stop email"]):
        email_enabled = False
        return "✅ 邮件功能已关闭（Email polling disabled）"

    # 飞书开关（支持英文）
    if any(k in msg for k in ["开启飞书", "打开飞书", "enable feishu", "start feishu"]):
        feishu_enabled = True
        return "✅ 飞书功能已开启（Feishu enabled）"
    if any(k in msg for k in ["关闭飞书", "停止飞书", "disable feishu", "stop feishu"]):
        feishu_enabled = False
        return "✅ 飞书功能已关闭（Feishu disabled）"

    # 系统状态（支持英文）
    if any(k in msg for k in ["系统状态", "status", "system status", "查看状态"]):
        return (f"📊 **MultiAgentSwarm Status**\n"
                f"• Version: v5\n"
                f"• Email Polling: {'✅ Enabled' if email_enabled else '❌ Disabled'}\n"
                f"• Feishu: {'✅ Enabled' if feishu_enabled else '❌ Disabled'}\n"
                f"• Sessions: {len(swarms)}\n"
                f"• Password: {'Disabled (empty)' if not SWARM_ADMIN_PASSWORD else 'Enabled'}")

    # 清理上传文件
    if any(k in msg for k in ["清理上传文件", "清理附件", "clean uploads", "clear files"]):
        import shutil
        upload_dir = Path("uploads")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            upload_dir.mkdir()
        return "🧹 所有上传附件已清理完成！（All uploads cleared!）"

    return None


# ====================== API 端点 ======================
@app.on_event("startup")
async def startup_event():
    global feishu_config
    global email_enabled, feishu_enabled
    with open("swarm_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 🔥 新增：解析环境变量占位符（最小改动点）
    def resolve_env_placeholders(config_dict):
        """递归替换 {env:VAR_NAME} 为环境变量值（支持所有字符串字段）"""
        pattern = re.compile(r'^\{env:(\w+)\}$')  # 严格匹配 {env:XXX}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                resolve_env_placeholders(value)  # 递归子字典
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        resolve_env_placeholders(item)
            elif isinstance(value, str):
                match = pattern.match(value)
                if match:
                    env_var = match.group(1)
                    env_value = os.environ.get(env_var, '')  # 默认空，避免崩溃
                    config_dict[key] = env_value
                    if not env_value:
                        print(f"⚠️ 环境变量 {env_var} 未设置，使用空值（Secret可能无效）")
                    else:
                        print(f"✅ 已从环境变量加载 {env_var}")

    resolve_env_placeholders(cfg)  # 调用替换（针对整个配置）
    feishu_config = cfg.get("feishu", {})

    # 飞书启动
    app_id = feishu_config.get("app_id", "").strip()
    app_secret = feishu_config.get("app_secret", "").strip()

    feishu_enabled = feishu_config.get("enabled", False)
    if app_id and app_secret and feishu_enabled:
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
    email_enabled = email_config.get("enabled", False)
    if email_config.get("imap_user") and email_enable:
        threading.Thread(
            target=start_email_poller,
            args=(email_config,),
            daemon=True,
            name="Email-Poller"
        ).start()
        print("🚀 邮件配置有效，正在启动邮箱连接服务...")

    # 初始化全局 Swarm（配置API仍共用）
    global global_swarm, feishu_swarm, email_swarm
    if not global_swarm:
        print("🚀 初始化全局 Swarm（配置API专用）")
        global_swarm = MultiAgentSwarm(config_path="swarm_config.yaml")

    # 🔥 给飞书和邮件各自独立实例（彻底避免干扰）
    if not feishu_swarm:
        print("🚀 创建飞书专用 Swarm 实例")
        feishu_swarm = MultiAgentSwarm(config_path="swarm_config.yaml")
    if not email_swarm:
        print("🚀 创建邮件专用 Swarm 实例")
        email_swarm = MultiAgentSwarm(config_path="swarm_config.yaml")


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

            # 🔥 取消功能
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
                await websocket.send_json({"type": "error", "content": "❌ 消息不能为空"})
                continue

            # 🔥 先获取 session_id（关键修复！）
            session_id = get_or_create_session(data.get("session_id"))

            # 🔥 新增：管理员指令系统（最高优先级拦截）
            admin_response = handle_admin_command(message, session_id)
            if admin_response:
                await websocket.send_json({"type": "stream", "agent": "System", "content": admin_response})
                await websocket.send_json({"type": "end"})
                continue  # ← 改成 continue 更稳妥

            current_swarm = get_or_create_swarm(session_id)

            use_memory = data.get("use_memory", False)
            memory_key = data.get("memory_key", "default")
            force_complexity = data.get("force_complexity")

            current_swarm._reset_cancel_flag()

            # ====================== 【WebUI双层控制】 ======================
            if force_complexity in ("simple", "balanced", "complex"):
                print(f"🌐 [WebUI] 用户手动选择 {force_complexity.capitalize()}")
                current_swarm._webui_auto_mode = False
            else:
                print(f"🌐 [WebUI] 自动模式 → Complex 降级保护已启用")
                current_swarm._webui_auto_mode = True
                force_complexity = None
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

            # ============ 新增：解决 Markdown 流式渲染乱码问题 ============
            current_response = ""          # 累积完整回复内容（关键！）
            # ============================================================

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
                    nonlocal current_response
                    if not content:  # 防空 chunk
                        return
                    current_response += content  # 继续累积（用于最终保存）

                    asyncio.run_coroutine_threadsafe(
                        stream_queue.put({
                            "type": "stream",
                            "agent": agent_name,
                            "content": content,  # ← 关键修改：只发送本次新增 chunk
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
                if not feishu_enabled: return

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
                answer = feishu_swarm.solve(
                    full_message,
                    use_memory=True,
                    memory_key="feishu_long"
                )

                # ==================== 【终极稳定版】post 富文本（永不 200621） ====================
                clean_answer = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', answer)[:4000]
                clean_answer = clean_answer.replace('{{', '{ {').replace('}}', '} }')

                # 构造 post 富文本
                post_content = {
                    "zh_cn": {
                        "title": "🤖 MultiAgentSwarm AI 回复",
                        "content": [
                            [
                                {
                                    "tag": "text",
                                    "text": clean_answer or "（回复内容为空）"
                                }
                            ]
                        ]
                    }
                }

                try:
                    request = CreateMessageRequest.builder() \
                        .receive_id_type("chat_id") \
                        .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .msg_type("post")  # ← 关键：改为 post
                        .content(json.dumps(post_content, ensure_ascii=False))
                        .build()
                    ).build()

                    response = feishu_client.im.v1.message.create(request)
                    if response.success():
                        print("✅ 已回复（post 富文本 - 稳定版）")
                    else:
                        raise Exception(response.msg)

                except Exception as e:
                    print(f"⚠️ post 也失败，切换纯文本兜底: {str(e)[:80]}")
                    # 最终兜底（一定能发出去）
                    fallback = clean_answer[:1800] or "MultiAgentSwarm 已处理完成。"
                    request = CreateMessageRequest.builder() \
                        .receive_id_type("chat_id") \
                        .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .msg_type("text")
                        .content(json.dumps({"text": fallback}, ensure_ascii=False))
                        .build()
                    ).build()
                    feishu_client.im.v1.message.create(request)
                    print("✅ 已使用普通文本回复（兜底成功）")

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

    # 启动日志
    trigger_keywords = config.get('trigger_keywords', [])
    print(f"📧 Email Poller 已启动（间隔 {interval} 秒）")
    print(f"🎯 触发关键词（仅匹配主题，大小写不敏感）: {trigger_keywords}")

    def fetch_recent_unseen():
        """获取最近24小时未读邮件（终极增强版）"""
        try:
            mail = imaplib.IMAP4_SSL(config["imap_server"], 993)

            # IMAP ID（保持不变）
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
            keywords = [k.lower() for k in config.get('trigger_keywords', [])]

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

                    # ==================== 【核心修改】触发词过滤 ====================
                    # 只检查 Subject + 大小写不敏感

                    subject_lower = subject.lower()

                    if keywords and not any(kw in subject_lower for kw in keywords):
                        print(f"⏭️ 跳过（主题无触发词）: {subject[:50]}")
                        continue

                    print(f"✅ 触发词匹配成功: {subject[:50]}")
                    # ============================================================

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

                            # ✅ 智能判断是否为有效内容（修复后）
                            if body:
                                # 统计所有有效字符（中文 + 英文数字 + 常用标点）
                                valid_chars = sum(1 for c in body if (
                                        c.isalnum() or  # 英文/数字
                                        '\u4e00' <= c <= '\u9fff' or  # 中文
                                        '\u3040' <= c <= '\u30ff' or  # 日文
                                        c in '，。！？、；：""''（）《》【】…—'  # 中文标点
                                ))

                                # ✅ 只要有1个以上有效字符就认为有效（原阈值 3 太高）
                                if valid_chars < 1:
                                    body = "[邮件正文无法解析 - 可能是纯图片或格式化邮件]"
                                    print(f"  ⚠️  正文无效（有效字符: {valid_chars}）")
                                else:
                                    print(f"  ✅ HTML 转文本成功（{len(body)} 字符，{valid_chars} 有效字符）")

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
        """发送专业HTML邮件 - Markdown自动美化（已优化版）"""
        try:
            msg = MIMEMultipart('alternative')

            msg['From'] = formataddr((Header("MultiAgentSwarm AI", 'utf-8').encode(), config['imap_user']))
            msg['To'] = to
            msg['Subject'] = Header(subject, 'utf-8')

            # ==================== 纯文本版本（兼容纯文本客户端） ====================
            plain_text = re.sub(r'\*\*|\#+\s|\`|\[.*?\]\(.*?\)', '', body)[:2500]
            plain_text = re.sub(r'\n{3,}', '\n\n', plain_text)

            # ==================== Markdown → 专业HTML（核心优化） ====================
            try:
                html_content = markdown.markdown(
                    body,
                    extensions=[
                        'extra',  # tables, fenced_code, nl2br 等
                        'attr_list',
                        'tables',  # 明确支持表格
                        'fenced_code'  # 代码块高亮支持
                    ]
                )
                print("✅ Markdown 已成功转为专业HTML（含表格+代码块）")
            except Exception as e:
                print(f"⚠️ Markdown 转换失败，使用纯文本兜底: {e}")
                html_content = body.replace('\n', '<br>')

            # ==================== 美观HTML模板（已增强表格样式） ====================
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ 
                        font-family: Arial, "Helvetica Neue", "Microsoft YaHei", sans-serif; 
                        line-height: 1.75; 
                        color: #333; 
                        max-width: 780px; 
                        margin: 0 auto; 
                        padding: 30px 20px; 
                        background: #f9f9f9;
                    }}
                    h1, h2, h3 {{ color: #1e88e5; }}
                    h1 {{ border-bottom: 3px solid #eee; padding-bottom: 12px; }}
                    h2 {{ margin-top: 32px; color: #1976d2; }}
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin: 20px 0; 
                        background: white;
                    }}
                    th, td {{ 
                        border: 1px solid #ddd; 
                        padding: 12px 15px; 
                        text-align: left;
                    }}
                    th {{ background: #f0f7ff; font-weight: 600; }}
                    tr:nth-child(even) {{ background: #f8f9fa; }}
                    pre {{ 
                        background: #f5f7f9; 
                        padding: 16px; 
                        border-radius: 8px; 
                        overflow-x: auto; 
                        border: 1px solid #e0e0e0;
                        font-family: Consolas, monospace;
                    }}
                    code {{ 
                        font-family: Consolas, monospace; 
                        background: #f0f0f0; 
                        padding: 2px 6px; 
                        border-radius: 4px; 
                    }}
                    blockquote {{ 
                        border-left: 5px solid #1e88e5; 
                        padding-left: 20px; 
                        color: #555; 
                        margin: 24px 0;
                    }}
                    hr {{ border: none; border-top: 2px solid #eee; margin: 35px 0; }}
                    .footer {{ color:#666; font-size:13px; text-align:center; margin-top:40px; }}
                </style>
            </head>
            <body>
                {html_content}
                <hr>
                <div class="footer">
                    本邮件由 MultiAgentSwarm AI 系统自动生成<br>
                    如有任何疑问，请直接回复此邮件
                </div>
            </body>
            </html>
            """

            # 附加纯文本 + HTML（标准 alternative 格式）
            msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
            msg.attach(MIMEText(full_html, 'html', 'utf-8'))

            # ==================== 附件处理（完全保持原逻辑） ====================
            if attachments:
                from email.mime.base import MIMEBase
                from email import encoders
                for file_path in attachments:
                    try:
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                part = MIMEBase('application', 'octet-stream')
                                part.set_payload(f.read())
                                encoders.encode_base64(part)
                                filename = os.path.basename(file_path)
                                part.add_header('Content-Disposition',
                                                f'attachment; filename="{Header(filename, "utf-8").encode()}"')
                                msg.attach(part)
                            print(f"📎 已附加文件: {filename}")
                    except Exception as e:
                        print(f"⚠️ 附件添加失败 {file_path}: {e}")

            # ==================== 发送逻辑 ====================
            port = config.get("smtp_port", 465)
            if port == 465:
                server = smtplib.SMTP_SSL(config["smtp_server"], port)
            else:
                server = smtplib.SMTP(config["smtp_server"], port)
                server.starttls()

            server.login(config["imap_user"], config["imap_pass"])
            server.send_message(msg)
            server.quit()

            print(f"✅ 专业美化邮件已成功发送 → {to}")
            return True

        except Exception as e:
            print(f"❌ 邮件发送失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # 主循环
    while True:
        try:
            if not email_enabled:
                time.sleep(interval)
                continue
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

                        # 保持和飞书/WebUI 完全一致的干净输入
                        # full_question = f"[来自邮件]\n{email_data['subject']}\n\n{email_data['body']}"
                        # 第 1046 行改为:
                        full_question = f"[来自邮件] {email_data['body']}"
                        if email_data['attachments']:
                            full_question += "\n\n[附件]\n" + "\n".join(f"- {a}" for a in email_data['attachments'])

                        if email_data['attachments']:
                            full_question += "\n\n[附件]\n" + "\n".join(f"- {a}" for a in email_data['attachments'])

                        print("🤖 AI 处理中...（使用与 WebUI/飞书一致的 prompt）")

                        # ✅ 完全一致的调用（无任何强制指令）
                        answer = email_swarm.solve(
                            full_question,
                            use_memory=True,
                            memory_key="email_auto"
                            # 故意不传 force_complexity，让它走默认路由，和其他渠道完全一致
                        )

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