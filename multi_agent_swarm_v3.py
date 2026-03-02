#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多智能体协作系统 (Multi-Agent Swarm) v3.0.0
✨ 新增功能：
- 多层对抗辩论 (Adversarial Debate)
- Meta-Critic 元批评机制
- 动态 Agent 工厂 + 任务分解
- 主动知识图谱 + 蒸馏
- 自适应反思深度
"""

import yaml
import logging
import importlib.util
import requests
import random
import time
import threading
import base64
import mimetypes
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import os
from datetime import datetime, timedelta
from pathlib import Path
import glob

# ====================== 时间统计工具 ======================
from contextlib import contextmanager


class TimeTracker:
    """时间统计工具类"""

    def __init__(self):
        self.start_time = None
        self.checkpoints = {}

    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self.start_time

    def checkpoint(self, name: str):
        """记录检查点"""
        if self.start_time is None:
            self.start()
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed

    def get_elapsed(self) -> float:
        """获取总耗时"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}分{secs:.1f}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}小时{minutes}分{secs:.0f}秒"

    def summary(self) -> str:
        """生成耗时摘要"""
        total = self.get_elapsed()
        lines = [f"\n{'=' * 60}"]
        lines.append(f"⏱️  总耗时: {self.format_time(total)}")
        lines.append(f"{'─' * 60}")

        if self.checkpoints:
            lines.append("📊 各阶段耗时:")
            for name, elapsed in self.checkpoints.items():
                percentage = (elapsed / total * 100) if total > 0 else 0
                lines.append(f"   {name}: {self.format_time(elapsed)} ({percentage:.1f}%)")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


@contextmanager
def timer(description: str):
    """上下文管理器：自动计时并打印"""
    start = time.time()
    print(f"⏱️  开始: {description}", flush=True)
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"✅ 完成: {description} | 耗时: {TimeTracker().format_time(elapsed)}", flush=True)


# ====================== 线程安全的工具缓存 ======================
tool_cache = {}
cache_count = 0
cache_lock = threading.Lock()


def clean_cache():
    """自动清理工具缓存（线程安全）"""
    global cache_count
    with cache_lock:
        if cache_count > 50:
            tool_cache.clear()
            cache_count = 0
            logging.info("🧹 自动清理工具缓存")


# ====================== 工具函数 ======================
def web_search(query: str, num_results: int = 5) -> str:
    """DuckDuckGo 网页搜索（带缓存 + 随机延时）"""
    global cache_count
    clean_cache()

    with cache_lock:
        if query in tool_cache:
            return tool_cache[query]

    time.sleep(random.uniform(0.5, 2.0))

    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]

        result = "\n".join([
            f"标题: {r['title']}\n摘要: {r['body']}\n链接: {r['href']}"
            for r in results
        ])

        with cache_lock:
            tool_cache[query] = result
            cache_count += 1

        return result
    except Exception as e:
        logging.error(f"搜索失败: {e}")
        return f"搜索失败: {str(e)}"


def browse_page(url: str) -> str:
    """浏览网页并提取文本（带缓存）"""
    global cache_count
    clean_cache()

    with cache_lock:
        if url in tool_cache:
            return tool_cache[url]

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        result = "\n".join(chunk for chunk in chunks if chunk)

        with cache_lock:
            tool_cache[url] = result
            cache_count += 1

        return result
    except Exception as e:
        logging.error(f"浏览失败 {url}: {e}")
        return f"浏览失败: {str(e)}"


def run_python(code: str) -> str:
    """
    沙箱执行 Python 代码（10秒超时）
    注意：threading.Timer 无法真正终止阻塞代码，仅作软超时
    """
    result_container = {"output": None, "done": False}

    def target():
        try:
            restricted_globals = {
                "__builtins__": {
                    "print": print,
                    "range": range,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "sum": sum,
                    "min": min,
                    "max": max,
                }
            }
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            result_container["output"] = str(local_vars.get("result", "执行成功，无返回结果"))
        except Exception as e:
            result_container["output"] = f"执行错误: {str(e)}"
        finally:
            result_container["done"] = True

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=10.0)

    if not result_container["done"]:
        return "⏱️ 执行超时（10秒）"

    return result_container["output"]


# ====================== 向量记忆 ======================
class VectorMemory:
    """
    基于 ChromaDB 和 SentenceTransformer 的向量记忆系统
    ✅ 优先使用缓存模型，避免重复下载
    """

    def __init__(
            self,
            persist_directory: str = "./memory_db",
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir: str = "./cached_model/"
    ):
        """
        初始化向量记忆系统

        Args:
            persist_directory: ChromaDB 数据库路径
            model_name: SentenceTransformer 模型名称
            cache_dir: 模型缓存目录（优先从此处加载）
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.cache_dir = os.path.abspath(cache_dir)

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 初始化嵌入模型（优先使用缓存）
        self._init_embedding_model()

        # 初始化 ChromaDB
        self._init_chromadb()

    def _init_embedding_model(self):
        """初始化嵌入模型（优先从缓存加载）"""
        try:
            # 检查缓存是否存在
            cached_model_path = os.path.join(self.cache_dir, self.model_name.replace('/', '_'))

            if os.path.exists(cached_model_path):
                logging.info(f"📦 从缓存加载向量模型: {cached_model_path}")
                print(f"📦 从缓存加载向量模型: {self.model_name}")
                self.embedding_model = SentenceTransformer(cached_model_path)
            else:
                logging.info(f"⬇️  下载向量模型: {self.model_name} → {cached_model_path}")
                print(f"⬇️  首次使用，正在下载向量模型: {self.model_name}")
                print(f"   下载后将缓存到: {cached_model_path}")

                # 下载模型
                self.embedding_model = SentenceTransformer(self.model_name)

                # 保存到缓存
                self.embedding_model.save(cached_model_path)
                logging.info(f"✅ 模型已缓存到: {cached_model_path}")
                print(f"✅ 模型已缓存，下次将直接使用")

        except Exception as e:
            logging.error(f"❌ 向量模型初始化失败: {e}")
            raise

    def _init_chromadb(self):
        """初始化 ChromaDB 客户端"""
        try:
            os.makedirs(os.path.dirname(self.persist_directory) if os.path.dirname(
                self.persist_directory) else ".", exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name="swarm_memory",
                metadata={"description": "Agent memory storage"}
            )

            logging.info(f"✅ ChromaDB 初始化成功: {self.persist_directory}")

        except Exception as e:
            logging.error(f"❌ ChromaDB 初始化失败: {e}")
            self.collection = None

    def add(self, text: str, metadata: Optional[Dict] = None):
        """添加记忆到向量数据库"""
        if not self.collection:
            return

        try:
            # 生成嵌入向量
            embedding = self.embedding_model.encode(text).tolist()

            # 生成 ID
            memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # 添加到数据库
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {"timestamp": datetime.now().isoformat()}]
            )

            logging.info(f"✅ 记忆已保存: {memory_id}")

        except Exception as e:
            logging.error(f"❌ 保存记忆失败: {e}")

    def search(self, query: str, n_results: int = 5) -> str:
        """搜索相关记忆"""
        if not self.collection:
            return ""

        try:
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode(query).tolist()

            # 搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # 格式化结果
            if results and results["documents"]:
                return "\n\n---\n\n".join(results["documents"][0])

        except Exception as e:
            logging.error(f"❌ 搜索记忆失败: {e}")

        return ""


class PrimalMemory:
    """PrimalClaw极简版：树状日志 + 原子KB + 衰退 + 完整QMD检索（已优化）"""
    def __init__(self, base_dir: str = "./memory", vector_memory=None):  # ← 改这里
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / "logs"
        self.kb_dir = self.base_dir / "kb"
        self.archive_dir = self.base_dir / "archive"
        for d in [self.logs_dir, self.kb_dir / "lessons", self.kb_dir / "decisions", self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.vector_memory = vector_memory   # ← 新增这一行

    def save_episode(self, task: str, history: list, final_answer: str, memory_key: str):
        """写入Layer2树状日志 + 原子KB（不覆盖）"""
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime('%H%M%S')
        log_path = self.logs_dir / today / f"{memory_key}_{timestamp}.md"
        log_path.parent.mkdir(exist_ok=True)

        content = f"# Episode: {memory_key}\n**时间**: {datetime.now()}\n**任务**: {task[:200]}\n\n"
        content += "## 讨论历史\n" + "\n".join([f"- **{h.get('speaker','')}:** {h.get('content','')[:300]}..." for h in history[-5:]])
        content += f"\n\n## 最终答案\n{final_answer[:2000]}\n"

        log_path.write_text(content, encoding="utf-8")

        # 原子KB（不覆盖，用时间戳）
        kb_path = self.kb_dir / "lessons" / f"{memory_key}_{timestamp}.md"
        kb_text = f"## lesson-{memory_key}-{timestamp}\n"
        kb_text += f"**Title**: 从 {memory_key} 提炼的关键经验\n"
        kb_text += f"**Content**: {final_answer[:800].replace('**','')}\n"
        kb_text += f"**Source**: {log_path.relative_to(self.base_dir)}\n"
        kb_text += f"**Importance**: 0.85\n**LastAccess**: {datetime.now().isoformat()}\n"
        kb_path.write_text(kb_text, encoding="utf-8")

    def get_relevant_memory(self, query: str, n: int = 3) -> str:
        """完整QMD混合检索：Vector语义优先 + 文件关键词"""
        results = []

        # 1. 优先用已有VectorMemory（语义强）
        if hasattr(self, 'vector_memory') and self.vector_memory:  # 注意：需在MultiAgentSwarm中传引用
            vec_result = self.vector_memory.search(query, n_results=n)
            if vec_result:
                results.append(f"🔍 Vector语义记忆:\n{vec_result}")

        # 2. 文件精确匹配（fallback）
        for f in glob.glob(str(self.kb_dir / "**/*.md"), recursive=True)[:n]:
            try:
                text = Path(f).read_text(encoding="utf-8")[:600]
                if any(k in text.lower() for k in query.lower().split()[:4]):
                    results.append(f"📁 {Path(f).name}:\n{text}")
            except:
                pass

        # return "\n\n---\n\n".join(results[:n]) if results else ""
        result = "\n\n---\n\n".join(results[:n]) if results else ""
        # return result[:12000]  # ← 新增：单次注入最多12K字符（≈6K tokens）
        return result[:12000]  # ← 新增：单次注入最多12K字符（≈6K tokens）

    def decay(self):
        """指数衰退 + GC（更安全）"""
        threshold = 0.4
        for md_file in glob.glob(str(self.kb_dir / "**/*.md"), recursive=True):
            try:
                text = Path(md_file).read_text(encoding="utf-8")
                if "Importance:" in text and "LastAccess:" in text:
                    imp_line = text.split("Importance:")[1].split("\n")[0].strip()
                    last_line = text.split("LastAccess:")[1].split("\n")[0].strip()
                    imp = float(imp_line)
                    last = datetime.fromisoformat(last_line)
                    days = (datetime.now() - last).days
                    new_imp = imp * (0.985 ** max(days, 0))
                    if new_imp < threshold:
                        archive_path = self.archive_dir / Path(md_file).name
                        Path(md_file).rename(archive_path)
            except:
                pass

# ====================== ✨ 知识图谱管理器 ======================
class KnowledgeGraph:
    """
    轻量级知识图谱 + 自动蒸馏
    用于追踪关键概念、关系和发现
    """

    def __init__(self, enable_distillation: bool = True):
        self.graph = {}  # {entity: {"type": str, "relations": [(rel, target)], "evidence": [str]}}
        self.enable_distillation = enable_distillation
        self.distilled_knowledge = []

    def add_entity(self, entity: str, entity_type: str = "concept", evidence: str = ""):
        """添加实体"""
        if entity not in self.graph:
            self.graph[entity] = {
                "type": entity_type,
                "relations": [],
                "evidence": [evidence] if evidence else []
            }
        elif evidence:
            self.graph[entity]["evidence"].append(evidence)

    def add_relation(self, source: str, relation: str, target: str):
        """添加关系"""
        if source in self.graph:
            self.graph[source]["relations"].append((relation, target))
        else:
            self.add_entity(source)
            self.graph[source]["relations"].append((relation, target))

    def distill(self, max_items: int = 10) -> str:
        """
        蒸馏知识图谱（提取核心概念和关系）
        """
        if not self.enable_distillation or not self.graph:
            return ""

        # 按关系数量排序（最重要的概念）
        sorted_entities = sorted(
            self.graph.items(),
            key=lambda x: len(x[1]["relations"]),
            reverse=True
        )[:max_items]

        distilled = ["🧠 核心知识蒸馏:"]
        for entity, data in sorted_entities:
            relations_str = ", ".join([f"{rel}→{tgt}" for rel, tgt in data["relations"][:3]])
            distilled.append(f"• {entity} ({data['type']}): {relations_str}")

        result = "\n".join(distilled)
        self.distilled_knowledge.append(result)
        return result

    def get_context(self, entity: str, depth: int = 1) -> str:
        """获取实体的上下文"""
        if entity not in self.graph:
            return ""

        context = [f"📌 {entity} ({self.graph[entity]['type']})"]

        # 一级关系
        for rel, target in self.graph[entity]["relations"]:
            context.append(f"  └─ {rel} → {target}")

            # 二级关系（如果 depth > 1）
            if depth > 1 and target in self.graph:
                for sub_rel, sub_target in self.graph[target]["relations"][:2]:
                    context.append(f"     └─ {sub_rel} → {sub_target}")

        return "\n".join(context)


# ====================== Skill 动态加载器 ======================
def load_skills(skills_dir: str = "skills"):
    """
    递归加载 skills 目录下的所有 Python 工具和 Markdown 知识文件
    支持子目录结构
    """
    tool_registry = {}
    shared_knowledge = []

    if not os.path.exists(skills_dir):
        logging.warning(f"⚠️ Skills 目录不存在: {skills_dir}")
        return tool_registry, ""

    # 递归扫描所有 .py 文件
    py_files = glob.glob(os.path.join(skills_dir, "**/*.py"), recursive=True)

    for py_file in py_files:
        try:
            rel_path = os.path.relpath(py_file, skills_dir)

            # 动态导入模块
            spec = importlib.util.spec_from_file_location(
                os.path.splitext(os.path.basename(py_file))[0],
                py_file
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # 查找工具函数和 schema
            if hasattr(mod, "tool_function") and hasattr(mod, "tool_schema"):
                tool_name = mod.tool_schema["function"]["name"]
                tool_registry[tool_name] = {
                    "func": mod.tool_function,
                    "schema": mod.tool_schema
                }
                logging.info(f"✅ 加载 Skill (py): {tool_name} | 来自: {rel_path}")
            else:
                logging.warning(f"⚠️ 跳过无效 Skill 文件: {rel_path}")

        except Exception as e:
            logging.error(f"❌ 加载 Skill 失败: {py_file} | 错误: {e}")

    # 递归扫描所有 .md 文件
    md_files = glob.glob(os.path.join(skills_dir, "**/*.md"), recursive=True)

    for md_file in md_files:
        try:
            rel_path = os.path.relpath(md_file, skills_dir)
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    shared_knowledge.append(f"### 📚 来自 {rel_path} ###\n{content}")
                    logging.info(f"📚 加载知识文件 (md): {rel_path}")
        except Exception as e:
            logging.error(f"❌ 读取知识文件失败: {md_file} | 错误: {e}")

    logging.info(f"📊 Skills 加载完成: {len(tool_registry)} 个工具, {len(shared_knowledge)} 个知识文件")

    shared_knowledge_str = "\n\n".join(shared_knowledge)

    return tool_registry, shared_knowledge_str


# ====================== Agent 类 ======================
class Agent:
    """单个智能体代理"""

    def __init__(
            self,
            config: Dict,
            default_model: str,
            default_max_tokens: int,
            tool_registry: Dict,
            shared_knowledge: str = "",
            vector_memory: Optional[VectorMemory] = None,
            knowledge_graph: Optional[KnowledgeGraph] = None,
            context_limit_k: str = "64"  # ← 新增这一行
    ):
        self.name = config["name"]
        self.role = config["role"]
        self.shared_knowledge = shared_knowledge
        self.vector_memory = vector_memory
        self.knowledge_graph = knowledge_graph
        self.context_limit_k = context_limit_k  # ← 新增这一行，保存下来

        # OpenAI 客户端配置
        self.client = OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        self.model = config.get("model", default_model)
        self.temperature = config.get("temperature", 0.7)
        self.stream = config.get("stream", False)
        self.max_tokens = config.get("max_tokens", default_max_tokens)

        # 工具配置
        enabled = config.get("enabled_tools", [])
        self.tools = [
            tool_registry[name]["schema"]
            for name in enabled
            if name in tool_registry
        ]
        self.tool_map = {
            name: tool_registry[name]["func"]
            for name in enabled
            if name in tool_registry
        }

        if self.tools:
            logging.debug(f"  {self.name} 已启用工具: {list(self.tool_map.keys())}")

    def _execute_tool(self, tool_call) -> Dict:
        """执行工具调用"""
        func_name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
            logging.info(f"🔧 {self.name} 调用工具: {func_name}({args})")

            result = self.tool_map[func_name](**args)

            # ✅ 修复：role 必须是 'tool'（NVIDIA 兼容），但内容前缀【Observation】让前端和日志完美显示红色 Observation 效果
            return {
                "role": "tool",  # ← 改回 tool（必须！）
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": f"【Observation】\n{result}"  # ← 关键：前缀让用户一眼看出是 Observation
            }
        except Exception as e:
            logging.error(f"工具执行失败 {func_name}: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": f"【Observation】\nTool error: {str(e)}"
            }

    def generate_response(
            self,
            history: List[Dict],
            round_num: int,
            system_extra: str = "",
            force_non_stream: bool = False,
            critique_previous: bool = False,
            stream_callback=None,
            log_callback=None
    ) -> str:
        """
        生成 Agent 响应（增强版：工具调用超限兜底）

        Args:
            history: 对话历史
            round_num: 当前轮次
            system_extra: 额外系统提示词
            force_non_stream: 强制关闭流式输出
            critique_previous: 是否启用批判模式
            stream_callback: 流式回调函数 callback(agent_name, chunk)
            log_callback: 日志回调函数 callback(message)

        Returns:
            str: Agent 的完整响应
        """
        # ✅ 判断是否使用流式输出（如果有 stream_callback，强制启用）
        use_stream = (
                (self.stream or stream_callback is not None) and
                not force_non_stream and
                not self.tools  # 有工具时暂时不用流式
        )

        # ✨ 批判模式增强
        if critique_previous and len(history) > 3:
            critique_prompt = (
                "🔍 在给出你的贡献前，请先用 [CRITIQUE] 标记指出上一轮讨论中：\n"
                "1. 至少 1 个潜在逻辑漏洞或矛盾点\n"
                "2. 至少 1 个可改进或补充的地方\n"
                "然后再给出你的建设性贡献。"
            )
            system_extra = critique_prompt + "\n\n" + system_extra

        # 🔥【Plan 注入】每个 Agent 都知道当前 Master Plan（最小改动）
        plan_summary = next(
            (h["content"] for h in history if "Master Plan" in str(h.get("content", ""))),
            ""
        )[:300]

        # 构建系统提示词（Plan 摘要放在最前面，让所有 Agent 对齐）
        system_prompt = (
            f"{self.role}\n"
            f"【当前Master Plan摘要】\n{plan_summary}\n\n"
            f"{self.shared_knowledge}\n"
            f"{system_extra}\n"
            "你是多智能体协作团队的一员，请提供有价值、准确、有深度的贡献。\n\n"

            # ==================== 【新增】强制 ReAct 三段式（对应架构图） ====================
            "【强制思考格式 - 必须严格遵守】\n"
            "无论是否调用工具，都请在回复最开头先按以下格式输出：\n"
            "Thinking: （怎么解决用户的问题，原因分析）\n"
            "Action: （需要调用的 Function 名称，或写 Final Answer）\n"
            "Action Input: （Function 的参数JSON，或最终答案摘要）\n\n"
            "示例：\n"
            "Thinking: 用户问 Transformer 注意力机制，我需要先回忆原理再查最新进展。\n"
            "Action: web_search\n"
            "Action Input: {\"query\": \"Transformer attention mechanism latest\"}\n\n"
            "用户和前端会直接看到这个思考过程，提升透明度和调试能力。"
            # =============================================================================
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 处理历史消息
        for h in history:
            if h["speaker"] == "User":
                messages.append({"role": "user", "content": h["content"]})
            elif h["speaker"] == "System":
                messages.append({"role": "system", "content": h["content"]})
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"[{h['speaker']}] {h.get('content', '')}"
                })

        est_tokens = sum(len(str(m.get("content", ""))) // 2 for m in messages)
        if est_tokens > 110000 and "128" in str(self.context_limit_k):
            logging.warning(f"⚠️ 上下文接近128K上限！当前估算 {est_tokens} tokens")
        start_time = time.time()

        try:
            # ✅ 添加开始日志
            if log_callback:
                log_callback(f"[{self.name}] 开始生成响应 (轮次 {round_num})")

            if use_stream:
                print(f"\n💬 【{self.name}】正在思考... ", end="", flush=True)
                if log_callback:
                    log_callback(f"[{self.name}] 正在思考...")

            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
                stream=use_stream,
            )

            full_response = ""

            # ===== 流式输出 =====
            if use_stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        print(delta, end="", flush=True)
                        full_response += delta

                        # ✅ 流式回调
                        if stream_callback:
                            stream_callback(self.name, delta)
                print()

            # ===== 非流式输出 =====
            else:
                full_response = response.choices[0].message.content or ""

                # ✅ 即使非流式，也调用回调（模拟流式效果）
                if stream_callback and full_response:
                    chunk_size = 20
                    for i in range(0, len(full_response), chunk_size):
                        chunk = full_response[i:i + chunk_size]
                        stream_callback(self.name, chunk)
                        time.sleep(0.02)

            # ===== 🔧 工具调用处理（支持多轮循环 + 超限兜底）=====
            if (not use_stream and
                    hasattr(response.choices[0].message, 'tool_calls') and
                    response.choices[0].message.tool_calls):

                max_tool_iterations = 10
                iteration = 0
                tool_call_history = []  # 🔥 记录所有工具调用

                while (hasattr(response.choices[0].message, 'tool_calls') and
                       response.choices[0].message.tool_calls and
                       iteration < max_tool_iterations):

                    iteration += 1
                    print(f"\n🔧 [{self.name}] 工具调用 (第 {iteration} 轮)")

                    if log_callback:
                        log_callback(f"[{self.name}] 工具调用 (第 {iteration} 轮)")

                    # 添加 assistant 消息（包含 tool_calls）
                    messages.append(response.choices[0].message.model_dump())

                    # 执行所有工具调用
                    for tool_call in response.choices[0].message.tool_calls:
                        tool_result = self._execute_tool(tool_call)
                        messages.append(tool_result)

                        # 🔥 记录工具调用
                        tool_call_history.append(tool_result['name'])

                        # 显示工具调用结果（截断预览）
                        result_preview = tool_result.get("content", "")[:150]
                        if len(tool_result.get("content", "")) > 150:
                            result_preview += "..."
                        print(f"   ✅ {tool_result['name']}: {result_preview}")

                        if log_callback:
                            log_callback(f"[{self.name}] 工具: {tool_result['name']}")

                    # 重新调用 API（带工具结果）
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=self.tools if self.tools else None,
                        tool_choice="auto" if self.tools else None,
                        stream=False
                    )

                    # ✅ 如果不再调用工具，提取最终答案并发送给前端
                    if not (hasattr(response.choices[0].message, 'tool_calls') and
                            response.choices[0].message.tool_calls):
                        full_response = response.choices[0].message.content or ""

                        # ✅ 模拟流式发送
                        if stream_callback and full_response:
                            chunk_size = 20
                            for i in range(0, len(full_response), chunk_size):
                                chunk = full_response[i:i + chunk_size]
                                stream_callback(self.name, chunk)
                                time.sleep(0.02)

                        print(f"   💬 [{self.name}] 工具调用完成，生成最终答案")
                        if log_callback:
                            log_callback(f"[{self.name}] 工具调用完成")
                        break

                # 🔥🔥🔥 核心修复：工具调用超限后的兜底逻辑 🔥🔥🔥
                if iteration >= max_tool_iterations:
                    print(f"   ⚠️ [{self.name}] 工具调用达到上限 ({max_tool_iterations} 轮)")

                    # 🔥 检查是否有有效回复
                    if not full_response or not full_response.strip():
                        # 生成友好兜底消息
                        unique_tools = list(set(tool_call_history))
                        full_response = (
                            f"⚠️ 工具调用达到上限（{max_tool_iterations}次），已收集部分信息。\n\n"
                            f"**已调用工具**: {', '.join(unique_tools)}\n\n"
                            "**建议操作**：\n"
                            "1. 尝试简化问题或分步提问\n"
                            "2. 检查附件大小（建议<5MB）\n"
                            "3. 如需完整分析，请明确指定分析范围\n\n"
                            "💡 **提示**: 当前任务可能过于复杂，建议将其拆分为多个小任务分别处理。"
                        )

                        print(f"   ⚠️ 未生成有效回复，已使用兜底消息")
                        if log_callback:
                            log_callback(f"[{self.name}] ⚠️ 工具超限，使用兜底回复")

                    # ✅ 新增：兜底消息一次性发送（避免空白分块）
                    if stream_callback and full_response:
                        stream_callback(self.name, full_response)  # ← 一次性发送完整消息

                    if log_callback:
                        log_callback(f"[{self.name}] 工具调用超限")

            # ===== 计算并显示耗时 =====
            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed:.2f}秒" if elapsed < 60 else f"{int(elapsed // 60)}分{elapsed % 60:.1f}秒"

            if not use_stream:
                print(f"⏱️  【{self.name}】响应完成 | 耗时: {elapsed_str}")

            if log_callback:
                log_callback(f"[{self.name}] 响应完成 (耗时 {elapsed_str})")

            logging.info(f"⏱️  {self.name} 响应耗时: {elapsed_str}")

            # 🔥 最终兜底检查（防止所有情况漏网）
            if not full_response or not full_response.strip():
                full_response = (
                    f"⚠️ [{self.name}] 未能生成有效回复。\n\n"
                    "可能原因：\n"
                    "- 工具调用超限或失败\n"
                    "- 网络异常或模型超时\n"
                    "- 输入内容无法处理\n\n"
                    "**建议操作**：\n"
                    "1. 检查输入内容是否完整\n"
                    "2. 简化问题后重试\n"
                    "3. 联系技术支持"
                )
                logging.warning(f"⚠️ {self.name} 未生成有效回复，使用兜底消息")

            return full_response.strip()

        except Exception as e:
            elapsed = time.time() - start_time
            err = f"[Error in {self.name}]: {str(e)}"
            logging.error(f"{err} | 耗时: {elapsed:.2f}秒")
            print(f"❌ 【{self.name}】执行失败 | 耗时: {elapsed:.2f}秒")

            if log_callback:
                log_callback(f"[{self.name}] ❌ 执行失败: {str(e)[:50]}")

            # 🔥 异常情况也返回友好消息
            return (
                f"❌ [{self.name}] 执行失败\n\n"
                f"**错误信息**: {str(e)[:200]}\n\n"
                "**建议操作**：\n"
                "1. 检查网络连接\n"
                "2. 确认 API 配置正确\n"
                "3. 稍后重试或联系管理员"
            )


# ====================== 主类 MultiAgentSwarm ======================
class MultiAgentSwarm:
    """多智能体群智慧框架 v3.0.0"""

    def __init__(self, config_path: str = "swarm_config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ 配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # OpenAI 配置
        oai = cfg.get("openai", {})
        self.default_model = oai.get("default_model", "gpt-4o-mini")
        self.default_max_tokens = oai.get("default_max_tokens", 4096)
        self.context_limit_k = oai.get("context_limit_k", "64")

        # Swarm 配置
        swarm = cfg.get("swarm", {})
        self.mode = swarm.get("mode", "fixed")
        # self.max_rounds = swarm.get("max_rounds", 3 if self.mode == "fixed" else 10)
        # 根据上下文限制动态轮次（64K时保守，128K时激进）
        self.max_rounds = 8 if "64" in str(self.context_limit_k) else 12  # 或者读配置
        self.max_concurrent_agents = swarm.get("max_concurrent_agents", 2)
        self.reflection_planning = swarm.get("reflection_planning", True)
        self.enable_web_search = swarm.get("enable_web_search", False)
        self.max_images = swarm.get("max_images", 2)

        self.log_file = swarm.get("log_file", "swarm.log")
        self.skills_dir = swarm.get("skills_dir", "skills")
        self.memory_file = swarm.get("memory_file", "memory.json")
        self.max_memory_items = swarm.get("max_memory_items", 50)

        # ✨ 新增增强配置
        advanced = cfg.get("advanced_features", {})
        debate_cfg = advanced.get("adversarial_debate", {})
        self.adversarial_debate_strategy = debate_cfg.get("trigger_strategy", "quality_based")
        self.adversarial_debate_threshold = debate_cfg.get("trigger_threshold", 82)
        self.adversarial_debate_interval = debate_cfg.get("trigger_interval", 2)

        self.enable_adversarial_debate = advanced.get("adversarial_debate", {}).get("enabled", True)
        self.enable_meta_critic = advanced.get("meta_critic", {}).get("enabled", True)
        self.enable_task_decomposition = advanced.get("task_decomposition", {}).get("enabled", True)
        self.enable_knowledge_graph = advanced.get("knowledge_graph", {}).get("enabled", True)
        self.enable_adaptive_depth = advanced.get("adaptive_reflection", {}).get("enabled", True)

        self.max_reflection_rounds = advanced.get("adaptive_reflection", {}).get("max_rounds", 3)
        self.reflection_quality_threshold = advanced.get("adaptive_reflection", {}).get("quality_threshold", 85)
        self.stop_quality_threshold = advanced.get("adaptive_reflection", {}).get("stop_threshold", 80)
        self.quality_convergence_delta = advanced.get("adaptive_reflection", {}).get("convergence_delta", 3)

        # ✨✨✨ 智能路由配置（新增）✨✨✨
        routing_cfg = cfg.get("intelligent_routing", {})
        self.intelligent_routing_enabled = routing_cfg.get("enabled", True)
        self.force_complexity = routing_cfg.get("force_complexity", None)  # "simple"/"medium"/"complex"/None

        # 向量记忆配置
        vector_cfg = swarm.get("vector_memory", {})
        self.vector_memory_enabled = vector_cfg.get("enabled", False)
        self.vector_persist_directory = vector_cfg.get("persist_directory", "./memory_db")
        self.vector_model_cache_dir = vector_cfg.get("model_cache_dir", "./cached_model/")
        self.vector_embedding_model = vector_cfg.get("embedding_model",
                                                     "sentence-transformers/distiluse-base-multilingual-cased-v2")

        # 日志配置
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            encoding="utf-8",
            force=True
        )
        logging.getLogger().addHandler(logging.StreamHandler())

        # 打印启动信息
        self._print_startup_banner()

        # 加载 Skills
        self.tool_registry, self.shared_knowledge = load_skills(self.skills_dir)

        # ====================== 【新增】支持 YAML 中的 shared_knowledge ======================
        yaml_shared = cfg.get("shared_knowledge", "") or ""
        if yaml_shared.strip():
            self.shared_knowledge = (yaml_shared.strip() + "\n\n" + self.shared_knowledge).strip()
            logging.info(f"✅ 已合并 YAML shared_knowledge（{len(yaml_shared)} 字符）")
            print(f"✅ 已加载 YAML 全局知识（{len(yaml_shared)} 字符）")

        # 添加内置网络搜索工具
        if self.enable_web_search:
            self.tool_registry["web_search"] = {
                "func": web_search,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "实时网页搜索最新信息（DuckDuckGo）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "搜索关键词"},
                                "num_results": {"type": "integer", "description": "返回结果数量", "default": 5}
                            },
                            "required": ["query"]
                        }
                    }
                }
            }
            logging.info("✅ 已启用网络搜索工具")

        # 初始化持久化记忆
        self.memory = self._load_memory()

        # 初始化向量记忆
        self.vector_memory = None
        if self.vector_memory_enabled:
            try:
                self.vector_memory = VectorMemory(
                    persist_directory=self.vector_persist_directory,
                    model_name=self.vector_embedding_model,
                    cache_dir=self.vector_model_cache_dir
                )
                logging.info("✅ 向量记忆系统初始化成功")
            except Exception as e:
                logging.warning(f"⚠️ 向量记忆初始化失败: {e}")
                self.vector_memory_enabled = False

        # ✨ PrimalClaw极简记忆系统（最小改动核心）
        self.primal_memory = PrimalMemory(
            base_dir="./memory",
            vector_memory=self.vector_memory  # ← 直接传入，更优雅
        )
        logging.info("✅ PrimalMemory (树状日志+原子KB+衰退) 初始化成功")

        # ✨ 初始化知识图谱
        self.knowledge_graph = None
        if self.enable_knowledge_graph:
            self.knowledge_graph = KnowledgeGraph(enable_distillation=True)
            logging.info("✅ 知识图谱系统初始化成功")

        # 初始化 Agents
        self.agents = []
        for a_cfg in cfg.get("agents", [])[:swarm.get("num_agents", 4)]:
            agent = Agent(
                a_cfg,
                self.default_model,
                self.default_max_tokens,
                self.tool_registry,
                self.shared_knowledge,
                self.vector_memory,
                self.knowledge_graph,
                context_limit_k=self.context_limit_k
            )
            self.agents.append(agent)
            logging.info(f"✅ Agent 加载: {agent.name} | Model: {agent.model}")

        if not self.agents:
            raise ValueError("❌ 至少需要配置一个 Agent")

        # ✨ 新增：取消标志（线程安全）
        self._cancel_flag = threading.Event()
        self._cancel_lock = threading.Lock()

        self.leader = self.agents[0]
        logging.info(f"👑 Leader: {self.leader.name}")

    def cancel_current_task(self):
        """取消当前正在执行的任务（线程安全）"""
        with self._cancel_lock:
            self._cancel_flag.set()
            logging.info("🛑 收到取消请求")

    def _reset_cancel_flag(self):
        """重置取消标志（新任务开始时调用）"""
        with self._cancel_lock:
            self._cancel_flag.clear()

    def _check_cancellation(self) -> bool:
        """检查是否被取消"""
        return self._cancel_flag.is_set()

    def _generate_detailed_plan(self, task: str, history: List[Dict]) -> str:
        """【最小改动核心】生成结构化Master Plan"""
        plan_prompt = (
            f"任务：{task}\n\n"  # ← 保持原样
            "请立即制定一个**清晰、可执行、阶段性**的Master Plan（用中文编号列表）：\n"
            "1. 任务分解为3-5个主要阶段（Phase）\n"
            "2. 每个阶段：目标 + 主要负责Agent + 预期输出\n"
            "3. 关键检查点（quality gate）和成功指标\n"
            "4. 潜在风险及应对（Benjamin特别关注逻辑漏洞）\n"
            f"5. 总轮次控制建议（不超过{self.max_rounds}轮）\n\n"  # ← 只改这一行，加 f 和 self.
            "要求：极简清晰、可直接作为System Prompt使用。不要多余废话。"
        )
        try:
            plan_response = self.leader.generate_response(
                [{"speaker": "System", "content": plan_prompt}],
                0,
                force_non_stream=True
            )
            return f"📋 【Master Plan】\n{plan_response}"
        except Exception as e:
            logging.warning(f"Plan生成失败: {e}")
            return ""

    def _print_startup_banner(self):
        """打印启动横幅"""
        banner = f"""
    {'=' * 80}
    🚀 MultiAgentSwarm v3.1.0 初始化（智能路由版）
    {'=' * 80}
    📊 基础配置:
       Mode: {self.mode} | Max Rounds: {self.max_rounds}
       Max Concurrent: {self.max_concurrent_agents}
       Reflection: {self.reflection_planning} | Web Search: {self.enable_web_search}
       Vector Memory: {self.vector_memory_enabled}

    ✨ 增强功能:
       🥊 对抗辩论 (Adversarial Debate): {'✅ 启用' if self.enable_adversarial_debate else '❌ 禁用'}
          └─ 策略: {self.adversarial_debate_strategy} | 阈值: {self.adversarial_debate_threshold}
       🎯 元批评 (Meta-Critic): {'✅ 启用' if self.enable_meta_critic else '❌ 禁用'}
       🏭 任务分解 (Task Decomposition): {'✅ 启用' if self.enable_task_decomposition else '❌ 禁用'}
       🧠 知识图谱 (Knowledge Graph): {'✅ 启用' if self.enable_knowledge_graph else '❌ 禁用'}
       📈 自适应反思 (Adaptive Depth): {'✅ 启用' if self.enable_adaptive_depth else '❌ 禁用'}
       🧭 智能路由 (Intelligent Routing): {'✅ 启用' if self.intelligent_routing_enabled else '❌ 禁用'}
          └─ 强制模式: {self.force_complexity or '自动判断'}
    {'=' * 80}
    """
        print(banner)
        logging.info(banner)

    def _load_memory(self) -> Dict:
        """加载持久化记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    memory = json.load(f)
                logging.info(f"📖 加载记忆文件: {self.memory_file} ({len(memory)} keys)")
                return memory
            except Exception as e:
                logging.error(f"加载记忆失败: {e}")
        return {}

    def _save_memory(self, key: str, summary: str):
        """保存记忆到持久化文件"""
        if key not in self.memory:
            self.memory[key] = []

        self.memory[key].append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary[:3000]
        })

        if len(self.memory[key]) > self.max_memory_items:
            self.memory[key] = self.memory[key][-self.max_memory_items:]

        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            logging.info(f"💾 保存记忆: {key}")
        except Exception as e:
            logging.error(f"保存记忆失败: {e}")

    def _decompose_task(self, task: str) -> str:
        """
        ✨ 任务分解器
        将复杂任务分解为子任务并分配给最适合的 Agent
        """
        if not self.enable_task_decomposition or len(self.agents) <= 1:
            return ""

        logging.info("🏭 启动任务分解...")

        decompose_prompt = (
            f"请将以下任务分解为 {min(len(self.agents) + 2, 7)} 个可并行或顺序执行的子任务。\n"
            f"任务: {task}\n\n"
            f"可用 Agent 及其专长：\n"
            + "\n".join([f"- {a.name}: {a.role[:100]}" for a in self.agents])
            + "\n\n格式要求：\n"
              "```json\n"
              '{"subtasks": [{"id": 1, "description": "子任务描述", "assigned_agent": "Agent名称", "priority": "high/medium/low"}]}\n'
              "```"
        )

        try:
            decomposition = self.leader.generate_response(
                [{"speaker": "System", "content": decompose_prompt}],
                0,
                force_non_stream=True
            )

            logging.info(f"📋 任务分解结果:\n{decomposition[:500]}...")
            return f"📋 任务分解:\n{decomposition}"

        except Exception as e:
            logging.error(f"任务分解失败: {e}")
            return ""

    def _adversarial_debate(self, history: List[Dict], round_num: int) -> Tuple[int, str]:
        """
        ✨ 对抗式辩论机制
        三角色并行辩论：Pro（建设者）、Con（批判者）、Judge（裁判）
        返回：(质量分数 0-100, 决策 "continue"/"stop")
        """
        if not self.enable_adversarial_debate:
            return 50, "continue"

        logging.info(f"\n{'─' * 80}")
        logging.info(f"🥊 启动对抗式辩论 (第 {round_num} 轮)")
        logging.info(f"{'─' * 80}")

        # 角色分配（利用现有 Agent）
        debate_agents = {
            "Pro": self.agents[1] if len(self.agents) > 1 else self.leader,  # Harper - 创意建设者
            "Con": self.agents[2] if len(self.agents) > 2 else self.leader,  # Benjamin - 严格批判者
            "Judge": self.leader  # Grok - 综合裁判
        }

        # 辩论提示词
        reflection_prompts = {
            "Pro": (
                "🟢 你是乐观建设者，请：\n"
                "1. 指出本轮讨论的 3 个最大亮点\n"
                "2. 提供 2-3 个建设性改进建议\n"
                "3. 给出质量评分（0-100）"
            ),
            "Con": (
                "🔴 你是严格批判者，请：\n"
                "1. 找出至少 3 个逻辑漏洞、事实风险或遗漏点\n"
                "2. 指出可能导致错误结论的假设\n"
                "3. 给出风险评分（0-100，越高风险越大）"
            ),
            "Judge": (
                "⚖️ 你是最终裁判，请：\n"
                "1. 综合 Pro 和 Con 的观点\n"
                "2. 给出综合质量评分（0-100）\n"
                "3. 决策是否继续讨论（continue/stop）\n"
                "格式：```json\n{\"quality_score\": 0-100, \"decision\": \"continue/stop\", \"reason\": \"原因\"}\n```"
            )
        }

        # 并行执行辩论
        reflections = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_role = {
                executor.submit(
                    agent.generate_response,
                    history.copy(),
                    round_num,
                    system_extra=reflection_prompts[role],
                    critique_previous=True,  # 强制批判模式
                    force_non_stream=True
                ): role
                for role, agent in debate_agents.items()
            }

            for future in as_completed(future_to_role):
                role = future_to_role[future]
                try:
                    reflections[role] = future.result()
                    logging.info(f"✅ {role} 完成辩论")
                except Exception as e:
                    logging.error(f"❌ {role} 辩论失败: {e}")
                    reflections[role] = f"[执行失败: {str(e)}]"

        # Meta-Critic 综合评估
        if self.enable_meta_critic:
            synthesis_prompt = (
                f"🎯 Meta-Critic 综合评估\n\n"
                f"Pro 观点:\n{reflections.get('Pro', 'N/A')[:800]}\n\n"
                f"Con 观点:\n{reflections.get('Con', 'N/A')[:800]}\n\n"
                f"Judge 观点:\n{reflections.get('Judge', 'N/A')[:800]}\n\n"
                f"请综合三方辩论，给出最终决策（JSON 格式）：\n"
                f"```json\n"
                f'{{"quality_score": 0-100, "decision": "continue/stop", "reason": "综合原因", "key_issues": ["问题1", "问题2"]}}\n'
                f"```"
            )

            final_eval = self.leader.generate_response(
                history + [{"speaker": "System", "content": synthesis_prompt}],
                round_num,
                force_non_stream=True
            )
        else:
            final_eval = reflections.get("Judge", "{}")

        # 解析决策
        try:
            eval_json = json.loads(
                final_eval.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            quality_score = eval_json.get("quality_score", 50)
            decision = eval_json.get("decision", "continue").lower()

            logging.info(f"📊 辩论结果: 质量分数 {quality_score}/100 | 决策: {decision}")
            logging.info(f"💡 原因: {eval_json.get('reason', 'N/A')[:200]}")

            return quality_score, decision

        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"❌ 解析辩论结果失败: {e}")
            return 50, "continue"

    def solve(
            self,
            task: str,
            use_memory: bool = False,
            memory_key: str = "default",
            image_paths: Optional[List[str]] = None,
            force_complexity: Optional[str] = None,
            stream_callback=None,
            log_callback=None
    ) -> str:
        """
        解决任务的主入口（智能路由版 v3.1.0 + 取消支持）

        Args:
            task: 任务描述
            use_memory: 是否使用持久化记忆
            memory_key: 记忆键名
            image_paths: 图片路径列表（最多 max_images 张）
            force_complexity: 强制指定复杂度 "simple"/"medium"/"complex"（优先级最高，用于调试）
            stream_callback: 流式输出回调 func(agent_name, content)
            log_callback: 日志回调 func(message)

        Returns:
            最终答案字符串
        """
        # ✅ 任务开始时重置取消标志
        self._reset_cancel_flag()

        tracker = TimeTracker()
        # 🔥 防御性自动多模态图片检测（飞书/文本路径自动生效）
        if (image_paths is None or len(image_paths) == 0) and isinstance(task, str):
            import re
            matches = re.findall(r'(?:uploads[/\\]|/uploads/)([^"\s]+\.(?:png|jpg|jpeg|gif|bmp))', task, re.IGNORECASE)
            if matches:
                image_paths = []
                upload_dir = Path("uploads")
                for m in matches:
                    candidate = upload_dir / m.strip()
                    if candidate.exists():
                        image_paths.append(str(candidate))
                if image_paths:
                    logging.info(f"🔍 自动识别图片附件（飞书/文本路径）：{image_paths}")
                    if log_callback:
                        log_callback(f"📸 自动加载 {len(image_paths)} 张图片为多模态输入")
        tracker.start()
        logging.info(f"\n{'=' * 80}")
        logging.info(f"📋 新任务: {task[:100]}{'...' if len(task) > 100 else ''}")

        # ✅ 检查点 0：任务开始前
        if self._check_cancellation():
            return "⏸️ 任务已被用户取消"

        # 🔥【核心修复】仅 12 行，彻底解决历史污染
        classification_task = task
        if isinstance(task, str) and "=== 💬 当前问题 ===" in task:
            try:
                classification_task = task.split("=== 💬 当前问题 ===")[-1].strip()
                if classification_task.startswith("User:") or classification_task.startswith("User："):
                    classification_task = classification_task.split(":", 1)[-1].strip()
                classification_task = classification_task[:300]
            except Exception:
                classification_task = task[:200]

        logging.info(f"📊 分类使用纯查询: {classification_task[:80]}{'...' if len(classification_task) > 80 else ''}")

        try:
            # ✅ 检查点 1：分类前检查
            if self._check_cancellation():
                return "⏸️ 任务已被用户取消"

            if self.intelligent_routing_enabled:
                complexity = (force_complexity or
                              self.force_complexity or
                              self._classify_task_complexity(classification_task))
            else:
                complexity = "complex"

            tracker.checkpoint("1️⃣ 任务分类")

            if log_callback:
                log_callback(f"📊 任务复杂度: {complexity.upper()}")

        except Exception as e:
            logging.error(f"❌ 任务分类失败: {e}，回退到完整模式")
            if log_callback:
                log_callback(f"⚠️ 任务分类失败，使用完整模式")
            complexity = "complex"

        # 处理图像
        if image_paths:
            image_paths = image_paths[:self.max_images]
            logging.info(f"📷 处理 {len(image_paths)} 张图片")
            if log_callback:
                log_callback(f"📷 处理 {len(image_paths)} 张图片")

        # 🔥 附件路径强规范化（兼容 Windows \ 和动态 UUID 文件名污染）
        import re
        if "uploads" in task:
            # 统一转正斜杠 + 清理多余路径
            task = re.sub(r'uploads\\?/?', 'uploads/', task)          # 统一前缀
            task = re.sub(r'uploads/[^ \n]+', lambda m: m.group(0).replace('\\', '/'), task)
            logging.info(f"📎 已规范化附件路径 → {task.split('uploads/')[-1]}")

        history: List[Dict] = []

        # 构建初始 history
        if image_paths:
            image_content = [{"type": "text", "text": task}]
            for idx, path in enumerate(image_paths, 1):
                if not os.path.exists(path):
                    logging.warning(f"⚠️ 图片不存在: {path}")
                    continue
                try:
                    mime_type, _ = mimetypes.guess_type(path)
                    if not mime_type or not mime_type.startswith("image/"):
                        mime_type = "image/jpeg"
                    with open(path, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode('utf-8')
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                    })
                except Exception as e:
                    logging.error(f"  ❌ 读取图片失败 {path}: {e}")
            history.append({"speaker": "User", "content": image_content})
        else:
            history.append({"speaker": "User", "content": task})

        # ✨ PrimalClaw：注入相关历史记忆（QMD检索）
        if use_memory and hasattr(self, 'primal_memory'):
            relevant = self.primal_memory.get_relevant_memory(task)
            if relevant:
                history.insert(0, {"speaker": "System", "content": f"📚 Primal记忆（相关经验）：\n{relevant}"})
                if log_callback:
                    log_callback("📚 已注入Primal相关记忆")

        # ✅ 检查点 2：执行前检查
        if self._check_cancellation():
            return "⏸️ 任务已被用户取消"

        # ✨✨✨ 三级路由执行（带异常降级）✨✨✨
        final_answer = ""
        execution_mode = complexity

        try:
            if complexity == "simple":
                final_answer = self._solve_simple(
                    task, history,
                    stream_callback, log_callback
                )

            elif complexity == "medium":
                final_answer = self._solve_medium(
                    task, history, tracker,
                    stream_callback, log_callback
                )

            else:  # complex
                final_answer = self._solve_complex(
                    task, history, tracker, use_memory, memory_key,
                    stream_callback, log_callback
                )

        except Exception as e:
            # ✅ 检查是否因取消而异常
            if self._check_cancellation():
                return "⏸️ 任务已被用户取消"

            logging.error(f"❌ {complexity.upper()} 模式执行失败: {e}")
            print(f"\n{'!' * 80}")
            print(f"⚠️  {complexity.upper()} 模式执行失败: {str(e)[:100]}")
            print(f"🔄 自动降级到 COMPLEX 完整模式...")
            print(f"{'!' * 80}\n")

            if log_callback:
                log_callback(f"⚠️ {complexity.upper()} 模式失败，降级到 COMPLEX")

            execution_mode = "complex (降级)"
            try:
                final_answer = self._solve_complex(
                    task, history, tracker, use_memory, memory_key,
                    stream_callback, log_callback
                )
            except Exception as fallback_error:
                if self._check_cancellation():
                    return "⏸️ 任务已被用户取消"

                logging.error(f"❌ 降级执行也失败: {fallback_error}")
                final_answer = f"[系统错误] 任务执行失败:\n原始错误: {str(e)}\n降级错误: {str(fallback_error)}"
                if log_callback:
                    log_callback(f"❌ 系统错误: 任务执行失败")

        # ✅ 检查点 3：执行后检查
        if self._check_cancellation():
            partial_result = final_answer[:500] + "..." if len(final_answer) > 500 else final_answer
            return f"⏸️ 任务已被取消\n\n**部分结果**：\n{partial_result}" if final_answer else "⏸️ 任务已被用户取消"

        # ===== 统一输出（三种模式共用） =====
        print("\n" + "=" * 100)
        print(f"🎯 【最终答案】（执行模式: {execution_mode.upper()}）")
        print("=" * 100)
        print(final_answer)
        print("=" * 100)

        if log_callback:
            log_callback(f"✅ 任务完成 (模式: {execution_mode.upper()})")

        # 知识图谱蒸馏（仅 complex 模式）
        if complexity == "complex" and self.knowledge_graph:
            kg_summary = self.knowledge_graph.distill(max_items=10)
            if kg_summary:
                print("\n" + "─" * 100)
                print("🧠 【知识图谱蒸馏】")
                print("─" * 100)
                print(kg_summary)
                print("─" * 100)

        # 时间统计
        print(tracker.summary())
        logging.info(tracker.summary())

        return final_answer

    def _solve_complex(
            self,
            task: str,
            history: List[Dict],
            tracker: TimeTracker,
            use_memory: bool,
            memory_key: str,
            stream_callback=None,
            log_callback=None
    ) -> str:
        """
        🔴 完整模式：全功能协作（新增取消检查）
        """
        logging.info("🔴 执行完整模式（全功能协作）")
        print(f"\n{'=' * 80}")
        print("🔴 检测到复杂任务，启用全功能协作模式")
        print(f"{'=' * 80}\n")

        if log_callback:
            log_callback("🔴 执行完整模式（全功能协作）")

        # ✅ 检查点：开始前
        if self._check_cancellation():
            return "⏸️ 任务在开始前被取消"

        # ===== 任务分解 =====
        if self.enable_task_decomposition and self.mode == "intelligent":
            if self._check_cancellation():
                return "⏸️ 任务在分解前被取消"

            decomposition = self._decompose_task(task)
            if decomposition:
                history.insert(0, {"speaker": "System", "content": decomposition})
                tracker.checkpoint("2️⃣ 任务分解")
                if log_callback:
                    log_callback("📋 任务分解完成")

        # ✅ 检查点：Master Plan 生成前
        if self._check_cancellation():
            return "⏸️ 任务在规划前被取消"

        # 🔥【新增】显式Master Plan
        plan = self._generate_detailed_plan(task, history)
        if plan:
            history.insert(0, {"speaker": "System", "content": plan})
            if log_callback:
                log_callback("📋 Master Plan 已生成并注入")
            tracker.checkpoint("2.5️⃣ Master Plan 生成")

        # ===== 加载历史记忆 =====
        if use_memory and memory_key in self.memory:
            memory_text = "\n".join([
                f"- {item['summary']}"
                for item in self.memory[memory_key][-5:]
            ])
            history.insert(0, {
                "speaker": "System",
                "content": f"📚 历史记忆（{memory_key}）：\n{memory_text}"
            })
            if log_callback:
                log_callback(f"📚 加载历史记忆: {memory_key}")

        # ✅ 检查点：主循环前
        if self._check_cancellation():
            return "⏸️ 任务在讨论开始前被取消"

        # ===== 主循环（多轮讨论 + 辩论）=====
        round_num = 0
        previous_quality = 0

        while True:
            # ✅ 关键检查点：每轮开始前
            if self._check_cancellation():
                logging.info(f"🛑 第 {round_num} 轮被取消")
                if log_callback:
                    log_callback(f"⏸️ 第 {round_num} 轮被取消")
                break

            round_num += 1
            if round_num > self.max_rounds:
                logging.info(f"⏸️  达到最大轮次 {self.max_rounds}，停止讨论")
                if log_callback:
                    log_callback(f"⏸️ 达到最大轮次 {self.max_rounds}")
                break

            logging.info(f"\n{'─' * 80}")
            logging.info(f"🔄 第 {round_num} 轮讨论开始")
            logging.info(f"{'─' * 80}")

            if log_callback:
                log_callback(f"🔄 第 {round_num}/{self.max_rounds} 轮讨论开始")

            round_start = time.time()

            # 并发执行 Agent 讨论
            with ThreadPoolExecutor(max_workers=self.max_concurrent_agents) as executor:
                future_to_agent = {
                    executor.submit(
                        agent.generate_response,
                        history.copy(),
                        round_num,
                        critique_previous=(round_num > 1 and self.enable_adversarial_debate),
                        log_callback=log_callback
                    ): agent
                    for agent in self.agents
                }

                for future in as_completed(future_to_agent):
                    # ✅ 检查点：每个 Agent 完成后
                    if self._check_cancellation():
                        logging.info(f"🛑 Agent 讨论被取消，正在清理...")
                        # 取消未完成的任务
                        for f in future_to_agent:
                            f.cancel()
                        break

                    agent = future_to_agent[future]
                    try:
                        contribution = future.result()
                        history.append({
                            "speaker": agent.name,
                            "content": contribution
                        })

                        # 知识图谱提取
                        if self.knowledge_graph and len(contribution) > 50:
                            words = contribution.split()
                            for i, word in enumerate(words[:50]):
                                if word.istitle() and len(word) > 3:
                                    self.knowledge_graph.add_entity(
                                        word, "concept", f"{agent.name}提及"
                                    )
                    except Exception as e:
                        logging.error(f"❌ {agent.name} 执行失败: {e}")
                        if log_callback:
                            log_callback(f"❌ {agent.name} 执行失败")

            # ✅ 检查点：本轮讨论后
            if self._check_cancellation():
                logging.info(f"🛑 第 {round_num} 轮讨论后被取消")
                if log_callback:
                    log_callback(f"⏸️ 第 {round_num} 轮讨论被取消")
                break

            round_elapsed = time.time() - round_start
            tracker.checkpoint(f"3️⃣ 第{round_num}轮讨论 ({round_elapsed:.1f}秒)")

            # ✨ 自动压缩历史
            history = self._compress_history(history)
            tracker.checkpoint(f"3.5️⃣ 历史压缩")

            # ===== 对抗式辩论与自适应反思 =====
            if (self.mode == "intelligent"
                    and self.reflection_planning
                    and self.enable_adversarial_debate
                    and self._should_trigger_debate(round_num, previous_quality)):
                # ✅ 检查点：辩论前
                if self._check_cancellation():
                    logging.info(f"🛑 辩论前被取消")
                    if log_callback:
                        log_callback(f"⏸️ 辩论被取消")
                    break

                if log_callback:
                    log_callback(f"🥊 启动对抗式辩论 (第 {round_num} 轮)")

                quality_score, decision = self._adversarial_debate(history, round_num)
                tracker.checkpoint(f"4️⃣ 第{round_num}轮辩论")
                # ✨【动态更新 Master Plan】——对应架构图“更新prompt”循环箭头
                if round_num % 3 == 0 or quality_score < 75:  # 每3轮或质量低时刷新
                    new_plan = self._generate_detailed_plan(task, history)
                    if new_plan:
                        # 替换旧 Plan（保持 history 干净）
                        for i, msg in enumerate(history):
                            if "Master Plan" in str(msg.get("content", "")):
                                history[i] = {"speaker": "System", "content": new_plan}
                                break
                        else:
                            history.insert(0, {"speaker": "System", "content": new_plan})
                        if log_callback:
                            log_callback("📋 Master Plan 已动态刷新")
                        tracker.checkpoint(f"2.6️⃣ Plan 动态更新")

                # ✅ 检查点：辩论后
                if self._check_cancellation():
                    logging.info(f"🛑 辩论后被取消")
                    if log_callback:
                        log_callback(f"⏸️ 辩论被取消")
                    break

                if log_callback:
                    log_callback(f"📊 辩论结果: 质量 {quality_score}/100, 决策 {decision}")

                if self.enable_adaptive_depth:
                    # 质量达标立即停止
                    if quality_score >= self.reflection_quality_threshold:
                        logging.info(f"✅ 质量达标 ({quality_score} >= {self.reflection_quality_threshold})，停止讨论")
                        if log_callback:
                            log_callback(f"✅ 质量达标 ({quality_score}分)")
                        break

                    # 裁判建议停止 + 质量可接受
                    if decision == "stop" and quality_score >= self.stop_quality_threshold:
                        logging.info(f"✅ 裁判建议停止 + 质量可接受 ({quality_score} >= {self.stop_quality_threshold})")
                        if log_callback:
                            log_callback(f"✅ 裁判建议停止 (质量 {quality_score}分)")
                        break

                    # 质量收敛判定
                    if round_num > 1 and previous_quality > 0:
                        quality_delta = quality_score - previous_quality
                        if abs(quality_delta) < self.quality_convergence_delta:
                            logging.info(
                                f"📉 质量收敛 (Δ={quality_delta:.1f} < {self.quality_convergence_delta})，停止讨论")
                            if log_callback:
                                log_callback(f"📉 质量收敛 (Δ={quality_delta:.1f})")
                            break

                    previous_quality = quality_score
                else:
                    # 非自适应模式：仅听从裁判决策
                    if decision == "stop":
                        logging.info("✅ 裁判建议停止，结束讨论")
                        if log_callback:
                            log_callback("✅ 裁判建议停止")
                        break

        # ✅ 最终检查：综合前
        if self._check_cancellation():
            # 返回部分结果
            if history and len(history) > 0:
                last_content = ""
                for h in reversed(history):
                    if h.get("speaker") not in ["System", "User"]:
                        last_content = h.get("content", "")
                        break

                partial = last_content[:800] + "..." if len(last_content) > 800 else last_content
                return f"⏸️ 任务已被取消\n\n**部分结果**（来自第 {round_num} 轮讨论）：\n\n{partial}"
            else:
                return "⏸️ 任务已被取消，暂无结果"

        # ===== 最终综合 =====
        if log_callback:
            log_callback("🎯 开始最终综合")

        kg_context = ""
        if self.knowledge_graph:
            kg_context = self.knowledge_graph.distill(max_items=10)
            if kg_context:
                history.insert(-1, {"speaker": "System", "content": kg_context})

        history.append({
            "speaker": "System",
            "content": (
                "请综合以上全部讨论，给出**最准确、最完整、最高质量**的最终答案。\n\n"

                "**智能文件交付决策框架（必须严格执行）：**\n"
                "在生成最终答案前，请先进行以下内部思考（不需要输出思考过程）：\n"
                "1. 用户本次查询是否希望得到一个**可保存、可下载、可分享的结构化文件**？\n"
                "   常见信号包括但不限于：报告、整理、总结、分析、编辑、修改、导出、生成文档、给我下载链接、保存版本、export、save as 等。\n"
                "2. 如果用户想要的是持久化输出（尤其是编辑后、修改后、整理后的结果），则**必须先调用 write_file 工具**生成 Markdown 文件。\n"
                "3. 文件名建议格式：`{核心主题}_{今天日期}.md`（例如 iran_news_20260302.md 或 edited_report_20260302.md）。\n"
                "4. 生成文件成功后，必须在答案最底部添加清晰的下载链接。\n\n"

                "**决策原则（更智能、更鲁棒）：**\n"
                "- 如果用户明确或隐含需要『整理成报告/编辑后版本/修改结果/给我下载/生成文档』等 → **强制生成文件**\n"
                "- 如果是普通解释、聊天、快速问答、脑暴想法 → **直接输出文本答案**\n"
                "- 当意图模糊时，优先倾向于生成文件（用户体验更好）\n\n"

                "**当需要生成文件时的标准输出格式（必须严格遵守）：**\n\n"
                "[这里输出完整、专业、美观的报告/文档正文]\n\n"
                "**📥 下载生成的文件**\n\n"
                "- [文件名_20260302.md](/uploads/文件名_20260302.md)\n\n"

                "现在请根据用户本次查询的**真实意图**，智能决定是否需要生成可下载文件，然后直接输出最终答案。"
            )
        })

        # ✅ 检查点：最终生成前
        if self._check_cancellation():
            return "⏸️ 任务在最终综合前被取消"

        final_answer = self.leader.generate_response(
            history,
            round_num + 1,
            force_non_stream=False,
            stream_callback=stream_callback,
            log_callback=log_callback
        )

        tracker.checkpoint("5️⃣ 最终综合")

        # ✅ 检查点：最终生成后
        if self._check_cancellation():
            partial = final_answer[:800] + "..." if len(final_answer) > 800 else final_answer
            return f"⏸️ 任务已被取消\n\n**部分结果**：\n\n{partial}" if final_answer else "⏸️ 任务已被取消"

        # ===== 保存记忆 =====
        if use_memory:
            try:
                if log_callback:
                    log_callback("💾 保存记忆中...")

                summary = self.leader.generate_response(
                    history + [{
                        "speaker": "System",
                        "content": "请用500字总结：核心结论、关键发现、可复用经验、遗留问题"
                    }],
                    round_num + 1,
                    force_non_stream=True
                )
                self._save_memory(memory_key, summary)

                # ✨ PrimalClaw：保存结构化记忆 + 自动提炼
                if hasattr(self, 'primal_memory'):
                    self.primal_memory.save_episode(task, history, final_answer, memory_key)
                    self.primal_memory.decay()

                # 向量记忆
                if self.vector_memory:
                    self.vector_memory.add(
                        summary,
                        metadata={"task": task[:100], "memory_key": memory_key}
                    )

                tracker.checkpoint("6️⃣ 保存记忆")
                if log_callback:
                    log_callback("💾 记忆保存完成")

            except Exception as e:
                logging.error(f"❌ 保存记忆失败: {e}")
                if log_callback:
                    log_callback(f"⚠️ 记忆保存失败")

        return final_answer

    def _should_trigger_debate(self, round_num: int, prev_quality: int) -> bool:
        """按需触发对抗辩论（最终推荐版 - 最清晰）"""
        # 全局开关关闭时直接跳过
        if not self.enable_adversarial_debate:
            return False

        strategy = self.adversarial_debate_strategy  # ← 直接使用你配置的变量

        if strategy == "always":
            return True
        elif strategy == "every_n_rounds":
            return round_num % self.adversarial_debate_interval == 0
        else:  # quality_based（强烈推荐，默认策略）
            threshold = self.adversarial_debate_threshold
            # 第一轮必须辩论（建立质量基准）
            # 之后只有质量低于阈值才触发 → 大幅节省开销
            return round_num == 1 or (prev_quality > 0 and prev_quality < threshold)

    def _classify_task_complexity(self, task: str) -> str:
        """
        ✨ 智能任务分类器
        返回: "simple" | "medium" | "complex"
        """
        # 快速规则过滤（0ms，无API调用）
        task_lower = task.lower().strip()

        # 新增：对话跟进场景（“继续”“详细说说”“再解释一下”）强制 medium，防止简单问题被误判 complex
        if any(kw in task_lower for kw in ["继续", "详细", "再", "然后", "为什么", "怎么", "解释一下",
                                           "more details", "elaborate", "next"]):
            if len(task) < 150:  # 短跟进一定是 medium
                logging.info("🟡 对话跟进 → MEDIUM 模式")
                return "medium"

        # 简单任务特征（直接判定）
        simple_patterns = [
            # 问候类
            len(task) < 20 and any(word in task_lower for word in ["你好", "hi", "hello", "hey", "谢谢", "thank", "嘿", "thank", "ok", "好的"]),
            # 简单问答
            task.endswith("?") and len(task) < 30,
            # 单一查询
            task.startswith(("什么是", "who is", "when", "where")) and len(task) < 50,
        ]

        if any(simple_patterns):
            logging.info("🟢 任务分类: SIMPLE (规则匹配)")
            return "simple"

        # 复杂任务特征（直接判定）
        complex_patterns = [
            # 明确要求协作
            any(word in task_lower for word in ["分析报告", "深度", "对比", "评估", "战略", "方案", "代码审查"]),
            # 多步骤任务
            task.count("并且") + task.count("然后") + task.count("同时") + task.count("and then") >= 2,
            # 文件操作
            any(word in task_lower for word in ["写入", "保存", "生成文件", "write to", "save to"]),
            # 长文本
            len(task) > 200,
        ]

        if any(complex_patterns):
            logging.info("🔴 任务分类: COMPLEX (规则匹配)")
            return "complex"

        # 中等复杂度：用 Leader 快速判断（单次 API 调用，~0.5秒）
        try:
            classify_prompt = (
                f"任务: {task}\n\n"
                "请判断此任务的复杂度（仅回复一个词）：\n"
                "- simple: 简单问候/单句问答/查询\n"
                "- medium: 需要分析但不复杂（如解释概念、简单建议）\n"
                "- complex: 需要深度分析/多步骤/协作\n\n"
                "回复格式: 仅输出 simple/medium/complex"
            )

            response = self.leader.client.chat.completions.create(
                model=self.leader.model,
                messages=[
                    {"role": "system", "content": "你是任务复杂度分类器，仅回复 simple/medium/complex"},
                    {"role": "user", "content": classify_prompt}
                ],
                temperature=0.0,
                max_tokens=10,
                stream=False
            )

            # ✅ 核心修复：处理 None 返回值
            content = response.choices[0].message.content

            if content is None or not content.strip():
                logging.warning("⚠️ API 返回空值，默认使用 medium")
                return "medium"

            complexity = content.strip().lower()

            if complexity in ["simple", "medium", "complex"]:
                logging.info(f"🟡 任务分类: {complexity.upper()} (AI判断)")
                return complexity
            else:
                logging.warning(f"⚠️ AI分类返回无效值: {complexity}，默认使用 medium")
                return "medium"

        except Exception as e:
            logging.error(f"❌ 任务分类失败: {e}，默认使用 medium")
            return "medium"

    def _solve_simple(
            self,
            task: str,
            history: List[Dict],
            stream_callback=None,  # ✅ 新增
            log_callback=None  # ✅ 新增
    ) -> str:
        """
        🟢 简单模式：单 Agent 直接回答
        """
        logging.info("🟢 执行简单模式（单Agent直答）")
        print(f"\n{'=' * 80}")
        print("🟢 检测到简单任务，使用快速模式")
        print(f"{'=' * 80}\n")

        # ✅ 发送日志
        if log_callback:
            log_callback("🟢 执行简单模式")

        # 直接用 Leader 回答（允许流式输出）
        answer = self.leader.generate_response(
            history,
            round_num=1,
            system_extra="请简洁、直接地回答用户问题。",
            force_non_stream=False,
            stream_callback=stream_callback,  # ✅ 传递
            log_callback=log_callback  # ✅ 传递
        )

        return answer

    def _solve_medium(
            self,
            task: str,
            history: List[Dict],
            tracker: TimeTracker,
            stream_callback=None,  # ✅ 新增
            log_callback=None  # ✅ 新增
    ) -> str:
        """
        🟡 中等模式：2 Agents + 单轮讨论
        """
        logging.info("🟡 执行中等模式（2 Agents + 1轮）")
        print(f"\n{'=' * 80}")
        print("🟡 检测到中等任务，使用精简协作模式")
        print(f"{'=' * 80}\n")

        # ✅ 发送日志
        if log_callback:
            log_callback("🟡 执行中等模式（2 Agents + 1轮）")

        # 选择 2 个最适合的 Agent（Leader + 1个专家）
        selected_agents = [self.leader]

        if len(self.agents) > 1:
            # 简单策略：选择第二个Agent（通常是创意/分析专家）
            selected_agents.append(self.agents[1])

        # 单轮并发讨论
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_agent = {
                executor.submit(
                    agent.generate_response,
                    history.copy(),
                    1,
                    log_callback=log_callback  # ✅ 传递日志（不传 stream 避免混乱）
                ): agent
                for agent in selected_agents
            }

            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    contribution = future.result()
                    history.append({
                        "speaker": agent.name,
                        "content": contribution
                    })
                except Exception as e:
                    logging.error(f"❌ {agent.name} 执行失败: {e}")
                    if log_callback:
                        log_callback(f"❌ {agent.name} 执行失败")

        tracker.checkpoint("2️⃣ 单轮讨论")

        # Leader 快速综合 + 轻量智能文件决策（核心优化）
        history.append({
            "speaker": "System",
            "content": (
                "请简洁但高质量地综合以上观点，给出清晰答案。\n\n"

                "**轻量文件交付决策（必须执行）：**\n"
                "在输出前快速判断：用户本次查询是否希望得到一个**可保存、可下载的结构化文件**？（信号包括：报告、整理、总结、编辑、修改、导出、给我下载、生成文档等）。\n"
                "- 如果是 → **必须先调用 write_file 工具**生成 Markdown 文件（文件名建议带日期，如 topic_20260302.md）\n"
                "- 如果不是 → 直接输出清晰答案即可\n\n"

                "**当需要生成文件时的格式（必须包含）：**\n\n"
                "[这里是完整答案正文]\n\n"
                "**📥 下载生成的文件**\n\n"
                "- [文件名_20260302.md](/uploads/文件名_20260302.md)\n\n"

                "现在请根据用户真实意图，直接输出最终答案。"
            )
        })

        final_answer = self.leader.generate_response(
            history,
            2,
            force_non_stream=False,
            stream_callback=stream_callback,  # ✅ 传递流式
            log_callback=log_callback  # ✅ 传递日志
        )

        tracker.checkpoint("3️⃣ 快速综合")

        return final_answer

    def _compress_history(self, history: List[Dict], max_tokens_approx: int = 20000) -> List[Dict]:
        """极简滚动压缩：只保留最近N轮 + Primal摘要"""
        if len(history) < 8:  # 短历史不压缩
            return history

        # 估算tokens（粗略：中文≈2char=1token）
        total_chars = sum(len(str(h.get("content", ""))) for h in history)
        if total_chars < max_tokens_approx * 2:
            return history

        # 保留：系统提示 + 最近3轮 + Primal记忆 + 最终答案
        compressed = [h for h in history if h["speaker"] == "System" and "Primal记忆" in str(h.get("content", ""))]
        compressed.extend(history[-6:])  # 保留最近3轮讨论（每轮2条左右）

        # 如果还是太长，让Leader做一次超短总结
        if len(str(compressed)) > max_tokens_approx * 3:
            summary_prompt = "请用800字以内总结以上全部历史，只保留关键决策、教训和结论，不要重复细节。"
            short_summary = self.leader.generate_response(
                [{"speaker": "System", "content": summary_prompt}] + compressed[-4:],
                0, force_non_stream=True
            )
            compressed = [{"speaker": "System", "content": f"📜 历史压缩总结：\n{short_summary}"}]

        return compressed

    def nightly_reflect(self):
        """夜间/任务后反思（可手动或定时调用）"""
        if not hasattr(self, 'primal_memory'):
            return
        # 让Leader做一次全局反思（复用现有generate_response）
        prompt = "请扫描/memory/kb/所有lessons，提炼≤5条跨任务通用智慧，更新到kb/decisions/general.md"
        self.leader.generate_response([{"speaker": "System", "content": prompt}], 0, force_non_stream=True)


# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        swarm = MultiAgentSwarm()

        print("\n" + "🧪" * 40)
        print("开始测试智能路由功能")
        print("🧪" * 40 + "\n")

        # ===== 测试 1：简单模式 =====
        print("\n📝 测试 1: 简单问候（预期: SIMPLE 模式，~1-2秒）")
        print("=" * 80)
        msg = swarm.solve("你好，今天天气怎么样？")
        print('测试1 回答:\n', msg)
        input("\n按回车继续下一个测试...")

        # ===== 测试 2：中等模式 =====
        print("\n📝 测试 2: 概念解释（预期: MEDIUM 模式，~10-20秒）")
        print("=" * 80)
        msg = swarm.solve("请解释一下什么是 Transformer 注意力机制")
        print('测试2 回答:\n', msg)
        input("\n按回车继续下一个测试...")

        # ===== 测试 3：复杂模式 =====
        print("\n📝 测试 3: 深度分析（预期: COMPLEX 模式，~40-60秒）")
        print("=" * 80)
        msg = swarm.solve(
            "请写一篇关于大语言模型训练技术的深度分析报告，"
            "包括数据准备、模型架构、训练策略的对比分析",
            use_memory=True,
            memory_key="llm_training"
        )
        print('测试3 回答:\n', msg)
        input("\n按回车继续下一个测试...")

        # ===== 测试 4：强制模式 =====
        print("\n📝 测试 4: 强制使用 COMPLEX 模式处理简单任务")
        print("=" * 80)
        msg = swarm.solve("你好", force_complexity="complex")
        print('测试4 回答:\n', msg)

        # 示例5：图像分析（需要提供真实图片路径）
        # swarm.solve(
        #     "请分析这些图片中的代码问题",
        #     image_paths=["./screenshot1.png", "./screenshot2.png"]
        # )

        print("\n" + "✅" * 40)
        print("所有测试完成！")
        print("✅" * 40 + "\n")

    except Exception as e:
        logging.error(f"❌ 程序异常: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")