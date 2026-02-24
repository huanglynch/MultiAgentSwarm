#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多智能体协作系统 (Multi-Agent Swarm) v2.9.2
- 支持并发控制（max_concurrent_agents）
- 支持向量记忆（优先使用缓存模型）
- 支持耗时统计
- 支持多模态输入（文本 + 图像）
"""

import yaml
import logging
import os
import glob
import importlib.util
import requests
import random
import time
import threading
import base64
import mimetypes
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from openai import OpenAI
import json
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

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
            vector_memory: Optional[VectorMemory] = None
    ):
        self.name = config["name"]
        self.role = config["role"]
        self.shared_knowledge = shared_knowledge
        self.vector_memory = vector_memory

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

            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": str(result)
            }
        except Exception as e:
            logging.error(f"工具执行失败 {func_name}: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": f"Tool error: {str(e)}"
            }

    def generate_response(
            self,
            history: List[Dict],
            round_num: int,
            system_extra: str = "",
            force_non_stream: bool = False
    ) -> str:
        """生成响应"""
        start_time = time.time()

        use_stream = self.stream and not force_non_stream and not self.tools

        # 构建系统提示词
        system_prompt = (
            f"{self.role}\n"
            f"{self.shared_knowledge}\n"
            f"{system_extra}\n"
            "你是多智能体协作团队的一员，请提供有价值、准确、有深度的贡献。"
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

        try:
            if use_stream:
                print(f"\n💬 【{self.name}】正在思考... ", end="", flush=True)

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

            # 流式输出
            if use_stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        print(delta, end="", flush=True)
                        full_response += delta
                print()
            else:
                full_response = response.choices[0].message.content or ""

            # 工具调用处理
            if (not use_stream and
                    hasattr(response.choices[0].message, 'tool_calls') and
                    response.choices[0].message.tool_calls):

                messages.append(response.choices[0].message.model_dump())

                for tool_call in response.choices[0].message.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append(tool_result)

                # 获取工具调用后的最终响应
                final_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                full_response = final_resp.choices[0].message.content or ""

            # 计算并显示耗时
            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed:.2f}秒" if elapsed < 60 else f"{int(elapsed // 60)}分{elapsed % 60:.1f}秒"

            if not use_stream:
                print(f"⏱️  【{self.name}】响应完成 | 耗时: {elapsed_str}")

            logging.info(f"⏱️  {self.name} 响应耗时: {elapsed_str}")

            return full_response.strip()

        except Exception as e:
            elapsed = time.time() - start_time
            err = f"[Error in {self.name}]: {str(e)}"
            logging.error(f"{err} | 耗时: {elapsed:.2f}秒")
            print(f"❌ 【{self.name}】执行失败 | 耗时: {elapsed:.2f}秒")
            return err


# ====================== 主类 MultiAgentSwarm ======================
class MultiAgentSwarm:
    """多智能体群智慧框架 v2.9.2"""

    def __init__(self, config_path: str = "swarm_config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ 配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # OpenAI 配置
        oai = cfg.get("openai", {})
        self.default_model = oai.get("default_model", "gpt-4o-mini")
        self.default_max_tokens = oai.get("default_max_tokens", 4096)

        # Swarm 配置
        swarm = cfg.get("swarm", {})
        self.mode = swarm.get("mode", "fixed")
        self.max_rounds = swarm.get("max_rounds", 3 if self.mode == "fixed" else 10)
        self.max_concurrent_agents = swarm.get("max_concurrent_agents", 2)  # ✅ 新增
        self.reflection_planning = swarm.get("reflection_planning", True)
        self.enable_web_search = swarm.get("enable_web_search", False)
        self.max_images = swarm.get("max_images", 2)

        self.log_file = swarm.get("log_file", "swarm.log")
        self.skills_dir = swarm.get("skills_dir", "skills")
        self.memory_file = swarm.get("memory_file", "memory.json")
        self.max_memory_items = swarm.get("max_memory_items", 50)

        self.max_reflection_rounds = swarm.get("max_reflection_rounds", 3)
        self.reflection_quality_threshold = swarm.get("reflection_quality_threshold", 9)
        self.stop_quality_threshold = swarm.get("stop_quality_threshold", 8)
        self.quality_convergence_delta = swarm.get("quality_convergence_delta", 0.5)

        # ✅ 向量记忆配置
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

        logging.info(f"{'=' * 80}")
        logging.info(f"🚀 MultiAgentSwarm v2.9.2 初始化")
        logging.info(f"   Mode: {self.mode} | Max Rounds: {self.max_rounds}")
        logging.info(f"   Max Concurrent: {self.max_concurrent_agents}")  # ✅ 新增
        logging.info(f"   Reflection: {self.reflection_planning} | Web Search: {self.enable_web_search}")
        logging.info(f"   Vector Memory: {self.vector_memory_enabled}")  # ✅ 新增
        logging.info(f"{'=' * 80}")

        # 加载 Skills
        self.tool_registry, self.shared_knowledge = load_skills(self.skills_dir)

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

        # 初始化持久化记忆（必须在向量记忆之前）
        self.memory = self._load_memory()

        # ✅ 初始化向量记忆
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

        # 初始化 Agents
        self.agents = []
        for a_cfg in cfg.get("agents", [])[:swarm.get("num_agents", 4)]:
            agent = Agent(
                a_cfg,
                self.default_model,
                self.default_max_tokens,
                self.tool_registry,
                self.shared_knowledge,
                self.vector_memory
            )
            self.agents.append(agent)
            logging.info(f"✅ Agent 加载: {agent.name} | Model: {agent.model}")

        if not self.agents:
            raise ValueError("❌ 至少需要配置一个 Agent")

        self.leader = self.agents[0]
        logging.info(f"👑 Leader: {self.leader.name}")

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

    def solve(
            self,
            task: str,
            use_memory: bool = False,
            memory_key: str = "default",
            image_paths: Optional[List[str]] = None
    ) -> str:
        """
        解决任务的主入口

        Args:
            task: 任务描述
            use_memory: 是否使用持久化记忆
            memory_key: 记忆键名
            image_paths: 图像文件路径列表

        Returns:
            最终答案
        """
        tracker = TimeTracker()
        tracker.start()

        logging.info(f"\n{'=' * 80}")
        logging.info(f"📋 新任务: {task}")
        logging.info(f"   记忆模式: {use_memory} | Key: {memory_key}")
        logging.info(f"   图片数量: {len(image_paths) if image_paths else 0}")
        logging.info(f"{'=' * 80}")

        print(f"\n{'=' * 80}")
        print(f"🚀 任务开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 80}\n")

        if image_paths:
            image_paths = image_paths[:self.max_images]
            logging.info(f"📷 处理 {len(image_paths)} 张图片")

        history: List[Dict] = []

        # 图像处理
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
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                    logging.info(f"  ✅ 图片 {idx}: {path} ({mime_type})")

                except Exception as e:
                    logging.error(f"  ❌ 读取图片失败 {path}: {e}")
                    image_content.append({
                        "type": "text",
                        "text": f"[无法读取图片 {idx}: {os.path.basename(path)}]"
                    })

            history.append({"speaker": "User", "content": image_content})
        else:
            history.append({"speaker": "User", "content": task})

        tracker.checkpoint("1️⃣ 初始化")

        # 加载历史记忆
        if use_memory and memory_key in self.memory:
            memory_text = "\n".join([
                f"- {item['summary']}"
                for item in self.memory[memory_key][-5:]
            ])
            history.insert(0, {
                "speaker": "System",
                "content": f"📚 历史记忆（{memory_key}）：\n{memory_text}"
            })
            logging.info(f"📖 加载历史记忆: {memory_key} ({len(self.memory[memory_key])} 条)")

        # 主循环
        round_num = 0
        while True:
            round_num += 1

            if round_num > self.max_rounds:
                logging.warning(f"⏱️ 达到最大轮次 {self.max_rounds}，强制结束")
                break

            logging.info(f"\n{'─' * 80}")
            logging.info(f"🔄 第 {round_num} 轮讨论开始")
            logging.info(f"{'─' * 80}")

            round_start = time.time()

            # ✅ 使用 max_concurrent_agents 限制并发数
            with ThreadPoolExecutor(max_workers=self.max_concurrent_agents) as executor:
                future_to_agent = {
                    executor.submit(
                        agent.generate_response,
                        history.copy(),
                        round_num
                    ): agent
                    for agent in self.agents
                }

                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        contribution = future.result()
                        history.append({
                            "speaker": agent.name,
                            "content": contribution
                        })
                        logging.info(f"✅ {agent.name} 完成第 {round_num} 轮")
                    except Exception as e:
                        logging.error(f"❌ {agent.name} 执行失败: {e}")
                        history.append({
                            "speaker": agent.name,
                            "content": f"[执行失败: {str(e)}]"
                        })

            round_elapsed = time.time() - round_start
            round_time_str = tracker.format_time(round_elapsed)
            print(f"\n⏱️  第 {round_num} 轮讨论完成 | 耗时: {round_time_str}\n")
            logging.info(f"⏱️  第 {round_num} 轮讨论耗时: {round_time_str}")

            tracker.checkpoint(f"2️⃣ 第{round_num}轮讨论")

            # 反思与规划
            if self.mode == "intelligent" and self.reflection_planning:
                reflection_start = time.time()

                logging.info(f"\n{'─' * 80}")
                logging.info(f"🤔 Leader Multi-Round Reflection (第 {round_num} 轮)")
                logging.info(f"{'─' * 80}")

                plan_prompt = (
                    "请以 JSON 格式规划下一轮的重点方向。\n"
                    "格式: {\"focus_areas\": [\"方向1\", \"方向2\"], \"expected_improvement\": \"预期改进\"}"
                )
                plan = self.leader.generate_response(
                    history + [{"speaker": "System", "content": plan_prompt}],
                    round_num,
                    force_non_stream=True
                )
                logging.info(f"📋 Plan: {plan[:200]}...")

                max_reflection_rounds = self.max_reflection_rounds
                final_decision = "continue"
                final_quality = 0
                previous_quality = 0

                for reflection_round in range(1, max_reflection_rounds + 1):
                    logging.info(f"\n🔍 Reflection Round {reflection_round}/{max_reflection_rounds}")

                    if reflection_round == 1:
                        reflect_prompt = (
                            "请反思本轮讨论结果，给出质量评分和决策。\n"
                            "评估标准：\n"
                            "- 信息完整性（是否覆盖关键点）\n"
                            "- 逻辑严密性（是否有矛盾或跳跃）\n"
                            "- 深度与洞察（是否有独到见解）\n"
                            "JSON 格式: {\"quality_score\": 1-10, \"decision\": \"continue/stop\", "
                            "\"reason\": \"原因\", \"suggestions\": [\"建议1\", \"建议2\"]}"
                        )
                    elif reflection_round == 2:
                        reflect_prompt = (
                            f"这是第 {reflection_round} 次深度反思。\n"
                            f"上次评分：{final_quality}/10\n"
                            f"上次建议：已在讨论中部分体现\n\n"
                            "请更深入地分析：\n"
                            "- 是否还有隐藏的逻辑漏洞？\n"
                            "- 论据是否充分支撑结论？\n"
                            "- 表达是否清晰易懂？\n"
                            "JSON 格式: {\"quality_score\": 1-10, \"decision\": \"continue/stop\", "
                            "\"reason\": \"原因\", \"critical_issues\": [\"关键问题1\", \"关键问题2\"]}"
                        )
                    else:
                        reflect_prompt = (
                            f"这是第 {reflection_round} 次（最终）反思。\n"
                            f"上次评分：{final_quality}/10\n"
                            f"质量提升幅度：{final_quality - previous_quality if previous_quality > 0 else 'N/A'}\n\n"
                            "请做最终综合判断：\n"
                            "- 当前质量是否达到可交付标准？\n"
                            "- 继续讨论的边际收益如何？\n"
                            "- 是否存在致命缺陷必须修复？\n"
                            "JSON 格式: {\"quality_score\": 1-10, \"decision\": \"continue/stop\", "
                            "\"reason\": \"原因\", \"final_verdict\": \"综合评价\"}"
                        )

                    leader_eval = self.leader.generate_response(
                        history + [{"speaker": "System", "content": reflect_prompt}],
                        round_num,
                        force_non_stream=True
                    )

                    logging.info(f"💭 Reflection {reflection_round}: {leader_eval[:150]}...")

                    try:
                        eval_json = json.loads(
                            leader_eval.strip()
                            .replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )

                        previous_quality = final_quality
                        final_quality = eval_json.get("quality_score", 0)
                        final_decision = eval_json.get("decision", "").lower()

                        logging.info(f"📊 质量评分: {final_quality}/10 | 决策: {final_decision}")

                        if final_quality >= self.reflection_quality_threshold:
                            logging.info(f"✅ 质量达到 {final_quality}/10，无需继续反思")
                            break

                        if final_decision == "stop" and final_quality >= self.stop_quality_threshold:
                            logging.info(f"✅ Leader 判断质量 {final_quality}/10 可接受，停止反思")
                            break

                        if reflection_round > 1 and previous_quality > 0:
                            quality_delta = final_quality - previous_quality
                            if abs(quality_delta) < self.quality_convergence_delta:
                                logging.info(f"🔴 质量提升停滞 (Δ={quality_delta:.1f})，停止反思")
                                break

                    except json.JSONDecodeError:
                        logging.warning(f"⚠️ 反思 {reflection_round} JSON 解析失败")
                        final_quality = max(final_quality, 5)
                        continue
                    except Exception as e:
                        logging.error(f"❌ 反思 {reflection_round} 处理失败: {e}")
                        continue

                reflection_elapsed = time.time() - reflection_start
                reflection_time_str = tracker.format_time(reflection_elapsed)
                print(f"⏱️  反思阶段完成 | 耗时: {reflection_time_str}\n")
                logging.info(f"⏱️  反思阶段耗时: {reflection_time_str}")

                tracker.checkpoint(f"3️⃣ 第{round_num}轮反思")

                if final_decision == "stop" and final_quality >= self.stop_quality_threshold:
                    logging.info(f"🎯 经过 {reflection_round} 轮反思，质量达到 {final_quality}/10，停止讨论")
                    break
                else:
                    logging.info(f"🔄 质量 {final_quality}/10，继续下一轮讨论优化")

        # 最终综合
        final_synthesis_start = time.time()

        logging.info(f"\n{'=' * 80}")
        logging.info("🎯 Leader 最终综合")
        logging.info(f"{'=' * 80}")

        history.append({
            "speaker": "System",
            "content": (
                "请综合以上全部讨论，给出最准确、最完整、最高质量的最终答案。\n"
                "要求：\n"
                "1. 逻辑严密，论证充分\n"
                "2. 信息完整，细节丰富\n"
                "3. 结构清晰，易于理解\n"
                "4. 如涉及代码或文件操作，请确保已正确执行"
            )
        })

        final_answer = self.leader.generate_response(
            history,
            round_num + 1,
            force_non_stream=False
        )

        tracker.checkpoint("4️⃣ 最终综合")

        # 保存记忆
        if use_memory:
            summary_prompt = (
                "请用 500 字以内总结本次任务的：\n"
                "1. 核心结论\n"
                "2. 关键发现\n"
                "3. 可复用经验\n"
                "4. 遗留问题（如有）"
            )
            summary = self.leader.generate_response(
                history + [{"speaker": "System", "content": summary_prompt}],
                round_num + 1,
                force_non_stream=True
            )
            self._save_memory(memory_key, summary)

            # 向量记忆
            if self.vector_memory:
                self.vector_memory.add(
                    summary,
                    metadata={"task": task[:100], "memory_key": memory_key}
                )

            tracker.checkpoint("5️⃣ 保存记忆")

        # 输出最终答案
        print("\n" + "=" * 100)
        print("🎯 【最终最高质量答案】")
        print("=" * 100)
        print(final_answer)
        print("=" * 100)

        print(tracker.summary())
        logging.info(tracker.summary())

        logging.info(f"\n{'=' * 80}")
        logging.info("✅ 任务完成")
        logging.info(f"{'=' * 80}\n")

        return final_answer


# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        swarm = MultiAgentSwarm()

        # 示例1：基础任务
        # swarm.solve("请帮我分析一下人工智能的发展趋势")

        # 示例2：带记忆的深度报告
        swarm.solve(
            "请帮我写一篇关于人工智能的深度分析报告，并保存到 ./reports/ai_report.md",
            use_memory=True,
            memory_key="ai_topic"
        )

        # 示例3：图像分析（需要提供真实图片路径）
        # swarm.solve(
        #     "请分析这些图片中的代码问题",
        #     image_paths=["./screenshot1.png", "./screenshot2.png"]
        # )

    except Exception as e:
        logging.error(f"❌ 程序异常: {e}", exc_info=True)