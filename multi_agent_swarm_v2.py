import yaml
import logging
import os
import glob
import importlib.util
import requests
import random
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
from openai import OpenAI
import json
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS

# ====================== 工具函数（已加入延时） ======================
def web_search(query: str, num_results: int = 5) -> str:
    """实时网页搜索（带随机延时）"""
    time.sleep(random.uniform(0.5, 2.0))   # ← 新增随机延时，防止请求过快
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
        return "\n".join([f"标题: {r['title']}\n摘要: {r['body']}\n链接: {r['href']}" for r in results])
    except Exception as e:
        return f"搜索失败: {str(e)}"

def browse_page(url: str) -> str:
    """浏览网页"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)
    except Exception as e:
        return f"浏览失败: {str(e)}"

def run_python(code: str) -> str:
    """安全沙箱 Python 执行"""
    try:
        restricted_globals = {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int}}
        local_vars = {}
        exec(code, restricted_globals, local_vars)
        return str(local_vars.get("result", "执行成功，无返回结果"))
    except Exception as e:
        return f"执行错误: {str(e)}"

# ====================== 向量记忆 ======================
class VectorMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./memory_db")
        self.collection = self.client.get_or_create_collection(
            name="swarm_memory",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def add(self, text: str, metadata: Dict = None):
        if not metadata:
            metadata = {"timestamp": datetime.now().isoformat()}
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[datetime.now().isoformat()]
        )

    def search(self, query: str, n_results: int = 5) -> str:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if results and results["documents"]:
            return "\n\n".join(results["documents"][0])
        return ""

# ====================== Skill 动态加载器 ======================
def load_skills(skills_dir: str = "skill"):
    tool_registry: Dict[str, Dict] = {}
    shared_knowledge = ""

    if not os.path.exists(skills_dir):
        logging.warning(f"⚠️ skill/ 目录不存在")
        return tool_registry, shared_knowledge

    for py_file in glob.glob(os.path.join(skills_dir, "*.py")):
        if "__init__" in py_file: continue
        module_name = os.path.splitext(os.path.basename(py_file))[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "execute") and hasattr(module, "schema"):
                name = getattr(module, "name", module_name)
                tool_registry[name] = {"func": module.execute, "schema": module.schema}
                logging.info(f"✅ 加载 Skill (py): {name}")
        except Exception as e:
            logging.error(f"加载 Skill {py_file} 失败: {e}")

    for md_file in glob.glob(os.path.join(skills_dir, "*.md")):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                shared_knowledge += f"\n\n### 来自 {os.path.basename(md_file)} ###\n{content}"
            logging.info(f"✅ 加载知识 (md): {os.path.basename(md_file)}")
        except Exception as e:
            logging.error(f"加载知识 {md_file} 失败: {e}")

    return tool_registry, shared_knowledge

# ====================== Agent 类 ======================
class Agent:
    def __init__(self, config: Dict, default_model: str, default_max_tokens: int, tool_registry: Dict, shared_knowledge: str = "", vector_memory=None):
        self.name = config["name"]
        self.role = config["role"]
        self.shared_knowledge = shared_knowledge
        self.vector_memory = vector_memory

        self.client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url"))
        self.model = config.get("model", default_model)
        self.temperature = config.get("temperature", 0.7)
        self.stream = config.get("stream", False)
        self.max_tokens = config.get("max_tokens", default_max_tokens)

        enabled = config.get("enabled_tools", [])
        self.tools = [tool_registry[name]["schema"] for name in enabled if name in tool_registry]
        self.tool_map = {name: tool_registry[name]["func"] for name in enabled if name in tool_registry}

    def _execute_tool(self, tool_call) -> Dict:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            result = self.tool_map[func_name](**args)
            return {"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": str(result)}
        except Exception as e:
            return {"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": f"Tool error: {str(e)}"}

    def generate_response(self, history: List[Dict], round_num: int, system_extra: str = "", force_non_stream: bool = False) -> str:
        use_stream = self.stream and not force_non_stream and not self.tools

        system_prompt = f"{self.role}\n{self.shared_knowledge}\n{system_extra}\n你是多智能体协作团队的一员，请提供有价值、准确、有深度的贡献。"
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            if h["speaker"] == "User":
                messages.append({"role": "user", "content": h["content"]})
            else:
                messages.append({"role": "user", "content": f"[{h['speaker']}] {h.get('content', '')}"})

        try:
            if use_stream:
                print(f"\n【{self.name}】正在思考... ", end="", flush=True)

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
            if use_stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        print(delta, end="", flush=True)
                        full_response += delta
                print()
            else:
                full_response = response.choices[0].message.content or ""

            if not use_stream and hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                messages.append(response.choices[0].message.model_dump())
                for tool_call in response.choices[0].message.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append(tool_result)
                final_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                full_response = final_resp.choices[0].message.content or ""

            return full_response.strip()

        except Exception as e:
            err = f"[Error in {self.name}]: {str(e)}"
            logging.error(err)
            return err


# ====================== 主类 v2.5.1 ======================
class MultiAgentSwarm:
    def __init__(self, config_path: str = "swarm_config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        oai = cfg.get("openai", {})
        self.default_model = oai.get("default_model", "gpt-4o-mini")
        self.default_max_tokens = oai.get("default_max_tokens", 4096)

        swarm = cfg.get("swarm", {})
        self.mode = swarm.get("mode", "fixed")
        self.max_rounds = swarm.get("max_rounds", 3 if swarm.get("mode", "fixed") == "fixed" else 10)
        self.reflection_planning = swarm.get("reflection_planning", True)
        self.enable_web_search = swarm.get("enable_web_search", False)   # ← 新增，默认关闭

        self.log_file = swarm.get("log_file", "swarm.log")
        self.skills_dir = swarm.get("skills_dir", "skill")
        self.memory_file = swarm.get("memory_file", "memory.json")
        self.max_memory_items = swarm.get("max_memory_items", 50)

        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s",
                            encoding="utf-8", force=True)
        logging.getLogger().addHandler(logging.StreamHandler())

        logging.info(f"=== MultiAgentSwarm v2.5.1 初始化完成 | Mode: {self.mode} | WebSearch: {self.enable_web_search} ===")

        self.tool_registry, self.shared_knowledge = load_skills(self.skills_dir)

        # 动态注册 web_search
        if self.enable_web_search:
            self.tool_registry["web_search"] = {
                "func": web_search,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "实时网页搜索最新信息",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
                            "required": ["query"]
                        }
                    }
                }
            }

        self.vector_memory = VectorMemory()

        self.agents = []
        for a_cfg in cfg.get("agents", [])[:swarm.get("num_agents", 4)]:
            agent = Agent(a_cfg, self.default_model, self.default_max_tokens, self.tool_registry, self.shared_knowledge, self.vector_memory)
            self.agents.append(agent)
            logging.info(f"✅ Agent 加载: {agent.name}")

        self.leader = self.agents[0]

    # _load_memory、_save_memory、solve 方法与 v2.5 完全相同（省略以节省篇幅，请保留你上一个版本中的这三个方法）

if __name__ == "__main__":
    swarm = MultiAgentSwarm()
    swarm.solve("请帮我写一篇关于人工智能的深度分析报告，并保存到 ./reports/ai_report.md", use_memory=True, memory_key="ai_topic")
