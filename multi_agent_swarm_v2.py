import yaml
import logging
import os
import glob
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
from openai import OpenAI
import json

# ====================== Skill 动态加载器 ======================
def load_skills(skills_dir: str = "skill"):
    tool_registry: Dict[str, Dict] = {}
    shared_knowledge = ""

    if not os.path.exists(skills_dir):
        logging.warning(f"⚠️ skill/ 目录不存在（{skills_dir}），将使用空工具集")
        return tool_registry, shared_knowledge

    # 1. 加载 .py 文件 → 可执行 Tool
    for py_file in glob.glob(os.path.join(skills_dir, "*.py")):
        if "__init__" in py_file:
            continue
        module_name = os.path.splitext(os.path.basename(py_file))[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "execute") and hasattr(module, "schema"):
                name = getattr(module, "name", module_name)
                tool_registry[name] = {
                    "func": module.execute,
                    "schema": module.schema
                }
                logging.info(f"✅ 加载 Skill (py): {name}")
        except Exception as e:
            logging.error(f"加载 Skill {py_file} 失败: {e}")

    # 2. 加载 .md 文件 → 共享知识
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
    def __init__(self, config: Dict, default_model: str, default_max_tokens: int, tool_registry: Dict, shared_knowledge: str = ""):
        self.name = config["name"]
        self.role = config["role"]
        self.shared_knowledge = shared_knowledge

        self.client = OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        self.model = config.get("model", default_model)
        self.temperature = config.get("temperature", 0.7)
        self.stream = config.get("stream", False)
        self.max_tokens = config.get("max_tokens", default_max_tokens)

        # 工具（只启用 yaml 中声明的）
        enabled = config.get("enabled_tools", [])
        self.tools = [tool_registry[name]["schema"] for name in enabled if name in tool_registry]
        self.tool_map: Dict[str, Callable] = {name: tool_registry[name]["func"] for name in enabled if name in tool_registry}

    def _execute_tool(self, tool_call) -> Dict:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            result = self.tool_map[func_name](**args)
            return {"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": str(result)}
        except Exception as e:
            return {"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": f"Tool error: {str(e)}"}

    def generate_response(self, history: List[Dict], round_num: int) -> str:
        system_prompt = f"{self.role}\n{self.shared_knowledge}\n你是多智能体协作团队的一员，请提供有价值、准确、有深度的贡献。"
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            if h["speaker"] == "User":
                messages.append({"role": "user", "content": h["content"]})
            else:
                messages.append({"role": "user", "content": f"[{h['speaker']}] {h.get('content', '')}"})

        try:
            if self.stream:
                print(f"\n【{self.name}】正在思考... ", end="", flush=True)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
                stream=self.stream,
            )

            full_response = ""
            if self.stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        print(delta, end="", flush=True)
                        full_response += delta
                print()
            else:
                full_response = response.choices[0].message.content or ""

            # Tool Calling（单轮）
            message_obj = response.choices[0].message
            if not self.stream and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                messages.append(message_obj.model_dump())
                for tool_call in message_obj.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append(tool_result)
                    logging.info(f"[{self.name}] 执行工具: {tool_call.function.name}")

                final_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                full_response = final_resp.choices[0].message.content or ""

            logging.info(f"[Round {round_num}] {self.name} 完成")
            return full_response.strip()

        except Exception as e:
            err = f"[Error in {self.name}]: {str(e)}"
            logging.error(err)
            return err


# ====================== 主类 ======================
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
        self.num_agents = swarm.get("num_agents", 4)
        self.max_rounds = swarm.get("max_rounds", 3)
        self.log_file = swarm.get("log_file", "swarm.log")
        self.skills_dir = swarm.get("skills_dir", "skill")   # ← 新增

        # 日志
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s",
                            encoding="utf-8", force=True)
        logging.getLogger().addHandler(logging.StreamHandler())

        logging.info("=== MultiAgentSwarm v2.2 (Skill 独立目录) 初始化 ===")

        # 动态加载 Skill
        self.tool_registry, self.shared_knowledge = load_skills(self.skills_dir)
        logging.info(f"共加载 {len(self.tool_registry)} 个可执行 Skill，知识库长度 {len(self.shared_knowledge)} 字符")

        # 加载 Agent
        self.agents: List[Agent] = []
        for a_cfg in cfg.get("agents", [])[:self.num_agents]:
            agent = Agent(a_cfg, self.default_model, self.default_max_tokens,
                          self.tool_registry, self.shared_knowledge)
            self.agents.append(agent)
            logging.info(f"✅ Agent 加载: {agent.name} | Model: {agent.model} | max_tokens: {agent.max_tokens}")

        self.leader = self.agents[0]

    def solve(self, task: str) -> str:
        logging.info(f"【新任务】{task}")
        history: List[Dict] = [{"speaker": "User", "content": task}]

        for r in range(1, self.max_rounds + 1):
            logging.info(f"--- 第 {r} 轮并行讨论开始 ---")
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_to_agent = {executor.submit(agent.generate_response, history.copy(), r): agent for agent in self.agents}
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        contribution = future.result()
                        history.append({"speaker": agent.name, "content": contribution})
                    except Exception as e:
                        logging.error(f"{agent.name} 异常: {e}")

        logging.info("--- Leader 最终综合 ---")
        history.append({"speaker": "System", "content": "请综合以上全部讨论，给出最准确、最完整、最优的最终答案。"})
        final_answer = self.leader.generate_response(history, self.max_rounds + 1)

        print("\n" + "="*90)
        print("🎯 【最终答案】")
        print(final_answer)
        print("="*90)
        return final_answer


if __name__ == "__main__":
    swarm = MultiAgentSwarm()
    swarm.solve("请读取 skill/knowledge.md 中的内容，然后帮我写一篇关于人工智能的短文，并保存到 ./output/ai_essay.md")
