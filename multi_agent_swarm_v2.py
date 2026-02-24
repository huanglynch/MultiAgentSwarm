import yaml
import logging
import os
import glob
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
from openai import OpenAI
import json
from datetime import datetime

# ====================== Skill 动态加载器 ======================
def load_skills(skills_dir: str = "skills"):
    tool_registry: Dict[str, Dict] = {}
    shared_knowledge = ""

    if not os.path.exists(skills_dir):
        logging.warning(f"⚠️ skills/ 目录不存在，将使用空工具集")
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

    def generate_response(self, history: List[Dict], round_num: int, system_extra: str = "") -> str:
        system_prompt = f"{self.role}\n{self.shared_knowledge}\n{system_extra}\n你是多智能体协作团队的一员，请提供有价值、准确、有深度的贡献。"
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

            # Tool Calling 单轮
            message_obj = response.choices[0].message
            if not self.stream and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                messages.append(message_obj.model_dump())
                for tool_call in message_obj.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append(tool_result)
                final_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                full_response = final_resp.choices[0].message.content or ""

            return full_response.strip()

        except Exception as e:
            err = f"[Error in {self.name}]: {str(e)}"
            logging.error(err)
            return err


# ====================== 主类 v2.4 ======================
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
        self.log_file = swarm.get("log_file", "swarm.log")
        self.skills_dir = swarm.get("skills_dir", "skills")
        self.memory_file = swarm.get("memory_file", "memory.json")      # 新增
        self.max_memory_items = swarm.get("max_memory_items", 50)       # 新增

        # 日志
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s",
                            encoding="utf-8", force=True)
        logging.getLogger().addHandler(logging.StreamHandler())

        logging.info(f"=== MultiAgentSwarm v2.4 初始化完成 | Mode: {self.mode} ===")

        # 加载 skills
        self.tool_registry, self.shared_knowledge = load_skills(self.skills_dir)

        # 加载记忆
        self.memory = self._load_memory()

        # 加载 Agent
        self.agents = []
        for a_cfg in cfg.get("agents", [])[:swarm.get("num_agents", 4)]:
            agent = Agent(a_cfg, self.default_model, self.default_max_tokens, self.tool_registry, self.shared_knowledge)
            self.agents.append(agent)
            logging.info(f"✅ Agent 加载: {agent.name}")

        self.leader = self.agents[0]

    def _load_memory(self) -> Dict:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_memory(self, key: str, summary: str):
        if key not in self.memory:
            self.memory[key] = []
        self.memory[key].append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary[:3000]
        })
        if len(self.memory[key]) > self.max_memory_items:
            self.memory[key] = self.memory[key][-self.max_memory_items:]
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def solve(self, task: str, use_memory: bool = False, memory_key: str = "default") -> str:
        logging.info(f"【新任务】{task} | 记忆模式: {use_memory} | key: {memory_key}")
        history: List[Dict] = [{"speaker": "User", "content": task}]

        # 加载记忆
        if use_memory and memory_key in self.memory:
            memory_text = "\n".join([item["summary"] for item in self.memory[memory_key][-5:]])
            history.insert(0, {"speaker": "System", "content": f"历史记忆（{memory_key}）：\n{memory_text}"})

        round_num = 0
        while True:
            round_num += 1
            if round_num > self.max_rounds:
                logging.warning(f"达到最大轮次 {self.max_rounds}，强制结束")
                break

            logging.info(f"--- 第 {round_num} 轮并行讨论开始 ---")
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_to_agent = {executor.submit(agent.generate_response, history.copy(), round_num): agent for agent in self.agents}
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    contribution = future.result()
                    history.append({"speaker": agent.name, "content": contribution})

            # 智能模式评价
            if self.mode == "intelligent":
                logging.info(f"--- Leader 智能评价第 {round_num} 轮 ---")
                eval_prompt = "请评估当前讨论质量，输出合法 JSON：{\"quality_score\": 整数(1-10), \"decision\": \"continue\"或\"stop\", \"reason\": \"简短理由\"}"
                eval_history = history + [{"speaker": "System", "content": eval_prompt}]
                leader_eval = self.leader.generate_response(eval_history, round_num)

                try:
                    eval_json = json.loads(leader_eval.strip().strip("```json").strip("```"))
                    if eval_json.get("decision", "").lower() == "stop":
                        logging.info("✅ Leader 判断已达最高质量，停止讨论")
                        break
                except:
                    pass

        # Leader 最终综合
        logging.info("--- Leader 最终综合 ---")
        history.append({"speaker": "System", "content": "请综合以上全部讨论，给出最准确、最完整、最高质量的最终答案。"})
        final_answer = self.leader.generate_response(history, round_num + 1)

        # 保存记忆
        if use_memory:
            summary_prompt = "请用 500 字以内总结本次任务的核心结论、关键发现和可复用经验。"
            summary = self.leader.generate_response(history + [{"speaker": "System", "content": summary_prompt}], round_num + 1)
            self._save_memory(memory_key, summary)

        print("\n" + "="*100)
        print("🎯 【最终最高质量答案】")
        print(final_answer)
        print("="*100)
        return final_answer


if __name__ == "__main__":
    swarm = MultiAgentSwarm()
    swarm.solve("请帮我写一篇关于人工智能的深度分析报告，并保存到 ./reports/ai_report.md", use_memory=True, memory_key="ai_topic")
