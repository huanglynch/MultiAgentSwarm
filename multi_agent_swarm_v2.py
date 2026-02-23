import yaml
import logging
import os
import glob
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
from openai import OpenAI
import json

# ====================== Skill 动态加载器（保持不变） ======================
def load_skills(skills_dir: str = "skill"):
    tool_registry: Dict[str, Dict] = {}
    shared_knowledge = ""

    if not os.path.exists(skills_dir):
        logging.warning(f"⚠️ skill/ 目录不存在，将使用空工具集")
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


# ====================== Agent 类（保持不变） ======================
class Agent:
    def __init__(self, config: Dict, default_model: str, default_max_tokens: int, tool_registry: Dict, shared_knowledge: str = ""):
        self.name = config["name"]
        self.role = config["role"]
        self.shared_knowledge = shared_knowledge
        self.client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url"))
        self.model = config.get("model", default_model)
        self.temperature = config.get("temperature", 0.7)
        self.stream = config.get("stream", False)
        self.max_tokens = config.get("max_tokens", default_max_tokens)
        enabled = config.get("enabled_tools", [])
        self.tools = [tool_registry[name]["schema"] for name in enabled if name in tool_registry]
        self.tool_map: Dict[str, Callable] = {name: tool_registry[name]["func"] for name in enabled if name in tool_registry}

    def _execute_tool(self, tool_call) -> Dict:
        # （保持不变，省略以节省篇幅，实际代码与 v2.2 完全相同）
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
            if self.stream and self.name == "Grok":  # 只在 Leader 流式显示最终综合更清晰
                print(f"\n【{self.name}】正在思考... ", end="", flush=True)

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                max_tokens=self.max_tokens, tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None, stream=self.stream,
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

            # Tool Calling 单轮（保持不变）
            message_obj = response.choices[0].message
            if not self.stream and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                messages.append(message_obj.model_dump())
                for tool_call in message_obj.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append(tool_result)
                final_resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
                full_response = final_resp.choices[0].message.content or ""

            return full_response.strip()

        except Exception as e:
            err = f"[Error in {self.name}]: {str(e)}"
            logging.error(err)
            return err


# ====================== 主类 v2.3 ======================
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
        self.mode = swarm.get("mode", "fixed")                    # 新增：fixed / intelligent
        self.max_rounds = swarm.get("max_rounds", 3 if swarm.get("mode", "fixed") == "fixed" else 10)
        self.log_file = swarm.get("log_file", "swarm.log")
        self.skills_dir = swarm.get("skills_dir", "skill")

        # 日志设置（不变）
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", encoding="utf-8", force=True)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(f"=== MultiAgentSwarm v2.3 初始化完成 | Mode: {self.mode} | Max Rounds: {self.max_rounds} ===")

        self.tool_registry, self.shared_knowledge = load_skills(self.skills_dir)

        self.agents = []
        for a_cfg in cfg.get("agents", [])[:swarm.get("num_agents", 4)]:
            agent = Agent(a_cfg, self.default_model, self.default_max_tokens, self.tool_registry, self.shared_knowledge)
            self.agents.append(agent)
            logging.info(f"✅ Agent 加载: {agent.name} | Model: {agent.model}")

        self.leader = self.agents[0]

    def solve(self, task: str) -> str:
        logging.info(f"【新任务启动】{task} | 模式: {self.mode}")
        history: List[Dict] = [{"speaker": "User", "content": task}]
        round_num = 0

        while True:
            round_num += 1
            if round_num > self.max_rounds:
                logging.warning(f"达到最大轮次上限 {self.max_rounds}，强制结束")
                break

            logging.info(f"--- 第 {round_num} 轮并行讨论开始 ---")
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_to_agent = {executor.submit(agent.generate_response, history.copy(), round_num): agent for agent in self.agents}
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    contribution = future.result()
                    history.append({"speaker": agent.name, "content": contribution})

            # ==================== 智能模式核心：Leader 智能评价 ====================
            if self.mode == "intelligent":
                logging.info(f"--- Leader 智能评价第 {round_num} 轮质量 ---")
                eval_prompt = (
                    "你现在是团队质量控制官。请严格评估当前所有讨论的质量。\n"
                    "输出必须是合法 JSON 对象，格式如下：\n"
                    "{\n"
                    '  "quality_score": 整数(1-10),\n'
                    '  "decision": "continue" 或 "stop",\n'
                    '  "reason": "简短理由",\n'
                    '  "suggestions": "如果继续，下一步改进点"\n'
                    "}\n"
                    "质量 >=8 且你认为已足够完美时 decision=stop，并直接给出最终高质量答案。"
                )
                eval_history = history + [{"speaker": "System", "content": eval_prompt}]
                leader_eval = self.leader.generate_response(eval_history, round_num, "请以 JSON 格式回复。")

                try:
                    eval_json = json.loads(leader_eval.strip().strip("```json").strip("```"))
                    score = eval_json.get("quality_score", 5)
                    decision = eval_json.get("decision", "continue")
                    logging.info(f"Leader 评价: 分数={score} | 决策={decision} | 理由={eval_json.get('reason')}")
                    if decision.lower() == "stop":
                        logging.info("✅ Leader 判断已达最高质量，停止讨论")
                        # 最终答案已在 eval_json 或直接用 leader_eval 后的内容
                        final_answer = eval_json.get("final_answer") or leader_eval
                        break
                except:
                    logging.warning("JSON 解析失败，继续下一轮")
                    continue

            # 固定模式或未停止时继续
            if self.mode == "fixed" and round_num >= self.max_rounds:
                break

        # Leader 最终综合（智能模式下可能已在 eval 中完成）
        if self.mode == "fixed" or "final_answer" not in locals():
            logging.info("--- Leader 最终综合 ---")
            history.append({"speaker": "System", "content": "请综合以上全部讨论，给出最准确、最完整、最高质量的最终答案。"})
            final_answer = self.leader.generate_response(history, round_num + 1)

        print("\n" + "="*100)
        print("🎯 【最终最高质量答案】")
        print(final_answer)
        print("="*100)
        logging.info("✅ 任务完成")
        return final_answer


if __name__ == "__main__":
    swarm = MultiAgentSwarm()
    swarm.solve("请帮我写一篇关于『2026 年东京人工智能产业趋势』的深度报告，并保存到 ./reports/tokyo_ai_2026.md")
