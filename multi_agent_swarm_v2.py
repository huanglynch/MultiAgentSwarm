import yaml
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
from openai import OpenAI
import json

# ====================== 默认内置 Skill ======================
def read_file(path: str) -> str:
    """读取本地文件内容"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {str(e)}"

def write_file(path: str, content: str) -> str:
    """写入内容到本地文件（自动创建目录）"""
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件已成功写入: {path}"
    except Exception as e:
        return f"写入失败: {str(e)}"

def list_dir(path: str = ".") -> str:
    """列出目录下的文件和文件夹"""
    try:
        items = os.listdir(path)
        return "\n".join([f"📄 {item}" if os.path.isfile(os.path.join(path, item)) else f"📁 {item}/" for item in items])
    except Exception as e:
        return f"列目录失败: {str(e)}"

DEFAULT_TOOLS = {
    "read_file": {
        "func": read_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取指定路径的本地文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "文件路径"}},
                    "required": ["path"]
                }
            }
        }
    },
    "write_file": {
        "func": write_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "将内容写入指定路径的文件（自动创建目录）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"},
                        "content": {"type": "string", "description": "要写入的内容"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    },
    "list_dir": {
        "func": list_dir,
        "schema": {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "列出指定目录下的文件和文件夹",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "目录路径，默认为当前目录"}},
                    "required": []
                }
            }
        }
    }
}

# ====================== Agent 类 ======================
class Agent:
    def __init__(self, config: Dict, default_model: str, default_max_tokens: int):
        self.name = config["name"]
        self.role = config["role"]
        
        self.client = OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        self.model = config.get("model", default_model)
        self.temperature = config.get("temperature", 0.7)
        self.stream = config.get("stream", False)
        self.max_tokens = config.get("max_tokens", default_max_tokens)
        
        enabled = config.get("enabled_tools", [])
        self.tools = [DEFAULT_TOOLS[name]["schema"] for name in enabled if name in DEFAULT_TOOLS]
        self.tool_map: Dict[str, Callable] = {name: DEFAULT_TOOLS[name]["func"] for name in enabled if name in DEFAULT_TOOLS}

    def _execute_tool(self, tool_call) -> Dict:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            func = self.tool_map.get(func_name)
            if func:
                result = func(**args)
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(result)
                }
        except Exception as e:
            return {"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": f"Tool error: {str(e)}"}
        return {"role": "tool", "content": "Tool not found"}

    def generate_response(self, history: List[Dict], round_num: int) -> str:
        messages = [{"role": "system", "content": f"{self.role}\n你是多智能体协作团队的一员，请提供有价值、准确、有深度的贡献。"}]
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

            # Tool Calling（单轮，最可靠）
            message_obj = response.choices[0].message
            if not self.stream and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                messages.append(message_obj.model_dump())
                for tool_call in message_obj.tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append(tool_result)
                    logging.info(f"[{self.name}] 执行工具: {tool_call.function.name}")
                
                # Tool 执行后再次调用得到最终回答
                final_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                full_response = final_resp.choices[0].message.content or ""

            logging.info(f"[Round {round_num}] {self.name} 完成贡献")
            return full_response.strip()

        except Exception as e:
            err_msg = f"[Error in {self.name}]: {str(e)}"
            logging.error(err_msg)
            return err_msg

# ====================== 主类 ======================
class MultiAgentSwarm:
    def __init__(self, config_path: str = "swarm_config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}（请根据下方示例创建）")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        oai = cfg.get("openai", {})
        self.default_model = oai.get("default_model", "gpt-4o-mini")
        self.default_max_tokens = oai.get("default_max_tokens", 4096)

        swarm = cfg.get("swarm", {})
        self.num_agents = swarm.get("num_agents", 4)
        self.max_rounds = swarm.get("max_rounds", 3)
        self.log_file = swarm.get("log_file", "swarm.log")

        # 日志
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            encoding="utf-8",
            force=True
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

        logging.info("=== MultiAgentSwarm v2.1 初始化完成 ===")

        self.agents: List[Agent] = []
        for a_cfg in cfg.get("agents", [])[:self.num_agents]:
            agent = Agent(a_cfg, self.default_model, self.default_max_tokens)
            self.agents.append(agent)
            logging.info(f"✅ Agent 加载: {agent.name} | Model: {agent.model} | max_tokens: {agent.max_tokens} | Stream: {agent.stream}")

        if not self.agents:
            raise ValueError("至少需要配置 1 个 Agent")
        self.leader = self.agents[0]

    def solve(self, task: str) -> str:
        logging.info(f"【新任务启动】{task}")
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
                        logging.error(f"{agent.name} 执行异常: {e}")

        # Leader 最终综合
        logging.info("--- Leader 最终综合 ---")
        history.append({"speaker": "System", "content": "请综合以上所有智能体的讨论，给出最准确、最完整、最优的最终答案。"})
        
        final_answer = self.leader.generate_response(history, self.max_rounds + 1)

        print("\n" + "="*90)
        print("🎯 【最终答案】")
        print(final_answer)
        print("="*90)

        logging.info("✅ 任务完成")
        return final_answer


if __name__ == "__main__":
    swarm = MultiAgentSwarm()
    swarm.solve("请帮我写一篇关于「人工智能如何改变软件开发」的深度分析报告，并保存到 ./reports/ai_impact_report.md")
