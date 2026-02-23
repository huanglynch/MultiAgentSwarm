# Multi-Agent Swarm v2.2

**一个极简、可靠、高效的多智能体群智慧框架**  
仅需 **1 个 Python 文件 + 1 个 YAML 配置 + skill/ 目录**，即可实现类似 Grok 4.2 的多智能体协作。

支持：
- **并行执行**（每轮所有智能体同时思考）
- **每个智能体独立 LLM**（不同 API Key、base_url、model）
- **流式输出**（per-agent 可开关）
- **固定轮次讨论**（默认 3 轮，**绝对无死循环**）
- **动态 Skill 系统**（`skill/` 目录下 `.py` 自动转为 Tool Calling，`.md` 自动注入共享知识）
- **OpenAI 完全兼容接口**（OpenAI、Groq、DeepSeek、Ollama、vLLM、本地等）
- **完整日志**（`swarm.log`）
- **max_tokens 默认 4096**（推荐值，可全局/单 Agent 配置）

---

## 特性亮点（第一性原理设计）

- **最小改动**：单文件主程序，无需任何框架
- **最可靠**：固定轮次 + 单轮 Tool Calling + 异常隔离
- **最易扩展**：Skill 放 `skill/` 即可，YAML 配置一切
- **并行加速**：4 个智能体 ≈ 最慢 1 个的时间
- **生产可用**：日志、错误处理、流式、文件读写全部内置

---

## 目录结构

```
multi-agent-swarm/
├── multi_agent_swarm_v2.py     # 主程序（唯一需要运行的文件）
├── swarm_config.yaml           # 全部配置
├── swarm.log                   # 自动生成的运行日志
├── skill/                      # 动态 Skill 目录
│   ├── read_file.py
│   ├── write_file.py
│   ├── list_dir.py
│   └── knowledge.md            # 任意 .md 文件 → 自动成为全团队共享知识
├── reports/                    # 示例输出目录（可自行创建）
└── output/                     # 示例输出目录（可自行创建）
```

---

## 安装

```bash
pip install openai pyyaml
```

**无需其他依赖**（Python 3.8+）

---

## 配置（swarm_config.yaml）

```yaml
openai:
  default_model: "gpt-4o-mini"
  default_max_tokens: 4096          # 我推荐的全局默认值

swarm:
  num_agents: 4                     # 默认智能体数量
  max_rounds: 3                     # 默认讨论轮次（固定，不会死循环）
  log_file: "swarm.log"
  skills_dir: "skill"               # Skill 目录路径

agents:
  - name: Grok
    role: "你是团队领导者，风格幽默实用，负责协调与最终高质量输出。"
    api_key: "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    temperature: 0.75
    stream: true
    max_tokens: 8192                # 可覆盖全局默认
    enabled_tools: ["read_file", "write_file", "list_dir"]

  - name: Harper
    role: "你是创意研究员，专注新颖想法和跨领域洞见。"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    temperature: 0.9
    stream: true
    max_tokens: 4096
    enabled_tools: ["read_file", "list_dir"]

  - name: Benjamin
    role: "你是严谨逻辑分析师，专注事实检查和严密推理。"
    api_key: "sk-..."
    base_url: "https://api.deepseek.com"
    model: "deepseek-chat"
    temperature: 0.5
    stream: false
    max_tokens: 4096
    enabled_tools: ["list_dir"]

  - name: Lucas
    role: "你是执行与总结专家，擅长把观点整合成可落地方案。"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    temperature: 0.65
    stream: true
    max_tokens: 4096
    enabled_tools: ["write_file"]
```

---

## 使用方法

### 1. 直接运行（最简单）

```bash
python multi_agent_swarm_v2.py
```

程序会自动读取配置并执行示例任务。

### 2. 在其他代码中调用（推荐）

```python
from multi_agent_swarm_v2 import MultiAgentSwarm

swarm = MultiAgentSwarm()                     # 自动读取 swarm_config.yaml
answer = swarm.solve("请帮我写一篇关于量子计算的科普文章，并保存到 ./reports/quantum.md")
print(answer)
```

---

## 示例任务

```python
# 示例1：读取知识库并生成报告
swarm.solve("读取 skill/knowledge.md 中的内容，结合当前东京房价趋势，写一篇分析报告并保存到 ./reports/tokyo_housing.md")

# 示例2：多智能体 brainstorm
swarm.solve("用最通俗语言解释量子纠缠，并说明它在量子计算中的应用")

# 示例3：代码相关任务
swarm.solve("帮我设计一个 FastAPI 用户注册接口，并把完整代码写入 ./output/user_api.py")
```

---

## 如何添加新 Skill（30 秒搞定）

1. 在 `skill/` 目录新建 `my_skill.py`

```python
# skill/my_skill.py
def execute(param1: str, param2: int = 10) -> str:
    return f"执行成功！param1={param1}, param2={param2}"

schema = {
    "type": "function",
    "function": {
        "name": "my_skill",
        "description": "我的自定义工具描述",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "参数1"},
                "param2": {"type": "integer", "description": "参数2"}
            },
            "required": ["param1"]
        }
    }
}
```

2. 在任意 Agent 的 `enabled_tools` 中添加 `"my_skill"`

3. 重启程序即可使用！

**支持格式**：
- `.py` → Tool Calling（必须包含 `execute` 函数 + `schema`）
- `.md` → 自动注入全团队 System Prompt（共享知识库）

---

## 工作流程（透明说明）

1. 启动 → 加载 YAML + 动态扫描 `skill/`
2. **每轮并行**：所有智能体同时调用各自 LLM（支持流式）
3. 支持 Tool Calling（单轮，最可靠）
4. 固定 `max_rounds` 轮后 → Leader 最终综合输出
5. 全程记录 `swarm.log`

**绝对不会死循环**（固定 for 循环）

---

## 高级配置

- 想限制并发：修改 `ThreadPoolExecutor(max_workers=2)`
- 想异步版本：告诉我，我 1 分钟给你 async 版
- 想更多内置 Skill：直接放 `skill/` 即可

---

## 注意事项

- 请确保 `api_key` 和 `base_url` 正确
- 大模型费用按实际 token 消耗计算
- `skill/` 目录必须存在（否则会警告但仍可运行）
- 推荐将 `reports/`、`output/` 加入 `.gitignore`

---

**Made with ❤️ from 第一性原理**  
任何问题、想要新功能（持久化历史、Web UI、自动评估等），直接告诉我，我继续用**最小改动**给你升级！

**当前版本：v2.2（2026.02）**

