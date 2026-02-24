# Multi-Agent Swarm v2.5.2

**一个极简、可靠、高效的多智能体群智慧框架**

支持固定模式与智能模式，可实现代码审查、Bug调查、深度报告写作等多种复杂任务，像 Grok 4.2 团队一样进行协作式智能工作。

**v2.5.2 主要升级**：
- Skill 目录统一改为 `skills/`（复数）
- 强化 Reflection + Planning 循环（真正的 Plan → Act → Reflect）
- 网络搜索默认关闭 + 自动随机延时（更安全）
- 持久化记忆（JSON + Chroma 向量数据库）
- 完整 Code Review 和 Bug Investigation 使用指南

---

## 目录结构

```
multi-agent-swarm/
├── multi_agent_swarm_v2.5.2.py     # 主程序
├── swarm_config.yaml               # 配置
├── memory.json                     # 持久化记忆
├── memory_db/                      # Chroma 向量数据库
├── skills/                         # Skill 目录（复数）
│   ├── read_file.py
│   ├── write_file.py
│   ├── list_dir.py
│   └── knowledge.md
├── reports/                        # 输出目录
├── output/                         # 输出目录
└── swarm.log                       # 运行日志
```

---

## 安装

```bash
pip install openai pyyaml duckduckgo-search beautifulsoup4 chromadb
```

---

## 配置（swarm_config.yaml 完整示例）

```yaml
openai:
  default_model: "gpt-4o-mini"
  default_max_tokens: 4096

swarm:
  mode: "intelligent"               # fixed 或 intelligent（推荐智能模式）
  num_agents: 4
  max_rounds: 12                    # fixed 模式建议 3，intelligent 模式建议 8~15
  reflection_planning: true         # 是否开启 Plan-Act-Reflect 循环（默认开启）
  enable_web_search: false          # 默认关闭网络搜索
  memory_file: "memory.json"
  max_memory_items: 50
  skills_dir: "skills"              # Skill 目录（复数）

agents:
  - name: Grok
    role: "你是团队领导者，负责协调、质量把控与最终高质量输出。"
    api_key: "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    temperature: 0.75
    stream: true
    max_tokens: 8192
    enabled_tools: ["read_file", "write_file", "list_dir", "web_search"]   # Leader 可搜索

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

### 基础调用
```python
from multi_agent_swarm_v2.5.2 import MultiAgentSwarm

swarm = MultiAgentSwarm()
answer = swarm.solve("你的任务")
```

### 带记忆调用（推荐长期任务）
```python
answer = swarm.solve(
    task="继续上次的 Bug 调查",
    use_memory=True,           # 是否加载历史记忆
    memory_key="login_bug"     # 按主题分组记忆
)
```

---

## 主要应用场景

### Code Review（强烈推荐）
```python
swarm.solve("请对 ./code/main.py 进行全面代码审查，输出详细报告并生成修复版保存到 ./review/fixed_main.py")
```

### Bug Investigation（强烈推荐）
```python
swarm.solve("请调查登录失败 Bug（ERR-401），读取 ./logs/app.log 和 ./src/auth.py，给出完整根因分析和修复方案")
```

### 深度报告写作
```python
swarm.solve("写一篇关于人工智能的深度分析报告（3000字以上），并保存到 ./reports/ai_report.md", use_memory=True, memory_key="ai_topic")
```

---

## Reflection + Planning 循环（v2.5.2 核心功能）

开启 `reflection_planning: true` 后，智能模式下每轮会自动执行：
1. **Plan**：Leader 规划本轮重点方向
2. **Act**：所有 Agent 执行计划
3. **Reflect**：Leader 反思结果，给出质量评估和改进建议
4. 决定继续或停止

这使得系统能**自我迭代优化**，显著提升输出质量。

---

## 如何添加新 Skill

1. 在 `skills/` 目录新建 `my_skill.py`
2. 使用以下模板：

```python
def execute(param1: str) -> str:
    return f"执行结果: {param1}"

schema = {
    "type": "function",
    "function": {
        "name": "my_skill",
        "description": "工具描述",
        "parameters": {
            "type": "object",
            "properties": {"param1": {"type": "string"}},
            "required": ["param1"]
        }
    }
}
```

3. 在 Agent 的 `enabled_tools` 中添加名称即可。

---

## 注意事项

- `enable_web_search` 默认关闭，需手动开启
- 网络搜索每次自动随机延时 0.5~2 秒，降低风险
- `./reports/`、`./output/` 等目录会自动创建
- 智能模式质量更高，但耗时和 token 消耗也更大
- 首次运行会自动创建 `memory_db/` 目录

**当前版本：v2.5.2**  
如需进一步优化（异步版本、Web UI、更多工具等），随时告诉我！

欢迎使用与反馈！
