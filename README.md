# Multi-Agent Swarm v2.5

**一个极简、可靠、高效的多智能体群智慧框架**

支持固定模式与智能模式，可实现代码审查、Bug调查、报告写作、持续学习等多种复杂任务，像 Grok 4.2 团队一样进行协作式智能工作。

**v2.5 主要特性**：
- 双模式切换：`fixed`（快速固定轮次） / `intelligent`（智能评价 + 自动迭代改进）
- 持久化记忆：跨多次任务保留关键结论（JSON + Chroma 向量数据库）
- 动态 Skill 系统：`skill/` 目录下 .py 自动转为 Tool Calling，.md 自动注入共享知识
- 并行执行 + 流式输出
- 网络搜索默认关闭（可配置开启，并自动随机延时）
- 安全沙箱 Python 执行
- 完整日志记录

---

## 目录结构

```
multi-agent-swarm/
├── multi_agent_swarm_v2.5.py     # 主程序
├── swarm_config.yaml             # 全部配置
├── memory.json                   # 持久化记忆文件
├── memory_db/                    # Chroma 向量数据库目录
├── skill/                        # Skill 目录（推荐结构）
│   ├── read_file.py
│   ├── write_file.py
│   ├── list_dir.py
│   └── knowledge.md              # 任意 .md 自动成为团队知识库
├── reports/                      # 示例输出目录
├── output/                       # 示例输出目录
└── swarm.log                     # 详细运行日志
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
  mode: "intelligent"               # fixed 或 intelligent（推荐智能模式用于深度任务）
  num_agents: 4
  max_rounds: 12                    # fixed 模式建议 3，intelligent 模式建议 8~15
  enable_web_search: false          # 默认关闭网络搜索（安全考虑）
  reflection_planning: true         # 是否开启反思规划循环（默认开启）
  memory_file: "memory.json"
  max_memory_items: 50
  skills_dir: "skill"

agents:
  - name: Grok
    role: "你是团队领导者，负责协调、质量把控与最终高质量输出。"
    api_key: "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    temperature: 0.75
    stream: true
    max_tokens: 8192
    enabled_tools: ["read_file", "write_file", "list_dir", "web_search"]  # Leader 可搜索

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
    enabled_tools: ["read_file", "list_dir"]

  - name: Lucas
    role: "你是执行与总结专家，擅长整合观点并输出可落地方案。"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    temperature: 0.65
    stream: true
    max_tokens: 4096
    enabled_tools: ["read_file", "write_file"]
```

---

## 使用方法

### 基础调用
```python
from multi_agent_swarm_v2.5 import MultiAgentSwarm

swarm = MultiAgentSwarm()

answer = swarm.solve("你的任务描述")
```

### 带记忆调用（推荐长期任务）
```python
answer = swarm.solve(
    task="继续上次的 Bug 调查",
    use_memory=True,           # 是否加载历史记忆
    memory_key="login_bug"     # 按主题分组记忆
)
```
### 图像输入示例：
```python
swarm.solve("请分析这两张图片中的代码 Bug", image_paths=["./img1.png", "./img2.png"])
```
---

## 主要应用场景示例

### 1. Code Review
```python
swarm.solve("请对 ./code/main.py 进行全面代码审查，输出详细报告并生成修复版本保存到 ./review/fixed_main.py")
```

### 2. Bug Investigation
```python
swarm.solve("请调查登录失败 Bug（ERR-401），读取 ./logs/app.log 和 ./src/auth.py，给出完整根因分析和修复方案")
```

### 3. 深度报告写作
```python
swarm.solve("写一篇关于人工智能的深度分析报告（3000字以上），并保存到 ./reports/ai_report.md", use_memory=True, memory_key="ai_topic")
```

### 4. 结合网络搜索的实时任务
（先把 `enable_web_search: true`）
```python
swarm.solve("搜索2026年最新AI趋势，结合本地知识写一篇报告")
```

---

## 如何添加新 Skill

1. 在 `skill/` 目录新建 `my_skill.py`
2. 模板：

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

3. 在 Agent 的 `enabled_tools` 中添加 `"my_skill"`

---

## 工作流程

**固定模式**：固定轮次 → 并行贡献 → Leader 综合 → 结束  
**智能模式**：并行贡献 → Leader 智能评价（质量打分 + 决策）→ 决定继续改进或停止 → 最终输出

---

## 注意事项

- `enable_web_search` 默认关闭，建议只在需要实时信息时开启
- 智能模式质量更高，但耗时和 token 消耗也更大
- `./reports/`、`./output/` 等目录会自动创建
- 网络搜索每次自动随机延时 0.5~2 秒，降低风险
- 首次使用会自动创建 `memory_db/` 目录

**当前版本：v2.5**  
如需进一步优化（异步版本、Web UI、更多工具等），随时告诉我！

欢迎使用与反馈！
