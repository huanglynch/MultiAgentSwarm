# Multi-Agent Swarm v2

**一个极简、可靠、高效的多智能体群智慧框架**

支持固定模式与智能模式，可实现代码审查、Bug调查、深度报告写作、图像分析等多种复杂任务，像 Grok 4.2 团队一样进行协作式智能工作。

**v2.9 主要特性**：
- 双模式：`fixed`（快速固定轮次） / `intelligent`（智能评价 + 自动迭代改进）
- 真正图像输入支持（最多2张图片，可在配置文件中设置）
- 持久化记忆（JSON + Chroma向量数据库）
- 动态 Skill 系统（skills/ 目录下 .py 自动转为工具，.md 自动注入知识）
- 并行执行 + 流式输出
- 网络搜索默认关闭 + 自动随机延时（更安全）
- 加强沙箱 + 执行超时保护
- Reflection + Planning 多轮反思循环

---

## 目录结构

```
multi-agent-swarm/
├── multi_agent_swarm_v2.9.py       # 主程序
├── swarm_config.yaml               # 配置
├── memory.json                     # 持久化记忆
├── memory_db/                      # Chroma向量数据库
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
  mode: "intelligent"               # fixed 或 intelligent（推荐智能模式用于深度任务）
  num_agents: 4
  max_rounds: 12                    # fixed模式建议3，intelligent模式建议8~15
  reflection_planning: true         # 是否开启多轮反思规划循环（默认开启）
  enable_web_search: false          # 默认关闭网络搜索
  max_images: 2                     # 最多支持2张图片输入
  memory_file: "memory.json"
  max_memory_items: 50
  skills_dir: "skills"              # Skill 目录（复数）

agents:
  - name: Grok
    role: "你是团队领导者，风格幽默实用，负责协调与最终高质量输出。"
    api_key: "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    temperature: 0.75
    stream: true
    max_tokens: 8192
    enabled_tools: ["read_file", "write_file", "list_dir", "web_search"]

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
from multi_agent_swarm_v2.9 import MultiAgentSwarm

swarm = MultiAgentSwarm()
answer = swarm.solve("你的任务")
```

### 带记忆 + 图像输入
```python
answer = swarm.solve(
    task="请分析这两张图片中的代码 Bug",
    use_memory=True,
    memory_key="bug_topic",
    image_paths=["./screenshot1.png", "./screenshot2.png"]
)
```

---

## 主要应用场景

### Code Review（强烈推荐）
```python
swarm.solve("请对 ./code/main.py 进行全面代码审查，输出详细报告并生成修复版")
```

### Bug Investigation（强烈推荐）
```python
swarm.solve("请调查登录失败 Bug（ERR-401），读取 ./logs/app.log 和 ./src/auth.py")
```

### 深度报告写作
```python
swarm.solve("写一篇关于人工智能的深度分析报告（3000字以上），保存到 ./reports/ai_report.md", use_memory=True, memory_key="ai_topic")
```

### 图像分析
```python
swarm.solve("请分析这两张图片中的问题", image_paths=["./img1.png", "./img2.png"])
```

---

## Reflection + Planning 循环（v2.5.2 核心功能）

开启后，每轮自动执行：
1. **Plan**：Leader 规划本轮重点
2. **Act**：Agent 执行
3. **Reflect**：Leader 反思结果
4. **Decide**：决定继续或停止

显著提升输出质量和逻辑性。

---

## 如何添加新 Skill

1. 在 `skills/` 新建 `my_skill.py`
2. 使用模板（见之前版本）

---

## 注意事项

- `enable_web_search` 默认关闭，建议只在需要时开启
- 网络搜索每次自动随机延时 0.5~2 秒
- 目录（如 reports/）会自动创建
- 智能模式质量更高，但耗时和 token 消耗也更大

**当前版本：v2.9**  
如需进一步优化或新增功能，随时告诉我！

欢迎使用与反馈！
