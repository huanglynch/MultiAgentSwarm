# Multi-Agent Swarm v2.3

**一个极简、可靠、高效的多智能体群智慧框架**  
**支持固定模式 + 智能模式**，像 Grok 4.2 团队一样实现真正涌现的群智慧。

**核心升级（v2.3）**：
- **智能模式**（intelligent）：不再受固定轮次约束，**Leader 每轮智能评价质量 + 决定是否继续改进**，力求输出**最高质量**答案。
- **固定模式**（fixed）：保持原有极简可靠（默认 3 轮）。
- **安全机制**：智能模式下仍有**硬性最大轮次上限**（默认 10），**绝对不会死循环**。
- 其余特性全部保留：并行执行、独立 LLM、流式输出、动态 Skill（skill/ 目录）、OpenAI 完全兼容、完整日志。

---

## 特性亮点（第一性原理设计）

- **最小改动**：仍然**单文件主程序 + 一个 YAML 配置**
- **双模式灵活**：
  - `fixed`：速度最快、成本最低、极度可靠
  - `intelligent`：质量最高、自动迭代改进、接近真实团队 brainstorm
- **并行加速**：每轮所有智能体**真正同时工作**
- **动态 Skill 系统**：`skill/` 下 `.py` 自动变 Tool Calling，`.md` 自动注入共享知识
- **生产级可靠**：固定/智能双保险 + 异常隔离 + 完整日志

---

## 目录结构

```
multi-agent-swarm/
├── multi_agent_swarm_v2.py     # 主程序（唯一运行文件）
├── swarm_config.yaml           # 全部配置（新增 mode）
├── swarm.log                   # 自动生成的详细日志
├── skill/                      # Skill 目录
│   ├── read_file.py
│   ├── write_file.py
│   ├── list_dir.py
│   └── knowledge.md            # 任意 .md 自动成为团队共享知识
├── reports/                    # 示例输出目录（建议创建）
└── output/                     # 示例输出目录（建议创建）
```

---

## 安装

```bash
pip install openai pyyaml
```

**Python 3.8+** 即可，无其他依赖。

---

## 配置（swarm_config.yaml）

```yaml
openai:
  default_model: "gpt-4o-mini"
  default_max_tokens: 4096          # 推荐全局默认值

swarm:
  mode: "intelligent"               # ← 新增！固定用 "fixed"，追求最高质量用 "intelligent"
  num_agents: 4
  max_rounds: 10                    # fixed 模式默认 3，intelligent 模式默认 10（安全上限）
  log_file: "swarm.log"
  skills_dir: "skill"

agents:
  - name: Grok
    role: "你是团队领导者，风格幽默实用，负责协调与最终高质量输出。"
    api_key: "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    temperature: 0.75
    stream: true
    max_tokens: 8192
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

### 2. 在其他代码中调用（推荐）

```python
from multi_agent_swarm_v2 import MultiAgentSwarm

swarm = MultiAgentSwarm()   # 自动读取 swarm_config.yaml
answer = swarm.solve("请帮我写一篇关于『2026 年东京人工智能产业趋势』的深度报告，并保存到 ./reports/tokyo_ai_2026.md")
print(answer)
```

---

## 示例任务

```python
# 示例1：智能模式下追求最高质量
swarm.solve("用最通俗语言解释量子纠缠，并说明它在量子计算中的应用")

# 示例2：结合 Skill 生成报告
swarm.solve("读取 skill/knowledge.md 中的内容，结合当前东京房价趋势，写一篇分析报告并保存到 ./reports/tokyo_housing.md")

# 示例3：代码生成任务
swarm.solve("帮我设计一个 FastAPI 用户注册接口，并把完整代码写入 ./output/user_api.py")
```

---

## 工作流程（透明说明）

### 固定模式（mode: "fixed"）
1. 初始化历史
2. **固定轮次**（默认 3 轮）并行让所有 Agent 贡献
3. Leader 最终综合 → 输出

**优点**：速度快、可预测、成本低。

### 智能模式（mode: "intelligent"）← 新增核心
1. 第 1 轮并行贡献
2. **Leader 智能评价**（输出结构化 JSON：quality_score、decision=continue/stop、reason、suggestions）
3. 如果 Leader 判断已达最高质量（score ≥8 且 decision=stop）→ 立即输出最终答案
4. 否则根据 suggestions 继续下一轮改进（所有 Agent 自动看到改进方向）
5. **硬上限**：达到 `max_rounds`（默认 10）强制停止

**优点**：真正追求“最好最高质量”，自动迭代，像真实团队一样自我改进。

**绝对不会死循环**：无论哪种模式都有硬性轮次上限 + 日志全记录。

---

## 如何添加新 Skill（30 秒）

1. 在 `skill/` 新建 `my_skill.py`（模板同 v2.2）
2. 在任意 Agent 的 `enabled_tools` 中加入名称
3. 重启即可（自动加载）

---

## 高级配置与注意事项

- **切换模式**：只需改 `swarm.mode` 为 `fixed` 或 `intelligent`
- **调整上限**：`max_rounds` 越大，智能模式质量越高（但成本也越高）
- **流式输出**：仅 Leader 在最终综合时流式显示更清晰
- **费用提醒**：智能模式 token 消耗更高，请根据需求选择
- **推荐**：复杂/高质量需求用 `intelligent`，快速验证用 `fixed`

---

**Made with ❤️ from 第一性原理**  
当前版本：**v2.3（2026.02）**  
