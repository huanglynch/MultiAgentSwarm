# Multi-Agent Swarm v2.4

**一个极简、可靠、高效的多智能体群智慧框架**  
**支持固定模式 + 智能模式**，像 Grok 4.2 团队一样实现真正涌现的群智慧。

**核心升级（v2.3）**：
- **智能模式**（intelligent）：**不再受固定轮次约束**，Leader 每轮自动进行**智能评价 + 质量打分 + 改进建议**，直到 Leader 判断“已达最高质量”为止。
- **固定模式**（fixed）：保持原有极简可靠，默认 3 轮。
- **安全机制**：智能模式仍有**硬性最大轮次上限**（默认 10），**绝对不会死循环**。
- 其余全部特性保留：并行执行、每个智能体独立 LLM、流式输出、动态 Skill 系统（skill/ 目录下 .py/.md 自动加载）、OpenAI 完全兼容接口、完整日志、max_tokens 默认 4096 等。

**v2.4 新增**：
- **持久化记忆机制**：跨多次 solve() 调用保留关键记忆（JSON 文件）
- **use_memory 参数**：单次任务可选择是否加载历史记忆
- 完整 Code Review 和 Bug Investigation 专区
---

## 特性亮点（第一性原理设计）

- **持久化记忆（跨会话）**
- **双模式灵活切换**：
  - `fixed`：速度最快、成本最低、极度可预测
  - `intelligent`：质量最高、自动迭代改进、追求“最好最高质量输出”
- **真正并行**：每轮所有智能体**同时**思考、调用 LLM、执行 Tool（速度提升 3~4 倍）
- **动态 Skill 系统**：把 Skill 放到 `skill/` 即可自动加载，无需改主代码
- **OpenAI 完全兼容接口**
- **极简架构（单文件主程序）**，只有 **1 个 Python 文件** + **1 个 YAML 配置** + **skill/ 目录**
- **生产级可靠**：异常隔离、单轮 Tool Calling、固定/智能双保险 + 全程日志
- **易扩展**：后续加 Tool、异步、Web UI 等都只需极小改动

---

## 目录结构

```
multi-agent-swarm/
├── multi_agent_swarm_v2.py          # 主程序（唯一需要运行的文件）
├── swarm_config.yaml                # 全部配置（新增 mode 字段）
├── swarm.log                        # 自动生成的完整运行日志（每轮、每 Agent、Tool 调用全记录）
├── skill/                           # Skill 独立目录（推荐）
│   ├── read_file.py
│   ├── write_file.py
│   ├── list_dir.py
│   └── knowledge.md                 # 任意 .md 文件 → 自动注入全团队共享知识库
├── reports/                         # 示例输出目录（建议手动创建）
├── output/                          # 示例输出目录（建议手动创建）
└── README.md                        # 本文档
```

---

## 安装（最简）

```bash
pip install openai pyyaml
```

**仅需这两个包**，Python 3.8+ 即可运行，无任何其他依赖。

---

## 配置（swarm_config.yaml）完整示例

```yaml
openai:
  default_model: "gpt-4o-mini"
  default_max_tokens: 4096          # 我推荐的全局默认值（足够详细输出）

swarm:
  mode: "intelligent"               # ← 新增！"fixed" 或 "intelligent"
  num_agents: 4                     # 默认智能体数量（可配置 2~20）
  max_rounds: 10                    # fixed 模式默认 3，intelligent 模式默认 10（安全上限）
  memory_file: "memory.json"      # 新增
  max_memory_items: 50            # 新增
  log_file: "swarm.log"
  skills_dir: "skills"               # Skill 目录路径

agents:
  - name: Grok
    role: "你是团队领导者，风格幽默实用，负责协调、质量把控与最终高质量输出。"
    api_key: "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    temperature: 0.75
    stream: true
    max_tokens: 8192
    enabled_tools: ["read_file", "write_file", "list_dir"]

  - name: Harper
    role: "你是创意研究员，专注新颖想法、跨领域洞见和脑洞大开的可能性。"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    temperature: 0.9
    stream: true
    max_tokens: 4096
    enabled_tools: ["read_file", "list_dir"]

  - name: Benjamin
    role: "你是严谨逻辑分析师，专注事实检查、找出漏洞、提供严密推理。"
    api_key: "sk-..."
    base_url: "https://api.deepseek.com"
    model: "deepseek-chat"
    temperature: 0.5
    stream: false
    max_tokens: 4096
    enabled_tools: ["list_dir"]

  - name: Lucas
    role: "你是执行与总结专家，擅长把所有观点整合成可落地、可操作的最终方案。"
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

### 1. 直接运行（最简单方式）

```bash
python multi_agent_swarm_v2.py
```

程序会自动读取配置并执行 `if __name__ == "__main__"` 中的示例任务。

### 2. 在其他 Python 代码中导入调用（推荐生产方式）

```python
from multi_agent_swarm_v2 import MultiAgentSwarm

# 初始化（自动读取 swarm_config.yaml）
swarm = MultiAgentSwarm()

# 调用 solve 方法
answer = swarm.solve("请帮我写一篇关于『2026 年东京人工智能产业趋势』的深度报告，并保存到 ./reports/tokyo_ai_2026.md")

print("\n=== 最终答案 ===")
print(answer)
```

```python
from multi_agent_swarm_v2 import MultiAgentSwarm
swarm = MultiAgentSwarm()

# 普通调用
answer = swarm.solve("你的任务")

# 带记忆调用
answer = swarm.solve(
    task="继续上次的 Bug 调查",
    use_memory=True,
    memory_key="login_bug"   # 按主题分组记忆
)
```
---

## 示例任务（直接复制使用）

```python
# 示例1：智能模式下追求最高质量
swarm.solve("用最通俗语言解释量子纠缠，并说明它在量子计算中的潜在应用")

# 示例2：结合 Skill 生成并保存文件
swarm.solve("读取 skill/knowledge.md 中的内容，结合当前东京房价趋势，写一篇详细分析报告并保存到 ./reports/tokyo_housing.md")

# 示例3：代码生成 + 文件写入
swarm.solve("帮我设计一个 FastAPI 用户注册接口（包含密码哈希），把完整可运行代码写入 ./output/user_api.py")

# 示例4：快速固定模式测试
# （先把 swarm_config.yaml 中的 mode 改成 "fixed"）
swarm.solve("简单总结一下 Grok 4.2 的多智能体设计思路")
```

---

## Code Review 使用专区（v2.3 特别推荐）

**是的，这个 Swarm 天生就是做高质量 Code Review 的神器！**  
开启 **智能模式** 后，效果远超单个 LLM，能实现多视角、迭代式、极致详细的代码审查（支持“凑字数”长篇报告）。

### 为什么特别适合 Code Review？
- **4 个智能体并行**：同时从安全、性能、架构、可读性、扩展性等不同角度审查
- **智能迭代**：Leader 每轮自动打分（1-10）、给出改进建议，直到质量达到最高（通常 9~10 分）
- **支持文件读写**：直接读取你的代码文件，审查完还能自动生成修复版或完整报告
- **输出极详细**：问题分级（高/中/低）、风险评分、修复代码片段、整体总结、改进后完整代码
- **安全无死循环**：硬上限 10~15 轮

### 推荐 Code Review 配置（直接复制替换 swarm_config.yaml 中的 agents 部分）

```yaml
swarm:
  mode: "intelligent"      # ← 必须用 intelligent
  num_agents: 4
  max_rounds: 12           # 可调到 12~15，追求极致深度

agents:
  - name: Grok
    role: "你是首席代码审查官（Leader），负责统筹、打分、最终输出结构化长篇报告。风格专业、严谨、建设性、极度详细。"
    api_key: "..."
    base_url: "..."
    model: "gpt-4o"
    temperature: 0.7
    stream: true
    max_tokens: 8192
    enabled_tools: ["read_file", "write_file", "list_dir"]

  - name: Harper
    role: "你是架构与设计审查专家，专注代码结构、设计模式、可扩展性、可维护性、未来演进建议。"
    enabled_tools: ["read_file"]

  - name: Benjamin
    role: "你是安全与性能审查专家，专注漏洞、SQL注入、XSS、性能瓶颈、并发问题、内存/CPU 优化。"
    enabled_tools: ["read_file"]

  - name: Lucas
    role: "你是代码规范与可读性专家，专注 PEP8、命名规范、注释、测试覆盖率、重构建议、Clean Code 原则。"
    enabled_tools: ["read_file", "write_file"]
```

### Code Review 使用示例

```python
from multi_agent_swarm_v2 import MultiAgentSwarm
swarm = MultiAgentSwarm()

# 示例1：审查单个文件（最常用）
review = swarm.solve("请对文件 ./code/my_app.py 进行全面代码审查。要求：1. 列出所有问题并分级（高/中/低）；2. 给出每个问题的修复代码片段；3. 输出改进后的完整版本并保存到 ./review/fixed_my_app.py；4. 最后给出整体评分（1-10分）和详细改进总结。")

# 示例2：直接粘贴代码片段审查
review = swarm.solve("""
请对以下代码进行深度 Code Review（要求输出 3000+ 字详细报告）：

```python
def process_user_data(data):
    # 这里粘贴你的代码
    ...
```

# 示例3：审查整个项目目录
review = swarm.solve("先用 list_dir 列出 ./src/ 目录所有 .py 文件，然后逐个进行代码审查，最后输出一份完整的项目审查报告（包含优先级排序和重构路线图）。")
```

**想让 review 更长、更详细？**  
- 把 `max_rounds` 设为 15  
- 在 Leader 的 role 里加上“输出必须非常详细、举例充分、包含多角度分析和最佳实践对比”  
- 智能模式下会自动迭代，直到 Leader 满意为止
```
---

## Bug Investigation 使用专区（v2.3 新增 · 强烈推荐）

**是的，这个 Swarm 非常适合做 Bug 调查与根因分析（Root Cause Analysis）！**

尤其是智能模式下，多智能体能从日志、代码、假设、环境等多维度并行调查，自动迭代直到确认根因，输出极详细的结构化报告。

### 为什么特别适合？
- 并行多视角（日志分析、代码审查、假设生成、修复验证）
- 智能迭代：Leader 每轮打分 + 建议下一步，直到质量最高
- 直接读写文件：轻松处理日志、stack trace、源码、config
- 输出极详尽：复现步骤、证据链、概率排序、修复 patch、回归测试建议

### 推荐 Bug Investigation 配置

```yaml
swarm:
  mode: "intelligent"
  max_rounds: 15

agents:
  - name: Grok
    role: "你是首席 Bug 调查官（Leader），负责统筹线索、构建证据链、最终输出极详细结构化报告。"
    stream: true
    max_tokens: 8192
    enabled_tools: ["read_file", "write_file", "list_dir"]

  - name: Harper
    role: "假设生成与创新调查专家，专注多种可能根因和隐藏关联。"
    enabled_tools: ["read_file"]

  - name: Benjamin
    role: "证据验证与日志分析专家，专注时间线、并发、竞态条件。"
    enabled_tools: ["read_file"]

  - name: Lucas
    role: "修复与验证专家，专注最小 patch、可复现测试、风险评估。"
    enabled_tools: ["read_file", "write_file"]
```

### 使用示例

```python
# 示例1：日志 + 代码联合调查
swarm.solve("请调查登录失败 Bug（ERR-401）。已提供 ./logs/app.log 和 ./src/auth.py。请输出完整报告并保存到 ./bug_reports/ERR-401.md")

# 示例2：直接贴错误信息
swarm.solve("请调查以下 TypeError 根因并给出修复：（贴 traceback）")

# 示例3：项目级调查
swarm.solve("列出 ./src/ 最近修改文件，读取 error.log，输出完整 Bug 调查报告。")
```

**想让报告更长更详细？** 把 `max_rounds` 调高 + 在 Grok role 中强调“极度详尽、多表格、多代码片段”即可。

## 工作流程（透明说明）

### 固定模式（mode: "fixed"）
1. 用户输入任务 → history 初始化
2. **固定轮次**（默认 3 轮）并行让所有 Agent 同时贡献
3. Leader 进行最终综合 → 输出答案

### 智能模式（mode: "intelligent"）← v2.3 核心
1. 第 1 轮并行贡献
2. **Leader 智能评价**（自动输出结构化 JSON：quality_score 1-10、decision "continue/stop"、reason、suggestions）
3. 如果 Leader 判断质量已足够高（通常 score ≥8 且 decision=stop）→ 立即停止并输出最终最高质量答案
4. 否则根据 suggestions 继续下一轮（所有 Agent 会自动看到改进方向，继续迭代）
5. **安全上限**：达到 `max_rounds`（默认 10）时强制结束

**绝对不会死循环**：无论哪种模式都有硬性轮次上限 + 完整日志记录。

---

## 如何添加新 Skill（30 秒搞定）

1. 在 `skill/` 目录新建 `my_new_skill.py`
2. 复制下面模板（略，与之前相同）
3. 在任意 Agent 的 `enabled_tools` 列表里加上名称
4. 重启程序即可使用！

---

## 高级配置与注意事项

- **切换模式**：只需修改 `swarm.mode` 为 `fixed` 或 `intelligent`
- **调整质量追求**：智能模式下把 `max_rounds` 设为 15~20 可获得更高品质（但 token 消耗也会增加）
- **流式输出**：每个 Agent 可独立开启，Leader 在最终综合时会更清晰地流式显示
- **费用提醒**：智能模式 + Code Review 通常消耗更多 token，请根据任务复杂度选择
- **日志查看**：每次运行后查看 `swarm.log`，可追溯每一轮、每个 Agent、每个 Tool 调用
- **推荐用法**：
  - 快速验证 / 简单任务 → `fixed`
  - 深度分析 / 报告 / **Code Review** / 需要最高质量 → `intelligent`
- **目录建议**：把 `reports/`、`output/`、`review/` 加入 `.gitignore`

---

**Made with ❤️ from 第一性原理**  
当前版本：**v2.3（2026.02）**  
