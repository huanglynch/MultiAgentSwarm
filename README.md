# MultiAgentSwarm v4.1.0 (ReAct Visualization Edition)
**Self-Adaptive Digital Team**  
**A fully visible, self-evolving ReAct Digital Team** that intelligently decides *when to be fast* and *when to go deep*.
**MultiAgentSwarm v4.1.0** 不再是简单的“多个LLM并行聊天”，而是一个**完全可视化、智能决策、自进化 ReAct 数字团队** —— 它完美还原经典 ReAct 架构，同时注入群体智能、动态规划、分层终身记忆与生产级交付能力。

<p align="center">
  <strong>
    <a href="#english-version">🇬🇧 English Version</a>
      |  
    <a href="#chinese-version">🇨🇳 中文版</a>
  </strong>
</p>

<p align="center">
  <img src="images/architecture-diagram.png" alt="MultiAgentSwarm v3.2.0 Architecture" width="95%" style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
</p>

---

## English Version <a id="english-version"></a>

**MultiAgentSwarm v4.1.0**  
**A fully visible, self-evolving ReAct Digital Team** that intelligently decides *when to be fast* and *when to go deep*.

### Core Philosophy (First Principles)

We rejected the “use everything on every task” anti-pattern.  
Instead, we built a swarm that **knows itself** — automatically choosing the right level of intelligence for each request.

This is **the only swarm that**:
- **Intelligently routes** every task: Simple / Medium / Balanced / Complex
- **Forces visible ReAct thinking** in every single agent response
- Delivers **Balanced mode** as the default sweet spot (3 agents, planning + tools, best quality/speed ratio)
- Runs **Tree-of-Thoughts** with 3 concurrent exploration branches
- Spawns **Dynamic Agents** on-the-fly with true parallel subtask execution (Hierarchical Supervisor)
- Maintains **lifelong hierarchical memory** (PrimalMemory + Vector + Knowledge Graph) with automatic decay & distillation
- Performs **all self-evolution completely in the background** (Auto-Eval + Active Distillation + memory persistence) — **zero freezing after answer delivery**
- Generates & validates **Master Plan** with Benjamin review pass + dynamic refresh
- Supports **instant task cancellation**, streaming output, smart image compression, and tool overrun friendly fallback
- Ships production-ready: **WebUI + Feishu long-polling + Email integration + smart file delivery**

### Result

A local swarm that feels like a real team:  
**Fast when you need speed. Deep when you need excellence. Never makes you wait.**


---

### ✨ What Makes v3.2.0 Different

| Feature                        | MultiAgentSwarm v3.2.0                  | Others (AutoGen/CrewAI/LangGraph) |
|--------------------------------|-----------------------------------------|-----------------------------------|
| **Task Routing**               | ✅ Auto Simple/Medium/Balanced/Complex   | ❌ Always full swarm              |
| **ReAct Visibility**           | ✅ **Forced** Thinking/Action in every response | ❌ Opaque or optional            |
| **Balanced Mode**              | ✅ **Fixed 3 agents** (Grok+Harper+Benjamin) + Master Plan + 1 light debate | ❌ No dedicated sweet-spot mode  |
| **Adversarial Debate**         | ✅ Pro/Con/Judge + Meta-Critic (quality-based) | ❌ Manual only                   |
| **Master Plan + Benjamin Review** | ✅ Auto-generated + validated every run | ❌ None                           |
| **Memory System**              | ✅ PrimalMemory (tree+atomic+decay) + Vector + Knowledge Graph | ❌ Ephemeral                     |
| **WebUI**                      | ✅ Real-time streaming + cancel + expandable ReAct panel + file download | ❌ Requires extra work           |
| **Feishu Integration**         | ✅ Official long-connection + auto 👍     | ❌ Not supported                 |
| **Token Efficiency**           | Balanced saves 60-80% vs Complex        | ❌ Always expensive              |

---

### ✨ Key Features (v3.2.0)

**1. 🧭 Full ReAct Visualization (Core of v3.2.0)**  
- Every agent response **must** begin with:  
  `Thinking:` → `Action:` → `Action Input:`  
- Tool results clearly marked **`【Observation】`** (red highlighted)  
- Real-time WebSocket streaming → you literally watch every agent think.

**2. 📋 Dynamic Master Plan + Benjamin Validation**  
- Auto-generates structured plan  
- Benjamin (Agent[2]) performs dedicated Plan Review Pass  
- Refreshes plan when quality drops or every 3 rounds

**3. 🟠 Balanced Mode — The Sweet Spot**  
- **Fixed 3 agents**: Grok (leader) + Harper (creative) + Benjamin (critic)  
- Master Plan injection + Benjamin review + **exactly 2 rounds**  
- Light adversarial debate **only on round 2**  
- Quality ≈ Complex 80-85% at ~140% speed of Medium

**4. Intelligent Routing (v4 Classifier)**  
- Rule + LLM hybrid  
- Balanced is the default for analysis/reports/structured output  
- Complex only when truly needed (real-time tracking + deep reflection) or user forces it (中英日 keywords supported)

**5. Production WebUI & Integrations**  
- True per-agent streaming + expandable Thinking panel  
- Upload images/PDFs → auto multi-modal or text parsing  
- One-click Markdown export + smart file generation with `/uploads` download links  
- Feishu long-connection with instant 👍 reaction  
- Task cancel button + heartbeat keep-alive

**6. Memory & Knowledge Systems**  
- PrimalMemory (tree logs + atomic KB + exponential decay)  
- VectorMemory (ChromaDB)  
- Active Knowledge Graph + automatic distillation

---

### 📊 Performance (Real Measured)

| Mode       | Agents | Rounds | Debate | Quality (vs Complex) | Typical Time | Best For                  |
|------------|--------|--------|--------|----------------------|--------------|---------------------------|
| Simple     | 1      | 1      | 0      | 60%                  | 1-3s         | Greetings, quick facts    |
| Medium     | 2      | 1      | 0      | 75%                  | 8-15s        | Concept explanation       |
| **Balanced** | **3**  | **2**  | **1**  | **80-85%**           | **18-28s**   | **Analysis, reports, files** |
| Complex    | 4+     | 4-8    | 2-4    | 100%                 | 40-80s+      | Real-time deep tracking   |

**Balanced is now the recommended default for 70%+ of real tasks.**

---

### 🚀 Quick Start

```bash
git clone https://github.com/yourname/MultiAgentSwarm.git
cd MultiAgentSwarm
uv pip install -r requirements.txt
```

**Recommended: Start with OpenSandbox** (hard isolation)  

---

**WebUI (recommended)**: `python webui.py` → http://localhost:8060

---
**Via EMAIL**
- Configure email information in the configuration file.
- Send an email to the recipient's address (using the predefined subject).

---

**CLI test**: `python multi_agent_swarm_v3.py`
---


### 📋 Flexible Configuration (One YAML to Rule Them All)

```yaml
openai:
  default_model: "gpt-4o-mini"          
  context_limit_k: "128"                
  base_url: "http://localhost:11434/v1" # Ollama / vLLM / any compatible

advanced_features:
  adversarial_debate:
    enabled: true
    trigger_strategy: "quality_based"
  adaptive_reflection:
    quality_threshold: 85

intelligent_routing:
  enabled: true
  force_complexity: null                # "simple" / "medium" / "complex"
```

All features are hot-reloadable via WebUI `/api/config`.

### 🛠️ What You Can Do Today

- Beautiful WebUI with real-time streaming + cancel button + expandable ReAct panel
- Upload images/PDFs/TXT/MD → auto multi-modal or text parsing
- Talk directly to Feishu (group/private) → bot replies automatically with 👍
- Generate downloadable reports with one click
- Export any conversation as Markdown
- **25+ Skills** + instant skill extension via `/skills/`
- Persistent PrimalMemory + Vector DB across restarts
- Full local model support (Ollama/vLLM or any OpenAI-compatible endpoint)
- Email support

### 📁 Project Structure

```
MultiAgentSwarm/
├── webui.py                    # FastAPI + WebSocket + Feishu (main entry) + Email
├── multi_agent_swarm_v3.py     # Core swarm logic
├── skills/                     # 25+ ready skills + custom .py/.md
├── uploads/                    # Uploads + generated files (directly downloadable)
├── static/index.html           # Frontend
├── requirements.txt            # Full dependencies
├── swarm_config.yaml           # All settings
└── memory/                     # PrimalMemory + Vector DB + Knowledge Graph
```

### Installation & Requirements

```bash
Python 3.10+
uv pip install -r requirements.txt   # or pip
```

No GPU required. Model caching automatic. OpenSandbox optional (auto fallback).

**Star ⭐ if you like the direction — it keeps us motivated!**

## 中文版 <a id="chinese-version"></a>

**MultiAgentSwarm WebUI**  
**一个真正看得见Agent思考的自适应数字团队**

**MultiAgentSwarm v4.1.0** 不再是简单的“多个LLM并行聊天”，而是一个**完全可视化、智能决策、自进化 ReAct 数字团队** —— 它完美还原经典 ReAct 架构，同时注入群体智能、动态规划、分层终身记忆与生产级交付能力。

### 我们为什么要做这个项目（第一性原理）

市面上的多智能体框架大多采用同一种粗暴思路：  
**“不管什么任务都把所有Agent扔上去 → 大量烧token → 祈祷出好结果”**

我们从第一性原理出发，基于两个核心公理重新构建：

1. **真正的智能不是用更多Agent，而是知道什么时候该快、什么时候该深、该并行什么、该记住什么**。
2. **高质量绝不能以让用户等待为代价** —— 答案必须即时呈现，所有自进化过程必须完全后台异步完成。

**成果：MultiAgentSwarm v4.1.0** —— 目前唯一同时做到以下全部特性的本地多智能体系统：

- **智能任务路由**：自动判断并切换 Simple / Medium / Balanced / Complex 四种模式
- **Balanced 甜点模式**：默认最优选择（规划 + 工具 + 轻量反思，质量接近 Complex，速度接近 Medium）
- **Tree-of-Thoughts 多分支并行探索**：3条思考路径同时推进
- **动态Agent工厂 + Hierarchical Supervisor**：真正并行执行子任务（按需生成专家并发工作）
- **强制可见 ReAct 思考**：每个Agent都必须先输出 Thinking → Action（完全透明）
- **分层终身记忆系统**：PrimalMemory + Vector + Knowledge Graph + 自动衰退与蒸馏
- **完全异步后台自进化**：答案显示后立即可用，记忆保存、Auto-Eval、Active Distillation 全在后台运行（**零冻结**）
- **Master Plan 生成 + Benjamin 验证**：动态规划 + 严格审查 + 实时刷新
- **生产级集成**：开箱即用的 WebUI + 飞书长连接 + 邮件智能回复 + 智能文件生成与下载

**一句话总结**：  
这是一个**看得见思考、懂得节制、持续进化**的本地数字团队 —— 既能秒回简单问题，也能高质量完成复杂报告，还永远不会让你等后台处理。


---

### ✨ 与其他框架的本质区别

| 特性                     | MultiAgentSwarm v3               | AutoGen / CrewAI / LangGraph |
|--------------------------|----------------------------------|------------------------------|
| 智能路由                 | ✅ 自动简单/中等/复杂             | ❌ 永远全量群聊              |
| 对抗式辩论               | ✅ Pro/Con/Judge + 元批评         | ❌（需手动实现）             |
| Primal终身记忆           | ✅ 树状日志 + 原子KB + 衰退       | ❌ 临时记忆                  |
| 知识图谱 + 蒸馏          | ✅ 实时概念图谱                   | ❌ 无                       |
| 自适应反思深度           | ✅ 质量达标自动停止               | ❌ 固定轮数                  |
| 强制ReAct透明格式        | ✅ 每轮都显示思考过程             | ❌ 黑箱                     |
| WebUI + 流式 + 取消按钮  | ✅ 开箱即用                       | ❌ 需要额外开发              |
| 飞书企业级长连接         | ✅ 零配置                         | ❌ 不支持                   |
| 智能文件生成与下载       | ✅ 自动识别意图并生成可下载报告   | ❌ 手动                     |
| Token & 时间节省         | 简单任务节省60-80%               | ❌ 永远昂贵                 |
| 本地模型支持             | ✅ 任何兼容OpenAI接口的LLM（Ollama、vLLM、DeepSeek、Qwen等） | ❌ 支持有限或需额外配置     |

---

**核心升级**：
核心特性：
- 智能任务路由（Simple / Medium / Balanced / Complex）+ Balanced甜点模式
- Tree-of-Thoughts 多分支并行探索（3 Agent 并发）
- 动态 Agent 工厂 + Hierarchical Supervisor 真正并行子任务执行
- 分层记忆系统（PrimalMemory + Vector + Knowledge Graph）
- 完全异步后台处理（记忆保存 + Auto-Eval + Active Distillation + 无任何冻结）
- Master Plan 生成 + Benjamin 验证 Pass + 动态刷新
- 任务取消支持 + 流式输出 + 图片智能压缩 + 工具超限友好兜底
- Balanced 模式**严格锁定 3 个 Agent**（Grok + Harper + Benjamin）——质量与速度最佳平衡点
- 所有 Agent 强制输出 ReAct 三段式思考过程（实时可见）
- WebUI 完整流式 + 文件智能生成下载 + 飞书长连接 👍

### ✨ 核心特性

**1. 🧭 显式 ReAct 思考过程（架构图 100% 对齐）** 
- 每条 Agent 回复**必须**以以下格式开头：  
  `Thinking:`（原因分析）  
  `Action:`（调用工具名称或 Final Answer）  
  `Action Input:`（参数 JSON 或最终答案摘要）  
- 工具返回结果独立标记为 **【Observation】**（红色醒目）  
- WebSocket 实时流式输出，用户和开发者可完整看到思考链路。

**2. 📋 动态 Master Plan 刷新（动态规划闭环）**  
- 每 3 轮或质量 < 75 分时**自动刷新** Master Plan  
- 完美闭合架构图“更新prompt”循环  
- 所有 Agent 始终对齐最新规划。

**3. 🧭 Intelligent Routing（智能任务路由）** ★ 旗舰特性  
- 自动判断 Simple / Medium / Complex  
- 规则 + LLM 双重判断 + 自动降级

**4. 🥊 Adversarial Debate + Meta-Critic**  
- Pro / Con / Judge 三角色并行辩论，每轮强制先挑刺  
- Meta-Critic 二次综合评估

**5. 🏭 Dynamic Task Decomposition + 🧠 Active Knowledge Graph + PrimalMemory**  
- 自动拆解 4-7 个子任务并智能分配  
- 实时实体-关系提取 + 重要性蒸馏 + 树状日志 + 原子 KB + 衰退机制

**6. 📈 Adaptive Reflection Depth（自适应反思深度）**  
- 质量 ≥85 分立即停止  
- 质量收敛（Δ<3）自动停止  
- 全部参数通过 API 实时可调

**7. 🌐 美观生产级 WebUI（v3.2.0 增强版）** ★ **全新**  
- 真实逐 Agent WebSocket 流式输出 + 可展开「🤔 思考过程」面板  
- Master Plan 动态刷新实时日志可见  
- 多会话管理 + 一键导出 Markdown  
- 文件上传（PDF/图片/文本，最大10MB）**+ 中文文件名自动净化**  
- `/uploads` 静态挂载 → **修改后的报告/Excel/PDF 可直接点击下载**  
- 任务取消按钮 + 30秒心跳保活  
- 完整**飞书官方 SDK 长连接** + 收到消息立即自动👍反应
- 邮件交互

**8. 🔒 OpenSandbox 双模式代码执行器**  
- 已安装 → Docker 硬隔离（推荐）  
- 未安装 → 自动回退 + 醒目安装提示

**9. 🌍 本地模型支持 + 25+ 开箱即用 Skill**  
- 任何兼容 OpenAI 接口的 LLM（Ollama、vLLM、DeepSeek、Qwen 等）  
- **25+ Skill** 开箱即用 + **极易扩展**：把 `.py` 或 `.md` 扔进 `/skills/` 即可立即加载

### 📊 性能对比

| 指标                  | v2.9.2     | v3.1.0      | v3.2.0（现在）          | 提升幅度       |
|-----------------------|------------|-------------|-------------------------|----------------|
| 简单任务耗时          | 8-12s      | 1-3s        | **1-3s**                | -75%          |
| 复杂任务质量          | 8.0/10     | 9.5/10      | **9.7/10**              | +21%          |
| 思考过程透明度        | 无         | 部分        | **完整实时可见**        | 革命性提升    |
| 规划漂移（5+轮）      | 中         | 低          | **几乎为 0**            | 彻底解决      |
| Token 消耗            | 基准       | -40~60%     | **-45~65%**             | 进一步节省    |
| 文件处理              | 无         | 基础        | **中文名净化 + 下载支持** | 生产级可用    |

### 🚀 快速开始

```bash
git clone https://github.com/yourname/MultiAgentSwarm.git
cd MultiAgentSwarm
uv pip install -r requirements.txt
```

**启动 OpenSandbox（推荐）**  
```bash
opensandbox-server init-config ~/.sandbox.toml --example docker
opensandbox-server
```

**启动 WebUI**  
```bash
python webui.py
```
访问 → **http://localhost:8060**

**通过 EMAIL**  
- 在配置文件中配置邮箱信息
- 发送邮件到收信邮件地址（使用设定好的标题）


**CLI 测试**  
```bash
python multi_agent_swarm_v3.py
```

---

**享受构建属于你自己的完全透明数字团队吧！** 🚀

**最后更新**：2026 年 3 月 3 日  
**版本**：v3.2.0
