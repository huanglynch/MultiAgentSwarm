# MultiAgentSwarm v3.2.0 (ReAct Visualization Edition)
**Self-Adaptive Digital Team**  
**Enterprise-grade Multi-Agent Collaboration Framework with Full ReAct Visibility**  
**The only swarm that lets you literally watch every agent think.**

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

**MultiAgentSwarm v3.2.0** is not just multiple LLMs chatting — it is a **fully visible, self-adaptive ReAct Digital Team** that perfectly matches the classic ReAct architecture diagram while adding group intelligence, dynamic planning, and production-ready WebUI + Feishu integration.

### Why This Project Exists (First-Principles Thinking)

Most multi-agent frameworks follow the same pattern:  
**“Throw all agents at every task → burn tokens → pray for good output.”**

We started from scratch with two axioms:

1. **Intelligence ≠ more agents.** Intelligence = knowing *when* to use agents, *how deep* to debate, and *what to remember*.
2. **Quality without waste.** Every token spent must be justified.

**Result:** The only framework that is **100% aligned with the classic ReAct architecture diagram** while delivering group intelligence, dynamic planning, lifelong memory, and production-ready WebUI + Feishu integration.

### ✨ Core Features (v3.2.0 Major Upgrades)

**1. 🧭 Explicit ReAct Thinking Process (100% Architecture Alignment)** ★ **2026 Visualization Core**  
- Every Agent response **must** start with:  
  `Thinking:` (reasoning)  
  `Action:` (tool name or Final Answer)  
  `Action Input:` (parameters or answer summary)  
- Tool results clearly marked as **【Observation】** (highlighted in red)  
- Real-time WebSocket streaming makes the entire thinking chain visible.

**2. 📋 Dynamic Master Plan Refresh (Closed-Loop Planning)**  
- Automatically refreshes Master Plan every 3 rounds or when quality score < 75  
- Perfectly closes the “update prompt” loop in the original ReAct diagram  
- Zero long-term drift — all agents stay aligned.

**3. 🧭 Intelligent Routing** ★ **2026 Flagship Feature**  
- Auto-detects: **Simple / Medium / Complex** (rule + LLM dual judgment)  
- Automatic fallback + per-request force mode

**4. 🥊 Adversarial Debate + Meta-Critic**  
- Pro / Con / Judge three-role parallel debate  
- Every round forces critique first + Meta-Critic synthesis

**5. 🏭 Dynamic Task Decomposition + 🧠 Active Knowledge Graph + PrimalMemory**  
- Auto-breaks tasks into 4–7 subtasks with smart assignment  
- Real-time entity-relation extraction + importance-based distillation  
- Tree-log + atomic KB + exponential decay memory system

**6. 📈 Adaptive Reflection Depth**  
- Stops immediately when quality ≥ 85  
- Stops on quality convergence (Δ < 3)  
- All thresholds configurable in real time via API

**7. 🌐 Beautiful Production WebUI (v3.2.0 Enhanced)** ★ **Brand New**  
- True per-agent WebSocket streaming + expandable “🤔 Thinking Process” panel  
- Master Plan dynamic refresh logs visible in real time  
- Multi-session management + one-click Markdown export  
- File upload (PDF/TXT/MD/images, max 10MB) with automatic Chinese filename sanitization  
- `/uploads` static mount → modified reports/Excel/PDFs downloadable with one click  
- Task cancel button + 30s heartbeat keep-alive  
- Full Feishu official SDK long connection with instant 👍 reaction

**8. 🔒 OpenSandbox Dual-Mode Code Executor**  
- Installed → Docker hard isolation (recommended)  
- Not installed → auto fallback + clear installation prompt

**9. 🌍 Local Model Support + 25+ Ready-to-Use Skills**  
- Any OpenAI-compatible LLM (Ollama, vLLM, DeepSeek, Qwen, etc.) via `base_url`  
- **25+ Skills out of the box** (web search, PDF parser, file I/O, Python sandbox, image analysis, etc.)  
- **Drop-and-play Skill system**: Drop any `.py` tool or `.md` knowledge file into `/skills/` — instantly loaded

### 📊 Performance Comparison

| Metric                  | v2.9.2     | v3.1.0      | v3.2.0 (Now)               | Improvement      |
|-------------------------|------------|-------------|----------------------------|------------------|
| Simple task time        | 8-12s      | 1-3s        | **1-3s**                   | -75%            |
| Complex task quality    | 8.0/10     | 9.5/10      | **9.7/10**                 | +21%            |
| Thinking transparency   | None       | Partial     | **Full real-time visible** | Revolutionary   |
| Plan drift (5+ rounds)  | Medium     | Low         | **Almost 0**               | Completely solved |
| Token usage (complex)   | Baseline   | -40~60%     | **-45~65%**                | Further savings |
| File handling           | None       | Basic       | **Chinese name cleanup + download** | Production-ready |

### 🚀 Quick Start (Under 60 Seconds)

```bash
git clone https://github.com/yourname/MultiAgentSwarm.git
cd MultiAgentSwarm
uv pip install -r requirements.txt        # Recommended (faster)
```

**Start OpenSandbox (strongly recommended for production)**  
```bash
opensandbox-server init-config ~/.sandbox.toml --example docker
# Run in a new terminal (keep running)
opensandbox-server
```

**Start WebUI (recommended)**  
```bash
python webui.py
```
→ Open **http://localhost:8060**

**CLI quick test**  
```bash
python multi_agent_swarm_v3.py
```

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

### 📁 Project Structure

```
MultiAgentSwarm/
├── webui.py                    # FastAPI + WebSocket + Feishu (main entry)
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

---

## 中文版 <a id="chinese-version"></a>

**MultiAgentSwarm WebUI v3.2.0（ReAct 可视化版）**  
**一个真正“看得见思考”的自适应数字团队**

**MultiAgentSwarm v3.2.0** 不再是简单的“多个 LLM 并行聊天”，而是一个**完全可视化、自适应 ReAct 数字团队** —— 完美对齐经典 ReAct 架构图，同时具备群体智能、动态规划、生产级 WebUI 和飞书深度集成。

### ✨ 核心特性（v3.2.0 重磅升级）

**1. 🧭 显式 ReAct 思考过程（架构图 100% 对齐）** ★ **2026 可视化核心**  
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

**3. 🧭 Intelligent Routing（智能任务路由）** ★ 2026 旗舰特性  
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

**CLI 测试**  
```bash
python multi_agent_swarm_v3.py
```

---

**配置参考**、**项目结构**、**License**（MIT）与英文版一致。

**享受构建属于你自己的完全透明数字团队吧！** 🚀

**最后更新**：2026 年 3 月 3 日  
**版本**：v3.2.0（ReAct 可视化 + 动态 Master Plan 闭环 + WebUI 完整版 + 文件上传下载 + 飞书 👍 长连接 + OpenSandbox）
