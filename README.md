# 🚀 TrendMaster-MCP: The Quant 4.0 Architecture
**Web3 + AI Agent 量化交易底座 | 基于 Model Context Protocol 的微服务架构**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Ready](https://img.shields.io/badge/Model_Context_Protocol-Ready-green.svg)](https://anthropic.com/mcp)

[English](#english-version) | [中文说明](#中文说明)

---

## 🌟 中文说明

**TrendMaster-MCP** 是一个专为 Web3 与 Crypto 市场设计的下一代量化交易系统底座。它彻底抛弃了传统量化中僵化的 `if-else` 策略流，全面拥抱 **Quant 4.0 理念**：将大语言模型（LLM）作为中枢大脑，通过标准的 [Model Context Protocol (MCP)](https://anthropic.com/mcp) 动态拉取（Pull）独立运行的底层算力微服务。

### 🔥 核心突破：解决 LLM 交易的两大死穴

传统 AI 交易系统往往面临两大痛点：**上下文核爆（Token 浪费）** 与 **串行推理延迟（错过行情）**。TrendMaster-MCP 提出了终极解法：

1. **高维特征数据脱水 (Data Dehydration)**：Agent 绝不直接读取几千行的原始 K 线或盘口数组。底层的 `Indicator-MCP` 使用 Pandas/NumPy 矩阵运算，瞬间将海量数据转化为极简的脱水特征（如 HMM 状态、SMC 结构、OFI 订单流）。
2. **宏观聚合工具 (Fat-Tool 架构)**：将原本需要 Agent 串行调用 4 次的 API 往返（耗时数分钟），压缩至物理底层的单次并发拉取。**决策延迟从 10 分钟级直接坍缩至 2 秒级！**

### 🏗️ 架构蓝图 (Architecture)

系统被严格物理隔离为四层：

- 🧠 **Agent Client (中枢与记忆层)**：基于 ReAct 架构，注入 `Memory Stream`（记忆流），支持跨周期的连贯思考。严格遵守 `Strategy-SOP.skill` 铁律，实现“变盘深度重估，震荡极速风控”。
- 👁️ **Market-MCP (感知层)**：只负责高频、稳定地对接 CCXT，搬运并清洗 K 线与 Orderbook 深度数据。
- ⚙️ **Indicator-MCP (认知层)**：系统的算力黑盒。内建：
  - `HMM Engine`: 隐马尔可夫模型，精准识别市场波动率与牛熊震荡状态。
  - `SMC Engine`: 聪明钱概念，向量化识别 FVG（公允价值缺口）与 BOS（结构突破）。
  - `Orderflow Engine`: 订单流引擎，结合 OFI 与主动成交（Aggressor Trades）洞察主力资金真实意图。
  - `Kronos Forecast`: 本地时间序列预测入口，启动阶段预热本地模型；不可用时自动降级为启发式预测，并通过 Fat-Tool 注入 `kronos_forecast` 与共振风控。
- 🛡️ **Execution-MCP (执行层)**：绝对安全的物理沙盒。内建硬编码风控，若大模型指令触发最大滑点或仓位上限，拥有“一票否决权”。

### 📂 目录结构 (Directory Structure)

```text
TrendMaster-MCP/
├── agent_client.py           # Agent 中枢入口 (记忆流 + ReAct 循环)
├── agent_memory.json         # Agent 本地记忆数据库 (自动生成)
├── shared/                   # 跨微服务共享模块
│   ├── models.py             # 统一数据契约 (Pydantic 标准化输出)
│   └── logger.py             # 彩色日志与日志隔离引擎
├── skills/                   # Agent 认知法则层
│   └── Strategy-SOP.skill    # 严格的交易纪律与状态响应手册
├── servers/                  # MCP 微服务层
│   ├── market-mcp/           # 行情感知服务
│   ├── indicator-mcp/        # 指标算力服务 (HMM/SMC/Orderflow)
│   └── execution-mcp/        # 沙盒执行服务 (带硬核风控)
└── .env.example              # 环境变量与安全配置模板

🚀 快速启动 (Quick Start)
1. 环境准备:
git clone [https://github.com/yourusername/TrendMaster-MCP.git](https://github.com/yourusername/TrendMaster-MCP.git)
cd TrendMaster-MCP
pip install -r requirements.txt

2. 配置秘钥:
复制 .env.example 为 .env，填入你的 LLM API Key (支持 DeepSeek/Claude/OpenAI) 及交易所 API Key。
如需启用本地 Kronos，补齐 `KRONOS_LOCAL_REPO_PATH`、`KRONOS_MODEL_ID`、`KRONOS_TOKENIZER_ID` 与 `KRONOS_DEVICE`；配置不完整时系统会保守降级，不影响 Indicator-MCP 启动。

3. 启动系统:
python agent_client.py


🌟 English Version
TrendMaster-MCP is a next-generation quantitative trading foundational layer designed specifically for the Web3 and Crypto markets. It completely abandons the rigid if-else strategy flows of classical quant, fully embracing the Quant 4.0 philosophy: using Large Language Models (LLMs) as the central brain to dynamically pull independent, high-performance microservices via the standard Model Context Protocol (MCP).

🔥 Core Breakthroughs
It solves the two fatal flaws of traditional AI trading bots: Context Window Explosion and Serial Reasoning Latency.
Data Dehydration: The Agent never reads raw OHLCV arrays. The underlying Indicator-MCP uses Pandas/NumPy matrix operations to instantly transform massive data into concise, dehydrated features (e.g., HMM Regimes, SMC Structures, OFI).
Fat-Tool Architecture: It compresses what used to be 4 serial API roundtrips (taking minutes) into a single concurrent physical pull. Decision latency collapses from 10 minutes to under 2 seconds!
(For detailed architecture and setup, please refer to the Chinese documentation above or explore the codebase.)
