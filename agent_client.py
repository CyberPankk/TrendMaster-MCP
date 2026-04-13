import asyncio
import os
import json
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from anthropic import AsyncAnthropic
from anthropic.types.message_param import MessageParam
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from colorama import Fore, Style, init

from shared.audit_feedback import format_order_lessons_context, get_recent_order_lessons
from shared.data_fetcher import fetch_fear_and_greed, fetch_funding_rate
from shared.logger import get_logger
from openai_client import DeepSeekAdapter # 导入适配器

# 初始化 Colorama
init(autoreset=True)

# 加载环境变量
load_dotenv(override=True)

logger = get_logger("Agent-Client")

DEFAULT_SYSTEM_PROMPT = """你是 TrendMaster PRO 的量化交易决策中枢，负责对加密货币市场进行审慎、可执行、可审计的分析。

你的核心职责：
1. 优先保护资金安全，任何信号必须先经过风险审查，再讨论收益空间。
2. 必须综合技术面、资金面、情绪面和事件风险，避免只凭单一指标下结论。
3. 当信息不完整、指标冲突、波动异常或存在重大事件风险时，默认输出 WAIT。
4. 严禁臆测未提供的数据，严禁伪造价格、成交量、仓位、资金费率或新闻事件。
5. 若市场结构与历史记忆冲突，应以最新数据为准，并说明冲突原因。

你的分析顺序：
1. 先识别趋势、波动、关键支撑阻力和市场结构。
2. 再评估多维因子，包括恐慌贪婪、资金费率、多空比以及其他注入的上下文。
3. 最后输出交易动作、核心逻辑、风险提示和无效条件。

你的强制约束：
1. 只允许基于输入中提供的数据进行推理，禁止假设已经调用了不存在的工具。
2. 如果多维因子显示市场过热、拥挤或事件风险升高，必须下调激进程度。
3. 如果出现高不确定性，必须明确写出为什么 WAIT 比交易更优。
4. 不要输出模糊建议，必须给出清晰动作标签。

你的输出必须包含以下 XML 标签：
<ACTION>BUY|SELL|WAIT</ACTION>
<REASONING>简洁说明技术面、多维因子与风险约束如何共同支持该决策</REASONING>
<RISK>说明主要风险、失效条件或不确定性来源</RISK>

请始终使用专业、克制、面向执行的语气。"""

class AgentMemory:
    def __init__(self, filepath="agent_memory.json"):
        self.filepath = filepath
        self.memory = self._load_db()

    def _load_db(self):
        """加载本地 JSON 记忆库"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"读取记忆库失败: {e}")
        return {}

    def _save_db(self):
        """持久化保存到 JSON"""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"写入记忆库失败: {e}")

    def get_last_memory(self, symbol: str) -> str:
        """格式化输出上一次的记忆"""
        mem = self.memory.get(symbol)
        if not mem:
            return "无历史记忆。"
        
        return (f"【历史记忆流】你上一次在 {mem['timestamp']} 对 {symbol} 的决策是：\n"
                f"操作: {mem['last_action']}\n"
                f"逻辑: {mem['reasoning']}")

    def update_memory(self, symbol: str, ai_response: str):
        """解析 AI 的回复并更新记忆"""
        # 升级版正则：强制匹配 XML 标签提取核心指令
        action_match = re.search(r'<ACTION>([A-Z]+)</ACTION>', ai_response, re.IGNORECASE)
        action = action_match.group(1).upper() if action_match else "UNKNOWN"

        self.memory[symbol] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_action": action,
            # 截取 AI 回复的前 200 个字符作为核心逻辑摘要，防止记忆无限膨胀
            "reasoning": ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
        }
        self._save_db()
        logger.info(f"💾 {symbol} 的决策记忆已归档 (Action: {action})。")

class TrendMasterAgent:
    def __init__(self):
        self.llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        logger.info(f"🚀 当前使用的 LLM 提供商: {Fore.GREEN}{self.llm_provider.upper()}{Style.RESET_ALL}")

        if self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("未找到 ANTHROPIC_API_KEY，请检查 .env 文件")
                sys.exit(1)
            self.anthropic = AsyncAnthropic(api_key=api_key)
        elif self.llm_provider in ["deepseek", "ofox"]:
            # 使用 OpenAI 兼容适配器
            self.deepseek = DeepSeekAdapter(provider=self.llm_provider)
        else:
            logger.error(f"不支持的 LLM_PROVIDER: {self.llm_provider}")
            sys.exit(1)
            
        # 记录 tool_name 对应哪个 MCP Session
        self.tool_routing_map: Dict[str, ClientSession] = {} 
        self.available_tools: List[Dict[str, Any]] = []
        self.memory_stream = AgentMemory() # 实例化记忆库
        
        # ⚠️ 彻底抛弃 AsyncExitStack，改为纯手工生命周期管理
        # 这也是 mcp 官方文档中针对多客户端管理的推荐范式
        self.sessions: List[ClientSession] = []
        self.transports: List[Any] = []
        
        # 我们的 3 个微服务路径 (这里使用 python 命令启动)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        self.system_prompt_path = os.path.join(self.repo_root, "data", "system_prompt.txt")
        self.feed_config_path = os.path.join(self.repo_root, "data", "feed_config.json")
        self.ensure_system_prompt_file()
        self.ensure_feed_config_file()
        self.mcp_servers = {
            "Market": os.path.join(base_dir, "servers/market_mcp/server.py"),
            "Indicator": os.path.join(base_dir, "servers/indicator_mcp/server.py"),
            "Execution": os.path.join(base_dir, "servers/execution_mcp/server.py"),
            "Sentiment": os.path.join(base_dir, "servers/sentiment_mcp/server.py"),
            "FactorLab": os.path.join(base_dir, "servers/factor_lab_mcp/server.py"),
            "Strategy": os.path.join(base_dir, "servers/strategy_mcp/server.py")
        }

    def ensure_system_prompt_file(self) -> str:
        """确保动态 Prompt 文件存在且可读，首次启动时自动写入默认模板。"""
        os.makedirs(os.path.dirname(self.system_prompt_path), exist_ok=True)
        if os.path.exists(self.system_prompt_path):
            try:
                with open(self.system_prompt_path, "r", encoding="utf-8") as file:
                    if file.read().strip():
                        return self.system_prompt_path
            except Exception as exc:
                logger.warning(f"读取现有 system_prompt.txt 失败，将重建默认模板: {exc}")

        with open(self.system_prompt_path, "w", encoding="utf-8") as file:
            file.write(DEFAULT_SYSTEM_PROMPT)
        logger.info(f"✅ 已初始化动态 system_prompt.txt: {self.system_prompt_path}")
        return self.system_prompt_path

    def ensure_feed_config_file(self) -> str:
        os.makedirs(os.path.dirname(self.feed_config_path), exist_ok=True)
        default_config = {
            "fear_greed": True,
            "funding_rate": False,
            "long_short": False,
        }
        if os.path.exists(self.feed_config_path):
            try:
                with open(self.feed_config_path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
                if isinstance(loaded, dict):
                    normalized = {
                        "fear_greed": bool(loaded.get("fear_greed", default_config["fear_greed"])),
                        "funding_rate": bool(loaded.get("funding_rate", default_config["funding_rate"])),
                        "long_short": bool(loaded.get("long_short", default_config["long_short"])),
                    }
                    with open(self.feed_config_path, "w", encoding="utf-8") as file:
                        json.dump(normalized, file, indent=2, ensure_ascii=False)
                    return self.feed_config_path
            except Exception as exc:
                logger.warning(f"读取现有 feed_config.json 失败，将重建默认配置: {exc}")

        with open(self.feed_config_path, "w", encoding="utf-8") as file:
            json.dump(default_config, file, indent=2, ensure_ascii=False)
        logger.info(f"✅ 已初始化 feed_config.json: {self.feed_config_path}")
        return self.feed_config_path

    def read_feed_config(self) -> Dict[str, bool]:
        self.ensure_feed_config_file()
        try:
            with open(self.feed_config_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            return {
                "fear_greed": bool(loaded.get("fear_greed", True)),
                "funding_rate": bool(loaded.get("funding_rate", False)),
                "long_short": bool(loaded.get("long_short", False)),
            }
        except Exception as exc:
            logger.warning(f"读取 feed_config.json 失败，回退默认配置: {exc}")
            return {
                "fear_greed": True,
                "funding_rate": False,
                "long_short": False,
            }

    def read_latest_system_prompt(self) -> str:
        """读取仓库中的最新系统提示词，若不存在则回退到基础 SOP。"""
        try:
            self.ensure_system_prompt_file()
            if os.path.exists(self.system_prompt_path):
                with open(self.system_prompt_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()
                if content:
                    logger.info(f"✅ 已加载最新 system_prompt.txt: {self.system_prompt_path}")
                    return content
                logger.warning("⚠️ system_prompt.txt 为空，回退到默认模板")
        except Exception as exc:
            logger.error(f"读取最新 system prompt 失败，回退至默认模板: {exc}")

        return DEFAULT_SYSTEM_PROMPT

    def write_system_prompt(self, content: str) -> str:
        """持久化写入 system_prompt.txt，供 API 与 Agent 热更新共用。"""
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("system_prompt.txt 内容不能为空")

        os.makedirs(os.path.dirname(self.system_prompt_path), exist_ok=True)
        with open(self.system_prompt_path, "w", encoding="utf-8") as file:
            file.write(normalized_content)

        logger.info(f"✅ system_prompt.txt 热更新完成: {self.system_prompt_path}")
        return self.system_prompt_path

    def build_runtime_system_prompt(self) -> str:
        """组装运行时系统提示词，确保每轮都读取最新 Prompt。"""
        return "\n\n".join(
            [
                self.read_latest_system_prompt(),
                self.load_skill_sop(),
                self.load_skill("Trend-Sniper"),
                self.load_skill("Mean-Reversion"),
                self.load_skill("Portfolio-Risk"),
            ]
        )

    async def build_multifactor_payload(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """按配置构建真实多维因子载荷，禁止注入假数据。"""
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feed_config = self.read_feed_config()
        factor_payload: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": generated_at,
            "feed_config": feed_config,
            "factor_lines": [],
        }

        async_tasks: List[tuple[str, asyncio.Future]] = []
        if feed_config.get("fear_greed"):
            async_tasks.append(("fear_greed", asyncio.create_task(fetch_fear_and_greed())))
        if feed_config.get("funding_rate"):
            async_tasks.append(("funding_rate", asyncio.create_task(fetch_funding_rate(symbol))))

        if async_tasks:
            results = await asyncio.gather(*(task for _, task in async_tasks), return_exceptions=True)
            for (factor_name, _), result in zip(async_tasks, results):
                if isinstance(result, Exception) or result is None:
                    logger.warning(f"⚠️ 真实因子 {factor_name} 拉取失败，已跳过该维度")
                    continue

                if factor_name == "fear_greed":
                    factor_payload["fear_greed_index"] = result
                    factor_payload["factor_lines"].append(
                        f"当前市场恐慌贪婪指数为 {result['value']} ({result['classification']})。"
                    )
                elif factor_name == "funding_rate":
                    factor_payload["funding_rate"] = result
                    factor_payload["factor_lines"].append(
                        f"当前 {result['symbol']} 资金费率为 {result['formatted']}。"
                    )

        factor_payload["enabled_factors"] = [
            key for key, enabled in feed_config.items() if enabled
        ]

        return factor_payload

    def format_multifactor_prompt(self, factor_payload: Dict[str, Any]) -> str:
        """将多维因子载荷格式化为稳定的 Prompt 注入块。"""
        factor_lines = list(factor_payload.get("factor_lines", []))
        enabled_factors = factor_payload.get("enabled_factors", [])

        prompt_lines = [
            "【多维因子上下文】",
            f"- 生成时间: {factor_payload.get('generated_at', 'N/A')}",
        ]

        if factor_lines:
            prompt_lines.extend(f"- {line}" for line in factor_lines)
        elif enabled_factors:
            prompt_lines.append("- 已启用真实因子，但本轮拉取失败，禁止以假数据代替。")
        else:
            prompt_lines.append("- 当前未启用额外真实因子开关。")

        return "\n".join(prompt_lines)

    async def build_experience_context(self, symbol: str, limit: int = 5) -> str:
        """读取最近实战教训并格式化为尾部唤醒块，避免经验池与 Skill 混写。"""
        lessons = await get_recent_order_lessons(limit=limit, symbol=symbol)
        return format_order_lessons_context(lessons)

    async def build_decision_context_prefix(self, symbol: str, limit: int = 5) -> str:
        """只返回短期记忆前缀，确保长期经验池仅注入到 user_message 尾部。"""
        del limit
        return self.memory_stream.get_last_memory(symbol)

    def load_skill_sop(self) -> str:
        """加载 Agent 的核心交易纪律"""
        try:
            local_path = os.path.join(self.base_dir, "skills", "Strategy-SOP.skill")
            root_path = os.path.join(self.repo_root, "skills", "Strategy-SOP.skill")
            path = local_path if os.path.exists(local_path) else root_path
            with open(path, "r", encoding="utf-8") as f:
                sop = f.read()
                logger.info("✅ 成功加载 Strategy-SOP.skill")
                return sop
        except Exception as e:
            logger.error(f"无法读取 SOP 规则: {e}")
            return "You are a helpful trading assistant."

    def load_skill(self, skill_name: str) -> str:
        """
        按策略名加载技能书（.skill），例如 Trend-Sniper -> skills/Trend-Sniper.skill
        文件不存在时回退到基础安全提示词。
        """
        safe_fallback = (
            "You are a cautious trading assistant. "
            "If information is insufficient or risk is high, always output <ACTION>WAIT</ACTION>."
        )
        if not skill_name:
            return safe_fallback
        try:
            filename = f"{skill_name}.skill"
            local_path = os.path.join(self.base_dir, "skills", filename)
            root_path = os.path.join(self.repo_root, "skills", filename)
            path = local_path if os.path.exists(local_path) else root_path
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"无法读取技能书 {skill_name}: {e}")
            return safe_fallback

    async def init_mcp_connections(self):
        """同时启动并连接多个 MCP Server，手动管理生命周期"""
        logger.info(f"{Fore.CYAN}正在唤醒 MCP 微服务集群...{Style.RESET_ALL}")
        
        for name, script_path in self.mcp_servers.items():
            if not os.path.exists(script_path):
                logger.error(f"找不到 {script_path}，请检查路径！")
                continue
                
            server_params = StdioServerParameters(
                command=sys.executable, # 使用当前虚拟环境的 python
                args=[script_path],
                # 必须继承当前环境变量，否则子进程拿不到 API Keys
                env=os.environ.copy() 
            )
            
            try:
                # 1. 启动并管理 stdio_client
                # ⚠️ 修复：直接获取 stdio_client 的 context manager，并在其中包装 session
                # 这样可以确保 anyio 的 task group 不会在多个 task 之间乱窜
                stdio_cm = stdio_client(server_params)
                read, write = await stdio_cm.__aenter__()
                self.transports.append(stdio_cm) # 记录下来以便 cleanup 时 __aexit__
                
                # 2. 启动并管理 ClientSession
                session = ClientSession(read, write)
                await session.__aenter__()
                self.sessions.append(session)
                
                # 3. 初始化连接
                await session.initialize()
                
                # 4. 获取该 Server 的 Tools 并注册到路由表
                tools_response = await session.list_tools()
                
                for tool in tools_response.tools:
                    self.tool_routing_map[tool.name] = session
                    
                    # 将 MCP Tool 格式转换为 Anthropic JSON Schema 格式
                    anthropic_tool = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                    self.available_tools.append(anthropic_tool)
                    logger.info(f"  🔧 注册 Tool: {Fore.YELLOW}{tool.name}{Style.RESET_ALL} -> [{name}-MCP]")
                    
                # 手动注入可能由于网络原因未在启动时声明，但服务端已包含的 Tool (容错机制)
                if name == "Market" and "get_ticker" not in self.tool_routing_map:
                    self.tool_routing_map["get_ticker"] = session
                    logger.info(f"  🔧 手动补充注册 Tool: {Fore.YELLOW}get_ticker{Style.RESET_ALL} -> [{name}-MCP]")
                    
                logger.info(f"{Fore.GREEN}✅ {name}-MCP 准备就绪{Style.RESET_ALL}")
                
            except Exception as e:
                logger.error(f"连接 {name}-MCP 失败: {e}")

        logger.info(f"共加载了 {len(self.available_tools)} 个可用工具。")

    async def call_mcp_tool_directly(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """
        绕过 LLM，直接调用 MCP Tool
        Args:
            server_name: 仅用于日志记录，实际路由通过 tool_name 查找
            tool_name: MCP 工具名称
            arguments: 工具参数字典
        """
        if tool_name not in self.tool_routing_map:
            error_msg = f"未找到工具 {tool_name} 的路由映射"
            logger.error(error_msg)
            return error_msg

        session = self.tool_routing_map[tool_name]
        try:
            logger.info(f"⚡ [{server_name}] 直接调用工具: {tool_name} args={arguments}")
            # 设置 15s 超时防止卡死
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=15.0
            )
            # result.content 是一个列表，通常第一个元素包含文本
            text = result.content[0].text
            try:
                return json.loads(text)
            except Exception:
                return text
        except Exception as e:
            logger.error(f"直接调用 {tool_name} 失败: {e}")
            return f"调用失败: {str(e)}"

    async def run_chat_loop(self):
        """启动 Agent 交互主循环"""
        try:
            # 启动所有连接，如果失败会抛出异常
            await self.init_mcp_connections()
            
            # OpenAI 格式需要将 system prompt 放入 messages 列表
            messages: List[Dict[str, str]] = []
                
            logger.info("\n" + "="*60)
            logger.info(f"{Fore.MAGENTA}🤖 TrendMaster 核心中枢已上线 (Powered by {self.llm_provider.upper()} & MCP){Style.RESET_ALL}")
            logger.info(f"{Fore.MAGENTA}🧠 记忆流模块已就绪 (agent_memory.json){Style.RESET_ALL}")
            logger.info("="*60 + "\n")

            # Anthropic messages 列表 (不包含 system)
            anthropic_messages: List[MessageParam] = []
            
            while True:
                user_input = await asyncio.to_thread(input, f"\n{Fore.BLUE}🧑‍💻 长官，请下达指令 (输入 'exit' 退出): {Style.RESET_ALL}")
                if user_input.lower() in ['exit', 'quit']:
                    self.memory_stream._save_db() # 退出前强制保存一次记忆
                    break
                if not user_input.strip():
                    continue

                system_prompt = self.build_runtime_system_prompt()
                if self.llm_provider in ["deepseek", "ofox"]:
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = system_prompt
                    else:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                
                # 智能提取 Symbol (假设用户输入中包含 BTC 或 ETH 等)
                symbol_to_analyze = "BTC/USDT" # 默认 fallback
                if "ETH" in user_input.upper():
                    symbol_to_analyze = "ETH/USDT"
                elif "BTC" in user_input.upper():
                    symbol_to_analyze = "BTC/USDT"
                    
                # ==========================================
                # 🚀 优化点：数据前置注入 (绕过 LLM 的工具选择)
                # ==========================================
                logger.info(f"{Fore.YELLOW}⚡ 绕过 LLM 路由，客户端直接并发请求 Fat-Tool 数据...{Style.RESET_ALL}")
                
                # 直接通过 session.call_tool 获取脱水数据
                fat_data = "获取失败"
                if "get_full_market_context" in self.tool_routing_map:
                    session = self.tool_routing_map["get_full_market_context"]
                    try:
                        # 显式重置超时时间，防止底层MCP调用被挂起
                        # 这里使用了 asyncio.wait_for 来防止 session.call_tool 永久卡死
                        result = await asyncio.wait_for(
                            session.call_tool("get_full_market_context", arguments={"symbol": symbol_to_analyze, "timeframe": "1h"}),
                            timeout=15.0
                        )
                        fat_data = result.content[0].text
                        logger.info(f"{Fore.GREEN}✅ 极速脱水技术数据已获取{Style.RESET_ALL}")
                    except Exception as e:
                        logger.error(f"直接调用 Fat-Tool 失败: {e}")
                        fat_data = f"调用失败: {e}"
                else:
                    logger.error("未找到 get_full_market_context 工具的路由映射")
                    
                # [Phase 32] 补充获取多维宏观与资金面因子
                logger.info(f"{Fore.YELLOW}🌍 正在构建多维因子上下文...{Style.RESET_ALL}")
                factor_payload = await self.build_multifactor_payload(symbol_to_analyze, "1h")
                macro_factors = self.format_multifactor_prompt(factor_payload)
                logger.info(f"{Fore.GREEN}✅ 宏观与资金面因子已就绪，准备组装终极 Prompt{Style.RESET_ALL}")
                    
                # 记忆流与经验池严格分离：短期记忆放前缀，长期经验池放 user_message 尾部
                memory_context = await self.build_decision_context_prefix(symbol_to_analyze)
                lesson_context = await self.build_experience_context(symbol_to_analyze, limit=5)
                
                # 组装终极 Prompt
                enriched_input = (
                    f"{memory_context}\n\n"
                    f"【系统强制注入的底层技术数据】\n{fat_data}\n\n"
                    f"{macro_factors}\n\n"
                    f"当前长官指令：{user_input}\n"
                    f"请直接综合上述【技术面】与【资金面】数据输出最终交易决策，严禁调用任何外部工具！\n\n"
                    f"{lesson_context}"
                )
                    
                # 统一添加 User Message
                if self.llm_provider in ["deepseek", "ofox"]:
                    messages.append({"role": "user", "content": enriched_input})
                else:
                    anthropic_messages.append({"role": "user", "content": enriched_input})
                
                # ---------------------------------------------------------
                # ReAct 核心循环 (Reasoning + Acting)
                # ---------------------------------------------------------
                # 暂存大模型回复，用于写入记忆
                final_reply_content = ""

                logger.info(f"{Fore.CYAN}🧠 Agent 正在进行 One-Shot 极速推理...{Style.RESET_ALL}")
                print(f"\n{Fore.MAGENTA}🤖 TrendMaster: {Style.RESET_ALL}", end="", flush=True)
                
                try:
                    if self.llm_provider == "anthropic":
                        # Anthropic 流式调用逻辑
                        response_stream = await self.anthropic.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=4096,
                            system=system_prompt,
                            messages=anthropic_messages,
                            stream=True
                        )
                        
                        async for event in response_stream:
                            if event.type == "text_delta":
                                text = event.delta.text
                                print(text, end="", flush=True)
                                final_reply_content += text
                                
                        anthropic_messages.append({"role": "assistant", "content": final_reply_content})
                        
                    elif self.llm_provider in ["deepseek", "ofox"]:
                        # OpenAI 兼容流式调用逻辑
                        response_stream = await self.deepseek.chat_completion(
                            messages=messages
                        )
                        
                        async for chunk in response_stream:
                            if chunk.choices and chunk.choices[0].delta.content:
                                text = chunk.choices[0].delta.content
                                print(text, end="", flush=True)
                                final_reply_content += text
                                
                        messages.append({"role": "assistant", "content": final_reply_content})
                        
                    print("\n") # 换行收尾

                    # -----------------------------------------------------
                    # 🚀 客户端直连执行 (Action Interception)
                    # -----------------------------------------------------
                    if final_reply_content:
                        self.memory_stream.update_memory(symbol_to_analyze, final_reply_content)
                        
                        # 正则提取 ACTION
                        action_match = re.search(r'<ACTION>([A-Z]+)</ACTION>', final_reply_content, re.IGNORECASE)
                        action = action_match.group(1).upper() if action_match else None
                        
                        if action == "EMERGENCY_LIQUIDATE":
                            logger.error(f"{Fore.RED}🚨 捕获到核按钮指令 [EMERGENCY_LIQUIDATE]，立即触发强平！{Style.RESET_ALL}")
                            try:
                                res = await self.call_mcp_tool_directly(server_name="Execution", tool_name="kill_all_positions_global", arguments={})
                                logger.info(f"{Fore.GREEN}✅ 强平调用结果: {res}{Style.RESET_ALL}")
                            except Exception as e:
                                logger.error(f"❌ 全局强平调用失败: {e}")
                                try:
                                    res = await self.call_mcp_tool_directly(
                                        server_name="Execution",
                                        tool_name="kill_all_positions",
                                        arguments={"symbol": symbol_to_analyze},
                                    )
                                    logger.info(f"{Fore.GREEN}✅ 单品种强平调用结果: {res}{Style.RESET_ALL}")
                                except Exception as e2:
                                    logger.error(f"❌ 单品种强平调用失败: {e2}")
                        elif action == "REBALANCE":
                            weights_match = re.search(r"<WEIGHTS>(.*?)</WEIGHTS>", final_reply_content, re.IGNORECASE | re.DOTALL)
                            if not weights_match:
                                logger.error("❌ 未找到 <WEIGHTS>...</WEIGHTS>，按 HOLD 处理。")
                            else:
                                weights_text = weights_match.group(1).strip()
                                try:
                                    weights = json.loads(weights_text)
                                    if not isinstance(weights, dict):
                                        raise ValueError("WEIGHTS JSON must be an object/dict")
                                    res = await self.call_mcp_tool_directly(
                                        server_name="Strategy",
                                        tool_name="rebalance_capital_allocation",
                                        arguments={"strategy_weights": weights},
                                    )
                                    logger.info(f"{Fore.GREEN}✅ 调仓调用结果: {res}{Style.RESET_ALL}")
                                except Exception as e:
                                    logger.error(f"❌ WEIGHTS 解析或调仓失败: {e}。按 HOLD 处理。")
                        elif action == "HOLD":
                            logger.info(f"{Fore.BLUE}🧊 收到 HOLD 指令，维持现状。{Style.RESET_ALL}")
                        elif action in ["BUY", "SELL"]:
                            logger.info(f"{Fore.YELLOW}⚔️ 捕获到交易指令 [{action}]，Python 客户端接管执行流程...{Style.RESET_ALL}")
                            # 固定下单 500 USDT (您可以根据需要通过正则进一步提取金额，这里按要求执行一笔固定数量交易，或默认 500)
                            # 从 user_input 提取金额
                            amount_match = re.search(r'(\d+)\s*USDT', user_input, re.IGNORECASE)
                            amount_usd = float(amount_match.group(1)) if amount_match else 500.0
                            
                            if "execute_smart_order" in self.tool_routing_map:
                                exec_session = self.tool_routing_map["execute_smart_order"]
                                try:
                                    logger.info(f"{Fore.YELLOW}🚀 发往下单工具: symbol={symbol_to_analyze}, side={action.lower()}, amount={amount_usd}{Style.RESET_ALL}")
                                    # 使用 asyncio.wait_for 防止订单执行卡死
                                    exec_result = await asyncio.wait_for(
                                        exec_session.call_tool("execute_smart_order", arguments={
                                            "symbol": symbol_to_analyze, 
                                            "side": action.lower(), 
                                            "amount_usd": amount_usd
                                        }),
                                        timeout=10.0
                                    )
                                    logger.info(f"{Fore.GREEN}✅ 交易执行结果: {exec_result.content[0].text}{Style.RESET_ALL}")
                                except Exception as e:
                                    logger.error(f"❌ 交易执行失败: {e}")
                            else:
                                logger.error("未找到 execute_smart_order 工具的路由映射")
                        elif action == "WAIT":
                            logger.info(f"{Fore.BLUE}⏳ 收到观望指令，本轮分析结束。{Style.RESET_ALL}")
                        else:
                            logger.warn(f"⚠️ 未能从模型输出中提取到标准的 <ACTION> 标签。")

                except Exception as e:
                    logger.error(f"推理请求失败: {e}")
        except Exception as e:
            logger.error(f"Agent 交互主循环发生异常: {e}")

    async def cleanup(self):
        """优雅清理资源"""
        logger.info("🔌 正在关闭 MCP 集群连接 (Agent Client)...")
        try:
            # 清空路由表
            self.tool_routing_map.clear()
            
            # 1. 逐个关闭 session (先停止收发消息)
            for session in reversed(self.sessions):
                try:
                    # 避免使用 __aexit__，直接取消后台收发任务
                    if hasattr(session, '_cancel_tasks'):
                        session._cancel_tasks()
                except Exception:
                    pass
            self.sessions.clear()
            
            # 2. 对于底层的 stdio 管道，不调用 __aexit__ 以免触发 anyio 的跨任务异常
            # 直接通过系统垃圾回收来清理进程即可
            self.transports.clear()
            
            logger.info("✅ MCP 集群连接已断开")
        except Exception as e:
            logger.warning(f"⚠️ 清理 MCP 集群时发生异常 (可忽略): {e}")

if __name__ == "__main__":
    import sys
    agent = TrendMasterAgent()
    asyncio.run(agent.run_chat_loop())
