import asyncio
import os
import json
import sys
import re
from datetime import datetime
from contextlib import AsyncExitStack
from typing import Dict, List, Any
from dotenv import load_dotenv

from anthropic import AsyncAnthropic
from anthropic.types.message_param import MessageParam
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from colorama import Fore, Style, init

from shared.logger import get_logger
from openai_client import DeepSeekAdapter # 导入适配器

# 初始化 Colorama
init(autoreset=True)

# 加载环境变量
load_dotenv(override=True)

logger = get_logger("Agent-Client")

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
        self.mcp_servers = {
            "Market": os.path.join(base_dir, "servers/market_mcp/server.py"),
            "Indicator": os.path.join(base_dir, "servers/indicator_mcp/server.py"),
            "Execution": os.path.join(base_dir, "servers/execution_mcp/server.py"),
            "Sentiment": os.path.join(base_dir, "servers/sentiment_mcp/server.py"),
            "FactorLab": os.path.join(base_dir, "servers/factor_lab_mcp/server.py"),
            "Strategy": os.path.join(base_dir, "servers/strategy_mcp/server.py")
        }

    def load_skill_sop(self) -> str:
        """加载 Agent 的核心交易纪律"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        try:
            local_path = os.path.join(base_dir, "skills", "Strategy-SOP.skill")
            root_path = os.path.join(repo_root, "skills", "Strategy-SOP.skill")
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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
            local_path = os.path.join(base_dir, "skills", filename)
            root_path = os.path.join(repo_root, "skills", filename)
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
            
            # [Phase 32] 动态读取最新的 system_prompt.txt
            prompt_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), "data", "system_prompt.txt")
            try:
                if os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        dynamic_prompt = f.read()
                else:
                    dynamic_prompt = self.load_skill_sop()
            except Exception as e:
                logger.error(f"读取动态 prompt 失败，回退至默认 SOP: {e}")
                dynamic_prompt = self.load_skill_sop()
                
            system_prompt = dynamic_prompt
            
            # [由于 agent_client 独有的多技能加载，我们仍保留这些附加技能]
            system_prompt += "\n\n" + self.load_skill("Trend-Sniper")
            system_prompt += "\n\n" + self.load_skill("Mean-Reversion")
            system_prompt += "\n\n" + self.load_skill("Portfolio-Risk")
            system_prompt += "\n\n【系统强制指令】\n在你开始任何交易分析前，必须首先阅读上方的 Strategy-SOP 规则。\n**分析行情时，只能调用 `get_full_market_context` 工具一次，不要尝试分开获取指标！**\n你必须在思考链 (Chain of Thought) 中明确展示你是如何将 HMM 状态、SMC 信号与资金面因子进行对齐的。绝对禁止在没有任何数据支撑的情况下瞎猜点位。"
            system_prompt += "\n\n【舆情强制指令】\n当且仅当你准备输出 BUY/SELL 时，必须先调用一次 `get_comprehensive_sentiment` 获取舆情与宏观日历；若存在 impact=CRITICAL 的宏观事件，必须输出 WAIT。"
            
            # OpenAI 格式需要将 system prompt 放入 messages 列表
            messages = []
            if self.llm_provider in ["deepseek", "ofox"]:
                messages.append({"role": "system", "content": system_prompt})
                
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
                logger.info(f"{Fore.YELLOW}🌍 正在拉取宏观情绪与资金面因子...{Style.RESET_ALL}")
                fear_greed_index = 45  # 1-100，模拟数据
                funding_rate = 0.01    # 资金费率，模拟数据
                ls_ratio = 1.2         # 多空比，模拟数据
                
                macro_factors = (
                    f"【宏观与资金面因子】\n"
                    f"- Fear & Greed Index (恐慌贪婪指数): {fear_greed_index} (Neutral)\n"
                    f"- Funding Rate (资金费率): {funding_rate}%\n"
                    f"- Long/Short Ratio (多空比): {ls_ratio}\n"
                )
                logger.info(f"{Fore.GREEN}✅ 宏观与资金面因子已就绪，准备组装终极 Prompt{Style.RESET_ALL}")
                    
                # 注入前置记忆
                past_memory = self.memory_stream.get_last_memory(symbol_to_analyze)
                
                # 组装终极 Prompt
                enriched_input = (
                    f"{past_memory}\n\n"
                    f"【系统强制注入的底层技术数据】\n{fat_data}\n\n"
                    f"{macro_factors}\n\n"
                    f"当前长官指令：{user_input}\n"
                    f"请直接综合上述【技术面】与【资金面】数据输出最终交易决策，严禁调用任何外部工具！"
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
