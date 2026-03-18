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
load_dotenv()

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
            self.anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            if not self.anthropic.api_key:
                logger.error("未找到 ANTHROPIC_API_KEY 环境变量！")
        elif self.llm_provider == "deepseek":
            self.deepseek = DeepSeekAdapter()
        else:
            logger.error(f"不支持的 LLM 提供商: {self.llm_provider}")
            
        # 记录 tool_name 对应哪个 MCP Session
        self.tool_routing_map: Dict[str, ClientSession] = {} 
        self.available_tools: List[Dict[str, Any]] = []
        self.memory_stream = AgentMemory() # 实例化记忆库
        self.exit_stack = AsyncExitStack()
        
        # 我们的 3 个微服务路径 (这里使用 python 命令启动)
        self.mcp_servers = {
            "Market": "servers/market_mcp/server.py",
            "Indicator": "servers/indicator_mcp/server.py",
            "Execution": "servers/execution_mcp/server.py"
        }

    def load_skill_sop(self) -> str:
        """加载 Agent 的核心交易纪律"""
        try:
            with open("skills/Strategy-SOP.skill", "r", encoding="utf-8") as f:
                sop = f.read()
                logger.info("✅ 成功加载 Strategy-SOP.skill")
                return sop
        except Exception as e:
            logger.error(f"无法读取 SOP 规则: {e}")
            return "You are a helpful trading assistant."

    async def init_mcp_connections(self):
        """同时启动并连接多个 MCP Server，使用 AsyncExitStack 管理生命周期"""
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
                # 1. 进入 stdio_client 上下文
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                
                # 2. 进入 ClientSession 上下文
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                
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
                    
                logger.info(f"{Fore.GREEN}✅ {name}-MCP 准备就绪{Style.RESET_ALL}")
                
            except Exception as e:
                logger.error(f"连接 {name}-MCP 失败: {e}")

        logger.info(f"共加载了 {len(self.available_tools)} 个可用工具。")

    async def run_chat_loop(self):
        """启动 Agent 交互主循环"""
        try:
            # 启动所有连接，如果失败会抛出异常
            await self.init_mcp_connections()
            
            system_prompt = self.load_skill_sop()
            system_prompt += "\n\n【系统强制指令】\n在你开始任何交易分析前，必须首先阅读上方的 Strategy-SOP 规则。\n**分析行情时，只能调用 `get_full_market_context` 工具一次，不要尝试分开获取指标！**\n你必须在思考链 (Chain of Thought) 中明确展示你是如何将 HMM 状态与 SMC 信号进行对齐的。绝对禁止在没有任何数据支撑的情况下瞎猜点位。"
            
            # OpenAI 格式需要将 system prompt 放入 messages 列表
            messages = []
            if self.llm_provider == "deepseek":
                messages.append({"role": "system", "content": system_prompt})
            
            logger.info("\n" + "="*60)
            logger.info(f"{Fore.MAGENTA}🤖 TrendMaster 核心中枢已上线 (Powered by {self.llm_provider.upper()} & MCP){Style.RESET_ALL}")
            logger.info(f"{Fore.MAGENTA}🧠 记忆流模块已就绪 (agent_memory.json){Style.RESET_ALL}")
            logger.info("="*60 + "\n")

            # Anthropic messages 列表 (不包含 system)
            anthropic_messages: List[MessageParam] = []
            
            while True:
                user_input = input(f"\n{Fore.BLUE}🧑‍💻 长官，请下达指令 (输入 'exit' 退出): {Style.RESET_ALL}")
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue
                
                # 智能提取 Symbol (假设用户输入中包含 BTC 或 ETH 等)
                symbol_to_analyze = "BTC/USDT" # 默认 fallback
                if "ETH" in user_input.upper():
                    symbol_to_analyze = "ETH/USDT"
                elif "BTC" in user_input.upper():
                    symbol_to_analyze = "BTC/USDT"
                    
                # 注入前置记忆
                past_memory = self.memory_stream.get_last_memory(symbol_to_analyze)
                enriched_input = f"{past_memory}\n\n当前指令：{user_input}\n请结合历史记忆进行连贯性分析。"
                    
                # 统一添加 User Message
                if self.llm_provider == "deepseek":
                    messages.append({"role": "user", "content": enriched_input})
                else:
                    anthropic_messages.append({"role": "user", "content": enriched_input})
                
                # ---------------------------------------------------------
                # ReAct 核心循环 (Reasoning + Acting)
                # ---------------------------------------------------------
                # 暂存大模型回复，用于写入记忆
                final_reply_content = ""

                while True:
                    logger.info(f"{Fore.CYAN}🧠 Agent 正在思考...{Style.RESET_ALL}")
                    
                    try:
                        if self.llm_provider == "anthropic":
                            # Anthropic 调用逻辑
                            response = await self.anthropic.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                max_tokens=4096,
                                system=system_prompt,
                                messages=anthropic_messages,
                                tools=self.available_tools
                            )
                            # 记录 Assistant 的完整回复
                            anthropic_messages.append({"role": "assistant", "content": response.content})
                            
                            tool_calls = [block for block in response.content if block.type == "tool_use"]
                            text_blocks = [block.text for block in response.content if block.type == "text"]
                            if text_blocks:
                                final_reply_content = text_blocks[0]
                                print(f"\n{Fore.MAGENTA}🤖 TrendMaster:{Style.RESET_ALL} {text_blocks[0]}")
                                
                        elif self.llm_provider == "deepseek":
                            # DeepSeek 调用逻辑
                            response_msg = await self.deepseek.chat_completion(
                                messages=messages,
                                tools=self.available_tools
                            )
                            # 记录 Assistant 回复
                            messages.append(response_msg) # response_msg 是 OpenAI 格式的对象或字典
                            
                            tool_calls = response_msg.tool_calls # OpenAI 格式的 tool_calls
                            if response_msg.content:
                                final_reply_content = response_msg.content
                                print(f"\n{Fore.MAGENTA}🤖 TrendMaster:{Style.RESET_ALL} {response_msg.content}")

                        # -----------------------------------------------------
                        # 统一处理 Tool Calls
                        # -----------------------------------------------------
                        if not tool_calls:
                            # 循环结束，归档记忆
                            if final_reply_content:
                                self.memory_stream.update_memory(symbol_to_analyze, final_reply_content)
                            break # 无工具调用，结束本轮对话
                            
                        # 如果有 Tool Calls，执行它们并将结果返还给大模型
                        tool_results = []
                        
                        # Anthropic 和 OpenAI 的 tool_calls 结构略有不同，需要适配
                        normalized_tool_calls = []
                        if self.llm_provider == "anthropic":
                            for tc in tool_calls:
                                normalized_tool_calls.append({
                                    "id": tc.id,
                                    "name": tc.name,
                                    "args": tc.input,
                                    "original": tc
                                })
                        else: # deepseek (OpenAI format)
                            for tc in tool_calls:
                                normalized_tool_calls.append({
                                    "id": tc.id,
                                    "name": tc.function.name,
                                    "args": json.loads(tc.function.arguments),
                                    "original": tc
                                })

                        for tc in normalized_tool_calls:
                            tool_name = tc["name"]
                            tool_args = tc["args"]
                            tool_id = tc["id"]
                            
                            # 路由逻辑
                            session = self.tool_routing_map.get(tool_name)
                            if not session:
                                error_msg = f"找不到工具 {tool_name} 对应的 MCP Server。"
                                logger.error(error_msg)
                                tool_results.append({
                                    "tool_use_id": tool_id,
                                    "content": error_msg,
                                    "is_error": True
                                })
                                continue
                                
                            logger.info(f"{Fore.YELLOW}📡 [Router] 路由 tool '{tool_name}' 至对应的 MCP Server...{Style.RESET_ALL}")
                            logger.info(f"   参数: {json.dumps(tool_args, ensure_ascii=False)}")
                            
                            try:
                                # 发起 RPC 调用
                                result = await session.call_tool(tool_name, arguments=tool_args)
                                result_text = result.content[0].text if result.content else "No Output"
                                
                                logger.info(f"{Fore.GREEN}🟢 [Router] 工具 '{tool_name}' 执行成功。{Style.RESET_ALL}")
                                
                                tool_results.append({
                                    "tool_use_id": tool_id,
                                    "content": result_text,
                                    "is_error": False
                                })
                            except Exception as e:
                                logger.error(f"🔴 [Router] 工具 '{tool_name}' 执行报错: {e}")
                                tool_results.append({
                                    "tool_use_id": tool_id,
                                    "content": str(e),
                                    "is_error": True
                                })
                        
                        # 将结果回传给 LLM
                        if self.llm_provider == "anthropic":
                            # Anthropic 格式回传
                            content_list = []
                            for tr in tool_results:
                                content_list.append({
                                    "type": "tool_result",
                                    "tool_use_id": tr["tool_use_id"],
                                    "content": tr["content"],
                                    "is_error": tr.get("is_error", False)
                                })
                            anthropic_messages.append({"role": "user", "content": content_list})
                            
                        elif self.llm_provider == "deepseek":
                            # OpenAI 格式回传
                            for tr in tool_results:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tr["tool_use_id"],
                                    "content": tr["content"]
                                })

                    except Exception as e:
                        logger.error(f"大模型通信或处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        break

        except Exception as e:
            logger.error(f"Agent 初始化失败: {e}")
        finally:
            # 无论发生什么，优雅关闭所有 MCP 进程
            logger.info("正在关闭所有 MCP 连接...")
            await self.exit_stack.aclose()
            logger.info("退出完成。")

if __name__ == "__main__":
    import sys
    agent = TrendMasterAgent()
    asyncio.run(agent.run_chat_loop())
