import os
import json
from openai import AsyncOpenAI
from shared.logger import get_logger

logger = get_logger("OpenAI-Adapter")

class DeepSeekAdapter:
    """
    适配兼容 OpenAI 格式接口的 LLM 提供商 (如 SiliconFlow 的 DeepSeek，或 Ofox 的 GPT-5.4-Mini)。
    用于替代 Anthropic 客户端，实现统一的调用逻辑。
    """
    def __init__(self, provider="deepseek"):
        self.provider = provider
        
        if provider == "ofox":
            self.api_key = os.getenv("OFOX_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OFOX_BASE_URL", "https://api.ofox.ai/v1")
            self.model = os.getenv("LLM_MODEL_FAST") or os.getenv("OFOX_MODEL", "openai/gpt-5.4-mini")
        else:
            self.api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OFOX_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("SILICONFLOW_BASE_URL") or os.getenv("OFOX_BASE_URL", "https://api.siliconflow.cn/v1")
            self.model = os.getenv("LLM_MODEL_FAST") or os.getenv("SILICONFLOW_MODEL") or os.getenv("OFOX_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")
        
        if not self.api_key:
            logger.error(f"未找到 {provider.upper()}_API_KEY，适配器无法初始化！")
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=int(os.getenv("LLM_API_MAX_RETRIES", "2") or 2),
            timeout=float(os.getenv("LLM_API_TIMEOUT_SEC", "60") or 60),
        )

    def convert_tools_to_openai_format(self, anthropic_tools: list) -> list:
        """
        将 Anthropic 格式的 Tools 转换为 OpenAI/DeepSeek 兼容的格式。
        
        Anthropic Format:
        {
            "name": "get_ticker",
            "description": "...",
            "input_schema": { ... }
        }
        
        OpenAI Format:
        {
            "type": "function",
            "function": {
                "name": "get_ticker",
                "description": "...",
                "parameters": { ... }
            }
        }
        """
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        return openai_tools

    async def chat_completion(self, messages: list, tools: list = None, stream: bool = True):
        """
        发送聊天请求，兼容 Anthropic 的 message 格式。
        """
        # 1. 转换消息格式 (Anthropic -> OpenAI)
        # Anthropic: [{"role": "user", "content": "..."}] (有时 content 是 list)
        # OpenAI: [{"role": "user", "content": "..."}] (content 通常是 str)
        
        openai_messages = []
        for msg in messages:
            # 兼容：如果 msg 是对象 (比如上一次请求返回的 ChatCompletionMessage)，先转为 dict
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump(exclude_none=True)
            elif hasattr(msg, "dict"):
                msg = msg.dict(exclude_none=True)
                
            role = msg.get("role")
            content = msg.get("content")
            
            # OpenAI 原生消息包含 tool_calls
            if role == "assistant" and msg.get("tool_calls"):
                openai_messages.append(msg)
                continue
            
            # 处理 Tool Result 格式回传
            if isinstance(content, list):
                # 检查是否是 tool_result
                if content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                    for item in content:
                        if item.get("type") == "tool_result":
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"]
                            })
                    continue # 跳过本次循环，因为 tool result 已经作为独立 message 添加
                
                # 检查是否是 tool_use (DeepSeek 返回的 Assistant Message)
                # 通常不需要手动转换发回去，OpenAI 会自动维护 context
                # 但如果是手动构建的历史记录，需要注意
                
                # 简单处理：如果是文本列表，合并为字符串
                text_parts = [item["text"] for item in content if item.get("type") == "text"]
                if text_parts:
                    openai_messages.append({
                        "role": role,
                        "content": "\n".join(text_parts)
                    })
            else:
                # 普通文本消息
                openai_messages.append({
                    "role": role,
                    "content": content
                })
        
        # 2. 转换 Tools
        openai_tools = self.convert_tools_to_openai_format(tools) if tools else None
        
        # 3. 发起请求
        try:
            # 移除 kwargs 中不支持的参数 (如果需要流式输出，可以增加 stream=True 参数处理)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                # 不再传递 tools，因为在 One-Shot 模式下大模型被剥夺了工具权限
                # tools=openai_tools,
                stream=stream # 可选流式输出
            )
            return response
        except Exception as e:
            logger.error(f"{self.provider.upper()} API 请求失败: {e}")
            raise e
