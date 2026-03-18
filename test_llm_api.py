import asyncio
import os
import time
import json
from dotenv import load_dotenv
from colorama import Fore, Style, init
from openai_client import DeepSeekAdapter
from shared.logger import get_logger

# 初始化
init(autoreset=True)
load_dotenv()
logger = get_logger("LLM-Test")

async def test_llm_api():
    print(f"\n{Fore.CYAN}🚀 开始 LLM API 端到端测试流程...{Style.RESET_ALL}\n")
    
    # ------------------------------------------------------------------
    # 1. 配置加载验证
    # ------------------------------------------------------------------
    print(f"{Fore.YELLOW}[Step 1] 验证环境配置...{Style.RESET_ALL}")
    provider = os.getenv("LLM_PROVIDER", "").lower()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    model = os.getenv("SILICONFLOW_MODEL")
    base_url = os.getenv("SILICONFLOW_BASE_URL")
    
    print(f"  - Provider: {provider}")
    print(f"  - Model: {model}")
    print(f"  - Base URL: {base_url}")
    print(f"  - API Key: {'*' * 8 + api_key[-4:] if api_key else 'MISSING'}")
    
    if provider != "deepseek" or not api_key:
        print(f"{Fore.RED}❌ 配置验证失败: 请确保 LLM_PROVIDER=deepseek 且 API Key 存在。{Style.RESET_ALL}")
        return

    try:
        adapter = DeepSeekAdapter()
        print(f"{Fore.GREEN}✅ 适配器初始化成功。{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}❌ 适配器初始化失败: {e}{Style.RESET_ALL}")
        return

    # ------------------------------------------------------------------
    # 2. 基础连通性测试 (Simple Chat)
    # ------------------------------------------------------------------
    print(f"{Fore.YELLOW}[Step 2] 测试基础连通性 (Hello World)...{Style.RESET_ALL}")
    test_msg = [{"role": "user", "content": "Hello, reply with 'PONG' only."}]
    
    start_time = time.time()
    try:
        response = await adapter.chat_completion(messages=test_msg)
        elapsed = time.time() - start_time
        
        content = response.content
        print(f"  - Request: {json.dumps(test_msg, ensure_ascii=False)}")
        print(f"  - Response: {content}")
        print(f"  - Latency: {elapsed:.4f}s")
        
        if "PONG" in content.upper():
            print(f"{Fore.GREEN}✅ 连通性测试通过。{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.RED}❌ 响应内容不符合预期。{Style.RESET_ALL}\n")
            
    except Exception as e:
        print(f"{Fore.RED}❌ 连通性测试失败: {e}{Style.RESET_ALL}\n")
        return

    # ------------------------------------------------------------------
    # 3. 工具调用测试 (Tool Calling)
    # ------------------------------------------------------------------
    print(f"{Fore.YELLOW}[Step 3] 测试工具调用能力 (Function Calling)...{Style.RESET_ALL}")
    
    # 定义一个简单的测试工具
    mock_tools = [{
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }]
    
    tool_msg = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    
    try:
        print(f"  - Sending Tool Definition: {json.dumps(mock_tools, indent=2)}")
        response = await adapter.chat_completion(messages=tool_msg, tools=mock_tools)
        
        if response.tool_calls:
            tc = response.tool_calls[0]
            print(f"  - Tool Call Received: {tc.function.name}")
            print(f"  - Arguments: {tc.function.arguments}")
            
            if tc.function.name == "get_weather" and "Tokyo" in tc.function.arguments:
                print(f"{Fore.GREEN}✅ 工具调用测试通过。{Style.RESET_ALL}\n")
            else:
                print(f"{Fore.RED}❌ 工具参数解析错误。{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.RED}❌ 模型未返回工具调用请求。{Style.RESET_ALL}\n")
            print(f"  - Actual Content: {response.content}")

    except Exception as e:
        print(f"{Fore.RED}❌ 工具调用测试失败: {e}{Style.RESET_ALL}\n")

    # ------------------------------------------------------------------
    # 4. 错误处理测试 (Invalid API Key)
    # ------------------------------------------------------------------
    print(f"{Fore.YELLOW}[Step 4] 测试错误处理 (模拟无效 Key)...{Style.RESET_ALL}")
    
    # 临时篡改 Key
    original_key = adapter.client.api_key
    adapter.client.api_key = "sk-invalid-key-for-test"
    
    try:
        await adapter.chat_completion(messages=test_msg)
        print(f"{Fore.RED}❌ 未捕获预期错误 (AuthenticationError)。{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"  - Caught Expected Error: {str(e)}")
        print(f"{Fore.GREEN}✅ 错误处理测试通过。{Style.RESET_ALL}\n")
    finally:
        adapter.client.api_key = original_key # 恢复 Key

    print(f"{Fore.CYAN}🎉 所有测试步骤执行完毕。{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(test_llm_api())
