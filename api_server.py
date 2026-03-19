import asyncio
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from shared.logger import get_logger

# 引入我们已经打磨完美的 Agent 核心组件
from agent_client import TrendMasterAgent 

logger = get_logger("API-Gateway")

app = FastAPI(title="TrendMaster Quant 4.0 API", version="1.0")
agent = TrendMasterAgent()

# 启动时初始化 MCP 连接
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 正在启动 API 网关并连接 MCP 底层服务...")
    await agent.init_mcp_connections()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 正在关闭 API 网关及 MCP 连接...")
    await agent.exit_stack.aclose()

class TradeRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    amount_usd: float = 500.0
    instruction: str = "分析最近的行情，查找合适的开单时机"

class TradeResponse(BaseModel):
    symbol: str
    action: str
    reasoning: str
    execution_status: str

@app.post("/api/v1/analyze_and_trade", response_model=TradeResponse)
async def analyze_and_trade(req: TradeRequest):
    logger.info(f"📥 收到主系统调用请求: {req.symbol}")
    
    try:
        # 1. 客户端直连获取 Fat-Tool 脱水数据
        logger.info("⚡ 并发请求 Fat-Tool 数据...")
        fat_data = await agent.call_mcp_tool_directly(
            "Indicator-MCP", 
            "get_full_market_context", 
            {"symbol": req.symbol, "timeframe": req.timeframe}
        )
        
        # 2. 提取记忆并组装 One-Shot Prompt
        past_memory = agent.memory_stream.get_last_memory(req.symbol)
        enriched_input = (
            f"{past_memory}\n\n"
            f"【系统强制注入的底层数据】\n{fat_data}\n\n"
            f"当前主系统指令：{req.instruction}，计划开仓金额：{req.amount_usd} USDT\n"
            f"请直接根据上述数据输出最终交易决策，严禁调用任何外部工具！"
        )
        
        # 3. 发起非流式极速推理 (因为是 API 调用，主系统不需要看打字机效果)
        logger.info("🧠 大模型正在进行极速推理...")
        
        system_prompt = agent.load_skill_sop()
        system_prompt += "\n\n【系统强制指令】\n在你开始任何交易分析前，必须首先阅读上方的 Strategy-SOP 规则。\n**分析行情时，只能调用 `get_full_market_context` 工具一次，不要尝试分开获取指标！**\n你必须在思考链 (Chain of Thought) 中明确展示你是如何将 HMM 状态与 SMC 信号进行对齐的。绝对禁止在没有任何数据支撑的情况下瞎猜点位。"
        
        reply = ""
        
        if agent.llm_provider == "anthropic":
            # Anthropic 调用
            messages = [{"role": "user", "content": enriched_input}]
            response = await agent.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                stream=False
            )
            reply = response.content[0].text
            
        elif agent.llm_provider in ["deepseek", "ofox"]:
            # DeepSeek/Ofox 调用 (OpenAI 兼容)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enriched_input}
            ]
            # 调用适配器，传入 stream=False
            response = await agent.deepseek.chat_completion(
                messages=messages,
                stream=False
            )
            # OpenAI 非流式响应结构: response.choices[0].message.content
            reply = response.choices[0].message.content
            
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported LLM provider: {agent.llm_provider}")
        
        # 4. 解析 Action 并归档记忆
        action_match = re.search(r'<ACTION>([A-Z]+)</ACTION>', reply, re.IGNORECASE)
        action = action_match.group(1).upper() if action_match else "UNKNOWN"
        agent.memory_stream.update_memory(req.symbol, reply)
        
        # 5. 拦截执行逻辑
        exec_status = "Skipped (Action was WAIT)"
        if action in ["BUY", "SELL"]:
            logger.info(f"⚔️ 触发执行层: {action} {req.symbol} ${req.amount_usd}")
            exec_res = await agent.call_mcp_tool_directly(
                "Execution-MCP", 
                "execute_smart_order", 
                {"symbol": req.symbol, "side": action.lower(), "amount_usd": req.amount_usd}
            )
            exec_status = str(exec_res)
        elif action == "WAIT":
             logger.info(f"⏳ 收到观望指令，跳过执行。")
            
        return TradeResponse(
            symbol=req.symbol,
            action=action,
            reasoning=reply[:500], # 返回前500字的核心推理供主系统落库
            execution_status=exec_status
        )

    except Exception as e:
        logger.error(f"API 处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
