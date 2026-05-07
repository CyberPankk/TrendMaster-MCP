import sys
import os
import time
import json

# 将主项目根目录加入到 sys.path 中，以便能读取到 shared/telegram_notifier.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import asyncio
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn
from shared.logger import get_logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from decimal import Decimal
import uuid

# 引入我们已经打磨完美的 Agent 核心组件
from agent_client import TrendMasterAgent 
from shared.telegram_notifier import send_tg_alert
from shared.db_manager import init_db, insert_trade_log, get_today_summary, get_recent_shadow_trades, get_system_metrics
from shared.shadow_ledger import ShadowLedger, estimate_cost_usdt, safe_decimal
from mvp_system.core.dynamic_allocator import DynamicAllocator
from datetime import datetime
from skills_pool.shadow.shadow_manager import ShadowPoolManager
from skills_pool.lifecycle_manager import StrategyLifecycleManager

logger = get_logger("API-Gateway")

app = FastAPI(title="TrendMaster Quant 4.0 API", version="1.0")

# 实例化动态资金分配器
dynamic_allocator = DynamicAllocator(config={
    "risk_control": {
        "base_risk_pct": 0.05,
        "max_position_pct": 0.15
    }
})

# 配置 CORS 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = TrendMasterAgent()
scheduler = AsyncIOScheduler()
shadow_manager = ShadowPoolManager()
lifecycle_manager = StrategyLifecycleManager()
shadow_ledger = ShadowLedger(initial_usdt=safe_decimal(os.getenv("SHADOW_LEDGER_INITIAL_USDT", "0")))
shadow_ledger_enabled = os.getenv("SHADOW_LEDGER_ENABLED", "true").lower() in ("true", "1", "yes")

START_TIME = time.time()

@app.get("/api/v1/health")
async def health_check():
    uptime_seconds = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # scheduler.state == 1 means RUNNING, 2 means PAUSED
    scheduler_state = "running" if scheduler.state == 1 else "paused" if scheduler.state == 2 else "stopped"
    
    return {
        "status": "success",
        "data": {
            "uptime": uptime_str,
            "scheduler_state": scheduler_state,
            "mcp_status": "connected"
        }
    }

@app.post("/api/v1/system/kill")
async def system_kill():
    logger.warning("🚨 [KILL SWITCH] 收到全局熔断指令！")
    try:
        # 第一步：暂停调度器
        scheduler.pause()
        logger.info("✅ 调度器已暂停，停止自动巡航。")
        
        # 第二步：调用底层 MCP，强制撤单平仓
        await agent.call_mcp_tool_directly("Execution-MCP", "kill_all_positions_global", {})
        logger.info("✅ 底层 MCP 强制撤单平仓指令已发送。")
        
        # 第三步：发送 TG 报警
        await send_tg_alert("🚨 警告！触发全局熔断机制 (KILL SWITCH)！已强制撤单平仓并停止自动巡航！")
        
        return {"status": "success", "message": "Global Kill Switch Activated"}
    except Exception as e:
        logger.error(f"❌ [KILL SWITCH] 执行熔断失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/system/resume")
async def system_resume():
    logger.info("✅ [RESUME] 收到系统恢复指令！")
    try:
        scheduler.resume()
        logger.info("✅ 调度器已恢复，重新开始自动巡航。")
        
        await send_tg_alert("✅ 系统已解除熔断，自动巡航恢复。")
        
        return {"status": "success", "message": "System Resumed"}
    except Exception as e:
        logger.error(f"❌ [RESUME] 系统恢复失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======= 新增配置：被监控与交易的目标代币池 =======
# 动态读取共享配置
def get_target_symbols():
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    import json, os
    symbols_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "monitored_symbols.json")
    if os.path.exists(symbols_file):
        try:
            with open(symbols_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list) and len(saved) > 0:
                    symbols = saved
        except Exception:
            pass
    return symbols

async def run_shadow_pool_job():
    """沙盒策略池自动巡航任务"""
    target_symbols = get_target_symbols()
    logger.info(f"👻 [Shadow-Pool] 触发定时沙盒巡航，目标交易对: {target_symbols}...")
    try:
        # 1. 尝试动态加载最新策略（支持热更新）
        shadow_manager.load_skills()
        timeframe = "1h"
        
        # 2. 遍历所有配置的交易对，获取行情数据并驱动沙盒
        for symbol in target_symbols:
            logger.info(f"👻 [Shadow-Pool] 正在获取 {symbol} 行情数据供沙盒分析...")
            try:
                # 调用 Indicator-MCP 获取 K线数据（复用已有能力）
                klines_data = await agent.call_mcp_tool_directly(
                    "Indicator-MCP", 
                    "get_full_market_context", 
                    {"symbol": symbol, "timeframe": timeframe}
                )
                
                # 尝试获取当前最新价格，如果无法获取，可以用一个占位符，或者尝试解析 klines_data
                # 这里我们调用 Market-MCP 或直接在数据中获取
                current_price = 0.0
                try:
                    ticker_res = await agent.call_mcp_tool_directly("Market-MCP", "get_ticker", {"symbol": symbol})
                    if isinstance(ticker_res, dict) and "data" in ticker_res:
                        import ast
                        ticker_data = ast.literal_eval(ticker_res["data"])
                        current_price = float(ticker_data.get("last", 0.0))
                    elif isinstance(ticker_res, dict):
                        current_price = float(ticker_res.get("last", 0.0))
                except Exception as e:
                    logger.warning(f"⚠️ [Shadow-Pool] {symbol} 获取最新价格失败，将使用默认价格 0.0: {e}")
                    
                # 3. 驱动沙盒管理器执行虚拟交易
                await shadow_manager.execute_virtual_trading(klines_data, current_price)
                
            except Exception as e:
                logger.error(f"❌ [Shadow-Pool] {symbol} 沙盒执行异常: {e}")
            
            # API 限频保护机制：每个币种错峰请求
            await asyncio.sleep(3.0)
            
    except Exception as e:
        logger.error(f"❌ [Shadow-Pool] 沙盒巡航任务异常崩溃: {e}")

async def auto_cruise_job():
    """后台自动巡航任务"""
    logger.info(f"⚙️ [Auto-Cruise] 触发定时巡航，目标交易对: {TARGET_SYMBOLS}...")
    try:
        for symbol in TARGET_SYMBOLS:
            logger.info(f"🔍 [Auto-Cruise] 正在执行巡航任务: {symbol}")
            try:
                req = TradeRequest(symbol=symbol, timeframe="1h")
                await analyze_and_trade(req)
            except Exception as e:
                logger.error(f"❌ [Auto-Cruise] {symbol} 巡航任务异常: {e}")
            
            # 增加异步休眠以错峰请求，严防限流封禁
            await asyncio.sleep(3.0)
            
    except Exception as e:
        logger.error(f"❌ [Auto-Cruise] 巡航任务全局异常崩溃: {e}")

async def daily_report_job():
    """每日量化财报推送任务"""
    logger.info("📊 正在生成并推送今日量化财报...")
    try:
        summary = await get_today_summary()
        
        # 组装精美的 Markdown 报告
        report_msg = (
            f"📊 *TrendMaster 每日量化财报*\n\n"
            f"📅 日期：`{datetime.now().strftime('%Y-%m-%d')}`\n"
            f"⚡️ 总交易笔数：`{summary['total_trades']}` 笔\n"
            f"🟢 做多次数 (BUY)：`{summary['buys']}` 笔\n"
            f"🔴 做空次数 (SELL)：`{summary['sells']}` 笔\n"
            f"💰 日内总交易额：`${summary['total_amount_usd']:.2f}`\n\n"
            f"✨ _自动巡航引擎将持续为您守护资产。_"
        )
        
        await send_tg_alert(report_msg)
    except Exception as e:
        logger.error(f"❌ [Daily-Report] 财报生成任务异常: {e}")

async def run_lifecycle_evaluation_job():
    """每日沙盒策略考核结算任务"""
    logger.info("⚔️ [Lifecycle] 触发每日角斗场考核任务...")
    try:
        await lifecycle_manager.evaluate_shadow_skills()
    except Exception as e:
        logger.error(f"❌ [Lifecycle] 角斗场考核任务异常: {e}")

# 启动时初始化 MCP 连接
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 正在启动 API 网关并连接 MCP 底层服务...")
    await agent.init_mcp_connections()
    
    # 初始化 SQLite 账本（彻底异步化，全部使用 aiosqlite）
    await init_db()
    
    # 挂载定时任务
    scheduler.add_job(auto_cruise_job, 'interval', minutes=15)
    scheduler.add_job(run_shadow_pool_job, 'interval', minutes=15)
    scheduler.add_job(run_lifecycle_evaluation_job, 'cron', hour=0, minute=0)
    scheduler.add_job(daily_report_job, 'cron', hour=23, minute=50)
    scheduler.start()
    
    # 发送 Telegram 通知
    await send_tg_alert("🚀 *TrendMaster Quant 4.0* 已上线，API 网关启动成功！\n⚙️ 自动巡航引擎已启动，周期：15分钟\n👻 沙盒策略引擎已启动，周期：15分钟\n⚔️ 角斗场考核已部署，将在每天 00:00 进行策略结算。\n📊 财报调度器已激活，将在每天 23:50 推送今日战报。")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 正在关闭 API 网关及 MCP 连接...")
    scheduler.shutdown()
    await agent.cleanup()

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


class PromptUpdateRequest(BaseModel):
    prompt: str | None = None
    content: str | None = None

    def get_prompt_text(self) -> str:
        """兼容 prompt/content 两种字段名，统一提取热更新内容。"""
        prompt_text = self.prompt if self.prompt is not None else self.content
        if prompt_text is None:
            raise ValueError("请求体缺少 prompt 字段")
        return prompt_text


@app.get("/api/v1/strategy/prompt", response_class=PlainTextResponse)
async def get_strategy_prompt():
    """读取 system_prompt.txt 最新内容，供前端与后端统一查看当前 Prompt。"""
    prompt_content = agent.read_latest_system_prompt()
    return prompt_content


@app.post("/api/v1/strategy/prompt")
async def update_strategy_prompt(req: PromptUpdateRequest):
    """热更新 system_prompt.txt，后续 API 调用与 CLI 会自动读取最新内容。"""
    try:
        prompt_path = agent.write_system_prompt(req.get_prompt_text())
        return {
            "status": "success",
            "message": "system_prompt.txt updated successfully",
            "data": {
                "path": prompt_path,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"❌ 热更新 system_prompt.txt 失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/api/v1/shadow/trades")
async def api_get_shadow_trades(limit: int = 50):
    try:
        trades = await get_recent_shadow_trades(limit)
        return {"success": True, "data": trades}
    except Exception as e:
        logger.error(f"获取影子账本失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/shadow/metrics")
async def api_get_shadow_metrics():
    try:
        metrics = await get_system_metrics()
        return {"success": True, "data": metrics}
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/shadow/ledger")
async def api_get_shadow_ledger_snapshot():
    """透出影子账本资金预扣快照，便于前端与审计定位并发超卖风险。"""
    try:
        snapshot = await shadow_ledger.get_snapshot()
        return {"success": True, "data": snapshot, "enabled": shadow_ledger_enabled}
    except Exception as e:
        logger.error(f"获取影子账本快照失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # [Phase 32] 并发获取多维数据因子
        logger.info("🌍 正在拉取多维因子占位上下文...")
        factor_payload = await agent.build_multifactor_payload(req.symbol, req.timeframe)
        macro_factors = agent.format_multifactor_prompt(factor_payload)
        logger.info(
            "DB_RECORD %s",
            json.dumps(
                {
                    "module": "api_server",
                    "event": "multifactor_context_built",
                    "symbol": req.symbol,
                    "timeframe": req.timeframe,
                    "placeholder_mode": factor_payload.get("placeholder_mode", True),
                    "generated_at": factor_payload.get("generated_at"),
                },
                ensure_ascii=False,
            ),
        )
        
        # 2. 提取记忆与失败单经验池，并组装 One-Shot Prompt
        experience_context = await agent.build_decision_context_prefix(req.symbol)
        enriched_input = (
            f"{experience_context}\n\n"
            f"【系统强制注入的底层技术数据】\n{fat_data}\n\n"
            f"{macro_factors}\n\n"
            f"当前主系统指令：{req.instruction}，计划开仓金额：{req.amount_usd} USDT\n"
            f"请直接综合上述【技术面】与【资金面】数据输出最终交易决策，严禁调用任何外部工具！"
        )
        
        # 3. 发起非流式极速推理 (因为是 API 调用，主系统不需要看打字机效果)
        logger.info("🧠 大模型正在进行极速推理...")
        
        system_prompt = agent.build_runtime_system_prompt()
        
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
        # 尝试 JSON 解析，兼容 markdown wrapper
        reply_cleaned = reply.strip()
        if reply_cleaned.startswith("```json"):
            reply_cleaned = reply_cleaned[7:]
        elif reply_cleaned.startswith("```"):
            reply_cleaned = reply_cleaned[3:]
        if reply_cleaned.endswith("```"):
            reply_cleaned = reply_cleaned[:-3]
        reply_cleaned = reply_cleaned.strip()
        
        action = "UNKNOWN"
        try:
            parsed_json = json.loads(reply_cleaned)
            action = parsed_json.get("action", "UNKNOWN").upper()
        except json.JSONDecodeError:
            # Fallback 到旧的正则模式
            action_match = re.search(r'<ACTION>([A-Z]+)</ACTION>', reply, re.IGNORECASE)
            action = action_match.group(1).upper() if action_match else "UNKNOWN"
            
        agent.memory_stream.update_memory(req.symbol, reply)
        
        # 5. 拦截执行逻辑
        exec_status = "Skipped (Action was WAIT)"
        if action in ["BUY", "SELL"]:
            logger.info(f"💰 正在计算动态可用资金头寸...")
            try:
                # 获取真实余额
                balance_res = await agent.call_mcp_tool_directly("Execution-MCP", "get_account_balance", {})
                
                # 安全解析 USDT 余额
                usdt_balance = 0.0
                if isinstance(balance_res, dict):
                    if balance_res.get("status") == "success" and isinstance(balance_res.get("data"), dict):
                        balance_data = balance_res["data"]
                        usdt_balance = float(balance_data.get("free") or balance_data.get("total") or 0.0)
                    elif "data" in balance_res:
                        import ast
                        try:
                            balance_data = ast.literal_eval(balance_res["data"])
                            usdt_balance = float(balance_data.get("free") or balance_data.get("total") or 0.0)
                        except Exception:
                            usdt_balance = float(balance_res.get("data", 0.0))
                    else:
                        usdt_balance = float(balance_res.get("free") or balance_res.get("total") or 0.0)
                else:
                    import ast
                    try:
                        balance_data = ast.literal_eval(balance_res)
                        usdt_balance = float(balance_data.get("free") or balance_data.get("total") or 0.0)
                    except Exception:
                        usdt_balance = float(balance_res)
                        
                logger.info(f"💵 当前账户 USDT 余额: {usdt_balance:.2f}")
            except Exception as e:
                logger.error(f"❌ 动态头寸计算失败，回退至默认资金。原因: {e}")
                usdt_balance = 1000.0

            # 模块化解耦：剥离网关层硬编码的仓位逻辑，下放至专门的 DynamicAllocator
            # 由于 MCP Server 当前流程中并未完全打通 Oracle 预测和确信度，这里我们使用默认中等确信度进行兜底
            confidence = 0.8  # 可以后续扩展让 Agent 传回
            dynamic_amount = dynamic_allocator.calculate_position_size(
                symbol=req.symbol,
                confidence=confidence,
                action=action,
                current_balance=usdt_balance
            )
            
            if dynamic_amount < 10.0:
                logger.warning(f"⚠️ 调度器批复的头寸 {dynamic_amount:.2f} USDT 低于最小发单金额，跳过本次发单！")
                exec_status = "Skipped (Amount too low)"
            else:
                ledger_order_id = f"LEDGER_{uuid.uuid4().hex[:10]}"
                if shadow_ledger_enabled:
                    sync_res = await shadow_ledger.sync_with_exchange_free(safe_decimal(usdt_balance))
                    ok, reason = await shadow_ledger.pre_deduct(
                        order_id=ledger_order_id,
                        symbol=req.symbol,
                        amount_usdt=safe_decimal(dynamic_amount),
                    )
                    if not ok:
                        logger.warning(f"🛡️ [ShadowLedger] 预扣失败，拒绝发单: {reason} | sync={sync_res}")
                        exec_status = f"Skipped (ShadowLedger rejected: {reason})"
                        return TradeResponse(
                            symbol=req.symbol,
                            action="WAIT",
                            reasoning=reply[:500],
                            execution_status=exec_status,
                        )

                logger.info(f"⚔️ 触发执行层: {action} {req.symbol} ${dynamic_amount:.2f} (5% of Balance)")
                exec_res = await agent.call_mcp_tool_directly(
                    "Execution-MCP", 
                    "execute_smart_order", 
                    {"symbol": req.symbol, "side": action.lower(), "amount_usd": dynamic_amount}
                )
                exec_result = exec_res if isinstance(exec_res, dict) else {"status": "unknown", "message": str(exec_res)}

                if shadow_ledger_enabled:
                    result_status = str(exec_result.get("status") or "").lower()
                    if result_status in {"success", "success_dry_run", "partial_success"}:
                        actual_cost = estimate_cost_usdt(
                            exec_result.get("filled_qty"),
                            exec_result.get("average_price") or exec_result.get("average"),
                            exec_result.get("fee"),
                        )
                        ledger_result = await shadow_ledger.reconcile(ledger_order_id, actual_cost, "FILLED")
                    else:
                        await shadow_ledger.release(ledger_order_id)
                        ledger_result = {"status": "released", "reason": result_status or "UNKNOWN"}

                    exec_result["shadow_ledger"] = {"order_id": ledger_order_id, "result": ledger_result}

                exec_status = json.dumps(exec_result, ensure_ascii=False)
                
                # 推送 Telegram 战报
                action_emoji = "🟢" if action == "BUY" else "🔴"
                
                # 尝试从执行结果提取均价
                avg_price = "N/A"
                if isinstance(exec_result, dict):
                    avg_val = exec_result.get("average_price") or exec_result.get("average")
                    if avg_val is not None:
                        avg_price = f"${avg_val}"

                if exec_result.get("status") in {"success", "success_dry_run", "partial_success"}:
                    report_msg = (
                        f"🚀 *TrendMaster Quant 执行战报*\n\n"
                        f"交易对：`{req.symbol}`\n"
                        f"动作：{action_emoji} *{action}*\n"
                        f"开仓金额：`${dynamic_amount:.2f}`\n"
                        f"成交均价：{avg_price}\n"
                        f"当前余额：`${usdt_balance:.2f}`\n\n"
                    )
                    if exec_result.get("status") == "partial_success":
                        report_msg += f"⚠️ *注意*：{exec_result.get('message', '主单成功但部分附加操作失败')}\n\n"
                    report_msg += (
                        f"🧠 *AI 决策核心逻辑*：\n"
                        f"_{reply[:500]}..._"
                    )
                else:
                    fail_reason = exec_result.get("message") or exec_result.get("reason") or exec_result.get("error_code") or exec_status
                    logger.error(f"❌ Execution-MCP 返回失败: {exec_result}")
                    report_msg = (
                        f"❌ *TrendMaster Quant 发单失败*\n\n"
                        f"交易对：`{req.symbol}`\n"
                        f"动作：{action_emoji} *{action}*\n"
                        f"开仓金额：`${dynamic_amount:.2f}`\n"
                        f"当前余额：`${usdt_balance:.2f}`\n"
                        f"失败原因：`{fail_reason}`\n\n"
                        f"🧠 *AI 决策核心逻辑*：\n"
                        f"_{reply[:500]}..._"
                    )
                asyncio.create_task(send_tg_alert(report_msg))
                
                # 记录交易账本
                await insert_trade_log(req.symbol, action, dynamic_amount, reply, exec_status)
                
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
    # 关闭 reload，避免日志或缓存文件更新导致服务自动重启
    uvicorn.run("api_server:app", host="0.0.0.0", port=8002, reload=False)
