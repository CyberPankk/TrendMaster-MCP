# Execution Engine: Powered by ccxt.async_support

import asyncio
import json
import os
import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import ccxt.async_support as ccxt
from decimal import Decimal

# 禁用 ccxt 和 aiohttp 在退出时的烦人警告
try:
    from ccxt.async_support.base.exchange import Exchange
    if hasattr(Exchange, '__del__'):
        Exchange.__del__ = lambda self: None
except Exception:
    pass

try:
    from aiohttp.client import ClientSession
    if hasattr(ClientSession, '__del__'):
        ClientSession.__del__ = lambda self: None
except Exception:
    pass

try:
    from aiohttp.connector import BaseConnector
    if hasattr(BaseConnector, '__del__'):
        BaseConnector.__del__ = lambda self: None
except Exception:
    pass

from mcp.server.fastmcp import FastMCP
import time

# 添加项目根目录到 sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
# 将主工程的根目录也加进去，以便能导入全局的 shared
sys.path.append(str(root_dir.parent.parent))

# 加载 .env 环境变量
load_dotenv(root_dir / ".env", override=True)

from shared.logger import get_logger
from shared.models import MCPErrorResponse
from shared.cache_manager import balance_cache, cache_key_builder
from shared.circuit_breaker import execution_breaker
from shared.shadow_ledger import ShadowLedger, safe_decimal
from shared.db_manager import MAIN_TRADES_DB_PATH
from asyncache import cached

# 加载 .env 文件，确保能够读取到最新配置
load_dotenv()

logger = get_logger("Execution-Server")
mcp = FastMCP("Execution-Server", port=8000)

# 在执行层实例化一个本地的 ShadowLedger，用于 TWAP 后台任务的切片资金预扣和管理
# 初始化可用资金先置为0，在每次执行真实发单前会通过 fetch_balance 同步
execution_shadow_ledger = ShadowLedger(initial_usdt=Decimal("0"))

TWAP_TASKS: dict[str, dict] = {}
TWAP_ACTIVE_BY_SYMBOL: dict[str, str] = {}
TWAP_ASYNC_TASKS: dict[str, asyncio.Task] = {}
TWAP_CONTROL_LOCK: asyncio.Lock | None = None
PROTECTION_TASKS: dict[str, dict] = {}

def _get_twap_control_lock() -> asyncio.Lock:
    global TWAP_CONTROL_LOCK
    if TWAP_CONTROL_LOCK is None:
        TWAP_CONTROL_LOCK = asyncio.Lock()
    return TWAP_CONTROL_LOCK

def _init_protection_tables() -> None:
    try:
        conn = sqlite3.connect(MAIN_TRADES_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS protection_tasks (
                task_id TEXT PRIMARY KEY,
                symbol TEXT,
                kind TEXT,
                close_side TEXT,
                amount_str TEXT,
                trigger_price REAL,
                tag_prefix TEXT,
                status TEXT,
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 0,
                delay_sec REAL DEFAULT 0.0,
                last_error TEXT,
                created_at INTEGER,
                updated_at INTEGER
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def _upsert_protection_task(row: dict) -> None:
    try:
        conn = sqlite3.connect(MAIN_TRADES_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO protection_tasks (
                task_id, symbol, kind, close_side, amount_str, trigger_price, tag_prefix,
                status, attempts, max_attempts, delay_sec, last_error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                status=excluded.status,
                attempts=excluded.attempts,
                last_error=excluded.last_error,
                updated_at=excluded.updated_at,
                trigger_price=excluded.trigger_price
            """,
            (
                row.get("task_id"),
                row.get("symbol"),
                row.get("kind"),
                row.get("close_side"),
                row.get("amount_str"),
                row.get("trigger_price"),
                row.get("tag_prefix"),
                row.get("status"),
                int(row.get("attempts") or 0),
                int(row.get("max_attempts") or 0),
                float(row.get("delay_sec") or 0.0),
                row.get("last_error"),
                int(row.get("created_at") or int(time.time())),
                int(row.get("updated_at") or int(time.time())),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def _load_pending_protection_tasks() -> list[dict]:
    try:
        conn = sqlite3.connect(MAIN_TRADES_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM protection_tasks
            WHERE status IN ('running', 'failed')
              AND attempts < max_attempts
            ORDER BY updated_at ASC
            LIMIT 200
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows] if rows else []
    except Exception:
        return []

_init_protection_tables()

def _normalize_client_tag_symbol(symbol: str) -> str:
    """
    生成可用于 newClientOrderId 的稳定 symbol 标签，便于后续只撤销本系统创建的保护单。
    """
    return symbol.replace("/", "").replace(":", "_").replace("-", "_").upper()

def _extract_client_order_id(order: dict) -> str | None:
    """
    从 CCXT order 结构中提取 clientOrderId（不同交易所/CCXT 版本字段可能不同）。
    """
    if not isinstance(order, dict):
        return None
    if order.get("clientOrderId"):
        return str(order.get("clientOrderId"))
    info = order.get("info") if isinstance(order.get("info"), dict) else None
    if not isinstance(info, dict):
        return None
    for key in ("clientOrderId", "origClientOrderId", "newClientOrderId"):
        val = info.get(key)
        if val:
            return str(val)
    return None

# --- 动态加载风控阈值与安全开关 ---
RISK_CONFIG = {
    "MAX_ORDER_USD": Decimal(os.getenv("MAX_ORDER_USD", "5000")),
    "MAX_POSITION_USD": Decimal(os.getenv("MAX_POSITION_USD", "20000")),
    "MAX_SPREAD_PCT": Decimal(os.getenv("MAX_SPREAD_PCT", "0.005")),
    "VOLATILITY_THRESHOLD": Decimal("0.02"), # 波动率阈值 (2%)，超过则禁止市价单
    "DRY_RUN_MODE": os.getenv("DRY_RUN_MODE", "False").lower() in ("true", "1", "yes")
}

if RISK_CONFIG["DRY_RUN_MODE"]:
    logger.warning("⚠️ 当前处于 模拟盘(DRY RUN) 模式")
else:
    logger.info("🔥 当前处于 实盘(LIVE) 模式")

class ExecutionEngine:
    def __init__(self, exchange_id='binanceusdm'):
        self.exchange_id = exchange_id
        self.ex = None
        self._protection_resumed = False
        # 算法单配置
        self.twap_threshold = Decimal(os.getenv("TWAP_THRESHOLD_USD", "2000"))
        self.twap_chunk_size = Decimal(os.getenv("TWAP_CHUNK_USD", "500"))
        self.twap_interval_sec = int(os.getenv("TWAP_INTERVAL_SEC", 5))
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.secret = os.getenv("BINANCE_SECRET")
        
        if not self.api_key or not self.secret:
            logger.error("API Key or Secret not found in environment variables!")
            # 可以在这里抛出异常或在 init_exchange 时处理

    def _build_symbol_candidates(self, symbol: str) -> list[str]:
        """
        生成同一合约在 CCXT/Binance Futures 下的候选 symbol 列表。

        设计原因：
        - Web 端大多使用 `BTC/USDT`，但 Binance U 本位持仓在 CCXT 中经常表现为 `BTC/USDT:USDT`；
        - ROI 回撤风控触发 kill switch 时，如果 execution_mcp 只按单一格式查询，
          会出现“调用成功但未命中真实持仓”的假成功。
        """
        normalized_symbol = str(symbol or "").strip()
        if not normalized_symbol:
            return []

        candidates: list[str] = [normalized_symbol]
        if ":" in normalized_symbol:
            base_symbol = normalized_symbol.split(":", maxsplit=1)[0].strip()
            if base_symbol and base_symbol not in candidates:
                candidates.append(base_symbol)
        elif normalized_symbol.endswith("/USDT"):
            futures_symbol = f"{normalized_symbol}:USDT"
            if futures_symbol not in candidates:
                candidates.append(futures_symbol)

        return candidates

    def _normalize_symbol_identity(self, symbol: str) -> str:
        """
        将不同表现形式的 symbol 归一到同一比较键。

        设计原因：
        - `BTC/USDT` 与 `BTC/USDT:USDT` 本质指向同一条 USDT 永续；
        - kill switch、保护单更新与持仓查询必须共享统一的符号匹配口径，
          否则会在高波动阶段出现风控命中但执行层查不到仓位的问题。
        """
        normalized_symbol = str(symbol or "").strip()
        if not normalized_symbol:
            return ""
        normalized_symbol = normalized_symbol.split(":", maxsplit=1)[0]
        return (
            normalized_symbol.replace("/", "")
            .replace("-", "")
            .replace("_", "")
            .upper()
        )

    def _symbol_matches(self, left: str, right: str) -> bool:
        """
        判断两个 symbol 是否指向同一条合约。
        """
        left_key = self._normalize_symbol_identity(left)
        right_key = self._normalize_symbol_identity(right)
        return bool(left_key and right_key and left_key == right_key)

    async def _fetch_positions_for_symbol(self, symbol: str) -> tuple[list[dict], str]:
        """
        按 symbol 获取匹配持仓，并在必要时退化到全量扫描。

        设计原因：
        - Binance Futures 的 symbol 过滤在不同 CCXT 版本下表现并不稳定；
        - 当风控需要立即平仓时，必须支持 `fetch_positions(None)` 全量扫描，
          避免因为 `BTC/USDT` / `BTC/USDT:USDT` 口径差异漏掉真实仓位。
        """
        await self.init_exchange()
        candidates = self._build_symbol_candidates(symbol)
        matched_positions: list[dict] = []

        for candidate in candidates:
            try:
                positions = await self.ex.fetch_positions([candidate])
            except Exception as exc:
                logger.warning(f"按候选 symbol 查询持仓失败: requested={symbol} candidate={candidate} error={exc}")
                continue

            for pos in positions or []:
                pos_symbol = str(pos.get("symbol") or candidate)
                if self._symbol_matches(pos_symbol, symbol):
                    matched_positions.append(pos)

            if matched_positions:
                resolved_symbol = str(matched_positions[0].get("symbol") or candidate)
                if resolved_symbol != symbol:
                    logger.info(f"持仓 symbol 解析映射: requested={symbol} resolved={resolved_symbol}")
                return matched_positions, resolved_symbol

        try:
            positions = await self.ex.fetch_positions(None)
        except Exception as exc:
            logger.warning(f"全量扫描持仓失败: requested={symbol} error={exc}")
            fallback_symbol = candidates[0] if candidates else str(symbol or "").strip()
            return [], fallback_symbol

        for pos in positions or []:
            pos_symbol = str(pos.get("symbol") or "")
            if self._symbol_matches(pos_symbol, symbol):
                matched_positions.append(pos)

        resolved_symbol = (
            str(matched_positions[0].get("symbol"))
            if matched_positions
            else (candidates[0] if candidates else str(symbol or "").strip())
        )
        if matched_positions and resolved_symbol != symbol:
            logger.info(f"全量扫描命中持仓 symbol 映射: requested={symbol} resolved={resolved_symbol}")
        return matched_positions, resolved_symbol

    async def _fetch_open_orders_for_symbol(self, symbol: str) -> tuple[list[dict], str]:
        """
        按 symbol 获取匹配挂单，并在必要时退化到全量扫描。

        设计原因：
        - 风控平仓前必须先撤保护单，否则 ReduceOnly 数量可能被旧条件单占用；
        - 如果挂单查询也受 symbol 口径影响，就会让 kill switch 在 -2022 场景下持续失效。
        """
        await self.init_exchange()
        candidates = self._build_symbol_candidates(symbol)

        for candidate in candidates:
            try:
                orders = await self.ex.fetch_open_orders(candidate)
            except Exception as exc:
                logger.warning(f"按候选 symbol 查询挂单失败: requested={symbol} candidate={candidate} error={exc}")
                continue

            matched_orders = [
                order
                for order in (orders or [])
                if self._symbol_matches(str(order.get("symbol") or candidate), symbol)
            ]
            if matched_orders:
                resolved_symbol = str(matched_orders[0].get("symbol") or candidate)
                if resolved_symbol != symbol:
                    logger.info(f"挂单 symbol 解析映射: requested={symbol} resolved={resolved_symbol}")
                return matched_orders, resolved_symbol

        try:
            orders = await self.ex.fetch_open_orders()
        except Exception as exc:
            logger.warning(f"全量扫描挂单失败: requested={symbol} error={exc}")
            fallback_symbol = candidates[0] if candidates else str(symbol or "").strip()
            return [], fallback_symbol

        matched_orders = [
            order
            for order in (orders or [])
            if self._symbol_matches(str(order.get("symbol") or ""), symbol)
        ]
        resolved_symbol = (
            str(matched_orders[0].get("symbol"))
            if matched_orders
            else (candidates[0] if candidates else str(symbol or "").strip())
        )
        if matched_orders and resolved_symbol != symbol:
            logger.info(f"全量扫描命中挂单 symbol 映射: requested={symbol} resolved={resolved_symbol}")
        return matched_orders, resolved_symbol

    async def _cancel_twap_task(self, task_id: str, reason: str) -> None:
        """
        取消指定 TWAP 任务（用于反向信号/清仓等场景，避免后台切片继续对打与消耗手续费）。
        """
        task = TWAP_TASKS.get(task_id)
        if isinstance(task, dict):
            task["status"] = "cancelled"
            task["last_error"] = reason
            task["updated_at"] = int(time.time())

        async_task = TWAP_ASYNC_TASKS.get(task_id)
        if isinstance(async_task, asyncio.Task) and not async_task.done():
            async_task.cancel()

        TWAP_ASYNC_TASKS.pop(task_id, None)

    async def cancel_twap_for_symbol(self, symbol: str, reason: str) -> str | None:
        """
        按 symbol 取消当前在飞的 TWAP 任务（如存在），并清理映射。
        """
        async with _get_twap_control_lock():
            active_id = TWAP_ACTIVE_BY_SYMBOL.get(symbol)
            if not active_id:
                return None

            await self._cancel_twap_task(active_id, reason)
            if TWAP_ACTIVE_BY_SYMBOL.get(symbol) == active_id:
                TWAP_ACTIVE_BY_SYMBOL.pop(symbol, None)
            return active_id

    async def _has_managed_stop_loss(self, symbol: str, tag_prefix: str) -> bool:
        """
        判断交易所当前是否已经存在本系统创建的止损单，避免重试过程重复创建。
        """
        open_orders, _ = await self._fetch_open_orders_for_symbol(symbol)

        for o in open_orders or []:
            otype = str(o.get("type") or "").lower()
            info = o.get("info") if isinstance(o.get("info"), dict) else {}
            has_stop = ("stop" in otype) or ("stopPrice" in info) or ("stopprice" in {k.lower() for k in info.keys()})
            if not has_stop:
                continue
            client_id = _extract_client_order_id(o)
            if client_id and client_id.startswith(tag_prefix):
                return True
        return False

    async def _retry_create_stop_loss(
        self,
        symbol: str,
        close_side: str,
        amount_str: str,
        sl_price: float,
        tag_prefix: str,
        max_attempts: int,
        delay_sec: float,
        task_id: str | None = None,
    ) -> None:
        task_id = task_id or f"{tag_prefix}_pending_sl"
        PROTECTION_TASKS[task_id] = {
            "task_id": task_id,
            "symbol": symbol,
            "type": "stop_loss",
            "status": "running",
            "attempts": 0,
            "last_error": None,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
        _upsert_protection_task(
            {
                "task_id": task_id,
                "symbol": symbol,
                "kind": "stop_loss",
                "close_side": close_side,
                "amount_str": amount_str,
                "trigger_price": float(sl_price),
                "tag_prefix": tag_prefix,
                "status": "running",
                "attempts": 0,
                "max_attempts": int(max_attempts),
                "delay_sec": float(delay_sec),
                "last_error": None,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            }
        )

        for attempt in range(1, max(1, max_attempts) + 1):
            PROTECTION_TASKS[task_id]["attempts"] = attempt
            PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
            _upsert_protection_task(
                {
                    "task_id": task_id,
                    "symbol": symbol,
                    "kind": "stop_loss",
                    "close_side": close_side,
                    "amount_str": amount_str,
                    "trigger_price": float(sl_price),
                    "tag_prefix": tag_prefix,
                    "status": "running",
                    "attempts": attempt,
                    "max_attempts": int(max_attempts),
                    "delay_sec": float(delay_sec),
                    "last_error": PROTECTION_TASKS[task_id].get("last_error"),
                    "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                    "updated_at": int(time.time()),
                }
            )

            if await self._has_managed_stop_loss(symbol, tag_prefix):
                PROTECTION_TASKS[task_id]["status"] = "completed"
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "stop_loss",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(sl_price),
                        "tag_prefix": tag_prefix,
                        "status": "completed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": None,
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                return

            try:
                sl_price_str = self.ex.price_to_precision(symbol, float(safe_decimal(sl_price)))
                client_id = f"{tag_prefix}_{int(time.time() * 1000)}"
                await self.ex.create_order(
                    symbol=symbol,
                    type="STOP_MARKET",
                    side=close_side,
                    amount=float(amount_str),
                    params={"stopPrice": float(sl_price_str), "reduceOnly": True, "newClientOrderId": client_id},
                )
                PROTECTION_TASKS[task_id]["status"] = "completed"
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "stop_loss",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(sl_price),
                        "tag_prefix": tag_prefix,
                        "status": "completed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": None,
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                return
            except Exception as e:
                PROTECTION_TASKS[task_id]["last_error"] = str(e)
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "stop_loss",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(sl_price),
                        "tag_prefix": tag_prefix,
                        "status": "running" if attempt < max_attempts else "failed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": str(e),
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                if attempt < max_attempts:
                    await asyncio.sleep(max(0.1, delay_sec))
                    continue
                PROTECTION_TASKS[task_id]["status"] = "failed"
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "stop_loss",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(sl_price),
                        "tag_prefix": tag_prefix,
                        "status": "failed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": str(e),
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                return

    async def _has_managed_take_profit(self, symbol: str, tag_prefix: str) -> bool:
        open_orders, _ = await self._fetch_open_orders_for_symbol(symbol)
        for o in open_orders or []:
            otype = str(o.get("type") or "").lower()
            info = o.get("info") if isinstance(o.get("info"), dict) else {}
            has_tp = ("take_profit" in otype) or ("takeprofit" in otype) or ("stopPrice" in info) or ("stopprice" in {k.lower() for k in info.keys()})
            if not has_tp:
                continue
            client_id = _extract_client_order_id(o)
            if client_id and client_id.startswith(tag_prefix):
                return True
        return False

    async def _retry_create_take_profit(
        self,
        symbol: str,
        close_side: str,
        amount_str: str,
        tp_price: float,
        tag_prefix: str,
        max_attempts: int,
        delay_sec: float,
        task_id: str | None = None,
    ) -> None:
        task_id = task_id or f"{tag_prefix}_pending_tp"
        PROTECTION_TASKS[task_id] = {
            "task_id": task_id,
            "symbol": symbol,
            "type": "take_profit",
            "status": "running",
            "attempts": 0,
            "last_error": None,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
        _upsert_protection_task(
            {
                "task_id": task_id,
                "symbol": symbol,
                "kind": "take_profit",
                "close_side": close_side,
                "amount_str": amount_str,
                "trigger_price": float(tp_price),
                "tag_prefix": tag_prefix,
                "status": "running",
                "attempts": 0,
                "max_attempts": int(max_attempts),
                "delay_sec": float(delay_sec),
                "last_error": None,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            }
        )

        for attempt in range(1, max(1, max_attempts) + 1):
            PROTECTION_TASKS[task_id]["attempts"] = attempt
            PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
            _upsert_protection_task(
                {
                    "task_id": task_id,
                    "symbol": symbol,
                    "kind": "take_profit",
                    "close_side": close_side,
                    "amount_str": amount_str,
                    "trigger_price": float(tp_price),
                    "tag_prefix": tag_prefix,
                    "status": "running",
                    "attempts": attempt,
                    "max_attempts": int(max_attempts),
                    "delay_sec": float(delay_sec),
                    "last_error": PROTECTION_TASKS[task_id].get("last_error"),
                    "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                    "updated_at": int(time.time()),
                }
            )

            if await self._has_managed_take_profit(symbol, tag_prefix):
                PROTECTION_TASKS[task_id]["status"] = "completed"
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "take_profit",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(tp_price),
                        "tag_prefix": tag_prefix,
                        "status": "completed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": None,
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                return

            try:
                tp_price_str = self.ex.price_to_precision(symbol, float(safe_decimal(tp_price)))
                client_id = f"{tag_prefix}_{int(time.time() * 1000)}"
                await self.ex.create_order(
                    symbol=symbol,
                    type="TAKE_PROFIT_MARKET",
                    side=close_side,
                    amount=float(amount_str),
                    params={"stopPrice": float(tp_price_str), "reduceOnly": True, "newClientOrderId": client_id},
                )
                PROTECTION_TASKS[task_id]["status"] = "completed"
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "take_profit",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(tp_price),
                        "tag_prefix": tag_prefix,
                        "status": "completed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": None,
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                return
            except Exception as e:
                PROTECTION_TASKS[task_id]["last_error"] = str(e)
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "take_profit",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(tp_price),
                        "tag_prefix": tag_prefix,
                        "status": "running" if attempt < max_attempts else "failed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": str(e),
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                if attempt < max_attempts:
                    await asyncio.sleep(max(0.1, delay_sec))
                    continue
                PROTECTION_TASKS[task_id]["status"] = "failed"
                PROTECTION_TASKS[task_id]["updated_at"] = int(time.time())
                _upsert_protection_task(
                    {
                        "task_id": task_id,
                        "symbol": symbol,
                        "kind": "take_profit",
                        "close_side": close_side,
                        "amount_str": amount_str,
                        "trigger_price": float(tp_price),
                        "tag_prefix": tag_prefix,
                        "status": "failed",
                        "attempts": attempt,
                        "max_attempts": int(max_attempts),
                        "delay_sec": float(delay_sec),
                        "last_error": str(e),
                        "created_at": PROTECTION_TASKS[task_id].get("created_at"),
                        "updated_at": int(time.time()),
                    }
                )
                return

    def _new_twap_task(self, symbol: str, side: str, total_amount_usd: Decimal, chunk_usd: Decimal) -> str:
        """
        创建一个可审计的 TWAP 任务记录，便于网关层展示“已下单/已执行到哪一步”，避免误报。
        """
        task_id = f"twap_{int(time.time() * 1000)}"
        TWAP_TASKS[task_id] = {
            "task_id": task_id,
            "symbol": symbol,
            "side": side,
            "total_amount_usd": str(total_amount_usd),
            "chunk_usd": str(chunk_usd),
            "remaining_usd": str(total_amount_usd),
            "status": "running",
            "orders": [],
            "slice_count": 0,
            "last_error": None,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
        return task_id

    async def _execute_twap_slice(self, symbol: str, side: str, slice_usd: Decimal, task_id: str, slice_index: int) -> dict:
        """
        执行单次 TWAP 切片，并在成功/失败时同步更新影子账本与任务状态。
        """
        await self.init_exchange()

        order_id_mock = f"{task_id}_{slice_index}"
        if side == 'buy':
            balance_info = await self.get_balance()
            await execution_shadow_ledger.sync_with_exchange_free(Decimal(str(balance_info['free'])))
            if not await execution_shadow_ledger.pre_deduct(order_id_mock, symbol, slice_usd):
                raise RuntimeError(f"TWAP 影子账本余额不足，无法执行切片 ${slice_usd}")

        ticker = await self.ex.fetch_ticker(symbol)
        price = safe_decimal(ticker['last'])
        amount_dec = slice_usd / price
        amount_str = self.ex.amount_to_precision(symbol, float(amount_dec))
        order = await self.ex.create_market_order(symbol, side, float(amount_str))

        filled_usd = safe_decimal(order.get('cost', slice_usd))
        if side == 'buy':
            await execution_shadow_ledger.reconcile(order_id_mock, filled_usd, "FILLED")

        order_record = {
            "slice_index": slice_index,
            "order_id": order.get("id"),
            "status": order.get("status"),
            "filled_qty": str(order.get("filled", amount_str)),
            "average_price": str(order.get("average") or price),
            "cost_usd": str(filled_usd),
        }

        task = TWAP_TASKS.get(task_id)
        if isinstance(task, dict):
            task["orders"].append(order_record)
            task["slice_count"] = slice_index
            remaining_usd = safe_decimal(task.get("remaining_usd", "0")) - slice_usd
            task["remaining_usd"] = str(max(Decimal("0"), remaining_usd))
            task["updated_at"] = int(time.time())

        return order_record

    def _extract_usdt_balance(self, balance: dict) -> dict:
        usdt_direct = balance.get('USDT', {})
        total_balance = usdt_direct.get('total')
        free_balance = usdt_direct.get('free')
        used_balance = usdt_direct.get('used')

        if total_balance is None:
            total_balance = balance.get('total', {}).get('USDT', "0.0")
        if free_balance is None:
            free_balance = balance.get('free', {}).get('USDT', "0.0")
        if used_balance is None:
            used_balance = balance.get('used', {}).get('USDT', "0.0")

        return {
            "asset": "USDT",
            "total": str(safe_decimal(total_balance)),
            "free": str(safe_decimal(free_balance)),
            "used": str(safe_decimal(used_balance))
        }

    def _do_init_sync(self):
        """后台线程中执行沉重的初始化 (同步)"""
        import ccxt
        if not self.api_key or not self.secret:
             raise ValueError("Missing API credentials")
             
        exchange_config = {
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'} # 默认合约交易
        }
        
        # 动态加载代理
        proxy_url = os.getenv("CCXT_PROXY")
        if proxy_url:
            exchange_config['proxies'] = {
                'http': proxy_url,
                'https': proxy_url
            }
             
        # 实例化 ccxt 异步引擎 (虽然是异步类，但实例化过程包含复杂的正则预编译和配置加载)
        ex = getattr(ccxt.async_support, self.exchange_id)(exchange_config)
        
        # 如果是 Testnet，开启沙盒模式
        if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
            ex.set_sandbox_mode(True)
            
        return ex

    async def init_exchange(self):
        if not self.ex:
            logger.info("⏳ [Execution Gateway] 正在后台线程执行沉重的引擎初始化...")
            self.ex = await asyncio.to_thread(self._do_init_sync)
            
            # 异步加载市场数据
            logger.info("⏳ [Execution Gateway] 正在后台任务中加载市场数据 (load_markets)...")
            await asyncio.create_task(self.ex.load_markets())
            logger.info("✅ Execution-MCP 已启动，事件循环无阻塞，市场数据已加载。")

        if not self._protection_resumed:
            self._protection_resumed = True
            asyncio.create_task(self.resume_protection_tasks())

    async def resume_protection_tasks(self) -> None:
        await self.init_exchange()
        tasks = _load_pending_protection_tasks()
        for row in tasks:
            kind = str(row.get("kind") or "")
            if kind == "stop_loss":
                asyncio.create_task(
                    self._retry_create_stop_loss(
                        symbol=str(row.get("symbol")),
                        close_side=str(row.get("close_side")),
                        amount_str=str(row.get("amount_str")),
                        sl_price=float(row.get("trigger_price") or 0.0),
                        tag_prefix=str(row.get("tag_prefix")),
                        max_attempts=int(row.get("max_attempts") or 0),
                        delay_sec=float(row.get("delay_sec") or 1.0),
                        task_id=str(row.get("task_id")),
                    )
                )
            elif kind == "take_profit":
                asyncio.create_task(
                    self._retry_create_take_profit(
                        symbol=str(row.get("symbol")),
                        close_side=str(row.get("close_side")),
                        amount_str=str(row.get("amount_str")),
                        tp_price=float(row.get("trigger_price") or 0.0),
                        tag_prefix=str(row.get("tag_prefix")),
                        max_attempts=int(row.get("max_attempts") or 0),
                        delay_sec=float(row.get("delay_sec") or 1.0),
                        task_id=str(row.get("task_id")),
                    )
                )

    async def validate_risk(
        self,
        symbol: str,
        amount_usd: Decimal,
        order_type: str = 'market',
        volatility: float = 0.0,
        side: str = None,
        is_algo_order: bool = False,
    ):
        """
        硬核风控检查
        """
        volatility_dec = safe_decimal(volatility)
        # 1. 检查单笔金额
        if (not is_algo_order) and amount_usd > RISK_CONFIG["MAX_ORDER_USD"]:
            return False, f"订单金额 ${amount_usd} 超过限额 ${RISK_CONFIG['MAX_ORDER_USD']}"
        
        # 2. 检查波动率限制 (如果波动率过大，禁止市价单)
        if volatility_dec > RISK_CONFIG["VOLATILITY_THRESHOLD"] and order_type == 'market':
             return False, f"当前波动率 ({volatility_dec:.2%}) 过高，禁止执行市价单，请改用限价单 (Limit Order)。"

        # 3. 检查最大持仓价值 (防穿仓) & 动态风险预算 (Dynamic Risk Budget)
        try:
            # 动态风险预算：获取当前账户总权益
            balance_info = await self.get_balance()
            total_equity = safe_decimal(balance_info['total'])
            
            # 动态敞口上限：默认基础限额，如果账户盈利且总权益增厚，允许放宽至总权益的 2 倍（2x 杠杆敞口）
            dynamic_max_position = max(RISK_CONFIG["MAX_POSITION_USD"], total_equity * Decimal("2.0"))
            
            positions, _ = await self._fetch_positions_for_symbol(symbol)
            current_long_notional = Decimal("0")
            current_short_notional = Decimal("0")
            
            for pos in positions:
                size = safe_decimal(pos.get('contracts', 0))
                mark_price = safe_decimal(pos.get('markPrice', 0))
                pos_side = pos.get('side', '').lower()
                
                if size > Decimal("0") and mark_price > Decimal("0"):
                    notional = size * mark_price
                    if pos_side == 'long':
                        current_long_notional += notional
                    elif pos_side == 'short':
                        current_short_notional += notional
                    else:
                        # 兼容 net 模式或未指定方向，默认当作增加总敞口
                        current_long_notional += notional
            
            # 预估下单后的持仓价值
            if side:
                side = side.lower()
                if side == 'buy':
                    # 如果当前有空单，buy 会优先平空单，多出部分算多单开仓
                    if current_short_notional > 0:
                        new_short = max(Decimal("0"), current_short_notional - amount_usd)
                        new_long = current_long_notional + max(Decimal("0"), amount_usd - current_short_notional)
                        estimated_notional = new_long + new_short
                    else:
                        estimated_notional = current_long_notional + amount_usd
                elif side == 'sell':
                    # 如果当前有多单，sell 会优先平多单，多出部分算空单开仓
                    if current_long_notional > 0:
                        new_long = max(Decimal("0"), current_long_notional - amount_usd)
                        new_short = current_short_notional + max(Decimal("0"), amount_usd - current_long_notional)
                        estimated_notional = new_long + new_short
                    else:
                        estimated_notional = current_short_notional + amount_usd
                else:
                    estimated_notional = current_long_notional + current_short_notional + amount_usd
            else:
                estimated_notional = current_long_notional + current_short_notional + amount_usd
                    
            if estimated_notional > dynamic_max_position:
                return False, f"开仓后总持仓价值 (${estimated_notional}) 将超过动态风险限额 ${dynamic_max_position} (当前基础限额 ${RISK_CONFIG['MAX_POSITION_USD']})"
        except Exception as e:
            logger.warning(f"获取持仓或权益失败，略过持仓限制检查: {e}")

        # 4. 检查盘口流动性 (防滑点)
        try:
            orderbook = await self.ex.fetch_order_book(symbol, limit=5)
            if not orderbook or not orderbook.get('asks') or not orderbook.get('bids'):
                 return False, "无法获取完整的盘口数据(asks/bids为空)"
                 
            best_ask = safe_decimal(orderbook['asks'][0][0])
            best_bid = safe_decimal(orderbook['bids'][0][0])
            
            spread = (best_ask - best_bid) / best_bid
            if spread > RISK_CONFIG["MAX_SPREAD_PCT"]:
                return False, f"价差过大 ({spread:.4%}), 超过阈值 {RISK_CONFIG['MAX_SPREAD_PCT']:.2%}, 放弃执行"
                
            return True, "Success"
        except Exception as e:
            return False, f"风控检查时发生错误: {str(e)}"

    async def get_balance(self):
        await self.init_exchange()
        balance = await self.ex.fetch_balance()
        return self._extract_usdt_balance(balance)
    
    async def cancel_all_orders(self, symbol: str):
        """
        取消指定 symbol 的全部挂单，并自动适配 Binance Futures 的 symbol 变体。
        """
        await self.init_exchange()
        open_orders, resolved_symbol = await self._fetch_open_orders_for_symbol(symbol)
        target_symbols = [resolved_symbol]
        for candidate in self._build_symbol_candidates(symbol):
            if candidate not in target_symbols:
                target_symbols.append(candidate)

        last_exc: Exception | None = None
        for candidate in target_symbols:
            try:
                await self.ex.cancel_all_orders(candidate)
                if candidate != symbol:
                    logger.info(f"撤单 symbol 解析映射: requested={symbol} resolved={candidate}")
                return
            except Exception as exc:
                last_exc = exc

        if open_orders and last_exc is not None:
            raise last_exc

    def _extract_position_side_param(self, position: dict) -> str | None:
        """
        提取 Binance 对冲模式需要的 positionSide 参数。
        """
        raw_info = position.get("info") if isinstance(position.get("info"), dict) else {}
        raw_position_side = str(
            raw_info.get("positionSide")
            or position.get("positionSide")
            or position.get("position_side")
            or ""
        ).upper()
        if raw_position_side in {"LONG", "SHORT"}:
            return raw_position_side

        side = str(position.get("side") or "").lower()
        if side == "long":
            return "LONG"
        if side == "short":
            return "SHORT"
        return None

    def _is_reduce_only_not_required_error(self, message: str) -> bool:
        """
        判断交易所是否返回了“reduceOnly 参数不需要”的语义错误。

        设计原因：
        - Binance 某些平仓路径下会返回 -1106，表示当前请求场景无需显式传 reduceOnly；
        - 如果直接把该错误当成失败，会导致 kill switch 在明明可平仓的情况下被误判为异常。
        """
        normalized = str(message or "").lower()
        return (
            "-1106" in normalized
            and "reduceonly" in normalized
            and "not required" in normalized
        )

    def _is_reduce_only_rejected_error(self, message: str) -> bool:
        """
        判断交易所是否返回了 ReduceOnly 被拒绝的错误。

        设计原因：
        - -2022 常见于旧保护单占用数量、对冲模式参数不完整等场景；
        - 将识别逻辑单独抽出，便于主平仓流程和后续重试逻辑共用。
        """
        normalized = str(message or "").lower()
        return "-2022" in normalized or "reduceonly order is rejected" in normalized

    async def _submit_market_close_order(
        self,
        symbol: str,
        close_side: str,
        amount_str: str,
        position_side: str | None,
        allow_reduce_only_retry: bool = True,
    ) -> tuple[dict, str]:
        """
        提交市价平仓单，并在交易所返回 -1106 时自动去掉 reduceOnly 重试。

        设计原因：
        - kill switch 的目标是优先清仓，而不是执着于固定参数组合；
        - 只有识别到交易所明确声明“reduceOnly 不需要”时，才执行一次无 reduceOnly 重试，
          避免把其他真正的交易所错误误判成可恢复异常。
        """
        primary_params: dict[str, object] = {"reduceOnly": True}
        if position_side:
            primary_params["positionSide"] = position_side

        try:
            order = await self.ex.create_market_order(
                symbol,
                close_side,
                float(amount_str),
                params=primary_params,
            )
            return order, "reduce_only"
        except Exception as exc:
            message = str(exc)
            if not allow_reduce_only_retry or not self._is_reduce_only_not_required_error(message):
                raise

            retry_params = {
                "positionSide": position_side,
            } if position_side else {}
            logger.warning(
                "⚠️ %s 平仓遇到 -1106 reduceOnly not required，自动改为无 reduceOnly 重试。"
                " close_side=%s amount=%s position_side=%s",
                symbol,
                close_side,
                amount_str,
                position_side or "NONE",
            )
            order = await self.ex.create_market_order(
                symbol,
                close_side,
                float(amount_str),
                params=retry_params,
            )
            return order, "no_reduce_only_retry"

    async def _cancel_protection_orders(self, symbol: str) -> int:
        """
        只清理保护性质的条件单，避免 ReduceOnly 数量被旧保护单占用。
        """
        open_orders, resolved_symbol = await self._fetch_open_orders_for_symbol(symbol)

        cancelled = 0
        for order in open_orders or []:
            info = order.get("info") if isinstance(order.get("info"), dict) else {}
            order_type = str(order.get("type") or info.get("type") or "").upper()
            client_id = _extract_client_order_id(order) or ""
            reduce_only_flag = str(
                order.get("reduceOnly")
                or info.get("reduceOnly")
                or info.get("closePosition")
                or ""
            ).lower() in {"true", "1"}
            is_protection_order = (
                reduce_only_flag
                or order_type in {"STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"}
                or client_id.startswith("TM_SL_")
                or client_id.startswith("TM_TP_")
            )
            if not is_protection_order:
                continue

            order_id = order.get("id")
            if not order_id:
                continue
            try:
                await self.ex.cancel_order(order_id, resolved_symbol)
                cancelled += 1
            except Exception as exc:
                logger.warning(f"取消保护单失败 {resolved_symbol} order_id={order_id}: {exc}")
        return cancelled

    async def _close_position_with_fallback(self, symbol: str, position: dict) -> dict:
        """
        对单个仓位执行带兜底的平仓逻辑。

        设计原因：
        - Binance 在存在残余保护单或账户处于对冲模式时，ReduceOnly 市价单可能返回 -2022；
        - 当出现该错误时，先确保撤掉保护单，再带 positionSide 重试一次，避免风控触发后仓位滞留；
        - 另外兼容 -1106 reduceOnly not required，在交易所明确声明无需该参数时自动无 reduceOnly 重试。
        """
        size = safe_decimal(position.get("contracts", 0))
        if size <= Decimal("0"):
            return {"status": "skipped", "message": f"{symbol} 无需平仓"}

        side = str(position.get("side") or "").lower()
        close_side = "sell" if side == "long" else "buy"
        amount_str = self.ex.amount_to_precision(symbol, float(size))
        position_side = self._extract_position_side_param(position)

        try:
            order, submit_mode = await self._submit_market_close_order(
                symbol=symbol,
                close_side=close_side,
                amount_str=amount_str,
                position_side=position_side,
            )
            return {
                "status": "success",
                "mode": submit_mode,
                "order_id": order.get("id"),
                "amount": amount_str,
                "close_side": close_side,
                "reduce_only_applied": submit_mode == "reduce_only",
            }
        except Exception as exc:
            message = str(exc)
            if not self._is_reduce_only_rejected_error(message):
                raise

            logger.warning(
                f"⚠️ {symbol} ReduceOnly 平仓被拒绝，尝试执行保护单清理 + positionSide 兜底重试: {message}"
            )
            await self._cancel_protection_orders(symbol)
            await asyncio.sleep(0.2)

            refreshed_positions, resolved_symbol = await self._fetch_positions_for_symbol(symbol)
            refreshed_position = None
            for pos in refreshed_positions or []:
                if self._symbol_matches(str(pos.get("symbol") or ""), resolved_symbol) and safe_decimal(pos.get("contracts", 0)) > Decimal("0"):
                    refreshed_position = pos
                    break

            if not refreshed_position:
                return {
                    "status": "success",
                    "mode": "post_reject_refresh",
                    "message": f"{symbol} 在 -2022 后刷新持仓时已无仓位，视为已清空。",
                }

            fallback_position_side = self._extract_position_side_param(refreshed_position)
            refreshed_size = safe_decimal(refreshed_position.get("contracts", 0))
            execution_symbol = str(refreshed_position.get("symbol") or resolved_symbol or symbol)
            refreshed_amount_str = self.ex.amount_to_precision(execution_symbol, float(refreshed_size))
            fallback_side = str(refreshed_position.get("side") or "").lower()
            fallback_close_side = "sell" if fallback_side == "long" else "buy"
            order, submit_mode = await self._submit_market_close_order(
                symbol=execution_symbol,
                close_side=fallback_close_side,
                amount_str=refreshed_amount_str,
                position_side=fallback_position_side,
            )
            return {
                "status": "success",
                "mode": (
                    "position_side_fallback"
                    if submit_mode == "reduce_only"
                    else "position_side_no_reduce_only_fallback"
                ),
                "order_id": order.get("id"),
                "amount": refreshed_amount_str,
                "close_side": fallback_close_side,
                "reduce_only_applied": submit_mode == "reduce_only",
            }

    async def close_all_positions(self, symbol: str):
        """
        市价平掉指定 symbol 的所有持仓
        """
        await self.init_exchange()
        try:
            await self.cancel_all_orders(symbol)
        except Exception as exc:
            logger.warning(f"取消 {symbol} 全部挂单失败，继续执行持仓清理: {exc}")
        await self._cancel_protection_orders(symbol)
        positions, resolved_symbol = await self._fetch_positions_for_symbol(symbol)
        
        closed_count = 0
        for position in positions:
            size = safe_decimal(position.get('contracts', 0))
            if size > Decimal("0"):
                execution_symbol = str(position.get("symbol") or resolved_symbol or symbol)
                close_result = await self._close_position_with_fallback(execution_symbol, position)
                logger.info(f"✅ {execution_symbol} 平仓结果: {close_result}")
                closed_count += 1
                
        return closed_count

    async def sync_trailing_stop(self, symbol: str, trailing_pct: float = 0.05, roe_activation: float = 0.05) -> list:
        """
        追踪止损联动 (Trailing Stop-Loss):
        为已有盈利的仓位自动更新/挂出止损单，以保护既得利润。
        采用绝对价格回撤与 ROE 混合机制。
        
        :param trailing_pct: 价格回撤容忍百分比 (如 0.05 表示最高点回落 5% 触发止损)
        :param roe_activation: ROE 激活阈值 (如 0.05 表示 ROE 达到 5% 才激活追踪止损)
        :return: 更新的止损单详情列表
        """
        await self.init_exchange()
        positions, resolved_symbol = await self._fetch_positions_for_symbol(symbol)
        updated_sl_orders = []
        
        for pos in positions:
            size = safe_decimal(pos.get('contracts', 0))
            if size <= Decimal("0"):
                continue
                
            execution_symbol = str(pos.get("symbol") or resolved_symbol or symbol)
            side = pos.get('side', '').lower()
            mark_price = safe_decimal(pos.get('markPrice', 0))
            entry_price = safe_decimal(pos.get('entryPrice', pos.get('entry_price', 0)))
            unrealized_pnl = safe_decimal(pos.get('unrealizedPnl', 0))
            initial_margin = safe_decimal(pos.get('initialMargin', pos.get('initial_margin', 0)))
            leverage = safe_decimal(pos.get('leverage', 1))
            
            # 计算 Binance 同步真实 ROE
            binance_roe = Decimal("0")
            if initial_margin > Decimal("0"):
                binance_roe = unrealized_pnl / initial_margin
                
            # 计算系统本地推算 ROE
            system_roe = Decimal("0")
            if entry_price > Decimal("0"):
                if side == 'long':
                    system_roe = (mark_price - entry_price) / entry_price * leverage
                elif side == 'short':
                    system_roe = (entry_price - mark_price) / entry_price * leverage
                    
            logger.info(f"📊 {symbol} 仓位 ROE 混合分析 -> 系统推算 ROE: {system_roe:.4%}, Binance 真实 ROE: {binance_roe:.4%}")
            
            # 取两者较优值作为激活判定标准
            effective_roe = max(system_roe, binance_roe)
            
            # 只有在有浮盈，且盈利率达到了激活阈值 (ROE >= roe_activation) 且价格有效的情况下，才启动或上调追踪止损
            if unrealized_pnl > Decimal("0") and effective_roe >= safe_decimal(roe_activation) and mark_price > Decimal("0") and entry_price > Decimal("0"):
                trailing_dec = safe_decimal(trailing_pct)
                amount_str = self.ex.amount_to_precision(execution_symbol, float(size))
                
                new_sl_price = Decimal("0")
                close_side = 'sell'
                
                # 多单：价格从高点回落触发绝对价格止损
                if side == 'long':
                    # 假定 mark_price 是近期的局部高点，SL = mark_price * (1 - 回撤)
                    new_sl_price = mark_price * (Decimal("1.0") - trailing_dec)
                    # 底线：不应低于开仓价，保证“赢来的钱不吐回去” (稍微留点利润如 0.5% 覆盖手续费)
                    breakeven_price = entry_price * Decimal("1.005")
                    new_sl_price = max(new_sl_price, breakeven_price)
                    close_side = 'sell'
                    
                # 空单：价格从低点反弹触发绝对价格止损
                elif side == 'short':
                    # SL = mark_price * (1 + 回撤)
                    new_sl_price = mark_price * (Decimal("1.0") + trailing_dec)
                    breakeven_price = entry_price * Decimal("0.995")
                    new_sl_price = min(new_sl_price, breakeven_price)
                    close_side = 'buy'
                
                if new_sl_price > Decimal("0"):
                    price_str = self.ex.price_to_precision(execution_symbol, float(new_sl_price))
                    
                    if RISK_CONFIG["DRY_RUN_MODE"]:
                        logger.info(f"🛡️ [DRY-RUN] 模拟挂出混合驱动追踪止损单: {close_side} {execution_symbol} {amount_str} @ 触发价 {price_str} (当前综合ROE {effective_roe:.2%})")
                        updated_sl_orders.append({"symbol": execution_symbol, "side": close_side, "amount": amount_str, "sl_price": price_str, "mode": "dry_run", "system_roe": float(system_roe), "binance_roe": float(binance_roe)})
                    else:
                        try:
                            # 真实环境：撤销现有的所有条件单，重新挂载
                            await self.cancel_all_orders(execution_symbol)
                            
                            # 根据交易所类型，发送止损单。以下为币安 U本位合约的通用 Stop Market 单
                            sl_params = {
                                'stopPrice': float(price_str),
                                'reduceOnly': True
                            }
                            order = await self.ex.create_order(execution_symbol, 'STOP_MARKET', close_side, float(amount_str), None, sl_params)
                            
                            # 将 ROE 数据一并返回给调用方（如 Agent 或网关）
                            order['system_roe'] = float(system_roe)
                            order['binance_roe'] = float(binance_roe)
                            
                            logger.info(f"✅ 成功更新混合追踪止损单: {close_side} {execution_symbol} {amount_str} @ 触发价 {price_str} (当前综合ROE {effective_roe:.2%})")
                            updated_sl_orders.append(order)
                        except Exception as e:
                            logger.error(f"❌ 更新混合追踪止损单失败: {e}")
                            
        return updated_sl_orders

    async def run_twap_background(self, task_id: str, symbol: str, side: str, remaining_usd: Decimal, start_slice_index: int = 0):
        """[核心算法] 后台静默执行 TWAP 时间加权算法单"""
        effective_chunk_usd = min(self.twap_chunk_size, RISK_CONFIG["MAX_ORDER_USD"])
        logger.info(f"🚀 启动 TWAP 算法引擎 [{task_id}]: 剩余 ${remaining_usd}, 每刀 ${effective_chunk_usd}, 间隔 {self.twap_interval_sec}s")

        slice_index = start_slice_index
        try:
            while remaining_usd > Decimal("0"):
                task = TWAP_TASKS.get(task_id)
                if isinstance(task, dict) and task.get("status") not in {"running"}:
                    logger.warning(f"🛑 TWAP [{task_id}] 已被取消/终止，停止后台切片。status={task.get('status')}")
                    break

                slice_index += 1
                current_slice_usd = min(effective_chunk_usd, remaining_usd)
                logger.info(f"⏳ TWAP [{task_id}] 第 {slice_index} 刀: 准备 {side} {symbol} ${current_slice_usd} (剩余未执行: ${remaining_usd})")

                if RISK_CONFIG["DRY_RUN_MODE"]:
                    if side == 'buy':
                        order_id_mock = f"{task_id}_{slice_index}"
                        await execution_shadow_ledger.reconcile(order_id_mock, current_slice_usd, "FILLED")
                    task = TWAP_TASKS.get(task_id)
                    if isinstance(task, dict):
                        task["slice_count"] = slice_index
                        task["remaining_usd"] = str(max(Decimal("0"), safe_decimal(task.get("remaining_usd", "0")) - current_slice_usd))
                        task["updated_at"] = int(time.time())
                    remaining_usd -= current_slice_usd
                else:
                    try:
                        await self._execute_twap_slice(symbol, side, current_slice_usd, task_id, slice_index)
                        remaining_usd -= current_slice_usd
                    except Exception as e:
                        logger.error(f"❌ TWAP [{task_id}] 切片执行失败: {e}")
                        task = TWAP_TASKS.get(task_id)
                        if isinstance(task, dict):
                            task["status"] = "failed"
                            task["last_error"] = str(e)
                            task["updated_at"] = int(time.time())
                        break

                if remaining_usd > Decimal("0"):
                    await asyncio.sleep(self.twap_interval_sec)

            task = TWAP_TASKS.get(task_id)
            if isinstance(task, dict) and task.get("status") not in {"failed", "cancelled"}:
                task["status"] = "completed"
                task["updated_at"] = int(time.time())
            logger.info(f"🏁 TWAP 算法单 [{task_id}] 执行结束，状态={TWAP_TASKS.get(task_id, {}).get('status')}")
            balance_cache.clear()
        except asyncio.CancelledError:
            task = TWAP_TASKS.get(task_id)
            if isinstance(task, dict):
                task["status"] = "cancelled"
                task["updated_at"] = int(time.time())
            raise
        except Exception as e:
            logger.error(f"❌ TWAP [{task_id}] 异常中断: {e}")
            task = TWAP_TASKS.get(task_id)
            if isinstance(task, dict):
                task["status"] = "failed"
                task["last_error"] = str(e)
                task["updated_at"] = int(time.time())
        finally:
            async with _get_twap_control_lock():
                active_id = TWAP_ACTIVE_BY_SYMBOL.get(symbol)
                if active_id == task_id:
                    TWAP_ACTIVE_BY_SYMBOL.pop(symbol, None)
            TWAP_ASYNC_TASKS.pop(task_id, None)

engine = ExecutionEngine()

@mcp.tool()
async def get_account_balance() -> str:
    """
    [Agent 工具] 获取账户当前的资金情况 (USDT)。
    (注：为了解决 asyncache 对无参函数的 cache_key_builder 传参错误，已暂时移除 @cached，
     如果后续遇到限频，可改用内部类变量实现简单的 TTL)
    """
    logger.info("🌐 正在向交易所请求真实账户余额...")
    try:
        await engine.init_exchange()
        balance = await engine.get_balance()
        logger.info(f"查询余额成功: {balance}")
        return json.dumps({"status": "success", "data": balance}, ensure_ascii=False)
    except Exception as e:
        logger.exception("查询余额失败")
        return MCPErrorResponse(status="error", error_code="BALANCE_QUERY_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def get_twap_task_status(task_id: str) -> str:
    """
    查询 TWAP 任务状态，用于网关层/审计台确认“是否真实产生了交易所订单”与当前进度。
    """
    task = TWAP_TASKS.get(task_id)
    if not isinstance(task, dict):
        return MCPErrorResponse(status="error", error_code="TWAP_TASK_NOT_FOUND", message=f"TWAP task not found: {task_id}").model_dump_json()
    return json.dumps({"status": "success", "data": task}, ensure_ascii=False)

@mcp.tool()
async def get_active_twap_tasks(symbol: str = "") -> str:
    """
    返回当前在飞 TWAP 任务（用于观测与审计；不建议策略层依赖该接口做交易决策）。
    """
    normalized = str(symbol or "").strip()
    async with _get_twap_control_lock():
        if normalized:
            task_id = TWAP_ACTIVE_BY_SYMBOL.get(normalized)
            task = TWAP_TASKS.get(task_id) if task_id else None
            payload = {"task_id": task_id, "task": task} if task_id else None
            return json.dumps({"status": "success", "data": {"by_symbol": {normalized: payload}}}, ensure_ascii=False)

        by_symbol: dict[str, dict] = {}
        for sym, task_id in list(TWAP_ACTIVE_BY_SYMBOL.items()):
            task = TWAP_TASKS.get(task_id)
            if not isinstance(task, dict):
                continue
            if str(task.get("status") or "") != "running":
                continue
            by_symbol[sym] = {"task_id": task_id, "task": task}
        return json.dumps({"status": "success", "data": {"by_symbol": by_symbol}}, ensure_ascii=False)

@mcp.tool()
async def get_position_snapshot(symbol: str) -> str:
    try:
        await engine.init_exchange()
        positions, resolved_symbol = await engine._fetch_positions_for_symbol(symbol)
        position = None
        for pos in positions or []:
            if safe_decimal(pos.get("contracts", 0)) > Decimal("0"):
                position = pos
                break

        if not position:
            return json.dumps({"status": "success", "data": None}, ensure_ascii=False)

        contracts = safe_decimal(position.get("contracts", 0))
        entry_price = safe_decimal(position.get("entryPrice", position.get("entry_price", 0)))
        mark_price = safe_decimal(position.get("markPrice", position.get("mark_price", 0)))
        unrealized_pnl = safe_decimal(position.get("unrealizedPnl", position.get("unrealized_pnl", 0)))
        side = position.get("side") or position.get("positionSide") or position.get("position_side")
        leverage = float(position.get("leverage", 20))
        initial_margin = safe_decimal(position.get("initialMargin", position.get("initial_margin", 0)))
        
        roe = 0.0
        if initial_margin > 0:
            roe = float(unrealized_pnl / initial_margin)
        elif entry_price > 0 and contracts > 0:
            # 兼容：如果拿不到 margin，根据杠杆推算 ROE
            price_diff = mark_price - entry_price if side == "long" else entry_price - mark_price
            price_move_pct = float(price_diff / entry_price)
            roe = price_move_pct * leverage

        data = {
            "symbol": str(position.get("symbol") or resolved_symbol or symbol),
            "contracts": float(contracts),
            "side": side,
            "entry_price": float(entry_price),
            "mark_price": float(mark_price),
            "unrealized_pnl": float(unrealized_pnl),
            "leverage": leverage,
            "roe": roe
        }

        return json.dumps({"status": "success", "data": data}, ensure_ascii=False)
    except Exception as e:
        logger.exception("获取持仓快照失败")
        return MCPErrorResponse(status="error", error_code="POSITION_QUERY_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def get_all_position_snapshots() -> str:
    """
    查询账户当前所有真实持仓快照，供 API 风控做全仓硬止损巡检。
    """
    try:
        await engine.init_exchange()
        positions = await engine.ex.fetch_positions(None)
        snapshots: list[dict] = []

        for position in positions or []:
            contracts = safe_decimal(position.get("contracts", 0))
            if contracts <= Decimal("0"):
                continue

            entry_price = safe_decimal(position.get("entryPrice", position.get("entry_price", 0)))
            mark_price = safe_decimal(position.get("markPrice", position.get("mark_price", 0)))
            unrealized_pnl = safe_decimal(position.get("unrealizedPnl", position.get("unrealized_pnl", 0)))
            side = position.get("side") or position.get("positionSide") or position.get("position_side")
            leverage = float(position.get("leverage", 20))
            initial_margin = safe_decimal(position.get("initialMargin", position.get("initial_margin", 0)))

            roe = 0.0
            if initial_margin > 0:
                roe = float(unrealized_pnl / initial_margin)
            elif entry_price > 0 and contracts > 0:
                price_diff = mark_price - entry_price if side == "long" else entry_price - mark_price
                price_move_pct = float(price_diff / entry_price)
                roe = price_move_pct * leverage

            snapshots.append(
                {
                    "symbol": str(position.get("symbol") or ""),
                    "contracts": float(contracts),
                    "side": side,
                    "entry_price": float(entry_price),
                    "mark_price": float(mark_price),
                    "unrealized_pnl": float(unrealized_pnl),
                    "leverage": leverage,
                    "roe": roe,
                }
            )

        return json.dumps({"status": "success", "data": snapshots}, ensure_ascii=False)
    except Exception as e:
        logger.exception("获取全仓持仓快照失败")
        return MCPErrorResponse(status="error", error_code="ALL_POSITION_QUERY_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def get_protection_snapshot(symbol: str) -> str:
    """
    查询当前 symbol 是否存在本系统管理的 reduceOnly 止损保护单。
    """
    try:
        await engine.init_exchange()
        open_orders, resolved_symbol = await engine._fetch_open_orders_for_symbol(symbol)
        tag_prefix = f"TM_SL_{_normalize_client_tag_symbol(resolved_symbol or symbol)}"
        protection_orders: list[dict] = []

        for order in open_orders or []:
            otype = str(order.get("type") or "").lower()
            info = order.get("info") if isinstance(order.get("info"), dict) else {}
            has_stop = ("stop" in otype) or ("stopPrice" in info) or ("stopprice" in {k.lower() for k in info.keys()})
            if not has_stop:
                continue

            reduce_only = False
            if isinstance(info, dict):
                reduce_only = str(info.get("reduceOnly") or info.get("reduce_only") or "").lower() in {"true", "1"}

            client_id = _extract_client_order_id(order)
            is_managed = bool(client_id and client_id.startswith(tag_prefix))
            if reduce_only and is_managed:
                protection_orders.append(
                    {
                        "id": order.get("id"),
                        "client_order_id": client_id,
                        "type": order.get("type"),
                        "side": order.get("side"),
                        "amount": order.get("amount"),
                        "stop_price": (
                            order.get("stopPrice")
                            or (info.get("stopPrice") if isinstance(info, dict) else None)
                            or (info.get("stopPrice".lower()) if isinstance(info, dict) else None)
                        ),
                    }
                )

        return json.dumps(
            {
                "status": "success",
                "data": {
                    "symbol": str(resolved_symbol or symbol),
                    "tag_prefix": tag_prefix,
                    "has_stop_loss": len(protection_orders) > 0,
                    "protection_orders": protection_orders,
                },
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("获取保护单快照失败")
        return MCPErrorResponse(status="error", error_code="PROTECTION_QUERY_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def execute_smart_order(symbol: str, side: str, amount_usd: float, order_type: str = 'market', price: float = None, sl_price: float = None, tp_price: float = None, volatility: float = 0.0) -> str:
    """
    [Agent 专用工具] 执行带风控校验的智能下单指令。
    
    Args:
        symbol (str): 交易对，如 'BTC/USDT'。
        side (str): 'buy' 或 'sell'。
        amount_usd (float): 下单金额 (USDT)。
        order_type (str): 'market' (市价) 或 'limit' (限价)。
        price (float, optional): 限价单必须指定价格。
        sl_price (float, optional): 止损触发价格 (若设置，会自动挂止损单)。
        tp_price (float, optional): 止盈触发价格 (若设置，会自动挂止盈单)。
        volatility (float, optional): 当前市场波动率 (来自 Indicator-MCP)，用于风控判断。
        
    Returns:
        str: 订单结果 JSON 字符串。
    """
    amount_usd_dec = safe_decimal(amount_usd)
    
    # 1. 物理层面的第一道门：熔断检查
    allowed, reason = execution_breaker.is_allowed()
    if not allowed:
        logger.error(f"🛡️ 熔断拦截开仓请求: {reason}")
        return MCPErrorResponse(status="rejected", error_code="CIRCUIT_BREAKER_OPEN", message=f"全局熔断已触发: {reason}").model_dump_json()

    try:
        await engine.init_exchange()

        should_twap = amount_usd_dec >= engine.twap_threshold
        
        # 1. 执行前置风控
        is_safe, reason = await engine.validate_risk(
            symbol,
            amount_usd_dec,
            order_type,
            volatility,
            side,
            is_algo_order=should_twap,
        )
        if not is_safe:
            logger.warning(f"风控拦截 ({symbol}): {reason}")
            return MCPErrorResponse(status="rejected", error_code="RISK_CHECK_FAILED", message=reason).model_dump_json()

        if should_twap:
            effective_chunk_usd = min(engine.twap_chunk_size, RISK_CONFIG["MAX_ORDER_USD"], amount_usd_dec)
            async with _get_twap_control_lock():
                active_id = TWAP_ACTIVE_BY_SYMBOL.get(symbol)
                if active_id:
                    active = TWAP_TASKS.get(active_id)
                    active_status = str(active.get("status")) if isinstance(active, dict) else ""
                    active_side = str(active.get("side") or "").lower() if isinstance(active, dict) else ""
                    if active_status == "running":
                        if active_side == str(side).lower():
                            return MCPErrorResponse(
                                status="rejected",
                                error_code="TWAP_IN_FLIGHT",
                                message=f"{symbol} 存在在飞 TWAP 任务 {active_id}，拒绝叠加同向任务。",
                            ).model_dump_json()
                        await engine._cancel_twap_task(active_id, f"reverse signal: {active_side}->{side}")
                    if TWAP_ACTIVE_BY_SYMBOL.get(symbol) == active_id:
                        TWAP_ACTIVE_BY_SYMBOL.pop(symbol, None)

                task_id = engine._new_twap_task(symbol, side, amount_usd_dec, effective_chunk_usd)
                TWAP_ACTIVE_BY_SYMBOL[symbol] = task_id

            if RISK_CONFIG["DRY_RUN_MODE"]:
                twap_async = asyncio.create_task(engine.run_twap_background(task_id, symbol, side, amount_usd_dec, start_slice_index=0))
                TWAP_ASYNC_TASKS[task_id] = twap_async
                return json.dumps({
                    "status": "success_dry_run",
                    "execution_mode": "twap",
                    "twap_task_id": task_id,
                    "message": f"[DRY-RUN] 订单金额 ${amount_usd_dec} 触发 TWAP 大额路由，已进入后台切片模拟执行。"
                }, ensure_ascii=False)

            try:
                first_slice = await engine._execute_twap_slice(symbol, side, effective_chunk_usd, task_id, 1)
            except Exception as e:
                task = TWAP_TASKS.get(task_id)
                if isinstance(task, dict):
                    task["status"] = "failed"
                    task["last_error"] = str(e)
                    task["updated_at"] = int(time.time())
                async with _get_twap_control_lock():
                    if TWAP_ACTIVE_BY_SYMBOL.get(symbol) == task_id:
                        TWAP_ACTIVE_BY_SYMBOL.pop(symbol, None)
                return MCPErrorResponse(status="error", error_code="TWAP_FIRST_SLICE_FAILED", message=str(e)).model_dump_json()

            remaining_usd = amount_usd_dec - effective_chunk_usd
            if remaining_usd > Decimal("0"):
                twap_async = asyncio.create_task(engine.run_twap_background(task_id, symbol, side, remaining_usd, start_slice_index=1))
                TWAP_ASYNC_TASKS[task_id] = twap_async
            else:
                task = TWAP_TASKS.get(task_id)
                if isinstance(task, dict):
                    task["status"] = "completed"
                    task["updated_at"] = int(time.time())
                async with _get_twap_control_lock():
                    if TWAP_ACTIVE_BY_SYMBOL.get(symbol) == task_id:
                        TWAP_ACTIVE_BY_SYMBOL.pop(symbol, None)

            return json.dumps({
                "status": "success",
                "execution_mode": "twap",
                "twap_task_id": task_id,
                "message": f"订单金额 ${amount_usd_dec} 触发 TWAP 大额路由：首刀已执行(${effective_chunk_usd})，剩余将后台切片执行。",
                "order_id": first_slice.get("order_id"),
                "filled_qty": first_slice.get("filled_qty"),
                "average_price": first_slice.get("average_price"),
                "fee": "0.0"
            }, ensure_ascii=False)

        # 2. 计算下单数量
        ticker = await engine.ex.fetch_ticker(symbol)
        current_price = safe_decimal(ticker['last'])
        
        # 如果是限价单，且未提供价格，则报错
        if order_type == 'limit' and not price:
             return MCPErrorResponse(status="error", error_code="INVALID_PARAMS", message="Limit order requires a price.").model_dump_json()
        
        exec_price_dec = safe_decimal(price) if order_type == 'limit' else current_price
        amount_dec = amount_usd_dec / exec_price_dec
        
        # 精度调整: 使用 exchange.amount_to_precision 和 price_to_precision
        amount_str = engine.ex.amount_to_precision(symbol, float(amount_dec))
        price_str = engine.ex.price_to_precision(symbol, float(exec_price_dec)) if order_type == 'limit' else None

        logger.info(f"准备执行: {side.upper()} {symbol} {amount_str} @ {order_type} (Est. Price: {exec_price_dec})")

        # ---------------------------------------------------------
        # ⚠️ 核心安全拦截 (Dry-Run 模式检查)
        # ---------------------------------------------------------
        if RISK_CONFIG["DRY_RUN_MODE"]:
            logger.warning(f"🛡️ [DRY-RUN 拦截] 模拟下单成功，未发送至交易所: {side.upper()} {symbol} {amount_str} @ {exec_price_dec}")
            
            # 主动清空余额缓存，确保下一次查询是最新数据
            balance_cache.clear()
            logger.info("♻️ 订单已执行(Dry-Run)，已主动清空本地资产缓存。")
            
            # 模拟返回成功订单结构
            mock_order_id = f"mock_{int(time.time()*1000)}"
            return json.dumps({
                "status": "success_dry_run", 
                "execution_mode": "dry_run",
                "order_id": mock_order_id, 
                "sl_order_id": f"{mock_order_id}_sl" if sl_price else None,
                "filled_qty": amount_str,
                "average_price": str(exec_price_dec),
                "fee": "0.0",
                "note": "This is a simulated order generated in DRY_RUN_MODE."
            }, ensure_ascii=False)

        # 3. 发送物理主订单
        try:
            if order_type == 'market':
                order = await engine.ex.create_market_order(symbol, side, float(amount_str))
            else:
                order = await engine.ex.create_limit_order(symbol, side, float(amount_str), float(price_str))
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
            logger.exception(f"❌ [CCXT 认证失败] {type(e).__name__}: {e}")
            return MCPErrorResponse(status="error", error_code="AUTHENTICATION_ERROR", message=str(e)).model_dump_json()
        except ccxt.InsufficientFunds as e:
            logger.exception(f"❌ [CCXT 余额不足] {type(e).__name__}: {e}")
            return MCPErrorResponse(status="error", error_code="INSUFFICIENT_FUNDS", message=str(e)).model_dump_json()
        except ccxt.InvalidOrder as e:
            logger.exception(f"❌ [CCXT 非法订单] {type(e).__name__}: {e}")
            return MCPErrorResponse(status="error", error_code="INVALID_ORDER", message=str(e)).model_dump_json()
        except ccxt.ExchangeError as e:
            logger.exception(f"❌ [CCXT 交易所错误] {type(e).__name__}: {e}")
            emsg = str(e)
            emsg_lower = emsg.lower()
            if ("notional" in emsg_lower and "no smaller than" in emsg_lower) or ("\"code\":-4164" in emsg_lower) or ("code': -4164" in emsg_lower):
                return MCPErrorResponse(status="rejected", error_code="MIN_NOTIONAL", message=emsg).model_dump_json()
            if any(k in emsg_lower for k in ["rate limit", "too many requests", "ddos", "temporarily banned", "service unavailable", "timeout", "timed out"]):
                execution_breaker.record_failure()
            return MCPErrorResponse(status="error", error_code="EXCHANGE_ERROR", message=emsg).model_dump_json()
        except Exception as e:
            logger.exception(f"❌ [CCXT 未知下单错误] {type(e).__name__}: {e}")
            execution_breaker.record_failure()
            return MCPErrorResponse(status="error", error_code="CREATE_ORDER_FAILED", message=str(e)).model_dump_json()
            
        logger.info(f"主订单物理下单成功: ID={order['id']}, Status={order['status']}")
        
        # 主动清空余额缓存，确保下一次查询是最新数据
        balance_cache.clear()
        execution_breaker.record_success() # 成功发单，重置熔断器
        logger.info("♻️ 订单已执行，已主动清空本地资产缓存。")
        
        fee = 0.0
        if isinstance(order.get("fee"), dict):
            fee = float(order["fee"].get("cost") or 0.0)

        protection_pending = False
        protection_messages: list[str] = []

        sl_order_id = None
        if sl_price:
            sl_retries = int(os.getenv("PROTECTION_SL_RETRY", "3") or 3)
            sl_retry_delay_sec = float(os.getenv("PROTECTION_SL_RETRY_DELAY_SEC", "0.4") or 0.4)
            sl_tag_prefix = f"TM_SL_{_normalize_client_tag_symbol(symbol)}"
            last_exc: Exception | None = None

            for attempt in range(1, max(1, sl_retries) + 1):
                try:
                    sl_price_str = engine.ex.price_to_precision(symbol, sl_price)
                    client_id = f"{sl_tag_prefix}_{int(time.time() * 1000)}"
                    sl_order = await engine.ex.create_order(
                        symbol=symbol,
                        type="STOP_MARKET",
                        side="buy" if side == "sell" else "sell",
                        amount=float(amount_str),
                        params={
                            "stopPrice": float(sl_price_str),
                            "reduceOnly": True,
                            "newClientOrderId": client_id,
                        },
                    )
                    sl_order_id = sl_order.get("id")
                    logger.info(f"止损单设置成功: ID={sl_order_id} @ {sl_price_str}")
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    logger.warning(f"止损单设置失败(尝试 {attempt}/{sl_retries}): {e}")
                    if attempt < sl_retries:
                        await asyncio.sleep(sl_retry_delay_sec)

            if last_exc is not None and not sl_order_id:
                logger.exception(f"止损单设置失败 (主单已成): {str(last_exc)}")
                retry_attempts = int(os.getenv("PROTECTION_SL_BACKGROUND_RETRY", "20") or 20)
                retry_delay_sec = float(os.getenv("PROTECTION_SL_BACKGROUND_DELAY_SEC", "1.0") or 1.0)
                close_side = "buy" if side == "sell" else "sell"
                asyncio.create_task(
                    engine._retry_create_stop_loss(
                        symbol=symbol,
                        close_side=close_side,
                        amount_str=str(amount_str),
                        sl_price=float(sl_price),
                        tag_prefix=sl_tag_prefix,
                        max_attempts=retry_attempts,
                        delay_sec=retry_delay_sec,
                    )
                )
                protection_pending = True
                protection_messages.append(f"止损单创建失败，已进入后台重试: {str(last_exc)}")

        tp_order_id = None
        if tp_price:
            tp_retries = int(os.getenv("PROTECTION_TP_RETRY", "2") or 2)
            tp_retry_delay_sec = float(os.getenv("PROTECTION_TP_RETRY_DELAY_SEC", "0.4") or 0.4)
            tp_tag_prefix = f"TM_TP_{_normalize_client_tag_symbol(symbol)}"
            last_exc: Exception | None = None
            close_side = "buy" if side == "sell" else "sell"

            for attempt in range(1, max(1, tp_retries) + 1):
                try:
                    tp_price_str = engine.ex.price_to_precision(symbol, tp_price)
                    client_id = f"{tp_tag_prefix}_{int(time.time() * 1000)}"
                    tp_order = await engine.ex.create_order(
                        symbol=symbol,
                        type="TAKE_PROFIT_MARKET",
                        side=close_side,
                        amount=float(amount_str),
                        params={
                            "stopPrice": float(tp_price_str),
                            "reduceOnly": True,
                            "newClientOrderId": client_id,
                        },
                    )
                    tp_order_id = tp_order.get("id")
                    logger.info(f"止盈单设置成功: ID={tp_order_id} @ {tp_price_str}")
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    logger.warning(f"止盈单设置失败(尝试 {attempt}/{tp_retries}): {e}")
                    if attempt < tp_retries:
                        await asyncio.sleep(tp_retry_delay_sec)

            if last_exc is not None and not tp_order_id:
                retry_attempts = int(os.getenv("PROTECTION_TP_BACKGROUND_RETRY", "10") or 10)
                retry_delay_sec = float(os.getenv("PROTECTION_TP_BACKGROUND_DELAY_SEC", "1.0") or 1.0)
                asyncio.create_task(
                    engine._retry_create_take_profit(
                        symbol=symbol,
                        close_side=close_side,
                        amount_str=str(amount_str),
                        tp_price=float(tp_price),
                        tag_prefix=tp_tag_prefix,
                        max_attempts=retry_attempts,
                        delay_sec=retry_delay_sec,
                    )
                )
                protection_pending = True
                protection_messages.append(f"止盈单创建失败，已进入后台重试: {str(last_exc)}")

        status = "partial_success" if protection_pending else "success"
        message = " ; ".join(protection_messages) if protection_messages else None

        return json.dumps({
            "status": status,
            "execution_mode": "live",
            "order_id": order['id'],
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id,
            "message": message,
            "filled_qty": order.get('filled', amount_str),
            "average_price": order.get('average', str(exec_price_dec)),
            "fee": fee
        }, ensure_ascii=False)

    except ccxt.NetworkError as e:
        execution_breaker.record_failure()
        logger.exception(f"执行层网络崩溃: {str(e)}")
        return MCPErrorResponse(status="error", error_code="EXCHANGE_NETWORK_ERROR", message=f"交易所连接失败: {str(e)}").model_dump_json()

    except Exception as e:
        logger.error(f"下单执行异常: {str(e)}")
        return MCPErrorResponse(status="error", error_code="EXECUTION_ERROR", message=str(e)).model_dump_json()

@mcp.tool()
async def update_trailing_stop(symbol: str, trailing_pct: float = 0.05, roe_activation: float = 0.05) -> str:
    """
    [Agent 专用工具] 追踪止损联动。
    检查该 symbol 是否存在浮盈仓位，如果有，则自动计算最新的高位回撤止损价，
    并挂出或更新 Stop Market 条件单，以锁定利润。
    采用了绝对价格与 ROE (Return on Equity) 混合的机制。
    
    Args:
        symbol (str): 交易对，如 'BTC/USDT'。
        trailing_pct (float): 回撤容忍比例，默认 0.05 (即 5%)。
        roe_activation (float): 触发追踪止损的最小 ROE 阈值，默认 0.05 (即 5%)。
        
    Returns:
        str: 止损单更新结果 JSON 字符串，包含系统计算的 ROE 和 Binance 同步 ROE。
    """
    try:
        updated_orders = await engine.sync_trailing_stop(symbol, trailing_pct, roe_activation)
        if not updated_orders:
            return json.dumps({
                "status": "success",
                "message": "当前没有浮盈仓位或 ROE 未达到追踪止损激活条件，无需更新追踪止损单。"
            }, ensure_ascii=False)
            
        return json.dumps({
            "status": "success",
            "message": f"基于绝对价格回撤与 ROE 混合机制，成功更新了 {len(updated_orders)} 个追踪止损单。",
            "details": updated_orders
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"追踪止损执行失败: {e}")
        return MCPErrorResponse(status="error", error_code="TRAILING_STOP_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def update_stop_loss(symbol: str, position_side: str, sl_price: float) -> str:
    """
    将指定 symbol 的交易所止损单更新到新的触发价。
    """
    try:
        await engine.init_exchange()

        if RISK_CONFIG["DRY_RUN_MODE"]:
            return json.dumps(
                {
                    "status": "success_dry_run",
                    "message": f"[DRY-RUN] 已模拟更新止损单: {symbol} position_side={position_side} sl_price={sl_price}",
                },
                ensure_ascii=False,
            )

        positions, resolved_symbol = await engine._fetch_positions_for_symbol(symbol)
        position = None
        for pos in positions or []:
            if safe_decimal(pos.get("contracts", 0)) > Decimal("0"):
                position = pos
                break

        if not position:
            return MCPErrorResponse(status="rejected", error_code="NO_POSITION", message=f"{symbol} 当前无持仓，无法更新止损单").model_dump_json()

        execution_symbol = str(position.get("symbol") or resolved_symbol or symbol)
        contracts = safe_decimal(position.get("contracts", 0))
        amount_str = engine.ex.amount_to_precision(execution_symbol, float(contracts))
        close_side = None

        raw_side = (position.get("side") or "").lower()
        if raw_side in {"long", "short"}:
            close_side = "sell" if raw_side == "long" else "buy"
        else:
            ps = str(position_side or "").upper()
            close_side = "sell" if ps == "BUY" else "buy"

        sl_price_dec = safe_decimal(sl_price)
        if sl_price_dec <= Decimal("0"):
            return MCPErrorResponse(status="error", error_code="INVALID_PARAMS", message="sl_price 必须大于 0").model_dump_json()

        sl_price_str = engine.ex.price_to_precision(execution_symbol, float(sl_price_dec))
        tag_prefix = f"TM_SL_{_normalize_client_tag_symbol(execution_symbol)}"

        new_client_id = f"{tag_prefix}_{int(time.time() * 1000)}"
        sl_order = await engine.ex.create_order(
            symbol=execution_symbol,
            type="STOP_MARKET",
            side=close_side,
            amount=float(amount_str),
            params={"stopPrice": float(sl_price_str), "reduceOnly": True, "newClientOrderId": new_client_id},
        )
        new_order_id = sl_order.get("id")

        cancelled_order_ids: list[str] = []
        try:
            open_orders, resolved_open_symbol = await engine._fetch_open_orders_for_symbol(execution_symbol)
        except Exception:
            open_orders = []
            resolved_open_symbol = execution_symbol

        for o in open_orders or []:
            oid = o.get("id")
            if oid and new_order_id and str(oid) == str(new_order_id):
                continue
            otype = str(o.get("type") or "").lower()
            info = o.get("info") if isinstance(o.get("info"), dict) else {}
            has_stop = ("stop" in otype) or ("stopPrice" in info) or ("stopprice" in {k.lower() for k in info.keys()})
            reduce_only = False
            if isinstance(info, dict):
                reduce_only = str(info.get("reduceOnly") or info.get("reduce_only") or "").lower() in {"true", "1"}
            client_id = _extract_client_order_id(o)
            is_managed = bool(client_id and client_id.startswith(tag_prefix))
            if has_stop and reduce_only and is_managed and oid:
                try:
                    await engine.ex.cancel_order(oid, resolved_open_symbol)
                    cancelled_order_ids.append(str(oid))
                except Exception:
                    pass

        return json.dumps(
            {
                "status": "success",
                "message": f"止损单已更新: {execution_symbol} -> {sl_price_str}",
                "cancelled_order_ids": cancelled_order_ids,
                "sl_order_id": new_order_id,
                "sl_price": sl_price_str,
                "amount": amount_str,
                "close_side": close_side,
                "symbol": execution_symbol,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"更新止损单失败: {e}")
        return MCPErrorResponse(status="error", error_code="UPDATE_STOP_LOSS_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def update_take_profit(symbol: str, position_side: str, tp_price: float) -> str:
    try:
        await engine.init_exchange()

        if RISK_CONFIG["DRY_RUN_MODE"]:
            return json.dumps(
                {
                    "status": "success_dry_run",
                    "message": f"[DRY-RUN] 已模拟更新止盈单: {symbol} position_side={position_side} tp_price={tp_price}",
                },
                ensure_ascii=False,
            )

        positions, resolved_symbol = await engine._fetch_positions_for_symbol(symbol)
        position = None
        for pos in positions or []:
            if safe_decimal(pos.get("contracts", 0)) > Decimal("0"):
                position = pos
                break

        if not position:
            return MCPErrorResponse(status="rejected", error_code="NO_POSITION", message=f"{symbol} 当前无持仓，无法更新止盈单").model_dump_json()

        execution_symbol = str(position.get("symbol") or resolved_symbol or symbol)
        contracts = safe_decimal(position.get("contracts", 0))
        amount_str = engine.ex.amount_to_precision(execution_symbol, float(contracts))

        raw_side = (position.get("side") or "").lower()
        if raw_side in {"long", "short"}:
            close_side = "sell" if raw_side == "long" else "buy"
        else:
            ps = str(position_side or "").upper()
            close_side = "sell" if ps == "BUY" else "buy"

        tp_price_dec = safe_decimal(tp_price)
        if tp_price_dec <= Decimal("0"):
            return MCPErrorResponse(status="error", error_code="INVALID_PARAMS", message="tp_price 必须大于 0").model_dump_json()

        tp_price_str = engine.ex.price_to_precision(execution_symbol, float(tp_price_dec))
        tag_prefix = f"TM_TP_{_normalize_client_tag_symbol(execution_symbol)}"

        new_client_id = f"{tag_prefix}_{int(time.time() * 1000)}"
        tp_order = await engine.ex.create_order(
            symbol=execution_symbol,
            type="TAKE_PROFIT_MARKET",
            side=close_side,
            amount=float(amount_str),
            params={"stopPrice": float(tp_price_str), "reduceOnly": True, "newClientOrderId": new_client_id},
        )
        new_order_id = tp_order.get("id")

        cancelled_order_ids: list[str] = []
        try:
            open_orders, resolved_open_symbol = await engine._fetch_open_orders_for_symbol(execution_symbol)
        except Exception:
            open_orders = []
            resolved_open_symbol = execution_symbol

        for o in open_orders or []:
            oid = o.get("id")
            if oid and new_order_id and str(oid) == str(new_order_id):
                continue
            otype = str(o.get("type") or "").lower()
            info = o.get("info") if isinstance(o.get("info"), dict) else {}
            has_tp = ("take_profit" in otype) or ("takeprofit" in otype) or ("stopPrice" in info) or ("stopprice" in {k.lower() for k in info.keys()})
            reduce_only = False
            if isinstance(info, dict):
                reduce_only = str(info.get("reduceOnly") or info.get("reduce_only") or "").lower() in {"true", "1"}
            client_id = _extract_client_order_id(o)
            is_managed = bool(client_id and client_id.startswith(tag_prefix))
            if has_tp and reduce_only and is_managed and oid:
                try:
                    await engine.ex.cancel_order(oid, resolved_open_symbol)
                    cancelled_order_ids.append(str(oid))
                except Exception:
                    pass

        return json.dumps(
            {
                "status": "success",
                "message": f"止盈单已更新: {execution_symbol} -> {tp_price_str}",
                "cancelled_order_ids": cancelled_order_ids,
                "tp_order_id": new_order_id,
                "tp_price": tp_price_str,
                "amount": amount_str,
                "close_side": close_side,
                "symbol": execution_symbol,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"更新止盈单失败: {e}")
        return MCPErrorResponse(status="error", error_code="UPDATE_TAKE_PROFIT_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def get_order_info(symbol: str, order_id: str) -> str:
    """
    读取交易所侧的订单详情，用于复盘“为何盈利单最终亏损平仓”（例如 stop 触发价、成交均价、是否 reduceOnly 等）。
    """
    try:
        await engine.init_exchange()
        order = await engine.ex.fetch_order(order_id, symbol)
        return json.dumps({"status": "success", "data": order}, ensure_ascii=False)
    except ccxt.AuthenticationError as e:
        return MCPErrorResponse(status="error", error_code="AUTHENTICATION_ERROR", message=str(e)).model_dump_json()
    except ccxt.ExchangeError as e:
        return MCPErrorResponse(status="error", error_code="EXCHANGE_ERROR", message=str(e)).model_dump_json()
    except Exception as e:
        logger.error(f"查询订单失败: {e}")
        return MCPErrorResponse(status="error", error_code="FETCH_ORDER_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def cancel_order(symbol: str, order_id: str) -> str:
    """
    统一撤销指定订单。

    设计原因：
    - API Gateway / Patrol Bot 通过 ExecutionGateway 共用同一条撤单链路；
    - 服务端只负责执行单笔撤单，不在此层重复实现节流，避免与网关治理口径分叉。
    """
    try:
        await engine.init_exchange()
        normalized_symbol = str(symbol or "").strip()
        normalized_order_id = str(order_id or "").strip()
        if not normalized_symbol:
            return MCPErrorResponse(
                status="error",
                error_code="INVALID_PARAMS",
                message="symbol is required",
            ).model_dump_json()
        if not normalized_order_id:
            return MCPErrorResponse(
                status="error",
                error_code="INVALID_PARAMS",
                message="order_id is required",
            ).model_dump_json()

        result = await engine.ex.cancel_order(normalized_order_id, normalized_symbol)
        logger.info(
            "✅ 撤单成功: symbol=%s order_id=%s",
            normalized_symbol,
            normalized_order_id,
        )
        return json.dumps(
            {
                "status": "success",
                "symbol": normalized_symbol,
                "order_id": normalized_order_id,
                "data": result,
            },
            ensure_ascii=False,
        )
    except ccxt.OrderNotFound as e:
        return MCPErrorResponse(status="error", error_code="ORDER_NOT_FOUND", message=str(e)).model_dump_json()
    except ccxt.AuthenticationError as e:
        return MCPErrorResponse(status="error", error_code="AUTHENTICATION_ERROR", message=str(e)).model_dump_json()
    except ccxt.ExchangeError as e:
        return MCPErrorResponse(status="error", error_code="EXCHANGE_ERROR", message=str(e)).model_dump_json()
    except Exception as e:
        logger.error(f"撤单失败: {e}")
        return MCPErrorResponse(status="error", error_code="CANCEL_ORDER_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def kill_all_positions(symbol: str) -> str:
    """
    [Agent 工具] 一键清仓撤单 (Kill Switch)。撤销所有挂单并市价平掉所有持仓。
    """
    # Kill Switch 是逃生通道，不应该受普通熔断器限制，除非物理断网
    # 但为了防止死循环报错，也可以加一个简单的判断，或者赋予它最高优先级（绕过熔断）
    # 这里选择绕过熔断，因为这是用来救命的
    
    try:
        await engine.init_exchange()
        logger.warning(f"🚨 收到清仓指令: {symbol}")

        await engine.cancel_twap_for_symbol(symbol, reason="kill switch invoked")
        
        # 1. 撤销所有挂单
        await engine.cancel_all_orders(symbol)
        logger.info(f"已撤销 {symbol} 所有挂单。")
        
        # 2. 平掉所有仓位
        closed_count = await engine.close_all_positions(symbol)
        logger.info(f"已平仓 {symbol} 的 {closed_count} 个持仓。")

        # 再次复查交易所持仓，避免 symbol 口径异常或交易所拒单时出现“假成功”。
        remaining_positions, resolved_symbol = await engine._fetch_positions_for_symbol(symbol)
        live_positions = [
            pos
            for pos in (remaining_positions or [])
            if safe_decimal(pos.get("contracts", 0)) > Decimal("0")
        ]
        if live_positions:
            logger.error(
                "❌ Kill switch 执行后仍有持仓残留: requested=%s resolved=%s remaining=%s",
                symbol,
                resolved_symbol,
                [
                    {
                        "symbol": str(pos.get("symbol") or resolved_symbol or symbol),
                        "contracts": str(pos.get("contracts", 0)),
                        "side": pos.get("side"),
                    }
                    for pos in live_positions
                ],
            )
            return MCPErrorResponse(
                status="error",
                error_code="KILL_SWITCH_POSITION_REMAINING",
                message=(
                    f"Kill switch 执行后 {symbol} 仍检测到未平仓持仓，"
                    f"resolved_symbol={resolved_symbol}"
                ),
            ).model_dump_json()

        result_message = (
            f"Kill switch executed for {symbol}. Orders cancelled, {closed_count} positions closed."
            if closed_count > 0
            else f"Kill switch executed for {symbol}. No live positions remained after verification."
        )
        return json.dumps(
            {
                "status": "success",
                "message": result_message,
                "closed_count": closed_count,
                "resolved_symbol": resolved_symbol,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"熔断失败: {str(e)}")
        return MCPErrorResponse(status="error", error_code="KILL_SWITCH_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def kill_all_positions_global() -> str:
    try:
        await engine.init_exchange()
        logger.warning("🚨 收到全局清仓指令: GLOBAL")

        symbols = set()
        try:
            positions = await engine.ex.fetch_positions()
            for position in positions or []:
                try:
                    contracts = safe_decimal(position.get("contracts", 0))
                except Exception:
                    contracts = Decimal("0")
                if contracts > Decimal("0"):
                    sym = position.get("symbol")
                    if sym:
                        symbols.add(sym)
        except Exception as e:
            logger.error(f"获取全局持仓失败: {e}")

        try:
            open_orders = await engine.ex.fetch_open_orders()
            for o in open_orders or []:
                sym = o.get("symbol")
                if sym:
                    symbols.add(sym)
        except Exception as e:
            logger.error(f"获取全局挂单失败: {e}")

        cancelled = 0
        closed = 0
        errors = []

        for sym in sorted(symbols):
            try:
                await engine.cancel_twap_for_symbol(sym, reason="global kill switch invoked")
                await engine.cancel_all_orders(sym)
                cancelled += 1
            except Exception as e:
                errors.append({"symbol": sym, "stage": "cancel_all_orders", "error": str(e)})
            try:
                c = await engine.close_all_positions(sym)
                closed += int(c or 0)
            except Exception as e:
                errors.append({"symbol": sym, "stage": "close_all_positions", "error": str(e)})

        return json.dumps(
            {
                "status": "success",
                "message": "Global kill switch executed.",
                "symbols_affected": sorted(symbols),
                "orders_cancelled_symbols": cancelled,
                "positions_closed_count": closed,
                "errors": errors,
            },
            ensure_ascii=False
        )
    except Exception as e:
        logger.error(f"全局熔断失败: {str(e)}")
        return MCPErrorResponse(status="error", error_code="KILL_SWITCH_GLOBAL_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def ping_exchange() -> str:
    """
    [Agent 工具] 测试与交易所 API 的真实连通延迟 (毫秒)。
    """
    try:
        await engine.init_exchange()
        start = time.perf_counter()
        
        # 尝试调用 fetch_time 或 fetch_status 来获取最轻量级的 API 响应时间
        if hasattr(engine.ex, 'fetch_time'):
            await engine.ex.fetch_time()
        else:
            await engine.ex.fetch_status()
            
        latency = int((time.perf_counter() - start) * 1000)
        return json.dumps({"status": "success", "latency_ms": latency}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Ping exchange failed: {e}")
        return MCPErrorResponse(status="error", error_code="PING_FAILED", message=str(e)).model_dump_json()

if __name__ == "__main__":
    mcp.run(transport="sse")
