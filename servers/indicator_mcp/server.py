import sys
import os
import importlib
from pathlib import Path
from dotenv import load_dotenv
import time
import pandas as pd
import httpx
from mcp.server.fastmcp import FastMCP
from typing import Any, Awaitable, Callable

# 添加项目根目录到 sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# 加载 .env 环境变量
load_dotenv(root_dir / ".env", override=True)

from shared.logger import get_logger
from shared.models import MarketContext, MarketRegime, VolatilityLevel, MCPErrorResponse
from shared.binance_futures import (
    BinanceApiError,
    BinanceNetworkError,
    build_exchange_from_config,
)
from shared.circuit_breaker import get_route_circuit_breaker
from servers.indicator_mcp.engines.smc_engine import SMCEngine
from servers.indicator_mcp.engines.hmm_engine import HMMEngine
from servers.indicator_mcp.engines.orderflow_engine import OrderflowEngine
from servers.indicator_mcp.engines.ws_engine import WsMemoryPool  # 引入新引擎
import numpy as np
import asyncio
import json
import time
from shared.cache_manager import (
    cache_key_builder,
    get_or_set_ttl_cache,
    market_cache,
    ohlcv_cache,
    ohlcv_request_locks,
)
from shared.validation import build_db_record
from asyncache import cached

# 初始化
logger = get_logger("Indicator-Server")
mcp = FastMCP("Indicator-Server")

# 实例化全局内存池
ws_pool = WsMemoryPool()

# 单例引擎实例
smc_engine_cache = {} # SMC 引擎通常是无状态的，或者每次重新实例化，这里仅作占位
hmm_engine = HMMEngine(n_components=3) # HMM 也可以每次实例化，但如果模型很大可以持久化
orderflow_engine = OrderflowEngine()   # Orderflow 必须持久化以计算 OFI 增量

# ==========================================
# 🧠 注入全局记忆流：记录各个资产的宏观状态
# ==========================================
STATE_MEMORY = {}

logger.info("Indicator-MCP Server initializing...")

KRONOS_RUNTIME_STATE: dict[str, Any] = {
    "initialized": False,
    "runtime": None,
    "status": {},
    "init_attempt_count": 0,
    "load_origin": "uninitialized",
    "initialized_at": None,
}

def get_exchange_config(proxy_url: str | None = None) -> dict[str, Any]:
    """动态获取带有代理的 CCXT 配置。"""
    config = {
        'enableRateLimit': True,
        'timeout': get_exchange_timeout_ms(),
        'options': {'defaultType': 'swap'}
    }
    resolved_proxy_url = (
        os.getenv("CCXT_PROXY", "http://127.0.0.1:7890").strip()
        if proxy_url is None
        else proxy_url.strip()
    )
    if resolved_proxy_url:
        config['proxies'] = {
            'http': resolved_proxy_url,
            'https': resolved_proxy_url,
        }
    return config


def build_exchange_config_candidates() -> list[tuple[str, dict[str, Any]]]:
    """构建 Binance 主网请求链路候选列表，优先代理，失败后直连。"""
    candidates: list[tuple[str, dict[str, Any]]] = []
    proxy_url = os.getenv("CCXT_PROXY", "http://127.0.0.1:7890").strip()
    if proxy_url:
        candidates.append(("proxy", get_exchange_config(proxy_url=proxy_url)))
    candidates.append(("direct", get_exchange_config(proxy_url="")))
    return candidates


def get_indicator_engine_priority() -> list[str]:
    """返回固定的 Indicator 路由策略：localhost HTTP 优先，超时后回退官方 SDK。"""
    return ["local_httpx", "sdk"]


def get_indicator_route_breaker_name(scope: str, route_name: str) -> str:
    """构建 Indicator-MCP 链路熔断器键名。"""
    return f"indicator_{scope}_{route_name}"


def get_indicator_data_source_label(route_name: str) -> str:
    """将 Indicator 路由名映射为统一的数据源日志标签。"""
    normalized_route = str(route_name).strip().lower()
    if normalized_route.startswith("sdk_") or normalized_route in {"proxy", "direct", "sdk"}:
        return "[Binance-Official-SDK]"
    return "[Hummingbot-Gateway]"


def get_indicator_exchange_route_name(exchange: Any) -> str:
    """读取交易所实例上绑定的 Indicator 路由名。"""
    return str(
        getattr(exchange, "_indicator_route_name", None)
        or getattr(exchange, "route_name", None)
        or "sdk_unknown"
    )


def log_indicator_fetch_event(
    route_name: str,
    action: str,
    symbol: str | None = None,
    timeframe: str | None = None,
    extra: str | None = None,
) -> None:
    """输出统一的 Indicator 数据获取来源日志。"""
    log_parts = [get_indicator_data_source_label(route_name), action]
    if symbol:
        log_parts.append(f"symbol={symbol}")
    if timeframe:
        log_parts.append(f"timeframe={timeframe}")
    log_parts.append(f"route={route_name}")
    if extra:
        log_parts.append(extra)
    logger.info(" ".join(log_parts))


def normalize_binance_rest_symbol(symbol: str) -> str:
    """将 CCXT 交易对格式转换为 Binance Futures REST 所需格式。"""
    return symbol.replace("/", "").replace(":", "").upper()


def get_binance_futures_rest_base_url() -> str:
    """返回当前环境对应的 Binance Futures REST 基础地址。"""
    if should_use_indicator_testnet():
        return "https://testnet.binancefuture.com"
    return "https://fapi.binance.com"


def get_exchange_timeout_ms() -> int:
    """读取 CCXT/CCXT Pro 单次链路超时，控制失败链路的等待上限。"""
    raw_value = os.getenv("INDICATOR_EXCHANGE_TIMEOUT_MS", os.getenv("CCXT_TIMEOUT_MS", "8000")).strip()
    try:
        return max(int(raw_value), 1000)
    except ValueError:
        logger.warn(f"INDICATOR_EXCHANGE_TIMEOUT_MS={raw_value} 非法，回退 8000ms")
        return 8000


def get_public_rest_timeout_seconds() -> float:
    """读取本地 localhost HTTP 网关超时，默认 1.5s 后立即回退官方 SDK。"""
    raw_value = os.getenv("INDICATOR_PUBLIC_REST_TIMEOUT_SECONDS", "1.5").strip()
    try:
        return max(float(raw_value), 0.5)
    except ValueError:
        logger.warn(f"INDICATOR_PUBLIC_REST_TIMEOUT_SECONDS={raw_value} 非法，回退 1.5s")
        return 1.5


def get_indicator_local_gateway_base_url() -> str:
    """返回 Indicator 本地 HTTP 网关地址，默认命中 localhost:8000。"""
    return os.getenv("INDICATOR_LOCAL_GATEWAY_URL", "http://127.0.0.1:8000").strip().rstrip("/")


def get_kronos_local_timeout_seconds() -> float:
    """读取本地 Kronos 推理超时，避免 Fat-Tool 被本地模型长尾阻塞。"""
    raw_value = os.getenv("KRONOS_LOCAL_TIMEOUT_SECONDS", "12").strip()
    try:
        return max(float(raw_value), 1.0)
    except ValueError:
        logger.warn(f"KRONOS_LOCAL_TIMEOUT_SECONDS={raw_value} 非法，回退 12s")
        return 12.0


async def execute_exchange_request(
    request_name: str,
    request_builder: Callable[[Any], Awaitable[Any]],
) -> tuple[Any, str]:
    """按 SDK Gateway 的代理优先、直连兜底顺序执行公共行情请求。"""
    last_error: Exception | None = None
    skipped_routes: list[str] = []

    for route_name, config in build_exchange_config_candidates():
        breaker_name = get_indicator_route_breaker_name("sdk", route_name)
        breaker = get_route_circuit_breaker(breaker_name)
        is_allowed, breaker_state = breaker.allow_request()
        if not is_allowed:
            skipped_routes.append(f"{route_name}({breaker_state})")
            logger.warn(
                f"[Binance-Official-SDK] ⏭️ {request_name} 跳过 {route_name} 链路，原因: 熔断器={breaker_state}"
            )
            continue

        exchange = None
        try:
            log_indicator_fetch_event(
                route_name=f"sdk_{route_name}",
                action=f"开始请求 {request_name}",
                extra="transport=ccxt_async_binance_futures",
            )
            logger.info(f"🌐 {request_name} 使用 SDK Gateway/{route_name} 链路访问 Binance Futures")
            exchange = build_exchange_from_config(config)
            setattr(exchange, "_indicator_route_name", f"sdk_{route_name}")
            if should_use_indicator_testnet():
                exchange.set_sandbox_mode(True)
            await exchange.load_markets()
            result = await request_builder(exchange)
            breaker.record_success()
            return result, route_name
        except Exception as exc:
            last_error = exc
            breaker.record_failure(exc)
            logger.warn(
                f"[Binance-Official-SDK] ⚠️ {request_name} 通过 {route_name} 链路失败，将尝试下一条链路: {exc}"
            )
        finally:
            if exchange is not None:
                await exchange.close()

    if last_error is not None:
        raise last_error
    if skipped_routes:
        raise RuntimeError(
            f"{request_name} 所有候选链路均处于熔断跳过状态: {', '.join(skipped_routes)}"
        )
    raise RuntimeError(f"{request_name} 未获取到可用 Binance 链路")


async def fetch_public_rest_json(
    path: str,
    params: dict[str, Any],
) -> tuple[Any, str]:
    """优先访问 localhost:8000 本地 HTTP 网关，命中超时后交由上层回退官方 SDK。"""
    route_name = "local_httpx"
    breaker_name = get_indicator_route_breaker_name("httpx", route_name)
    breaker = get_route_circuit_breaker(breaker_name)
    is_allowed, breaker_state = breaker.allow_request()
    if not is_allowed:
        raise RuntimeError(
            f"Indicator 本地 HTTP 网关处于熔断跳过状态: {route_name}({breaker_state})"
        )

    base_url = get_indicator_local_gateway_base_url()
    timeout_seconds = get_public_rest_timeout_seconds()
    request_url = f"{base_url}{path}"
    try:
        symbol = str(params.get("symbol", "MARKET"))
        timeframe = str(params.get("interval", "") or "").strip() or None
        log_indicator_fetch_event(
            route_name=route_name,
            action=f"开始请求 {path}",
            symbol=symbol,
            timeframe=timeframe,
            extra=f"url={request_url} timeout={timeout_seconds:.1f}s",
        )
        logger.info(
            f"🎯 Indicator 命中本地 HTTP 优先链路: {request_url} timeout={timeout_seconds:.1f}s"
        )
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(request_url, params=params)
            response.raise_for_status()
            breaker.record_success()
            logger.info(
                build_db_record(
                    module="Indicator-MCP",
                    event="indicator_route_hit",
                    symbol=str(params.get("symbol", "MARKET")),
                    payload={
                        "route": route_name,
                        "url": request_url,
                        "timeout_seconds": timeout_seconds,
                    },
                )
            )
            return response.json(), route_name
    except Exception as exc:
        breaker.record_failure(exc)
        raise


async def fetch_rest_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int,
) -> tuple[list[list[float]], str]:
    """优先通过 localhost HTTP 网关拉取 OHLCV，超时后回退官方 SDK。"""

    async def _request(exchange: Any) -> list[list[float]]:
        """执行单次 OHLCV 请求。"""
        return await get_cached_ohlcv(exchange, symbol, timeframe, limit=limit)

    try:
        payload, route_name = await fetch_public_rest_json(
            "/fapi/v1/klines",
            {
                "symbol": normalize_binance_rest_symbol(symbol),
                "interval": timeframe,
                "limit": limit,
            },
        )
        normalized_ohlcv = [
            [
                int(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
            ]
            for item in payload
        ]
        return normalized_ohlcv, route_name
    except httpx.TimeoutException as timeout_error:
        logger.warn(
            f"[Hummingbot-Gateway] ⏱️ Indicator 本地 HTTP OHLCV 超时 ({get_public_rest_timeout_seconds():.1f}s)，"
            f"回退官方 SDK: {timeout_error}"
        )
    except Exception as local_http_error:
        logger.warn(
            f"[Hummingbot-Gateway] ⚠️ Indicator 本地 HTTP OHLCV 失败，回退官方 SDK: {local_http_error}"
        )

    ohlcv, route_name = await execute_exchange_request(
        request_name=f"OHLCV {symbol} {timeframe}",
        request_builder=_request,
    )
    logger.info(
        build_db_record(
            module="Indicator-MCP",
            event="indicator_route_fallback",
            symbol=symbol,
            payload={
                "from_route": "local_httpx",
                "to_route": f"sdk_{route_name}",
                "reason": "timeout_or_local_http_failure",
            },
        )
    )
    return ohlcv, f"sdk_{route_name}"


async def fetch_rest_market_snapshot(
    symbol: str,
    timeframe: str,
) -> tuple[list[list[float]], dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    """优先通过 localhost HTTP 网关拉取完整快照，超时后回退官方 SDK。"""

    async def _request(exchange: Any) -> tuple[
        list[list[float]],
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        """并发执行市场快照请求。"""
        return await asyncio.gather(
            get_cached_ohlcv(exchange, symbol, timeframe, limit=500),
            get_cached_orderbook(exchange, symbol, limit=20),
            get_cached_ticker(exchange, symbol),
            get_cached_trades(exchange, symbol, limit=200),
        )

    try:
        rest_symbol = normalize_binance_rest_symbol(symbol)
        klines_result, orderbook_result, ticker_result, trades_result, stats_result = await asyncio.gather(
            fetch_public_rest_json(
                "/fapi/v1/klines",
                {"symbol": rest_symbol, "interval": timeframe, "limit": 500},
            ),
            fetch_public_rest_json(
                "/fapi/v1/depth",
                {"symbol": rest_symbol, "limit": 20},
            ),
            fetch_public_rest_json(
                "/fapi/v1/ticker/price",
                {"symbol": rest_symbol},
            ),
            fetch_public_rest_json(
                "/fapi/v1/trades",
                {"symbol": rest_symbol, "limit": 200},
            ),
            fetch_public_rest_json(
                "/fapi/v1/ticker/24hr",
                {"symbol": rest_symbol},
            ),
        )
        klines, klines_route = klines_result
        orderbook, _ = orderbook_result
        ticker, _ = ticker_result
        trades, _ = trades_result
        stats_24h, _ = stats_result
        normalized_ohlcv = [
            [
                int(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
            ]
            for item in klines
        ]
        normalized_orderbook = {
            "bids": [[float(price), float(amount)] for price, amount in orderbook.get("bids", [])],
            "asks": [[float(price), float(amount)] for price, amount in orderbook.get("asks", [])],
        }
        normalized_ticker = {
            "last": float(ticker.get("price", 0.0) or 0.0),
            "percentage": float(stats_24h.get("priceChangePercent", 0.0) or 0.0),
            "high": float(stats_24h.get("highPrice", 0.0) or 0.0),
            "low": float(stats_24h.get("lowPrice", 0.0) or 0.0),
            "quoteVolume": float(stats_24h.get("quoteVolume", 0.0) or 0.0),
            "timestamp": int(stats_24h.get("closeTime", 0) or 0),
        }
        normalized_trades = [
            {
                "price": float(item.get("price", 0.0) or 0.0),
                "amount": float(item.get("qty", 0.0) or 0.0),
                "side": "sell" if item.get("isBuyerMaker") else "buy",
                "timestamp": int(item.get("time", 0) or 0),
            }
            for item in trades
        ]
        return (
            normalized_ohlcv,
            normalized_orderbook,
            normalized_ticker,
            normalized_trades,
            klines_route,
        )
    except httpx.TimeoutException as timeout_error:
        logger.warn(
            f"[Hummingbot-Gateway] ⏱️ Indicator 本地 HTTP 市场快照超时 ({get_public_rest_timeout_seconds():.1f}s)，"
            f"回退官方 SDK: {timeout_error}"
        )
    except Exception as local_http_error:
        logger.warn(
            f"[Hummingbot-Gateway] ⚠️ Indicator 本地 HTTP 市场快照失败，回退官方 SDK: {local_http_error}"
        )

    market_snapshot, route_name = await execute_exchange_request(
        request_name=f"REST 市场快照 {symbol} {timeframe}",
        request_builder=_request,
    )
    ohlcv, orderbook, ticker, trades = market_snapshot
    logger.info(
        build_db_record(
            module="Indicator-MCP",
            event="indicator_route_fallback",
            symbol=symbol,
            payload={
                "from_route": "local_httpx",
                "to_route": f"sdk_{route_name}",
                "reason": "timeout_or_local_http_failure",
            },
        )
    )
    return ohlcv, orderbook, ticker, trades, f"sdk_{route_name}"


def normalize_ohlcv_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """清洗 OHLCV 数据，保证 Kronos/HMM/SMC 使用一致输入。"""
    df = raw_df.copy()
    numeric_columns = ["open", "high", "low", "close", "volume"]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64")
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    df = df.dropna(subset=numeric_columns).reset_index(drop=True)
    return df


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """将交易所 timeframe 转换为 Pandas 频率字符串。"""
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
    }
    return mapping.get(timeframe, "1h")


def build_future_timestamps(last_timestamp_ms: int, timeframe: str, pred_len: int) -> pd.Series:
    """基于最后一根 K 线时间戳，构建未来预测时间轴。"""
    start_ts = pd.to_datetime(last_timestamp_ms, unit="ms", utc=True)
    freq = timeframe_to_pandas_freq(timeframe)
    future_index = pd.date_range(start=start_ts, periods=pred_len + 1, freq=freq)[1:]
    return pd.Series(future_index)


def clamp_confidence(confidence: float) -> float:
    """约束置信度到 [0, 1] 区间。"""
    return float(max(0.0, min(1.0, confidence)))


def parse_bool_env(env_name: str, default: bool) -> bool:
    """按布尔语义解析环境变量。"""
    raw_value = os.getenv(env_name, str(default)).strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def should_use_indicator_testnet() -> bool:
    """读取 Indicator-MCP 专用 Testnet 开关，缺省时回退到全局 USE_TESTNET。"""
    if os.getenv("INDICATOR_USE_TESTNET") is not None:
        return parse_bool_env("INDICATOR_USE_TESTNET", False)
    return parse_bool_env("USE_TESTNET", False)


def has_explicit_kronos_local_config(local_config: dict[str, Any]) -> bool:
    """判断是否提供了可执行真实本地推理的完整配置。"""
    required_fields = (
        str(local_config.get("repo_path", "")).strip(),
        str(local_config.get("python_module", "")).strip(),
        str(local_config.get("model_id", "")).strip(),
        str(local_config.get("tokenizer_id", "")).strip(),
    )
    return all(required_fields)


def get_kronos_local_config() -> dict[str, Any]:
    """读取 Kronos 本地 Python 库推理配置。"""
    repo_path = os.getenv("KRONOS_LOCAL_REPO_PATH", "").strip()
    safe_repo_path = ""
    if repo_path:
        safe_repo_path = str(Path(repo_path).expanduser().resolve())

    return {
        "enabled": parse_bool_env("KRONOS_ENABLE_LOCAL", True),
        "require_local": parse_bool_env("KRONOS_REQUIRE_LOCAL", True),
        "repo_path": safe_repo_path,
        "python_module": os.getenv("KRONOS_PYTHON_MODULE", "model").strip() or "model",
        "model_id": os.getenv("KRONOS_MODEL_ID", "").strip(),
        "tokenizer_id": os.getenv("KRONOS_TOKENIZER_ID", "").strip(),
        "device": os.getenv("KRONOS_DEVICE", "auto").strip() or "auto",
        "max_context": int(os.getenv("KRONOS_MAX_CONTEXT", "512")),
        "lookback": int(os.getenv("KRONOS_LOOKBACK", "400")),
        "temperature": float(os.getenv("KRONOS_TEMPERATURE", "1.0")),
        "top_p": float(os.getenv("KRONOS_TOP_P", "0.9")),
        "sample_count": int(os.getenv("KRONOS_SAMPLE_COUNT", "1")),
    }


def validate_kronos_local_config(local_config: dict[str, Any]) -> dict[str, Any]:
    """校验 Kronos 本地 Python 库配置，便于 API/日志复用。"""
    errors: list[str] = []
    warnings: list[str] = []
    repo_path = str(local_config.get("repo_path", "")).strip()
    module_name = str(local_config.get("python_module", "model")).strip() or "model"
    model_id = str(local_config.get("model_id", "")).strip()
    tokenizer_id = str(local_config.get("tokenizer_id", "")).strip()
    explicit_local_configured = has_explicit_kronos_local_config(local_config)

    if not local_config.get("enabled", True):
        warnings.append("KRONOS_ENABLE_LOCAL=False，本地 Kronos 推理已被关闭。")

    if repo_path and not Path(repo_path).exists():
        errors.append(f"KRONOS_LOCAL_REPO_PATH 不存在: {repo_path}")

    for key in ("max_context", "lookback", "sample_count"):
        try:
            if int(local_config.get(key, 0)) <= 0:
                errors.append(f"{key} 必须大于 0")
        except (TypeError, ValueError):
            errors.append(f"{key} 必须是整数")

    for key in ("temperature", "top_p"):
        try:
            float(local_config.get(key, 0.0))
        except (TypeError, ValueError):
            errors.append(f"{key} 必须是数值")

    if not module_name:
        errors.append("KRONOS_PYTHON_MODULE 不能为空")

    if not local_config.get("enabled", True):
        warnings.append("真实 Kronos 集成已停用，将始终返回模拟/降级预测。")
    elif not explicit_local_configured:
        missing_fields: list[str] = []
        if not repo_path:
            missing_fields.append("KRONOS_LOCAL_REPO_PATH")
        if not model_id:
            missing_fields.append("KRONOS_MODEL_ID")
        if not tokenizer_id:
            missing_fields.append("KRONOS_TOKENIZER_ID")

        warnings.append(
            "未检测到完整本地模型配置，已禁用真实 Kronos 加载并切换到模拟/降级 JSON。"
        )
        if missing_fields:
            warnings.append(f"缺少配置项: {', '.join(missing_fields)}")

    return {
        "is_valid": len(errors) == 0 and explicit_local_configured and bool(local_config.get("enabled", True)),
        "errors": errors,
        "warnings": warnings,
        "explicit_local_configured": explicit_local_configured,
    }


def _build_uninitialized_kronos_status(disable_reason: str) -> dict[str, Any]:
    """构建未预热状态的 Kronos 运行时摘要，避免调用链路内再触发懒加载。"""
    local_config = get_kronos_local_config()
    config_validation = validate_kronos_local_config(local_config)
    device_resolution = resolve_kronos_device(str(local_config.get("device", "cpu")))
    merged_warnings = list(
        dict.fromkeys(
            list(config_validation.get("warnings", [])) + list(device_resolution.get("warnings", []))
        )
    )
    config_validation["warnings"] = merged_warnings
    return {
        "enabled": local_config.get("enabled", True),
        "require_local": local_config.get("require_local", True),
        "runtime_loaded": False,
        "python_module": local_config.get("python_module", "model"),
        "repo_path": local_config.get("repo_path", ""),
        "device": device_resolution.get("resolved_device", local_config.get("device", "cpu")),
        "requested_device": device_resolution.get("requested_device", local_config.get("device", "cpu")),
        "device_resolution": device_resolution,
        "model_id": local_config.get("model_id"),
        "tokenizer_id": local_config.get("tokenizer_id"),
        "max_context": local_config.get("max_context"),
        "config_validation": config_validation,
        "disable_reason": disable_reason,
        "fallback_reason": disable_reason,
    }


def reset_local_kronos_runtime_cache(
    preload: bool = True,
    load_origin: str = "manual_reset",
) -> None:
    """清空 Kronos 单例缓存，并按需立即重新预热。"""
    KRONOS_RUNTIME_STATE.update(
        {
            "initialized": False,
            "runtime": None,
            "status": _build_uninitialized_kronos_status("runtime_reset_pending_preload"),
            "load_origin": "reset_pending_preload",
            "initialized_at": None,
        }
    )
    if preload:
        preload_local_kronos_runtime(load_origin=load_origin)


def get_torch_backend_status() -> dict[str, Any]:
    """探测当前 Python 环境中的 Torch 后端可用性。"""
    backend_status = {
        "torch_available": False,
        "cuda_available": False,
        "mps_available": False,
    }
    try:
        torch_module = importlib.import_module("torch")
    except Exception:
        return backend_status

    backend_status["torch_available"] = True
    backend_status["cuda_available"] = bool(torch_module.cuda.is_available())
    backend_status["mps_available"] = bool(
        hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    )
    return backend_status


def resolve_kronos_device(requested_device: str) -> dict[str, Any]:
    """规范化 Kronos device 配置，兼容 auto/gpu/mps:0 等输入。"""
    raw_device = str(requested_device or "cpu").strip()
    normalized_device = raw_device.lower()
    backend_status = get_torch_backend_status()
    warnings: list[str] = []

    def _best_available_device() -> str:
        """按优先级选择当前环境最优设备。"""
        if backend_status["cuda_available"]:
            return "cuda:0"
        if backend_status["mps_available"]:
            return "mps"
        return "cpu"

    if normalized_device in {"", "auto", "default"}:
        resolved_device = _best_available_device()
    elif normalized_device == "gpu":
        if backend_status["cuda_available"]:
            resolved_device = "cuda:0"
        elif backend_status["mps_available"]:
            resolved_device = "mps"
            warnings.append("KRONOS_DEVICE=gpu 在当前 Apple Silicon 环境已自动映射为 mps。")
        else:
            resolved_device = "cpu"
            warnings.append("KRONOS_DEVICE=gpu 但未检测到可用 GPU，已回退为 cpu。")
    elif normalized_device.startswith("cuda"):
        if backend_status["cuda_available"]:
            resolved_device = raw_device
        else:
            resolved_device = _best_available_device()
            warnings.append(
                f"KRONOS_DEVICE={raw_device} 但当前环境 CUDA 不可用，已回退为 {resolved_device}。"
            )
    elif normalized_device.startswith("mps"):
        if backend_status["mps_available"]:
            resolved_device = "mps"
            if normalized_device != "mps":
                warnings.append(f"KRONOS_DEVICE={raw_device} 已规范化为 mps。")
        else:
            resolved_device = _best_available_device()
            warnings.append(
                f"KRONOS_DEVICE={raw_device} 但当前环境 MPS 不可用，已回退为 {resolved_device}。"
            )
    elif normalized_device == "cpu":
        resolved_device = "cpu"
    else:
        resolved_device = _best_available_device()
        warnings.append(
            f"KRONOS_DEVICE={raw_device} 非法或不可识别，已自动回退为 {resolved_device}。"
        )

    return {
        "requested_device": raw_device or "cpu",
        "resolved_device": resolved_device,
        "warnings": warnings,
        "backend_status": backend_status,
    }


def build_kronos_heuristic_forecast(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    pred_len: int,
    runtime_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在 Kronos 不可用时，使用轻量启发式模型生成降级预测。"""
    window_df = df.tail(min(len(df), 128)).copy()
    close_series = window_df["close"]
    last_close = float(close_series.iloc[-1])
    returns = close_series.pct_change().dropna()

    if returns.empty:
        trend = "neutral"
        confidence = 0.35
        drift = 0.0
        realized_vol = 0.0
    else:
        short_momentum = float(returns.tail(min(len(returns), 12)).mean())
        medium_drift = float(returns.tail(min(len(returns), 48)).mean())
        drift = (0.65 * short_momentum) + (0.35 * medium_drift)
        realized_vol = float(returns.tail(min(len(returns), 48)).std(ddof=0) or 0.0)
        projected_return = drift * max(pred_len / 8, 1)
        if projected_return > 0.003:
            trend = "bullish"
        elif projected_return < -0.003:
            trend = "bearish"
        else:
            trend = "neutral"

        signal_to_noise = abs(projected_return) / max(realized_vol, 1e-6)
        confidence = clamp_confidence(0.48 + min(signal_to_noise * 0.08, 0.24))

    projected_return = drift * max(pred_len / 8, 1)
    expected_close = max(last_close * (1 + projected_return), 1e-8)
    range_width_pct = max(realized_vol * np.sqrt(max(pred_len, 1)) * 1.2, 0.0035)
    low_target = max(expected_close * (1 - range_width_pct), 1e-8)
    high_target = max(expected_close * (1 + range_width_pct), low_target)
    last_timestamp = int(window_df["timestamp"].iloc[-1])
    fallback_reason = (
        (runtime_status or {}).get("disable_reason")
        or (runtime_status or {}).get("fallback_reason")
        or "Kronos local python runtime unavailable"
    )

    return {
        "status": "degraded",
        "model": "heuristic-fallback",
        "source": "heuristic_fallback",
        "symbol": symbol,
        "timeframe": timeframe,
        "prediction": {
            "trend": trend,
            "confidence": clamp_confidence(confidence),
            "target_range": [float(low_target), float(high_target)],
            "expected_close": float(expected_close),
            "horizon_bars": int(pred_len),
            "predicted_at": pd.Timestamp.utcnow().isoformat(),
            "forecast_timestamps": [
                ts.isoformat() for ts in build_future_timestamps(last_timestamp, timeframe, pred_len).tolist()
            ],
        },
        "meta": {
            "lookback_used": int(len(window_df)),
            "backend": "heuristic_fallback",
            "local_only": True,
            "fallback_reason": fallback_reason,
            "simulation_mode": True,
            "runtime_status": runtime_status or {},
        },
    }


def build_kronos_neutral_payload(
    symbol: str,
    timeframe: str,
    pred_len: int,
    reason: str,
    runtime_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在数据链路或模型链路不可用时返回中性预测。"""
    return {
        "status": "degraded",
        "model": "kronos-unavailable",
        "source": "service_unavailable",
        "symbol": symbol,
        "timeframe": timeframe,
        "prediction": {
            "trend": "neutral",
            "confidence": 0.0,
            "target_range": [],
            "expected_close": 0.0,
            "horizon_bars": int(pred_len),
            "predicted_at": pd.Timestamp.utcnow().isoformat(),
            "forecast_timestamps": [],
        },
        "meta": {
            "backend": "service_unavailable",
            "local_only": True,
            "fallback_reason": reason,
            "runtime_status": runtime_status or {},
        },
        "validation": {
            "is_valid": False,
            "errors": [reason],
            "warnings": [],
        },
    }

def preload_local_kronos_runtime(load_origin: str = "module_import") -> dict[str, Any] | None:
    """在 Indicator-MCP 初始化阶段预热 Kronos 单例，后续请求仅复用缓存。"""
    if KRONOS_RUNTIME_STATE["initialized"]:
        return KRONOS_RUNTIME_STATE["runtime"]

    KRONOS_RUNTIME_STATE["init_attempt_count"] += 1
    local_config = get_kronos_local_config()
    config_validation = validate_kronos_local_config(local_config)
    runtime: dict[str, Any] | None = None
    disable_reason = ""
    device_resolution = resolve_kronos_device(str(local_config.get("device", "cpu")))
    merged_warnings = list(
        dict.fromkeys(
            list(config_validation.get("warnings", [])) + list(device_resolution.get("warnings", []))
        )
    )
    config_validation["warnings"] = merged_warnings
    try:
        if not local_config.get("enabled", True):
            disable_reason = "KRONOS_ENABLE_LOCAL=False"
            raise RuntimeError(disable_reason)

        if not config_validation.get("explicit_local_configured", False):
            disable_reason = "Kronos local model is not fully configured"
            raise RuntimeError(disable_reason)

        if config_validation.get("errors"):
            raise ValueError("; ".join(config_validation.get("errors", [])))

        repo_path = str(local_config.get("repo_path", "")).strip()
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        kronos_module = importlib.import_module(str(local_config.get("python_module", "model")))
        kronos_tokenizer_cls = getattr(kronos_module, "KronosTokenizer")
        kronos_model_cls = getattr(kronos_module, "Kronos")
        kronos_predictor_cls = getattr(kronos_module, "KronosPredictor")

        tokenizer_id = str(local_config.get("tokenizer_id"))
        model_id = str(local_config.get("model_id"))
        device = str(device_resolution.get("resolved_device", "cpu"))
        max_context = int(local_config.get("max_context", 512))

        tokenizer = kronos_tokenizer_cls.from_pretrained(tokenizer_id)
        model = kronos_model_cls.from_pretrained(model_id)
        predictor = kronos_predictor_cls(
            model,
            tokenizer,
            device=device,
            max_context=max_context,
        )
        runtime = {
            "predictor": predictor,
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "max_context": max_context,
            "device": device,
            "requested_device": device_resolution.get("requested_device"),
            "device_resolution": device_resolution,
            "python_module": str(local_config.get("python_module", "model")),
            "repo_path": repo_path,
            "config_validation": config_validation,
        }
        logger.info(
            f"✅ 本地 Kronos 模型已加载: {model_id} "
            f"module={local_config.get('python_module')} repo={repo_path or '<PYTHONPATH>'} "
            f"device={device} origin={load_origin}"
        )
        for device_warning in device_resolution.get("warnings", []):
            logger.warn(f"Kronos device 调整: {device_warning}")
    except Exception as exc:
        logger.warn(
            f"本地 Kronos Python 库未就绪，将使用启发式回退: {exc} "
            f"(origin={load_origin})"
        )
        if not disable_reason:
            disable_reason = str(exc)

    KRONOS_RUNTIME_STATE.update(
        {
            "initialized": True,
            "runtime": runtime,
            "status": {
                "enabled": local_config.get("enabled", True),
                "require_local": local_config.get("require_local", True),
                "runtime_loaded": runtime is not None,
                "python_module": local_config.get("python_module", "model"),
                "repo_path": local_config.get("repo_path", ""),
                "device": device_resolution.get("resolved_device", local_config.get("device", "cpu")),
                "requested_device": device_resolution.get(
                    "requested_device",
                    local_config.get("device", "cpu"),
                ),
                "device_resolution": device_resolution,
                "model_id": local_config.get("model_id"),
                "tokenizer_id": local_config.get("tokenizer_id"),
                "max_context": local_config.get("max_context"),
                "config_validation": config_validation,
                "disable_reason": disable_reason,
                "fallback_reason": disable_reason,
                "load_origin": load_origin,
            },
            "load_origin": load_origin,
            "initialized_at": pd.Timestamp.utcnow().isoformat(),
        }
    )
    return runtime


def get_local_kronos_runtime() -> dict[str, Any] | None:
    """返回已预热的 Kronos 单例运行时，不再在请求路径内触发冷加载。"""
    return KRONOS_RUNTIME_STATE["runtime"]


def get_local_kronos_runtime_status() -> dict[str, Any]:
    """返回本地 Kronos 运行时状态，用于 API 与降级链路透传。"""
    if KRONOS_RUNTIME_STATE["status"]:
        return KRONOS_RUNTIME_STATE["status"]
    return _build_uninitialized_kronos_status("runtime_not_preloaded")


def get_local_kronos_runtime_metrics() -> dict[str, Any]:
    """返回 Kronos 预热指标，便于验证请求路径是否仍存在冷加载。"""
    return {
        "initialized": KRONOS_RUNTIME_STATE["initialized"],
        "init_attempt_count": KRONOS_RUNTIME_STATE["init_attempt_count"],
        "load_origin": KRONOS_RUNTIME_STATE["load_origin"],
        "initialized_at": KRONOS_RUNTIME_STATE["initialized_at"],
        "runtime_loaded": bool(KRONOS_RUNTIME_STATE["runtime"]),
    }


preload_local_kronos_runtime(load_origin="module_import")




async def predict_with_local_kronos(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    pred_len: int,
) -> dict[str, Any] | None:
    """使用本地 Kronos 模型执行预测。"""
    feature_columns = ["open", "high", "low", "close", "volume"]

    def _run_prediction() -> tuple[dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame] | None:
        """在线程中执行本地模型加载与推理，避免阻塞事件循环。"""
        runtime = get_local_kronos_runtime()
        if not runtime:
            return None

        predictor = runtime["predictor"]
        max_context = int(runtime["max_context"])
        working_df = df.tail(min(len(df), max_context)).copy()
        x_timestamp = pd.to_datetime(working_df["timestamp"], unit="ms", utc=True)
        y_timestamp = build_future_timestamps(int(working_df["timestamp"].iloc[-1]), timeframe, pred_len)
        prediction_df = predictor.predict(
            df=working_df[feature_columns],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=float(os.getenv("KRONOS_TEMPERATURE", "1.0")),
            top_p=float(os.getenv("KRONOS_TOP_P", "0.9")),
            sample_count=int(os.getenv("KRONOS_SAMPLE_COUNT", "1")),
        )
        return runtime, working_df, y_timestamp, prediction_df

    try:
        prediction_timeout = get_kronos_local_timeout_seconds()
        prediction_result = await asyncio.wait_for(
            asyncio.to_thread(_run_prediction),
            timeout=prediction_timeout,
        )
        if prediction_result is None:
            return None

        runtime, working_df, y_timestamp, prediction_df = prediction_result
        if prediction_df is None or prediction_df.empty:
            return None

        prediction_df = prediction_df.reset_index(drop=True)
        last_close = float(working_df["close"].iloc[-1])
        expected_close = float(prediction_df["close"].iloc[-1])
        low_target = float(prediction_df["low"].min())
        high_target = float(prediction_df["high"].max())
        close_delta = (expected_close - last_close) / max(last_close, 1e-8)
        volatility_proxy = float(working_df["close"].pct_change().dropna().tail(48).std(ddof=0) or 0.0)

        if close_delta > 0.002:
            trend = "bullish"
        elif close_delta < -0.002:
            trend = "bearish"
        else:
            trend = "neutral"

        confidence = clamp_confidence(0.55 + min(abs(close_delta) / max(volatility_proxy, 1e-6) * 0.06, 0.25))
        return {
            "status": "success",
            "model": runtime["model_id"],
            "source": "local_model",
            "symbol": symbol,
            "timeframe": timeframe,
            "prediction": {
                "trend": trend,
                "confidence": confidence,
                "target_range": [low_target, high_target],
                "expected_close": expected_close,
                "horizon_bars": int(pred_len),
                "predicted_at": pd.Timestamp.utcnow().isoformat(),
                "forecast_timestamps": [ts.isoformat() for ts in y_timestamp.tolist()],
            },
            "meta": {
                "lookback_used": int(len(working_df)),
                "backend": "local_python_library",
                "local_only": True,
                "python_module": runtime.get("python_module"),
                "repo_path": runtime.get("repo_path"),
                "device": runtime.get("device"),
                "requested_device": runtime.get("requested_device"),
                "device_resolution": runtime.get("device_resolution"),
                "max_context": runtime.get("max_context"),
                "config_validation": runtime.get("config_validation", {}),
            },
        }
    except asyncio.TimeoutError:
        logger.warn(
            f"本地 Kronos 推理超时，已切换启发式回退: "
            f"symbol={symbol} timeframe={timeframe} timeout={get_kronos_local_timeout_seconds():.1f}s"
        )
        return None
    except Exception as exc:
        logger.warn(f"本地 Kronos 预测失败，将进入回退路径: {exc}")
        return None


def validate_kronos_payload(
    payload: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """验证并标准化 Kronos 预测输出。"""
    normalized = payload.copy()
    prediction = normalized.setdefault("prediction", {})
    validation_errors: list[str] = []
    validation_warnings: list[str] = []

    trend = str(prediction.get("trend", "neutral")).lower()
    if trend not in {"bullish", "bearish", "neutral"}:
        validation_errors.append(f"非法 trend: {trend}")
        trend = "neutral"
    prediction["trend"] = trend

    try:
        confidence = clamp_confidence(float(prediction.get("confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
        validation_errors.append("confidence 不是有效数字")
    prediction["confidence"] = confidence

    target_range = prediction.get("target_range", [])
    if not isinstance(target_range, list) or len(target_range) != 2:
        validation_errors.append("target_range 缺失或格式非法")
        low_target = current_price
        high_target = current_price
    else:
        try:
            low_target = float(target_range[0])
            high_target = float(target_range[1])
            low_target, high_target = sorted([low_target, high_target])
        except (TypeError, ValueError):
            validation_errors.append("target_range 无法转换为数值")
            low_target = current_price
            high_target = current_price

    if low_target <= 0 or high_target <= 0:
        validation_errors.append("target_range 必须为正数")
        low_target = max(current_price, 1e-8)
        high_target = max(current_price, 1e-8)

    if current_price > 0:
        move_ratio = max(abs(low_target - current_price), abs(high_target - current_price)) / current_price
        if move_ratio > 0.25:
            validation_warnings.append("预测区间偏离当前价格超过 25%，已标记为高风险输出")

    prediction["target_range"] = [float(low_target), float(high_target)]
    prediction["expected_close"] = float(prediction.get("expected_close", (low_target + high_target) / 2))
    normalized["validation"] = {
        "is_valid": len(validation_errors) == 0,
        "errors": validation_errors,
        "warnings": validation_warnings,
    }
    return normalized


def build_kronos_guardrail(
    kronos_payload: dict[str, Any],
    hmm_regime: str,
    smc_result: dict[str, Any],
) -> dict[str, Any]:
    """根据 Kronos/HMM/SMC 共振关系生成执行降级策略。"""
    prediction = kronos_payload.get("prediction", {})
    validation = kronos_payload.get("validation", {})
    trend = prediction.get("trend", "neutral")
    confidence = float(prediction.get("confidence", 0.0))
    source = str(kronos_payload.get("source", "unknown"))
    smc_structure = smc_result.get("structure", {}).get("structure", "UNKNOWN")
    smc_event = smc_result.get("structure", {}).get("last_event", "NONE")

    risk_level = "normal"
    forced_action = "ALLOW"
    execution_allowed = True
    position_scale = 1.0
    reasons: list[str] = []

    if not validation.get("is_valid", False):
        risk_level = "critical"
        forced_action = "WAIT"
        execution_allowed = False
        position_scale = 0.0
        reasons.append("Kronos 输出校验失败")
    elif trend == "neutral":
        risk_level = "high"
        forced_action = "WAIT"
        execution_allowed = False
        position_scale = 0.0
        reasons.append("Kronos 未给出明确方向")
    elif confidence < 0.6:
        risk_level = "high"
        forced_action = "WAIT"
        execution_allowed = False
        position_scale = 0.0
        reasons.append(f"Kronos 置信度过低({confidence:.2f})")
    elif source == "heuristic_fallback":
        risk_level = "medium"
        forced_action = "REDUCE_SIZE"
        position_scale = 0.25
        reasons.append("Kronos 已退化为启发式回退，仅允许 25% 仓位试探")

    if smc_structure == "BULL" and trend == "bearish":
        risk_level = "high"
        forced_action = "WAIT"
        execution_allowed = False
        position_scale = 0.0
        reasons.append("Kronos 与 SMC 方向冲突(BULL vs bearish)")
    elif smc_structure == "BEAR" and trend == "bullish":
        risk_level = "high"
        forced_action = "WAIT"
        execution_allowed = False
        position_scale = 0.0
        reasons.append("Kronos 与 SMC 方向冲突(BEAR vs bullish)")

    if hmm_regime == "QUIET_SIDEWAYS" and execution_allowed:
        risk_level = "medium" if risk_level == "normal" else risk_level
        forced_action = "REDUCE_SIZE"
        position_scale = min(position_scale, 0.5)
        reasons.append("HMM=QUIET_SIDEWAYS，限制为半仓以下试探")

    if hmm_regime == "UNKNOWN":
        risk_level = "high"
        forced_action = "WAIT"
        execution_allowed = False
        position_scale = 0.0
        reasons.append("HMM 状态未知，禁止执行")

    system_instruction = "✅ Kronos/HMM/SMC 共振通过，可按标准流程执行。"
    if forced_action == "REDUCE_SIZE":
        system_instruction = (
            f"⚠️ 风险降级：仅允许执行 {position_scale:.0%} 标准仓位，"
            "并优先选择更保守的入场与止损。"
        )
    elif forced_action == "WAIT":
        system_instruction = (
            "🚨 风险降级：Kronos 校验未通过或多模型冲突。"
            "请强制输出 <ACTION>WAIT</ACTION>，禁止下单。"
        )

    return {
        "risk_level": risk_level,
        "forced_action": forced_action,
        "execution_allowed": execution_allowed,
        "position_scale": float(position_scale),
        "reasons": reasons,
        "hmm_regime": hmm_regime,
        "smc_structure": smc_structure,
        "smc_event": smc_event,
        "kronos_trend": trend,
        "kronos_confidence": confidence,
        "system_instruction": system_instruction,
    }


async def build_kronos_prediction(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    pred_len: int,
    current_price: float,
) -> dict[str, Any]:
    """统一构建 Kronos 预测，仅使用本地 Python 库并在失败时启发式回退。"""
    local_prediction = await predict_with_local_kronos(symbol, timeframe, df, pred_len)
    if local_prediction:
        return validate_kronos_payload(local_prediction, current_price)

    runtime_status = get_local_kronos_runtime_status()
    heuristic_prediction = build_kronos_heuristic_forecast(
        symbol=symbol,
        timeframe=timeframe,
        df=df,
        pred_len=pred_len,
        runtime_status=runtime_status,
    )
    validated_prediction = validate_kronos_payload(heuristic_prediction, current_price)
    if runtime_status.get("config_validation", {}).get("errors"):
        validated_prediction.setdefault("validation", {}).setdefault("warnings", []).append(
            "本地 Kronos 配置未通过校验，当前结果已降级为启发式预测。"
        )
    return validated_prediction

# @mcp.tool()
async def analyze_smc(symbol: str, timeframe: str = '1h') -> str:
    """
    基于 SMC (Smart Money Concepts) 分析市场结构。
    """
    start_time = time.time()
    logger.info(f"Received SMC analysis request for {symbol} on {timeframe}")
    
    config = get_exchange_config()
    exchange = build_exchange_from_config(config)
    if should_use_indicator_testnet():
        exchange.set_sandbox_mode(True)
    
    try:
        await exchange.load_markets()
        # 1. 获取 K 线数据 (默认最近 100 根足够分析近期结构)
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if not ohlcv:
            raise ValueError(f"No OHLCV data returned for {symbol}")
            
        # 转换为 DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. 运行 SMC 引擎
        engine = SMCEngine(df)
        analysis_result = engine.analyze()
        
        structure_data = analysis_result['structure']
        fvg_data = analysis_result['fvg']
        
        # 3. 映射到 shared.models.MarketContext
        # 将字符串枚举转为 Enum 对象
        regime_map = {
            "BULL": MarketRegime.BULL,
            "BEAR": MarketRegime.BEAR,
            "SIDEWAYS": MarketRegime.SIDEWAYS
        }
        regime = regime_map.get(structure_data['structure'], MarketRegime.SIDEWAYS)
        
        # 简单估算波动率 (ATR 简化版：High - Low 的平均值)
        atr = (df['high'] - df['low']).mean()
        volatility = VolatilityLevel.MEDIUM # 暂时硬编码，后续可接 ATR 逻辑
        
        # 提取支撑阻力位
        support_levels = []
        if structure_data['key_support']:
            support_levels.append(structure_data['key_support'])
            
        resistance_levels = []
        if structure_data['key_resistance']:
            resistance_levels.append(structure_data['key_resistance'])
            
        # 生成脱水 Insight
        insight = f"当前市场结构为 {structure_data['structure']}。"
        if structure_data['last_event'] != "NONE":
            insight += f" 最近发生了 {structure_data['last_event']}。"
            
        if fvg_data['has_fvg']:
            insight += f" 并在 {fvg_data['gap_bottom']:.2f}-{fvg_data['gap_top']:.2f} 存在 {fvg_data['type']} FVG 缺口，"
            if fvg_data['type'] == 'BULLISH':
                insight += "建议在此区间寻找顺势买入机会。"
                support_levels.append(fvg_data['gap_top']) # FVG 上沿作为支撑
            else:
                insight += "建议在此区间寻找顺势卖出机会。"
                resistance_levels.append(fvg_data['gap_bottom']) # FVG 下沿作为阻力
        else:
            insight += " 附近无明显 FVG 缺口，建议观望。"

        # 构造最终 Context
        context = MarketContext(
            symbol=symbol,
            regime=regime,
            volatility=volatility,
            atr=float(atr),
            support_levels=[float(x) for x in support_levels],
            resistance_levels=[float(x) for x in resistance_levels],
            insight=f"[SMC分析] {insight}"
        )
        
        elapsed = time.time() - start_time
        logger.info(f"SMC Analysis completed for {symbol} in {elapsed:.4f}s")
        
        return context.model_dump_json()

    except Exception as e:
        logger.error(f"Error in analyze_smc: {str(e)}")
        return MCPErrorResponse(
            status="error",
            error_code="SMC_ANALYSIS_ERROR",
            message=str(e)
        ).model_dump_json()
        
    finally:
        await exchange.close()

def interpret_hmm_state(means: np.ndarray, _covars: np.ndarray, current_state: int) -> tuple[MarketRegime, VolatilityLevel, str]:
    """
    辅助函数：根据 HMM 模型参数解读当前状态含义。
    
    逻辑：
    - 比较各状态的波动率 (range 的均值) 和 收益率 (log_return 的方差)
    - 波动率最高的状态 -> VOLATILE_TREND
    - 波动率最低的状态 -> QUIET_SIDEWAYS
    - 其他 -> REVERSAL_ZONE / NORMAL
    """
    # means 结构: [n_components, n_features] -> 特征顺序 [log_return, range]
    # 我们主要关注第二个特征 'range' (波动率) 的均值来区分状态
    
    volatility_means = means[:, 1] # 获取所有状态的波动率均值
    
    # 排序状态：按波动率从小到大
    sorted_indices = np.argsort(volatility_means)
    
    # 定义状态映射
    state_labels = {}
    
    # 波动率最小的状态 -> 低波动震荡
    state_labels[sorted_indices[0]] = {
        "regime": MarketRegime.SIDEWAYS,
        "volatility": VolatilityLevel.LOW,
        "desc": "QUIET_SIDEWAYS (安静震荡)"
    }
    
    # 波动率最大的状态 -> 高波动趋势
    state_labels[sorted_indices[-1]] = {
        "regime": MarketRegime.BULL if means[sorted_indices[-1], 0] > 0 else MarketRegime.BEAR, # 简单通过收益率均值正负判断方向
        "volatility": VolatilityLevel.HIGH,
        "desc": "VOLATILE_TREND (高波动趋势)"
    }
    
    # 中间状态 -> 均值回归/反转区
    if len(sorted_indices) > 2:
        state_labels[sorted_indices[1]] = {
            "regime": MarketRegime.SIDEWAYS,
            "volatility": VolatilityLevel.MEDIUM,
            "desc": "REVERSAL_ZONE (反转/过渡区)"
        }
        
    result = state_labels.get(current_state, {
        "regime": MarketRegime.SIDEWAYS,
        "volatility": VolatilityLevel.MEDIUM,
        "desc": "UNKNOWN"
    })
    
    return result["regime"], result["volatility"], result["desc"]

# @mcp.tool()
async def get_market_regime_hmm(symbol: str, timeframe: str = '1h') -> str:
    """
    使用 HMM (隐马尔可夫模型) 识别当前市场状态 (Regime)。
    """
    start_time = time.time()
    logger.info(f"Received HMM regime analysis request for {symbol} on {timeframe}")
    
    config = get_exchange_config()
    exchange = build_exchange_from_config(config)
    if should_use_indicator_testnet():
        exchange.set_sandbox_mode(True)
    
    try:
        await exchange.load_markets()
        # 1. 获取足够的历史数据用于训练 (500根)
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=500)
        
        if not ohlcv or len(ohlcv) < 100:
            raise ValueError(f"Insufficient OHLCV data for HMM training: {len(ohlcv) if ohlcv else 0}")
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. 运行 HMM 引擎
        engine = HMMEngine(n_components=3)
        
        train_start = time.time()
        hmm_result = engine.train_and_predict(df)
        train_elapsed = time.time() - train_start
        
        if train_elapsed > 0.5:
            logger.warn(f"HMM training took {train_elapsed:.4f}s (> 0.5s) for {symbol}")
            
        if not hmm_result:
            raise ValueError("HMM model failed to converge or predict.")
            
        # 3. 解读状态
        regime, volatility, state_desc = interpret_hmm_state(
            hmm_result['means'], 
            hmm_result['covars'], 
            hmm_result['current_state']
        )
        
        # 4. 生成 Insight
        insight = f"HMM 模型识别当前市场处于: {state_desc}。"
        
        if state_desc.startswith("VOLATILE_TREND"):
            insight += " 波动率正在扩张，建议缩减杠杆并切换至突破策略，注意设置移动止损。"
        elif state_desc.startswith("QUIET_SIDEWAYS"):
            insight += " 市场波动率低，适合网格交易或区间刷单策略。"
        elif state_desc.startswith("REVERSAL_ZONE"):
            insight += " 市场处于过渡阶段，可能面临变盘或均值回归，建议降低仓位观望。"
            
        # 构造 Context
        # 注意：HMM 不直接提供支撑阻力位，这里留空或复用简单计算
        atr = (df['high'] - df['low']).mean()
        
        context = MarketContext(
            symbol=symbol,
            regime=regime,
            volatility=volatility,
            atr=float(atr),
            support_levels=[], # HMM 不负责 SR
            resistance_levels=[],
            insight=f"[HMM分析] {insight}"
        )
        
        elapsed = time.time() - start_time
        logger.info(f"HMM Analysis completed for {symbol} in {elapsed:.4f}s (State: {hmm_result['current_state']})")
        
        return context.model_dump_json()

    except Exception as e:
        logger.error(f"Error in get_market_regime_hmm: {str(e)}")
        return MCPErrorResponse(
            status="error",
            error_code="HMM_ANALYSIS_ERROR",
            message=str(e)
        ).model_dump_json()
        
    finally:
        await exchange.close()

# @mcp.tool()
async def analyze_orderflow(symbol: str) -> str:
    """
    分析订单流 (Orderflow) 数据，包括 OFI 和成交主动性。
    """
    start_time = time.time()
    logger.info(f"Received Orderflow analysis request for {symbol}")
    
    config = get_exchange_config()
    exchange = build_exchange_from_config(config)
    if should_use_indicator_testnet():
        exchange.set_sandbox_mode(True)
    
    try:
        await exchange.load_markets()
        # 并发获取 盘口(Orderbook) 和 成交(Trades)
        # fetch_order_book limit=20, fetch_trades limit=100
        task_ob = exchange.fetch_order_book(symbol, limit=20)
        task_trades = exchange.fetch_trades(symbol, limit=100)
        
        orderbook, trades = await asyncio.gather(task_ob, task_trades)
        
        # 1. 计算 OFI
        ofi = orderflow_engine.calculate_ofi(symbol, orderbook)
        
        # 归一化 OFI (简单缩放，实际可能需要历史 Rolling Window Z-score)
        # 这里为了直观，假设单次 tick 变化量通常在一定范围内，简单 clip 到 -1 ~ 1 用于展示
        # 注意：这里的 OFI 是绝对量，归一化通常需要历史数据，这里仅做符号和量级参考
        # 暂且用 sigmoid 或简单分段来映射 score
        ofi_score = np.tanh(ofi / 100.0) # 假设 100 是一个较大的 volume 变动单位
        
        # 2. 分析成交主动性
        dominance_info = orderflow_engine.analyze_trades_dominance(trades)
        dominance = dominance_info['dominance']
        buy_ratio = dominance_info['buy_ratio']
        
        # 3. 综合信号
        pressure_signal = orderflow_engine.get_pressure_signal(ofi, dominance)
        
        # 4. 生成 Insight
        insight = f"Orderflow 分析: OFI={ofi:.2f}, 主动买入占比={buy_ratio:.0%} ({dominance})。"
        insight += f" 结论: {pressure_signal}。"
        
        if ofi > 0 and dominance == "BULLISH":
            insight += " [强劲看多] 挂单支撑增加且主动买盘强劲。"
        elif ofi < 0 and dominance == "BEARISH":
            insight += " [强劲看空] 挂单抛压增加且主动卖盘强劲。"
            
        # 构造返回结果 (使用 dict 转 json，或者定义新的 Pydantic 模型)
        # 这里为了灵活，直接返回结构化字典
        result = {
            "symbol": symbol,
            "ofi_raw": ofi,
            "ofi_score": float(ofi_score),
            "trade_dominance": dominance,
            "buy_ratio": buy_ratio,
            "pressure_signal": pressure_signal,
            "insight": insight,
            "timestamp": int(time.time() * 1000)
        }
        
        elapsed = time.time() - start_time
        logger.info(f"Orderflow Analysis completed for {symbol} in {elapsed:.4f}s")
        
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in analyze_orderflow: {str(e)}")
        return MCPErrorResponse(
            status="error",
            error_code="ORDERFLOW_ERROR",
            message=str(e)
        ).model_dump_json()
        
    finally:
        await exchange.close()

async def get_cached_ohlcv(ex, symbol, timeframe, limit=500):
    """
    通过全局 60 秒内存缓存与异步并发锁获取 OHLCV，避免同 Key 并发双请求。
    缓存 Key 仅由 symbol + timeframe 构成，底层统一拉取 500 根后在内存中切片返回。

    Args:
        ex: CCXT 异步交易所实例。
        symbol: 交易对。
        timeframe: K 线周期。
        limit: 拉取条数。

    Returns:
        list[list[float]]: 标准化前的原始 OHLCV 数据。
    """
    requested_limit = max(int(limit or 500), 1)
    fetch_limit = 500
    cache_key = f"{symbol}_{timeframe}"

    async def _load_ohlcv() -> list[list[float]]:
        """
        在缓存未命中时统一拉取 500 根 OHLCV，供不同 limit 请求共享同一缓存。
        """
        log_indicator_fetch_event(
            route_name=get_indicator_exchange_route_name(ex),
            action="Cache Miss 拉取 OHLCV",
            symbol=symbol,
            timeframe=timeframe,
            extra=f"requested_limit={requested_limit}, fetch_limit={fetch_limit}",
        )
        logger.info(f"🌐 [Cache Miss] 从交易所拉取真实数据: OHLCV {symbol}")
        return await ex.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)

    cached_ohlcv = await get_or_set_ttl_cache(
        cache=ohlcv_cache,
        key=cache_key,
        loader=_load_ohlcv,
        request_locks=ohlcv_request_locks,
        cache_name="ohlcv_cache",
    )
    return list(cached_ohlcv[-requested_limit:])

@cached(cache=market_cache, key=cache_key_builder)
async def get_cached_orderbook(ex, symbol, limit=20):
    """在缓存未命中时拉取 orderbook，并输出来源标签日志。"""
    log_indicator_fetch_event(
        route_name=get_indicator_exchange_route_name(ex),
        action="Cache Miss 拉取 OrderBook",
        symbol=symbol,
        extra=f"limit={limit}",
    )
    return await ex.fetch_order_book(symbol, limit=limit)

@cached(cache=market_cache, key=cache_key_builder)
async def get_cached_ticker(ex, symbol):
    """在缓存未命中时拉取 ticker，并输出来源标签日志。"""
    log_indicator_fetch_event(
        route_name=get_indicator_exchange_route_name(ex),
        action="Cache Miss 拉取 Ticker",
        symbol=symbol,
    )
    return await ex.fetch_ticker(symbol)

@cached(cache=market_cache, key=cache_key_builder)
async def get_cached_trades(ex, symbol, limit=200):
    """在缓存未命中时拉取 trades，并输出来源标签日志。"""
    log_indicator_fetch_event(
        route_name=get_indicator_exchange_route_name(ex),
        action="Cache Miss 拉取 Trades",
        symbol=symbol,
        extra=f"limit={limit}",
    )
    return await ex.fetch_trades(symbol, limit=limit)



@mcp.tool()
async def get_kronos_prediction(
    symbol: str,
    timeframe: str = "1h",
    pred_len: int = 24,
) -> str:
    """生成 Kronos 预测结果，并在本地 Python 库不可用时自动启发式回退。"""
    try:
        ohlcv, market_route = await fetch_rest_ohlcv(
            symbol,
            timeframe,
            limit=min(max(int(os.getenv("KRONOS_LOOKBACK", "400")), 120), 500),
        )
        if not ohlcv:
            raise ValueError(f"Kronos 输入数据为空: {symbol} {timeframe}")

        df = normalize_ohlcv_dataframe(
            pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        )
        current_price = float(df["close"].iloc[-1])
        prediction = await build_kronos_prediction(symbol, timeframe, df, pred_len, current_price)
        prediction.setdefault("meta", {})
        prediction["meta"]["market_data_source"] = f"rest_{market_route}"
        logger.info(
            f"✅ Kronos 预测完成: {symbol} {timeframe} "
            f"source={prediction.get('source')} status={prediction.get('status')}"
        )
        return json.dumps(prediction, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"get_kronos_prediction 执行失败: {exc}")
        neutral_payload = build_kronos_neutral_payload(
            symbol=symbol,
            timeframe=timeframe,
            pred_len=pred_len,
            reason=f"Kronos unavailable: {exc}",
            runtime_status=get_local_kronos_runtime_status(),
        )
        return json.dumps(neutral_payload, ensure_ascii=False)

@mcp.tool()
async def get_full_market_context(symbol: str, timeframe: str = "1h") -> str:
    """
    [最高优先级分析工具] 一键获取指定资产的所有市场维度的脱水研报！
    当你需要分析行情时，必须且只需调用此工具一次。它会同时返回：
    当前价格、HMM市场状态、SMC结构与FVG缺口、以及Orderflow订单流不平衡(OFI)数据。
    """
    logger.info(f"🚀 启动 Fat-Tool 聚合分析: {symbol} @ {timeframe}")

    try:
        engine_priority = get_indicator_engine_priority()
        logger.info(f"⚙️ Indicator 双引擎优先级: {engine_priority}")

        logger.info("🌐 正在执行 Indicator 固定路由: localhost:8000 -> 官方 SDK")
        ohlcv, orderbook, ticker, trades, route_name = await fetch_rest_market_snapshot(
            symbol=symbol,
            timeframe=timeframe,
        )
        market_data_source = route_name
        fallback_reasons: list[str] = []
        if route_name.startswith("sdk_"):
            fallback_reasons.append("local_httpx_unavailable")
            logger.info(f"✅ Indicator 回退命中官方 SDK: {symbol} route={route_name}")
        else:
            logger.info(f"✅ Indicator 首选命中 localhost HTTP: {symbol} route={route_name}")

        if not ohlcv or not orderbook.get('bids'):
            logger.error(f"{symbol} localhost HTTP + 官方 SDK 均未返回完整数据。")
            return json.dumps({
                "error": "行情数据链路不可用",
                "market_data_source": market_data_source,
                "fallback_reasons": fallback_reasons,
                "system_instruction": "🚨 底层数据流连接不稳定，无法获取完整行情。请强制输出 <ACTION>WAIT</ACTION> 并结束本次分析。"
            }, ensure_ascii=False)

        df = normalize_ohlcv_dataframe(
            pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        )
        current_price = float(ticker.get('last') or df['close'].iloc[-1])
        
        # 3. 实例化并调用算力引擎 (CPU 密集型操作)
        logger.info("🧠 正在进行高维特征矩阵运算 (HMM + SMC + OFI)...")
        
        # SMC
        smc_engine = SMCEngine(df)
        smc_res = smc_engine.analyze()
        
        # HMM
        hmm_res = hmm_engine.train_and_predict(df)
        # hmm_res 已经是 "QUIET_SIDEWAYS" 这样的字符串了
        
        # OFI & Aggressor
        ofi_score = orderflow_engine.calculate_ofi(symbol, orderbook)
        aggressor_data = orderflow_engine.analyze_aggressor_trades(trades) # 计算真实成交
        kronos_runtime_status = getattr(get_local_kronos_runtime, "_status", {}) or {}
        kronos_prediction = validate_kronos_payload(
            build_kronos_heuristic_forecast(
                symbol=symbol,
                timeframe=timeframe,
                df=df,
                pred_len=int(os.getenv("KRONOS_DEFAULT_PRED_LEN", "24")),
                runtime_status=kronos_runtime_status,
            ),
            current_price,
        )
        kronos_prediction.setdefault("meta", {})
        kronos_prediction["meta"]["prefetch_mode"] = "fat_tool_fast_path"
        risk_guardrail = build_kronos_guardrail(kronos_prediction, hmm_res, smc_res)
        
        # ==========================================
        # 3. 记忆流对比与动态 Prompt 生成
        # ==========================================
        state_key = f"{symbol}_{timeframe}"
        previous_state = STATE_MEMORY.get(state_key)
        
        # 判断是否发生状态切换 (第一次运行也视为 true)
        is_state_changed = (previous_state != hmm_res)
        
        # 更新记忆到最新状态
        STATE_MEMORY[state_key] = hmm_res
        
        # 为大模型生成动态指令
        if is_state_changed:
            instruction = f"🚨 警告：环境已从 {previous_state or '未知'} 切换为 {hmm_res}！请立即执行【路径 B：宏观环境切换】进行深度共振重估。"
            logger.warn(f"变盘预警! {symbol} 状态从 {previous_state} -> {hmm_res}")
        else:
            instruction = f"✅ 环境稳定维持在 {hmm_res}。请执行【路径 A：环境稳定期】仅做轻量级风控校验，快速响应，禁止长篇大论。"
            logger.info(f"环境稳定 ({hmm_res})，已下发轻量级风控指令。")

        fvg_data = smc_res.get("fvg", {})
        if fvg_data.get("has_fvg"):
            fvg_insight = f"{fvg_data.get('type')} FVG ({fvg_data.get('gap_bottom')} - {fvg_data.get('gap_top')})"
        else:
            fvg_insight = "No FVG"

        # 5. 组装终极脱水研报 (Fat Payload)
        # 【极致压缩】：仅保留结论性数据，剔除任何原始数组
        fat_payload = {
            "asset": symbol,
            "current_price": current_price,
            "state_changed": is_state_changed,
            "market_data_source": market_data_source,
            "market_regime": hmm_res,
            "previous_regime": previous_state,
            "kronos_forecast": kronos_prediction,
            "risk_guardrail": risk_guardrail,
            "smart_money_concepts": {
                "structure": smc_res.get("structure", {}).get("structure", "UNKNOWN"),
                "key_support": smc_res.get("structure", {}).get("key_support"),
                "key_resistance": smc_res.get("structure", {}).get("key_resistance"),
                "fvg_insight": fvg_insight
            } if isinstance(smc_res, dict) else smc_res,
            "orderflow": {
                "ofi_score": ofi_score,
                "aggressor_dominance": aggressor_data.get("dominance", "UNKNOWN")
            },
            "system_instruction": f"{instruction}\n{risk_guardrail.get('system_instruction')}",
            "validation": {
                "kronos_valid": kronos_prediction.get("validation", {}).get("is_valid", False),
                "kronos_warnings": kronos_prediction.get("validation", {}).get("warnings", []),
            }
        }
        
        logger.info(
            f"✅ Fat-Tool 聚合分析完成: source={market_data_source} "
            f"kronos={kronos_prediction.get('source')} "
            f"forced_action={risk_guardrail.get('forced_action')}"
        )
        return json.dumps(fat_payload, ensure_ascii=False)
        
    except BinanceNetworkError as ne:
        logger.error(f"网络异常 (NetworkError): {ne}")
        return json.dumps({
            "error": "Network Connection Failed",
            "system_instruction": "🚨 交易所 API 连接超时或被拒绝。禁止任何分析，强制输出：<ACTION>WAIT</ACTION>"
        }, ensure_ascii=False)
    except BinanceApiError as ee:
        logger.error(f"交易所异常 (ExchangeError): {ee}")
        return json.dumps({
            "error": "Exchange API Error",
            "system_instruction": "🚨 交易所内部错误或处于维护状态。禁止任何分析，强制输出：<ACTION>WAIT</ACTION>"
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Fat-Tool 聚合执行失败: {str(e)}")
        return json.dumps({"error": f"聚合分析失败: {str(e)}"}, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
