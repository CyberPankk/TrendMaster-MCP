import asyncio
from cachetools import TTLCache
from asyncache import cached
from shared.logger import get_logger

logger = get_logger("Cache-Manager")

# 定义各类数据的专用缓存池
# maxsize=100 表示最多缓存 100 个交易对的数据
ohlcv_cache = TTLCache(maxsize=100, ttl=60.0)      # K线缓存 60秒
market_cache = TTLCache(maxsize=100, ttl=2.0)      # 盘口/成交/Ticker缓存 2秒
balance_cache = TTLCache(maxsize=1, ttl=10.0)      # 账户余额缓存 10秒
ohlcv_request_locks: dict[str, asyncio.Lock] = {}

def cache_key_builder(func, *args, **kwargs):
    """
    自定义缓存键生成器：忽略 exchange 实例，只取 symbol 和 timeframe 作为 Key
    防止因 CCXT 实例内存地址不同导致缓存失效
    """
    # asyncache 的 @cached 会默认将 func 放在 kwargs 的第一个位置或 kwargs 外部。
    # 为了防止无参函数如 get_account_balance 抛出 IndexError，进行安全提取
    if not args and not kwargs:
        return getattr(func, '__name__', 'unknown_func')

    # 安全提取 args 中的第二个参数 (即 symbol)，前提是 args 长度大于1
    symbol = kwargs.get('symbol', None)
    if not symbol and len(args) > 1:
        symbol = args[1]
    elif not symbol:
        symbol = 'global'
        
    timeframe = kwargs.get('timeframe', '')
    limit = kwargs.get('limit', '')
    
    key = f"{getattr(func, '__name__', 'func')}_{symbol}_{timeframe}_{limit}"
    return key


async def get_or_set_ttl_cache(
    *,
    cache: TTLCache,
    key: str,
    loader,
    request_locks: dict[str, asyncio.Lock] | None = None,
    cache_name: str = "ttl_cache",
):
    """
    在 TTLCache 外层补一层按 key 的异步锁，避免并发 miss 时重复打交易所。

    设计原因：
    - Indicator-MCP 的 OHLCV 拉取既昂贵又容易触发交易所限频；
    - 同一 symbol/timeframe 在缓存失效瞬间只允许一个协程回源，其他协程复用结果。
    """
    if key in cache:
        return cache[key]

    lock_pool = request_locks if request_locks is not None else {}
    lock = lock_pool.get(key)
    if lock is None:
        lock = asyncio.Lock()
        lock_pool[key] = lock

    async with lock:
        if key in cache:
            return cache[key]

        logger.info(f"♻️ {cache_name} miss，开始回源加载: {key}")
        value = await loader()
        cache[key] = value
        return value
