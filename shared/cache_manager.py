import time
from cachetools import TTLCache
from asyncache import cached
from shared.logger import get_logger

logger = get_logger("Cache-Manager")

# 定义各类数据的专用缓存池
# maxsize=100 表示最多缓存 100 个交易对的数据
ohlcv_cache = TTLCache(maxsize=100, ttl=60.0)      # K线缓存 60秒
market_cache = TTLCache(maxsize=100, ttl=2.0)      # 盘口/成交/Ticker缓存 2秒
balance_cache = TTLCache(maxsize=1, ttl=10.0)      # 账户余额缓存 10秒

def cache_key_builder(func, *args, **kwargs):
    """
    自定义缓存键生成器：忽略 exchange 实例，只取 symbol 和 timeframe 作为 Key
    防止因 CCXT 实例内存地址不同导致缓存失效
    """
    # 提取 symbol，如果没有则默认为 'global'
    symbol = kwargs.get('symbol') or (args[1] if len(args) > 1 else 'global')
    timeframe = kwargs.get('timeframe', '')
    limit = kwargs.get('limit', '')
    
    key = f"{func.__name__}_{symbol}_{timeframe}_{limit}"
    # logger.debug(f"Cache Key Evaluated: {key}") # 调试用
    return key