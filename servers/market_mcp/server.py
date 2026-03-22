import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# 将项目根目录加入到 sys.path 中，以便能正确引用 shared 模块
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# 加载环境变量
load_dotenv(root_dir / ".env", override=True)

import ccxt.async_support as ccxt

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


from shared.logger import get_logger
from shared.models import TickerData, MCPErrorResponse, OrderbookSnapshot
import time
from datetime import datetime

# 1. 初始化日志工具
logger = get_logger("Market-Server")

# 2. 初始化 FastMCP 服务器
mcp = FastMCP("Market-Server")

logger.info("Market-MCP Server initializing...")

def get_exchange_config():
    """动态获取带有代理和 Testnet 配置的 CCXT 配置"""
    config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    }
    
    # 动态加载代理
    proxy_url = os.getenv("CCXT_PROXY")
    if proxy_url:
        config['proxies'] = {
            'http': proxy_url,
            'https': proxy_url,
        }
        
    # 检测是否应该使用 Testnet (根据 API Key 特征或新增的环境变量开关)
    # 既然我们用的是 Testnet API Key，应该开启 sandbox mode
    # 这里我们添加一个简单的判断逻辑，或者直接通过环境变量控制
    # 建议在 .env 中增加一个 USE_TESTNET=True
    if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
        logger.info("Using Binance Testnet...")
        # ccxt.binanceusdm 开启 testnet 需要设置 sandbox mode
        # 也可以在 config 中直接设置 'urls'，但 set_sandbox_mode 是标准做法
        # 不过在构造函数中无法直接 set_sandbox_mode，只能通过 options 里的 defaultType='future' 配合 testnet=True?
        # CCXT 的标准做法是在实例化后调用 set_sandbox_mode(True)
        # 或者在 config 字典中不直接支持 testnet 参数? 
        # 查阅 CCXT 文档: 'options': {'defaultType': 'swap', 'sandboxMode': True} (部分 exchange 支持)
        pass 
        
    return config

@mcp.tool()
async def get_ticker(symbol: str) -> str:
    """
    [Agent 工具] 获取指定交易对（如 BTC/USDT）的最新市场行情 (Ticker)。
    """
    start_time = time.time()
    logger.info(f"Received request for get_ticker: {symbol}")
    
    # 初始化 ccxt binance 异步客户端
    config = get_exchange_config()
    exchange = ccxt.binanceusdm(config)
    
    # 如果开启了 Testnet，必须显式设置
    if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
        exchange.set_sandbox_mode(True)
    
    try:
        # 异步请求行情数据
        ticker = await exchange.fetch_ticker(symbol)
        
        # 将 ccxt 返回的数据映射并封装到 Pydantic 定义的 TickerData 契约中
        ticker_data = TickerData(
            symbol=symbol,
            last_price=float(ticker.get('last', 0.0) or 0.0),
            change_24h_pct=float(ticker.get('percentage', 0.0) or 0.0),
            high_24h=float(ticker.get('high', 0.0) or 0.0),
            low_24h=float(ticker.get('low', 0.0) or 0.0),
            volume_24h=float(ticker.get('quoteVolume', 0.0) or 0.0),
            timestamp=int(ticker.get('timestamp', 0) or 0)
        )
        
        logger.info(f"Successfully fetched ticker for {symbol}")
        # 输出标准的 JSON 格式供 Agent 解析
        return ticker_data.model_dump_json()

    except ccxt.NetworkError as e:
        error_msg = f"Network error when fetching {symbol}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "status": "error",
            "error_code": "NETWORK_ERROR",
            "message": error_msg
        })
        
    except ccxt.ExchangeError as e:
        error_msg = f"Exchange error for {symbol}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "status": "error",
            "error_code": "EXCHANGE_ERROR",
            "message": error_msg
        })
        
    except Exception as e:
        error_msg = f"Unexpected error for {symbol}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "message": error_msg
        })
        
    finally:
        # 确保释放和关闭异步连接，防止资源泄露
        await exchange.close()

# @mcp.tool()
async def get_orderbook_ofi(symbol: str) -> str:
    """
    获取指定交易对的盘口数据，计算订单流不平衡指标 (OFI) 并生成自然语言描述。
    """
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{scan_time}] Starting orderbook scan and OFI calculation for {symbol}")
    
    config = get_exchange_config()
    exchange = ccxt.binanceusdm(config)
    
    if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
        exchange.set_sandbox_mode(True)
    
    try:
        # 获取深度为20的盘口数据
        orderbook = await exchange.fetch_order_book(symbol, limit=20)
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            raise ValueError(f"Empty orderbook received for {symbol}")
            
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        bid_ask_spread = best_ask - best_bid
        
        # 计算买卖盘的总挂单量 (Size)
        sum_bid_size = sum(bid[1] for bid in bids)
        sum_ask_size = sum(ask[1] for ask in asks)
        
        # 计算 OFI (Order Flow Imbalance)
        total_size = sum_bid_size + sum_ask_size
        if total_size == 0:
            ofi = 0.0
        else:
            ofi = (sum_bid_size - sum_ask_size) / total_size
            
        # 生成自然语言的深度描述 (depth_summary)
        if ofi > 0.3:
            depth_summary = "买盘明显占优，下档支撑强劲，具备向上动能。"
        elif ofi > 0.05:
            depth_summary = "买盘略微占优，短期偏向多头。"
        elif ofi < -0.3:
            depth_summary = "卖盘严重压制，上档抛压沉重，建议观望或防范回落风险。"
        elif ofi < -0.05:
            depth_summary = "卖盘略微占优，短期偏向空头。"
        else:
            depth_summary = "买卖力量相对均衡，多空博弈胶着，盘面震荡。"
            
        # 封装并返回 OrderbookSnapshot 契约模型
        snapshot = OrderbookSnapshot(
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            ofi=ofi,
            bid_ask_spread=bid_ask_spread,
            depth_summary=depth_summary
        )
        
        logger.info(f"Successfully generated OrderbookSnapshot for {symbol}. OFI: {ofi:.4f}")
        return snapshot.model_dump_json()

    except ccxt.NetworkError as e:
        error_msg = f"Network error when fetching orderbook for {symbol}: {str(e)}"
        logger.error(error_msg)
        return MCPErrorResponse(
            status="error",
            error_code="NETWORK_ERROR",
            message=error_msg
        ).model_dump_json()
        
    except ccxt.ExchangeError as e:
        error_msg = f"Exchange error for {symbol}: {str(e)}"
        logger.error(error_msg)
        return MCPErrorResponse(
            status="error",
            error_code="EXCHANGE_ERROR",
            message=error_msg
        ).model_dump_json()
        
    except Exception as e:
        error_msg = f"Unexpected error calculating OFI for {symbol}: {str(e)}"
        logger.error(error_msg)
        return MCPErrorResponse(
            status="error",
            error_code="INTERNAL_ERROR",
            message=error_msg
        ).model_dump_json()
        
    finally:
        await exchange.close()

if __name__ == "__main__":
    logger.info("Market-MCP Server starting...")
    # 启动 MCP server 实例，以 standard I/O (stdio) 方式运行
    mcp.run()
