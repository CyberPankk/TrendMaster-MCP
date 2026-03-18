import asyncio
from servers.indicator_mcp.server import get_full_market_context

async def main():
    print("Testing get_full_market_context...")
    res = await get_full_market_context("BTC/USDT", "1h")
    print(res)
    
if __name__ == "__main__":
    asyncio.run(main())
