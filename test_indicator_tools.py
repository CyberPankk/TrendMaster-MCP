import asyncio
from servers.indicator_mcp.server import analyze_smc, get_market_regime_hmm, analyze_orderflow

async def main():
    print("Testing analyze_smc...")
    smc_res = await analyze_smc("BTC/USDT", "1h")
    print(smc_res)
    
    print("\nTesting get_market_regime_hmm...")
    hmm_res = await get_market_regime_hmm("BTC/USDT", "1h")
    print(hmm_res)
    
    print("\nTesting analyze_orderflow...")
    of_res = await analyze_orderflow("BTC/USDT")
    print(of_res)

if __name__ == "__main__":
    asyncio.run(main())
