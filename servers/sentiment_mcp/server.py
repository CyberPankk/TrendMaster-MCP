import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import httpx
from datetime import datetime, timedelta
import random

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
load_dotenv(root_dir / ".env", override=True)

from mcp.server.fastmcp import FastMCP
from shared.logger import get_logger
from shared.cache_manager import cached, TTLCache, cache_key_builder

logger = get_logger("Sentiment-Server")
mcp = FastMCP("Sentiment-Server")

sentiment_cache = TTLCache(maxsize=50, ttl=300.0)


class SentimentEngine:
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=10.0)

    async def get_fear_and_greed(self) -> dict:
        try:
            response = await self.http_client.get("https://api.alternative.me/fng/")
            response.raise_for_status()
            data = response.json()
            latest = data["data"][0]

            return {
                "value": int(latest["value"]),
                "classification": latest["value_classification"],
                "insight": "数值越接近0越极度恐慌(底部特征)，越接近100越极度贪婪(顶部特征)。",
            }
        except Exception as e:
            logger.error(f"贪恐指数获取失败: {e}")
            return {"value": 50, "classification": "Neutral", "error": str(e)}

    async def get_social_sentiment(self, symbol: str) -> dict:
        coin = symbol.split("/")[0] if "/" in symbol else symbol
        mock_score = round(random.uniform(-0.5, 0.8), 2)
        classification = "BULLISH" if mock_score > 0.3 else ("BEARISH" if mock_score < -0.3 else "NEUTRAL")

        return {
            "asset": coin,
            "sentiment_score": mock_score,
            "classification": classification,
            "trending_keywords": [f"#{coin}", "Breakout", "ETF", "Whale"],
            "insight": f"社交媒体情绪得分 {mock_score}。>0.3 为看涨，<-0.3 为看跌。",
        }

    async def get_macro_calendar(self) -> list:
        today = datetime.now()
        mock_events = [
            {
                "event": "US CPI (YoY)",
                "impact": "HIGH",
                "time": (today + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "forecast": "3.1%",
                "previous": "3.2%",
            },
            {
                "event": "FOMC Press Conference",
                "impact": "CRITICAL",
                "time": (today + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
                "forecast": "N/A",
                "previous": "N/A",
            },
        ]
        return mock_events

    async def close(self):
        await self.http_client.aclose()


engine = SentimentEngine()


@mcp.tool()
@cached(cache=sentiment_cache, key=cache_key_builder)
async def get_comprehensive_sentiment(symbol: str, timeframe: str = "15m") -> str:
    logger.info(f"🌐 启动 Sentiment-MCP 聚合扫描: {symbol}...")

    try:
        fng_task = engine.get_fear_and_greed()
        social_task = engine.get_social_sentiment(symbol)
        macro_task = engine.get_macro_calendar()

        fng_data, social_data, macro_data = await asyncio.gather(fng_task, social_task, macro_task)

        sentiment_payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_fear_greed": fng_data,
            "social_sentiment": social_data,
            "upcoming_macro_events": macro_data,
            "system_instruction": "请结合 Indicator-MCP 的技术面数据。若技术面看多，但市场极度贪婪且社交情绪狂热，请警惕诱多；若面临 CRITICAL 级别宏观事件（如 FOMC），请强制输出 WAIT 观望。",
        }

        logger.info(f"✅ {symbol} 综合舆情脱水数据打包完成！")
        return json.dumps(sentiment_payload, ensure_ascii=False)

    except Exception as e:
        logger.error(f"❌ Sentiment-MCP 聚合失败: {e}")
        return json.dumps({
            "error": str(e),
            "status": "degraded_mode"
        }, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
