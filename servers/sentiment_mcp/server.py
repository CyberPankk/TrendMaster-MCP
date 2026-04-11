import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import httpx
from datetime import datetime, timedelta
import random
import json
from bs4 import BeautifulSoup
import feedparser

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

    async def get_telegram_alpha(self, channel_name="whale_alert_io") -> list:
        """
        [Phase 3] 零风险监听 Telegram 频道大额链上异动
        通过 BeautifulSoup 解析网页镜像版的 Telegram 消息，绝对避免 API 封号。
        提取最新包含 "transferred to" 或大额转账相关的文本。
        """
        try:
            url = f"https://t.me/s/{channel_name}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            # 发起网页请求
            response = await self.http_client.get(url, headers=headers, timeout=15.0)
            response.raise_for_status()
            
            # 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(response.content, "html.parser")
            
            # 定位所有的消息节点 tgme_widget_message_text
            message_nodes = soup.find_all("div", class_="tgme_widget_message_text")
            
            alerts = []
            # 目标关键词：只关心大额流入/流出交易所或未知钱包
            target_keywords = ["transferred to", "transferred from", "minted", "burned"]
            
            # 从最新(底部)开始遍历
            for node in reversed(message_nodes):
                text = node.get_text(separator=" ", strip=True)
                
                # 如果包含指定关键词
                if any(kw in text.lower() for kw in target_keywords):
                    alerts.append(text)
                    
                # 仅保留最新的 3 条异动信息
                if len(alerts) >= 3:
                    break
                    
            # 兜底：如果完全没有命中，提取最新的 2 条消息
            if not alerts and message_nodes:
                logger.info("未发现包含大额转账关键词的消息，返回最新频道内容作为兜底。")
                for node in reversed(message_nodes[-2:]):
                    alerts.append(node.get_text(separator=" ", strip=True))
                    
            return alerts
            
        except Exception as e:
            logger.error(f"获取 Telegram 链上异动失败: {e}")
            return ["Telegram Data API Unavailable"]

    async def get_twitter_alpha(self, author="elonmusk") -> list:
        """
        [Phase 3 预留] 监听 Twitter 核心人物动态
        通过 RSSHub 或第三方 API 接口抓取核心人物的推文。
        目前为代码骨架，返回 Mock 结构，待未来部署 RSSHub 后填充真实 URL。
        """
        try:
            # TODO: 未来替换为真实的 RSSHub URL，例如 "https://rsshub.app/twitter/user/{author}"
            # 或者通过 RapidAPI 等第三方推特抓取服务
            
            # url = f"https://api.example.com/twitter/{author}"
            # response = await self.http_client.get(url)
            # data = response.json()
            
            # 当前返回代码骨架/Mock 数据
            logger.info(f"预留 Twitter 监听骨架被调用: 目标人物 @{author}")
            return [f"Mock Data: @{author} is watching the crypto market closely."]
            
        except Exception as e:
            logger.error(f"获取 Twitter 动态失败: {e}")
            return ["Twitter API Unavailable"]

    async def get_macro_calendar(self) -> list:
        """
        获取真实的宏观日历 (ForexFactory)
        仅过滤出 country 为 USD 且 impact 为 High 的重大事件，并返回极度精简结构
        """
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
            response = await self.http_client.get(url, timeout=15.0)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "xml")
            events = soup.find_all("event")
            
            macro_events = []
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            
            for event in events:
                country = event.find("country")
                impact = event.find("impact")
                
                # 安全获取文本内容
                country_text = country.text.strip() if country else ""
                impact_text = impact.text.strip() if impact else ""
                
                if country_text == "USD" and impact_text == "High":
                    date_node = event.find("date")
                    time_node = event.find("time")
                    title_node = event.find("title")
                    
                    date_str = date_node.text.strip() if date_node else ""
                    time_str = time_node.text.strip() if time_node else ""
                    
                    try:
                        # ForexFactory XML 日期格式类似: 11-20-2024
                        event_date = datetime.strptime(date_str, "%m-%d-%Y").date()
                        
                        # 仅关注今天和明天的事件
                        if event_date == today or event_date == tomorrow:
                            macro_events.append(f"[{date_str} {time_str}] {title_node.text.strip() if title_node else 'Unknown'}")
                    except Exception as parse_e:
                        logger.warning(f"解析事件日期失败 ({date_str}): {parse_e}")
                        continue
                        
            return macro_events
            
        except Exception as e:
            logger.error(f"获取真实宏观日历失败: {e}")
            return ["Macro API Unavailable"]

    async def get_crypto_news(self) -> list:
        """
        [Phase 2] 获取 CryptoPanic 真实政策与新闻源
        通过 RSS 获取加密新闻，并过滤包含特定监管/政策关键词的标题
        """
        try:
            # 采用 CoinTelegraph 的 RSS 源，避免 CryptoPanic 常见的 403 拦截
            url = "https://cointelegraph.com/rss"
            
            # 添加常见 User-Agent 以绕过基础反爬机制
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            # 异步获取 RSS 内容以避免阻塞
            response = await self.http_client.get(url, headers=headers, timeout=15.0)
            response.raise_for_status()
            
            # 使用 feedparser 解析
            feed = feedparser.parse(response.content)
            
            # 定义关注的政策/监管关键词 (不区分大小写)
            target_keywords = ["sec", "etf", "ban", "rate", "fed", "regulation", "lawsuit", "sues", "approve"]
            
            filtered_news = []
            
            for entry in feed.entries:
                title = entry.title
                
                # 检查标题是否包含任一关键词
                if any(kw.lower() in title.lower() for kw in target_keywords):
                    filtered_news.append(f"[{entry.published if hasattr(entry, 'published') else 'Unknown'}] {title}")
                    
                # 提取最新的 5 条
                if len(filtered_news) >= 5:
                    break
                    
            # 如果没有匹配到关键词新闻，提供一些最新的热门新闻作为兜底
            if not filtered_news and feed.entries:
                logger.info("未匹配到特定政策/监管关键词新闻，返回最新普通新闻作为兜底。")
                for entry in feed.entries[:3]:
                     filtered_news.append(f"[{entry.published if hasattr(entry, 'published') else 'Unknown'}] {entry.title}")
                    
            return filtered_news
            
        except Exception as e:
            logger.error(f"获取加密新闻失败: {e}")
            return ["Crypto News API Unavailable"]

    async def close(self):
        await self.http_client.aclose()

engine = SentimentEngine()


@mcp.tool()
@cached(cache=sentiment_cache, key=cache_key_builder)
async def get_comprehensive_sentiment(symbol: str, timeframe: str = "15m") -> str:
    logger.info(f"🌐 启动 Sentiment-MCP 聚合扫描: {symbol}...")

    try:
        fng_task = engine.get_fear_and_greed()
        crypto_news_task = engine.get_crypto_news()
        telegram_task = engine.get_telegram_alpha()
        twitter_task = engine.get_twitter_alpha()
        macro_task = engine.get_macro_calendar()

        fng_data, news_data, tg_data, tw_data, macro_data = await asyncio.gather(
            fng_task, crypto_news_task, telegram_task, twitter_task, macro_task
        )

        sentiment_payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_fear_greed": fng_data,
            "social_and_news": {
                "crypto_news": news_data,
                "telegram_whale_alerts": tg_data,
                "twitter_alpha": tw_data
            },
            "upcoming_macro_events": macro_data,
            "system_instruction": (
                "请结合 Indicator-MCP 的技术面数据进行终极裁决。\n"
                "1. 宏观风控: 若面临 CRITICAL 级别宏观事件（如 FOMC/CPI），除非技术面有绝对的流动性确认，否则请强制输出 WAIT 观望。\n"
                "2. 链上与消息面: 密切关注 Telegram 的巨鲸充提币动向(流入交易所往往砸盘，流出往往拉盘)以及 CryptoNews 的监管动态。\n"
                "3. 情绪逆向: 若技术面看多，但市场极度贪婪(FGI > 80)，请警惕诱多陷阱；若极度恐慌(FGI < 20)且出现巨鲸提币，可能是黄金坑。"
            ),
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
