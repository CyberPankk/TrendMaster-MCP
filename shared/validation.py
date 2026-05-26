import json
from datetime import datetime, timezone
from typing import Any


def build_db_record(
    *,
    module: str,
    event: str,
    symbol: str | None = None,
    payload: dict[str, Any] | None = None,
    level: str = "INFO",
) -> str:
    """
    构造统一 DB_RECORD 日志字符串，供 MCP 子服务输出审计事件。

    设计原因：
    - Indicator-MCP 需要在链路切换时留下可解析的结构化日志；
    - 先统一日志骨架，后续无论写文件、落 SQLite 还是接监控都能复用同一口径。
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "level": level,
        "event_type": event,
        "symbol": symbol or "MARKET",
        "payload": payload or {},
    }
    return f"DB_RECORD {json.dumps(record, ensure_ascii=False, default=str)}"
