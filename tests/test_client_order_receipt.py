"""Receipt provenance is copied only from the exchange response."""

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1] / "servers" / "execution_mcp"
sys.path.insert(0, str(SERVER_DIR))
import server as execution_server  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_client_id", ["client-actual-1", None])
async def test_market_entry_passes_exchange_client_id_without_inventing_one(
    monkeypatch: pytest.MonkeyPatch, receipt_client_id: str | None,
) -> None:
    class FakeExchange:
        async def fetch_ticker(self, symbol):
            return {"last": 10}

        def amount_to_precision(self, symbol, amount):
            return str(amount)

        def price_to_precision(self, symbol, price):
            return str(price)

        async def create_market_order(self, symbol, side, amount):
            order = {"id": "exchange-1", "status": "closed", "filled": amount, "average": 10}
            if receipt_client_id:
                order["info"] = {"clientOrderId": receipt_client_id}
            return order

    class FakeEngine:
        ex = FakeExchange()
        twap_threshold = Decimal("10000")

        async def init_exchange(self):
            pass

        async def validate_risk(self, *args, **kwargs):
            return True, "ok"

        async def _resolve_authoritative_fill(self, symbol, order, amount):
            return {"filled_qty": amount, "average_price": "10", "source": "receipt", "fee": "0"}

    async def allow_leverage(*args):
        return True, "configured"

    monkeypatch.setattr(execution_server, "engine", FakeEngine())
    monkeypatch.setattr(execution_server.execution_breaker, "is_allowed", lambda: (True, "ok"))
    monkeypatch.setattr(execution_server, "_configure_entry_leverage", allow_leverage)
    monkeypatch.setitem(execution_server.RISK_CONFIG, "DRY_RUN_MODE", False)
    monkeypatch.setattr(execution_server, "USE_TESTNET_MODE", True)

    result = json.loads(await execution_server.execute_smart_order("BTC/USDT", "buy", 100))
    assert result["status"] == "success"
    assert result["order_id"] == "exchange-1"
    assert result["client_order_id"] == receipt_client_id


@pytest.mark.asyncio
async def test_twap_slice_records_actual_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = execution_server.ExecutionEngine()

    class FakeExchange:
        async def fetch_ticker(self, symbol):
            return {"last": 10}

        def amount_to_precision(self, symbol, amount):
            return str(amount)

        async def create_market_order(self, symbol, side, amount):
            return {"id": "slice-1", "status": "closed", "filled": amount,
                    "average": 10, "cost": 100, "clientOrderId": "client-slice-1"}

    async def noop_init():
        engine.ex = FakeExchange()

    monkeypatch.setattr(engine, "init_exchange", noop_init)
    monkeypatch.setitem(execution_server.TWAP_TASKS, "task-1", {"orders": [], "remaining_usd": "100"})
    record = await engine._execute_twap_slice("BTC/USDT", "sell", Decimal("100"), "task-1", 1)

    assert record["order_id"] == "slice-1"
    assert record["client_order_id"] == "client-slice-1"
    assert execution_server.TWAP_TASKS["task-1"]["orders"][0]["client_order_id"] == "client-slice-1"
