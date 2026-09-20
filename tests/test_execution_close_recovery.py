from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1] / "servers" / "execution_mcp"
sys.path.insert(0, str(SERVER_DIR))

import server as execution_server  # noqa: E402


def test_nullable_leverage_uses_safe_default() -> None:
    assert execution_server._coerce_positive_float(None, 20.0) == 20.0
    assert execution_server._coerce_positive_float("", 20.0) == 20.0
    assert execution_server._coerce_positive_float(0, 20.0) == 20.0
    assert execution_server._coerce_positive_float(float("nan"), 20.0) == 20.0
    assert execution_server._coerce_positive_float(float("inf"), 20.0) == 20.0
    assert execution_server._coerce_positive_float("8", 20.0) == 8.0


def test_client_tag_symbol_uses_same_identity_for_ccxt_future_suffix() -> None:
    assert execution_server._normalize_client_tag_symbol("ETH/USDT") == "ETHUSDT"
    assert execution_server._normalize_client_tag_symbol("ETH/USDT:USDT") == "ETHUSDT"


@pytest.mark.asyncio
async def test_open_order_lookup_merges_binance_algo_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = execution_server.ExecutionEngine()

    class FakeExchange:
        async def fetch_open_orders(self, symbol: str):
            return []

        @staticmethod
        def market_id(symbol: str) -> str:
            return "ETHUSDT"

        async def fapiPrivateGetOpenAlgoOrders(self, params: dict[str, object]):
            assert params == {"symbol": "ETHUSDT"}
            return [
                {
                    "algoId": 1000000210364943,
                    "clientAlgoId": "TM_SL_ETHUSDT_1",
                    "orderType": "STOP_MARKET",
                    "algoStatus": "NEW",
                    "symbol": "ETHUSDT",
                    "side": "SELL",
                    "quantity": "0.5",
                    "triggerPrice": "2636.78",
                    "reduceOnly": True,
                }
            ]

    engine.ex = FakeExchange()

    async def noop_init() -> None:
        return None

    monkeypatch.setattr(engine, "init_exchange", noop_init)
    orders, resolved = await engine._fetch_open_orders_for_symbol("ETH/USDT")

    assert resolved == "ETH/USDT"
    assert len(orders) == 1
    assert orders[0]["id"] == "1000000210364943"
    assert orders[0]["clientOrderId"] == "TM_SL_ETHUSDT_1"
    assert orders[0]["stopPrice"] == "2636.78"
    assert orders[0]["info"]["_trendmaster_algo_order"] is True


@pytest.mark.asyncio
async def test_open_order_lookup_fails_closed_when_algo_query_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = execution_server.ExecutionEngine()

    class FakeExchange:
        async def fetch_open_orders(self, symbol: str):
            return []

        @staticmethod
        def market_id(symbol: str) -> str:
            return "ETHUSDT"

        async def request(self, *args, **kwargs):
            raise RuntimeError("algo endpoint unavailable")

    engine.ex = FakeExchange()

    async def noop_init() -> None:
        return None

    monkeypatch.setattr(engine, "init_exchange", noop_init)
    with pytest.raises(RuntimeError, match="拒绝把保护状态误报为空"):
        await engine._fetch_open_orders_for_symbol("ETH/USDT")


@pytest.mark.asyncio
async def test_global_algo_scan_only_returns_trendmaster_managed_symbols() -> None:
    engine = execution_server.ExecutionEngine()

    class FakeExchange:
        async def fapiPrivateGetOpenAlgoOrders(self, params: dict[str, object]):
            assert params == {}
            return [
                {"symbol": "ETHUSDT", "clientAlgoId": "TM_SL_ETHUSDT_1"},
                {"symbol": "SOLUSDT", "clientAlgoId": "TM_TP_SOLUSDT_1"},
                {"symbol": "BTCUSDT", "clientAlgoId": "manual-protection"},
            ]

        @staticmethod
        def safe_symbol(symbol: str, *args) -> str:
            return {"ETHUSDT": "ETH/USDT:USDT", "SOLUSDT": "SOL/USDT:USDT"}.get(symbol, symbol)

    engine.ex = FakeExchange()

    symbols = await engine._fetch_managed_algo_order_symbols()

    assert symbols == {"ETH/USDT:USDT", "SOL/USDT:USDT"}


@pytest.mark.asyncio
async def test_unprotected_entry_rollback_closes_only_the_new_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = execution_server.ExecutionEngine()
    cancelled: list[tuple[str, str]] = []
    close_calls: list[dict[str, object]] = []

    class FakeExchange:
        async def cancel_order(self, order_id: str, symbol: str):
            raise RuntimeError("normal order endpoint does not contain algo order")

        @staticmethod
        def market_id(symbol: str) -> str:
            return "ETHUSDT"

        async def fapiPrivateDeleteAlgoOrder(self, params: dict[str, object]):
            cancelled.append((str(params["algoId"]), str(params["symbol"])))

    async def fake_submit(**kwargs):
        close_calls.append(kwargs)
        return {"id": "rollback-close-1"}, "reduce_only"

    engine.ex = FakeExchange()
    monkeypatch.setattr(engine, "_submit_market_close_order", fake_submit)

    result = await engine._rollback_unprotected_entry(
        symbol="ETH/USDT",
        entry_side="buy",
        filled_amount="0.5",
        created_protection_orders=[{"id": "1000000210364943", "info": {"algoId": 1000000210364943}}],
    )

    assert result["status"] == "success"
    assert cancelled == [("1000000210364943", "ETHUSDT")]
    assert close_calls == [
        {
            "symbol": "ETH/USDT",
            "close_side": "sell",
            "amount_str": "0.5",
            "position_side": None,
        }
    ]


def test_one_way_position_does_not_invent_hedge_side() -> None:
    engine = execution_server.ExecutionEngine()

    assert engine._extract_position_side_param(
        {"side": "long", "info": {"positionSide": "BOTH"}}
    ) is None
    assert engine._extract_position_side_param(
        {"side": "short", "info": {"positionSide": "SHORT"}}
    ) == "SHORT"
    assert engine._extract_position_side_param({"side": "long"}) == "LONG"


def test_reduce_only_rejection_refreshes_then_uses_minimal_one_way_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = execution_server.ExecutionEngine()

    class FakeExchange:
        @staticmethod
        def amount_to_precision(symbol: str, amount: float) -> str:
            assert symbol == "BTC/USDT:USDT"
            return f"{amount:.4f}"

    engine.ex = FakeExchange()
    submit_calls: list[dict[str, object]] = []

    async def fake_submit(**kwargs):
        submit_calls.append(kwargs)
        if len(submit_calls) == 1:
            raise RuntimeError("binanceusdm -2022 ReduceOnly Order is rejected")
        return {"id": "close-order-1"}, "minimal_one_way_close"

    async def fake_cancel_all_orders(symbol: str) -> None:
        assert symbol == "BTC/USDT:USDT"

    async def fake_cancel_protection_orders(symbol: str) -> int:
        assert symbol == "BTC/USDT:USDT"
        return 0

    refreshed = {
        "symbol": "BTC/USDT:USDT",
        "contracts": 0.0123,
        "side": "long",
        "info": {"positionSide": "BOTH"},
    }

    async def fake_fetch_positions(symbol: str):
        assert symbol == "BTC/USDT:USDT"
        return [refreshed], "BTC/USDT:USDT"

    monkeypatch.setenv("KILL_SWITCH_CANCEL_SETTLE_SEC", "0")
    monkeypatch.setattr(engine, "_submit_market_close_order", fake_submit)
    monkeypatch.setattr(engine, "cancel_all_orders", fake_cancel_all_orders)
    monkeypatch.setattr(engine, "_cancel_protection_orders", fake_cancel_protection_orders)
    monkeypatch.setattr(engine, "_fetch_positions_for_symbol", fake_fetch_positions)

    result = asyncio.run(engine._close_position_with_fallback("BTC/USDT:USDT", refreshed))

    assert submit_calls[0]["position_side"] is None
    assert submit_calls[0].get("force_minimal_params", False) is False
    assert submit_calls[1]["position_side"] is None
    assert submit_calls[1]["force_minimal_params"] is True
    assert submit_calls[1]["amount_str"] == "0.0123"
    assert result == {
        "status": "success",
        "mode": "cancel_all_refresh_no_reduce_only_fallback",
        "order_id": "close-order-1",
        "amount": "0.0123",
        "close_side": "sell",
        "reduce_only_applied": False,
    }
