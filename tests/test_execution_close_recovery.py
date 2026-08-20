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
