from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1] / "servers" / "execution_mcp"
sys.path.insert(0, str(SERVER_DIR))

import server as execution_server  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("regular", "algo", "expected_count"),
    [([], [], 0), ([{"symbol": "ETH/USDT:USDT", "id": "regular-1"}], [], 1),
     ([], [{"symbol": "ETH/USDT:USDT", "id": "algo-1"}], 1)],
)
async def test_recovery_open_order_snapshot_checks_both_endpoints(
    monkeypatch: pytest.MonkeyPatch, regular: list[dict], algo: list[dict], expected_count: int,
) -> None:
    recovery_engine = execution_server.ExecutionEngine()

    class FakeExchange:
        def market(self, symbol: str) -> dict:
            assert symbol == "ETH/USDT"
            return {"symbol": "ETH/USDT:USDT"}

        async def fetch_open_orders(self, symbol: str) -> list[dict]:
            assert symbol == "ETH/USDT:USDT"
            return regular

    async def fake_init() -> None:
        recovery_engine.ex = FakeExchange()

    async def fake_algo(symbol: str) -> list[dict]:
        assert symbol == "ETH/USDT:USDT"
        return algo

    monkeypatch.setattr(recovery_engine, "init_exchange", fake_init)
    monkeypatch.setattr(recovery_engine, "_fetch_open_algo_orders", fake_algo)
    monkeypatch.setattr(execution_server, "engine", recovery_engine)
    result = json.loads(await execution_server.get_open_orders_snapshot("ETH/USDT"))

    assert result["status"] == "success"
    assert result["data"]["open_order_count"] == expected_count


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_endpoint", ["regular", "algo"])
async def test_recovery_open_order_snapshot_fails_closed_on_query_error(
    monkeypatch: pytest.MonkeyPatch, failed_endpoint: str,
) -> None:
    recovery_engine = execution_server.ExecutionEngine()

    class FakeExchange:
        def market(self, symbol: str) -> dict:
            return {"symbol": "ETH/USDT:USDT"}

        async def fetch_open_orders(self, symbol: str) -> list[dict]:
            if failed_endpoint == "regular":
                raise TimeoutError("regular orders unavailable")
            return []

    async def fake_init() -> None:
        recovery_engine.ex = FakeExchange()

    async def fake_algo(symbol: str) -> list[dict]:
        if failed_endpoint == "algo":
            raise TimeoutError("algo orders unavailable")
        return []

    monkeypatch.setattr(recovery_engine, "init_exchange", fake_init)
    monkeypatch.setattr(recovery_engine, "_fetch_open_algo_orders", fake_algo)
    monkeypatch.setattr(execution_server, "engine", recovery_engine)
    result = json.loads(await execution_server.get_open_orders_snapshot("ETH/USDT"))

    assert result["status"] == "error"
    assert result["error_code"] == "OPEN_ORDERS_QUERY_FAILED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("positions", "expected_status", "expected_count"),
    [([], "success", 0), ([{"symbol": "ETH/USDT:USDT", "contracts": "-0.2"}], "success", 1),
     ([{"symbol": "ETH/USDT:USDT", "contracts": None}], "error", None),
     (None, "error", None)],
)
async def test_recovery_position_snapshot_never_hides_invalid_or_signed_positions(
    monkeypatch: pytest.MonkeyPatch, positions: list[dict] | None,
    expected_status: str, expected_count: int | None,
) -> None:
    recovery_engine = execution_server.ExecutionEngine()

    class FakeExchange:
        async def fetch_positions(self, symbols):
            assert symbols is None
            return positions

    async def fake_init() -> None:
        recovery_engine.ex = FakeExchange()

    monkeypatch.setattr(recovery_engine, "init_exchange", fake_init)
    monkeypatch.setattr(execution_server, "engine", recovery_engine)
    result = json.loads(await execution_server.get_all_position_snapshots())

    assert result["status"] == expected_status
    if expected_count is not None:
        assert len(result["data"]) == expected_count


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {}, {"orders": None}, {"orders": [None]}])
async def test_algo_order_query_rejects_incomplete_responses(response: object) -> None:
    recovery_engine = execution_server.ExecutionEngine()

    class FakeExchange:
        @staticmethod
        def market_id(symbol: str) -> str:
            return "ETHUSDT"

        async def fapiPrivateGetOpenAlgoOrders(self, params: dict[str, str]) -> object:
            return response

    recovery_engine.ex = FakeExchange()
    with pytest.raises(RuntimeError):
        await recovery_engine._fetch_open_algo_orders("ETH/USDT")


def test_nullable_leverage_uses_configured_safety_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATEGY_MAX_LEVERAGE", "3")

    assert execution_server._position_leverage({"leverage": None}) == (3.0, "configured_fallback")
    assert execution_server._position_leverage({"leverage": ""}) == (3.0, "configured_fallback")
    assert execution_server._position_leverage({"leverage": 0}) == (3.0, "configured_fallback")
    assert execution_server._position_leverage({"leverage": float("nan")}) == (3.0, "configured_fallback")
    assert execution_server._position_leverage({"leverage": "8"}) == (8.0, "exchange")


@pytest.mark.asyncio
async def test_authoritative_fill_prefers_user_trades() -> None:
    engine = execution_server.ExecutionEngine()

    class FakeExchange:
        async def fetch_order(self, order_id: str, symbol: str):
            return {"id": order_id, "status": "closed", "filled": 0.5, "average": 100.0}

        async def fetch_my_trades(self, symbol: str, since, limit: int, params: dict[str, object]):
            assert params == {"orderId": "entry-1"}
            return [
                {"order": "entry-1", "amount": 0.2, "cost": 20.2, "fee": {"cost": 0.01}},
                {"order": "entry-1", "amount": 0.3, "cost": 30.6, "fee": {"cost": 0.02}},
            ]

    engine.ex = FakeExchange()
    fill = await engine._resolve_authoritative_fill(
        "ETH/USDT",
        {"id": "entry-1", "status": "closed", "filled": 0.5, "average": 99.0},
        "0.5",
    )

    assert fill["source"] == "user_trades"
    assert fill["filled_qty"] == execution_server.Decimal("0.5")
    assert fill["average_price"] == execution_server.Decimal("101.6")
    assert fill["fee"] == execution_server.Decimal("0.03")


@pytest.mark.asyncio
async def test_cancel_order_by_kind_uses_algo_endpoint() -> None:
    engine = execution_server.ExecutionEngine()
    calls: list[dict[str, object]] = []

    class FakeExchange:
        @staticmethod
        def market_id(symbol: str) -> str:
            return "ETHUSDT"

        async def fapiPrivateDeleteAlgoOrder(self, params: dict[str, object]):
            calls.append(params)

        async def cancel_order(self, order_id: str, symbol: str):
            raise AssertionError("Algo orders must not use the normal cancel endpoint")

    engine.ex = FakeExchange()
    await engine._cancel_order_by_kind(
        "ETH/USDT",
        {"id": "123", "info": {"_trendmaster_algo_order": True, "algoId": "123"}},
    )

    assert calls == [{"symbol": "ETHUSDT", "algoId": "123"}]


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
async def test_smart_order_rejects_notional_that_falls_below_minimum_after_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExchange:
        async def fetch_ticker(self, symbol: str):
            return {"last": 80469.3}

        @staticmethod
        def amount_to_precision(symbol: str, amount: float) -> str:
            return "0.0006"

        @staticmethod
        def price_to_precision(symbol: str, price: float) -> str:
            return str(price)

    async def noop_init() -> None:
        return None

    async def allow_risk(*args, **kwargs):
        return True, "Success"

    monkeypatch.setattr(execution_server.engine, "ex", FakeExchange())
    monkeypatch.setattr(execution_server.engine, "init_exchange", noop_init)
    monkeypatch.setattr(execution_server.engine, "validate_risk", allow_risk)
    monkeypatch.setitem(execution_server.RISK_CONFIG, "DRY_RUN_MODE", True)
    monkeypatch.setitem(execution_server.RISK_CONFIG, "MIN_ORDER_NOTIONAL_USD", execution_server.Decimal("50"))

    raw = await execution_server.execute_smart_order("BTC/USDT", "buy", 55.0)
    result = execution_server.json.loads(raw)

    assert result["status"] == "rejected"
    assert result["error_code"] == "MIN_NOTIONAL_AFTER_PRECISION"
    assert "48.2816" in result["message"]


@pytest.mark.asyncio
async def test_physical_entry_configures_exchange_leverage_before_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str]] = []

    class FakeExchange:
        async def set_leverage(self, leverage: int, symbol: str):
            calls.append((leverage, symbol))

    monkeypatch.setattr(execution_server.engine, "ex", FakeExchange())
    monkeypatch.setitem(execution_server.RISK_CONFIG, "DRY_RUN_MODE", False)
    monkeypatch.setenv("STRATEGY_MAX_LEVERAGE", "3")

    ok, reason = await execution_server._configure_entry_leverage("ETH/USDT", 3)

    assert ok is True
    assert reason == "configured"
    assert calls == [(3, "ETH/USDT")]


@pytest.mark.asyncio
async def test_physical_entry_rejects_leverage_above_safety_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(execution_server.RISK_CONFIG, "DRY_RUN_MODE", False)
    monkeypatch.setenv("STRATEGY_MAX_LEVERAGE", "3")

    ok, reason = await execution_server._configure_entry_leverage("ETH/USDT", 20)

    assert ok is False
    assert "exceeds safety cap 3x" in reason


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

    async def zero_position(symbol: str) -> list[dict]:
        assert symbol == "ETH/USDT"
        return [{"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt": "0"}]

    monkeypatch.setattr(engine, "_fetch_binance_usdm_positions_direct", zero_position)

    result = await engine._rollback_unprotected_entry(
        symbol="ETH/USDT",
        entry_side="buy",
        filled_amount="0.5",
        created_protection_orders=[{"id": "1000000210364943", "info": {"algoId": 1000000210364943}}],
    )

    assert result["status"] == "success"
    assert result["position_verified_zero"] is True
    assert cancelled == [("1000000210364943", "ETHUSDT")]
    assert close_calls == [
        {
            "symbol": "ETH/USDT",
            "close_side": "sell",
            "amount_str": "0.5",
            "position_side": None,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("positions", "expected_status"),
    [
        ([{"symbol": "ETHUSDT", "positionAmt": "0.2"}], "position_remaining"),
        ([], "position_unknown"),
        ([{"symbol": "ETHUSDT", "positionAmt": None}], "position_unknown"),
    ],
)
async def test_atomic_rollback_never_reports_success_without_zero_position(
    monkeypatch: pytest.MonkeyPatch, positions: list[dict], expected_status: str,
) -> None:
    engine = execution_server.ExecutionEngine()
    close_calls = 0

    async def fake_close(**kwargs):
        nonlocal close_calls
        close_calls += 1
        return {"id": "close-1"}, "reduce_only"

    async def fake_positions(symbol: str) -> list[dict]:
        return positions

    monkeypatch.setattr(engine, "_submit_market_close_order", fake_close)
    monkeypatch.setattr(engine, "_fetch_binance_usdm_positions_direct", fake_positions)
    result = await engine._rollback_unprotected_entry("ETH/USDT", "buy", "0.5", [])

    assert result["status"] == expected_status
    assert result["position_verified_zero"] is False
    assert close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("positions", "expected_status"),
    [
        ([], "position_unknown"),
        ([{"symbol": "ETH/USDT", "contracts": "0"}], "success"),
        ([{"symbol": "ETH/USDT", "contracts": "-0.2"}], "position_remaining"),
    ],
)
async def test_atomic_rollback_generic_exchange_requires_explicit_zero_position(
    monkeypatch: pytest.MonkeyPatch, positions: list[dict], expected_status: str,
) -> None:
    engine = execution_server.ExecutionEngine()
    engine.exchange_id = "other"

    class FakeExchange:
        async def fetch_positions(self, symbols):
            assert symbols is None
            return positions

    async def fake_init_exchange():
        engine.ex = FakeExchange()

    async def fake_close(**kwargs):
        return {"id": "close-1"}, "reduce_only"

    monkeypatch.setattr(engine, "init_exchange", fake_init_exchange)
    monkeypatch.setattr(engine, "_submit_market_close_order", fake_close)
    result = await engine._rollback_unprotected_entry("ETH/USDT", "buy", "0.5", [])

    assert result["status"] == expected_status
    assert result["position_verified_zero"] is (expected_status == "success")


@pytest.mark.asyncio
async def test_atomic_rollback_timeout_checks_position_without_second_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = execution_server.ExecutionEngine()
    close_calls = 0

    async def timed_out_close(**kwargs):
        nonlocal close_calls
        close_calls += 1
        raise TimeoutError("close response timed out")

    async def zero_position(symbol: str) -> list[dict]:
        return [{"symbol": "ETHUSDT", "positionAmt": "0"}]

    monkeypatch.setattr(engine, "_submit_market_close_order", timed_out_close)
    monkeypatch.setattr(engine, "_fetch_binance_usdm_positions_direct", zero_position)
    result = await engine._rollback_unprotected_entry("ETH/USDT", "buy", "0.5", [])

    assert result["status"] == "success"
    assert result["close_error"] == "close response timed out"
    assert result["position_verified_zero"] is True
    assert close_calls == 1


@pytest.mark.asyncio
async def test_atomic_rollback_position_query_error_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = execution_server.ExecutionEngine()

    async def fake_close(**kwargs):
        return {"id": "close-1"}, "reduce_only"

    async def failed_positions(symbol: str) -> list[dict]:
        raise TimeoutError("positionRisk unavailable")

    monkeypatch.setattr(engine, "_submit_market_close_order", fake_close)
    monkeypatch.setattr(engine, "_fetch_binance_usdm_positions_direct", failed_positions)
    result = await engine._rollback_unprotected_entry("ETH/USDT", "buy", "0.5", [])

    assert result["status"] == "position_unknown"
    assert result["verification_error"] == "positionRisk unavailable"
    assert result["position_verified_zero"] is False


@pytest.mark.asyncio
async def test_atomic_rollback_cancel_failure_keeps_cleanup_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = execution_server.ExecutionEngine()

    class FakeExchange:
        async def cancel_order(self, order_id: str, symbol: str) -> None:
            raise RuntimeError("normal cancel unavailable")

    async def failed_algo_cancel(symbol: str, order: dict) -> None:
        raise RuntimeError("algo cancel unavailable")

    async def fake_close(**kwargs):
        return {"id": "close-1"}, "reduce_only"

    async def zero_position(symbol: str) -> list[dict]:
        return [{"symbol": "ETHUSDT", "positionAmt": "0"}]

    engine.ex = FakeExchange()
    monkeypatch.setattr(engine, "_cancel_algo_order", failed_algo_cancel)
    monkeypatch.setattr(engine, "_submit_market_close_order", fake_close)
    monkeypatch.setattr(engine, "_fetch_binance_usdm_positions_direct", zero_position)
    result = await engine._rollback_unprotected_entry(
        "ETH/USDT", "buy", "0.5", [{"id": "stop-1"}],
    )

    assert result["status"] == "cleanup_incomplete"
    assert result["position_verified_zero"] is True
    assert len(result["cancel_errors"]) == 1


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
