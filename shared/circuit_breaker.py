import time
from typing import Dict

from shared.logger import get_logger

logger = get_logger("Circuit-Breaker")


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        name: str = "default",
    ) -> None:
        self.failure_threshold = failure_threshold  # 容忍的连续失败次数
        self.recovery_timeout = recovery_timeout  # 熔断冷却时间 (秒)
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # 状态: CLOSED (通畅), OPEN (断开), HALF_OPEN (半开)
        self.name = name

    def record_failure(self, exc: Exception | None = None) -> None:
        """记录一次网络或 API 失败，并在阈值命中时打开熔断器。"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        suffix = f" | last_error={exc}" if exc is not None else ""
        logger.warn(
            f"⚠️ 熔断器[{self.name}] 记录异常，当前连续失败次数: {self.failure_count}{suffix}"
        )

        if self.failure_count >= self.failure_threshold and self.state != "OPEN":
            self.state = "OPEN"
            logger.error(
                f"🚨 熔断器[{self.name}] 触发 OPEN，系统锁定 {self.recovery_timeout} 秒后再尝试放行。"
            )

    def record_success(self) -> None:
        """记录一次成功，清零错误计数并关闭熔断器。"""
        if self.state != "CLOSED":
            logger.info(f"✅ 熔断器[{self.name}] 已恢复，重置为 CLOSED。")
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0

    def allow_request(self) -> tuple[bool, str]:
        """兼容 Indicator-MCP 的链路探测接口，返回是否允许放行与当前状态。"""
        if self.state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.warn(f"👀 熔断器[{self.name}] 冷却结束，进入 HALF_OPEN 探路状态。")
                return True, "HALF_OPEN"

            remaining = max(0, int(self.recovery_timeout - elapsed))
            return False, f"OPEN:{remaining}s"

        return True, self.state

    def is_allowed(self) -> tuple[bool, str]:
        """兼容 Execution-MCP 的旧接口。"""
        allowed, state = self.allow_request()
        if allowed:
            return True, "OK" if state == "CLOSED" else state
        if state.startswith("OPEN:"):
            remaining = state.split(":", 1)[1]
            return False, f"系统处于熔断保护中，{remaining} 后重试。"
        return False, state


_ROUTE_BREAKERS: Dict[str, CircuitBreaker] = {}


def get_route_circuit_breaker(
    route_name: str,
    *,
    failure_threshold: int = 2,
    recovery_timeout: int = 20,
) -> CircuitBreaker:
    """
    按链路名返回独立熔断器。

    设计原因：
    - Indicator-MCP 会在本地 HTTP 与官方 SDK 之间回退；
    - 每条链路单独熔断，避免某一路由抖动时拖垮全部行情抓取。
    """
    normalized_name = str(route_name or "default").strip() or "default"
    breaker = _ROUTE_BREAKERS.get(normalized_name)
    if breaker is None:
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=normalized_name,
        )
        _ROUTE_BREAKERS[normalized_name] = breaker
    return breaker


# 实例化全局熔断器供 Execution-MCP 使用
execution_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    name="execution_mcp",
)
