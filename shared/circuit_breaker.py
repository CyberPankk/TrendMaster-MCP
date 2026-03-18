import time
from shared.logger import get_logger

logger = get_logger("Circuit-Breaker")

class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self.failure_threshold = failure_threshold  # 容忍的连续失败次数
        self.recovery_timeout = recovery_timeout    # 熔断冷却时间 (秒)
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # 状态: CLOSED (通畅), OPEN (断开), HALF_OPEN (半开)

    def record_failure(self):
        """记录一次网络或API失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warn(f"⚠️ 记录异常，当前连续失败次数: {self.failure_count}")
        
        if self.failure_count >= self.failure_threshold and self.state == "CLOSED":
            self.state = "OPEN"
            logger.error(f"🚨 熔断器触发 (OPEN)! 交易所网络异常，系统锁定 {self.recovery_timeout} 秒，禁止发单。")

    def record_success(self):
        """记录一次成功，清零错误计数"""
        if self.state != "CLOSED":
            logger.info("✅ 交易所网络通信恢复，熔断器重置为 CLOSED。")
        self.failure_count = 0
        self.state = "CLOSED"

    def is_allowed(self) -> tuple[bool, str]:
        """检查当前是否允许通过"""
        if self.state == "OPEN":
            # 检查是否度过了冷却期
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.warn("👀 冷却期结束，熔断器进入 HALF_OPEN 状态，尝试放行探路请求...")
                return True, "HALF_OPEN"
            
            remaining = int(self.recovery_timeout - (time.time() - self.last_failure_time))
            return False, f"系统处于熔断保护中，{remaining} 秒后重试。"
        
        return True, "OK"

# 实例化全局熔断器供 Execution-MCP 使用
execution_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)