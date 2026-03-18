import logging
import sys
from pathlib import Path
from colorama import Fore, Style, init

# 初始化 colorama，适配 Windows/Unix
init(autoreset=True)

class QuantLogger:
    """
    高级量化日志模块，支持终端彩色输出和文件持久化。
    """
    def __init__(self, service_name: str):
        self.service_name = service_name.upper()
        self.logger = logging.getLogger(service_name)
        
        # 为了不干扰 MCP 的 JSONRPC stdio 通信，对于 MCP Server，日志只写文件，或者写到 stderr
        # 默认使用 stderr
        
        self.logger.setLevel(logging.INFO)
        
        # 确保日志目录存在
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 避免重复添加 Handler
        if not self.logger.handlers:
            # 1. 终端处理器 (Terminal Handler) - 带颜色，输出到 stderr 以免破坏 MCP stdio
            console_handler = logging.StreamHandler(sys.stderr)
            console_formatter = logging.Formatter(
                f"{Fore.CYAN}[%(asctime)s]{Style.RESET_ALL} "
                f"{Fore.YELLOW}[{self.service_name}]{Style.RESET_ALL} "
                f"%(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            
            # 2. 文件处理器 (File Handler) - 用于持久化记录
            file_handler = logging.FileHandler(log_dir / "system.log", encoding="utf-8")
            file_formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def info(self, msg: str):
        """记录 INFO 级别的日志"""
        self.logger.info(f"{Fore.GREEN}INFO{Style.RESET_ALL}: {msg}")

    def warn(self, msg: str):
        """记录 WARN 级别的日志"""
        self.logger.warning(f"{Fore.MAGENTA}WARN{Style.RESET_ALL}: {msg}")

    def error(self, msg: str):
        """记录 ERROR 级别的日志"""
        self.logger.error(f"{Fore.RED}ERROR{Style.RESET_ALL}: {msg}")

# 导出工具方法
def get_logger(service_name: str) -> QuantLogger:
    """
    获取量化服务专用的日志记录器
    
    Args:
        service_name (str): 服务名称，例如 "Market-Server"
        
    Returns:
        QuantLogger: 包装好的日志类实例
    """
    return QuantLogger(service_name)
