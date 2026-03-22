import os
import sys
import importlib.util
import logging
from typing import Callable, Any

logger = logging.getLogger("Indicator-MCP-Loader")

def load_dynamic_alpha_tools(mcp_server: Any, target_dir: str):
    """
    动态扫描并加载指定目录下的所有 alpha 因子，将它们的 analyze 方法注册为 MCP 工具。
    必须保证严格沙盒隔离，任何导入错误不得使主进程崩溃。
    """
    if not os.path.exists(target_dir):
        logger.warning(f"动态工具目录 {target_dir} 不存在，跳过加载。")
        return

    logger.info(f"开始动态扫描目录: {target_dir}")
    
    # 确保 target_dir 在 sys.path 中，方便相对导入或依赖寻找
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    for filename in os.listdir(target_dir):
        if filename.startswith("alpha_") and filename.endswith(".py"):
            module_name = filename[:-3]
            file_path = os.path.join(target_dir, filename)
            
            logger.info(f"发现潜在因子文件: {filename}")
            
            try:
                # 动态加载模块
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 检查是否包含主入口函数 analyze
                    if hasattr(module, "analyze") and callable(module.analyze):
                        analyze_func = getattr(module, "analyze")
                        
                        # 尝试提取 docstring
                        docstring = analyze_func.__doc__ or f"动态装载的 CogAlpha 因子工具: {module_name}"
                        
                        # 将这个函数作为工具注册到 MCP Server 中
                        # 工具名就用文件名去掉 .py
                        tool_name = module_name
                        
                        # 在 FastMCP 中添加工具
                        mcp_server.add_tool(analyze_func, name=tool_name, description=docstring)
                        logger.info(f"✅ 成功注册动态因子工具: {tool_name}")
                    else:
                        logger.warning(f"⚠️ 文件 {filename} 中未找到名为 analyze 的入口函数，跳过。")
            except Exception as e:
                # 沙盒隔离机制：只输出错误，绝对不能崩溃
                logger.error(f"❌ 加载因子 {filename} 失败: {str(e)}", exc_info=True)

    logger.info("动态工具扫描完毕。")
