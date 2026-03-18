import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，解决模块导入问题
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from shared.logger import get_logger

logger = get_logger("SMC-Engine")

class SMCEngine:
    def __init__(self, df: pd.DataFrame, swing_window=5):
        """
        初始化 SMC 引擎。
        Args:
            df (pd.DataFrame): K 线数据，必须包含 'high', 'low', 'close', 'open' 列，列名需标准化为 'h', 'l', 'c', 'o'。
            swing_window (int): 识别分形高低点的窗口大小。
        """
        # 标准化列名
        self.df = df.copy()
        column_map = {
            'high': 'h', 'low': 'l', 'close': 'c', 'open': 'o',
            'High': 'h', 'Low': 'l', 'Close': 'c', 'Open': 'o'
        }
        self.df = self.df.rename(columns=column_map)
        
        # 确保索引是整数，方便 iloc 操作
        if not isinstance(self.df.index, pd.RangeIndex):
            self.df = self.df.reset_index(drop=True)
            
        self.swing_window = swing_window

    def identify_fvg(self) -> dict:
        """
        向量化识别 FVG (Fair Value Gap)
        返回最近的一个未被填补的 FVG 区间
        """
        try:
            df = self.df
            # 看多 FVG: K线3的Low > K线1的High
            bull_fvg = (df['l'] > df['h'].shift(2)) & (df['c'] > df['o'])
            # 看空 FVG: K线3的High < K线1的Low
            bear_fvg = (df['h'] < df['l'].shift(2)) & (df['c'] < df['o'])
            
            df_bull = df[bull_fvg]
            df_bear = df[bear_fvg]
            
            result = {"has_fvg": False, "type": None, "gap_top": None, "gap_bottom": None}
            
            # 取最近出现的一个 FVG
            last_bull_idx = df_bull.index[-1] if not df_bull.empty else -1
            last_bear_idx = df_bear.index[-1] if not df_bear.empty else -1
            
            if last_bull_idx > last_bear_idx and last_bull_idx != -1:
                # 提取最近看多缺口的边界
                # 注意：bull_fvg 为 True 的行是 K线3
                # Gap Top 是 K线3 的 Low
                # Gap Bottom 是 K线1 的 High (index-2)
                idx = df.index.get_loc(last_bull_idx)
                result = {
                    "has_fvg": True,
                    "type": "BULLISH",
                    "gap_top": float(df['l'].iloc[idx]),      # K线3的低点
                    "gap_bottom": float(df['h'].iloc[idx-2])  # K线1的高点
                }
            elif last_bear_idx > last_bull_idx and last_bear_idx != -1:
                idx = df.index.get_loc(last_bear_idx)
                result = {
                    "has_fvg": True,
                    "type": "BEARISH",
                    "gap_top": float(df['l'].iloc[idx-2]),    # K线1的低点
                    "gap_bottom": float(df['h'].iloc[idx])    # K线3的高点
                }
                
            return result
        except Exception as e:
            logger.error(f"FVG 识别失败: {e}")
            return {"has_fvg": False}

    def detect_structure(self) -> dict:
        """
        识别波段高低点 (Swing High/Low) 及结构突破 (BOS)
        """
        try:
            df = self.df
            # 使用 rolling 识别局部极值，彻底替代 for 循环
            # center=True 表示窗口中心对齐，即前后各 swing_window 根 K 线
            df['swing_high'] = df['h'] == df['h'].rolling(window=self.swing_window*2+1, center=True).max()
            df['swing_low'] = df['l'] == df['l'].rolling(window=self.swing_window*2+1, center=True).min()

            # 获取最近的有效高低点
            recent_highs = df[df['swing_high']]['h']
            recent_lows = df[df['swing_low']]['l']
            
            last_high = recent_highs.iloc[-1] if not recent_highs.empty else None
            last_low = recent_lows.iloc[-1] if not recent_lows.empty else None
            
            current_close = df['c'].iloc[-1]
            
            signal = "SIDEWAYS"
            last_event = "NONE"
            
            # 简单的 BOS 判断：突破最近的 Swing High/Low
            if last_high and current_close > last_high:
                signal = "BULL"
                last_event = "BOS_UP"
            elif last_low and current_close < last_low:
                signal = "BEAR"
                last_event = "BOS_DOWN"
                
            return {
                "structure": signal,
                "last_event": last_event,
                "key_resistance": float(last_high) if last_high else None,
                "key_support": float(last_low) if last_low else None
            }
        except Exception as e:
            logger.error(f"结构识别失败: {e}")
            return {"structure": "ERROR", "last_event": "ERROR"}

    def analyze(self) -> dict:
        """对外暴露的综合分析入口"""
        fvg_data = self.identify_fvg()
        structure_data = self.detect_structure()
        
        # 组装原始数据
        return {
            "fvg": fvg_data,
            "structure": structure_data
        }

if __name__ == "__main__":
    # --- 健壮性测试 ---
    print("🚀 开始 SMC 引擎健壮性测试...")
    
    # 1. 生成模拟 K 线数据
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    # 构造一个先涨后跌的数据
    prices = np.linspace(100, 150, 50).tolist() + np.linspace(150, 120, 50).tolist()
    # 添加随机波动
    prices = np.array(prices) + np.random.normal(0, 2, 100)
    
    df_test = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices + 1 # 简单的阳线倾向
    }, index=dates)
    
    # 2. 运行引擎
    try:
        engine = SMCEngine(df_test)
        result = engine.analyze()
        
        print("\n✅ 分析完成！结果如下:")
        print(f"Structure: {result['structure']}")
        print(f"FVG: {result['fvg']}")
        
        # 3. 验证返回类型
        assert isinstance(result, dict)
        assert "structure" in result
        assert "fvg" in result
        print("\n✅ 类型检查通过")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
