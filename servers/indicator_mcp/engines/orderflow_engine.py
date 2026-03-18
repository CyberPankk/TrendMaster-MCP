import pandas as pd
from shared.logger import get_logger

logger = get_logger("Orderflow-Engine")

class OrderflowEngine:
    def __init__(self):
        self.prev_snapshot = None

    def calculate_ofi(self, symbol: str, current_book: dict) -> float:
        """计算静态挂单的订单流不平衡 (OFI)"""
        if self.prev_snapshot is None:
            self.prev_snapshot = current_book
            return 0.0

        try:
            bid_p_t, bid_v_t = current_book['bids'][0][0], current_book['bids'][0][1]
            ask_p_t, ask_v_t = current_book['asks'][0][0], current_book['asks'][0][1]
            
            bid_p_t_1, bid_v_t_1 = self.prev_snapshot['bids'][0][0], self.prev_snapshot['bids'][0][1]
            ask_p_t_1, ask_v_t_1 = self.prev_snapshot['asks'][0][0], self.prev_snapshot['asks'][0][1]

            delta_bid = bid_v_t if bid_p_t > bid_p_t_1 else (-bid_v_t if bid_p_t < bid_p_t_1 else bid_v_t - bid_v_t_1)
            delta_ask = ask_v_t if ask_p_t < ask_p_t_1 else (-ask_v_t if ask_p_t > ask_p_t_1 else ask_v_t - ask_v_t_1)
            
            self.prev_snapshot = current_book
            
            # 归一化处理，防止数值过大
            total_delta = abs(delta_bid) + abs(delta_ask)
            if total_delta == 0:
                return 0.0
                
            ofi = (delta_bid - delta_ask) / total_delta
            return round(float(ofi), 4)
        except Exception as e:
            logger.error(f"OFI 计算异常: {e}")
            return 0.0

    def analyze_aggressor_trades(self, trades: list) -> dict:
        """
        [核心升级] 分析真实成交的主动性 (Aggressor Analysis)
        计算 Taker 买单与 Taker 卖单的真实火拼比例
        """
        if not trades: 
            return {"buy_ratio": 0.5, "dominance": "NEUTRAL", "net_volume": 0}
        
        try:
            # 统计主动买入(吃掉卖单)和主动卖出(砸向买单)的真实数量
            buy_vol = sum(t['amount'] for t in trades if t['side'] == 'buy')
            sell_vol = sum(t['amount'] for t in trades if t['side'] == 'sell')
            total_vol = buy_vol + sell_vol
            
            buy_ratio = buy_vol / total_vol if total_vol > 0 else 0.5
            
            # 定义统治力阈值
            dominance = "NEUTRAL"
            if buy_ratio > 0.65:
                dominance = "STRONG_BUYER"
            elif buy_ratio < 0.35:
                dominance = "STRONG_SELLER"
                
            return {
                "buy_ratio": round(buy_ratio, 2),
                "net_volume": round(buy_vol - sell_vol, 4),
                "dominance": dominance,
                "insight": f"近期200笔成交中，主动买盘占比 {buy_ratio*100:.1f}%。状态: {dominance}"
            }
        except Exception as e:
            logger.error(f"Aggressor 分析异常: {e}")
            return {"buy_ratio": 0.5, "dominance": "ERROR", "net_volume": 0}
