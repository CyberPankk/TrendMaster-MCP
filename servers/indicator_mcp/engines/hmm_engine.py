import pandas as pd
import numpy as np
import warnings
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from shared.logger import get_logger

logger = get_logger("HMM-Engine")

class HMMEngine:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.scaler = StandardScaler()

    def _create_model(self):
        # 优化点 1: 大幅增加迭代次数，放宽收敛容差
        return GaussianHMM(
            n_components=self.n_components, 
            covariance_type="full", 
            n_iter=2000,     # 原来可能是默认的 10 或 100
            tol=1e-3,        # 容差阈值
            random_state=42
        )

    def prepare_features(self, df: pd.DataFrame):
        """提取特征并进行标准化"""
        df = df.copy()
        # 对数收益率
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        # K线振幅百分比
        df['range'] = (df['high'] - df['low']) / df['close']
        
        # 丢弃 NaN 行
        features_df = df[['log_return', 'range']].dropna()
        
        # 优化点 2: 强制数据标准化 (解决不收敛的终极核武)
        scaled_features = self.scaler.fit_transform(features_df)
        
        return scaled_features

    def train_and_predict(self, df: pd.DataFrame) -> str:
        """训练并返回具有明确物理含义的市场状态"""
        try:
            features = self.prepare_features(df)
            
            # 每次重新创建模型，避免 hmmlearn 在多次 fit 时产生属性被覆盖的 logging.warning 警告
            model = self._create_model()
            
            # 捕获并忽略底层的冗余警告，保持日志清爽
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features)
                
            if not model.monitor_.converged:
                logger.warn(f"HMM 在 {model.monitor_.iter} 次迭代后达到阈值，已取当前最优解。")
                
            states = model.predict(features)
            current_state_id = int(states[-1])
            
            # 优化点 3: 状态重排 (State Sorting by Volatility)
            # 获取每个状态的协方差矩阵中 'range' 特征（索引1）的方差
            variances = np.array([cov[1, 1] for cov in model.covars_])
            # 将方差从小到大排序的索引
            sorted_idx = np.argsort(variances)
            
            # 建立绝对映射关系
            state_mapping = {
                sorted_idx[0]: "QUIET_SIDEWAYS",  # 方差最小 = 安静震荡
                sorted_idx[1]: "TRENDING",        # 方差居中 = 稳步趋势
                sorted_idx[2]: "VOLATILE_TREND"   # 方差最大 = 高波趋势
            }
            
            final_regime = state_mapping.get(current_state_id, "UNKNOWN")
            logger.info(f"HMM 状态评估完毕，当前底层 ID: {current_state_id} -> 逻辑映射: {final_regime}")
            
            return final_regime
            
        except Exception as e:
            logger.error(f"HMM 训练失败: {str(e)}")
            return "UNKNOWN"
