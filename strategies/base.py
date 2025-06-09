from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from data.data_loader import DataLoader, DataProcessor

class Strategy(ABC):
    """策略基类"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.data_processor = DataProcessor()
        self.positions: Dict[str, float] = {}
        self.portfolio_value: float = 0.0
        self.current_data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def generate_signals(self) -> Dict[str, float]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def calculate_positions(self, signals: Dict[str, float]) -> Dict[str, float]:
        """计算目标仓位"""
        pass
    
    def update_portfolio(self, positions: Dict[str, float], returns: pd.Series):
        """更新组合价值"""
        portfolio_return = sum(positions[asset] * returns[asset] 
                             for asset in positions.keys() if asset in returns.index)
        self.portfolio_value *= (1 + portfolio_return)
        
    @abstractmethod
    def run(self, start_date: str, end_date: str) -> pd.Series:
        """运行策略"""
        pass
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """计算组合统计指标"""
        returns = self.get_portfolio_returns()
        stats = {
            'total_return': (self.portfolio_value - 1) * 100,
            'annual_return': returns.mean() * 252 * 100,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(returns) * 100
        }
        return stats
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return abs(drawdowns.min())
    
    def get_portfolio_returns(self) -> pd.Series:
        """获取组合收益率序列"""
        pass 