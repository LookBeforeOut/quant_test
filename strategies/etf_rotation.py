from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from strategies.base import Strategy
from data.data_loader import ETFDataLoader, DataProcessor
import logging

logger = logging.getLogger(__name__)

@dataclass
class ETFRotationConfig:
    momentum_window: int = 21
    vol_window: int = 21
    rsi_window: int = 14
    rebalance_freq: str = 'W'  # 'M' for monthly, 'W' for weekly
    risk_free_rate: float = 0.02
    max_drawdown_threshold: float = 0.15
    stop_loss_threshold: float = 0.1
    position_size_limit: float = 0.5  # 提高最大仓位到50%
    transaction_cost: float = 0.003
    momentum_weight: float = 0.3
    rsi_weight: float = 0.2
    macd_weight: float = 0.2
    volume_std_weight: float = 0.15  # 成交金额标准差权重
    return_factor_weight: float = 0.15  # 涨跌幅因子权重
    max_positions: int = 3  # 最大持仓数量，默认为3个ETF
    volume_std_window: int = 10  # 成交金额标准差计算窗口
    return_factor_window: int = 10  # 涨跌幅因子计算窗口

class ETFRotationStrategy(Strategy):
    """ETF轮动策略"""
    
    def __init__(self, symbols: List[str], config: ETFRotationConfig):
        super().__init__(ETFDataLoader(symbols))
        self.config = config
        self.symbols = symbols
        self.returns_history: List[float] = []
        self.position_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.data_processor = DataProcessor()
        
    def generate_signals(self, current_date: pd.Timestamp) -> Dict[str, float]:
        """生成交易信号"""
        signals = {}
        
        try:
            # 确保有足够的数据
            min_required_length = max(
                self.config.momentum_window,
                self.config.vol_window,
                self.config.rsi_window,
                self.config.volume_std_window,
                self.config.return_factor_window
            ) + 10

            data_before_current_date = self.current_data.loc[:current_date]

            # 对所有信号进行标准化
            all_momentum_signals = []
            all_rsi_signals = []
            all_macd_signals = []
            all_volume_std_signals = []
            all_return_factor_signals = []
            
            # 首先计算所有ETF的原始信号
            for symbol in self.symbols:
                try:
                    # 获取价格数据
                    prices = data_before_current_date[f'{symbol}_close'].iloc[-min_required_length:]
                    volumes = data_before_current_date[f'{symbol}_volume'].iloc[-min_required_length:]
                    
                    if len(prices) < min_required_length:
                        logger.warning(f"Insufficient data for {symbol}, skipping...")
                        continue
                        
                    # 检查价格数据的有效性
                    if prices.isnull().any() or (prices == 0).any():
                        logger.warning(f"Invalid price data found for {symbol}, skipping...")
                        continue
                    
                    # 计算动量指标
                    momentum = self.data_processor.calculate_momentum(
                        prices, self.config.momentum_window)
                    
                    # 计算波动率
                    returns = self.data_processor.calculate_returns(prices)
                    volatility = self.data_processor.calculate_volatility(
                        returns, self.config.vol_window)
                    
                    # 计算RSI
                    rsi = self.data_processor.calculate_rsi(
                        prices, self.config.rsi_window)
                    
                    # 计算MACD
                    macd = self.data_processor.calculate_macd(prices)
                    
                    # 计算成交金额标准差
                    volume_std = self.calculate_volume_std(prices, volumes)
                    
                    # 计算涨跌幅因子
                    return_factor = self.calculate_return_factor(prices)
                    
                    # 获取最新值
                    latest_momentum = momentum.iloc[-1]
                    latest_volatility = volatility.iloc[-1]
                    latest_rsi = rsi.iloc[-1]
                    latest_macd = macd.iloc[-1]
                    latest_volume_std = volume_std.iloc[-1]
                    latest_return_factor = return_factor.iloc[-1]
                    
                    # 检查计算结果的有效性
                    if any(pd.isna(x) for x in [latest_momentum, latest_volatility, latest_rsi, 
                                              latest_macd, latest_volume_std, latest_return_factor]):
                        logger.warning(f"Invalid indicator values for {symbol}, skipping...")
                        continue
                    
                    # 计算动量信号（考虑波动率）
                    momentum_signal = latest_momentum / prices.iloc[-1]  # 使用相对动量
                    
                    # 计算RSI信号（超买超卖）
                    rsi_signal = (latest_rsi - 50) / 50  # 归一化到[-1, 1]范围
                    
                    # MACD信号已经在计算时归一化
                    macd_signal = latest_macd
                    
                    # 成交金额标准差信号（标准化）
                    volume_std_signal = latest_volume_std / prices.iloc[-1]  # 相对成交金额标准差
                    
                    # 涨跌幅因子信号（标准化）
                    return_factor_signal = latest_return_factor
                    
                    all_momentum_signals.append((symbol, momentum_signal))
                    all_rsi_signals.append((symbol, rsi_signal))
                    all_macd_signals.append((symbol, macd_signal))
                    all_volume_std_signals.append((symbol, volume_std_signal))
                    all_return_factor_signals.append((symbol, return_factor_signal))
                    
                except Exception as e:
                    logger.error(f"Error calculating signals for {symbol}: {e}")
                    continue
            
            if not all_momentum_signals:
                logger.error("No valid signals could be calculated")
                return {symbol: 0.0 for symbol in self.symbols}
            
            # 对信号进行标准化
            def normalize_signals(signal_list):
                values = np.array([x[1] for x in signal_list])
                mean = np.mean(values)
                std = np.std(values)
                if std == 0:
                    return {x[0]: 0.0 for x in signal_list}
                normalized = (values - mean) / (std + 1e-6)  # 避免除以零
                # 使用sigmoid函数将信号映射到[0, 1]区间
                normalized = 1 / (1 + np.exp(-normalized))
                return {x[0]: y for x, y in zip(signal_list, normalized)}
            
            norm_momentum = normalize_signals(all_momentum_signals)
            norm_rsi = normalize_signals(all_rsi_signals)
            norm_macd = normalize_signals(all_macd_signals)
            norm_volume_std = normalize_signals(all_volume_std_signals)
            norm_return_factor = normalize_signals(all_return_factor_signals)
            
            # 合成最终信号
            for symbol in self.symbols:
                if all(symbol in norm for norm in [norm_momentum, norm_rsi, norm_macd, 
                                                 norm_volume_std, norm_return_factor]):
                    signals[symbol] = (
                        self.config.momentum_weight * norm_momentum[symbol] +
                        self.config.rsi_weight * norm_rsi[symbol] +
                        self.config.macd_weight * norm_macd[symbol] +
                        self.config.volume_std_weight * norm_volume_std[symbol] +
                        self.config.return_factor_weight * norm_return_factor[symbol]
                    )
                else:
                    signals[symbol] = 0.0
                
                logger.debug(f"Signals for {symbol}: momentum={norm_momentum.get(symbol, 0):.4f}, "
                          f"rsi={norm_rsi.get(symbol, 0):.4f}, macd={norm_macd.get(symbol, 0):.4f}, "
                          f"volume_std={norm_volume_std.get(symbol, 0):.4f}, "
                          f"return_factor={norm_return_factor.get(symbol, 0):.4f}, "
                          f"final={signals[symbol]:.4f}")
                
        except Exception as e:
            logger.error(f"Error in generate_signals: {e}")
            return {symbol: 0.0 for symbol in self.symbols}
            
        return signals
    
    def calculate_volume_std(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """计算成交金额标准差"""
        try:
            # 计算成交金额
            turnover = prices * volumes
            
            # 计算成交金额标准差
            volume_std = turnover.rolling(
                window=self.config.volume_std_window,
                min_periods=self.config.volume_std_window // 2
            ).std()
            
            return volume_std
            
        except Exception as e:
            logger.error(f"Error calculating volume std: {e}")
            return pd.Series(0, index=prices.index)
    
    def calculate_return_factor(self, prices: pd.Series) -> pd.Series:
        """计算涨跌幅因子"""
        try:
            # 计算收益率
            returns = prices.pct_change()
            
            # 计算涨跌幅因子（考虑上涨和下跌的幅度）
            up_returns = returns.where(returns > 0, 0)
            down_returns = abs(returns.where(returns < 0, 0))
            
            # 计算上涨和下跌的累积效应
            up_factor = up_returns.rolling(
                window=self.config.return_factor_window,
                min_periods=self.config.return_factor_window // 2
            ).sum()
            
            down_factor = down_returns.rolling(
                window=self.config.return_factor_window,
                min_periods=self.config.return_factor_window // 2
            ).sum()
            
            # 计算最终的涨跌幅因子
            return_factor = up_factor - down_factor
            
            return return_factor
            
        except Exception as e:
            logger.error(f"Error calculating return factor: {e}")
            return pd.Series(0, index=prices.index)
    
    def calculate_positions(self, signals: Dict[str, float]) -> Dict[str, float]:
        """计算目标仓位"""
        try:
            # 对信号进行排序
            sorted_signals = sorted(
                signals.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # 初始化仓位
            positions = {symbol: 0.0 for symbol in self.symbols}
            
            # 设置信号阈值（降低阈值以更容易进入市场）
            signal_threshold = 0.1  # 降低信号阈值
            
            # 根据信号强度分配仓位
            total_position = 0.0
            
            for symbol, signal in sorted_signals:
                if signal > signal_threshold and total_position < self.config.position_size_limit:
                    # 计算目标仓位大小
                    position_size = min(
                        self.config.position_size_limit / self.config.max_positions,  # 单个ETF的最大仓位
                        self.config.position_size_limit - total_position,  # 剩余可用仓位
                        self.config.position_size_limit * (signal - signal_threshold) / (1 - signal_threshold)  # 基于信号强度的仓位
                    )
                    
                    positions[symbol] = position_size
                    total_position += position_size
                    
                    if len([p for p in positions.values() if p > 0]) >= self.config.max_positions:
                        break
            
            if total_position > 0:
                logger.info(f"Taking positions: {', '.join([f'{s}: {p:.2%}' for s, p in positions.items() if p > 0])}")
            else:
                logger.info("No strong signals, staying in cash")
                
            return positions
            
        except Exception as e:
            logger.error(f"Error in calculate_positions: {e}")
            return {symbol: 0.0 for symbol in self.symbols}
    
    def apply_risk_management(self, positions: Dict[str, float], current_date: pd.Timestamp) -> Dict[str, float]:
        """应用风险管理规则"""

        data_before_current_date = self.current_data.loc[:current_date]

        try:
            # 检查最大回撤
            if len(self.returns_history) > 0:
                # 使用numpy数组进行计算以提高性能
                returns_array = np.array(self.returns_history)
                cumulative_returns = np.exp(np.cumsum(np.log1p(returns_array)))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (running_max - cumulative_returns) / running_max
                current_drawdown = drawdowns[-1]
                
                # 如果回撤超过阈值，逐步减仓
                if current_drawdown > self.config.max_drawdown_threshold * 0.8:
                    # 计算减仓比例
                    reduction_ratio = 1.0 - (current_drawdown - self.config.max_drawdown_threshold * 0.8) / (self.config.max_drawdown_threshold * 0.2)
                    reduction_ratio = max(0.0, min(1.0, reduction_ratio))
                    
                    logger.info(f"Reducing positions due to drawdown: {current_drawdown:.2%}, "
                              f"reduction ratio: {reduction_ratio:.2%}")
                    
                    # 应用减仓
                    for symbol in positions:
                        positions[symbol] *= reduction_ratio
                    
                    if reduction_ratio == 0.0:
                        logger.info("Maximum drawdown threshold exceeded, closing all positions")
                        return {symbol: 0.0 for symbol in self.symbols}
            
            # 检查个股止损
            for symbol in self.symbols:
                if positions[symbol] > 0:  # 只检查有持仓的ETF
                    try:
                        prices = data_before_current_date[f'{symbol}_close']
                        current_price = prices.iloc[-1]
                        
                        # 使用过去20天的最高价作为参考
                        highest_price = prices.iloc[-20:].max()
                        drawdown = (highest_price - current_price) / highest_price
                        
                        # 计算20日ATR
                        high = data_before_current_date[f'{symbol}_high'].iloc[-20:]
                        low = data_before_current_date[f'{symbol}_low'].iloc[-20:]
                        close = prices.iloc[-20:]
                        tr = pd.DataFrame({
                            'hl': high - low,
                            'hc': abs(high - close.shift(1)),
                            'lc': abs(low - close.shift(1))
                        }).max(axis=1)
                        atr = tr.mean()
                        
                        # 计算动态止损阈值
                        stop_loss_threshold = max(
                            self.config.stop_loss_threshold,
                            2 * atr / current_price  # 使用2倍ATR作为最小止损阈值
                        )
                        
                        # 如果回撤超过阈值，清空仓位
                        if drawdown > stop_loss_threshold:
                            logger.info(f"Stop loss triggered for {symbol}: "
                                      f"drawdown={drawdown:.2%}, threshold={stop_loss_threshold:.2%}")
                            positions[symbol] = 0.0
                            
                    except Exception as e:
                        logger.error(f"Error in stop loss check for {symbol}: {e}")
                        # 在出错时保持原有仓位
                        continue
                        
            return positions
            
        except Exception as e:
            logger.error(f"Error in apply_risk_management: {e}")
            return {symbol: 0.0 for symbol in self.symbols}
    
    def run(self, start_date: str, end_date: str) -> pd.Series:
        """运行策略"""
        try:
            # 获取数据
            self.current_data = self.data_loader.get_data(start_date, end_date)
            
            # 确保所有需要的列都存在
            required_columns = []
            for symbol in self.symbols:
                required_columns.extend([
                    f'{symbol}_open',
                    f'{symbol}_close',
                    f'{symbol}_high',
                    f'{symbol}_low',
                    f'{symbol}_volume'
                ])
            
            missing_columns = [col for col in required_columns 
                             if col not in self.current_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # 初始化组合，使用第一个收盘价作为基准
            benchmark_price = self.current_data[f'{self.symbols[0]}_close'].iloc[0]
            self.portfolio_value = benchmark_price
            portfolio_values = [self.portfolio_value]
            self.positions = {symbol: 0.0 for symbol in self.symbols}
            self.returns_history = []
            
            # 初始化仓位历史
            self.position_history = {symbol: [] for symbol in self.symbols}
            
            # 按日期遍历
            dates = self.current_data.index
            for i in range(len(dates)):
                try:
                    current_date = dates[i]
                    
                    # 记录仓位历史
                    for symbol in self.symbols:
                        self.position_history[symbol].append(self.positions.get(symbol, 0))
                    
                    if i > 0:
                        prev_date = dates[i-1]
                        # 计算当日收益
                        daily_returns = {}
                        for symbol in self.symbols:
                            try:
                                current_price = self.current_data[f'{symbol}_close'].iloc[i]
                                prev_price = self.current_data[f'{symbol}_close'].iloc[i-1]
                                if pd.isna(current_price) or pd.isna(prev_price) or prev_price == 0:
                                    daily_returns[symbol] = 0.0
                                    logger.warning(f"Invalid price data for {symbol} at {current_date}")
                                else:
                                    daily_returns[symbol] = current_price / prev_price - 1
                            except Exception as e:
                                logger.error(f"Error calculating returns for {symbol}: {e}")
                                daily_returns[symbol] = 0.0
                        
                        # 更新组合价值
                        portfolio_return = sum(
                            self.positions[symbol] * daily_returns[symbol]
                            for symbol in self.symbols
                        )
                        
                        self.portfolio_value *= (1 + portfolio_return)
                        self.returns_history.append(portfolio_return)
                    
                    portfolio_values.append(self.portfolio_value)
                    
                    # 判断是否需要调仓
                    if self.should_rebalance(current_date):
                        # 生成信号
                        signals = self.generate_signals(current_date)
                        
                        # 计算目标仓位
                        self.positions = self.calculate_positions(signals)
                        
                        # 应用风险管理
                        self.positions = self.apply_risk_management(self.positions, current_date)
                        
                        # 记录调仓信息
                        logger.info(f"Rebalancing at {current_date}")
                        for symbol, position in self.positions.items():
                            logger.info(f"Position for {symbol}: {position:.2%}")
                
                except Exception as e:
                    logger.error(f"Error processing date {current_date}: {e}")
                    continue
            
            # 确保返回的序列长度与日期索引匹配
            if len(portfolio_values) != len(dates):
                logger.warning(f"Portfolio values length ({len(portfolio_values)}) "
                             f"does not match dates length ({len(dates)})")
                # 如果长度不匹配，填充缺失值
                if len(portfolio_values) < len(dates):
                    portfolio_values.extend([portfolio_values[-1]] * 
                                         (len(dates) - len(portfolio_values)))
                else:
                    portfolio_values = portfolio_values[:len(dates)]
            
            return pd.Series(portfolio_values, index=dates)
            
        except Exception as e:
            logger.error(f"Error in strategy run: {e}")
            # 返回初始值序列
            return pd.Series([benchmark_price] * len(self.current_data.index), 
                           index=self.current_data.index)
    
    def should_rebalance(self, date: pd.Timestamp) -> bool:
        """判断是否需要调仓"""
        try:
            if self.config.rebalance_freq == 'M':
                return date.day == 1
            elif self.config.rebalance_freq == 'W':
                return date.weekday() == 0
            else:
                return True
        except Exception as e:
            logger.error(f"Error in should_rebalance: {e}")
            return False
    
    def get_portfolio_returns(self) -> pd.Series:
        """获取组合收益率序列"""
        return pd.Series(self.returns_history)
    
    def get_position_history(self) -> pd.DataFrame:
        """获取仓位历史"""
        return pd.DataFrame(self.position_history, index=self.current_data.index) 