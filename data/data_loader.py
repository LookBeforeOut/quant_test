import pandas as pd
import akshare as ak
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import numpy as np
import os
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row
from bokeh.models import HoverTool, CrosshairTool, Select, ColumnDataSource, CustomJS, TabPanel, Tabs
import json

logger = logging.getLogger(__name__)

class DataLoader(ABC):
    """数据加载器基类"""
    
    @abstractmethod
    def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取数据"""
        pass

    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据"""
        pass

    def visualize_data(self, df: pd.DataFrame, output_dir: str = "visualization") -> str:
        """将数据可视化并保存为HTML文件
        
        Args:
            df: 要可视化的数据框
            output_dir: 输出目录，默认为 "visualization"
            
        Returns:
            str: 生成的HTML文件路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(output_dir, f"data_visualization_{timestamp}.html")
        
        # 设置输出文件
        output_file(output_file_path)
        
        # 创建标签页列表
        tabs = []
        
        # 为每个ETF创建标签页
        for symbol in self.symbols:
            # 创建价格走势图
            price_fig = figure(
                title=f"{symbol} 价格走势",
                x_axis_label="日期",
                y_axis_label="价格",
                x_axis_type="datetime",
                width=1000,
                height=400
            )
            
            # 添加收盘价线
            price_fig.line(
                df.index,
                df[f"{symbol}_close"],
                line_color="blue",
                legend_label="收盘价"
            )
            
            # 添加成交量柱状图
            volume_fig = figure(
                title=f"{symbol} 成交量",
                x_axis_label="日期",
                y_axis_label="成交量",
                x_axis_type="datetime",
                width=1000,
                height=200
            )
            
            volume_fig.vbar(
                df.index,
                top=df[f"{symbol}_volume"],
                width=0.5,
                color="gray",
                alpha=0.5
            )
            
            # 添加交互工具
            for fig in [price_fig, volume_fig]:
                fig.add_tools(HoverTool())
                fig.add_tools(CrosshairTool())
            
            # 创建标签页
            tab = TabPanel(
                child=column(price_fig, volume_fig),
                title=symbol
            )
            tabs.append(tab)
        
        # 创建标签页组件并保存
        tabs_layout = Tabs(tabs=tabs)
        save(tabs_layout)
        
        logger.info(f"数据可视化已保存到: {output_file_path}")
        return output_file_path

class ETFDataLoader(DataLoader):
    """ETF数据加载器"""
    
    def __init__(self, symbols: List[str], cache_dir: str = "data_cache"):
        self.symbols = symbols
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self._init_cache_dir()
        
    def _init_cache_dir(self):
        """初始化缓存目录"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self._save_metadata({})
            
    def _save_metadata(self, metadata: dict):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
    def _load_metadata(self) -> dict:
        """加载元数据"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def _get_cache_file(self, symbol: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{symbol}.csv")
        
    def _is_data_stale(self, symbol: str, max_age_days: int = 1) -> bool:
        """检查数据是否过期"""
        metadata = self._load_metadata()
        if symbol not in metadata:
            return True
            
        last_update = datetime.fromisoformat(metadata[symbol]['last_update'])
        return (datetime.now() - last_update).days > max_age_days
        
    def _update_cache(self, symbol: str, df: pd.DataFrame):
        """更新缓存数据"""
        cache_file = self._get_cache_file(symbol)
        df.to_csv(cache_file)
        
        metadata = self._load_metadata()
        metadata[symbol] = {
            'last_update': datetime.now().isoformat(),
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d')
        }
        self._save_metadata(metadata)
        
    def _load_from_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_file = self._get_cache_file(symbol)
        if not os.path.exists(cache_file):
            return None
            
        df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
        df = df.loc[start_date:end_date]
        return df
        
    def get_data(self, start_date: str, end_date: str, force_update: bool = False) -> pd.DataFrame:
        """获取ETF数据, 支持离线缓存"""
        dfs = {}
        
        for symbol in self.symbols:
            try:
                # 检查是否需要更新数据
                if force_update or self._is_data_stale(symbol):
                    logger.info(f"Fetching fresh data for ETF {symbol}")
                    df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="hfq")
                    df = self.process_data(df, symbol)
                    self._update_cache(symbol, df)
                else:
                    logger.info(f"Loading cached data for ETF {symbol}")
                    df = self._load_from_cache(symbol, start_date, end_date)
                    if df is None or df.empty:
                        logger.warning(f"No cached data found for {symbol}, fetching fresh data")
                        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="hfq")
                        df = self.process_data(df, symbol)
                        self._update_cache(symbol, df)
                
                dfs[symbol] = df
                logger.info(f"Successfully loaded data for ETF {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for ETF {symbol}: {e}")
                continue
        
        # 合并所有ETF数据
        combined_df = pd.concat(dfs.values(), axis=1)
        logger.info(f"Combined data columns: {combined_df.columns}")
        
        combined_df = combined_df.loc[start_date:end_date]
        return combined_df
        
    def update_all_data(self):
        """更新所有ETF的缓存数据"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.get_data(start_date, end_date, force_update=True)
        
    def process_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """处理ETF数据"""
        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '开盘': f'{symbol}_open',
            '收盘': f'{symbol}_close',
            '最高': f'{symbol}_high',
            '最低': f'{symbol}_low',
            '成交量': f'{symbol}_volume'
        })
        
        # 转换数据类型
        for col in [f'{symbol}_open', f'{symbol}_close', f'{symbol}_high', f'{symbol}_low']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            
        # 设置索引
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df

class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """计算收益率"""
        returns = prices.pct_change(fill_method=None)
        return returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    @staticmethod
    def calculate_total_return(prices: pd.Series) -> float:
        """计算整个时间段的总收益率"""
        if prices.empty or prices.iloc[0] == 0:
            return np.nan  # 处理异常情况
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
        """计算波动率"""
        # 使用ewm来计算波动率，减少前期数据不足的影响
        volatility = returns.ewm(span=window, min_periods=window//2).std() * np.sqrt(252)
        return volatility.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    @staticmethod
    def calculate_momentum(prices: pd.Series | pd.DataFrame, window: int = 20) -> pd.Series | pd.DataFrame:
        """计算动量指标"""
        try:
            # 转换数据类型为float64以确保计算精度
            prices = prices.astype(np.float64)
            
            # 计算收益率
            if isinstance(prices, pd.DataFrame):
                returns = np.diff(np.log(prices.values), axis=0)
                returns = np.vstack([np.zeros(returns.shape[1]), returns])  # 补充第一行
            else:
                returns = np.diff(np.log(prices.values))
                returns = np.append([0], returns)  # 补充第一个值
            
            # 使用cumsum方法计算滚动窗口和
            cumsum = np.cumsum(returns, axis=0)
            if len(cumsum) > window:
                momentum = cumsum[window:] - cumsum[:-window]
                # 补充前window个值
                if isinstance(prices, pd.DataFrame):
                    momentum = np.vstack([np.zeros((window, momentum.shape[1])), momentum])
                else:
                    momentum = np.append(np.zeros(window), momentum)
            else:
                momentum = np.zeros_like(returns)
            
            # 转换回pandas对象
            if isinstance(prices, pd.DataFrame):
                momentum = pd.DataFrame(momentum, index=prices.index, columns=prices.columns)
            else:
                momentum = pd.Series(momentum, index=prices.index, name=prices.name)
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            # 根据输入类型返回相应的全0对象
            if isinstance(prices, pd.DataFrame):
                return pd.DataFrame(0, index=prices.index, columns=prices.columns)
            else:
                return pd.Series(0, index=prices.index, name=prices.name)
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI"""
        try:
            # 计算价格变化
            delta = prices.diff().fillna(0)
            
            # 使用numpy操作代替pandas where
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # 转换为Series以使用rolling
            gains = pd.Series(gains, index=prices.index)
            losses = pd.Series(losses, index=prices.index)
            
            # 使用SMA计算平均值
            avg_gains = gains.rolling(window=window, min_periods=window//2).mean()
            avg_losses = losses.rolling(window=window, min_periods=window//2).mean()
            
            # 计算RS和RSI
            rs = np.zeros(len(prices))
            valid_mask = avg_losses != 0
            rs[valid_mask] = avg_gains[valid_mask] / avg_losses[valid_mask]
            
            # 计算RSI
            rsi = 100 - (100 / (1 + rs))
            
            # 处理极端情况
            rsi = np.where(avg_losses == 0, 100, rsi)  # 如果没有下跌，RSI = 100
            rsi = np.where(avg_gains == 0, 0, rsi)     # 如果没有上涨，RSI = 0
            
            return pd.Series(rsi, index=prices.index).fillna(50)
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=prices.index)  # 在出错时返回中性值
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD"""
        # 计算快线和慢线
        exp1 = prices.ewm(span=fast, min_periods=fast//2).mean()
        exp2 = prices.ewm(span=slow, min_periods=slow//2).mean()
        
        # 计算MACD线和信号线
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, min_periods=signal//2).mean()
        
        # 计算MACD柱状图
        macd_hist = macd_line - signal_line
        
        # 归一化MACD
        std_macd = macd_hist.rolling(window=21, min_periods=10).std()
        norm_macd = macd_hist / std_macd.replace(0, np.nan)
        
        return norm_macd.replace([np.inf, -np.inf], np.nan).fillna(0) 