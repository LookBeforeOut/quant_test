import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import quantstats as qs
from bokeh.plotting import figure, show, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool, Div
from bokeh.palettes import Category10
import logging
from bokeh.io import output_file as bokeh_output_file
from bokeh.resources import CDN
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """策略性能分析器"""
    
    def __init__(self, portfolio_values: pd.Series, benchmark_values: Optional[pd.Series] = None):
        """初始化性能分析器"""
        try:
            # 数据验证
            if portfolio_values is None or len(portfolio_values) == 0:
                raise ValueError("Portfolio values cannot be None or empty")
            
            # 确保portfolio_values是Series类型且索引为datetime
            if not isinstance(portfolio_values, pd.Series):
                portfolio_values = pd.Series(portfolio_values)
            if not isinstance(portfolio_values.index, pd.DatetimeIndex):
                portfolio_values.index = pd.to_datetime(portfolio_values.index)
            
            # 处理无效值
            portfolio_values = portfolio_values.replace([np.inf, -np.inf], np.nan)
            portfolio_values = portfolio_values.ffill().bfill()

            # 如果有benchmark，确保其为Series类型且索引为datetime
            if benchmark_values is not None:
                if not isinstance(benchmark_values, pd.Series):
                    benchmark_values = pd.Series(benchmark_values)
                if not isinstance(benchmark_values.index, pd.DatetimeIndex):
                    benchmark_values.index = pd.to_datetime(benchmark_values.index)
                # 处理无效值
                benchmark_values = benchmark_values.replace([np.inf, -np.inf], np.nan)
                benchmark_values = benchmark_values.ffill().bfill()

            # 对齐数据
            if benchmark_values is not None:
                common_index = portfolio_values.index.intersection(benchmark_values.index)
                if len(common_index) == 0:
                    raise ValueError("No common dates between portfolio and benchmark")
                self.portfolio_values = portfolio_values[common_index]
                self.benchmark_values = benchmark_values[common_index]
            else:
                self.portfolio_values = portfolio_values
                self.benchmark_values = None
            
            # 计算收益率
            self.returns = self.portfolio_values.pct_change()
            self.returns = self.returns.replace([np.inf, -np.inf], 0).fillna(0)
            
            if self.benchmark_values is not None:
                self.benchmark_returns = self.benchmark_values.pct_change()
                self.benchmark_returns = self.benchmark_returns.replace([np.inf, -np.inf], 0).fillna(0)
            else:
                self.benchmark_returns = None
                
        except Exception as e:
            logger.error(f"Error in PerformanceAnalyzer initialization: {e}")
            raise
        
    def calculate_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        metrics = {
            'total_return': self.calculate_total_return(),
            'annual_return': self.calculate_annual_return(),
            'volatility': self.calculate_volatility(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor()
        }
        
        if self.benchmark_returns is not None:
            metrics.update({
                'beta': self.calculate_beta(),
                'information_ratio': self.calculate_information_ratio()
            })
            
        return metrics
    
    def calculate_total_return(self) -> float:
        """计算总收益率"""
        if len(self.returns) < 2:
            return 0.0
        return ((1 + self.returns).prod() - 1) * 100
    
    def calculate_annual_return(self) -> float:
        """计算年化收益率"""
        if len(self.returns) < 2:
            return 0.0
        total_days = (self.returns.index[-1] - self.returns.index[0]).days
        if total_days == 0:
            return 0.0
        total_return = self.calculate_total_return() / 100
        return (pow(1 + total_return, 365 / total_days) - 1) * 100
    
    def calculate_volatility(self) -> float:
        """计算波动率"""
        if len(self.returns) < 2:
            return 0.0
        return self.returns.std() * np.sqrt(252) * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(self.returns) < 2:
            return 0.0
        # 使用年化无风险收益率计算每日无风险收益率
        daily_rf_rate = pow(1 + risk_free_rate, 1/252) - 1
        excess_returns = self.returns - daily_rf_rate
        volatility = excess_returns.std() * np.sqrt(252)
        if volatility == 0:
            return 0.0
        return excess_returns.mean() * np.sqrt(252) / volatility
    
    def calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.portfolio_values) < 2:
            return 0.0
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return abs(drawdowns.min()) * 100
    
    def calculate_win_rate(self) -> float:
        """计算胜率"""
        if len(self.returns) < 2:
            return 0.0
        return (self.returns > 0).mean() * 100
    
    def calculate_profit_factor(self) -> float:
        """计算盈亏比"""
        if len(self.returns) < 2:
            return 0.0
        positive_returns = self.returns[self.returns > 0].sum()
        negative_returns = abs(self.returns[self.returns < 0].sum())
        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 0.0
        return positive_returns / negative_returns
    
    def calculate_beta(self) -> float:
        """计算Beta"""
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0
        # 计算协方差和方差
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        variance = np.var(self.benchmark_returns)
        return covariance / variance if variance != 0 else 0.0
    
    def calculate_information_ratio(self) -> float:
        """计算信息比率"""
        if self.benchmark_returns is None or len(self.returns) < 2:
            return 0.0
        # 计算超额收益
        excess_returns = self.returns - self.benchmark_returns
        # 计算跟踪误差
        tracking_error = excess_returns.std() * np.sqrt(252)
        if tracking_error == 0:
            return 0.0
        # 计算年化超额收益
        annual_excess_return = excess_returns.mean() * np.sqrt(252)
        return annual_excess_return / tracking_error
    
    def plot_performance(self, output_file: str = "strategy_performance.html"):
        """绘制策略表现图表"""
        try:
            if len(self.portfolio_values) < 2:
                logger.warning("Insufficient data points for plotting")
                return
                
            # 数据预处理
            plot_data = self.portfolio_values.copy()
            plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
            plot_data = plot_data.ffill().bfill()
            
            # 设置输出文件和资源
            bokeh_output_file(output_file, title='Strategy Performance Analysis')
            
            # 创建图表工具
            tools = [
                HoverTool(tooltips=[
                    ('Date', '@x{%F}'),
                    ('Value', '@y{0.00}')
                ], formatters={'@x': 'datetime'}),
                'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save',
                CrosshairTool()
            ]
            
            # 净值曲线
            p1 = figure(width=800, height=400, x_axis_type='datetime',
                       title='Portfolio Performance', tools=tools)
            
            # 将数据转换为ColumnDataSource格式
            source = ColumnDataSource({
                'x': plot_data.index,
                'y': plot_data.values
            })
            
            # 添加策略曲线
            p1.line('x', 'y', source=source, line_color='blue', legend_label='Strategy')
            
            # 如果有基准，添加基准曲线
            if self.benchmark_values is not None:
                benchmark_data = self.benchmark_values.copy()
                benchmark_data = benchmark_data.replace([np.inf, -np.inf], np.nan)
                benchmark_data = benchmark_data.ffill().bfill()
                
                benchmark_source = ColumnDataSource({
                    'x': benchmark_data.index,
                    'y': benchmark_data.values
                })
                p1.line('x', 'y', source=benchmark_source, line_color='red', legend_label='Benchmark')
            
            p1.legend.click_policy = 'hide'
            p1.legend.location = "top_left"
            
            # 回撤图
            drawdown = self.calculate_drawdown_series()
            drawdown = drawdown.replace([np.inf, -np.inf], np.nan)
            drawdown = drawdown.fillna(0)
            
            drawdown_source = ColumnDataSource({
                'x': drawdown.index,
                'y': drawdown.values * 100
            })
            
            p2 = figure(width=800, height=200, x_axis_type='datetime',
                       title='Drawdown', tools=tools)
            p2.line('x', 'y', source=drawdown_source, line_color='red', legend_label='Drawdown')
            p2.legend.location = "bottom_left"
            
            # 保存图表
            save(column(p1, p2), filename=output_file, resources=CDN)
            logger.info(f"Performance charts generated successfully: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")
            raise
    
    def calculate_drawdown_series(self) -> pd.Series:
        """计算回撤序列"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        return cumulative / running_max - 1
    
    def plot_position_history(self, position_history: pd.DataFrame, output_file: str = "position_history.html"):
        """绘制持仓历史图表"""
        try:
            if position_history is None or position_history.empty:
                logger.warning("No position history data available for plotting")
                return
                
            # 数据预处理
            plot_data = position_history.copy()
            plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
            plot_data = plot_data.ffill().bfill()
            
            # 设置输出文件和资源
            bokeh_output_file(output_file, title='Position History Analysis')
            
            # 创建图表工具
            tools = [
                HoverTool(tooltips=[
                    ('Date', '@x{%F}'),
                    ('Position', '@y{0.00%}')
                ], formatters={'@x': 'datetime'}),
                'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save',
                CrosshairTool()
            ]
            
            # 创建图表
            p = figure(width=800, height=400, x_axis_type='datetime',
                      title='Position History', tools=tools)
            
            # 为每个ETF添加一条线
            colors = Category10[10]  # 使用预定义的颜色方案
            for i, column in enumerate(plot_data.columns):
                source = ColumnDataSource({
                    'x': plot_data.index,
                    'y': plot_data[column].values
                })
                p.line('x', 'y', source=source, line_color=colors[i % len(colors)],
                      legend_label=column, line_width=2)
            
            p.legend.click_policy = 'hide'
            p.legend.location = "top_left"
            
            # 保存图表
            save(p, filename=output_file, resources=CDN)
            logger.info(f"Position history chart generated successfully: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating position history chart: {e}")
            raise

    def generate_report(self, output_file: str = "strategy_report.html", position_history: Optional[pd.DataFrame] = None):
        """生成策略报告"""
        try:
            # 创建报告目录
            report_dir = os.path.dirname(output_file)
            if report_dir and not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            # 生成性能图表
            performance_file = os.path.join(report_dir, "performance.html")
            self.plot_performance(performance_file)
            
            # 生成持仓历史图表
            if position_history is not None:
                position_file = os.path.join(report_dir, "position_history.html")
                self.plot_position_history(position_history, position_file)
            
            # 创建HTML报告
            html_content = f"""
            <html>
            <head>
                <title>Strategy Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ margin: 10px 0; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .chart-container {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Strategy Analysis Report</h1>
                    
                    <div class="section">
                        <h2>Performance Metrics</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Total Return</td>
                                <td>{self.calculate_total_return():.2f}%</td>
                            </tr>
                            <tr>
                                <td>Annual Return</td>
                                <td>{self.calculate_annual_return():.2f}%</td>
                            </tr>
                            <tr>
                                <td>Volatility</td>
                                <td>{self.calculate_volatility():.2f}%</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>{self.calculate_sharpe_ratio():.2f}</td>
                            </tr>
                            <tr>
                                <td>Max Drawdown</td>
                                <td>{self.calculate_max_drawdown():.2f}%</td>
                            </tr>
                            <tr>
                                <td>Win Rate</td>
                                <td>{self.calculate_win_rate():.2f}%</td>
                            </tr>
                            <tr>
                                <td>Profit Factor</td>
                                <td>{self.calculate_profit_factor():.2f}</td>
                            </tr>
                            <tr>
                                <td>Beta</td>
                                <td>{self.calculate_beta():.2f}</td>
                            </tr>
                            <tr>
                                <td>Information Ratio</td>
                                <td>{self.calculate_information_ratio():.2f}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Performance Charts</h2>
                        <div class="chart-container">
                            <iframe src="performance.html" width="100%" height="600px" frameborder="0"></iframe>
                        </div>
                    </div>
                    
                    {f'''
                    <div class="section">
                        <h2>Position History</h2>
                        <div class="chart-container">
                            <iframe src="position_history.html" width="100%" height="400px" frameborder="0"></iframe>
                        </div>
                    </div>
                    ''' if position_history is not None else ''}
                </div>
            </body>
            </html>
            """
            
            # 保存HTML报告
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Strategy report generated successfully: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating strategy report: {e}")
            raise

    def calculate_benchmark_returns(self, benchmark_etf: str) -> pd.Series:
        """计算基准收益"""
        try:
            # 使用与策略相同的计算方法
            benchmark_prices = self.df[f'{benchmark_etf}_close']
            benchmark_returns = benchmark_prices.pct_change()
            benchmark_returns = benchmark_returns.fillna(0)
            
            # 计算累积收益
            benchmark_nav = (1 + benchmark_returns).cumprod()
            
            return benchmark_nav
            
        except Exception as e:
            logger.error(f"Error calculating benchmark returns: {e}")
            return pd.Series(1.0, index=self.df.index) 