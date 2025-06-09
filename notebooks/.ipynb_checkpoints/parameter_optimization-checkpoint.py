import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bokeh.plotting import figure, show, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Select, Button
from bokeh.io import output_file
from itertools import product
import logging

from quant_system.strategies.etf_rotation import ETFRotationStrategy, ETFRotationConfig
from quant_system.data.data_loader import ETFDataLoader
from quant_system.analysis.performance import PerformanceAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    def __init__(self, etf_list, start_date, end_date):
        self.etf_list = etf_list
        self.start_date = start_date
        self.end_date = end_date
        self.data_loader = ETFDataLoader(etf_list)
        self.data = self.data_loader.get_data(start_date, end_date)
        
    def grid_search(self):
        """执行网格搜索"""
        logger.info("开始网格搜索...")
        
        # 定义参数网格
        param_grid = {
            'momentum_window': [20, 30, 40],
            'vol_window': [10, 15, 20],
            'rsi_window': [10, 14, 20],
            'momentum_weight': [0.3, 0.4, 0.5],
            'rsi_weight': [0.2, 0.3, 0.4],
            'macd_weight': [0.2, 0.3, 0.4]
        }
        
        results = []
        
        # 生成所有参数组合
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        total_combinations = len(param_combinations)
        
        for i, params in enumerate(param_combinations, 1):
            logger.info(f"测试参数组合 {i}/{total_combinations}")
            
            # 创建配置
            config = ETFRotationConfig(
                momentum_window=params['momentum_window'],
                vol_window=params['vol_window'],
                rsi_window=params['rsi_window'],
                rebalance_freq='W',
                risk_free_rate=0.02,
                max_drawdown_threshold=0.15,
                stop_loss_threshold=0.08,
                position_size_limit=0.5,
                transaction_cost=0.003,
                momentum_weight=params['momentum_weight'],
                rsi_weight=params['rsi_weight'],
                macd_weight=params['macd_weight']
            )
            
            # 运行策略
            strategy = ETFRotationStrategy(self.etf_list, config)
            portfolio_values = strategy.run(self.start_date, self.end_date)
            
            # 计算性能指标
            analyzer = PerformanceAnalyzer(portfolio_values, self.data['510300_close'])
            
            results.append({
                'params': params,
                'total_return': analyzer.calculate_total_return(),
                'sharpe_ratio': analyzer.calculate_sharpe_ratio(),
                'max_drawdown': analyzer.calculate_max_drawdown(),
                'annual_return': analyzer.calculate_annual_return(),
                'volatility': analyzer.calculate_volatility(),
                'portfolio_values': portfolio_values
            })
        
        return results
    
    def create_interactive_optimization(self):
        """创建交互式优化界面"""
        logger.info("创建交互式优化界面...")
        
        # 创建输出HTML文件
        output_file("parameter_optimization.html")
        
        # 创建交互式控件
        momentum_window = Slider(title="动量窗口", value=30, start=10, end=60, step=5)
        vol_window = Slider(title="波动率窗口", value=15, start=5, end=30, step=5)
        rsi_window = Slider(title="RSI窗口", value=14, start=5, end=30, step=1)
        momentum_weight = Slider(title="动量权重", value=0.4, start=0.1, end=0.6, step=0.1)
        rsi_weight = Slider(title="RSI权重", value=0.3, start=0.1, end=0.6, step=0.1)
        macd_weight = Slider(title="MACD权重", value=0.3, start=0.1, end=0.6, step=0.1)
        
        # 创建图表
        p1 = figure(title="策略收益曲线", x_axis_label="日期", y_axis_label="净值")
        p2 = figure(title="性能指标", x_axis_label="指标", y_axis_label="值")
        
        def update_plot(attr, old, new):
            # 创建配置
            config = ETFRotationConfig(
                momentum_window=momentum_window.value,
                vol_window=vol_window.value,
                rsi_window=rsi_window.value,
                rebalance_freq='W',
                risk_free_rate=0.02,
                max_drawdown_threshold=0.15,
                stop_loss_threshold=0.08,
                position_size_limit=0.5,
                transaction_cost=0.003,
                momentum_weight=momentum_weight.value,
                rsi_weight=rsi_weight.value,
                macd_weight=macd_weight.value
            )
            
            # 运行策略
            strategy = ETFRotationStrategy(self.etf_list, config)
            portfolio_values = strategy.run(self.start_date, self.end_date)
            
            # 计算性能指标
            analyzer = PerformanceAnalyzer(portfolio_values, self.data['510300_close'])
            
            # 更新收益曲线
            p1.renderers = []
            p1.line(range(len(portfolio_values)), portfolio_values, line_color="blue", legend_label="策略收益")
            p1.line(range(len(self.data['510300_close'])), self.data['510300_close'], line_color="red", legend_label="基准收益")
            
            # 更新性能指标
            metrics = {
                '总收益率': analyzer.calculate_total_return(),
                '夏普比率': analyzer.calculate_sharpe_ratio(),
                '最大回撤': analyzer.calculate_max_drawdown(),
                '年化收益率': analyzer.calculate_annual_return(),
                '波动率': analyzer.calculate_volatility()
            }
            
            p2.renderers = []
            p2.vbar(x=list(range(len(metrics))), top=list(metrics.values()), width=0.5)
            p2.xaxis.ticker = list(range(len(metrics)))
            p2.xaxis.major_label_overrides = {i: label for i, label in enumerate(metrics.keys())}
            
            # 输出当前参数组合的性能指标
            logger.info(f"当前参数组合性能指标：")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
        
        # 绑定更新事件
        for widget in [momentum_window, vol_window, rsi_window, momentum_weight, rsi_weight, macd_weight]:
            widget.on_change('value', update_plot)
        
        # 初始更新
        update_plot(None, None, None)
        
        # 显示布局
        controls = column(momentum_window, vol_window, rsi_window, momentum_weight, rsi_weight, macd_weight)
        plots = row(p1, p2)
        layout = column(controls, plots)
        
        # 保存到HTML文件
        save(layout)
        logger.info("交互式优化界面已保存到 parameter_optimization.html")
    
    def analyze_parameter_sensitivity(self, results):
        """分析参数敏感性"""
        logger.info("开始参数敏感性分析...")
        
        # 将结果转换为DataFrame
        results_df = pd.DataFrame([
            {**r['params'], 
             'total_return': r['total_return'],
             'sharpe_ratio': r['sharpe_ratio'],
             'max_drawdown': r['max_drawdown'],
             'annual_return': r['annual_return'],
             'volatility': r['volatility']}
            for r in results
        ])
        
        # 创建参数敏感性分析图表
        p = figure(title="参数敏感性分析", x_axis_label="参数值", y_axis_label="夏普比率")
        
        # 分析每个参数对夏普比率的影响
        for param in ['momentum_window', 'vol_window', 'rsi_window', 'momentum_weight', 'rsi_weight', 'macd_weight']:
            # 计算每个参数值对应的平均夏普比率
            param_sensitivity = results_df.groupby(param)['sharpe_ratio'].mean()
            
            # 绘制散点图
            p.scatter(param_sensitivity.index, param_sensitivity.values, 
                     size=10, legend_label=param)
        
        p.legend.click_policy = 'hide'
        
        # 保存敏感性分析图表
        output_file("parameter_sensitivity.html")
        save(p)
        logger.info("参数敏感性分析图表已保存到 parameter_sensitivity.html")
        
        # 输出参数相关性分析
        logger.info("参数相关性分析：")
        correlation_matrix = results_df.corr()['sharpe_ratio'].sort_values(ascending=False)
        logger.info(correlation_matrix)
        
        # 输出最优参数组合
        logger.info("\n最优参数组合：")
        logger.info(f"最佳夏普比率参数组合：\n{results_df.loc[results_df['sharpe_ratio'].idxmax()]}")
        logger.info(f"最佳收益率参数组合：\n{results_df.loc[results_df['total_return'].idxmax()]}")

def main():
    # 设置回测参数
    etf_list = ['510300', '159915', '518880', '513100']
    start_date = '2020-01-01'
    end_date = '2024-03-10'
    
    # 创建优化器实例
    optimizer = ParameterOptimizer(etf_list, start_date, end_date)
    
    # 执行网格搜索
    results = optimizer.grid_search()
    
    # 创建交互式优化界面
    optimizer.create_interactive_optimization()
    
    # 分析参数敏感性
    optimizer.analyze_parameter_sensitivity(results)

if __name__ == "__main__":
    main() 