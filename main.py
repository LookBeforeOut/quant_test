import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging
import os
import sys
from strategies.etf_rotation import ETFRotationStrategy, ETFRotationConfig
from strategies.etf_rotation_v2 import ETFRotationStrategyV2, ETFRotationConfigV2
from analysis.performance import PerformanceAnalyzer
from data.data_loader import ETFDataLoader


# 配置日志
def setup_logging():
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 生成日志文件名，包含时间戳
    log_filename = f'logs/quant_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别为DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Quant System...")

    try:
        # ETF列表
        etf_list = ['510300', '510500', '159915', '518880', '513100', '159928', '515000'] # 沪深300、中证500、创业板ETF、黄金ETF、纳指ETF、消费ETF、科技ETF

        # 创建策略配置
        # config = ETFRotationConfig(
        #     momentum_window=30,      # 降低动量窗口以提高灵敏度
        #     vol_window=15,           # 降低波动率窗口以更快捕捉市场变化
        #     rsi_window=14,           # 保持RSI窗口不变
        #     rebalance_freq='W',      # 每周调仓
        #     risk_free_rate=0.02,     # 无风险利率
        #     max_drawdown_threshold=0.15,  # 最大回撤阈值
        #     stop_loss_threshold=0.08,     # 止损阈值
        #     position_size_limit=0.5,      # 最大仓位
        #     transaction_cost=0.003,       # 交易成本
        #     momentum_weight=0.4,      # 动量权重
        #     rsi_weight=0.3,          # RSI权重
        #     macd_weight=0.3,         # MACD权重
        #     volume_std_weight=0.0,  # 成交金额标准差权重
        #     return_factor_weight=0.0,  # 涨跌幅因子权重
        #     max_positions=3,        # 最多持有3个ETF
        #     volume_std_window=10,     # 成交金额标准差计算窗口
        #     return_factor_window=10  # 涨跌幅因子计算窗口
        # )

        config = ETFRotationConfigV2(
            momentum_window=30,      # 降低动量窗口以提高灵敏度
            vol_window=15,           # 降低波动率窗口以更快捕捉市场变化
            rsi_window=14,           # 保持RSI窗口不变
            rebalance_freq='W',      # 每周调仓
            risk_free_rate=0.02,     # 无风险利率
            max_drawdown_threshold=0.15,  # 最大回撤阈值
            stop_loss_threshold=0.08,     # 止损阈值
            position_size_limit=1.0,      # 最大仓位
            transaction_cost=0.003,       # 交易成本
            momentum_weight=0.4,      # 动量权重
            rsi_weight=0.3,          # RSI权重
            macd_weight=0.3,         # MACD权重
            volume_std_weight=0.0,  # 成交金额标准差权重
            return_factor_weight=0.0,  # 涨跌幅因子权重
            max_positions=2,        # 最多持有2个ETF
            volume_std_window=10,     # 成交金额标准差计算窗口
            return_factor_window=10  # 涨跌幅因子计算窗口
        )

        # 设置回测时间范围
        start_date = '2020-01-01'
        end_date = '2025-03-10'

        # 创建数据加载器（使用默认的data_cache目录）
        data_loader = ETFDataLoader(etf_list, cache_dir="data/data_cache")
        
        # 首次运行或需要更新数据时，可以调用update_all_data()
        # data_loader.update_all_data()
        
        # 加载数据（会自动使用缓存）
        data = data_loader.get_data(start_date, end_date)
        html_path = data_loader.visualize_data(data)

        # 创建并运行策略
        strategy = ETFRotationStrategyV2(etf_list, config)
        portfolio_values = strategy.run(start_date, end_date)

        # 加载基准数据
        benchmark_values = None
        try:
            benchmark_values = data['510300_close']
        except KeyError:
            logger.warning("无法加载基准数据")

        # 创建性能分析器
        analyzer = PerformanceAnalyzer(portfolio_values, benchmark_values)

        # 计算性能指标
        total_return = analyzer.calculate_total_return()
        annual_return = analyzer.calculate_annual_return()
        volatility = analyzer.calculate_volatility()
        sharpe_ratio = analyzer.calculate_sharpe_ratio()
        max_drawdown = analyzer.calculate_max_drawdown()
        win_rate = analyzer.calculate_win_rate()
        profit_factor = analyzer.calculate_profit_factor()
        beta = analyzer.calculate_beta()
        information_ratio = analyzer.calculate_information_ratio()

        # 输出性能指标
        logger.info(f"总收益率: {total_return:.2%}")
        logger.info(f"年化收益率: {annual_return:.2%}")
        logger.info(f"波动率: {volatility:.2%}")
        logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        logger.info(f"最大回撤: {max_drawdown:.2%}")
        logger.info(f"胜率: {win_rate:.2%}")
        logger.info(f"盈亏比: {profit_factor:.2f}")
        logger.info(f"贝塔: {beta:.2f}")
        logger.info(f"信息比率: {information_ratio:.2f}")

        # 生成图表和报告
        try:
            # 获取持仓历史
            position_history = strategy.get_position_history()
            analyzer.generate_report("result/strategy_analysis.html", position_history)
        except Exception as e:
            logger.error(f"生成分析报告时出错: {str(e)}")

        # 获取持仓历史
        position_history = strategy.get_position_history()
        if position_history is not None:
            logger.info("持仓历史:")
            logger.info(position_history)

    except Exception as e:
        logger.error(f"主程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 