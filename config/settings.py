from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import yaml

@dataclass
class SystemConfig:
    data_dir: Path
    log_dir: Path
    output_dir: Path
    database_url: str
    api_keys: Dict[str, str]

@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float
    transaction_cost: float
    slippage: float

def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_system_config(config_dict: dict) -> SystemConfig:
    """构建系统配置"""
    return SystemConfig(
        data_dir=Path(config_dict['paths']['data_dir']),
        log_dir=Path(config_dict['paths']['log_dir']),
        output_dir=Path(config_dict['paths']['output_dir']),
        database_url=config_dict['database']['url'],
        api_keys=config_dict['api_keys']
    )

def get_backtest_config(config_dict: dict) -> BacktestConfig:
    """构建回测配置"""
    return BacktestConfig(
        start_date=config_dict['backtest']['start_date'],
        end_date=config_dict['backtest']['end_date'],
        initial_capital=config_dict['backtest']['initial_capital'],
        transaction_cost=config_dict['backtest']['transaction_cost'],
        slippage=config_dict['backtest']['slippage']
    ) 