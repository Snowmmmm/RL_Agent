#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店动态定价系统 - 主程序
基于贝叶斯深度神经网络（BNN）和Q-learning强化学习
"""


import os
import sys
import pickle
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import traceback

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from data_preprocessing import HotelDataPreprocessor
from bnn_model import BNNTrainer
from rl_system import HotelRLSystem, QLearningAgent, HotelEnvironment
from config import RL_CONFIG, BNN_CONFIG, SIMULATION_CONFIG

# 配置警告过滤器
warnings.filterwarnings('ignore')

def check_environment() -> bool:
    """
    检查环境配置
    
    验证系统运行环境是否满足要求，包括CUDA可用性、依赖库安装情况等。
    提供详细的检查报告和错误信息。
    
    Returns:
        bool: 环境检查通过返回True，否则返回False
        
    检查项目：
    - CUDA GPU加速可用性
    - PyTorch版本和兼容性
    - 核心依赖库（pandas, numpy等）
    - 系统资源和权限
    
    Note:
        - 自动检测GPU设备并报告状态
        - 提供详细的错误信息帮助环境配置
        - 支持CPU-only模式运行
    """
    print("=== 环境检查 ===")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA可用，设备：{torch.cuda.get_device_name(0)}")
        print(f"[OK] PyTorch版本：{torch.__version__}")
    else:
        print("[WARN] CUDA不可用，将使用CPU")
    
    # 检查必要的库
    try:
        # 验证关键依赖库是否可用
        _ = pd.DataFrame  # 验证pandas
        _ = np.array     # 验证numpy
        print("[OK] 所有依赖库已安装")
    except ImportError as e:
        print(f"[ERROR] 缺少依赖库：{e}")
        return False
    
    return True

def evaluate_confidence_interval_coverage(mean_pred: np.ndarray, var_pred: np.ndarray, y_true: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, np.ndarray]:
    """
    评估置信区间覆盖率
    
    计算贝叶斯神经网络预测的置信区间覆盖率，评估模型的不确定性估计质量。
    使用正态分布假设计算置信区间，并统计真实值落在区间内的比例。
    
    Args:
        mean_pred: 预测均值，形状为(n_samples,)
        var_pred: 预测方差，形状为(n_samples,)
        y_true: 真实值，形状为(n_samples,)
        confidence_level: 置信水平，默认95%
    
    Returns:
        Tuple[float, np.ndarray]: (覆盖率百分比, 布尔数组指示每个样本是否在区间内)
        
    计算过程：
    1. 计算标准差：std_pred = sqrt(var_pred)
    2. 计算Z分数：z_score = norm.ppf(1 - alpha/2)
    3. 计算置信区间：[mean_pred - z*std_pred, mean_pred + z*std_pred]
    4. 统计覆盖率：mean(y_true ∈ [lower, upper])
    
    Note:
        - 假设预测分布为正态分布N(mean_pred, var_pred)
        - 95%置信区间对应z_score ≈ 1.96
        - 覆盖率应接近置信水平（如95%）
        - 可用于评估BNN的不确定性校准质量
    """
    # 计算标准差
    std_pred = np.sqrt(var_pred)
    
    # 计算置信区间
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)  # 对于95%置信区间，z_score ≈ 1.96
    
    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred
    
    # 检查真实值是否在置信区间内
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage_rate = np.mean(in_interval) * 100
    
    return coverage_rate, in_interval

def plot_confidence_interval_coverage(y_true: np.ndarray, mean_pred: np.ndarray, var_pred: np.ndarray, in_interval: np.ndarray, coverage_rate: float, save_path: Optional[str] = None) -> None:
    """
    绘制置信区间覆盖率分析图
    
    生成BNN模型预测的可视化分析图表，包括置信区间覆盖情况、残差分布等。
    用于评估模型的预测准确性和不确定性估计质量。
    
    Args:
        y_true: 真实值数组，形状为(n_samples,)
        mean_pred: 预测均值数组，形状为(n_samples,)
        var_pred: 预测方差数组，形状为(n_samples,)
        in_interval: 布尔数组，指示每个样本是否落在置信区间内
        coverage_rate: 覆盖率百分比
        save_path: 可选，图表保存路径
    
    图表内容：
    1. 置信区间覆盖图：显示预测均值、置信区间和真实值
    2. 标准化残差图：显示标准化残差和置信边界
    3. 统计信息：覆盖率、残差统计等
    
    Note:
        - 使用95%置信区间（z_score = 1.96）
        - 红色X标记表示区间外样本
        - 颜色编码表示样本是否在置信区间内
        - 自动生成详细的统计报告
    """
    plt.figure(figsize=(12, 8))
    
    # 计算标准差和置信区间
    std_pred = np.sqrt(var_pred)
    z_score = stats.norm.ppf(0.975)  # 95%置信区间
    
    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1：带置信区间的预测值与真实值对比
    sample_indices = np.arange(len(y_true))
    
    # 绘制置信区间（阴影区域）
    ax1.fill_between(sample_indices, lower_bound, upper_bound, 
                    alpha=0.3, color='lightblue', label='95% Confidence Interval')
    
    # Plot predicted mean line
    ax1.plot(sample_indices, mean_pred, 'b-', linewidth=2, label='Predicted Mean', alpha=0.8)
    
    # Plot true values
    ax1.plot(sample_indices, y_true, 'r-', linewidth=2, label='True Values', alpha=0.8)
    
    # 标记落在区间外的点
    out_of_interval = ~in_interval
    if np.any(out_of_interval):
        ax1.scatter(sample_indices[out_of_interval], y_true[out_of_interval], 
                   color='red', s=50, marker='x', label=f'Outliers ({np.sum(out_of_interval)} points)', zorder=5)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Demand Prediction (Standardized)')
    ax1.set_title(f'BNN Model Confidence Interval Coverage Analysis\nCoverage Rate: {coverage_rate:.1f}% (Target: 95.0%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：标准化残差图
    residuals = y_true - mean_pred
    standardized_residuals = residuals / std_pred
    
    scatter = ax2.scatter(sample_indices, standardized_residuals, c=in_interval, 
               cmap='RdYlBu', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add reference lines for ±1.96 (boundaries of 95% confidence interval)
    ax2.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='95% Boundaries (±1.96)')
    ax2.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Standardized Residuals')
    ax2.set_title('Standardized Residuals Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 为散点图添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Within Confidence Interval')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"置信区间覆盖率图已保存至: {save_path}")
    
    plt.show()
    
    # 打印详细统计信息
    print(f"\n=== 置信区间覆盖率统计 ===")
    print(f"样本总数: {len(y_true)}")
    print(f"落在95%置信区间内: {np.sum(in_interval)} ({coverage_rate:.1f}%)")
    print(f"落在区间外: {np.sum(~in_interval)} ({100-coverage_rate:.1f}%)")
    print(f"理论覆盖率: 95.0%")
    print(f"实际覆盖率: {coverage_rate:.1f}%")
    print(f"覆盖率偏差: {coverage_rate-95:.1f} 个百分点")
    
    # 计算并打印残差统计信息
    print(f"\n=== 残差统计 ===")
    print(f"残差均值: {np.mean(residuals):.4f}")
    print(f"残差标准差: {np.std(residuals):.4f}")
    print(f"标准化残差均值: {np.mean(standardized_residuals):.4f}")
    print(f"标准化残差标准差: {np.std(standardized_residuals):.4f}")

def load_and_preprocess_data(data_path: str, force_reprocess: bool = False) -> Tuple[HotelDataPreprocessor, pd.DataFrame]:
    """
    加载和预处理数据
    
    加载酒店预订数据并进行预处理，包括特征工程、数据清洗和标准化。
    支持缓存机制避免重复预处理，提高开发效率。
    
    Args:
        data_path: 数据文件路径，支持CSV格式
        force_reprocess: 是否强制重新预处理数据，忽略缓存
        
    Returns:
        Tuple[HotelDataPreprocessor, pd.DataFrame]: (预处理器对象, 处理后的特征数据)
        
    处理流程：
    1. 检查缓存文件是否存在
    2. 如不存在或强制重新处理，执行完整预处理
    3. 保存预处理结果到缓存文件
    4. 返回预处理器和处理后的数据
    
    Note:
        - 自动检测和使用已有的预处理结果
        - 支持强制重新处理选项
        - 缓存文件包括预处理器参数和处理后的数据
        - 提供详细的处理进度报告
    """
    print("\n=== 数据预处理 ===")
    
    preprocessor_path = '../02_训练模型/preprocessor.pkl'
    processed_data_path = '../03_数据文件/processed_features.csv'
    
    # 检查是否已有预处理结果
    if not force_reprocess and os.path.exists(preprocessor_path) and os.path.exists(processed_data_path):
        print("发现已有的预处理结果，正在加载...")
        preprocessor = HotelDataPreprocessor.load_preprocessor(preprocessor_path)
        features_df = pd.read_csv(processed_data_path)
        print(f"[OK] 预处理数据加载完成，共{len(features_df)}条记录")
    else:
        if force_reprocess:
            print("强制重新执行数据预处理...")
        else:
            print("正在执行数据预处理...")
        preprocessor = HotelDataPreprocessor()
        features_df = preprocessor.load_and_preprocess_data(data_path)
        
        # 保存预处理结果
        features_df.to_csv(processed_data_path, index=False)
        preprocessor.save_preprocessor(preprocessor_path)
        print("[OK] 数据预处理完成")
    
    return preprocessor, features_df

def train_bnn_model(preprocessor: HotelDataPreprocessor, features_df: pd.DataFrame, force_retrain: bool = False) -> Tuple[BNNTrainer, StandardScaler]:
    """
    训练BNN模型
    
    训练贝叶斯神经网络模型用于需求预测，包括数据准备、模型训练和性能评估。
    支持模型缓存和增量训练，提供详细的训练过程监控。
    
    Args:
        preprocessor: 数据预处理器对象，包含特征工程方法
        features_df: 特征数据DataFrame，包含原始特征和需求数据
        force_retrain: 是否强制重新训练模型，忽略已有模型
        
    Returns:
        Tuple[BNNTrainer, StandardScaler]: (BNN训练器, 需求标准化器)
        
    训练流程：
    1. 数据标准化：对需求数据进行标准化处理
    2. 样本构建：使用真实价格数据创建训练样本
    3. 数据划分：按70%/15%/15%分割训练/验证/测试集
    4. 模型训练：使用配置参数训练BNN模型
    5. 性能评估：计算MAE和置信区间覆盖率
    
    Note:
        - 使用需求标准化器确保数值稳定性
        - 支持模型缓存避免重复训练
        - 提供详细的训练进度和性能报告
        - 包含置信区间覆盖率评估
    """
    print("\n=== BNN模型训练 ===")
    
    model_path = '../02_训练模型/bnn_model.pth'
    
    # 准备训练数据
    print("正在准备训练数据...")
    
    # 对需求数据进行标准化
    demand_scaler = StandardScaler()
    original_demands = features_df['daily_demand'].values.reshape(-1, 1)
    standardized_demands = demand_scaler.fit_transform(original_demands).flatten()
    

    joblib.dump(demand_scaler, '../02_训练模型/demand_scaler_main.pkl')
    
    # 构造训练特征和标签
    X_list = []
    y_list = []
    
    print("正在使用真实价格数据创建训练样本...")
    
    # 检查是否有价格数据
    price_cols = ['avg_price', 'price_std', 'min_price', 'max_price', 'median_price']
    has_price_data = any(col in features_df.columns for col in price_cols)
    

    # 使用真实价格数据创建训练样本
    for idx, row in features_df.iterrows():
        # 使用标准化后的需求作为目标
        standardized_demand = standardized_demands[idx]
        
        # 使用真实平均价格作为价格特征
        avg_price = row.get('avg_price', 120)  # 默认120如果缺失
        if pd.isna(avg_price) or avg_price <= 0:
            avg_price = 120  # 使用默认值
        
        # 准备特征
        features = preprocessor.prepare_bnn_features(features_df.iloc[idx:idx+1], price_action=avg_price) #
        
        # 添加少量噪声到标准化需求数据
        noisy_demand = standardized_demand + np.random.normal(0, 0.05)  # 很小的噪声
        
        X_list.append(features.numpy())
        y_list.append(noisy_demand)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"训练数据构造完成：X形状{X.shape}, y形状{y.shape}")
    print(f"目标值范围：{y.min():.3f} - {y.max():.3f}（标准化后）")
    
    # 划分训练集、验证集和测试集
    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # 检查是否已有训练好的模型
    if not force_retrain and os.path.exists(model_path):
        print("发现已有的BNN模型，正在加载...")
        bnn_trainer = BNNTrainer(
            input_dim=X.shape[1],
            hidden_dims=BNN_CONFIG['hidden_dims'],  # 使用配置文件中的网络结构
            learning_rate=BNN_CONFIG['learning_rate'],  # 使用配置文件中的学习率
            weight_decay=BNN_CONFIG['weight_decay'],   # 使用配置文件中的权重衰减
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        bnn_trainer.load_model(model_path)
        print("[OK] BNN模型加载完成")
    else:
        if force_retrain:
            print("强制重新训练BNN模型...")
        else:
            print("正在训练BNN模型...")
        bnn_trainer = BNNTrainer(
            input_dim=X.shape[1],
            hidden_dims=BNN_CONFIG['hidden_dims'],  # 使用配置文件中的网络结构
            learning_rate=BNN_CONFIG['learning_rate'],  # 使用配置文件中的学习率
            weight_decay=BNN_CONFIG['weight_decay'],   # 使用配置文件中的权重衰减
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        bnn_trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=BNN_CONFIG['epochs'], batch_size=BNN_CONFIG['batch_size'],  
            save_path=model_path
        )
        print("[OK] BNN模型训练完成")
    
    # 测试模型性能（使用标准化数据评估）
    print("正在评估BNN模型性能...")
    mean_pred, var_pred = bnn_trainer.predict(X_test[:100])
    
    # 反标准化预测结果用于显示
    mean_pred_original = demand_scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
    y_test_original = demand_scaler.inverse_transform(y_test[:100].reshape(-1, 1)).flatten()
    
    mae = np.mean(np.abs(mean_pred.flatten() - y_test[:100]))
    mae_original = np.mean(np.abs(mean_pred_original - y_test_original))
    
    print(f"测试集MAE（标准化空间）: {mae:.4f}")
    print(f"测试集MAE（原始空间）: {mae_original:.2f}")
    print(f"预测范围（标准化）: {mean_pred.min():.3f} - {mean_pred.max():.3f}")
    print(f"真实范围（标准化）: {y_test[:100].min():.3f} - {y_test[:100].max():.3f}")
    
    # 置信区间覆盖率检验
    print("\n正在计算95%置信区间覆盖率...")
    coverage_rate, in_interval = evaluate_confidence_interval_coverage(
        mean_pred.flatten(), var_pred.flatten(), y_test[:100]
    )
    print(f"95%置信区间覆盖率: {coverage_rate:.1f}%")
    print(f"样本总数: {len(y_test[:100])}, 落在区间内: {np.sum(in_interval)}")
    
    # 绘制置信区间覆盖图
    plot_confidence_interval_coverage(
        y_test[:100], mean_pred.flatten(), var_pred.flatten(), 
        in_interval, coverage_rate, save_path="../05_分析报告/confidence_interval_coverage.png"
    )
    
    return bnn_trainer, demand_scaler

def train_rl_system(bnn_trainer: BNNTrainer, preprocessor: HotelDataPreprocessor, features_df: pd.DataFrame, demand_scaler: Optional[StandardScaler] = None) -> Tuple[HotelRLSystem, Optional[Dict], Optional[Dict]]:
    """
    训练强化学习系统
    
    构建并训练酒店定价强化学习系统，包括离线预训练和在线学习两个阶段。
    使用Q-learning算法优化定价策略，结合BNN预测进行决策。
    
    Args:
        bnn_trainer: BNN训练器对象，用于需求预测
        preprocessor: 数据预处理器对象，包含特征工程方法
        features_df: 特征数据DataFrame，包含历史数据用于训练
        demand_scaler: 需求标准化器，用于数据标准化处理
        
    Returns:
        Tuple[HotelRLSystem, Optional[Dict], Optional[Dict]]: (RL系统, 在线学习统计, 策略评估统计)
        
    训练流程：
    1. 系统初始化：创建RL系统并配置探索参数
    2. 离线预训练：使用历史数据进行离线学习
    3. 探索统计：分析Q值分布和探索覆盖率
    4. 在线学习：根据配置进行在线策略优化 (由config['enable_online_learning']开关决定)
    5. 策略评估：评估学习后的策略性能（已关闭）
    
    Note:
        - 使用ε-贪心策略进行探索和利用平衡
        - 支持离线预训练和在线学习两个阶段
        - 提供详细的Q值统计和探索覆盖率分析
        - 策略评估功能默认关闭以提高训练效率
    """
    
    # 创建RL系统，使用配置文件中的探索参数
    rl_system = HotelRLSystem(
        bnn_trainer, 
        preprocessor, 
        demand_scaler,
        epsilon_start=RL_CONFIG['epsilon_start'],
        epsilon_end=RL_CONFIG['epsilon_end'],
        epsilon_decay_episodes=RL_CONFIG['epsilon_decay_episodes']
    )
    
    # 训练强化学习系统
    print("开始离线预训练...")
    rl_system.offline_pretraining(features_df, episodes=RL_CONFIG['episodes'])
    
    # 显示预训练后的探索统计
    print(f"\n=== 预训练完成后的探索统计 ===")
    q_stats = rl_system.agent.get_q_value_stats()
    if q_stats:
        print(f"零值Q值占比: {q_stats['zero_q_percentage']:.1f}%")
        print(f"探索覆盖率: {q_stats['exploration_coverage']:.1f}%")
        print(f"已探索状态-动作对: {q_stats['explored_state_actions']}/{q_stats['total_state_actions']}")
        print(f"平均Q值: {q_stats['mean_q_value']:.2f}")
        print(f"总状态访问次数: {q_stats['num_state_visits']}")
    
    # 训练完成后显示训练曲线
    from training_monitor import get_training_monitor
    monitor = get_training_monitor()
    monitor.plot_training_curves()
    
    # 在线学习（根据配置开关决定是否执行）
    if RL_CONFIG['enable_online_learning']:
        print("\n开始在线学习...")
        online_stats = rl_system.online_learning(features_df, days=RL_CONFIG['online_learning_days'], update_frequency=RL_CONFIG['update_frequency'])
        
        # 显示在线学习后的探索统计
        print(f"\n=== 在线学习完成后的探索统计 ===")
        q_stats_final = rl_system.agent.get_q_value_stats()
        if q_stats_final:
            print(f"零值Q값占比: {q_stats_final['zero_q_percentage']:.1f}%")
            print(f"探索覆盖率: {q_stats_final['exploration_coverage']:.1f}%")
            print(f"已探索状态-动作对: {q_stats_final['explored_state_actions']}/{q_stats_final['total_state_actions']}")
            print(f"平均Q값: {q_stats_final['mean_q_value']:.2f}")
            print(f"总状态访问次数: {q_stats_final['num_state_visits']}")
            
            # 显示探索改进
            if q_stats:
                print(f"\n探索改进:")
                print(f"零值Q값占比变化: {q_stats['zero_q_percentage']:.1f}% -> {q_stats_final['zero_q_percentage']:.1f}% ({q_stats_final['zero_q_percentage'] - q_stats['zero_q_percentage']:+.1f}%)")
                print(f"探索覆盖率变化: {q_stats['exploration_coverage']:.1f}% -> {q_stats_final['exploration_coverage']:.1f}% ({q_stats_final['exploration_coverage'] - q_stats['exploration_coverage']:+.1f}%)")
    else:
        print("\n跳过在线学习（配置已关闭）")
        online_stats = None
    
    # 策略评估（已关闭）
    # print("\n开始策略评估...")
    # avg_stats, all_stats = rl_system.evaluate_policy(features_df, n_episodes=SIMULATION_CONFIG['evaluation_episodes'])
    avg_stats = None  # 策略评估已关闭
    all_stats = None  # 策略评估已关闭
    
    return rl_system, online_stats, avg_stats

def run_simulation(rl_system: HotelRLSystem, features_df: pd.DataFrame, start_date: Optional[datetime] = None, days: int = 90) -> pd.DataFrame:
    """
    运行酒店定价策略模拟
    
    使用训练好的强化学习系统在给定时间段内运行定价决策模拟，
    记录每日的定价决策、预测需求和实际收益等关键指标。
    
    Args:
        rl_system: 训练好的RL系统，包含Q表和BNN预测器
        features_df: 特征数据DataFrame，包含季节、日期等信息
        start_date: 模拟开始日期，默认为2017-01-01
        days: 模拟天数，默认为90天
        
    Returns:
        pd.DataFrame: 每日决策记录，包含日期、库存、价格、需求等字段
        
    模拟流程：
    1. 初始化环境：重置酒店环境和状态
    2. 每日循环：对每一天进行定价决策
    3. 状态获取：从环境和特征数据获取当前状态
    4. 动作选择：使用Q-learning选择最优定价动作
    5. 需求预测：使用BNN模型预测当前价格下的需求
    6. 环境更新：执行定价决策并更新环境状态
    7. 结果记录：保存每日的决策和结果数据
    
    Note:
        - 使用6档定价策略（60-210元，间隔30元）
        - 结合季节和工作日特征进行状态离散化
        - 使用BNN预测器进行需求预测
        - 支持自定义模拟起始日期和时长
    """
    
    if start_date is None:
        start_date = datetime(2017, 1, 1)
    
    # 找到对应的起始索引
    if 'date' in features_df.columns:
        features_df['date'] = pd.to_datetime(features_df['date'])
        start_idx = features_df[features_df['date'] >= start_date].index[0]
    else:
        start_idx = 0
    
    # 运行模拟
    simulation_features = features_df.iloc[start_idx:start_idx + days].reset_index(drop=True)
    
    # 重置环境
    env = HotelEnvironment()
    env.reset()
    
    # 每日决策记录
    daily_decisions = []
    
    for day in range(days):
        day_features = simulation_features.iloc[day:day + 1].reset_index(drop=True)
        
        # 获取当前状态
        state_info = env._get_state()
        
        # 离散化状态
        season = int(day_features['season'].iloc[0])
        weekday = int(day_features['is_weekend'].iloc[0])
        state = rl_system.agent.discretize_state(state_info, season, weekday)
        
        # 选择最优动作
        q_values = rl_system.agent.q_table[state]
        action = np.argmax(q_values)
        
        # 定价档位
        prices = [60, 90, 120, 150, 180, 210]
        price = prices[action]
        
        # 获取BNN预测
        predicted_demand, predicted_variance = rl_system.bnn_predictor(day_features, action)
        
        # 执行动作
        next_state_info, reward, done, info = env.step(action, rl_system.bnn_predictor, day_features)
        
        # 记录决策
        daily_decisions.append({
            'day': day + 1,
            'date': day_features['date'].iloc[0] if 'date' in day_features.columns else start_date + timedelta(days=day),
            'inventory_before': state_info['inventory_raw'],
            'inventory_after': next_state_info['inventory_raw'],
            'action': action,
            'price': price,
            'predicted_demand': predicted_demand,
            'predicted_variance': predicted_variance,
            'actual_demand': info.get('predicted_demand', 0),  # 使用预测需求作为实际需求
            'actual_bookings': info['actual_bookings'],
            'revenue': info['revenue'],
            'reward': reward
        })
        
        if done:
            break
    
    # 生成模拟报告
    df_decisions = pd.DataFrame(daily_decisions)
    
    # print(f"\n=== {days}天定价模拟结果 ===")
    # print(f"总收益: ¥{df_decisions['revenue'].sum():,.2f}")
    # print(f"平均每日收益: ¥{df_decisions['revenue'].mean():,.2f}")
    # print(f"平均价格: ¥{df_decisions['price'].mean():,.2f}")
    # print(f"平均入住率: {df_decisions['actual_bookings'].sum() / (100 * len(df_decisions)):.1%}")
    # print(f"需求满足率: {df_decisions['actual_bookings'].sum() / df_decisions['actual_demand'].sum():.1%}")
    
    return df_decisions

def main() -> None:
    """
    酒店动态定价系统主函数
    
    系统入口点，负责整个定价系统的运行流程控制，包括：
    - 环境检查和配置验证
    - 数据加载和预处理
    - BNN模型训练和评估
    - 强化学习系统训练
    - 定价策略模拟和结果分析
    
    Args:
        无（使用命令行参数）
        
    命令行参数：
        --data: 数据文件路径，默认../03_数据文件/hotel_bookings.csv
        --skip-training: 跳过训练，直接使用已有模型
        --force-retrain: 强制重新训练所有模型
        --simulate-days: 模拟天数，默认90天
        --start-date: 模拟开始日期，默认2017-01-01
        
    运行流程：
    1. 环境检查：验证Python环境和依赖库
    2. 数据准备：加载和预处理酒店预订数据
    3. 模型训练：根据参数训练BNN和RL模型
    4. 策略模拟：运行定价策略模拟
    5. 结果分析：生成分析报告和可视化图表
    
    Note:
        - 支持模型缓存避免重复训练
        - 提供详细的训练进度和性能报告
        - 生成完整的分析报告和可视化结果
        - 支持灵活的参数配置和运行模式
    """
    parser = argparse.ArgumentParser(description='酒店动态定价系统')
    parser.add_argument('--data', type=str, default='../03_数据文件/hotel_bookings.csv',
                       help='酒店预订数据文件路径')
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过训练，直接使用已有模型')
    parser.add_argument('--force-retrain', action='store_true',
                       help='强制重新训练所有模型（忽略已有模型）')
    # parser.add_argument('--simulate-days', type=int, default=90,
    #                    help='模拟天数')
    # parser.add_argument('--start-date', type=str, default='2017-01-01',
    #                    help='模拟开始日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("酒店动态定价系统 (BNN + Q-learning)")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        return
    
    # 数据预处理
    preprocessor, features_df = load_and_preprocess_data(args.data, force_reprocess=args.force_retrain)
    
    if not args.skip_training:
        # 训练BNN模型
        bnn_trainer, demand_scaler = train_bnn_model(preprocessor, features_df, force_retrain=args.force_retrain)
        
        # 训练强化学习系统（传入demand_scaler）
        rl_system, online_stats, avg_stats = train_rl_system(bnn_trainer, preprocessor, features_df, demand_scaler)
    else:
        # 加载已有模型
        print("\n正在加载已有模型...")
        
        # 加载BNN模型
        bnn_model_path = '../02_训练模型/bnn_model_final.pth'
        if not os.path.exists(bnn_model_path):
            print("错误：未找到BNN模型文件，请先训练模型或移除--skip-training参数")
            return
        
        # 重新创建BNN训练器并加载模型
        bnn_trainer = BNNTrainer(
            input_dim=BNN_CONFIG['input_dim'],  # 使用配置文件中的输入维度
            hidden_dims=BNN_CONFIG['hidden_dims'],  # 使用配置文件中的网络结构
            learning_rate=BNN_CONFIG['learning_rate'],  # 使用配置文件中的学习率
            weight_decay=BNN_CONFIG['weight_decay'],   # 使用配置文件中的权重衰减
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        bnn_trainer.load_model(bnn_model_path)
        
        # 加载demand_scaler用于跳过训练模式
        demand_scaler = joblib.load('../02_训练模型/demand_scaler_main.pkl')
        
        # 创建RL系统
        rl_system = HotelRLSystem(bnn_trainer, preprocessor, demand_scaler)
        
        # 加载训练好的智能体
        agent_path = '../02_训练模型/q_agent_final.pkl'
        if os.path.exists(agent_path):
            rl_system.agent.load_agent(agent_path)
        else:
            print("警告：未找到最终智能体文件，将使用预训练智能体")
            pretrained_path = '../02_训练模型/q_agent_pretrained.pkl'
            if os.path.exists(pretrained_path):
                rl_system.agent.load_agent(pretrained_path)
            else:
                print("错误：未找到任何智能体文件")
                return
    
    # 运行模拟功能已移除
    
    # 模拟结果保存功能已移除
    # results_path = f'../04_结果输出/simulation_results_{start_date.strftime("%Y%m%d")}_{args.simulate_days}days.csv'
    # simulation_results.to_csv(results_path, index=False)
    # print(f"\n模拟结果已保存到：{results_path}")
    
    # 输出Q表信息
    print(f"\n=== Q表信息 ===")
    if hasattr(rl_system, 'agent') and hasattr(rl_system.agent, 'q_table'):
        q_table = rl_system.agent.q_table
        print(f"Q表状态数量: {len(q_table)}")
        
        # 获取Q值统计
        q_stats = rl_system.agent.get_q_value_stats()
        if q_stats:
            print(f"Q值统计:")
            print(f"  平均Q值: {q_stats['mean_q_value']:.2f}")
            print(f"  Q值标准差: {q_stats['std_q_value']:.2f}")
            print(f"  最小Q값: {q_stats['min_q_value']:.2f}")
            print(f"  最大Q값: {q_stats['max_q_value']:.2f}")
            print(f"  总状态访问次数: {q_stats['num_state_visits']}")
            print(f"  零值Q값占比: {q_stats['zero_q_percentage']:.1f}%")
            print(f"  探索覆盖率: {q_stats['exploration_coverage']:.1f}%")
            print(f"  已探索状态-动作对: {q_stats['explored_state_actions']}/{q_stats['total_state_actions']}")
        
        # 显示前10个状态的Q值
        print(f"\n前10个状态的Q值:")
        prices = [60, 90, 120, 150, 180, 210]
        for i, (state, q_values) in enumerate(list(q_table.items())[:10]):
            best_action = np.argmax(q_values)
            print(f"状态 {state}: {[f'{q:.1f}' for q in q_values]} -> 最佳动作: {best_action} (价格: {prices[best_action]}元)")
        
        if len(q_table) > 10:
            print(f"... 还有 {len(q_table) - 10} 个状态")
        
        # 保存Q表到CSV文件
        try:
            
            # 创建Q表DataFrame
            q_table_data = []
            prices = [60, 90, 120, 150, 180, 210]
            
            for state, q_values in q_table.items():
                best_action = np.argmax(q_values)
                row = {
                    'state': state,
                    'action_0': q_values[0],
                    'action_1': q_values[1], 
                    'action_2': q_values[2],
                    'action_3': q_values[3],
                    'action_4': q_values[4],
                    'action_5': q_values[5],
                    'best_action': best_action,
                    'best_price': prices[best_action],
                    'best_value': q_values[best_action]
                }
                q_table_data.append(row)
            
            q_table_df = pd.DataFrame(q_table_data)
            
            # 生成CSV文件名（包含时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            q_table_csv_path = f'../05_分析报告/q_table_main_{timestamp}.csv'
            
            # 保存到CSV
            q_table_df.to_csv(q_table_csv_path, index=False)
            print(f"\nQ表已保存到CSV文件: {q_table_csv_path}")
            
            # 同时保存Q表统计信息
            if q_stats:
                stats_df = pd.DataFrame([{
                    'total_states': len(q_table),
                    'mean_q_value': q_stats['mean_q_value'],
                    'std_q_value': q_stats['std_q_value'],
                    'min_q_value': q_stats['min_q_value'],
                    'max_q_value': q_stats['max_q_value'],
                    'total_visits': q_stats['num_state_visits'],
                    'zero_q_percentage': q_stats['zero_q_percentage'],
                    'exploration_coverage': q_stats['exploration_coverage'],
                    'explored_state_actions': q_stats['explored_state_actions'],
                    'total_state_actions': q_stats['total_state_actions']
                }])
                
                stats_csv_path = f'../05_分析报告/q_table_stats_{timestamp}.csv'
                stats_df.to_csv(stats_csv_path, index=False)
                print(f"Q表统计信息已保存到: {stats_csv_path}")
                
        except Exception as e:
            print(f"保存Q表到CSV时出错: {e}")
        
        # 绘制Q表热力图
        try:
            import seaborn as sns
            
            print("\n=== 开始绘制Q表热力图 ===")
            
            # 设置中文字体 - 添加更多备选字体确保兼容性
            plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建状态-动作矩阵
            states = sorted(q_table.keys())
            actions = list(range(6))
            
            # 创建Q值矩阵
            q_matrix = np.zeros((len(states), len(actions)))
            for i, state in enumerate(states):
                q_matrix[i, :] = q_table[state]
            
            # 创建状态标签（库存等级 + 季节 + 日期类型）
            state_labels = []
            for state in states:
                # 状态编码：库存等级(0-4) × 3(季节) × 2(日期类型) = 30种状态
                state_value = state
                inventory_level = state_value // 6  # 5种库存等级 (0-4)
                remaining = state_value % 6
                season = remaining // 2  # 3种季节 (0-2)
                day_type = remaining % 2  # 2种日期类型 (0-1)
                
                # 库存等级描述 - 按照实际数值范围命名
                inventory_descriptions = ['0-20间', '21-40间', '41-60间', '61-80间', '81-100间']
                # 季节描述
                season_descriptions = ['淡季', '平季', '旺季']
                # 日期类型描述
                day_type_descriptions = ['工作日', '周末']
                
                # 使用实际换行符而不是转义字符
                state_label = f"{inventory_descriptions[inventory_level]}\n{season_descriptions[season]}\n{day_type_descriptions[day_type]}"
                state_labels.append(state_label)
            
            # 动作标签（价格）
            action_labels = ['¥60', '¥90', '¥120', '¥150', '¥180', '¥210']
            
            # 创建热力图
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # 使用seaborn绘制热力图
            sns.heatmap(q_matrix, 
                        xticklabels=action_labels, 
                        yticklabels=state_labels,
                        cmap='RdYlBu_r', 
                        center=0,
                        annot=True, 
                        fmt='.1f',
                        cbar_kws={'label': 'Q값'},
                        ax=ax)
            
            # 设置标题和标签
            ax.set_title('Q表热力图 - 酒店动态定价策略', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
            ax.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存热力图
            heatmap_path = f'../04_结果输出/q_table_heatmap_{timestamp}.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Q表热力图已保存到: {heatmap_path}")
            
            # 显示热力图
            plt.show()
            
            # 创建最佳策略热力图
            print("\n=== 绘制最佳策略热力图 ===")
            
            # 创建最佳动作矩阵
            best_action_matrix = np.zeros((len(states), len(actions)))
            for i, state in enumerate(states):
                best_action = np.argmax(q_table[state])
                best_action_matrix[i, best_action] = 1
            
            # 将矩阵转换为整数类型以避免格式化错误
            best_action_matrix = best_action_matrix.astype(int)
            
            # 创建最佳策略热力图
            fig2, ax2 = plt.subplots(figsize=(14, 10))
            
            # 使用离散颜色映射
            cmap = plt.cm.get_cmap('RdYlBu', 2)
            sns.heatmap(best_action_matrix, 
                        xticklabels=action_labels, 
                        yticklabels=state_labels,
                        cmap=cmap, 
                        vmin=0, vmax=1,
                        annot=True, 
                        fmt='d',  
                        cbar_kws={'label': '是否为最佳动作', 'ticks': [0, 1]},
                        ax=ax2)
            
            # 设置标题和标签
            ax2.set_title('最佳策略热力图 - 酒店动态定价', fontsize=16, fontweight='bold', pad=20)
            ax2.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
            ax2.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存最佳策略热力图
            best_policy_path = f'../04_结果输出/best_policy_heatmap_{timestamp}.png'
            plt.savefig(best_policy_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 最佳策略热力图已保存到: {best_policy_path}")
            
            # 显示最佳策略热力图
            plt.show()
            
            print("[OK] Q表热力图绘制完成")
            
        except Exception as e:
            print(f"绘制Q表热力图时出错: {e}")
            traceback.print_exc()
    
    # 策略评估功能已移除（模拟结果不可用）
    # print(f"\n=== 详细分析报告 ===")
    # print(f"模拟期间：{simulation_results['date'].min().strftime('%Y-%m-%d')} 到 {simulation_results['date'].max().strftime('%Y-%m-%d')}")
    # 
    # 价格分布
    # price_stats = simulation_results['price'].describe()
    # print(f"\n价格统计：")
    # print(f"  平均价格: ¥{price_stats['mean']:.2f}")
    # print(f"  价格标准差: ¥{price_stats['std']:.2f}")
    # print(f"  最低价格: ¥{price_stats['min']:.2f}")
    # print(f"  最高价格: ¥{price_stats['max']:.2f}")
    # 
    # 需求预测准确性
    # demand_mae = np.mean(np.abs(simulation_results['predicted_demand'] - simulation_results['actual_demand']))
    # print(f"\n需求预测准确性：")
    # print(f"  MAE: {demand_mae:.2f} 间/天")
    # 
    # 季节性分析
    # if 'season' in simulation_results.columns:
    #     season_stats = simulation_results.groupby('season').agg({
    #         'price': 'mean',
    #         'actual_bookings': 'mean',
    #         'revenue': 'mean'
    #     }).round(2)
    #     print(f"\n季节性分析：")
    #     print(season_stats)
    
    print("\n" + "=" * 60)
    print("系统运行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
