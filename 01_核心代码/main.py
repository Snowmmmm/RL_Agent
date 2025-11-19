#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店动态定价系统 - 主程序
基于NGBoost和Q-learning强化学习
"""

# 标准库导入
import argparse
import os
import pickle
import sys
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

# 第三方库导入
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# 本地模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ngboost_model import NGBoostTrainer
from config import BQL_CONFIG, RL_CONFIG, SIMULATION_CONFIG, DATA_SPLIT_CONFIG, OPTUNA_CONFIG, BEST_NGBOOST_PARAMS
        
from data_preprocessing import HotelDataPreprocessor
from rl_system import HotelEnvironment, HotelRLSystem, QLearningAgent

# 配置警告过滤器
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
OPTUNA_AVAILABLE = True

# 添加随机因子控制 - 取消注释以启用固定随机种子
# import random
# random.seed(42)
# np.random.seed(42)

# 导入随机因子配置（自动设置随机模式）
from random_factor_config import current_random_config
from config import RANDOM_CONFIG
print(f"当前随机因子配置: {current_random_config['current_status']}")

# 确保所有随机种子设置与random_factor_config一致
if current_random_config['random_mode'] == 'fixed':
    # 固定模式：使用配置中的种子
    global_random_seed = RANDOM_CONFIG['fixed_seed']
    print(f"使用固定随机种子: {global_random_seed}")
else:
    # 随机模式：使用None作为种子
    global_random_seed = None
    print("使用随机模式，不设置固定种子")

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
    - Optuna可用性（用于超参数搜索）
    - 系统资源和权限
    
    Note:
        - 自动检测GPU设备并报告状态
        - 提供详细的错误信息帮助环境配置
        - 支持CPU-only模式运行
    """
    print("=== 环境检查 ===")
    
    # 检查Optuna
    if OPTUNA_AVAILABLE:
        print("[OK] Optuna已安装，支持超参数搜索")
    else:
        print("[WARN] Optuna未安装，超参数搜索功能不可用")
    
    # 检查必要的库
    # 验证关键依赖库是否可用
    _ = pd.DataFrame  # 验证pandas
    _ = np.array     # 验证numpy
    print("[OK] 所有依赖库已安装")
    
    return True

def evaluate_confidence_interval_coverage(mean_pred: np.ndarray, var_pred: np.ndarray, y_true: np.ndarray, confidence_level: float = 0.95, distribution: str = 'normal') -> Tuple[float, np.ndarray]:
    """
    评估置信区间覆盖率
    
    计算NGBoost预测的置信区间覆盖率，评估模型的不确定性估计质量。
    根据指定的分布类型计算置信区间，并统计真实值落在区间内的比例。
    
    Args:
        mean_pred: 预测均值，形状为(n_samples,)
        var_pred: 预测方差，形状为(n_samples,)
        y_true: 真实值，形状为(n_samples,)
        confidence_level: 置信水平，默认95%
        distribution: 分布类型 ('normal' 或 'lognormal')
    
    Returns:
        Tuple[float, np.ndarray]: (覆盖率百分比, 布尔数组指示每个样本是否在区间内)
        
    计算过程：
    1. 根据分布类型计算置信区间
    2. 统计覆盖率：mean(y_true ∈ [lower, upper])
    
    Note:
        - 支持正态分布和对数正态分布
        - 覆盖率应接近置信水平（如95%）
        - 可用于评估NGBoost的不确定性校准质量
    """
    # 计算标准差
    std_pred = np.sqrt(var_pred)
    
    # 根据分布类型计算置信区间
    alpha = 1 - confidence_level
    
    if distribution == 'normal':
        # 正态分布：使用Z分数
        z_score = stats.norm.ppf(1 - alpha/2)
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
    elif distribution == 'lognormal':
        # 对数正态分布：需要考虑对数变换
        # 如果mean_pred和std_pred是对数正态分布的参数，需要转换为对数空间的正态分布参数
        # 对数正态分布的置信区间计算较复杂，这里使用近似方法
        
        # 确保均值和标准差为正（对数正态分布要求）
        mean_pred = np.maximum(mean_pred, 1e-8)
        std_pred = np.maximum(std_pred, 1e-8)
        
        # 计算对数空间的参数
        # 对数正态分布：如果X~LogNormal(μ, σ)，则log(X)~Normal(μ, σ)
        # 但这里mean_pred和std_pred是X的均值和标准差，需要转换
        
        # 使用近似：假设预测的是对数正态分布的参数
        sigma_squared = np.log(1 + (std_pred / mean_pred) ** 2)
        sigma = np.sqrt(sigma_squared)
        mu = np.log(mean_pred) - sigma_squared / 2
        
        # 计算对数空间的置信区间
        z_score = stats.norm.ppf(1 - alpha/2)
        log_lower = mu - z_score * sigma
        log_upper = mu + z_score * sigma
        
        # 转换回原始空间
        lower_bound = np.exp(log_lower)
        upper_bound = np.exp(log_upper)
    else:
        # 默认使用正态分布
        z_score = stats.norm.ppf(1 - alpha/2)
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
    
    # 检查真实值是否在置信区间内
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage_rate = float(np.mean(in_interval)) * 100
    
    return coverage_rate, in_interval

def plot_confidence_interval_coverage(y_true: np.ndarray, mean_pred: np.ndarray, var_pred: np.ndarray, in_interval: np.ndarray, coverage_rate: float, save_path: Optional[str] = None, feature_names: Optional[List[str]] = None, show_plot: bool = True) -> None:
    """
    Plot confidence interval coverage analysis chart
    
    Generate visualization analysis for NGBoost predictions, including confidence interval coverage and residual distribution.
    Used to evaluate model prediction accuracy and uncertainty estimation quality.
    
    Args:
        y_true: True values array, shape (n_samples,)
        mean_pred: Predicted mean values array, shape (n_samples,)
        var_pred: Predicted variance array, shape (n_samples,)
        in_interval: Boolean array indicating whether each sample falls within confidence interval
        coverage_rate: Coverage rate percentage
        save_path: Optional, chart save path
        feature_names: Optional, list of feature names used in the model
    
    Chart contents:
    1. Confidence interval coverage plot: shows predicted mean, confidence intervals and true values
    2. Standardized residuals plot: shows standardized residuals and confidence boundaries
    3. Statistics: coverage rate, residual statistics, etc.
    
    Note:
        - Uses 95% confidence interval (z_score = 1.96)
        - Red X markers indicate out-of-interval samples
        - Color coding indicates whether samples are within confidence interval
        - Automatically generates detailed statistical report
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
    
    # Prepare feature information for title
    if save_path:
        # 从文件路径提取模型信息并转换中文为英文
        model_filename = save_path.split("/")[-1].replace("confidence_interval_coverage_", "").replace(".png", "")
        # 转换中文客户类型为英文
        if "线上用户" in model_filename:
            model_info = model_filename.replace("线上用户", "Online")
        elif "线下用户" in model_filename:
            model_info = model_filename.replace("线下用户", "Offline")
        else:
            model_info = model_filename
        # 转换需求类型
        model_info = model_info.replace("_booked", "_Booked").replace("_actual", "_Actual")
    else:
        model_info = "NGBoost Model"
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Demand Prediction (Standardized)')
    ax1.set_title(f'NGBoost Confidence Interval Coverage Analysis\nCoverage Rate: {coverage_rate:.1f}% (Target: 95.0%)\nModel: {model_info}')
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
    ax2.set_title('Standardized Residuals Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('In 95% CI')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Out of CI', 'In CI'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"置信区间覆盖率分析图已保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


class NGBoostHyperparameterOptimizer:
    """
    NGBoost超参数优化器
    
    使用Optuna进行NGBoost超参数搜索，支持双目标优化：
    1. 最小化MAE（平均绝对误差）
    2. 最大化置信区间覆盖率（接近95%）
    
    特性：
    - 支持多目标优化
    - 自动保存搜索结果
    - 可视化优化过程
    - 早停机制
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化超参数优化器
        
        Args:
            config: Optuna配置字典，默认为OPTUNA_CONFIG
        """
        # 设置matplotlib后端为非交互式，避免多线程冲突
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        self.config = config or OPTUNA_CONFIG
        self.study = None
        self.best_params = None
        self.trial_results = []
        
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna未安装，无法使用超参数优化功能。请运行: pip install optuna")
    
    def _create_objective_function(self, X_train, y_train, X_val, y_val, demand_scaler, customer_type, demand_type):
        """
        创建目标函数（基于概率评估指标）
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            demand_scaler: 需求标准化器
            customer_type: 客户类型（线上用户/线下用户）
            demand_type: 需求类型（booked/actual）
            
        Returns:
            目标函数，返回负对数似然（最小化）
        """
        def objective(trial):
            # 从trial中采样超参数
            params = self._sample_hyperparameters(trial)
            
            # 创建NGBoost训练器
            ngboost_trainer = NGBoostTrainer(
                distribution=params['distribution'],
                score='logscore',  # 使用对数似然作为评分规则
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                colsample_bytree=params['colsample_bytree'],
                verbose=False,
                random_state=global_random_seed
            )
            
            # 训练模型
            ngboost_trainer.train(
                X_train, y_train, X_val, y_val,
                early_stopping_rounds=params.get('early_stopping_rounds', 30)
            )
            
            # 在验证集上评估（使用完整的概率评估指标）
            evaluation_results = ngboost_trainer.evaluate(X_val, y_val)
            
            # 主要优化目标：最大化对数似然（转换为最小化问题）
            log_likelihood = evaluation_results.get('log_likelihood', -np.inf)
            primary_objective = -log_likelihood  # 负对数似然（最小化）
            
            # 辅助监测指标
            mae = evaluation_results.get('mae', np.inf)
            crps = evaluation_results.get('crps', np.inf)
            pit_mean = evaluation_results.get('pit_mean', np.nan)
            pit_ks_statistic = evaluation_results.get('pit_ks_statistic', np.nan)
            
            # 置信区间覆盖率（作为约束条件）
            coverage_rate = evaluation_results.get('coverage_95', 0.0)
            coverage_error = abs(coverage_rate - 95.0)
            
            # 计算综合目标值（对数似然为主，其他指标为辅助）
            # 使用对数似然作为主要指标，其他指标作为正则化项
            log_likelihood_weight = self.config['objective_weights'].get('log_likelihood_weight', 1.0)
            mae_weight = self.config['objective_weights'].get('mae_weight', 0.1)  # MAE作为辅助指标
            coverage_weight = self.config['objective_weights'].get('coverage_weight', 0.2)
            crps_weight = self.config['objective_weights'].get('crps_weight', 0.1)
            pit_weight = self.config['objective_weights'].get('pit_weight', 0.05)
            
            # 标准化各项指标
            mae_normalized = mae / np.std(y_val) if np.std(y_val) > 0 else mae
            crps_normalized = crps / np.std(y_val) if np.std(y_val) > 0 else crps
            
            # 综合目标：最小化负对数似然 + 正则化项
            combined_objective = (
                log_likelihood_weight * primary_objective +
                mae_weight * mae_normalized +
                coverage_weight * (coverage_error / 95.0) +
                crps_weight * crps_normalized +
                pit_weight * abs(pit_mean - 0.5)  # PIT均值应接近0.5
            )
            
            # 记录trial结果（包含完整的概率评估指标）
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'log_likelihood': log_likelihood,
                'primary_objective': primary_objective,
                'mae': mae,
                'crps': crps,
                'pit_mean': pit_mean,
                'pit_ks_statistic': pit_ks_statistic,
                'coverage_rate': coverage_rate,
                'coverage_error': coverage_error,
                'mae_normalized': mae_normalized,
                'crps_normalized': crps_normalized,
                'combined_objective': combined_objective
            }
            self.trial_results.append(trial_result)
            
            # 打印当前trial结果（突出概率评估指标）
            print(f"Trial {trial.number}: LogLikelihood={log_likelihood:.4f}, "
                  f"MAE={mae:.4f}, CRPS={crps:.4f}, Coverage={coverage_rate:.1f}%, "
                  f"PIT_mean={pit_mean:.3f}, Combined={combined_objective:.4f}")
            
            return combined_objective
        
        return objective
    
    def _sample_hyperparameters(self, trial):
        """
        从配置中采样超参数
        
        Args:
            trial: Optuna trial对象
            
        Returns:
            超参数字典
        """
        params = {}
        param_space = self.config['param_space']
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 0.01)
                    )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def optimize(self, X_train, y_train, X_val, y_val, demand_scaler, customer_type, demand_type):
        """
        运行超参数优化
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            demand_scaler: 需求标准化器
            customer_type: 客户类型
            demand_type: 需求类型
            
        Returns:
            最佳超参数
        """
        print(f"\n=== 开始NGBoost超参数优化 ===")
        print(f"客户类型: {customer_type}, 需求类型: {demand_type}")
        print(f"搜索试验次数: {self.config['n_trials']}")
        print(f"MAE权重: {self.config['objective_weights']['mae_weight']}, "
              f"覆盖率权重: {self.config['objective_weights']['coverage_weight']}")
        
        # 创建目标函数
        objective = self._create_objective_function(
            X_train, y_train, X_val, y_val, demand_scaler, customer_type, demand_type
        )
        
        # 创建study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=global_random_seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # 运行优化
        self.study.optimize(
            objective,
            n_trials=self.config['n_trials'],
            timeout=self.config['timeout'],
            n_jobs=self.config['n_jobs'],
            show_progress_bar=True
        )
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        
        print(f"\n=== 超参数优化完成 ===")
        print(f"最佳试验: {self.study.best_trial.number}")
        print(f"最佳综合目标值: {self.study.best_value:.4f}")
        print(f"最佳超参数: {self.best_params}")
        
        # 保存结果
        self._save_results(customer_type, demand_type)
        
        # 可视化结果
        if self.config['plot_results']:
            self._plot_results(customer_type, demand_type)
        
        return self.best_params
    
    def _save_results(self, customer_type, demand_type):
        """
        保存优化结果（为每个客户类型和需求类型组合分别保存）
        
        Args:
            customer_type: 客户类型
            demand_type: 需求类型
        """
        if not self.config['save_results']:
            return
        
        # 创建模型特定的文件名
        model_key = f"{customer_type}_{demand_type}"
        
        # 创建保存目录
        os.makedirs(os.path.dirname(self.config['results_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['study_path']), exist_ok=True)
        
        # 保存完整结果（模型特定）
        results = {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'best_trial_number': self.study.best_trial.number,
            'trial_results': self.trial_results,
            'study_summary': {
                'n_trials': len(self.study.trials),
                'best_trial': self.study.best_trial.number,
                'datetime': pd.Timestamp.now().isoformat(),
                'customer_type': customer_type,
                'demand_type': demand_type,
                'model_key': model_key
            }
        }
        
        # 模型特定的文件路径
        results_path_specific = self.config['results_path'].replace('.pkl', f'_{model_key}.pkl')
        best_params_path_specific = self.config['results_path'].replace('.pkl', f'_{model_key}_best_params.json')
        study_path_specific = self.config['study_path'].replace('.pkl', f'_{model_key}.pkl')
        
        # 保存完整结果（模型特定）
        with open(results_path_specific, 'wb') as f:
            pickle.dump(results, f)
        
        # 保存最佳参数（JSON格式，模型特定）
        import json
        # 添加score参数，因为NGBoostTrainer需要这个参数
        best_params_with_score = self.best_params.copy()
        if 'score' not in best_params_with_score:
            best_params_with_score['score'] = 'logscore'  # 默认使用logscore
            
        with open(best_params_path_specific, 'w', encoding='utf-8') as f:
            json.dump(best_params_with_score, f, indent=2, ensure_ascii=False)
        
        # 保存study对象（模型特定）
        with open(study_path_specific, 'wb') as f:
            pickle.dump(self.study, f)
        
        # 同时更新全局最佳参数文件（合并所有模型的最佳参数）
        self._update_global_best_params(model_key, self.best_params)
        
        print(f"优化结果已保存（{model_key}）:")
        print(f"  完整结果: {results_path_specific}")
        print(f"  最佳参数: {best_params_path_specific}")
        print(f"  Study对象: {study_path_specific}")
    
    def _update_global_best_params(self, model_key, best_params):
        """
        更新config.py中的全局最佳参数
        
        Args:
            model_key: 模型键（customer_type_demand_type）
            best_params: 该模型的最佳超参数
        """
        # 读取config.py文件内容
        config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 解析BEST_NGBOOST_PARAMS部分
        import re
        # 找到BEST_NGBOOST_PARAMS的开始和结束位置
        start_pattern = r'BEST_NGBOOST_PARAMS\s*=\s*\{'
        end_pattern = r'\}'
        
        match = re.search(start_pattern, config_content)
        if not match:
            print("警告：无法在config.py中找到BEST_NGBOOST_PARAMS定义")
            return
            
        start_pos = match.start()
        
        # 找到匹配的结束大括号
        brace_count = 0
        end_pos = start_pos
        for i, char in enumerate(config_content[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if brace_count != 0:
            print("警告：无法正确解析config.py中的BEST_NGBOOST_PARAMS")
            return
        
        # 提取BEST_NGBOOST_PARAMS部分
        best_params_section = config_content[start_pos:end_pos]
        
        # 使用ast.literal_eval安全地解析字典
        import ast
        try:
            best_params_dict = ast.literal_eval(best_params_section.split('=', 1)[1].strip())
        except (SyntaxError, ValueError) as e:
            print(f"警告：无法解析config.py中的BEST_NGBOOST_PARAMS: {e}")
            return
        
        # 更新当前模型的最佳参数
        # 添加score参数，因为NGBoostTrainer需要这个参数
        best_params_with_score = best_params.copy()
        if 'score' not in best_params_with_score:
            best_params_with_score['score'] = 'logscore'  # 默认使用logscore
            
        best_params_dict[model_key] = best_params_with_score
        
        # 生成新的BEST_NGBOOST_PARAMS代码
        new_best_params_code = "BEST_NGBOOST_PARAMS = {\n"
        for key, params in best_params_dict.items():
            new_best_params_code += f"    '{key}': {{\n"
            for param_name, param_value in params.items():
                if isinstance(param_value, str):
                    new_best_params_code += f"        '{param_name}': '{param_value}',\n"
                else:
                    new_best_params_code += f"        '{param_name}': {param_value},\n"
            new_best_params_code += "    },\n"
        new_best_params_code += "}"
        
        # 替换config.py中的BEST_NGBOOST_PARAMS部分
        new_config_content = config_content[:start_pos] + new_best_params_code + config_content[end_pos:]
        
        # 写回config.py文件
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_config_content)
        
        print(f"已更新config.py中的BEST_NGBOOST_PARAMS，包含模型: {list(best_params_dict.keys())}")
    
    def _plot_results(self, customer_type, demand_type):
        """
        绘制优化结果图表（已注释掉，跳过绘图以节省时间和避免潜在问题）
        
        Args:
            customer_type: 客户类型
            demand_type: 需求类型
        """
        if not OPTUNA_AVAILABLE:
            return
        
        # 绘图功能已注释掉，跳过图表生成
        print(f"跳过超参数优化结果图表绘制（{customer_type}_{demand_type}）")
        return
        
        # try:
        #     # 设置matplotlib后端为非交互式，避免多线程冲突
        #     import matplotlib
        #     matplotlib.use('Agg')  # 使用非交互式后端
        #     import matplotlib.pyplot as plt
        #     
        #     # 创建保存目录
        #     os.makedirs(self.config['plots_path'], exist_ok=True)
        #     
        #     # 生成文件名
        #     base_filename = f"optuna_{customer_type}_{demand_type}"
        #     
        #     # 绘制优化历史
        #     fig1 = plot_optimization_history(self.study)
        #     if hasattr(fig1, 'write_image'):
        #         fig1.write_image(os.path.join(self.config['plots_path'], f"{base_filename}_history.png"))
        #     else:
        #         fig1.savefig(os.path.join(self.config['plots_path'], f"{base_filename}_history.png"), 
        #                     dpi=300, bbox_inches='tight')
        #     plt.close(fig1) if hasattr(fig1, 'savefig') else None
        #     
        #     # 绘制参数重要性
        #     fig2 = plot_param_importances(self.study)
        #     if hasattr(fig2, 'write_image'):
        #         fig2.write_image(os.path.join(self.config['plots_path'], f"{base_filename}_importance.png"))
        #     else:
        #         fig2.savefig(os.path.join(self.config['plots_path'], f"{base_filename}_importance.png"), 
        #                     dpi=300, bbox_inches='tight')
        #     plt.close(fig2) if hasattr(fig2, 'savefig') else None
        #     
        #     print(f"优化结果图表已保存到: {self.config['plots_path']}")
        # except ImportError as e:
        #     print(f"警告：无法绘制优化图表 - {e}")
        #     print("请安装plotly库以启用图表功能：pip install plotly")
    
    def load_best_params(self):
        """
        加载之前保存的最佳参数
        
        Returns:
            最佳超参数，如果文件不存在则返回None
        """
        # 直接从config.py中读取BEST_NGBOOST_PARAMS
        import sys
        import importlib
        config_module = importlib.import_module('config')
        importlib.reload(config_module)
        self.best_params = config_module.BEST_NGBOOST_PARAMS
        print(f"已从config.py加载最佳超参数")
        return self.best_params

def load_and_preprocess_data(data_path: str, force_reprocess: bool = False) -> Tuple[HotelDataPreprocessor, pd.DataFrame, pd.DataFrame]:
    """
    加载和预处理数据（区分线上和线下用户）
    
    加载酒店预订数据并进行预处理，分别处理线上和线下用户的数据。
    支持缓存机制避免重复预处理，提高开发效率。
    
    Args:
        data_path: 数据文件路径，支持CSV格式
        force_reprocess: 是否强制重新预处理数据，忽略缓存
        
    Returns:
        Tuple[HotelDataPreprocessor, pd.DataFrame, pd.DataFrame]: (预处理器对象, 线上用户特征数据, 线下用户特征数据)
        
    处理流程：
    1. 检查缓存文件是否存在
    2. 如不存在或强制重新处理，执行完整预处理
    3. 分别处理线上和线下用户数据
    4. 保存预处理结果到缓存文件
    5. 返回预处理器和处理后的数据
    
    Note:
        - 自动检测和使用已有的预处理结果
        - 支持强制重新处理选项
        - 缓存文件包括预处理器参数和处理后的数据
        - 提供详细的处理进度报告
    """
    print("\n=== 数据预处理（区分线上和线下用户） ===")
    
    preprocessor_path = '../02_训练模型/preprocessor.pkl'
    online_data_path = '../03_数据文件/online_features.csv'
    offline_data_path = '../03_数据文件/offline_features.csv'
    
    # 检查是否已有预处理结果
    if not force_reprocess and os.path.exists(preprocessor_path) and os.path.exists(online_data_path) and os.path.exists(offline_data_path):
        print("发现已有的预处理结果，正在加载...")
        preprocessor = HotelDataPreprocessor.load_preprocessor(preprocessor_path)
        online_features_df = pd.read_csv(online_data_path)
        offline_features_df = pd.read_csv(offline_data_path)
        print(f"[OK] 线上用户数据加载完成，共{len(online_features_df)}条记录")
        print(f"[OK] 线下用户数据加载完成，共{len(offline_features_df)}条记录")
    else:
        if force_reprocess:
            print("强制重新执行数据预处理...")
        else:
            print("正在执行数据预处理...")
        preprocessor = HotelDataPreprocessor()
        
        # 加载原始数据
        raw_df = pd.read_csv(data_path)
        print(f"数据加载完成，共{len(raw_df)}条记录")
        
        # 筛选只保留City Hotel的数据
        initial_count = len(raw_df)
        raw_df = raw_df[raw_df['hotel'] == 'City Hotel'].copy()
        city_hotel_count = len(raw_df)
        print(f"过滤后只保留City Hotel数据，从{initial_count}条记录减少到{city_hotel_count}条记录")
        
        # 数据清洗
        cleaned_df = preprocessor.clean_data(raw_df)
        
        # 分别为线上和线下用户构造需求标签
        online_daily_stats, offline_daily_stats = preprocessor.construct_daily_demand_labels(cleaned_df)
        
        # 分别为线上和线下用户构造特征
        online_features_df = preprocessor.construct_features(online_daily_stats)
        offline_features_df = preprocessor.construct_features(offline_daily_stats)
        
        # 保存预处理结果
        online_features_df.to_csv(online_data_path, index=False)
        offline_features_df.to_csv(offline_data_path, index=False)
        preprocessor.save_preprocessor(preprocessor_path)
        print("[OK] 数据预处理完成")
    
    return preprocessor, online_features_df, offline_features_df

def train_ngboost_models(preprocessor: HotelDataPreprocessor, online_features_df: pd.DataFrame, 
                          offline_features_df: pd.DataFrame, force_retrain: bool = False, 
                          skip_hyperparameter_search: bool = False) -> Tuple[Dict[str, NGBoostTrainer], Dict[str, StandardScaler]]:
    """
    分别为线上和线下用户训练NGBoost模型（双需求：预定需求+实际需求）
    
    分别为线上和线下用户训练NGBoost模型用于需求预测，包括数据准备、模型训练和性能评估。
    支持模型缓存和增量训练，提供详细的训练过程监控。
    
    Args:
        preprocessor: 数据预处理器对象，包含特征工程方法
        online_features_df: 线上用户特征数据DataFrame
        offline_features_df: 线下用户特征数据DataFrame
        force_retrain: 是否强制重新训练模型，忽略已有模型
        
    Returns:
        Tuple[Dict[str, NGBoostTrainer], Dict[str, StandardScaler]]: 
            (训练器字典: {'online_booked': 线上预定模型, 'online_actual': 线上实际模型, 'offline_booked': 线下预定模型, 'offline_actual': 线下实际模型},
             标准化器字典: {'online_booked': 线上预定标准化器, 'online_actual': 线上实际标准化器, 'offline_booked': 线下预定标准化器, 'offline_actual': 线下实际标准化器})
        
    训练流程：
    1. 分别为线上和线下用户的预定需求和实际需求进行数据标准化
    2. 分别为四类需求构造训练样本
    3. 分别训练四个NGBoost模型
    4. 分别评估模型性能
    
    Note:
        - 分别为四类需求使用独立的需求标准化器
        - 支持模型缓存避免重复训练
        - 提供详细的训练进度和性能报告
        - 包含置信区间覆盖率评估
    """
    print("\n=== NGBoost模型训练（双需求：预定需求+实际需求） ===")
    
    model_paths = {
        'online_booked': '../02_训练模型/ngboost_model_online_booked.pkl',
        'online_actual': '../02_训练模型/ngboost_model_online_actual.pkl',
        'offline_booked': '../02_训练模型/ngboost_model_offline_booked.pkl',
        'offline_actual': '../02_训练模型/ngboost_model_offline_actual.pkl'
    }
    
    def train_single_ngboost_model(features_df, model_path, customer_type, demand_type, enable_hyperparameter_search=None):
        """为单个客户类型和特定需求类型训练NGBoost模型"""
        print(f"\n正在为{customer_type}的{demand_type}需求训练NGBoost模型...")
        
        # 确定是否启用超参数搜索
        if enable_hyperparameter_search is None:
            enable_hyperparameter_search = OPTUNA_CONFIG['enable_hyperparameter_search']
        
        # 选择对应的需求列
        demand_column = f'{demand_type}_demand'
        
        # 对需求数据进行标准化
        demand_scaler = StandardScaler()
        original_demands = features_df[demand_column].values.reshape(-1, 1)
        standardized_demands = demand_scaler.fit_transform(original_demands).flatten()
        
        # 保存标准化器
        scaler_path = f'../02_训练模型/demand_scaler_{customer_type}_{demand_type}.pkl'
        joblib.dump(demand_scaler, scaler_path)
        
        # 构造训练特征和标签
        X_list = []
        y_list = []
        
        print(f"正在为{customer_type}的{demand_type}需求使用真实价格数据创建训练样本...")
        
        # 使用真实价格数据创建训练样本
        for idx, row in features_df.iterrows():
            # 使用标准化后的需求作为目标
            standardized_demand = standardized_demands[idx]
            
            # 使用真实平均价格作为价格特征
            avg_price = row.get('avg_price', 120)  # 默认120如果缺失
            if pd.isna(avg_price) or avg_price <= 0:
                avg_price = 120  # 使用默认值
            
            # 准备特征（根据需求类型选择对应特征）
            features = preprocessor.prepare_ngboost_features(
                features_df.iloc[idx:idx+1], 
                price_action=avg_price,
                demand_type=demand_type,
                customer_type=customer_type
            )
            
            # 添加少量噪声到标准化需求数据
            noisy_demand = standardized_demand + np.random.normal(0, 0.05)  # 很小的噪声
            
            X_list.append(features)
            y_list.append(noisy_demand)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"{customer_type}的{demand_type}需求训练数据构造完成：X形状{X.shape}, y形状{y.shape}")
        print(f"{customer_type}的{demand_type}需求目标值范围：{y.min():.3f} - {y.max():.3f}（标准化后）")
        

        # 根据配置选择抽取方法
        if DATA_SPLIT_CONFIG['method'] in ['random_sample', 'sequential_sample']:
            # 随机抽取或顺序抽取
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.sample_data(
                X, y, 
                method=DATA_SPLIT_CONFIG['method'],
                train_samples=DATA_SPLIT_CONFIG['train_samples'],
                val_samples=DATA_SPLIT_CONFIG['val_samples'], 
                test_samples=DATA_SPLIT_CONFIG['test_samples'],
                random_seed=global_random_seed,
                stratify_by=DATA_SPLIT_CONFIG['stratify_by'],
                ensure_diversity=DATA_SPLIT_CONFIG['ensure_diversity']
            )
            
            print(f"使用{DATA_SPLIT_CONFIG['method']}抽取完成：")
            print(f"训练集：{len(X_train)}条记录（{len(X_train)/len(X)*100:.1f}%）")
            print(f"验证集：{len(X_val)}条记录（{len(X_val)/len(X)*100:.1f}%）")
            print(f"测试集：{len(X_test)}条记录（{len(X_test)/len(X)*100:.1f}%）")
            
        else:
            # 向后兼容：使用比例划分
            from sklearn.model_selection import train_test_split
            
            if DATA_SPLIT_CONFIG['method'] == 'random':
                # 随机划分，避免时间序列偏差
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=DATA_SPLIT_CONFIG['test_ratio'], 
                    random_state=global_random_seed, 
                    shuffle=DATA_SPLIT_CONFIG['shuffle']
                )
                
                # 再从临时集中划分验证集和测试集
                val_ratio_adjusted = DATA_SPLIT_CONFIG['val_ratio'] / (DATA_SPLIT_CONFIG['val_ratio'] + DATA_SPLIT_CONFIG['test_ratio'])
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_ratio_adjusted, 
                    random_state=global_random_seed, 
                    shuffle=DATA_SPLIT_CONFIG['shuffle']
                )
                
                print(f"使用随机划分完成：")
                print(f"训练集：{len(X_train)}条记录（{len(X_train)/len(X)*100:.1f}%）")
                print(f"验证集：{len(X_val)}条记录（{len(X_val)/len(X)*100:.1f}%）")
                print(f"测试集：{len(X_test)}条记录（{len(X_test)/len(X)*100:.1f}%）")
                
            else:
                # 时间顺序划分，保持时间序列特性
                total_samples = len(X)
                train_size = int(total_samples * DATA_SPLIT_CONFIG['train_ratio'])
                val_size = int(total_samples * DATA_SPLIT_CONFIG['val_ratio'])
                
                X_train = X[:train_size]
                X_val = X[train_size:train_size + val_size]
                X_test = X[train_size + val_size:]
                y_train = y[:train_size]
                y_val = y[train_size:train_size + val_size]
                y_test = y[train_size + val_size:]
                
                print(f"使用时间顺序划分完成：")
                print(f"训练集：{len(X_train)}条记录（{len(X_train)/len(X)*100:.1f}%）")
                print(f"验证集：{len(X_val)}条记录（{len(X_val)/len(X)*100:.1f}%）")
                print(f"测试集：{len(X_test)}条记录（{len(X_test)/len(X)*100:.1f}%）")

        # 超参数搜索或直接使用最佳参数
        if enable_hyperparameter_search:
            print(f"\n启用超参数搜索...")
            
            # 创建超参数优化器
            optimizer = NGBoostHyperparameterOptimizer()
            
            # 运行优化
            best_params = optimizer.optimize(
                X_train, y_train, X_val, y_val, demand_scaler, 
                customer_type, demand_type
            )
            
            # 使用最佳参数训练最终模型
            print(f"\n使用最佳超参数训练最终模型...")
            ngboost_trainer = NGBoostTrainer(
                distribution=best_params['distribution'],
                score='logscore',
                learning_rate=best_params['learning_rate'],
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                colsample_bytree=best_params['colsample_bytree'],
                verbose=best_params.get('verbose', 1),
                random_state=best_params.get('random_state', 42)
            )
            
            ngboost_trainer.train(
                X_train, y_train, X_val, y_val,
                early_stopping_rounds=best_params.get('early_stopping_rounds', 30),
                save_path=model_path
            )
            
            print(f"[OK] 使用最佳超参数的NGBoost模型训练完成")
            
            # 在超参数搜索分支中，使用best_params作为params_to_use
            params_to_use = best_params
            
        else:
            # 使用预设的最佳参数或默认配置
            print(f"\n跳过超参数搜索，使用预设最佳参数...")
            
            # 构建模型特定的键，用于获取对应的最佳参数
            model_key = f"{customer_type}_{demand_type}"
            
            # 直接使用config.py中的BEST_NGBOOST_PARAMS
            params_to_use = BEST_NGBOOST_PARAMS.get(model_key, BEST_NGBOOST_PARAMS['线上用户_booked']).copy()
            # 确保random_state与全局随机种子一致
            params_to_use['random_state'] = global_random_seed
            
            # 确保必要的参数存在，如果不存在则添加默认值
            if 'verbose' not in params_to_use:
                params_to_use['verbose'] = 1  # 设置默认verbose值
                
            print(f"使用config.py中的最佳参数（{model_key}）: {params_to_use}")
            
            # 检查是否已有训练好的模型
            if not force_retrain and os.path.exists(model_path):
                print(f"发现已有的{customer_type} NGBoost模型，正在加载...")
                ngboost_trainer = NGBoostTrainer(
                    distribution=params_to_use['distribution'],
                    score=params_to_use['score'],
                    learning_rate=params_to_use['learning_rate'],
                    n_estimators=params_to_use['n_estimators'],
                    max_depth=params_to_use['max_depth'],
                    min_samples_split=params_to_use['min_samples_split'],
                    min_samples_leaf=params_to_use['min_samples_leaf'],
                    colsample_bytree=params_to_use['colsample_bytree'],
                    verbose=params_to_use['verbose'],
                    random_state=params_to_use['random_state']
                )
                ngboost_trainer.load_model(model_path)
                print(f"[OK] {customer_type} NGBoost模型加载完成")
            else:
                if force_retrain:
                    print(f"强制重新训练{customer_type} NGBoost模型...")
                else:
                    print(f"正在训练{customer_type} NGBoost模型...")
                ngboost_trainer = NGBoostTrainer(
                    distribution=params_to_use['distribution'],
                    score=params_to_use['score'],
                    learning_rate=params_to_use['learning_rate'],
                    n_estimators=params_to_use['n_estimators'],
                    max_depth=params_to_use['max_depth'],
                    min_samples_split=params_to_use['min_samples_split'],
                    min_samples_leaf=params_to_use['min_samples_leaf'],
                    colsample_bytree=params_to_use['colsample_bytree'],
                    verbose=params_to_use['verbose'],
                    random_state=params_to_use['random_state']
                )
                
                ngboost_trainer.train(
                    X_train, y_train, X_val, y_val,
                    early_stopping_rounds=params_to_use.get('early_stopping_rounds', 30),
                    save_path=model_path
                )
                print(f"[OK] {customer_type} NGBoost模型训练完成")
        
        # 测试模型性能（使用标准化数据评估）
        print(f"正在评估{customer_type}的{demand_type}需求NGBoost模型性能...")
        mean_pred, var_pred = ngboost_trainer.predict(X_test[:100])
        
        # 反标准化预测结果用于显示
        mean_pred_original = demand_scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
        y_test_original = demand_scaler.inverse_transform(y_test[:100].reshape(-1, 1)).flatten()
        
        mae = np.mean(np.abs(mean_pred.flatten() - y_test[:100]))
        mae_original = np.mean(np.abs(mean_pred_original - y_test_original))
        
        print(f"{customer_type}的{demand_type}需求测试集MAE（标准化空间）: {mae:.4f}")
        print(f"{customer_type}的{demand_type}需求测试集MAE（原始空间）: {mae_original:.2f}")
        print(f"{customer_type}的{demand_type}需求预测范围（标准化）: {mean_pred.min():.3f} - {mean_pred.max():.3f}")
        print(f"{customer_type}的{demand_type}需求真实范围（标准化）: {y_test[:100].min():.3f} - {y_test[:100].max():.3f}")
        
        # 置信区间覆盖率检验
        print(f"\n正在计算{customer_type}的{demand_type}需求95%置信区间覆盖率...")
        # 获取分布类型（使用当前参数配置中的分布类型）
        distribution_type = params_to_use['distribution']
        coverage_rate, in_interval = evaluate_confidence_interval_coverage(
            mean_pred.flatten(), var_pred.flatten(), y_test[:100], 
            distribution=distribution_type
        )
        print(f"{customer_type}的{demand_type}需求95%置信区间覆盖率: {coverage_rate:.1f}%")
        print(f"{customer_type}的{demand_type}需求样本总数: {len(y_test[:100])}, 落在区间内: {np.sum(in_interval)}")
        
        # 绘制置信区间覆盖图
        feature_columns = features_df.columns.tolist()
        feature_columns = [col for col in feature_columns if col != 'date']  # 排除日期列
        plot_confidence_interval_coverage(
            y_test[:100], mean_pred.flatten(), var_pred.flatten(), 
            in_interval, coverage_rate, save_path=f"../05_分析报告/confidence_interval_coverage_{customer_type}_{demand_type}.png",
            feature_names=feature_columns,
            show_plot=False  # 在训练模式下不显示图形，避免阻塞
        )
        
        return ngboost_trainer, demand_scaler
    
    # 分别为线上和线下用户的预定需求和实际需求训练NGBoost模型
    trainers = {}
    scalers = {}
    
    # 根据参数决定是否跳过超参数搜索
    enable_hyperparameter_search = OPTUNA_CONFIG['enable_hyperparameter_search'] and not skip_hyperparameter_search
    if skip_hyperparameter_search:
        print("\n=== 跳过超参数搜索模式 ===")
        print("将直接使用预设的最佳超参数进行训练")
    
    # 线上用户模型
    trainers['online_booked'], scalers['online_booked'] = train_single_ngboost_model(
        online_features_df, model_paths['online_booked'], "线上用户", "booked", 
        enable_hyperparameter_search=enable_hyperparameter_search
    )
    trainers['online_actual'], scalers['online_actual'] = train_single_ngboost_model(
        online_features_df, model_paths['online_actual'], "线上用户", "actual",
        enable_hyperparameter_search=enable_hyperparameter_search
    )
    
    # 线下用户模型
    trainers['offline_booked'], scalers['offline_booked'] = train_single_ngboost_model(
        offline_features_df, model_paths['offline_booked'], "线下用户", "booked",
        enable_hyperparameter_search=enable_hyperparameter_search
    )
    trainers['offline_actual'], scalers['offline_actual'] = train_single_ngboost_model(
        offline_features_df, model_paths['offline_actual'], "线下用户", "actual",
        enable_hyperparameter_search=enable_hyperparameter_search
    )
    
    return trainers, scalers

def train_rl_system(ngboost_trainers: Dict[str, NGBoostTrainer], demand_scalers: Dict[str, StandardScaler], 
                   preprocessor: HotelDataPreprocessor, online_features_df: pd.DataFrame, 
                   offline_features_df: pd.DataFrame, use_bayesian_rl: bool = False) -> Tuple[HotelRLSystem, Optional[Dict], Optional[Dict]]:
    """
    训练强化学习系统（支持双需求：预定需求+实际需求）
    
    构建并训练酒店定价强化学习系统，包括离线预训练和在线学习两个阶段。
    使用Q-learning算法优化定价策略，结合四个NGBoost预测器（线上/线下 × 预定/实际需求）进行决策。
    
    Args:
        ngboost_trainers: NGBoost训练器字典，包含四个模型
        demand_scalers: 需求标准化器字典，包含四个标准化器
        preprocessor: 数据预处理器对象，包含特征工程方法
        online_features_df: 线上用户特征数据DataFrame
        offline_features_df: 线下用户特征数据DataFrame
        
    Returns:
        Tuple[HotelRLSystem, Optional[Dict], Optional[Dict]]: (RL系统, 在线学习统计, 策略评估统计)
        
    训练流程：
    1. 系统初始化：创建RL系统并配置探索参数
    2. 离线预训练：使用历史数据进行离线学习
    3. 探索统计：分析Q值分布和探索覆盖率
    4. 在线学习：根据配置进行在线策略优化 (由config['enable_online_learning']开关决定)
    5. 策略评估：评估学习后的策略性能（已关闭）
    
    Note:
        - 使用四个独立的NGBoost预测器分别预测线上/线下用户的预定/实际需求
        - 总预定需求 = 线上预定需求 + 线下预定需求
        - 总实际需求 = 线上实际需求 + 线下实际需求
        - 使用ε-贪心策略进行探索和利用平衡
        - 支持离线预训练和在线学习两个阶段
        - 提供详细的Q值统计和探索覆盖率分析
        - 策略评估功能默认关闭以提高训练效率
    """
    
    # 创建RL系统，使用四个NGBoost预测器
    rl_system = HotelRLSystem(
        ngboost_trainers=ngboost_trainers,
        preprocessor=preprocessor,
        demand_scalers=demand_scalers,
        epsilon_start=RL_CONFIG['epsilon_start'],
        epsilon_end=RL_CONFIG['epsilon_end'],
        epsilon_decay_episodes=RL_CONFIG['epsilon_decay_episodes'],
        use_bayesian_rl=use_bayesian_rl
    )
    
    # 训练强化学习系统（使用线上用户数据作为主要训练数据）
    print("开始离线预训练...")
    rl_system.offline_pretraining(online_features_df, episodes=RL_CONFIG['episodes'])
    
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
        online_stats = rl_system.online_learning(online_features_df, days=RL_CONFIG['online_learning_days'], update_frequency=RL_CONFIG['update_frequency'])
        
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
    5. 需求预测：使用NGBoost模型预测当前价格下的需求
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
    
    # 重置环境，设置房间数为400
    env = HotelEnvironment(initial_inventory=226)
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
        
        # 定价档位 - 基于数据分布优化的8个价格档位
        prices = [60, 80, 100, 110, 120, 130, 140, 150]
        price = prices[action]
        
        # 获取BNN预测
        predicted_demand, predicted_variance = rl_system.bnn_predictor(day_features, action)
        
        # 执行动作，使用四个NGBoost预测器
        next_state_info, reward, done, info = env.step(
            action, 
            rl_system.online_booked_predictor,
            rl_system.online_actual_predictor,
            rl_system.offline_booked_predictor,
            rl_system.offline_actual_predictor,
            day_features
        )
        
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
            'actual_demand': info.get('actual_demand', 0),  # 获取实际需求
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
    - NGBoost模型训练和评估
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
    parser.add_argument('--train-ngboost-only', action='store_true',
                       help='仅训练NGBoost模型，不训练Q-learning算法')
    parser.add_argument('--use-bayesian-rl', action='store_true',
                       help='使用贝叶斯Q-learning算法（默认使用标准Q-learning）')
    parser.add_argument('--skip-hyperparameter-search', action='store_true',
                       help='跳过超参数搜索，直接使用最佳超参数')
    parser.add_argument('--skip-ngboost-training', action='store_true',
                       help='跳过NGBoost训练，直接使用已有的NGBoost模型训练Q-learning')
    parser.add_argument('--run-uuid', type=str, default=None,
                       help='运行UUID，用于Q表存储和识别')
    # parser.add_argument('--simulate-days', type=int, default=90,
    #                    help='模拟天数')
    # parser.add_argument('--start-date', type=str, default='2017-01-01',
    #                    help='模拟开始日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    algorithm = "贝叶斯Q-learning" if args.use_bayesian_rl else "Q-learning"
    print(f"酒店动态定价系统 (NGBoost + {algorithm})")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        return
    
    # 数据预处理
    preprocessor, online_features_df, offline_features_df = load_and_preprocess_data(args.data, force_reprocess=args.force_retrain)
    
    if not args.skip_training:
        # 根据参数决定是否跳过NGBoost训练
        if args.skip_ngboost_training:
            print("\n=== 跳过NGBoost训练，直接使用已有NGBoost模型 ===")
            # 加载四个NGBoost模型（线上/线下用户的预定/实际需求）
            model_paths = {
                'online_booked': '../02_训练模型/ngboost_model_online_booked.pkl',
                'online_actual': '../02_训练模型/ngboost_model_online_actual.pkl',
                'offline_booked': '../02_训练模型/ngboost_model_offline_booked.pkl',
                'offline_actual': '../02_训练模型/ngboost_model_offline_actual.pkl'
            }
            
            # 检查所有模型文件是否存在
            missing_models = [path for path in model_paths.values() if not os.path.exists(path)]
            if missing_models:
                print(f"错误：未找到以下NGBoost模型文件：{missing_models}")
                print("请先训练模型或移除--skip-ngboost-training参数")
                return
            
            # 重新创建四个NGBoost训练器并加载模型
            ngboost_trainers = {}
            for model_key, model_path in model_paths.items():
                # 根据模型类型选择对应的超参数
                if 'online' in model_key:
                    if 'booked' in model_key:
                        params_key = '线上用户_booked'
                    else:
                        params_key = '线上用户_actual'
                else:  # offline
                    if 'booked' in model_key:
                        params_key = '线下用户_booked'
                    else:
                        params_key = '线下用户_actual'
                
                trainer = NGBoostTrainer(
                learning_rate=BEST_NGBOOST_PARAMS[params_key]['learning_rate'],
                n_estimators=BEST_NGBOOST_PARAMS[params_key]['n_estimators'],
                max_depth=BEST_NGBOOST_PARAMS[params_key]['max_depth'],
                min_samples_split=BEST_NGBOOST_PARAMS[params_key]['min_samples_split'],
                min_samples_leaf=BEST_NGBOOST_PARAMS[params_key]['min_samples_leaf'],
                colsample_bytree=BEST_NGBOOST_PARAMS[params_key]['colsample_bytree'],
                verbose=BEST_NGBOOST_PARAMS[params_key].get('verbose', 0),
                random_state=BEST_NGBOOST_PARAMS[params_key].get('random_state', 42)
            )
                trainer.load_model(model_path)
                ngboost_trainers[model_key] = trainer
            
            # 加载四个demand_scaler
            demand_scalers = {
                'online_booked': joblib.load('../02_训练模型/demand_scaler_线上用户_booked.pkl'),
                'online_actual': joblib.load('../02_训练模型/demand_scaler_线上用户_actual.pkl'),
                'offline_booked': joblib.load('../02_训练模型/demand_scaler_线下用户_booked.pkl'),
                'offline_actual': joblib.load('../02_训练模型/demand_scaler_线下用户_actual.pkl')
            }
            
            print("NGBoost模型加载完成")
            
            # 训练强化学习系统（使用加载的NGBoost模型）
            rl_system, online_stats, avg_stats = train_rl_system(ngboost_trainers, demand_scalers, preprocessor, online_features_df, offline_features_df, use_bayesian_rl=args.use_bayesian_rl)
        else:
            # 分别为线上和线下用户训练NGBoost模型（返回四个模型：线上/线下用户的预定/实际需求）
            ngboost_trainers, demand_scalers = train_ngboost_models(preprocessor, online_features_df, offline_features_df, force_retrain=args.force_retrain, skip_hyperparameter_search=args.skip_hyperparameter_search)
            
            # 根据参数决定是否训练Q-learning
            if args.train_ngboost_only:
                print("\n=== 跳过Q-learning训练（仅训练NGBoost模式） ===")
                # 创建RL系统但不训练，仅用于后续分析
                rl_system = HotelRLSystem(ngboost_trainers, preprocessor, demand_scalers, use_bayesian_rl=args.use_bayesian_rl)
                print("NGBoost模型训练完成，Q-learning训练已跳过")
                print("\n=== 仅训练NGBoost模式完成 ===")
                print("NGBoost模型已成功训练并保存")
                print("Q-learning训练已跳过")
                return  # 立即返回，避免执行后续代码
            else:
                # 训练强化学习系统（传入四个NGBoost训练器）
                rl_system, online_stats, avg_stats = train_rl_system(ngboost_trainers, demand_scalers, preprocessor, online_features_df, offline_features_df, use_bayesian_rl=args.use_bayesian_rl)
    else:
        # 加载已有模型
        print("\n正在加载已有模型...")
        
        # 加载四个NGBoost模型（线上/线下用户的预定/实际需求）
        model_paths = {
            'online_booked': '../02_训练模型/ngboost_model_online_booked.pkl',
            'online_actual': '../02_训练模型/ngboost_model_online_actual.pkl',
            'offline_booked': '../02_训练模型/ngboost_model_offline_booked.pkl',
            'offline_actual': '../02_训练模型/ngboost_model_offline_actual.pkl'
        }
        
        # 检查所有模型文件是否存在
        missing_models = [path for path in model_paths.values() if not os.path.exists(path)]
        if missing_models:
            print(f"错误：未找到以下NGBoost模型文件：{missing_models}")
            print("请先训练模型或移除--skip-training参数")
            return
        
        # 重新创建四个NGBoost训练器并加载模型
        ngboost_trainers = {}
        for model_key, model_path in model_paths.items():
            # 根据模型类型选择对应的超参数
            if 'online' in model_key:
                if 'booked' in model_key:
                    params_key = '线上用户_booked'
                else:
                    params_key = '线上用户_actual'
            else:  # offline
                if 'booked' in model_key:
                    params_key = '线下用户_booked'
                else:
                    params_key = '线下用户_actual'
            
            trainer = NGBoostTrainer(
                learning_rate=BEST_NGBOOST_PARAMS[params_key]['learning_rate'],
                n_estimators=BEST_NGBOOST_PARAMS[params_key]['n_estimators'],
                max_depth=BEST_NGBOOST_PARAMS[params_key]['max_depth'],
                min_samples_split=BEST_NGBOOST_PARAMS[params_key]['min_samples_split'],
                min_samples_leaf=BEST_NGBOOST_PARAMS[params_key]['min_samples_leaf'],
                colsample_bytree=BEST_NGBOOST_PARAMS[params_key]['colsample_bytree'],
                verbose=BEST_NGBOOST_PARAMS[params_key].get('verbose', 0),
                random_state=BEST_NGBOOST_PARAMS[params_key].get('random_state', 42)
            )
            trainer.load_model(model_path)
            ngboost_trainers[model_key] = trainer
        
        # 加载四个demand_scaler用于跳过训练模式
        demand_scalers = {
            'online_booked': joblib.load('../02_训练模型/demand_scaler_线上用户_booked.pkl'),
            'online_actual': joblib.load('../02_训练模型/demand_scaler_线上用户_actual.pkl'),
            'offline_booked': joblib.load('../02_训练模型/demand_scaler_线下用户_booked.pkl'),
            'offline_actual': joblib.load('../02_训练模型/demand_scaler_线下用户_actual.pkl')
        }
        
        # 创建RL系统，使用四个NGBoost预测器
        rl_system = HotelRLSystem(ngboost_trainers, preprocessor, demand_scalers, use_bayesian_rl=args.use_bayesian_rl)
        
        # 仅在非仅训练NGBoost模式下加载智能体
        if not args.train_ngboost_only:
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
        else:
            print("仅训练NGBoost模式：跳过智能体加载")
    
    # 运行模拟功能已移除
    
    # 模拟结果保存功能已移除
    # results_path = f'../04_结果输出/simulation_results_{start_date.strftime("%Y%m%d")}_{args.simulate_days}days.csv'
    # simulation_results.to_csv(results_path, index=False)
    # print(f"\n模拟结果已保存到：{results_path}")
    
    # 输出Q表信息（仅在非仅训练NGBoost模式下）
    
    print(f"\n=== {'贝叶斯' if args.use_bayesian_rl else ''}Q表信息 ===")
    if hasattr(rl_system, 'agent'):
        # 获取Q值统计
        q_stats = rl_system.agent.get_q_value_stats()
        if q_stats:
            print(f"{'贝叶斯' if args.use_bayesian_rl else ''}Q值统计:")
            print(f"  平均Q值: {q_stats['mean_q_value']:.2f}")
            if args.use_bayesian_rl:
                if 'mean_uncertainty' in q_stats:
                    print(f"  平均不确定性: {q_stats['mean_uncertainty']:.2f}")
                if 'std_uncertainty' in q_stats:
                    print(f"  不确定性标准差: {q_stats['std_uncertainty']:.2f}")
                if 'min_uncertainty' in q_stats:
                    print(f"  最小不确定性: {q_stats['min_uncertainty']:.2f}")
                if 'max_uncertainty' in q_stats:
                    print(f"  最大不确定性: {q_stats['max_uncertainty']:.2f}")
            else:
                print(f"  Q值标准差: {q_stats['std_q_value']:.2f}")
                print(f"  最小Q值: {q_stats['min_q_value']:.2f}")
                print(f"  最大Q值: {q_stats['max_q_value']:.2f}")
            print(f"  总状态访问次数: {q_stats['num_state_visits']}")
            print(f"  零值Q值占比: {q_stats['zero_q_percentage']:.1f}%")
            print(f"  探索覆盖率: {q_stats['exploration_coverage']:.1f}%")
            print(f"  已探索状态-动作对: {q_stats['explored_state_actions']}/{q_stats['total_state_actions']}")
        
        # 显示Q表内容（标准Q-learning）或Q值分布（贝叶斯Q-learning）
        if rl_system.is_standard_ql_agent():
            # 标准Q-learning
            q_table = rl_system.agent.q_table
            print(f"\nQ表状态数量: {len(q_table)}")
            
            # 显示前10个状态的Q值
            print(f"\n前10个状态的Q值:")
            # 36个动作组合配置（统一动作空间）
            online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个）
            offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个）
            
            # 生成36个动作的价格映射
            prices = []
            for online_idx in range(6):
                for offline_idx in range(6):
                    prices.append(f"线上{online_prices[online_idx]}线下{offline_prices[offline_idx]}")
            for i, (state, q_values) in enumerate(list(q_table.items())[:10]):
                best_action = np.argmax(q_values)
                # 确保best_action在有效范围内
                if best_action < len(prices):
                    print(f"状态 {state}: {[f'{q:.1f}' for q in q_values]} -> 最佳动作: {best_action} (价格: {prices[best_action]}元)")
                else:
                    print(f"状态 {state}: {[f'{q:.1f}' for q in q_values]} -> 最佳动作: {best_action} (价格: 索引超出范围)")
            
            if len(q_table) > 10:
                print(f"... 还有 {len(q_table) - 10} 个状态")
        elif rl_system.is_bayesian_ql_agent():
            # 贝叶斯Q-learning
            q_means = rl_system.agent.q_means
            q_vars = rl_system.agent.q_vars
            print(f"\nQ值分布数量: {len(q_means)}")
            
            # 显示前10个状态的Q值分布
            print(f"\n前10个状态的Q值分布（均值±标准差）:")
            # 36个动作组合配置（统一动作空间）
            online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个）
            offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个）
            
            # 生成36个动作的价格映射
            prices = []
            for online_idx in range(6):
                for offline_idx in range(6):
                    prices.append(f"线上{online_prices[online_idx]}线下{offline_prices[offline_idx]}")
            for i, (state, means) in enumerate(list(q_means.items())[:10]):
                variances = q_vars[state]
                uncertainties = np.sqrt(variances)
                best_action = np.argmax(means)
                q_distributions = [f'{m:.1f}±{u:.1f}' for m, u in zip(means, uncertainties)]
                # 确保best_action在有效范围内
                if best_action < len(prices):
                    print(f"状态 {state}: {q_distributions} -> 最佳动作: {best_action} (价格: {prices[best_action]}元)")
                else:
                    print(f"状态 {state}: {q_distributions} -> 最佳动作: {best_action} (价格: 索引超出范围)")
            
            if len(q_means) > 10:
                print(f"... 还有 {len(q_means) - 10} 个状态")
        
        # 保存Q表到CSV文件
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据算法类型获取Q值数据
        q_table_data = []
        # 36个动作组合配置（统一动作空间）
        online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个）
        offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个）
        
        # 生成36个动作的价格映射
        prices = []
        for online_idx in range(6):
            for offline_idx in range(6):
                prices.append(f"线上{online_prices[online_idx]}线下{offline_prices[offline_idx]}")
        
        if rl_system.is_standard_ql_agent():
            # 标准Q-learning
            q_data = rl_system.agent.q_table
            for state, q_values in q_data.items():
                best_action = np.argmax(q_values)
                
                # 动态创建动作列，支持8个价格档位
                row = {'state': state}
                
                # 添加所有动作的Q值
                for i in range(len(q_values)):
                    row[f'action_{i}'] = q_values[i]
                
                # 添加最佳动作信息（36个动作组合映射）
                if best_action < len(prices):
                    best_price = prices[best_action]
                    row.update({
                        'best_action': best_action,
                        'best_price': best_price,
                        'best_value': q_values[best_action]
                    })
                else:
                    # 对于超出36个动作的情况，显示警告但使用实际值
                    row.update({
                        'best_action': best_action,
                        'best_price': f'超出范围({best_action})',
                        'best_value': q_values[best_action]
                    })
                
                q_table_data.append(row)
                
        elif rl_system.is_bayesian_ql_agent():
            # 贝叶斯Q-learning
            q_means = rl_system.agent.q_means
            q_vars = rl_system.agent.q_vars
            for state, means in q_means.items():
                variances = q_vars[state]
                best_action = np.argmax(means)
                
                # 动态创建动作列，支持8个价格档位
                row = {'state': state}
                
                # 添加所有动作的均值、方差和不确定性
                for i in range(len(means)):
                    row[f'action_{i}'] = means[i]
                    row[f'variance_{i}'] = variances[i]
                    row[f'uncertainty_{i}'] = np.sqrt(variances[i])
                    row[f'action_{i}_mu_sigma'] = f'({means[i]:.0f}, {variances[i]:.1f})'
                
                # 添加最佳动作信息（36个动作组合映射）
                if best_action < len(prices):
                    best_price = prices[best_action]
                    row.update({
                        'best_action': best_action,
                        'best_price': best_price,
                        'best_value': means[best_action],
                        'best_uncertainty': np.sqrt(variances[best_action]),
                        'best_mu_sigma': f'({means[best_action]:.0f}, {variances[best_action]:.1f})'
                    })
                else:
                    # 对于超出36个动作的情况，显示警告但使用实际值
                    row.update({
                        'best_action': best_action,
                        'best_price': f'超出范围({best_action})',
                        'best_value': means[best_action],
                        'best_uncertainty': np.sqrt(variances[best_action]),
                        'best_mu_sigma': f'({means[best_action]:.0f}, {variances[best_action]:.1f})'
                    })
                
                q_table_data.append(row)
        
        if q_table_data:
            q_table_df = pd.DataFrame(q_table_data)
            
            # 保存到CSV
            q_table_csv_path = f'../05_分析报告/q_table_main_{timestamp}.csv'
            q_table_df.to_csv(q_table_csv_path, index=False)
            print(f"\nQ表已保存到CSV文件: {q_table_csv_path}")
            
            # 如果提供了run_uuid，则尝试将Q表数据存储到临时文件中
            if args.run_uuid:
                try:
                    # 将Q表数据转换为字符串格式
                    q_table_str = q_table_df.to_csv(index=False)
                    
                    # 创建临时文件存储Q表数据
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, f"q_table_{args.run_uuid}.csv")
                    
                    # 将Q表数据写入临时文件
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(q_table_str)
                    
                    print(f"Q表数据已存储到临时文件: {temp_file_path}")
                except Exception as e:
                    print(f"无法将Q表数据存储到临时文件: {e}")
            
            # 同时保存Q表统计信息
            if q_stats:
                stats_data = {
                    'total_states': len(q_table_data),
                    'mean_q_value': q_stats['mean_q_value'],
                    'total_visits': q_stats['num_state_visits'],
                    'zero_q_percentage': q_stats['zero_q_percentage'],
                    'exploration_coverage': q_stats['exploration_coverage'],
                    'explored_state_actions': q_stats['explored_state_actions'],
                    'total_state_actions': q_stats['total_state_actions']
                }
                
                # 根据算法类型添加相应的统计信息
                if args.use_bayesian_rl:
                    # 贝叶斯Q-learning的统计信息
                    if 'mean_uncertainty' in q_stats:
                        stats_data['mean_uncertainty'] = q_stats['mean_uncertainty']
                        stats_data['std_uncertainty'] = q_stats['std_uncertainty']
                        stats_data['min_uncertainty'] = q_stats['min_uncertainty']
                        stats_data['max_uncertainty'] = q_stats['max_uncertainty']
                    stats_data['algorithm'] = 'Bayesian Q-Learning'
                    stats_data['observation_noise_var'] = BQL_CONFIG['observation_noise_var']
                    stats_data['prior_mean'] = BQL_CONFIG['prior_mean']
                    stats_data['prior_var'] = BQL_CONFIG['prior_var']
                    stats_data['exploration_strategy'] = BQL_CONFIG['exploration_strategy']
                else:
                    # 标准Q-learning的统计信息
                    if 'std_q_value' in q_stats:
                        stats_data['std_q_value'] = q_stats['std_q_value']
                    if 'min_q_value' in q_stats:
                        stats_data['min_q_value'] = q_stats['min_q_value']
                    if 'max_q_value' in q_stats:
                        stats_data['max_q_value'] = q_stats['max_q_value']
                    stats_data['algorithm'] = 'Standard Q-Learning'
                
                stats_df = pd.DataFrame([stats_data])
                
                stats_csv_path = f'../05_分析报告/q_table_stats_{timestamp}.csv'
                stats_df.to_csv(stats_csv_path, index=False)
                print(f"Q表统计信息已保存到: {stats_csv_path}")
        
        # 绘制Q表热力图
        import seaborn as sns
        
        print("\n=== 开始绘制Q表热力图 ===")
        
        # 设置中文字体 - 添加更多备选字体确保兼容性
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 根据算法类型获取Q值数据
        if rl_system.is_standard_ql_agent():
            # 标准Q-learning
            q_data = rl_system.agent.q_table
            states = sorted(q_data.keys())
            # 获取实际的动作数量（从Q表中第一个状态获取）
            num_actions = len(q_data[states[0]]) if states else 36
            actions = list(range(num_actions))
            
            # 创建Q值矩阵（使用所有36个动作）
            q_matrix = np.zeros((len(states), num_actions))
            for i, state in enumerate(states):
                q_matrix[i, :] = q_data[state]
                
        elif rl_system.is_bayesian_ql_agent():
            # 贝叶斯Q-learning - 使用Q值均值
            q_means = rl_system.agent.q_means
            q_vars = rl_system.agent.q_vars
            states = sorted(q_means.keys())
            # 获取实际的动作数量（从Q表中第一个状态获取）
            num_actions = len(q_means[states[0]]) if states else 36
            actions = list(range(num_actions))
            
            # 创建Q值矩阵（使用均值，所有36个动作）
            q_matrix = np.zeros((len(states), num_actions))
            for i, state in enumerate(states):
                q_matrix[i, :] = q_means[state]
        else:
            print("[警告] 无法获取Q值数据，跳过热力图绘制")
            return
        
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
        
        # 动作标签（价格）- 6×6价格组合格式
        online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个动作）
        offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个动作）
        
        # 生成36个动作的标签（线上价格×线下价格组合）
        action_labels = []
        for online_idx in range(6):
            for offline_idx in range(6):
                online_price = online_prices[online_idx]
                offline_price = offline_prices[offline_idx]
                action_labels.append(f'线上¥{online_price}\n线下¥{offline_price}')
        
        # 创建Q值热力图 - 增加宽度防止文字重叠
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 创建注释矩阵显示(μ, σ²)格式
        if rl_system.is_bayesian_ql_agent():
            # 贝叶斯Q-learning: 显示(μ, σ²)格式（所有36个动作）
            annot_matrix = np.empty((len(states), len(actions)), dtype=object)
            for i, state in enumerate(states):
                means = q_means[state]
                vars = q_vars[state]
                for j in range(len(actions)):
                    annot_matrix[i, j] = f'({means[j]:.0f}, {vars[j]:.1f})'
            
            # 使用较小的字体以适应(μ, σ²)格式
            sns.heatmap(q_matrix, 
                        xticklabels=action_labels, 
                        yticklabels=state_labels,
                        cmap='RdYlBu_r', 
                        center=0,
                        annot=annot_matrix, 
                        fmt='',
                        annot_kws={'size': 8},
                        cbar_kws={'label': 'Q值均值'},
                        ax=ax)
        else:
            # 标准Q-learning: 保持原格式
            sns.heatmap(q_matrix, 
                        xticklabels=action_labels, 
                        yticklabels=state_labels,
                        cmap='RdYlBu_r', 
                        center=0,
                        annot=True, 
                        fmt='.1f',
                        cbar_kws={'label': 'Q值'},
                        ax=ax)
        
        # 设置标题和标签
        algorithm_name = '贝叶斯' if rl_system.is_bayesian_ql_agent() else '标准'
        if rl_system.is_bayesian_ql_agent():
            ax.set_title(f'{algorithm_name}Q值热力图 (μ, σ²) - 酒店动态定价策略', fontsize=16, fontweight='bold', pad=20)
        else:
            ax.set_title(f'{algorithm_name}Q值热力图 - 酒店动态定价策略', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
        ax.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
        
        # 改善坐标轴标签显示 - 防止重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存热力图
        heatmap_path = f'../04_结果输出/q_table_heatmap_{timestamp}.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Q值热力图已保存到: {heatmap_path}")
        
        # 显示热力图
        plt.show()
        
        # 创建最佳策略热力图
        print("\n=== 绘制最佳策略热力图 ===")
        
        # 创建最佳动作矩阵（考虑所有36个动作）
        best_action_matrix = np.zeros((len(states), len(actions)))
        for i, state in enumerate(states):
            if rl_system.is_standard_ql_agent():
                best_action = np.argmax(q_data[state])
            else:
                best_action = np.argmax(q_means[state])
            best_action_matrix[i, best_action] = 1
            
        # 显示策略分析信息
        print(f"\n=== 策略分析 ===")
        print(f"总状态数: {len(states)}")
        print(f"动作数: {num_actions} (6×6价格组合模式)")
        
        # 将矩阵转换为整数类型以避免格式化错误
        best_action_matrix = best_action_matrix.astype(int)
        
        # 创建最佳策略热力图 - 增加宽度防止文字重叠
        fig2, ax2 = plt.subplots(figsize=(16, 10))
        
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
        algorithm_name = '贝叶斯' if rl_system.is_bayesian_ql_agent() else '标准'
        ax2.set_title(f'{algorithm_name}最佳策略热力图 - 酒店动态定价', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
        ax2.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
        
        # 改善坐标轴标签显示 - 防止重叠
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax2.get_yticklabels(), fontsize=9)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存最佳策略热力图
        best_policy_path = f'../04_结果输出/best_policy_heatmap_{timestamp}.png'
        plt.savefig(best_policy_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 最佳策略热力图已保存到: {best_policy_path}")
        
        # 显示最佳策略热力图
        plt.show()
        
        print("[OK] Q值热力图绘制完成")
        
        # 如果是贝叶斯Q-learning，绘制不确定性热力图
        if rl_system.is_bayesian_ql_agent():
            print("\n=== 绘制不确定性热力图 ===")
            
            # 创建不确定性矩阵（所有36个动作）
            uncertainty_matrix = np.zeros((len(states), len(actions)))
            for i, state in enumerate(states):
                variances = q_vars[state]
                uncertainty_matrix[i, :] = np.sqrt(variances)  # 标准差作为不确定性
            
            # 创建不确定性热力图 - 增加宽度防止文字重叠
            fig3, ax3 = plt.subplots(figsize=(16, 10))
            
            sns.heatmap(uncertainty_matrix, 
                        xticklabels=action_labels, 
                        yticklabels=state_labels,
                        cmap='YlOrRd', 
                        annot=True, 
                        fmt='.2f',
                        cbar_kws={'label': '不确定性（标准差）'},
                        ax=ax3)
            
            # 设置标题和标签
            ax3.set_title('Q值不确定性热力图 - 贝叶斯Q-learning', fontsize=16, fontweight='bold', pad=20)
            ax3.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
            ax3.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
            
            # 改善坐标轴标签显示 - 防止重叠
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            plt.setp(ax3.get_yticklabels(), fontsize=9)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存不确定性热力图
            uncertainty_path = f'../04_结果输出/q_uncertainty_heatmap_{timestamp}.png'
            plt.savefig(uncertainty_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Q值不确定性热力图已保存到: {uncertainty_path}")
            
            # 显示不确定性热力图
            plt.show()
            
            # 创建精度热力图（方差的倒数）
            print("\n=== 绘制精度热力图 ===")
            
            # 创建精度矩阵（所有36个动作）
            precision_matrix = np.zeros((len(states), len(actions)))
            for i, state in enumerate(states):
                variances = q_vars[state]
                precision_matrix[i, :] = 1.0 / (variances + 1e-8)  # 避免除零
            
            # 创建精度热力图
            fig4, ax4 = plt.subplots(figsize=(14, 10))
            
            sns.heatmap(precision_matrix, 
                        xticklabels=action_labels, 
                        yticklabels=state_labels,
                        cmap='YlGnBu', 
                        annot=True, 
                        fmt='.2f',
                        cbar_kws={'label': '精度（1/方差）'},
                        ax=ax4)
            
            # 设置标题和标签
            ax4.set_title('Q值精度热力图 - 贝叶斯Q-learning', fontsize=16, fontweight='bold', pad=20)
            ax4.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
            ax4.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存精度热力图
            precision_path = f'../04_结果输出/q_precision_heatmap_{timestamp}.png'
            plt.savefig(precision_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Q值精度热力图已保存到: {precision_path}")
            
            # 显示精度热力图
            plt.show()
    
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
    # 添加超参数搜索控制逻辑
    if OPTUNA_CONFIG['enable_hyperparameter_search']:
        print("=" * 60)
        print("超参数搜索模式已启用")
        print("=" * 60)
        print(f"搜索目标：MAE权重={OPTUNA_CONFIG['objective_weights']['mae_weight']}, "
              f"置信区间覆盖率权重={OPTUNA_CONFIG['objective_weights']['coverage_weight']}")
        print(f"搜索次数：{OPTUNA_CONFIG['n_trials']} 次")
        print(f"早停策略：patience={OPTUNA_CONFIG['early_stopping']['patience']}, "
              f"min_delta={OPTUNA_CONFIG['early_stopping']['min_delta']}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("使用预设最佳超参数模式")
        print("=" * 60)
        print("跳过了超参数搜索，直接使用config.py中配置的最佳参数")
        print("=" * 60)
    
    main()
