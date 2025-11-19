#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 酒店动态定价系统 (BNN + Q-learning)

本配置文件包含了酒店动态定价系统的所有参数设置，包括：
- 数据路径和预处理配置
- 贝叶斯神经网络(BNN)模型参数
- 强化学习(Q-learning)算法参数
- 环境配置和定价策略
- 模拟和训练参数
- 系统性能和日志配置


"""


# 标准库导入
import os

# 第三方库导入已移除PyTorch依赖

# =============================================================================
# 数据配置
# =============================================================================
# 数据文件路径配置 - 支持相对路径和绝对路径
# 获取项目根目录，用于构建绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CONFIG = {
    'data_path': os.path.join(PROJECT_ROOT, '03_数据文件', 'hotel_bookings.csv'),  # 原始酒店预订数据
    'preprocessor_path': os.path.join(PROJECT_ROOT, '02_训练模型', 'preprocessor.pkl'),  # 数据预处理器序列化文件，保存特征缩放、编码等预处理参数
    'processed_data_path': os.path.join(PROJECT_ROOT, '03_数据文件', 'processed_features.csv'),  # 处理后的特征数据，用于模型训练和评估
    'analysis_path': os.path.join(PROJECT_ROOT, '05_分析报告', 'hotel_bookings_analysis.json'),  # 数据分析结果
    'unique_results_path': os.path.join(PROJECT_ROOT, '05_分析报告', 'unique_result_hotel_bookings.json')  # 唯一值统计
}
"""
数据配置说明：
- data_path: 酒店预订原始数据文件路径，包含预订记录、客户信息、入住信息等
- preprocessor_path: 数据预处理器序列化文件，保存特征缩放、编码等预处理参数
- processed_data_path: 预处理后的特征数据，用于模型训练和评估
- analysis_path: 数据分析结果文件，包含数据分布、统计特征等分析结果
- unique_results_path: 唯一值统计文件，记录各特征的唯一值分布情况

数据文件结构要求：
- 支持CSV格式，UTF-8编码
- 包含必要的特征列：入住日期、离店日期、房型、价格等
- 数据质量要求：无缺失值，格式统一
"""

# =============================================================================
# NGBoost模型配置
# =============================================================================
# NGBoost模型用于需求预测，输出需求量的均值和方差
# NGBoost优势：训练更快，预测更稳定，天然支持概率预测
NGBOOST_CONFIG = {
    # 模型架构参数
    'input_dim': 6,  # 输入特征维度，使用6个核心特征
    'output_dim': 2,  # 输出维度为2，分别预测需求的均值和方差
    
    # NGBoost核心参数 - 优化版本（基于参数调优结果调整）
    'distribution': 'normal',  # 预测分布类型：'normal'（正态）或 'lognormal'（对数正态）
    'score': 'logscore',  # 评分规则：'logscore'（对数评分）或 'crps'（CRPS评分）
    'n_estimators': 200,  # 基于参数调优的最佳值：平衡性能和训练效率
    'learning_rate': 0.01,  # 基于参数调优的最佳值：稳定收敛
    'max_depth': 3,  # 基于参数调优的最佳值：防止过拟合，保持泛化能力
    'min_samples_split': 25,  # 基于参数调优的最佳值：确保分裂稳定性
    'min_samples_leaf': 15,  # 基于参数调优的最佳值：防止叶节点过拟合
    'colsample_bytree': 0.8,  # 基于参数调优的最佳值：特征采样比例
    'verbose': False,  # 是否显示详细训练输出，False保持静默
    'random_state': None,  # 随机种子，设置为None以允许随机变化，设置为具体数值以确保结果可复现
    
    # 训练参数 - 优化早停策略
    'early_stopping_rounds': 30,  # 基于参数调优的最佳值：平衡训练效率和防止过拟合
    # 注意：validation_fraction 由数据划分配置控制，不在NGBoost参数中直接使用
    
    # 模型保存路径
    'model_path_online': os.path.join(PROJECT_ROOT, '02_训练模型', 'ngboost_model_online.pkl'),
    'model_path_offline': os.path.join(PROJECT_ROOT, '02_训练模型', 'ngboost_model_offline.pkl'),
    'model_path_final': os.path.join(PROJECT_ROOT, '02_训练模型', 'ngboost_model_final.pkl')
}
"""
NGBoost配置说明：
模型架构：
- input_dim: 输入特征维度，当前使用6个核心特征（season、is_weekend、avg_price、price_cv、demand_trend、price_trend）
- output_dim: 输出维度为2，分别预测需求的均值和方差
- distribution: 预测分布类型，正态分布适合大多数需求预测场景
- score: 评分规则，对数评分适合概率预测

训练策略：
- n_estimators: 200，适中的提升迭代次数，平衡训练时间和模型性能
- learning_rate: 0.01，标准学习率，确保稳定收敛
- max_depth: 3，适中的树深度，平衡模型复杂度和泛化能力
- min_samples_leaf: 15，防止过拟合的正则化参数
- colsample_bytree: 0.8，特征采样增加模型鲁棒性

优势特性：
- 训练速度：NGBoost训练速度快，不需要复杂的变分推理
- 预测稳定性：NGBoost预测稳定，不依赖蒙特卡洛采样
- 超参数调优：NGBoost超参数更少，更容易调优
- 概率预测：NGBoost天然支持概率预测和不确定性量化
"""

# =============================================================================
# 强化学习(Q-learning)配置
# =============================================================================
# Q-learning算法参数，用于学习最优定价策略
# 状态空间: 库存档位 × 季节 × 日期类型 = 5 × 3 × 2 = 30种状态
# 动作空间: 36个定价组合（线上6档 × 线下6档）
# 线上价格档位: [80, 90, 100, 110, 120, 130]元
# 线下价格档位: [90, 105, 120, 135, 150, 165]元
RL_CONFIG = {
    # Q-learning核心参数
    'learning_rate': 0.05,  # Q值学习率，降低学习率以提高稳定性（从0.1降至0.05）
    'discount_factor': 0.99,  # 折扣因子，提高对长期收益的重视（从0.95提高到0.99）
    'epsilon_start': 0.9,  # 初始探索率，适当降低初始探索（从0.95降至0.9）
    'epsilon_end': 0.01,  # 最终探索率，降低最终探索率以更专注利用（从0.05降至0.01）
    'epsilon_decay_episodes': 100,  # 探索率衰减轮数，延长衰减期（从200增加到300）
    'epsilon_min': 0.01,  # 最小探索率，与最终探索率保持一致
    
    # 训练配置
    'episodes': 320, # 离线预训练轮数（从150增加到300，确保充分训练）
    'online_learning_days': 90,  # 在线学习天数
    'update_frequency': 7,  # NGBoost模型更新频率（天）
    
    # 在线学习开关 
    'enable_online_learning': False,  # 是否启用在线学习，False则只使用离线训练
    
    # 定价档位配置 - 与环境保持一致
    'online_price_levels': [80, 90, 100, 110, 120, 130],  # 线上6个定价档位（元/晚）
    'offline_price_levels': [90, 105, 120, 135, 150, 165],  # 线下6个定价档位（元/晚）
    'n_actions': 36,  # 总动作数：6×6=36个价格组合
    
    # 智能体模型保存路径
    'agent_paths': {
        'pretrained': os.path.join(PROJECT_ROOT, '02_训练模型', 'q_agent_pretrained.pkl'),  # 离线预训练模型
        'final': os.path.join(PROJECT_ROOT, '02_训练模型', 'q_agent_final.pkl'),  # 最终模型（含在线学习）
        'online': os.path.join(PROJECT_ROOT, '02_训练模型', 'q_agent_online.pkl')  # 在线学习中间模型
    }
}
"""
RL配置说明：
Q-learning核心参数：
- learning_rate: Q值更新步长，0.1为适中的学习速度
- discount_factor: 0.95，较高的折扣因子重视长期收益
- epsilon策略：从0.9到0.1的衰减，平衡探索与利用
- epsilon_min: 保持最低1%的探索率避免陷入局部最优

训练策略：
- episodes: 离线训练轮数，可根据计算资源调整
- online_learning_days: 90天在线学习期
- update_frequency: 每7天更新一次BNN模型
- enable_online_learning: 在线学习开关，可切换训练模式

模型路径：
- pretrained: 离线预训练模型保存路径
- final: 包含在线学习的最终模型
- online: 在线学习过程中的中间模型

状态空间设计：
- 库存档位：5个离散化等级
- 季节特征：3个季节类别
- 日期类型：工作日/周末
- 总状态数：5×3×2=30种状态
"""

# =============================================================================
# 环境配置
# =============================================================================
# 酒店环境参数，模拟真实的酒店运营环境
ENV_CONFIG = {
    # 库存配置
    'initial_inventory': 226,  # 初始库存数量（房间总数）- 基于实际最大库存分析
    'max_inventory': 226,  # 最大库存容量 - 基于历史数据分析确定为226间
    'min_inventory': 0,  # 最小库存（不能为负）
    
    # 定价策略 - 支持36个动作组合（6×6）
    'online_price_levels': [80, 90, 100, 110, 120, 130],  # 线上6个定价档位（元/晚）
    'offline_price_levels': [90, 105, 120, 135, 150, 165],  # 线下6个定价档位（元/晚）
    'n_actions': 36,  # 总动作数：6×6=36个价格组合
    # 动作索引映射：action_idx = online_idx × 6 + offline_idx
    # 价格范围基于数据分析:
    # - 线上价格范围: 93.61-115.91元
    # - 线下价格范围: 99.96-152.80元
    # - 90%的数据在150元以下
    # 注意：确保与RL_CONFIG中的定价档位保持一致
    
    # 奖励函数权重
    'demand_weight': 0.7,  # 需求满足权重
    'inventory_weight': 0.3,  # 库存管理权重
    'revenue_weight': 1.0,  # 收益权重（主要目标）
    'booking_weight': 0.5  # 预订成功权重
}

# =============================================================================
# 模拟配置
# =============================================================================
# 系统模拟和评估参数
SIMULATION_CONFIG = {
    'default_days': 90,  # 默认模拟天数
    'default_start_date': '2017-01-01',  # 默认开始日期
    'evaluation_episodes': 10,  # 策略评估轮数
    'results_path': os.path.join(PROJECT_ROOT, '04_结果输出', 'simulation_results')  # 结果保存路径前缀
}

# =============================================================================
# 数据划分配置
# =============================================================================
# 训练集、验证集和测试集的抽取策略配置
# 支持两种模式：
# 1. 随机抽取：从整个数据池中随机选择指定数量的样本
# 2. 顺序抽取：按时间顺序选择指定数量的样本
DATA_SPLIT_CONFIG = {
    'method': 'random_sample',  # 抽取方法: 'random_sample' (随机抽取) 或 'sequential_sample' (顺序抽取)
    'train_samples': 643,     # 训练集样本数量（提高样本利用率）
    'val_samples': 50,       # 验证集样本数量（增加验证集大小）
    'test_samples': 100,      # 测试集样本数量（使用剩余样本）
    'random_seed': 42,        # 随机种子，确保可重复性
    'stratify_by': None,      # 分层抽样的列名，None表示不进行分层抽样
    'shuffle': True,          # 是否在随机抽取时打乱数据
    'ensure_diversity': True  # 是否确保抽取样本的多样性（时间、季节、价格等）
}

# =============================================================================
# Optuna超参数搜索配置
# =============================================================================
# 使用Optuna进行NGBoost超参数优化，支持双目标优化
OPTUNA_CONFIG = {
    # 搜索控制参数
    'enable_hyperparameter_search': True,  # 是否启用超参数搜索，False则使用预设最佳参数
    'n_trials': 64,  # 搜索试验次数
    'timeout': 3600,  # 搜索超时时间（秒），3600秒=1小时
    'n_jobs': 28,  # 并行工作数，-1表示使用所有CPU核心
    'random_seed': 42,  # 随机种子，确保搜索可重复
    
    # 超参数搜索空间
    'param_space': {
        'n_estimators': {'type': 'int', 'low': 100, 'high': 500, 'step': 50},  # 树的数量
        'learning_rate': {'type': 'float', 'low': 0.005, 'high': 0.1, 'log': True},  # 学习率
        'max_depth': {'type': 'int', 'low': 3, 'high': 8, 'step': 1},  # 树的最大深度
        'min_samples_split': {'type': 'int', 'low': 10, 'high': 50, 'step': 5},  # 分裂最小样本数
        'min_samples_leaf': {'type': 'int', 'low': 10, 'high': 30, 'step': 5},  # 叶节点最小样本数
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0, 'step': 0.1},  # 特征采样比例
        'distribution': {'type': 'categorical', 'choices': ['normal']},  # 分布类型：只使用正态分布，避免LogNormal对负值的要求
    },
    
    # 多目标优化权重（基于概率评估指标）
    'objective_weights': {
        'log_likelihood_weight': 1.0,  # 对数似然权重（主要优化目标）
        'mae_weight': 0.1,  # MAE目标权重（辅助监测指标）
        'coverage_weight': 10000000,  # 置信区间覆盖率权重 
        'crps_weight': 0.1,  # CRPS权重（连续秩概率得分）
        'pit_weight': 0.05,  # PIT权重（概率积分变换）
    },
    
    # 早停策略
    'early_stopping': {
        'patience': 10,  # 早停耐心轮数
        'min_delta': 0.001,  # 最小改善阈值
    },
    
    # 结果保存配置
    'save_results': True,  # 是否保存搜索结果
    'results_path': os.path.join(PROJECT_ROOT, '02_训练模型', 'optuna_results.pkl'),  # 搜索结果保存路径
    'study_path': os.path.join(PROJECT_ROOT, '02_训练模型', 'optuna_study.pkl'),  # Optuna study保存路径
    
    # 可视化配置
    'plot_results': True,  # 是否绘制搜索结果图表
    'plots_path': os.path.join(PROJECT_ROOT, '05_分析报告', 'optuna_optimization'),  # 图表保存目录
}

# 预设的最佳超参数（当enable_hyperparameter_search=False时使用）
# 注意：当启用超参数搜索时，搜索结果会自动更新这些参数到config.py中
# 每个客户类型和需求类型组合的最佳超参数
BEST_NGBOOST_PARAMS = {
    '线上用户_booked': {
        'n_estimators': 150,
        'learning_rate': 0.00554476117271747,
        'max_depth': 4,
        'min_samples_split': 45,
        'min_samples_leaf': 20,
        'colsample_bytree': 0.7,
        'distribution': 'normal',
        'score': 'logscore',
    },
    '线上用户_actual': {
        'n_estimators': 100,
        'learning_rate': 0.006897261486434552,
        'max_depth': 3,
        'min_samples_split': 25,
        'min_samples_leaf': 20,
        'colsample_bytree': 0.8,
        'distribution': 'normal',
        'score': 'logscore',
    },
    '线下用户_booked': {
        'n_estimators': 350,
        'learning_rate': 0.008388080806698468,
        'max_depth': 3,
        'min_samples_split': 50,
        'min_samples_leaf': 15,
        'colsample_bytree': 1.0,
        'distribution': 'normal',
        'score': 'logscore',
    },
    '线下用户_actual': {
        'n_estimators': 150,
        'learning_rate': 0.007199513515716703,
        'max_depth': 6,
        'min_samples_split': 45,
        'min_samples_leaf': 25,
        'colsample_bytree': 0.7,
        'distribution': 'normal',
        'score': 'logscore',
    },
}

# =============================================================================
# 随机因子配置
# =============================================================================
# 控制系统中的随机性，支持固定模式和随机模式
RANDOM_CONFIG = {
    'random_mode': 'random',  # 'fixed' 或 'random'
    'fixed_seed': 42,  # 固定模式下的随机种子
    'description': '随机因子控制配置 - 支持固定和随机两种模式'
}

# =============================================================================
# 系统配置
# =============================================================================
# 系统级配置参数，控制硬件使用和全局行为
SYSTEM_CONFIG = {
    'use_cuda': False,  # 是否使用CUDA GPU加速（如果可用）
    'device': 'cpu',  # 设备选择：'auto', 'cuda', 'cpu'
    'random_seed': 42,  # 全局随机种子
    'max_workers': 28,  # 最大工作进程数（用于并行处理）
    'memory_limit_gb': 24,  # 内存使用限制（GB）
    'enable_gpu_memory_growth': True,  # 是否启用GPU内存增长
    'mixed_precision': False,  # 是否使用混合精度训练
    'compile_models': False,  # 是否编译模型（PyTorch 2.0+）
    'profile_performance': False  # 是否启用性能分析
}

# =============================================================================
# 日志配置
# =============================================================================
# 系统日志和输出配置
LOG_CONFIG = {
    'log_level': 'INFO',  # 日志级别: DEBUG, INFO, WARNING, ERROR
    'log_file': os.path.join(PROJECT_ROOT, '06_临时文件', 'hotel_pricing.log'),  # 日志文件路径
    'console_output': True,  # 是否输出到控制台
    'save_intermediate_results': True  # 是否保存中间结果
}

# =============================================================================
# 贝叶斯强化学习(BQL)配置
# =============================================================================
# 贝叶斯Q-Learning算法参数，使用概率分布表示Q值信念
BQL_CONFIG = {
    # Q-learning核心参数
    'discount_factor': 0.97,  # 提高以重视长期收益，适配库存管理
    
    # 贝叶斯参数
    'observation_noise_var': 1.2,  # 增大以兼容需求预测噪声
    'prior_mean': 50.0,  # 基于实际净收益范围调整
    'prior_var': 15.0,  # 增大以鼓励初始探索
    
    # 探索策略
    'exploration_strategy': 'thompson',  # 适合挖掘高潜力定价策略
    'ucb_c': 2.5,  # 增大以强化旺季探索
    'ucb_bonus_scale': 2.0,  # 与ucb_c配合平衡均值与不确定性
    'epsilon_start': 0.9,  
    'epsilon_min': 0.1,  
    
    # 训练配置
    'episodes': 10,  # 增加以覆盖完整季节规律
    'online_learning_days': 90,  
    
    # 系统配置
    'random_seed': 42,  
    'save_frequency': 20,  # 减少保存频率
    'checkpoint_frequency': 50,  
    
    # 资源限制
    'max_memory_usage': 0.9,  
    'max_cpu_cores': 0,  
    
    # 性能优化
    'enable_memory_optimization': True,  
    'batch_data_loading': True,  
    'parallel_processing': True,  
    
    # 智能体模型保存路径（保持不变）
    'agent_paths': {
        'pretrained': os.path.join(PROJECT_ROOT, '02_训练模型', 'bql_agent_pretrained.pkl'),
        'final': os.path.join(PROJECT_ROOT, '02_训练模型', 'bql_agent_final.pkl'),
        'online': os.path.join(PROJECT_ROOT, '02_训练模型', 'bql_agent_online.pkl')
    }
}

"""
BQL配置说明：
贝叶斯参数：
- observation_noise_var: 观测噪声方差，控制TD目标的可信度
- prior_mean/var: 先验分布参数，初始信念
- 较小的观测噪声方差意味着更信任新观测

探索策略：
- ucb: 基于不确定性的上置信界探索
- thompson: Thompson采样，从后验分布采样
- epsilon_greedy: 标准的ε-贪心策略

核心特性：
- 维护Q值的概率分布而非点估计
- 不确定性随经验增加而降低
- 支持多种贝叶斯探索策略
- 提供不确定性量化
"""

def get_device() -> str:
    """
    获取计算设备
    
    由于移除了PyTorch依赖，此函数现在返回简单的设备标识符。
    
    Returns:
        str: 设备标识符，'cpu'
        
    Note:
        - NGBoost基于scikit-learn，主要使用CPU计算
        - 移除了GPU相关配置以简化系统
    """
    return 'cpu'


def setup_directories() -> None:
    """
    创建必要的目录
    
    根据项目结构创建所有必需的目录，确保数据、模型、结果等文件有正确的保存路径。
    使用当前脚本位置作为基准路径，确保路径正确性。
    
    创建的目录结构：
    - 02_训练模型: 保存训练好的模型文件
    - 03_数据文件: 存放原始数据和处理后的数据
    - 04_结果输出: 保存模拟和评估结果
    - 05_分析报告: 保存分析结果和图表
    - 06_临时文件: 存放日志、检查点等临时文件
    - 06_临时文件/checkpoints: 模型检查点
    - 06_临时文件/results: 中间结果
    - 06_临时文件/logs: 日志文件
    
    Note:
        - 使用exist_ok=True避免已存在目录报错
        - 打印创建过程便于调试和验证
        - 基于脚本位置构建相对路径，提高可移植性
    """
    # 获取当前脚本文件的目录，确保路径相对于脚本位置而不是工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 获取项目根目录（RL_Agent）
    
    directories = [
        os.path.join(project_root, '02_训练模型'),
        os.path.join(project_root, '03_数据文件'),
        os.path.join(project_root, '04_结果输出'),
        os.path.join(project_root, '05_分析报告'),
        os.path.join(project_root, '06_临时文件'),
        os.path.join(project_root, '06_临时文件', 'checkpoints'),
        os.path.join(project_root, '06_临时文件', 'results'),
        os.path.join(project_root, '06_临时文件', 'logs')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")  # 调试用，显示实际创建的目录路径

def validate_config() -> bool:
    """
    验证配置有效性
    
    检查配置文件中的各项参数是否有效，包括路径存在性、参数范围、逻辑一致性等。
    提供详细的错误信息帮助定位和修复配置问题。
    
    Returns:
        bool: 配置有效返回True，无效返回False
        
    验证项目：
    - 数据文件存在性检查
    - BNN参数范围验证（输入维度、隐藏层维度等）
    - RL参数逻辑验证（学习率、折扣因子等）
    - 路径格式和权限检查
    - 数值参数范围检查
    
    Note:
        - 打印详细的错误信息便于调试
        - 检查关键路径的存在性和可访问性
        - 验证数值参数的合理范围
        - 提供配置修复建议
    """
    import os
    import typing
    
    # 检查数据文件
    if not os.path.exists(DATA_CONFIG['data_path']):
        print(f"警告：数据文件不存在：{DATA_CONFIG['data_path']}")
        return False
    
    # BNN配置已移除，跳过相关检查
    
    # 检查RL配置
    epsilon_start = float(RL_CONFIG['epsilon_start'])  # type: ignore
    epsilon_end = float(RL_CONFIG['epsilon_end'])  # type: ignore
    discount_factor = float(RL_CONFIG['discount_factor'])  # type: ignore
    
    if epsilon_start < 0 or epsilon_start > 1:
        print("错误：epsilon_start必须在0和1之间")
        return False
    
    if epsilon_end < 0 or epsilon_end > 1:
        print("错误：epsilon_end必须在0和1之间")
        return False
    
    if discount_factor < 0 or discount_factor > 1:
        print("错误：折扣因子必须在0和1之间")
        return False
    
    # 检查环境配置
    initial_inventory = int(ENV_CONFIG['initial_inventory'])  # type: ignore
    online_price_levels = typing.cast(typing.List[int], ENV_CONFIG['online_price_levels'])  # type: ignore
    offline_price_levels = typing.cast(typing.List[int], ENV_CONFIG['offline_price_levels'])  # type: ignore
    n_actions = int(ENV_CONFIG['n_actions'])  # type: ignore
    
    if initial_inventory <= 0:
        print("错误：初始库存必须大于0")
        return False
    
    if len(online_price_levels) != 6 or len(offline_price_levels) != 6:
        print("错误：线上和线下价格档位都必须为6个")
        return False
    
    if n_actions != 36:
        print("错误：动作数量必须为36（6×6组合）")
        return False
    
    return True

# 初始化配置
# 获取项目根目录，用于构建所有相对路径的基准
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

setup_directories()
if not validate_config():
    print("配置验证失败，请检查配置文件")