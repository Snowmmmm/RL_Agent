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

import os
import torch

# =============================================================================
# 数据配置
# =============================================================================
# 数据文件路径配置 - 支持相对路径和绝对路径
# 获取项目根目录，用于构建绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CONFIG = {
    'data_path': os.path.join(PROJECT_ROOT, '03_数据文件', 'hotel_bookings.csv'),  # 原始酒店预订数据
    'preprocessor_path': os.path.join(PROJECT_ROOT, '02_训练模型', 'preprocessor.pkl'),  # 数据预处理器保存路径
    'processed_data_path': os.path.join(PROJECT_ROOT, '03_数据文件', 'processed_features.csv'),  # 处理后的特征数据
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
# 贝叶斯神经网络(BNN)配置
# =============================================================================
# BNN模型用于需求预测，输出需求量的均值和方差
# 网络结构: 输入层 -> 多个隐藏层 -> 双输出头(均值+方差)
BNN_CONFIG = {
    # 网络架构参数：增强模型拟合能力
    'input_dim': 3,  # 保持不变（若特征足够），若特征不足可后续增加
    'hidden_dims': [128, 64, 32],  # 加宽加深网络（原[64,32,16]容量不足），提升拟合复杂关系的能力
    'output_dim': 2,  # 保持不变（输出均值+方差）
    
    # 训练超参数：让模型更充分学习
    'learning_rate': 5e-5,  # 降低学习率（原13e-4=0.0013过高），避免训练震荡，便于收敛到更优解
    'batch_size': 32,  # 适当增大batch_size（原16），降低单批噪声，提升梯度估计稳定性（样本量足够时）
    'epochs': 1500,  # 适当增加最大轮数（原1000），给模型更多学习机会
    'early_stopping_patience': 30,  # 延长早停耐心（原20），避免因短期波动提前停止
    
    # 贝叶斯参数：减少过度约束，释放模型灵活性
    'beta_annealing_steps': 500,  # 缩短KL退火步数（原1000），让KL惩罚更慢生效（先学数据模式，再约束复杂度）
    'dropout_rate': 0.05,  # 降低dropout（原0.2过高），减少信息丢失，避免欠拟合
    'prior_mean': 0.0,  # 保持不变
    'prior_std': 1.5,  # 增大先验标准差（原0.5过小），放宽对参数的约束（先验越"宽松"，模型越容易学习数据波动）
    'weight_decay': 1e-5,  # 降低L2正则化（原1e-4过强），减少对权重的压制
    
    # 模型保存路径：保持不变
    'model_path': os.path.join(PROJECT_ROOT, '02_训练模型', 'bnn_model.pth'),
    'checkpoint_path': os.path.join(PROJECT_ROOT, '06_临时文件', 'bnn_checkpoint.pth')
}
"""
BNN配置说明：
网络架构：
- input_dim: 输入特征维度，当前使用3个核心特征
- hidden_dims: 隐藏层维度[128,64,32]，采用递减结构，增强特征提取能力
- output_dim: 输出维度为2，分别预测需求的均值和方差

训练策略：
- learning_rate: 5e-5，较小的学习率确保稳定收敛
- batch_size: 32，适中的批量大小平衡内存和稳定性
- epochs: 1500，充足的训练轮数确保模型充分学习
- early_stopping_patience: 30，避免过早停止训练

贝叶斯特性：
- beta_annealing_steps: KL散度退火步数，平衡数据拟合和模型复杂度
- dropout_rate: 0.05，轻度的dropout防止过拟合
- prior_mean/std: 先验分布参数，控制权重正则化强度
- weight_decay: L2正则化系数，进一步防止过拟合
"""

# =============================================================================
# 强化学习(Q-learning)配置
# =============================================================================
# Q-learning算法参数，用于学习最优定价策略
# 状态空间: 库存档位 × 季节 × 日期类型 = 5 × 3 × 2 = 30种状态
# 动作空间: 6个定价档位 [60, 90, 120, 150, 180, 210]元
RL_CONFIG = {
    # Q-learning核心参数
    'learning_rate': 0.1,  # Q值学习率，控制Q值更新速度
    'discount_factor': 0.95,  # 折扣因子，权衡即时与未来奖励
    'epsilon_start': 0.9,  # 初始探索率，前期多探索
    'epsilon_end': 0.1,  # 最终探索率，后期少探索
    'epsilon_decay_episodes': 100,  # 探索率衰减轮数
    'epsilon_min': 0.01,  # 最小探索率，保持少量探索
    
    # 训练配置
    'episodes': 10,  # 离线预训练轮数
    'online_learning_days': 90,  # 在线学习天数
    'update_frequency': 7,  # BNN模型更新频率（天）
    
    # 在线学习开关 
    'enable_online_learning': False,  # 是否启用在线学习，False则只使用离线训练
    
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
    'initial_inventory': 100,  # 初始库存数量（房间总数）
    'max_inventory': 100,  # 最大库存容量
    'min_inventory': 0,  # 最小库存（不能为负）
    
    # 定价策略
    'price_levels': [60, 90, 120, 150, 180, 210],  # 6个定价档位（元/晚）
    
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
# 训练集、验证集和测试集的划分策略配置
DATA_SPLIT_CONFIG = {
    'method': 'random',  # 划分方法: 'random' (随机) 或 'sequential' (时间顺序)
    'train_ratio': 0.7,  # 训练集比例
    'val_ratio': 0.15,   # 验证集比例
    'test_ratio': 0.15,  # 测试集比例
    'random_seed': 42,   # 随机种子，确保可重复性
    'stratify_by': None, # 分层抽样的列名，None表示不进行分层抽样
    'shuffle': True      # 是否在随机划分时打乱数据
}

# =============================================================================
# 系统配置
# =============================================================================
# 系统级配置参数，控制硬件使用和全局行为
SYSTEM_CONFIG = {
    'use_cuda': True,  # 是否使用CUDA GPU加速（如果可用）
    'device': 'auto',  # 设备选择：'auto', 'cuda', 'cpu'
    'random_seed': 42,  # 全局随机种子
    'max_workers': 4,  # 最大工作进程数（用于并行处理）
    'memory_limit_gb': 8,  # 内存使用限制（GB）
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

def get_device() -> torch.device:
    """
    获取计算设备
    
    根据系统配置和硬件可用性，返回最适合的计算设备（CUDA或CPU）。
    自动检测GPU可用性并选择合适的设备。
    
    Returns:
        torch.device: PyTorch设备对象，CUDA设备或CPU设备
        
    Note:
        - 优先使用CUDA GPU加速（如果可用且配置启用）
        - 自动处理设备检测和选择逻辑
        - 返回的设备可直接用于模型和张量操作
    """
    import torch
    if SYSTEM_CONFIG['use_cuda'] and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


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
    
    # 检查BNN配置
    input_dim = int(BNN_CONFIG['input_dim'])  # type: ignore
    if input_dim <= 0:
        print("错误：BNN输入维度必须大于0")
        return False
    
    hidden_dims = typing.cast(typing.List[int], BNN_CONFIG['hidden_dims'])  # type: ignore
    if not hidden_dims:
        print("错误：BNN隐藏层不能为空")
        return False
    
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
    price_levels = typing.cast(typing.List[int], ENV_CONFIG['price_levels'])  # type: ignore
    
    if initial_inventory <= 0:
        print("错误：初始库存必须大于0")
        return False
    
    if len(price_levels) < 2:
        print("错误：至少需要2个价格档位")
        return False
    
    return True

# 初始化配置
# 获取项目根目录，用于构建所有相对路径的基准
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

setup_directories()
if not validate_config():
    print("配置验证失败，请检查配置文件")
    exit(1)