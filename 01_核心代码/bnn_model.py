"""
贝叶斯神经网络模型模块

本模块实现了完整的贝叶斯神经网络系统，用于酒店预订需求的概率预测。
包含贝叶斯线性层、深度贝叶斯网络、训练器和数据集封装等核心组件。

主要功能：
- 贝叶斯线性层：实现权重的概率分布建模
- 贝叶斯神经网络：双输出头设计，预测均值和方差
- BNN训练器：支持变分推理和增量学习
- 数据集封装：酒店数据集的标准化处理
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from typing import Optional, Dict, List, Any, Tuple, Union
import warnings

class BayesianLinear(nn.Module):
    """
    贝叶斯线性层 - 修复KL散度计算
    
    实现了变分推理中的贝叶斯线性层，为权重和偏置学习概率分布而非固定值。
    该层使用重参数化技巧进行高效的梯度下降训练，并提供KL散度计算用于变分损失。
    
    主要特性：
    - 权重和偏置都服从正态分布（均值和方差参数化）
    - 支持前向传播时采样权重（训练）或使用均值（推理）
    - 提供KL散度计算，用于衡量学习分布与先验分布的差异
    - 使用softplus函数确保方差为正，提高数值稳定性
    
    Attributes:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        prior_mean (float): 先验分布的均值
        prior_std (float): 先验分布的标准差
        weight_mu (nn.Parameter): 权重的均值参数
        weight_rho (nn.Parameter): 权重的方差参数（通过softplus转换为标准差）
        bias_mu (nn.Parameter): 偏置的均值参数
        bias_rho (nn.Parameter): 偏置的方差参数（通过softplus转换为标准差）
    """
    
    def __init__(self, in_features: int, out_features: int, prior_mean: float = 0.0, prior_std: float = 1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # 权重参数（变分参数）
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # 初始化
        self.reset_parameters()
        
    def reset_parameters(self):
        # 初始化变分参数 - 使用更小的初始化值
        nn.init.normal_(self.weight_mu, self.prior_mean, 0.01)  # 减小初始化方差
        nn.init.normal_(self.weight_rho, -5, 0.1)  # 初始化为更小的值
        nn.init.normal_(self.bias_mu, self.prior_mean, 0.01)  # 减小初始化方差
        nn.init.normal_(self.bias_rho, -5, 0.1)
    
    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        """
        前向传播 - 贝叶斯线性层的前向计算
        
        实现贝叶斯线性层的前向传播，支持训练和推理两种模式。
        训练时从权重分布中采样，推理时使用权重均值，提供不确定性估计。
        
        Args:
            x (torch.Tensor): 输入张量，形状为[batch_size, in_features]
            sample_noise (bool, optional): 是否采样噪声，True表示训练模式，False表示推理模式
            
        Returns:
            torch.Tensor: 输出张量，形状为[batch_size, out_features]
            
        Note:
            - 训练或sample_noise=True时，从q(w|θ)分布中采样权重
            - 推理且sample_noise=False时，使用权重均值进行确定性计算
            - 使用重参数化技巧实现梯度回传
            - 通过log1p(exp(ρ))计算标准差，确保数值稳定性
        """
        if self.training or sample_noise:
            # 采样权重和偏置
            weight_epsilon = torch.randn_like(self.weight_mu)
            bias_epsilon = torch.randn_like(self.bias_mu)
            
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            
            weight = self.weight_mu + weight_sigma * weight_epsilon
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # 测试时使用均值
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度 - 使用更稳定的计算方式
        
        计算变分后验分布q(w|θ)与先验分布p(w)之间的KL散度，
        用于变分损失函数中的正则化项。
        
        KL(q||p) = ∫ q(w|θ) log(q(w|θ)/p(w)) dw
        
        对于正态分布，KL散度有解析解：
        KL(q||p) = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 0.5
        
        Returns:
            torch.Tensor: KL散度值，标量张量
            
        Note:
            - 使用softplus函数确保标准差为正，提高数值稳定性
            - 添加最小值保护(1e-8)避免除零错误
            - 分别计算权重和偏置的KL散度并求和
        """
        # 使用更稳定的softplus计算
        weight_sigma = F.softplus(self.weight_rho) + 1e-8
        bias_sigma = F.softplus(self.bias_rho) + 1e-8
        
        # KL(q||p) = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2)/(2*sigma_p^2) - 0.5
        kl_weight = torch.log(self.prior_std / weight_sigma) + \
                   (weight_sigma**2 + (self.weight_mu - self.prior_mean)**2) / (2 * self.prior_std**2) - 0.5
        kl_bias = torch.log(self.prior_std / bias_sigma) + \
                 (bias_sigma**2 + (self.bias_mu - self.prior_mean)**2) / (2 * self.prior_std**2) - 0.5
        
        return kl_weight.sum() + kl_bias.sum()

class BayesianNN(nn.Module):
    """
    贝叶斯神经网络（双输出：均值和方差）
    
    实现了深度贝叶斯神经网络，用于预测酒店需求的概率分布。
    网络输出均值和方差两个头，支持不确定性量化和置信区间估计。
    
    主要特性：
    - 多层贝叶斯线性层，每层都学习权重分布而非固定值
    - 双输出头设计：一个预测均值，一个预测方差
    - 支持跳跃连接，缓解深层网络的梯度消失问题
    - 集成Dropout层，提供额外的正则化
    - 提供完整的KL散度计算，支持变分推理
    
    网络结构：
    输入层 → 多个隐藏层（贝叶斯线性+ReLU+Dropout） → 
    ├── 均值输出头（多层）：预测需求均值
    └── 方差输出头（多层）：预测需求方差（通过softplus确保正数）
    
    Attributes:
        input_dim (int): 输入特征维度
        hidden_dims (List[int]): 隐藏层维度列表
        dropout_rate (float): Dropout比率
        hidden_layers (nn.ModuleList): 隐藏层模块列表
        skip_connections (nn.ModuleList): 跳跃连接模块列表
        mean_head (nn.Sequential): 均值预测输出头
        var_head (nn.Sequential): 方差预测输出头
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 96, 64, 48, 32], prior_mean: float = 0.0, prior_std: float = 1.0, dropout_rate: float = 0.1):
        super(BayesianNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 构建更深的网络层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_mean, prior_std))
            layers.append(nn.ReLU())
            
            # 添加Dropout层（除了最后一层隐藏层）
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            # 移除批归一化层，避免维度问题
            # layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # 添加跳跃连接的路径
        self.skip_connections = nn.ModuleList()
        if len(hidden_dims) >= 3:
            # 为深层网络添加跳跃连接
            for i in range(1, len(hidden_dims)):
                self.skip_connections.append(
                    BayesianLinear(hidden_dims[i-1], hidden_dims[i], prior_mean, prior_std)
                )
        
        # 均值输出头（多层）
        self.mean_head = nn.Sequential(
            BayesianLinear(prev_dim, prev_dim // 2, prior_mean, prior_std),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            BayesianLinear(prev_dim // 2, 1, prior_mean, prior_std)
        )
        
        # 方差输出头（多层）
        self.var_head = nn.Sequential(
            BayesianLinear(prev_dim, prev_dim // 2, prior_mean, prior_std),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            BayesianLinear(prev_dim // 2, 1, prior_mean, prior_std)
        )
        
    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 贝叶斯神经网络的前向计算
        
        实现贝叶斯神经网络的完整前向传播，包括隐藏层计算、跳跃连接、
        双输出头（均值和方差）预测。支持训练和推理模式。
        
        Args:
            x (torch.Tensor): 输入特征张量，形状为[batch_size, input_dim]
            sample_noise (bool, optional): 是否采样噪声，True表示训练模式，False表示推理模式
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (均值预测, 方差预测)，形状均为[batch_size, 1]
            
        Note:
            - 使用跳跃连接缓解深层网络梯度消失问题
            - 双输出头设计：一个预测均值，一个预测方差
            - 方差通过softplus激活确保为正，并添加最小值保护
            - 保存中间激活值用于跳跃连接计算
        """
        # 保存中间激活值用于跳跃连接
        intermediate_activations = []
        current_x = x
        layer_idx = 0
        skip_idx = 0
        
        # 前向传播通过隐藏层
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, BayesianLinear):
                # 应用贝叶斯线性层
                current_x = layer(current_x, sample_noise)
                
                # 保存激活值（在ReLU之前）
                if i < len(self.hidden_layers) - 1:  # 不是最后一层
                    intermediate_activations.append(current_x.clone())
                    
            elif isinstance(layer, nn.ReLU):
                current_x = layer(current_x)
                
                # 应用跳跃连接（如果有）
                if len(self.skip_connections) > 0 and layer_idx > 0 and layer_idx < len(self.skip_connections):
                    # 获取对应的跳跃连接
                    skip_x = intermediate_activations[layer_idx - 1]
                    skip_transform = self.skip_connections[layer_idx - 1]
                    skip_output = skip_transform(skip_x, sample_noise)
                    
                    # 调整维度后相加
                    if current_x.shape == skip_output.shape:
                        current_x = current_x + skip_output
                
                layer_idx += 1
                
            elif isinstance(layer, nn.BatchNorm1d):
                current_x = layer(current_x)
            elif isinstance(layer, nn.Dropout):
                current_x = layer(current_x)
        
        x = current_x
        
        # 输出均值和方差（使用多层输出头）
        mean = self.mean_head(x)
        var = self.var_head(x)
        
        # 输出层不再使用softplus激活，让模型学习真实的数据范围
        # 只确保方差为正即可
        var = F.softplus(var) + 1e-6    # 最小值保护
        
        return mean, var
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算总的KL散度 - 修复重复计算问题
        
        计算整个贝叶斯神经网络所有贝叶斯层的KL散度总和，
        用于变分损失函数中的正则化项。
        
        避免重复计算Sequential模块中的子模块，只计算顶层的贝叶斯线性层。
        
        Returns:
            torch.Tensor: 所有贝叶斯层的KL散度总和，标量张量
            
        Note:
            - 包括隐藏层、跳跃连接、输出头的KL散度
            - 使用named_modules避免重复计算Sequential中的模块
            - 通过条件'.' not in name只选择顶层模块
            - KL散度用于衡量学习分布与先验分布的差异
        """
        kl = torch.tensor(0.0)
        
        # 隐藏层的KL散度
        for layer in self.hidden_layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        
        # 跳跃连接的KL散度
        for layer in self.skip_connections:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        
        # 输出头的KL散度（避免重复计算Sequential中的模块）
        for name, module in self.mean_head.named_modules():
            if isinstance(module, BayesianLinear) and '.' not in name:  # 只计算顶层模块
                kl += module.kl_divergence()
                
        for name, module in self.var_head.named_modules():
            if isinstance(module, BayesianLinear) and '.' not in name:  # 只计算顶层模块
                kl += module.kl_divergence()
        
        return kl

class HotelDataset(Dataset):
    """
    酒店数据集
    
    PyTorch数据集类，用于封装酒店特征数据和目标数据，
    支持DataLoader的批量加载和随机打乱。
    
    主要功能：
    - 将numpy数组转换为PyTorch张量
    - 提供数据长度信息
    - 支持索引访问，返回特征和目标对
    
    Attributes:
        features (torch.FloatTensor): 特征数据，形状为[n_samples, n_features]
        targets (torch.FloatTensor): 目标数据，形状为[n_samples, 1]
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class BNNTrainer:
    """
    BNN训练器 - 适配深层网络
    
    贝叶斯神经网络的训练器，实现了变分推理的训练过程。
    支持ELBO损失函数、KL散度退火、早停机制、学习率调度等功能。
    
    主要特性：
    - 使用AdamW优化器，支持权重衰减正则化
    - 实现ELBO（Evidence Lower Bound）损失函数
    - 支持KL散度退火，平衡似然项和KL项
    - 提供早停机制，防止过拟合
    - 支持学习率调度，优化训练过程
    - 记录训练历史，便于分析和可视化
    
    训练过程：
    1. 前向传播：通过贝叶斯网络得到预测分布（均值和方差）
    2. 计算损失：ELBO = NLL（负对数似然）+ β × KL（KL散度）
    3. 反向传播：更新网络参数
    4. 验证评估：在验证集上评估性能
    5. 早停判断：根据验证损失决定是否提前停止
    
    Attributes:
        device (str): 计算设备（'cuda'或'cpu'）
        model (BayesianNN): 贝叶斯神经网络模型
        optimizer (optim.AdamW): AdamW优化器
        scheduler (optim.lr_scheduler.StepLR): 学习率调度器
        train_losses (List[float]): 训练损失历史
        val_losses (List[float]): 验证损失历史
        best_model_state (Dict): 最佳模型状态字典
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 96, 64, 48, 32], prior_mean: float = 0.0, prior_std: float = 1.0, 
                 dropout_rate: float = 0.1, learning_rate: float = 1e-4, weight_decay: float = 1e-5, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.model = BayesianNN(input_dim, hidden_dims, prior_mean, prior_std, dropout_rate).to(device)
        
        # 使用更稳定的优化器设置
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
        
        # 使用更温和的学习率调度
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
        self.n_train_samples = 5000  # 默认训练样本数量，将在train()中更新
        
    def elbo_loss(self, predictions: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor, kl_div: torch.Tensor, beta: float = 1.0, n_train_samples: int = 5000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO损失函数 - 最终修复版本
        
        计算变分推理中的ELBO（Evidence Lower Bound）损失函数，
        用于训练贝叶斯神经网络。
        
        ELBO损失 = NLL（负对数似然）+ β × KL散度
        
        其中：
        - NLL衡量预测分布与真实目标的匹配程度
        - KL散度衡量变分后验与先验分布的差异
        - β是退火参数，用于平衡似然项和KL项
        
        Args:
            predictions (Tuple[torch.Tensor, torch.Tensor]): 预测结果（均值，方差）
            targets (torch.Tensor): 真实目标值
            kl_div (torch.Tensor): KL散度值
            beta (float, optional): KL散度权重，默认为1.0
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (ELBO损失, NLL损失, KL散度)
            
        Note:
            - 使用高斯分布的负对数似然
            - KL散度按训练样本数量n_train_samples缩放，平衡NLL和KL的数量级
            - 方差通过softplus确保为正，并添加最小值保护
        """
        mean_pred, var_pred = predictions
        
        # 高斯负对数似然
        nll = 0.5 * torch.log(var_pred) + 0.5 * ((targets - mean_pred)**2) / var_pred + 0.5 * torch.log(2 * torch.tensor(np.pi))
        nll = nll.mean()
        
        # KL散度按训练样本数量归一化，确保正确的贝叶斯推断
        kl_weight = beta * kl_div / n_train_samples  # 使用实际训练样本数量进行缩放
        elbo = nll + kl_weight
        
        return elbo, nll, kl_div
    
    def train_epoch(self, dataloader: DataLoader, beta: float = 1.0) -> Tuple[float, float, float]:
        """
        训练一个epoch - 单次训练循环
        
        执行一个完整的训练epoch，包括前向传播、损失计算、反向传播和参数更新。
        计算并返回ELBO损失、负对数似然(NLL)和KL散度的平均值。
        
        Args:
            dataloader (DataLoader): 训练数据加载器
            beta (float, optional): KL散度权重系数，用于退火，默认为1.0
            
        Returns:
            Tuple[float, float, float]: (平均ELBO损失, 平均NLL损失, 平均KL散度)
            
        Note:
            - 设置模型为训练模式(self.model.train())
            - 对每个批次执行完整的前向-反向传播循环
            - 累积所有批次的损失值并计算平均值
            - KL散度权重beta用于平衡似然项和先验项
        """
        self.model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0
        
        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            mean_pred, var_pred = self.model(batch_features)
            
            # 计算KL散度
            kl_div = self.model.kl_divergence()
            
            # 计算损失
            loss, nll, kl = self.elbo_loss((mean_pred, var_pred), batch_targets, kl_div, beta, self.n_train_samples)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_nll += nll.item()
            total_kl += kl.item()
        
        return total_loss / len(dataloader), total_nll / len(dataloader), total_kl / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        验证模型 - 在验证集上评估性能
        
        在验证集上评估模型性能，计算平均ELBO损失。使用确定性推理模式
        （不采样噪声）以获得稳定的验证结果。
        
        Args:
            dataloader (DataLoader): 验证数据加载器
            
        Returns:
            float: 平均验证损失（ELBO）
            
        Note:
            - 设置模型为评估模式(self.model.eval())
            - 使用torch.no_grad()禁用梯度计算，提高推理速度
            - 前向传播时sample_noise=False，使用权重均值进行推理
            - 计算完整的ELBO损失，包括NLL和KL散度项
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # 前向传播
                mean_pred, var_pred = self.model(batch_features, sample_noise=False)
                
                # 计算KL散度
                kl_div = self.model.kl_divergence()
                
                # 计算损失 - 验证时也使用训练集大小进行KL散度归一化
                loss, _, _ = self.elbo_loss((mean_pred, var_pred), batch_targets, kl_div, 1.0, self.n_train_samples)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_features: np.ndarray, train_targets: np.ndarray, val_features: Optional[np.ndarray] = None, val_targets: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, beta_annealing: bool = True, save_path: Optional[str] = None) -> None:
        """
        训练模型
        
        完整的贝叶斯神经网络训练过程，包括数据准备、训练循环、验证评估、
        早停机制、模型保存等功能。
        
        训练流程：
        1. 创建数据加载器，支持批量训练
        2. 每个epoch进行KL散度退火，平衡似然项和KL项
        3. 训练阶段：前向传播、损失计算、反向传播、参数更新
        4. 验证阶段：在验证集上评估模型性能
        5. 早停判断：根据验证损失决定是否提前停止
        6. 模型保存：保存验证损失最低的模型
        
        Args:
            train_features (np.ndarray): 训练特征数据
            train_targets (np.ndarray): 训练目标数据
            val_features (Optional[np.ndarray], optional): 验证特征数据
            val_targets (Optional[np.ndarray], optional): 验证目标数据
            epochs (int, optional): 训练轮数，默认为100
            batch_size (int, optional): 批次大小，默认为32
            beta_annealing (bool, optional): 是否使用KL散度退火，默认为True
            save_path (Optional[str], optional): 模型保存路径
            
        Returns:
            None
            
        Note:
            - 支持KL散度退火，前50个epoch线性增加到1
            - 使用早停机制，patience为10个epoch
            - 每10个epoch打印训练进度
            - 自动保存验证损失最低的模型
        """
        
        # 保存训练样本数量用于KL散度归一化
        self.n_train_samples = len(train_features)
        
        # 创建数据加载器
        train_dataset = HotelDataset(train_features, train_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataloader = None
        if val_features is not None and val_targets is not None:
            val_dataset = HotelDataset(val_features, val_targets)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"开始训练，设备：{self.device}")
        print(f"训练样本数：{len(train_features)}")
        if val_features is not None:
            print(f"验证样本数：{len(val_features)}")
        
        for epoch in range(epochs):
            # Beta退火
            if beta_annealing:
                beta = min(1.0, epoch / 50)  # 前50个epoch线性增加到1
            else:
                beta = 1.0
            
            # 训练
            train_loss, train_nll, train_kl = self.train_epoch(train_dataloader, beta)
            
            # 验证
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型状态
                    self.best_model_state = self.model.state_dict().copy()
                    
                    # 保存最佳模型
                    if save_path is not None:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"早停于第{epoch+1}轮")
                    break
            
            self.train_losses.append(train_loss)
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Beta: {beta:.3f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                          f"Beta: {beta:.3f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        print("训练完成！")
        
        # 如果没有验证集，直接保存最终模型
        if save_path is not None and val_dataloader is None:
            self.save_model(save_path)
    
    def predict(self, features: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测 - 蒙特卡洛采样预测
        
        使用蒙特卡洛采样方法进行概率预测，通过多次前向传播采样
        来估计预测分布的均值和方差。支持批量样本预测。
        
        Args:
            features (np.ndarray): 输入特征数组，形状为[n_samples, n_features]
            n_samples (int, optional): 蒙特卡洛采样次数，默认为100
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (均值预测, 方差预测)，形状均为[n_samples, 1]
            
        Note:
            - 使用蒙特卡洛采样估计预测不确定性
            - 总方差 = 平均方差 + 方差的方差（认知不确定性+偶然不确定性）
            - 设置模型为评估模式，禁用梯度计算
            - 采样时启用噪声(sample_noise=True)以获得不同的网络输出
        """
        self.model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            mean_samples = []
            var_samples = []
            
            for _ in range(n_samples):
                mean, var = self.model(features_tensor, sample_noise=True)
                mean_samples.append(mean.cpu().numpy())
                var_samples.append(var.cpu().numpy())
            
            # 计算均值和方差的统计量
            mean_pred = np.mean(mean_samples, axis=0)
            var_pred = np.mean(var_samples, axis=0) + np.var(mean_samples, axis=0)
            
            return mean_pred, var_pred
    
    def predict_single(self, feature_vector: Union[np.ndarray, torch.Tensor], n_samples: int = 100) -> Tuple[float, float]:
        """
        单样本预测 - 便捷的单样本预测接口
        
        对单个输入样本进行概率预测，返回预测的均值和方差。
        内部调用predict方法，自动处理输入格式转换。
        
        Args:
            feature_vector (Union[np.ndarray, torch.Tensor]): 单个样本的特征向量
            n_samples (int, optional): 蒙特卡洛采样次数，默认为100
            
        Returns:
            Tuple[float, float]: (均值预测, 方差预测)
            
        Note:
            - 自动处理PyTorch张量和NumPy数组的输入
            - 将输入重塑为[1, n_features]形状以适配predict方法
            - 返回Python原生float类型，便于后续使用
            - 使用与predict方法相同的蒙特卡洛采样策略
        """
        if isinstance(feature_vector, torch.Tensor):
            feature_vector = feature_vector.cpu().numpy()
        
        mean_pred, var_pred = self.predict(feature_vector.reshape(1, -1), n_samples)
        
        return float(mean_pred[0]), float(var_pred[0])
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型 - 保存完整的训练状态和配置
        
        保存模型的完整状态，包括网络权重、优化器状态、训练进度和模型配置。
        支持断点续训和模型迁移。
        
        Args:
            filepath (str): 模型保存路径
            
        Note:
            - 保存模型状态字典(model.state_dict())
            - 保存优化器状态(optimizer.state_dict())
            - 保存训练进度(current_epoch)
            - 保存最佳损失(best_loss)
            - 保存模型配置(model_config)
            - 使用torch.save()进行序列化保存
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"模型已保存到：{filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载模型 - 从检查点恢复完整的训练状态
        
        从保存的检查点文件中恢复模型状态，包括网络权重、优化器状态、
        训练进度和最佳损失记录。支持断点续训和模型迁移。
        
        Args:
            filepath (str): 模型文件路径
            
        Note:
            - 使用torch.load()加载检查点文件
            - 自动处理设备映射(map_location=self.device)
            - 恢复模型状态(model.load_state_dict())
            - 恢复优化器状态(optimizer.load_state_dict())
            - 恢复学习率调度器状态(scheduler.load_state_dict())
            - 恢复训练历史(train_losses, val_losses)
            - 检查点文件应包含save_model方法保存的所有键
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"模型已从{filepath}加载")
    
    def incremental_update(self, new_features: np.ndarray, new_targets: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        """
        增量更新模型 - 使用新数据更新预训练模型
        
        在保持模型已有知识的基础上，使用新收集的数据对模型进行增量更新。
        通过选择性冻结部分网络层来防止灾难性遗忘，同时允许模型学习新数据的特征。
        
        Args:
            new_features (np.ndarray): 新样本的特征数据
            new_targets (np.ndarray): 新样本的目标值
            epochs (int, optional): 增量训练的轮数，默认为10
            batch_size (int, optional): 批次大小，默认为32
            
        Note:
            - 选择性冻结第一层隐藏层，保持底层特征提取能力
            - 使用与主训练相同的ELBO损失函数和KL散度约束
            - 训练完成后重新启用所有层的梯度
            - 适用于在线学习和数据持续到达的场景
            - 打印训练进度和最终完成信息
        """
        print(f"开始增量更新，新样本数：{len(new_features)}")
        
        # 冻结部分层（根据需求调整）
        for name, param in self.model.named_parameters():
            if 'hidden_layers.0' in name:  # 冻结第一层
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 训练新数据
        new_dataset = HotelDataset(new_features, new_targets)
        new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_features, batch_targets in new_dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                mean_pred, var_pred = self.model(batch_features)
                
                # 计算KL散度
                kl_div = self.model.kl_divergence()
                
                # 计算损失
                loss, _, _ = self.elbo_loss((mean_pred, var_pred), batch_targets, kl_div, 1.0, self.n_train_samples)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                print(f"增量更新 Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(new_dataloader):.4f}")
        
        # 重新启用所有层的梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        print("增量更新完成！")
