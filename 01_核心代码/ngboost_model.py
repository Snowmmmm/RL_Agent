"""
NGBoost模型模块

本模块实现了基于NGBoost的概率预测系统，用于酒店预订需求预测。
NGBoost（Natural Gradient Boosting）是一种基于自然梯度提升的概率预测方法，
能够提供预测分布的完整不确定性估计。

主要功能：
- NGBoost训练器：支持多种分布和评分规则
- 概率预测：输出预测分布的均值和方差
- 模型保存/加载：支持模型持久化
- 增量学习：通过数据合并实现模型更新（适合小数据）

与BNN相比的优势：
- 训练速度更快，不需要复杂的变分推理
- 预测稳定性更好，不依赖蒙特卡洛采样
- 更容易调优，超参数更少
- 天然支持概率预测和不确定性量化
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 标准库导入
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
from ngboost.scores import LogScore, CRPScore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import joblib
from scipy import stats
from scipy.stats import uniform, norm
from datetime import datetime


class NGBoostTrainer:
    """
    NGBoost训练器 - 概率预测模型
    
    基于NGBoost的概率预测训练器，支持多种分布和评分规则。
    提供完整的训练、预测、模型保存和增量更新功能。
    
    主要特性：
    - 支持正态分布和对数正态分布
    - 支持对数评分和CRPS评分规则
    - 提供预测分布的均值和方差
    - 支持模型保存和加载
    - 支持通过数据合并实现增量更新（适合小数据）
    - 集成特征标准化
    
    Attributes:
        distribution (str): 预测分布类型 ('normal' 或 'lognormal')
        score (str): 评分规则 ('logscore' 或 'crps')
        n_estimators (int): 提升迭代次数
        learning_rate (float): 学习率
        feature_scaler (StandardScaler): 特征标准化器
        model (NGBRegressor): NGBoost模型实例
        train_losses (List[float]): 训练损失历史
        val_losses (List[float]): 验证损失历史
    """
    
    def __init__(self, 
                 distribution: str = 'normal',
                 score: str = 'logscore', 
                 n_estimators: int = 500,
                 learning_rate: float = 0.01,
                 max_depth: int = 6,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 20,
                 subsample: float = 0.8,
                 colsample_bytree: float = 1.0,
                 verbose: bool = False,
                 random_state: Optional[int] = None):
        """
        初始化NGBoost训练器
        
        Args:
            distribution: 预测分布类型 ('normal' 或 'lognormal')
            score: 评分规则 ('logscore' 或 'crps')
            n_estimators: 提升迭代次数
            learning_rate: 学习率
            max_depth: 决策树最大深度
            min_samples_split: 分裂最小样本数
            min_samples_leaf: 叶节点最小样本数
            subsample: 子采样比例（用于 minibatch_frac）
            colsample: 特征采样比例
            verbose: 是否显示详细输出
            random_state: 随机种子
        """
        self.distribution = distribution
        self.score = score
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.verbose = verbose
        self.random_state = random_state
        
        # 特征标准化器
        self.feature_scaler = StandardScaler()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        # 创建NGBoost模型
        self._create_model()
        
    def _create_model(self) -> None:
        """
        创建NGBoost模型
        
        根据配置的分布和评分规则创建NGBoost回归器。
        """
        # 选择分布
        if self.distribution == 'normal':
            dist = Normal
        elif self.distribution == 'lognormal':
            dist = LogNormal
        else:
            raise ValueError(f"不支持的分布类型: {self.distribution}")
        
        # 选择评分规则
        if self.score == 'logscore':
            score_rule = LogScore
        elif self.score == 'crps':
            score_rule = CRPScore
        else:
            raise ValueError(f"不支持的评分规则: {self.score}")
        
        # 创建基础学习器
        base_learner_params = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }
        if self.random_state is not None:
            base_learner_params['random_state'] = self.random_state
        
        base_learner = DecisionTreeRegressor(**base_learner_params)
        
        # 创建模型
        model_params = {
            'Dist': dist,
            'Score': score_rule,
            'Base': base_learner,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'minibatch_frac': self.subsample,  # ✅ 修正：NGBoost 使用 minibatch_frac
            'col_sample': self.colsample_bytree,  # ✅ 添加特征采样比例
            'verbose': self.verbose  # ✅ 使用配置的verbose参数
        }
        if self.random_state is not None:
            model_params['random_state'] = self.random_state
        
        self.model = NGBRegressor(**model_params)
        
    def train(self, 
              train_features: np.ndarray, 
              train_targets: np.ndarray,
              val_features: Optional[np.ndarray] = None,
              val_targets: Optional[np.ndarray] = None,
              early_stopping_rounds: Optional[int] = 30,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        训练NGBoost模型
        
        Args:
            train_features: 训练特征数据 (n_samples, n_features)
            train_targets: 训练目标数据 (n_samples,)
            val_features: 验证特征数据（可选）
            val_targets: 验证目标数据（可选）
            early_stopping_rounds: 早停轮数（可选）
            save_path: 模型保存路径（可选）
            
        Returns:
            Dict[str, Any]: 训练结果统计信息
        """
        # 输入检查
        if self.distribution == 'lognormal' and np.any(train_targets <= 0):
            raise ValueError("LogNormal 分布要求目标值必须 > 0")
        
        print(f"开始训练NGBoost模型...")
        print(f"训练样本数: {len(train_features)}")
        if val_features is not None:
            print(f"验证样本数: {len(val_features)}")
        
        # 特征标准化
        train_features_scaled = self.feature_scaler.fit_transform(train_features)
        
        # 准备验证数据
        val_features_scaled = None
        if val_features is not None:
            val_features_scaled = self.feature_scaler.transform(val_features)
        
        # 训练模型
        if val_features_scaled is not None:
            self.model.fit(
                X=train_features_scaled,
                Y=train_targets,
                X_val=val_features_scaled,
                Y_val=val_targets,
                early_stopping_rounds=early_stopping_rounds
            )
        else:
            self.model.fit(
                X=train_features_scaled,
                Y=train_targets
            )
        
        # 获取训练历史（兼容不同版本的 ngboost）
        evals_result = getattr(self.model, 'evals_result_', getattr(self.model, 'evals_result', None))
        
        if evals_result:
            # 获取训练损失（取第一个指标）
            train_metrics = evals_result.get('train', {})
            if train_metrics:
                score_key = list(train_metrics.keys())[0]
                self.train_losses = train_metrics[score_key]
            
            # 获取验证损失
            val_metrics = evals_result.get('val', {})
            if val_metrics and score_key in val_metrics:
                self.val_losses = val_metrics[score_key]
        else:
            # 如果无法获取，用迭代次数填充（仅用于显示）
            best_iter = len(self.model.models) if hasattr(self.model, 'models') else self.n_estimators
            self.train_losses = [0.0] * best_iter
            self.val_losses = [0.0] * best_iter
        
        # 计算训练统计信息
        train_stats = self._calculate_train_stats(train_features, train_targets)
        
        print(f"训练完成！")
        print(f"最佳迭代轮数: {len(self.train_losses)}")
        if self.train_losses:
            print(f"最终训练损失: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"最终验证损失: {self.val_losses[-1]:.4f}")
        
        # 保存模型
        if save_path is not None:
            self.save_model(save_path)
        
        return train_stats
    
    def _calculate_train_stats(self, train_features: np.ndarray, train_targets: np.ndarray) -> Dict[str, Any]:
        """
        计算训练统计信息（包含概率评估指标）
        """
        mean_pred, var_pred = self.predict(train_features)
        std_pred = np.sqrt(var_pred)
        
        # 传统回归指标
        mse = mean_squared_error(train_targets, mean_pred)
        mae = mean_absolute_error(train_targets, mean_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((train_targets - mean_pred) / (train_targets + 1e-8))) * 100
        
        # 概率评估指标
        prob_metrics = self._calculate_probability_metrics(train_features, train_targets)
        
        return {
            # 传统回归指标
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            # 概率评估指标
            'log_likelihood': prob_metrics['log_likelihood'],
            'crps': prob_metrics['crps'],
            'pit_mean': prob_metrics['pit_mean'],
            'pit_std': prob_metrics['pit_std'],
            'pit_uniform_test': prob_metrics['pit_uniform_test'],
            # 预测分布统计
            'mean_prediction': float(np.mean(mean_pred)),
            'std_prediction': float(np.std(mean_pred)),
            'mean_variance': float(np.mean(var_pred)),
            'mean_target': float(np.mean(train_targets)),
            'std_target': float(np.std(train_targets))
        }
    
    def predict(self, features: np.ndarray, record_env_changes: bool = False, 
                user_type: str = None, demand_type: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测 - 输出预测分布的均值和方差，支持环境变化记录
        
        Args:
            features: 特征数据，包含7天时序特征
            record_env_changes: 是否记录环境变化
            user_type: 用户类型（'online'或'offline'）
            demand_type: 需求类型（'booked'或'actual'）
            
        Returns:
            (均值预测, 方差预测)
        """
        features_scaled = self.feature_scaler.transform(features)
        pred_dist = self.model.pred_dist(features_scaled)
        mean_pred = pred_dist.mean()
        var_pred = pred_dist.var  # NGBoost分布对象的var是属性，不是方法
        
        # 记录环境变化（如果启用）
        if record_env_changes and user_type and demand_type:
            self._record_prediction_environment_changes(
                features, mean_pred, var_pred, user_type, demand_type
            )
        
        return mean_pred, var_pred
    
    def _record_prediction_environment_changes(self, features: np.ndarray, mean_pred: np.ndarray, 
                                             var_pred: np.ndarray, user_type: str, demand_type: str) -> None:
        """
        记录预测器产生的环境变化
        
        Args:
            features: 原始特征数据（包含7天时序特征）
            mean_pred: 均值预测结果
            var_pred: 方差预测结果
            user_type: 用户类型
            demand_type: 需求类型
        """
        # 初始化环境变化记录（如果不存在）
        if not hasattr(self, '_environment_changes'):
            self._environment_changes = []
        
        # 提取时序特征信息
        feature_info = {
            'timestamp': datetime.now().isoformat(),
            'user_type': user_type,
            'demand_type': demand_type,
            'feature_shape': features.shape,
            'mean_prediction': float(np.mean(mean_pred)),
            'std_prediction': float(np.std(mean_pred)),
            'mean_variance': float(np.mean(var_pred)),
            'prediction_range': {
                'min': float(np.min(mean_pred)),
                'max': float(np.max(mean_pred)),
                'q25': float(np.percentile(mean_pred, 25)),
                'q50': float(np.percentile(mean_pred, 50)),
                'q75': float(np.percentile(mean_pred, 75))
            },
            'variance_range': {
                'min': float(np.min(var_pred)),
                'max': float(np.max(var_pred)),
                'mean': float(np.mean(var_pred))
            }
        }
        
        # 分析7天时序特征的变化
        if features.shape[1] >= 7:  # 至少有7个时序特征
            feature_info['temporal_features'] = {
                'day_1_mean': float(np.mean(features[:, 0])),
                'day_7_mean': float(np.mean(features[:, 6])),
                'temporal_trend': float(np.mean(features[:, 6]) - np.mean(features[:, 0])),
                'feature_variability': float(np.std(features.flatten()))
            }
        
        # 记录环境变化
        self._environment_changes.append(feature_info)
        
        # 保持记录大小合理（最多保留最近1000条记录）
        if len(self._environment_changes) > 1000:
            self._environment_changes = self._environment_changes[-1000:]
    
    def get_environment_changes(self, recent_n: int = None) -> List[Dict]:
        """
        获取环境变化记录
        
        Args:
            recent_n: 获取最近N条记录，None表示获取全部
            
        Returns:
            环境变化记录列表
        """
        if not hasattr(self, '_environment_changes'):
            return []
        
        if recent_n is None:
            return self._environment_changes
        else:
            return self._environment_changes[-recent_n:]
    
    def predict_single(self, feature_vector: Union[np.ndarray, List], record_env_changes: bool = False, 
                      user_type: str = None, demand_type: str = None) -> Tuple[float, float]:
        """
        单样本预测，支持环境变化记录
        """
        if isinstance(feature_vector, list):
            feature_vector = np.array(feature_vector)
        mean_pred, var_pred = self.predict(feature_vector.reshape(1, -1), record_env_changes, user_type, demand_type)
        return float(mean_pred[0]), float(var_pred[0])
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'distribution': self.distribution,
                'score': self.score,
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'subsample': self.subsample,
                'random_state': self.random_state
            }
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到：{filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载模型
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.train_losses = model_data.get('train_losses', [])
        self.val_losses = model_data.get('val_losses', [])
        config = model_data.get('config', {})
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        print(f"模型已从 {filepath} 加载")
    
    def incremental_update(self, 
                          new_features: np.ndarray, 
                          new_targets: np.ndarray,
                          old_features: Optional[np.ndarray] = None,
                          old_targets: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        增量更新模型（通过合并新旧数据重新训练）
        
        注意：NGBoost 不支持原生增量学习，此处采用数据合并策略。
        若未提供旧数据，则仅用新数据训练（不推荐）。
        
        Args:
            new_features: 新数据特征
            new_targets: 新数据目标
            old_features: 原始训练特征（可选，但推荐提供）
            old_targets: 原始训练目标（可选，但推荐提供）
        """
        if old_features is None or old_targets is None:
            print("警告：未提供旧数据，将仅使用新数据重新训练模型。")
            combined_features = new_features
            combined_targets = new_targets
        else:
            combined_features = np.vstack([old_features, new_features])
            combined_targets = np.hstack([old_targets, new_targets])
        
        print(f"增量更新：合并后总样本数 = {len(combined_features)}")
        return self.train(
            train_features=combined_features,
            train_targets=combined_targets,
            save_path=save_path
        )
    
    def predict_distribution(self, features: np.ndarray) -> Any:
        """
        获取完整的预测分布对象
        
        Returns:
            NGBoost分布对象，支持更多统计功能
        """
        features_scaled = self.feature_scaler.transform(features)
        return self.model.pred_dist(features_scaled)
    
    def predict_quantile(self, features: np.ndarray, quantile: float) -> np.ndarray:
        """
        预测任意分位数
        
        Args:
            features: 特征数据
            quantile: 分位数 (0-1之间)
            
        Returns:
            指定分位数的预测值
        """
        pred_dist = self.predict_distribution(features)
        return pred_dist.ppf(quantile)
    
    def predict_interval(self, features: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测置信区间
        
        Args:
            features: 特征数据
            confidence: 置信水平 (默认95%)
            
        Returns:
            (下限, 上限)
        """
        alpha = (1 - confidence) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha
        
        pred_dist = self.predict_distribution(features)
        lower_bound = pred_dist.ppf(lower_quantile)
        upper_bound = pred_dist.ppf(upper_quantile)
        
        return lower_bound, upper_bound
    
    def _calculate_probability_metrics(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        计算概率评估指标：对数似然、CRPS、PIT
        
        Args:
            features: 特征数据
            targets: 真实目标值
            
        Returns:
            包含log_likelihood, crps, pit相关指标的字典
        """
        # 获取预测分布
        pred_dist = self.predict_distribution(features)
        mean_pred = pred_dist.mean()
        std_pred = np.sqrt(pred_dist.var)
        
        # 1. 对数似然 (Log Likelihood)
        try:
            log_likelihood = np.mean(pred_dist.logpdf(targets))
        except:
            # 如果分布对象没有logpdf方法，使用数值计算
            log_likelihood = -np.inf
        
        # 2. 连续秩概率得分 (CRPS)
        crps_scores = []
        for i in range(len(targets)):
            try:
                # 使用scipy的crps计算（如果可用）
                from scipy.stats import norm as scipy_norm
                if self.distribution == 'normal':
                    crps = self._crps_normal(targets[i], mean_pred[i], std_pred[i])
                elif self.distribution == 'lognormal':
                    # 对数正态分布的CRPS计算较复杂，这里简化处理
                    crps = np.abs(targets[i] - mean_pred[i])  # 简化的CRPS近似
                else:
                    crps = np.abs(targets[i] - mean_pred[i])
                crps_scores.append(crps)
            except:
                crps_scores.append(np.abs(targets[i] - mean_pred[i]))
        
        crps = np.mean(crps_scores)
        
        # 3. 概率积分变换 (PIT)
        pit_values = pred_dist.cdf(targets)
        pit_mean = np.mean(pit_values)
        pit_std = np.std(pit_values)
        
        # PIT均匀性检验（使用Kolmogorov-Smirnov检验）
        try:
            # 检验PIT值是否服从均匀分布[0,1]
            ks_stat, ks_pvalue = stats.kstest(pit_values, 'uniform')
            pit_uniform_test = {'ks_statistic': float(ks_stat), 'p_value': float(ks_pvalue)}
        except:
            pit_uniform_test = {'ks_statistic': np.nan, 'p_value': np.nan}
        
        return {
            'log_likelihood': float(log_likelihood),
            'crps': float(crps),
            'pit_mean': float(pit_mean),
            'pit_std': float(pit_std),
            'pit_uniform_test': pit_uniform_test
        }
    
    def _crps_normal(self, observation: float, mean: float, std: float) -> float:
        """
        计算正态分布的CRPS得分
        
        Args:
            observation: 观测值
            mean: 预测均值
            std: 预测标准差
            
        Returns:
            CRPS得分
        """
        # 标准化观测值
        z = (observation - mean) / std
        # 标准正态分布的CDF和PDF
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        # CRPS公式
        crps = std * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
        return crps
    
    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能（包含完整的概率评估指标）
        """
        mean_pred, var_pred = self.predict(features)
        std_pred = np.sqrt(var_pred)
        
        # 传统回归指标
        mse = mean_squared_error(targets, mean_pred)
        mae = mean_absolute_error(targets, mean_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets - mean_pred) / (targets + 1e-8))) * 100
        
        # 95% 预测区间 - 使用分布特定的分位数计算
        lower_bound, upper_bound = self.predict_interval(features, confidence=0.95)
        coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
        avg_width = np.mean(upper_bound - lower_bound)
        
        # 概率评估指标
        prob_metrics = self._calculate_probability_metrics(features, targets)
        
        return {
            # 传统回归指标
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            # 预测区间指标
            'coverage_95': float(coverage),
            'avg_width_95': float(avg_width),
            # 概率评估指标
            'log_likelihood': prob_metrics['log_likelihood'],
            'crps': prob_metrics['crps'],
            'pit_mean': prob_metrics['pit_mean'],
            'pit_std': prob_metrics['pit_std'],
            'pit_ks_statistic': prob_metrics['pit_uniform_test']['ks_statistic'],
            'pit_ks_p_value': prob_metrics['pit_uniform_test']['p_value'],
            # 预测分布统计
            'mean_prediction': float(np.mean(mean_pred)),
            'std_prediction': float(np.std(mean_pred)),
            'mean_variance': float(np.mean(var_pred)),
            'mean_target': float(np.mean(targets)),
            'std_target': float(np.std(targets))
        }