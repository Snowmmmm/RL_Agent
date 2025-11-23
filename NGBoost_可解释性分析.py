#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NGBoost模型可解释性分析脚本

本脚本用于分析NGBoost模型的可解释性，包括：
1. 特征重要性分析
2. SHAP值分析（全局和局部）
3. 部分依赖图（PDP）分析
4. 模型预测分布分析
5. 特征对分布参数（μ和σ）的影响分析

支持的模型：
- ngboost_model_online_booked.pkl - 线上用户预定需求模型
- ngboost_model_online_actual.pkl - 线上用户实际需求模型  
- ngboost_model_offline_booked.pkl - 线下用户预定需求模型
- ngboost_model_offline_actual.pkl - 线下用户实际需求模型
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Tuple, Any, Optional
import matplotlib
from matplotlib.font_manager import FontProperties

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

import shap
SHAP_AVAILABLE = True

class NGBoostInterpretabilityAnalyzer:
    """
    NGBoost模型可解释性分析器
    
    提供多种可解释性分析方法，帮助理解NGBoost模型的预测行为。
    """
    
    def __init__(self, models_dir: str = "02_训练模型"):
        """
        初始化分析器
        
        Args:
            models_dir: 模型文件目录路径
        """
        self.models_dir = models_dir
        self.models = {}
        self.model_names = [
            "ngboost_model_online_booked.pkl",
            "ngboost_model_online_actual.pkl", 
            "ngboost_model_offline_booked.pkl",
            "ngboost_model_offline_actual.pkl"
        ]
        
        # 特征名称（根据data_preprocessing.py中的prepare_ngboost_features方法）
        self.feature_names = [
            'price', 'is_weekend', 'season', 'price_cv', 'demand_trend', 'price_trend'
        ]
        
        # 更直观的特征名称用于显示
        self.short_feature_names = [
            '价格', '是否周末', '季节', '价格变异系数', '需求趋势', '价格趋势'
        ]
        
        # 模型描述
        self.model_descriptions = {
            "ngboost_model_online_booked.pkl": "线上用户-预定需求模型",
            "ngboost_model_online_actual.pkl": "线上用户-实际需求模型", 
            "ngboost_model_offline_booked.pkl": "线下用户-预定需求模型",
            "ngboost_model_offline_actual.pkl": "线下用户-实际需求模型"
        }
        
        # 创建输出目录
        self.output_dir = "06_可解释性分析"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_models(self) -> Dict[str, Any]:
        """
        加载所有NGBoost模型和对应的标准化器（包括特征标准化器和需求标准化器）
        
        Returns:
            Dict[str, Any]: 加载的模型字典
        """
        print("正在加载NGBoost模型...")
        
        for model_name in self.model_names:
            model_path = os.path.join(self.models_dir, model_name)
            
            if os.path.exists(model_path):
                # 加载模型
                model_data = joblib.load(model_path)
                
                # 确保模型数据包含特征标准化器（从模型文件中加载）
                if isinstance(model_data, dict) and 'feature_scaler' in model_data:
                    print(f"✓ 模型文件中包含特征标准化器")
                elif hasattr(model_data, 'feature_scaler'):
                    # 如果模型对象直接包含特征标准化器属性
                    model_data = {
                        'model': model_data,
                        'feature_scaler': getattr(model_data, 'feature_scaler')
                    }
                    print(f"✓ 从模型对象中提取特征标准化器")
                else:
                    print(f"⚠ 警告: 模型中未找到特征标准化器")
                
                # 加载对应的需求标准化器
                # 构建正确的需求标准化器文件名映射
                demand_scaler_name = model_name.replace('ngboost_model_', 'demand_scaler_')
                # 将英文标识转换为中文标识
                demand_scaler_name = demand_scaler_name.replace('online', '线上用户').replace('offline', '线下用户')
                demand_scaler_path = os.path.join(self.models_dir, demand_scaler_name)
                
                if os.path.exists(demand_scaler_path):
                    demand_scaler = joblib.load(demand_scaler_path)
                    # 将需求标准化器添加到模型数据中
                    if isinstance(model_data, dict):
                        model_data['demand_scaler'] = demand_scaler
                    else:
                        # 如果模型数据不是字典，创建一个包含模型和标准化器的字典
                        model_data = {
                            'model': model_data,
                            'demand_scaler': demand_scaler
                        }
                    print(f"✓ 成功加载: {model_name} - {self.model_descriptions[model_name]} (包含特征标准化器和需求标准化器)")
                else:
                    print(f"⚠ 警告: 需求标准化器文件不存在: {demand_scaler_path}")
                    print(f"✓ 成功加载: {model_name} - {self.model_descriptions[model_name]} (包含特征标准化器，无需求标准化器)")
                
                self.models[model_name] = model_data
                
                # 分析模型结构
                self._analyze_model_structure(model_name, model_data)
            else:
                print(f"✗ 文件不存在: {model_path}")
        
        return self.models
    
    def _analyze_model_structure(self, model_name: str, model_data: Any) -> None:
        """
        分析模型结构
        
        Args:
            model_name: 模型文件名
            model_data: 模型数据
        """
        print(f"\n分析模型结构: {model_name}")
        print("-" * 50)
        
        # 检查模型类型
        if isinstance(model_data, dict):
            print("模型结构: 字典格式")
            print(f"字典键: {list(model_data.keys())}")
            
            # 检查是否包含NGBoost模型
            if 'model' in model_data:
                model_obj = model_data['model']
                print(f"模型类型: {type(model_obj)}")
                
                # 检查模型属性
                if hasattr(model_obj, 'estimators_'):
                    print(f"基学习器数量: {len(model_obj.estimators_)}")
                
                if hasattr(model_obj, 'feature_importances_'):
                    print("特征重要性: 可用")
                else:
                    print("特征重要性: 不可用")
                    
        else:
            print(f"模型结构: {type(model_data)}")
            
            # 检查是否是NGBoost模型对象
            if hasattr(model_data, 'estimators_'):
                print(f"基学习器数量: {len(model_data.estimators_)}")
                print("特征重要性: 可用")
            else:
                print("特征重要性: 不可用")
    
    def generate_sample_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """
        生成样本数据用于分析
        
        Args:
            n_samples: 样本数量
            
        Returns:
            Tuple[np.ndarray, List[str]]: 样本数据和特征名称
        """
        np.random.seed(42)
        
        # 生成模拟特征数据（根据prepare_ngboost_features方法的6个特征）
        # 1. price: 价格特征 (50-300之间的合理价格范围)
        price = np.random.uniform(50, 300, n_samples)
        
        # 2. is_weekend: 是否周末 (0或1)
        is_weekend = np.random.binomial(1, 0.3, n_samples)  # 30%概率为周末
        
        # 3. season: 季节 (0-2: 淡季/平季/旺季)
        season = np.random.randint(0, 3, n_samples)
        
        # 4. price_cv: 价格变异系数 (0.1-0.5之间的变异系数)
        price_cv = np.random.uniform(0.1, 0.5, n_samples)
        
        # 5. demand_trend: 需求趋势 (-1到1之间的趋势值)
        demand_trend = np.random.uniform(-1, 1, n_samples)
        
        # 6. price_trend: 价格趋势 (-0.5到0.5之间的趋势值)
        price_trend = np.random.uniform(-0.5, 0.5, n_samples)
        
        # 组合所有特征
        features = np.column_stack([
            price, is_weekend, season, price_cv, demand_trend, price_trend
        ])
        
        print(f"生成样本数据: {n_samples} 个样本, {features.shape[1]} 个特征")
        print(f"特征维度: {features.shape}")
        
        return features, self.short_feature_names
    
    def analyze_feature_importance(self, model_name: str, model_data: Any, 
                                 features: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        分析特征重要性
        
        Args:
            model_name: 模型文件名
            model_data: 模型数据
            features: 特征数据
            feature_names: 特征名称
            
        Returns:
            Dict[str, float]: 特征重要性字典
        """
        print(f"\n分析特征重要性: {model_name}")
        
        # 提取实际的NGBoost模型
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        
        feature_importance = {}
        
        # 定义辅助函数
        def get_scalar_importance(value):
            if isinstance(value, (int, float)):
                return value
            elif hasattr(value, '__len__') and len(value) == 1:
                return float(value[0])
            elif hasattr(value, '__len__'):
                # 如果是数组，取平均值
                return float(np.mean(value))
            else:
                return float(value)
        
        # 方法1: 使用模型内置的特征重要性（但检查是否均匀分布）
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 检查重要性是否均匀分布（所有值相同）
            if np.allclose(importances, importances[0]):
                print("模型内置特征重要性为均匀分布，使用改进方法")
                # 继续使用下面的改进方法，不返回
            else:
                # 确保特征数量匹配
                n_features = min(len(importances), len(feature_names))
                
                for i in range(n_features):
                    feature_importance[feature_names[i]] = importances[i]
                
                print("✓ 使用模型内置特征重要性")
                
                # 排序并返回
                sorted_importance = sorted(feature_importance.items(), key=lambda x: get_scalar_importance(x[1]), reverse=True)
                
                print("\n特征重要性排名:")
                for i, (feature, importance) in enumerate(sorted_importance[:10]):  # 显示前10个
                    scalar_importance = get_scalar_importance(importance)
                    print(f"{i+1:2d}. {feature:15s}: {scalar_importance:.4f}")
                
                return feature_importance
        
        # 如果内置重要性均匀或不可用，使用改进方法
        print("使用基于分裂增益的特征重要性计算")
        
        # 方法2.1: 使用permutation importance
        from sklearn.inspection import permutation_importance
        
        # 计算排列重要性（使用前100个样本加速计算）
        n_samples = min(100, len(features))
        sample_features = features[:n_samples]
        
        # 定义预测函数 - 返回与输入样本数量相同的预测数组
        def predict_function(X):
            if hasattr(model, 'predict'):
                preds = model.predict(X)
                # 确保返回的是数组，且与输入样本数量相同
                if isinstance(preds, np.ndarray):
                    return preds.flatten()
                else:
                    return np.array([preds] * len(X))
            else:
                pred_dist = model.pred_dist(X)
                means = pred_dist.mean()
                if isinstance(means, np.ndarray):
                    return means.flatten()
                else:
                    return np.array([means] * len(X))
        
        # 获取目标值（用于permutation_importance）
        y_true = predict_function(sample_features)
        
        result = permutation_importance(
            model, sample_features, y_true,
            n_repeats=5, random_state=42, n_jobs=-1
        )
        
        importances = result.importances_mean
        
        for i, name in enumerate(feature_names):
            if i < len(importances):
                feature_importance[name] = importances[i]
            else:
                feature_importance[name] = 0.0
        
        print("✓ 使用排列重要性")
        
        # 排序并打印重要性
        sorted_importance = sorted(feature_importance.items(), key=lambda x: get_scalar_importance(x[1]), reverse=True)
        
        print("\n特征重要性排名:")
        for i, (feature, importance) in enumerate(sorted_importance[:10]):  # 显示前10个
            scalar_importance = get_scalar_importance(importance)
            print(f"{i+1:2d}. {feature:15s}: {scalar_importance:.4f}")
        
        return feature_importance
    
    def analyze_with_shap(self, model_name: str, model_data: Any, 
                         features: np.ndarray, feature_names: List[str]) -> Optional[Dict[str, Any]]:
        """
        使用SHAP进行可解释性分析
        
        Args:
            model_name: 模型文件名
            model_data: 模型数据
            features: 特征数据
            feature_names: 特征名称
            
        Returns:
            Optional[Dict[str, Any]]: SHAP分析结果
        """
        if not SHAP_AVAILABLE:
            print("SHAP库未安装，跳过SHAP分析")
            return None
        
        print(f"\n进行SHAP分析: {model_name}")
        
        # 提取实际的NGBoost模型
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        
        # 获取特征标准化器（如果存在）- 在创建解释器之前获取
        feature_scaler = None
        if isinstance(model_data, dict) and 'feature_scaler' in model_data:
            feature_scaler = model_data['feature_scaler']
            print("✓ 从模型数据中获取特征标准化器")
        elif hasattr(model, 'feature_scaler'):
            feature_scaler = model.feature_scaler
            print("✓ 从模型对象获取特征标准化器")
        
        # 创建SHAP解释器
        explainer = None
        shap_values = None
        
        # 使用TreeExplainer（适用于树模型）
        if hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
            print("✓ 使用TreeExplainer（NGBoost内部已处理标准化）")
        
        # 如果TreeExplainer不可用，使用KernelExplainer
        if explainer is None:
            # 定义预测函数（包含特征标准化）
            def predict_function(X):
                # 如果存在特征标准化器，先对输入数据进行标准化
                if feature_scaler is not None:
                    X_scaled = feature_scaler.transform(X)
                else:
                    X_scaled = X
                
                if hasattr(model, 'predict'):
                    return model.predict(X_scaled)
                else:
                    pred_dist = model.pred_dist(X_scaled)
                    return pred_dist.mean()
            
            # 使用前50个样本作为背景数据，并进行标准化（如果需要）
            background_data = features[:min(50, len(features))]
            if feature_scaler is not None:
                background_data_scaled = feature_scaler.transform(background_data)
            else:
                background_data_scaled = background_data
            
            explainer = shap.KernelExplainer(predict_function, background_data_scaled) # 创建KernelExplainer
            print("✓ 使用KernelExplainer（已考虑特征标准化）")
        
        if explainer is None:
            print("✗ 所有SHAP解释器都失败")
            return None
        
        # 计算SHAP值（使用前100个样本加速计算）
        n_samples = min(100, len(features))
        sample_features = features[:n_samples]
        
        # 如果存在特征标准化器，对样本特征进行标准化
        if feature_scaler is not None:
            sample_features_scaled = feature_scaler.transform(sample_features)
            print("✓ 样本特征已标准化")
        else:
            sample_features_scaled = sample_features
            print("⚠ 未找到特征标准化器，使用原始特征")
        
        # 计算SHAP值（SHAP分析不需要目标值，只需要预测函数）
        
        # 获取预测结果用于反标准化（SHAP分析不需要目标值，但我们需要反标准化期望值）
        if isinstance(model_data, dict) and 'demand_scaler' in model_data:
            demand_scaler = model_data['demand_scaler'] # 提取需求标准化器
            # 对SHAP期望值进行反标准化
            expected_value_original = demand_scaler.inverse_transform( 
                np.array([[explainer.expected_value]]))[0][0]
            print("✓ SHAP期望值已反标准化到原始需求尺度")
        else:
            expected_value_original = explainer.expected_value
            print("⚠ 未找到需求标准化器，使用标准化后的SHAP期望值")
        
        shap_values = explainer.shap_values(sample_features_scaled)
        
        # 创建SHAP分析结果
        shap_results = {
            'explainer': explainer,
            'shap_values': shap_values,
            'expected_value': expected_value_original,
            'feature_names': feature_names,
            'sample_features': sample_features_scaled  # 使用标准化后的特征
        }
        
        print("✓ SHAP分析完成")
        
        # 生成SHAP摘要图
        self._create_shap_summary_plot(model_name, shap_results)
        
        return shap_results
    
    def _create_shap_summary_plot(self, model_name: str, shap_results: Dict[str, Any]) -> None:
        """
        创建SHAP摘要图
        
        Args:
            model_name: 模型文件名
            shap_results: SHAP分析结果
        """
        plt.figure(figsize=(12, 8))
        
        # SHAP摘要图
        shap.summary_plot(
            shap_results['shap_values'], 
            shap_results['sample_features'],
            feature_names=shap_results['feature_names'],
            show=False,
            plot_size=None
        )
        
        # 设置中文标题
        model_desc = self.model_descriptions.get(model_name, model_name)
        plt.title(f"SHAP摘要图 - {model_desc}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        filename = f"SHAP_Summary_{model_name.replace('.pkl', '')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP摘要图已保存: {filepath}")
    
    def create_partial_dependence_plots(self, model_name: str, model_data: Any,
                                      features: np.ndarray, feature_names: List[str],
                                      top_features: int = 6) -> None:
        """
        创建部分依赖图（PDP）
        
        Args:
            model_name: 模型文件名
            model_data: 模型数据
            features: 特征数据
            feature_names: 特征名称
            top_features: 显示最重要的几个特征
        """
        print(f"\n创建部分依赖图: {model_name}")
        
        # 提取实际的NGBoost模型
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        
        # 选择最重要的特征进行分析
        importance_dict = self.analyze_feature_importance(model_name, model_data, features, feature_names)
        
        # 确保特征重要性值是标量
        def get_scalar_importance(value):
            if isinstance(value, (int, float)):
                return value
            elif hasattr(value, '__len__') and len(value) == 1:
                return float(value[0])
            elif hasattr(value, '__len__'):
                # 如果是数组，取平均值
                return float(np.mean(value))
            else:
                return float(value)
        
        # 使用标量值进行排序
        top_features_names = sorted(importance_dict.items(), 
                                  key=lambda x: get_scalar_importance(x[1]), 
                                  reverse=True)[:top_features]
        top_features_names = [name for name, _ in top_features_names]
        
        # 如果特征重要性均匀分布，使用所有特征
        if len(set([get_scalar_importance(v) for v in importance_dict.values()])) == 1:
            print("特征重要性均匀分布，使用所有特征生成PDP图")
            top_features_names = feature_names[:top_features]
        
        # 创建PDP图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        model_desc = self.model_descriptions.get(model_name, model_name)
        fig.suptitle(f"部分依赖图 - {model_desc}", fontsize=16, fontweight='bold')
        
        for i, feature_name in enumerate(top_features_names):
            if i >= len(axes):
                break
                
            # 获取特征索引
            feature_idx = feature_names.index(feature_name)
            
            # 生成特征值范围
            feature_values = features[:, feature_idx]
            unique_vals = np.unique(feature_values)
            
            # 如果唯一值太多，进行采样
            if len(unique_vals) > 50:
                test_points = np.linspace(np.min(feature_values), np.max(feature_values), 50)
            else:
                test_points = unique_vals
            
            # 计算部分依赖
            pdp_values = []
            
            for val in test_points:
                # 创建测试数据（固定当前特征，其他特征取均值）
                test_data = features.copy()
                test_data[:, feature_idx] = val
                
                # 对测试数据进行特征标准化（与NGBoost模型训练时一致）
                if isinstance(model_data, dict) and 'feature_scaler' in model_data:
                    # 从模型数据字典中获取标准化器
                    feature_scaler = model_data['feature_scaler']
                    test_data_scaled = feature_scaler.transform(test_data)
                elif hasattr(model, 'feature_scaler'):
                    # 直接从模型对象获取标准化器
                    test_data_scaled = model.feature_scaler.transform(test_data)
                else:
                    test_data_scaled = test_data
                
                # 预测
                if hasattr(model, 'predict'):
                    predictions = model.predict(test_data_scaled)
                    # 确保predictions是标量
                    if isinstance(predictions, np.ndarray) and predictions.size > 1:
                        predictions = np.mean(predictions)
                else:
                    pred_dist = model.pred_dist(test_data_scaled)
                    predictions = pred_dist.mean()
                    # 确保predictions是标量
                    if isinstance(predictions, np.ndarray) and predictions.size > 1:
                        predictions = np.mean(predictions)
                
                # 对预测结果进行反标准化（如果存在需求标准化器）
                if isinstance(model_data, dict) and 'demand_scaler' in model_data:
                    demand_scaler = model_data['demand_scaler']
                    # 将预测结果转换为原始需求尺度
                    predictions_original = demand_scaler.inverse_transform(np.array([[predictions]]))[0][0]
                    pdp_values.append(float(predictions_original))
                else:
                    pdp_values.append(float(predictions))
            
            # 绘制PDP
            ax = axes[i]
            ax.plot(test_points, pdp_values, 'b-', linewidth=2)
            ax.set_xlabel(feature_name, fontsize=12)
            ax.set_ylabel('预测需求', fontsize=12)
            ax.set_title(f'PDP - {feature_name}', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        # 隐藏未使用的子图
        for i in range(len(top_features_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"PDP_{model_name.replace('.pkl', '')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 部分依赖图已保存: {filepath}")
    
    def analyze_prediction_distribution(self, model_name: str, model_data: Any, 
                                       features: np.ndarray) -> None:
        """
        分析预测分布特性
        
        Args:
            model_name: 模型文件名
            model_data: 模型数据
            features: 特征数据
        """
        print(f"\n分析预测分布: {model_name}")
        
        # 提取实际的NGBoost模型和特征标准化器
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            feature_scaler = model_data.get('feature_scaler')  # 获取特征标准化器
        else:
            model = model_data
            feature_scaler = None
        
        # ✅ 特征标准化处理
        if feature_scaler is not None:
            features_scaled = feature_scaler.transform(features)
            print("✓ 特征已标准化处理")
        else:
            features_scaled = features
            print("⚠ 未找到特征标准化器，使用原始特征")
        
        # 获取预测分布 - 使用标准化后的特征
        pred_dist = model.pred_dist(features_scaled)
        
        # 提取均值和方差
        mean_pred = pred_dist.mean()
        var_pred = pred_dist.var
        std_pred = np.sqrt(var_pred)
        
        # 对预测结果进行反标准化（如果存在需求标准化器）
        if isinstance(model_data, dict) and 'demand_scaler' in model_data:
            demand_scaler = model_data['demand_scaler']
            # 将预测结果转换为原始需求尺度
            mean_pred_original = demand_scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
            # 标准差也需要相应转换：如果数据被标准化为N(0,1)，标准差需要乘以缩放因子
            if hasattr(demand_scaler, 'scale_'):
                std_pred_original = std_pred * demand_scaler.scale_[0]
            else:
                std_pred_original = std_pred  # 如果没有scale_属性，保持原值
            print("✓ 预测结果已反标准化到原始需求尺度")
        else:
            mean_pred_original = mean_pred
            std_pred_original = std_pred
            print("⚠ 未找到需求标准化器，使用标准化后的预测值")
        
        # 分析分布特性（使用反标准化后的值）
        distribution_stats = {
            'mean_of_means': np.mean(mean_pred_original),
            'std_of_means': np.std(mean_pred_original),
            'mean_of_std': np.mean(std_pred_original),
            'std_of_std': np.std(std_pred_original),
            'min_mean': np.min(mean_pred_original),
            'max_mean': np.max(mean_pred_original),
            'min_std': np.min(std_pred_original),
            'max_std': np.max(std_pred_original),
            'cv_of_means': np.std(mean_pred_original) / np.mean(mean_pred_original),  # 变异系数
            'uncertainty_ratio': np.mean(std_pred_original) / np.mean(mean_pred_original)  # 不确定性比率
        }
        
        print("预测分布统计（原始需求尺度）:")
        for stat_name, value in distribution_stats.items():
            print(f"  {stat_name:15s}: {value:.4f}")
        
        # 创建分布可视化（使用反标准化后的值）
        self._create_distribution_visualization(model_name, mean_pred_original, std_pred_original)
    
    def _create_distribution_visualization(self, model_name: str, mean_pred: np.ndarray, 
                                          std_pred: np.ndarray) -> None:
        """
        创建预测分布可视化
        
        Args:
            model_name: 模型文件名
            mean_pred: 均值预测
            std_pred: 标准差预测
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_desc = self.model_descriptions.get(model_name, model_name)
        fig.suptitle(f"预测分布分析 - {model_desc}", fontsize=16, fontweight='bold')
        
        # 1. 均值分布直方图
        axes[0, 0].hist(mean_pred, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('预测均值', fontsize=12)
        axes[0, 0].set_ylabel('频数', fontsize=12)
        axes[0, 0].set_title('均值预测分布', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 标准差分布直方图
        axes[0, 1].hist(std_pred, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('预测标准差', fontsize=12)
        axes[0, 1].set_ylabel('频数', fontsize=12)
        axes[0, 1].set_title('不确定性分布', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 均值vs标准差散点图
        axes[1, 0].scatter(mean_pred, std_pred, alpha=0.6, color='green')
        axes[1, 0].set_xlabel('预测均值', fontsize=12)
        axes[1, 0].set_ylabel('预测标准差', fontsize=12)
        axes[1, 0].set_title('均值vs不确定性', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 不确定性比率
        uncertainty_ratio = std_pred / mean_pred
        axes[1, 1].hist(uncertainty_ratio, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('不确定性比率 (σ/μ)', fontsize=12)
        axes[1, 1].set_ylabel('频数', fontsize=12)
        axes[1, 1].set_title('不确定性比率分布', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"Distribution_{model_name.replace('.pkl', '')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 预测分布图已保存: {filepath}")
    
    def generate_comprehensive_report(self) -> None:
        """
        生成综合可解释性分析报告
        """
        print("=" * 70)
        print("NGBoost模型可解释性综合分析报告")
        print("=" * 70)
        
        # 加载模型
        if not self.models:
            self.load_models()
        
        if not self.models:
            print("没有可用的模型进行分析")
            return
        
        # 生成样本数据
        features, feature_names = self.generate_sample_data(1000)
        
        # 对每个模型进行分析
        for model_name, model_data in self.models.items():
            print(f"\n{'='*60}")
            print(f"分析模型: {model_name}")
            print(f"描述: {self.model_descriptions.get(model_name, 'N/A')}")
            print(f"{'='*60}")
            
            # 1. 特征重要性分析
            self.analyze_feature_importance(model_name, model_data, features, feature_names)
            
            # 2. SHAP分析
            self.analyze_with_shap(model_name, model_data, features, feature_names)
            
            # 3. 部分依赖图
            self.create_partial_dependence_plots(model_name, model_data, features, feature_names)
            
            # 4. 预测分布分析
            self.analyze_prediction_distribution(model_name, model_data, features)
            
            print(f"✓ {model_name} 分析完成")
        
        print("\n" + "=" * 70)
        print("分析完成!")
        print(f"所有结果已保存到目录: {self.output_dir}")
        print("=" * 70)

def main():
    """主函数"""
    # 创建分析器
    analyzer = NGBoostInterpretabilityAnalyzer()
    
    # 生成综合报告
    analyzer.generate_comprehensive_report()
    
    print("\n✓ 分析完成! 请查看输出目录中的图表和报告。")

if __name__ == "__main__":
    main()