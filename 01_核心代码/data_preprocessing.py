#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 标准库导入
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

class HotelDataPreprocessor:
    """
    酒店数据预处理器
    
    该类负责酒店预订数据的清洗、特征工程和预处理工作。
    主要功能包括：
    - 数据清洗（处理缺失值、异常值）
    - 构造每日需求标签（包含真实价格信息）
    - 特征工程（时间特征、滞后特征、滚动统计等）
    - 为BNN模型准备输入特征
    """
    
    def __init__(self) -> None:
        """
        初始化酒店数据预处理器
        
        初始化特征列列表、标准化器和分类编码器字典。
        这些属性将在后续的数据处理过程中被填充和使用。
        """
        self.feature_columns: Optional[List[str]] = None  # 特征列名列表
        self.scaler: Optional[Any] = None  # 标准化器
        self.categorical_encoders: Dict[str, Any] = {}  # 分类变量编码器字典
    def __init__(self) -> None:
        self.feature_columns: Optional[List[str]] = None
        self.scaler: Optional[Any] = None
        self.categorical_encoders: Dict[str, Any] = {}
        
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
        加载并预处理酒店预订数据
        
        这是主要的预处理流水线，按顺序执行以下步骤：
        1. 从CSV文件加载原始酒店预订数据
        2. 调用clean_data进行数据清洗
        3. 调用construct_daily_demand_labels构造每日需求标签
        4. 调用construct_features进行特征工程
        
        Args:
            file_path (str): 原始数据文件的路径
            
        Returns:
            pd.DataFrame: 包含完整特征和标签的预处理数据框
            
        Raises:
            FileNotFoundError: 当指定的文件路径不存在时
            pd.errors.EmptyDataError: 当CSV文件为空时
        """
        print("正在加载酒店预订数据...")
        df = pd.read_csv(file_path)
        print(f"数据加载完成，共{len(df)}条记录")
        
        # 数据清洗
        df = self.clean_data(df)
        
        # 构造每日需求标签
        daily_demand = self.construct_daily_demand_labels(df)
        
        # 构造特征
        features_df = self.construct_features(daily_demand)
        
        return features_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        对酒店预订数据进行全面的清洗处理，包括：
        1. 缺失值处理：对children、agent、company等字段填充0
        2. 异常值处理：
           - ADR（平均日房价）限制在0-500元范围内
           - 成人数限制在最多4人
           - 入住天数限制在最多5晚（周末+工作日）
        3. 客户分类：根据市场细分和分销渠道区分线上用户和线下用户
        
        这些清洗规则基于酒店业务的合理范围设定，可以有效去除数据中的异常记录。
        
        Args:
            df (pd.DataFrame): 原始酒店预订数据框
            
        Returns:
            pd.DataFrame: 清洗后的数据框，包含客户分类字段
            
        Note:
            - ADR超过500元的记录会被截断为500元
            - 成人数超过4人的记录会被截断为4人
            - 入住天数超过5晚的记录会被调整为2晚周末+3晚工作日
            - 客户分类逻辑：市场细分='Online TA' 或 分销渠道='TA/TO' 为线上用户
        """
        print("正在清洗数据...")
        
        # 处理缺失值
        df['children'] = df['children'].fillna(0)
        df['agent'] = df['agent'].fillna(0)
        df['company'] = df['company'].fillna(0)
        
        # 异常值处理
        # ADR异常值处理
        df.loc[df['adr'] > 500, 'adr'] = 500
        df.loc[df['adr'] < 0, 'adr'] = 0
        
        # 成人数异常值处理
        df.loc[df['adults'] > 4, 'adults'] = 4
        
        # 入住天数异常值处理
        df.loc[df['stays_in_weekend_nights'] + df['stays_in_week_nights'] > 5, 
               ['stays_in_weekend_nights', 'stays_in_week_nights']] = [2, 3]
        
        # 客户分类：区分线上用户与线下用户
        def classify_customer(row):
            if row['market_segment'] == 'Online TA' or row['distribution_channel'] == 'TA/TO':
                return '线上用户'
            else:
                return '线下用户'
        
        df['customer_type'] = df.apply(classify_customer, axis=1)
        
        # 统计客户类型分布
        customer_type_counts = df['customer_type'].value_counts()
        print(f"客户类型分布：{customer_type_counts.to_dict()}")
        
        return df
    
    def construct_daily_demand_labels(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        构造每日需求标签（区分线上和线下用户）
        
        基于酒店预订数据构造每日需求标签，分别统计线上和线下用户的需求：
        1. 创建到达日期
        2. 筛选有效预订（未取消且实际入住的订单）
        3. 根据客户类型分别统计每日需求、平均价格和价格分布
        4. 填充缺失的日期，确保时间序列连续性
        5. 对无订单的日期进行价格信息插值填充
        
        Args:
            df (pd.DataFrame): 清洗后的酒店预订数据
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 线上和线下用户的每日需求数据框
            列包括：date, daily_demand, avg_price, price_std, min_price, max_price, median_price, customer_type
            
        Note:
            - 只考虑未取消且实际入住的订单作为有效需求
            - 对缺失日期使用0填充需求量，价格信息通过插值填充
            - 价格插值使用相邻日期的价格信息，最后用整体平均值兜底
        """
        print("正在构造每日需求标签（区分线上和线下用户）...")
        
        # 创建到达日期
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' + 
            df['arrival_date_month'] + '-' + 
            df['arrival_date_day_of_month'].astype(str)
        )
        
        # 筛选有效预订（未取消且实际入住）
        valid_bookings = df[
            (df['is_canceled'] == 0) & 
            (df['stays_in_weekend_nights'] + df['stays_in_week_nights'] >= 1)
        ].copy()
        
        # 分别处理线上和线下用户
        online_bookings = valid_bookings[valid_bookings['customer_type'] == '线上用户'].copy()
        offline_bookings = valid_bookings[valid_bookings['customer_type'] == '线下用户'].copy()
        
        print(f"有效订单总数: {len(valid_bookings)}")
        print(f"线上用户订单数: {len(online_bookings)} ({len(online_bookings)/len(valid_bookings)*100:.1f}%)")
        print(f"线下用户订单数: {len(offline_bookings)} ({len(offline_bookings)/len(valid_bookings)*100:.1f}%)")
        
        def create_daily_stats(bookings, customer_type):
            """为指定客户类型创建每日统计"""
            if len(bookings) == 0:
                # 如果没有该类型的订单，创建空的数据框
                daily_stats = pd.DataFrame(columns=['date', 'daily_demand', 'avg_price', 'price_std', 'min_price', 'max_price', 'median_price'])
                return daily_stats
                
            # 按日期分组统计每日需求、平均价格和价格分布
            daily_stats = bookings.groupby('arrival_date').agg({
                'adr': ['count', 'mean', 'std', 'min', 'max', 'median']
            }).round(2)
            
            # 重命名列
            daily_stats.columns = ['daily_demand', 'avg_price', 'price_std', 'min_price', 'max_price', 'median_price']
            daily_stats = daily_stats.reset_index()
            daily_stats.rename(columns={'arrival_date': 'date'}, inplace=True)
            daily_stats['customer_type'] = customer_type
            
            return daily_stats
        
        # 分别为线上和线下用户创建每日统计
        online_daily_stats = create_daily_stats(online_bookings, '线上用户')
        offline_daily_stats = create_daily_stats(offline_bookings, '线下用户')
        
        # 获取完整的日期范围（基于所有有效订单）
        if len(valid_bookings) > 0:
            date_range = pd.date_range(valid_bookings['arrival_date'].min(), valid_bookings['arrival_date'].max())
            
            # 为线上用户数据填充缺失日期
            online_daily_stats = online_daily_stats.set_index('date').reindex(date_range, fill_value=0).reset_index()
            online_daily_stats.rename(columns={'index': 'date'}, inplace=True)
            online_daily_stats['customer_type'] = '线上用户'
            
            # 为线下用户数据填充缺失日期
            offline_daily_stats = offline_daily_stats.set_index('date').reindex(date_range, fill_value=0).reset_index()
            offline_daily_stats.rename(columns={'index': 'date'}, inplace=True)
            offline_daily_stats['customer_type'] = '线下用户'
            
            # 对没有订单的日期，填充价格统计信息
            for stats_df, bookings in [(online_daily_stats, online_bookings), (offline_daily_stats, offline_bookings)]:
                if len(bookings) > 0:
                    # 将0值替换为NaN以便插值
                    for col in ['avg_price', 'price_std', 'min_price', 'max_price', 'median_price']:
                        stats_df[col] = stats_df[col].replace(0, np.nan)
                    
                    # 用相邻日期的价格信息填充
                    stats_df['avg_price'] = stats_df['avg_price'].interpolate()
                    stats_df['price_std'] = stats_df['price_std'].interpolate()
                    stats_df['min_price'] = stats_df['min_price'].interpolate()
                    stats_df['max_price'] = stats_df['max_price'].interpolate()
                    stats_df['median_price'] = stats_df['median_price'].interpolate()
                    
                    # 如果还有缺失，用该客户类型的整体平均值填充
                    stats_df['avg_price'] = stats_df['avg_price'].fillna(bookings['adr'].mean())
                    stats_df['price_std'] = stats_df['price_std'].fillna(bookings['adr'].std())
                    stats_df['min_price'] = stats_df['min_price'].fillna(bookings['adr'].min())
                    stats_df['max_price'] = stats_df['max_price'].fillna(bookings['adr'].max())
                    stats_df['median_price'] = stats_df['median_price'].fillna(bookings['adr'].median())
        
        print(f"线上用户需求标签构造完成，共{len(online_daily_stats)}天数据")
        print(f"线下用户需求标签构造完成，共{len(offline_daily_stats)}天数据")
        
        if len(online_bookings) > 0:
            print(f"线上用户价格统计 - 平均价格: {online_bookings['adr'].mean():.2f}, 价格范围: {online_bookings['adr'].min():.2f}-{online_bookings['adr'].max():.2f}")
        if len(offline_bookings) > 0:
            print(f"线下用户价格统计 - 平均价格: {offline_bookings['adr'].mean():.2f}, 价格范围: {offline_bookings['adr'].min():.2f}-{offline_bookings['adr'].max():.2f}")
        
        return online_daily_stats, offline_daily_stats
    
    def construct_features(self, daily_stats: pd.DataFrame) -> pd.DataFrame:
        """
        构造特征工程
        
        基于每日需求和价格数据构造丰富的特征集合，用于机器学习模型训练：
        
        时间特征：
        - 基本时间特征：年、月、日、星期、季度、周数
        
        滞后特征（Lag Features）：
        - 需求和价格的1、2、3、7、14、30天滞后值
        
        滚动统计特征（Rolling Statistics）：
        - 需求和价格的3、7、14、30天移动平均和标准差
        - 价格区间（最高价-最低价）和价格变异系数（标准差/均值）
        
        节假日和季节性特征：
        - 周末标识、月初标识、月末标识
        - 季节编码（淡季、平季、旺季）
        
        需求-价格关系特征：
        - 需求价格比率、价格需求弹性
        
        趋势特征：
        - 基于7天窗口的需求和价格趋势（线性回归斜率）
        
        Args:
            daily_stats (pd.DataFrame): 包含每日需求量和价格统计的基础数据
            
        Returns:
            pd.DataFrame: 包含完整特征工程的数据框
            
        Note:
            - 使用前后向填充处理缺失值，确保时间序列连续性
            - 特征工程基于酒店业务理解，考虑了季节性、周期性、趋势性等因素
            - 滞后和滚动窗口的选择基于业务经验（短期1-3天，中期7-14天，长期30天）
        """
        print("正在构造特征...")
        
        df = daily_stats.copy()
        
        # 时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week
        
        # 滞后特征
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'demand_lag_{lag}'] = df['daily_demand'].shift(lag)
            df[f'price_lag_{lag}'] = df['avg_price'].shift(lag)
        
        # 滚动统计特征
        for window in [3, 7, 14, 30]:
            df[f'demand_ma_{window}'] = df['daily_demand'].rolling(window=window, min_periods=1).mean()
            df[f'price_ma_{window}'] = df['avg_price'].rolling(window=window, min_periods=1).mean()
            df[f'demand_std_{window}'] = df['daily_demand'].rolling(window=window, min_periods=1).std()
            df[f'price_std_{window}'] = df['avg_price'].rolling(window=window, min_periods=1).std()
        
        # 价格区间特征
        df['price_range'] = df['max_price'] - df['min_price']
        df['price_cv'] = df['price_std'] / (df['avg_price'] + 1e-8)  # 价格变异系数
        
        # 节假日特征（简化版）
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = (df['day'] <= 3).astype(int)
        df['is_month_end'] = (df['day'] >= 28).astype(int)
        
        # 季节性特征
        df['season'] = df['month'].apply(self.get_season)
        
        # 需求-价格关系特征
        df['demand_price_ratio'] = df['daily_demand'] / (df['avg_price'] + 1e-8)
        df['price_demand_elasticity'] = df['avg_price'].pct_change() / (df['daily_demand'].pct_change() + 1e-8)
        
        # 趋势特征
        df['demand_trend'] = df['daily_demand'].rolling(window=7, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        df['price_trend'] = df['avg_price'].rolling(window=7, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # 处理缺失值
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # 特征列列表
        self.feature_columns = [col for col in df.columns if col not in ['date']]
        
        print(f"特征构造完成，共{len(self.feature_columns)}个特征")
        print(f"特征列: {self.feature_columns[:10]}...")  # 只显示前10个
        
        return df
    
    def get_season(self, month: int) -> int:
        """
        获取季节编码
        
        基于月份将一年划分为三个季节，反映酒店业的季节性特征：
        - 淡季（11月-次年2月）：冬季，需求相对较低
        - 平季（3-5月，9-10月）：春秋季节，需求适中
        - 旺季（6-8月）：夏季，需求相对较高
        
        Args:
            month (int): 月份（1-12）
            
        Returns:
            int: 季节编码（0=淡季，1=平季，2=旺季）
            
        Note:
            这种季节划分基于北半球酒店业的典型季节性模式，
            可以根据具体地理位置和酒店类型进行调整。
        """
        if month in [11, 12, 1, 2]:
            return 0  # 淡季
        elif month in [3, 4, 5, 9, 10]:
            return 1  # 平季
        else:  # 6, 7, 8
            return 2  # 旺季
    
    def is_holiday(self, date: pd.Timestamp) -> int:
        """
        判断是否为节假日（简化版）
        
        目前实现为简化版本，仅考虑周末作为特殊日期。
        在实际应用中，应该集成完整的节假日API或数据库，
        包括国家法定节假日、地方节假日、特殊事件等。
        
        Args:
            date (pd.Timestamp): 日期时间戳
            
        Returns:
            int: 节假日标识（1=是节假日，0=不是节假日）
            
        Todo:
            - 集成中国法定节假日API
            - 添加地方特殊节假日
            - 考虑学校假期等影响因素
            - 添加重大事件（如展会、体育赛事）影响
        """
        # 这里可以添加具体的节假日逻辑
        # 为简化，只考虑周末
        return 1 if date.weekday() >= 5 else 0
    
    def prepare_bnn_features(self, features_df: pd.DataFrame, action: Optional[int] = None, price_action: Optional[float] = None) -> torch.Tensor:
        """
        准备BNN模型的输入特征（季节、工作日/周末、价格）
        
        为贝叶斯神经网络准备标准化的输入特征，主要包括三个核心特征：
        1. 季节性特征：反映酒店需求的季节性变化
        2. 工作日/周末特征：反映周周期性需求模式
        3. 价格特征：反映价格对需求的影响
        
        Args:
            features_df (pd.DataFrame): 特征数据框，包含日期、价格等基础特征
            action (Optional[int], optional): 可选的动作参数，预留用于强化学习
            price_action (Optional[float], optional): 可选的价格动作，如果提供则使用指定价格
            
        Returns:
            torch.Tensor: 标准化的特征张量，形状为[3]（季节、是否周末、价格）
            
        Note:
            - 该方法提取了影响酒店需求最核心的三个特征
            - 特征值都转换为浮点数，便于神经网络处理
            - 价格特征使用历史平均价格作为默认值
            - 所有特征都进行了合理的默认值处理，确保鲁棒性
        """
        
        # 获取季节特征
        if 'season' in features_df.columns:
            season = float(features_df['season'].iloc[0])
        else:
            season = 1.0  # 默认平季
        
        # 获取工作日/周末特征
        if 'is_weekend' in features_df.columns:
            is_weekend = float(features_df['is_weekend'].iloc[0])
        else:
            is_weekend = 0.0  # 默认工作日
        
        # 获取价格特征
        if price_action is not None:
            # 如果指定了价格动作，使用指定价格
            price = float(price_action)
        elif 'avg_price' in features_df.columns:
            # 使用历史平均价格
            price = float(features_df['avg_price'].iloc[0])
        else:
            # 默认值
            price = 100.0
        
        # 组合特征 [季节, 是否周末, 价格]
        features = [season, is_weekend, price]
        
        return torch.FloatTensor(features)
    
    def sample_data(self, X, y, method='random_sample', train_samples=400, 
                   val_samples=200, test_samples=193, random_seed=42, 
                   stratify_by=None, ensure_diversity=True):
        """
        随机抽取数据集，而不是简单划分
        
        支持两种抽取方式：
        1. 随机抽取：从整个数据池中随机选择指定数量的样本，确保多样性
        2. 顺序抽取：按时间顺序选择指定数量的样本
        
        Args:
            X: 特征数据
            y: 目标变量
            method: 抽取方法，'random_sample'表示随机抽取，'sequential_sample'表示顺序抽取
            train_samples: 训练集样本数量
            val_samples: 验证集样本数量
            test_samples: 测试集样本数量
            random_seed: 随机种子，确保可重复性
            stratify_by: 分层抽样的列名，None表示不进行分层抽样
            ensure_diversity: 是否确保抽取样本的多样性
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            
        Note:
            - 随机抽取确保从整个数据池中随机选择样本，避免局部偏差
            - 可以精确控制每个数据集的样本数量
            - 支持多样性检查，确保抽取样本在关键特征上的代表性
            - 适用于需要从大数据集中抽取代表性子集的场景
        """
        if method == 'random_sample':
            # 随机抽取，从整个数据池中随机选择指定数量的样本
            
            total_samples = len(X)
            print(f"总数据池大小: {total_samples} 样本")
            print(f"目标抽取数量: 训练集{train_samples} + 验证集{val_samples} + 测试集{test_samples} = {train_samples + val_samples + test_samples} 样本")
            
            # 检查样本数量是否足够
            if train_samples + val_samples + test_samples > total_samples:
                print(f"[警告] 请求的样本总数超过可用数据，将按比例调整")
                # 按比例调整样本数量
                total_requested = train_samples + val_samples + test_samples
                train_samples = int(train_samples * total_samples / total_requested)
                val_samples = int(val_samples * total_samples / total_requested)
                test_samples = total_samples - train_samples - val_samples
                print(f"调整后抽取数量: 训练集{train_samples} + 验证集{val_samples} + 测试集{test_samples} = {total_samples} 样本")
            
            # 设置随机种子确保可重复性
            np.random.seed(random_seed)
            
            # 创建样本索引
            all_indices = np.arange(total_samples)
            
            # 分层抽样处理
            if stratify_by and hasattr(X, 'columns') and stratify_by in X.columns:
                # 使用分层抽样确保各数据集中关键特征的平衡
                stratify_data = X[stratify_by].values
                
                # 首先抽取训练集
                train_indices, remaining_indices = train_test_split(
                    all_indices, train_size=train_samples, random_state=random_seed,
                    stratify=stratify_data[all_indices]
                )
                
                # 从剩余样本中抽取验证集
                val_size = min(val_samples, len(remaining_indices))
                if val_size > 0:
                    val_indices, test_remaining = train_test_split(
                        remaining_indices, train_size=val_size, random_state=random_seed + 1,
                        stratify=stratify_data[remaining_indices] if len(remaining_indices) > 0 else None
                    )
                else:
                    val_indices = np.array([])
                    test_remaining = remaining_indices
                
                # 剩余的作为测试集
                test_indices = test_remaining[:min(test_samples, len(test_remaining))]
                
            else:
                # 完全随机抽取
                np.random.shuffle(all_indices)
                
                train_indices = all_indices[:train_samples]
                val_indices = all_indices[train_samples:train_samples + val_samples]
                test_indices = all_indices[train_samples + val_samples:train_samples + val_samples + test_samples]
            
            # 根据索引抽取数据
            if hasattr(X, 'iloc'):  # DataFrame
                X_train = X.iloc[train_indices]
                X_val = X.iloc[val_indices] if len(val_indices) > 0 else X.iloc[:0]
                X_test = X.iloc[test_indices] if len(test_indices) > 0 else X.iloc[:0]
                y_train = y.iloc[train_indices]
                y_val = y.iloc[val_indices] if len(val_indices) > 0 else y.iloc[:0]
                y_test = y.iloc[test_indices] if len(test_indices) > 0 else y.iloc[:0]
            else:  # numpy数组
                X_train = X[train_indices]
                X_val = X[val_indices] if len(val_indices) > 0 else X[:0]
                X_test = X[test_indices] if len(test_indices) > 0 else X[:0]
                y_train = y[train_indices]
                y_val = y[val_indices] if len(val_indices) > 0 else y[:0]
                y_test = y[test_indices] if len(test_indices) > 0 else y[:0]
            
            print(f"随机抽取完成:")
            print(f"  训练集: {len(X_train)} 样本")
            print(f"  验证集: {len(X_val)} 样本")
            print(f"  测试集: {len(X_test)} 样本")
            
            # 多样性检查
            if ensure_diversity and hasattr(X_train, 'columns'):
                self._check_sample_diversity(X_train, X_val, X_test)
            
        elif method == 'sequential_sample':
            # 顺序抽取，按时间顺序选择指定数量的样本
            total_samples = len(X)
            print(f"总数据池大小: {total_samples} 样本")
            
            # 检查样本数量是否足够
            requested_total = train_samples + val_samples + test_samples
            if requested_total > total_samples:
                print(f"[警告] 请求的样本总数超过可用数据，将按比例调整")
                # 按比例调整样本数量
                train_samples = int(train_samples * total_samples / requested_total)
                val_samples = int(val_samples * total_samples / requested_total)
                test_samples = total_samples - train_samples - val_samples
            
            # 按顺序抽取
            if hasattr(X, 'iloc'):  # DataFrame
                X_train = X.iloc[:train_samples]
                X_val = X.iloc[train_samples:train_samples + val_samples]
                X_test = X.iloc[train_samples + val_samples:train_samples + val_samples + test_samples]
                y_train = y.iloc[:train_samples]
                y_val = y.iloc[train_samples:train_samples + val_samples]
                y_test = y.iloc[train_samples + val_samples:train_samples + val_samples + test_samples]
            else:  # numpy数组
                X_train = X[:train_samples]
                X_val = X[train_samples:train_samples + val_samples]
                X_test = X[train_samples + val_samples:train_samples + val_samples + test_samples]
                y_train = y[:train_samples]
                y_val = y[train_samples:train_samples + val_samples]
                y_test = y[train_samples + val_samples:train_samples + val_samples + test_samples]
            
            print(f"顺序抽取完成:")
            print(f"  训练集: {len(X_train)} 样本")
            print(f"  验证集: {len(X_val)} 样本")
            print(f"  测试集: {len(X_test)} 样本")
            
        else:
            # 向后兼容：旧的随机划分方法
            print(f"[警告] 使用方法 '{method}' 不存在，使用默认随机抽取")
            return self.sample_data(X, y, method='random_sample', train_samples=train_samples,
                                  val_samples=val_samples, test_samples=test_samples,
                                  random_seed=random_seed, stratify_by=stratify_by,
                                  ensure_diversity=ensure_diversity)
        
        # 分析划分结果的分布特征
        self._analyze_split_distribution(X_train, X_val, X_test, method == 'random_sample')
        
        # 返回抽取结果
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _check_sample_diversity(self, X_train, X_val, X_test):
        """
        检查抽取样本的多样性，确保代表性
        
        Args:
            X_train: 训练集
            X_val: 验证集
            X_test: 测试集
        """
        print(f"\n=== 样本多样性检查 ===")
        
        if hasattr(X_train, 'columns'):
            # 检查时间分布多样性
            if 'date' in X_train.columns:
                train_dates = pd.to_datetime(X_train['date'])
                val_dates = pd.to_datetime(X_val['date'])
                test_dates = pd.to_datetime(X_test['date'])
                
                print(f"时间多样性检查:")
                print(f"  训练集时间跨度: {(train_dates.max() - train_dates.min()).days} 天")
                print(f"  验证集时间跨度: {(val_dates.max() - val_dates.min()).days} 天")
                print(f"  测试集时间跨度: {(test_dates.max() - test_dates.min()).days} 天")
            
            # 检查价格分布多样性
            if 'avg_price' in X_train.columns:
                train_price_std = X_train['avg_price'].std()
                val_price_std = X_val['avg_price'].std()
                test_price_std = X_test['avg_price'].std()
                
                print(f"价格多样性检查:")
                print(f"  训练集价格标准差: {train_price_std:.2f}")
                print(f"  验证集价格标准差: {val_price_std:.2f}")
                print(f"  测试集价格标准差: {test_price_std:.2f}")
                
                # 检查价格范围覆盖
                min_price = min(X_train['avg_price'].min(), X_val['avg_price'].min(), X_test['avg_price'].min())
                max_price = max(X_train['avg_price'].max(), X_val['avg_price'].max(), X_test['avg_price'].max())
                print(f"  价格范围覆盖: {min_price:.0f} - {max_price:.0f} 元")
        
        print(f"=== 多样性检查完成 ===\n")
    
    def _analyze_split_distribution(self, X_train, X_val, X_test, is_random: bool = False) -> None:
        """
        分析数据划分后的分布特征
        
        比较训练集、验证集和测试集在时间分布、季节分布和价格分布上的差异，
        确保随机划分后各集合具有相似的分布特征。
        
        Args:
            X_train: 训练集特征数据
            X_val: 验证集特征数据
            X_test: 测试集特征数据
            is_random: 是否使用随机划分
            
        Returns:
            None
            
        Note:
            - 分析时间范围、季节分布、价格水平等关键特征
            - 打印各集合的分布统计信息，便于验证划分的合理性
        """
        print("\n=== 数据划分分布分析 ===")
        
        # 确保输入的是DataFrame
        if hasattr(X_train, 'columns'):
            # 时间范围分析
            if 'date' in X_train.columns:
                print(f"时间范围分析：")
                print(f"训练集：{X_train['date'].min()} 到 {X_train['date'].max()}（{len(X_train)}天）")
                print(f"验证集：{X_val['date'].min()} 到 {X_val['date'].max()}（{len(X_val)}天）")
                print(f"测试集：{X_test['date'].min()} 到 {X_test['date'].max()}（{len(X_test)}天）")
            
            # 季节分布分析
            if 'season' in X_train.columns:
                train_seasons = X_train['season']
                val_seasons = X_val['season']
                test_seasons = X_test['season']
                
                print(f"\n季节分布分析：")
                print(f"训练集季节分布：{dict(train_seasons.value_counts().sort_index())}")
                print(f"验证集季节分布：{dict(val_seasons.value_counts().sort_index())}")
                print(f"测试集季节分布：{dict(test_seasons.value_counts().sort_index())}")
            
            # 价格分布分析
            if 'avg_price' in X_train.columns:
                train_prices = X_train['avg_price']
                val_prices = X_val['avg_price']
                test_prices = X_test['avg_price']
                
                print(f"\n价格分布分析：")
                print(f"训练集价格：均值={train_prices.mean():.2f}，标准差={train_prices.std():.2f}")
                print(f"验证集价格：均值={val_prices.mean():.2f}，标准差={val_prices.std():.2f}")
                print(f"测试集价格：均值={test_prices.mean():.2f}，标准差={test_prices.std():.2f}")
            
            # 工作日/周末分布分析
            if 'is_weekend' in X_train.columns:
                train_weekend = X_train['is_weekend']
                val_weekend = X_val['is_weekend']
                test_weekend = X_test['is_weekend']
                
                print(f"\n工作日/周末分布分析：")
                print(f"训练集工作日占比：{(1-train_weekend.mean())*100:.1f}%，周末占比：{train_weekend.mean()*100:.1f}%")
                print(f"验证集工作日占比：{(1-val_weekend.mean())*100:.1f}%，周末占比：{val_weekend.mean()*100:.1f}%")
                print(f"测试集工作日占比：{(1-test_weekend.mean())*100:.1f}%，周末占比：{test_weekend.mean()*100:.1f}%")
        
        elif hasattr(X_train, 'iloc'):  # numpy数组但有iloc方法
            # 对于numpy数组，只分析数值特征
            print(f"数值特征分布分析：")
            print(f"训练集样本数：{len(X_train)}")
            print(f"验证集样本数：{len(X_val)}")
            print(f"测试集样本数：{len(X_test)}")
            
            if X_train.shape[1] > 0:  # 如果有特征
                print(f"特征维度：{X_train.shape[1]}")
        
        else:  # 纯numpy数组
            print(f"数组形状分析：")
            print(f"训练集形状：{X_train.shape}")
            print(f"验证集形状：{X_val.shape}")
            print(f"测试集形状：{X_test.shape}")
        
        # 显示划分方法信息
        if is_random:
            print(f"\n划分方法：随机划分（避免时间序列偏差）")
        else:
            print(f"\n划分方法：时间顺序划分（保持时间序列特性）")
        
        print("\n=== 分布分析完成 ===\n")
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        保存预处理器
        
        将HotelDataPreprocessor实例序列化保存到文件，包括所有配置参数、
        特征列列表、标准化器和编码器等内部状态。
        
        Args:
            filepath (str): 保存文件路径，建议使用.pkl扩展名
            
        Returns:
            None
            
        Note:
            - 使用pickle进行序列化，保存完整的对象状态
            - 保存的特征列列表对于后续特征提取很重要
            - 文件路径应该具有写权限
            - 保存的预处理器可以通过load_preprocessor方法加载
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"预处理器已保存到：{filepath}")
    
    @staticmethod
    def load_preprocessor(filepath: str) -> 'HotelDataPreprocessor':
        """
        加载预处理器
        
        从文件反序列化加载HotelDataPreprocessor实例，恢复保存时的完整状态，
        包括所有配置参数、特征列列表、标准化器和编码器等。
        
        Args:
            filepath (str): 预处理器文件路径，应该是之前通过save_preprocessor保存的文件
            
        Returns:
            HotelDataPreprocessor: 加载的预处理器实例
            
        Raises:
            FileNotFoundError: 如果指定文件不存在
            pickle.UnpicklingError: 如果文件格式不正确或损坏
            
        Note:
            - 使用pickle进行反序列化，恢复完整的对象状态
            - 加载的预处理器应该与保存时的版本兼容
            - 文件路径应该存在且可读
            - 通常与save_preprocessor方法配对使用
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"预处理器已从{filepath}加载")
        return preprocessor

if __name__ == "__main__":
    # 测试数据预处理
    preprocessor = HotelDataPreprocessor()
    
    # 加载和预处理数据
    features_df = preprocessor.load_and_preprocess_data('../03_数据文件/hotel_bookings.csv')
    
    # 保存预处理后的数据
    features_df.to_csv('../03_数据文件/processed_features.csv', index=False)
    print("预处理数据已保存到：processed_features.csv")
    
    # 保存预处理器
    preprocessor.save_preprocessor('../02_训练模型/preprocessor.pkl')
    
    # 测试特征准备
    test_features = preprocessor.prepare_bnn_features(features_df.iloc[:1], action=2)
    print(f"BNN输入特征维度：{test_features.shape}")
    print("数据预处理测试完成！")