import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import pickle
import warnings
from typing import Optional, Dict, List, Any, Union, Tuple
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
    
    def split_data(self, features_df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练、验证和测试数据
        
        按照时间顺序将数据划分为训练集、验证集和测试集，确保数据的时间序列特性。
        这种划分方式对于时间序列预测任务非常重要，避免了数据泄露问题。
        
        Args:
            features_df (pd.DataFrame): 完整的特征数据框
            train_ratio (float, optional): 训练集比例，默认为0.7
            val_ratio (float, optional): 验证集比例，默认为0.15
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 训练集、验证集、测试集
            
        Note:
            - 按照时间顺序划分，避免未来信息泄露
            - 测试集比例 = 1 - train_ratio - val_ratio
            - 适用于时间序列预测任务
            - 数据集大小会打印输出，便于调试
        """
        
        n = len(features_df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = features_df[:train_size]
        val_data = features_df[train_size:train_size + val_size]
        test_data = features_df[train_size + val_size:]
        
        print(f"数据划分完成：")
        print(f"训练集：{len(train_data)}条记录")
        print(f"验证集：{len(val_data)}条记录")
        print(f"测试集：{len(test_data)}条记录")
        
        return train_data, val_data, test_data
    
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