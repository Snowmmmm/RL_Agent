#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 标准库导入
import pickle
import random
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import numpy as np
import pandas as pd
import torch
from scipy import stats

# 本地模块导入
from config import BQL_CONFIG
from training_monitor import get_training_monitor

class HotelEnvironment:
    """
    酒店环境模拟器
    
    模拟酒店房间的动态定价环境，支持库存管理、需求预测、收益计算等功能。
    环境考虑了季节性、工作日类型、库存水平等因素对需求的影响。
    
    主要特性：
    - 多阶段库存管理：跟踪未来多天的可售房间数量
    - 动态需求预测：集成BNN模型进行需求预测
    - 收益优化：考虑当日收益和未来预期收益
    - 风险惩罚：基于预测方差的风险控制
    - 季节性调整：根据淡旺季调整定价策略
    
    状态空间：
    - inventory_level: 库存水平（离散化：0=极少，4=充足）
    - inventory_raw: 原始库存数量
    - future_inventory: 未来库存数组
    - day: 当前天数
    - season: 季节（0=淡季，1=平季，2=旺季）
    - weekday: 工作日类型（0=工作日，1=周末）
    
    动作空间：
    - 36个定价组合：线上6档 × 线下6档
    - 线上价格档位：[80, 90, 100, 110, 120, 130]元
    - 线下价格档位：[90, 105, 120, 135, 150, 165]元
    
    奖励函数：
    - 总收益 = 当日收益 + 未来预期收益
    - 风险惩罚 = λ × 预测方差
    - 最终奖励 = 总收益 - 风险惩罚
    
    Attributes:
        initial_inventory (int): 初始库存数量
        max_stay_nights (int): 最大入住天数
        cost_per_room (int): 每间房间的成本
        beta_distribution (List[float]): β系数分布，表示不同入住天数的比例
        future_inventory (List[int]): 未来库存数组
        current_inventory (int): 当前库存数量
        day (int): 当前天数
        total_revenue (float): 总收益
        total_bookings (int): 总预订数量
        daily_history (List[Dict]): 每日历史记录
        
    Note:
        - 状态编码：库存等级(0-4) × 季节(0-2) × 日期类型(0-1) = 30种状态
        - 价格档位：6档定价策略，间隔30元，覆盖60-210元区间
        - 风险惩罚系数按季节调整：旺季0.1，平季0.25，淡季0.5
        - 库存更新使用β系数分布，反映不同入住天数的影响
        - 支持90天周期模拟，支持自定义起始日期
    """
    
    def __init__(self, initial_inventory: int = None, max_stay_nights: int = 5, 
                 cost_per_room: int = 20, beta_distribution: Optional[List[float]] = None):
        
        # 从配置文件读取客房数量，如果没有显式传递参数
        if initial_inventory is None:
            from config import ENV_CONFIG
            self.initial_inventory = ENV_CONFIG['initial_inventory']
        else:
            self.initial_inventory = initial_inventory
        self.max_stay_nights = max_stay_nights # 最大入住天数
        self.cost_per_room = cost_per_room # 每间客房的成本
        self.beta_distribution = beta_distribution or [0.2] * max_stay_nights # Beta分布参数，用于模拟未来需求
        
        # 初始化未来库存数组：s_t^1, s_t^2, ..., s_t^{T-t+1}
        # 跟踪当前及未来max_stay_nights天的可售客房量
        self.future_inventory = None
        
        # 初始化4+1队列系统：4个需求队列和1个价格队列，每个队列记录7天数据
        self.demand_queues = {
            'online_booked': deque(maxlen=7),    # 线上用户预定需求队列
            'online_actual': deque(maxlen=7),    # 线上用户实际需求队列
            'offline_booked': deque(maxlen=7),   # 线下用户预定需求队列
            'offline_actual': deque(maxlen=7)     # 线下用户实际需求队列
        }
        self.price_queue = deque(maxlen=7)       # 价格队列（兼容旧逻辑）
        self.price_queue_online = deque(maxlen=7)    # 线上价格队列
        self.price_queue_offline = deque(maxlen=7)   # 线下价格队列
        
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """
        重置酒店环境到初始状态
        
        将酒店环境重置到初始状态，包括：
        1. 恢复初始库存数量
        2. 重置天数计数器
        3. 清空收益和预订统计
        4. 初始化历史记录
        5. 设置未来库存数组
        6. 清空4+1队列系统
        
        Returns:
            Dict[str, Any]: 初始状态字典，包含库存水平、季节、工作日类型等信息
            
        状态包含字段：
        - inventory_level: 库存水平（0=极少，1=较少，2=中等，3=较多，4=充足）
        - inventory_raw: 原始库存数量
        - future_inventory: 未来库存数组
        - day: 当前天数
        - season: 季节（0=淡季，1=平季，2=旺季）
        - weekday: 工作日类型（0=工作日，1=周末）
            
        Note:
            - 每次新的episode开始时调用此方法
            - 返回的状态用于强化学习智能体的初始观察
            - 历史记录用于后续分析和可视化
            - 状态编码：库存等级(0-4) × 季节(0-2) × 日期类型(0-1) = 30种状态
        """
        self.current_inventory = self.initial_inventory
        self.day = 0
        self.total_revenue = 0
        self.total_bookings = 0
        self.daily_history = []
        
        # 初始化未来库存数组：s_t^1, s_t^2, ..., s_t^{max_stay_nights+1}
        # 第t天起始时刻观察到当前及未来每一天的可售客房量
        self.future_inventory = [self.initial_inventory] * (self.max_stay_nights + 1)
        
        # 清空4+1队列系统
        for queue_name in self.demand_queues:
            self.demand_queues[queue_name].clear()
        self.price_queue.clear()
        self.price_queue_online.clear()
        self.price_queue_offline.clear()
        
        return self._get_state()
    
    def _get_state(self) -> Dict[str, Any]:
        """
        获取当前酒店环境状态
        
        计算当前环境状态，包括库存水平、季节、工作日类型等信息。
        
        Returns:
            Dict[str, Any]: 当前状态字典，包含以下字段：
                - inventory_level: 库存水平（0=极少，1=较少，2=中等，3=较多，4=充足）
                - inventory_raw: 原始库存数量
                - future_inventory: 未来库存数组
                - day: 当前天数
                - season: 季节（0=淡季，1=平季，2=旺季）
                - weekday: 工作日类型（0=工作日，1=周末）
                
        状态计算逻辑：
        1. 库存水平：根据当前库存数量离散化为5个等级
        2. 季节判断：基于当前天数计算月份，按季节划分规则确定
        3. 工作日类型：基于当前天数计算星期，周六日为周末
        
        Note:
            - 库存水平离散化：0-20=极少，21-40=较少，41-60=中等，61-80=较多，81-100=充足
            - 季节划分：11-2月=淡季，6-8月=旺季，其他=平季
            - 工作日类型：假设第0天为周一，周六日(5,6)为周末
            - 状态编码：库存等级(0-4) × 季节(0-2) × 日期类型(0-1) = 30种状态
        """
        # 当前库存离散化（s_t^1）- 注释掉库存数量区分
        current_inventory = self.future_inventory[0] if self.future_inventory else self.current_inventory
        
        # 注释掉库存等级区分，统一使用固定值
        # if current_inventory <= 20:
        #     inventory_level = 0
        # elif current_inventory <= 40:
        #     inventory_level = 1
        # elif current_inventory <= 60:
        #     inventory_level = 2
        # elif current_inventory <= 80:
        #     inventory_level = 3
        # else:
        #     inventory_level = 4
        
        # 统一使用库存等级2（中等库存）
        inventory_level = 3
        
        # 根据月份确定季节（方案要求：11-2月→淡季0，3-5/9-10月→平季1，6-8月→旺季2）
        month = (self.day // 30) % 12 + 1  # 简化：假设每月30天
        if month in [11, 12, 1, 2]:  # 11-2月：淡季
            season = 0
        elif month in [6, 7, 8]:  # 6-8月：旺季
            season = 2
        else:  # 3-5月, 9-10月：平季
            season = 1
        
        # 确定日期类型（工作日/周末）- 简化：假设第0天为周一
        weekday_type = 1 if (self.day % 7) in [5, 6] else 0  # 周六(5)、周日(6)为周末
        
        return {
            'inventory_level': inventory_level,
            'inventory_raw': current_inventory,
            'future_inventory': self.future_inventory.copy() if self.future_inventory else [],
            'day': self.day,
            'season': season,
            'weekday': weekday_type
        }
    
    def step(self, action: Union[int, List[int]], 
             ngboost_predictor_online_booked: Optional[Any] = None,
             ngboost_predictor_online_actual: Optional[Any] = None,
             ngboost_predictor_offline_booked: Optional[Any] = None,
             ngboost_predictor_offline_actual: Optional[Any] = None,
             date_features: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步酒店定价决策
        
        根据给定的定价动作，模拟一天的酒店运营过程，包括：
        1. 确定定价：将动作索引转换为具体价格（支持线上线下双价格）
        2. 需求预测：使用线上和线下BNN模型预测需求分布，并相加结果
        3. 预订处理：根据库存限制确定实际预订量（优先满足线下用户）
        4. 收益计算：计算当日收益和未来预期收益
        5. 风险惩罚：基于预测方差添加风险惩罚
        6. 库存更新：根据β系数更新未来库存
        7. 状态转移：获取新的环境状态
        
        Args:
            action (Union[int, List[int]]): 定价动作索引（int为单价格，List[int]为双价格[线上, 线下]）
            ngboost_predictor_online_booked (Optional[Any], optional): 线上用户预定需求NGBoost预测器
            ngboost_predictor_online_actual (Optional[Any], optional): 线上用户实际需求NGBoost预测器
            ngboost_predictor_offline_booked (Optional[Any], optional): 线下用户预定需求NGBoost预测器
            ngboost_predictor_offline_actual (Optional[Any], optional): 线下用户实际需求NGBoost预测器
            date_features (Optional[np.ndarray], optional): 日期特征数据
            
        Returns:
            Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: 
                - 新状态（包含库存、季节、工作日等信息）
                - 奖励（收益减去风险惩罚）
                - 是否结束（90天周期结束）
                - 额外信息（预测需求、方差、实际预订、收益等）
                
        收益计算逻辑：
        1. 当日收益 = (价格-成本) × 实际预订量 × 1.0（当天入住系数）
        2. 未来预期收益 = (价格-成本) × 预测需求 × Σβ₁₋₄（未来入住系数和）
        3. 总收益 = 当日收益 + 未来预期收益
        4. 风险惩罚 = λ × 预测方差（按季节调整λ系数）
        5. 最终奖励 = 总收益
                
        Note:
            - 价格档位：线上[80,90,100,110,120,130]，线下[90,105,120,135,150,165]
            - 需求预测为线上和线下BNN预测结果相加
            - 收益计算考虑当日入住和未来预期入住
            - 风险惩罚系数按季节调整（旺季0.1，平季0.25，淡季0.5）
            - 库存更新使用β系数分布，反映不同入住天数的影响
            - 支持90天周期模拟，episode在90天时结束
        """
        # print(f" \n {'+'*15}一个episode开始{'+'*15}")
        # 定价动作（线上线下双价格映射）
        # 从配置文件读取定价档位
        from config import ENV_CONFIG
        online_prices = ENV_CONFIG['online_price_levels']  # 线上价格档位（6个动作）
        offline_prices = ENV_CONFIG['offline_price_levels']  # 线下价格档位（6个动作）
        
        # 处理36个动作组合：action_idx = online_idx * 6 + offline_idx
        if hasattr(action, 'item'):
            action_idx = int(action.item())
        else:
            action_idx = int(action)
        
        # 36个动作组合映射到线上线下价格索引
        online_idx = action_idx // 6  # 线上价格索引 (0-5)
        offline_idx = action_idx % 6   # 线下价格索引 (0-5)
        
        price_online = online_prices[online_idx]
        price_offline = offline_prices[offline_idx]
        
        # 统一使用线上价格进行历史记录（兼容旧逻辑）
        price = price_online
        
        # 分别获取线上和线下的预定需求与实际需求预测
        predicted_booked_demand_online = 0
        predicted_booked_variance_online = 0
        predicted_actual_demand_online = 0
        predicted_actual_variance_online = 0
        predicted_booked_demand_offline = 0
        predicted_booked_variance_offline = 0
        predicted_actual_demand_offline = 0
        predicted_actual_variance_offline = 0
        
        # 确定用于预测的动作索引（36个动作组合映射）
        action_online = online_idx  # 线上价格索引 (0-5)
        action_offline = offline_idx  # 线下价格索引 (0-5)
        
        if date_features is not None:
            # 获取线上用户预定需求NGBoost预测（使用线上价格）
            if ngboost_predictor_online_booked is not None:
                booked_demand_online, booked_variance_online = ngboost_predictor_online_booked(
                    date_features, action_online, record_env_changes=True, user_type='online', demand_type='booked',
                    demand_queues=self.demand_queues, price_queue=self.price_queue_online
                )
                predicted_booked_demand_online = booked_demand_online
                predicted_booked_variance_online = booked_variance_online

            # 获取线上用户实际需求NGBoost预测（使用线上价格）
            if ngboost_predictor_online_actual is not None:
                actual_demand_online, actual_variance_online = ngboost_predictor_online_actual(
                    date_features, action_online, record_env_changes=True, user_type='online', demand_type='actual',
                    demand_queues=self.demand_queues, price_queue=self.price_queue_online
                )
                predicted_actual_demand_online = actual_demand_online
                predicted_actual_variance_online = actual_variance_online

            # 获取线下用户预定需求NGBoost预测（使用线下价格）
            if ngboost_predictor_offline_booked is not None:
                booked_demand_offline, booked_variance_offline = ngboost_predictor_offline_booked(
                    date_features, action_offline, record_env_changes=True, user_type='offline', demand_type='booked',
                    demand_queues=self.demand_queues, price_queue=self.price_queue_offline
                )
                predicted_booked_demand_offline = booked_demand_offline
                predicted_booked_variance_offline = booked_variance_offline

            # 获取线下用户实际需求NGBoost预测（使用线下价格）
            if ngboost_predictor_offline_actual is not None:
                actual_demand_offline, actual_variance_offline = ngboost_predictor_offline_actual(
                    date_features, action_offline, record_env_changes=True, user_type='offline', demand_type='actual',
                    demand_queues=self.demand_queues, price_queue=self.price_queue_offline
                )
                predicted_actual_demand_offline = actual_demand_offline
                predicted_actual_variance_offline = actual_variance_offline

        
        # 线上用户需求采样
        booked_demand_online_sampled = max(0, int(np.random.normal(predicted_booked_demand_online, np.sqrt(predicted_booked_variance_online))))
        actual_demand_online_sampled = max(0, int(np.random.normal(predicted_actual_demand_online, np.sqrt(predicted_actual_variance_online))))
        
        # 确保预定需求采样值大于实际需求采样值
        while booked_demand_online_sampled <= actual_demand_online_sampled:
            booked_demand_online_sampled = max(0, int(np.random.normal(predicted_booked_demand_online, np.sqrt(predicted_booked_variance_online))))
            actual_demand_online_sampled = max(0, int(np.random.normal(predicted_actual_demand_online, np.sqrt(predicted_actual_variance_online))))
        
        # 线下用户需求采样
        booked_demand_offline_sampled = max(0, int(np.random.normal(predicted_booked_demand_offline, np.sqrt(predicted_booked_variance_offline))))
        actual_demand_offline_sampled = max(0, int(np.random.normal(predicted_actual_demand_offline, np.sqrt(predicted_actual_variance_offline))))
        
        # 确保预定需求采样值大于实际需求采样值
        while booked_demand_offline_sampled <= actual_demand_offline_sampled:
            booked_demand_offline_sampled = max(0, int(np.random.normal(predicted_booked_demand_offline, np.sqrt(predicted_booked_variance_offline))))
            actual_demand_offline_sampled = max(0, int(np.random.normal(predicted_actual_demand_offline, np.sqrt(predicted_actual_variance_offline))))
        
        # # 总预定需求 = 线上预定需求 + 线下预定需求
        # predicted_booked_demand = predicted_booked_demand_online + predicted_booked_demand_offline
        # # 总实际需求 = 线上实际需求 + 线下实际需求
        # predicted_actual_demand = predicted_actual_demand_online + predicted_actual_demand_offline
        # # 总预定方差 = 线上预定方差 + 线下预定方差（假设独立）
        # predicted_booked_variance = predicted_booked_variance_online + predicted_booked_variance_offline
        # # 总实际方差 = 线上实际方差 + 线下实际方差（假设独立）
        # predicted_actual_variance = predicted_actual_variance_online + predicted_actual_variance_offline
        
        # 采样后的总需求
        total_booked_demand_sampled = booked_demand_online_sampled + booked_demand_offline_sampled
        total_actual_demand_sampled = actual_demand_online_sampled + actual_demand_offline_sampled
        
        # 更新4+1队列系统：将当前需求值和价格添加到队列中
        self.demand_queues['online_booked'].appendleft(booked_demand_online_sampled)
        self.demand_queues['online_actual'].appendleft(actual_demand_online_sampled)
        self.demand_queues['offline_booked'].appendleft(booked_demand_offline_sampled)
        self.demand_queues['offline_actual'].appendleft(actual_demand_offline_sampled)
        self.price_queue.appendleft(price)  # 使用线上价格进行历史记录（兼容旧逻辑）
        self.price_queue_online.appendleft(price_online)    # 记录线上价格
        self.price_queue_offline.appendleft(price_offline)  # 记录线下价格
        
        # 分别计算线上线下用户的成交量（受库存限制）
        current_original_inventory = self.initial_inventory
        
        # 优先满足线下用户（修改库存分配逻辑）
        actual_bookings_offline = min(actual_demand_offline_sampled, current_original_inventory)
        # 剩余库存
        remaining_inventory = max(0, current_original_inventory - actual_bookings_offline)
        # 线上用户成交量（基于剩余库存）
        actual_bookings_online = min(actual_demand_online_sampled, remaining_inventory)
        # 总成交量
        actual_bookings = actual_bookings_online + actual_bookings_offline
        
        # 分别计算线上线下用户的当日收益（使用各自的价格）
        # print(f"agent选择价格为{price},成本为{self.cost_per_room}")
        
        # 在成交确认后计算需求比例调整因子
        # 分别计算线上线下用户的需求比例调整因子
        demand_ratio_online = 1.0
        # if booked_demand_online_sampled > self.initial_inventory:
        #     demand_ratio_online = actual_demand_online_sampled / booked_demand_online_sampled if booked_demand_online_sampled > 0 else 1.0
        
        demand_ratio_offline = 1.0
        # if booked_demand_offline_sampled > self.initial_inventory:
        #     demand_ratio_offline = actual_demand_offline_sampled / booked_demand_offline_sampled if booked_demand_offline_sampled > 0 else 1.0
        
        # 总需求比例调整因子（用于历史记录）
        demand_ratio = 1.0
        # if total_booked_demand_sampled > self.initial_inventory:
        #     demand_ratio = total_actual_demand_sampled / total_booked_demand_sampled if total_booked_demand_sampled > 0 else 1.0
        
        today_revenue_online = (price_online - self.cost_per_room) * actual_bookings_online * demand_ratio_online
        today_revenue_offline = (price_offline - self.cost_per_room) * actual_bookings_offline * demand_ratio_offline
        
        # 总当日收益 = 线上当日收益 + 线下当日收益
        today_revenue = today_revenue_online + today_revenue_offline
        
        # # 分别计算线上线下用户的未来预期收益
        # future_expected_revenue_online = 0
        # future_expected_revenue_offline = 0
        # if self.beta_distribution and len(self.beta_distribution) > 1:
        #     # β[1]到β[n-1]表示未来入住的比例
        #     future_beta_sum = sum(self.beta_distribution[1:])
        #     # 线上用户未来预期收益
        #     future_expected_revenue_online = (price - self.cost_per_room) * booked_demand_online_sampled * future_beta_sum
        #     # 线下用户未来预期收益
        #     future_expected_revenue_offline = (price - self.cost_per_room) * booked_demand_offline_sampled * future_beta_sum
        
        # 总未来预期收益 = 线上未来收益 + 线下未来收益
        # future_expected_revenue = future_expected_revenue_online + future_expected_revenue_offline
        
        # # 总收益 = 当日收益 + 未来预期收益
        # total_revenue = today_revenue + future_expected_revenue

        # 总收益 = 当日收益
        total_revenue = today_revenue
        # print(f"总收益{total_revenue}")
        
        # # 动态风险惩罚系数（按季节调整）
        # if ngboost_predictor_online_booked is not None or ngboost_predictor_offline_booked is not None:
        #     # 获取当前状态信息
        #     state_info = self._get_state()
            
        #     # 获取当前季节
        #     season = state_info.get('season', 1)  # 默认为平季
            
        #     # 按季节调整风险系数
        #     if season == 2:  # 旺季
        #         lambda_coef = 0.1
        #     elif season == 0:  # 淡季
        #         lambda_coef = 0.5
        #     else:  # 平季
        #         lambda_coef = 0.25
                
        #     risk_penalty = lambda_coef * predicted_actual_variance  
        # else:
        #     risk_penalty = 0.0
        
        # # 总奖励 = 总收益 - 风险惩罚
        # reward = total_revenue - risk_penalty

        # 总奖励 = 总收益 (去除风险惩罚)
        reward = total_revenue
        
        # 更新库存
        self._update_inventory(actual_bookings)
        
        # 更新统计
        self.total_revenue += total_revenue
        self.total_bookings += actual_bookings
        self.day += 1
        
        # 记录历史
        self.daily_history.append({
            'day': self.day,
            'price': price,  # 线上价格用于历史记录（兼容旧逻辑）
            'price_online': price_online,  # 线上价格
            'price_offline': price_offline,  # 线下价格
            # 'predicted_booked_demand': predicted_booked_demand,
            # 'predicted_booked_variance': predicted_booked_variance,
            # 'predicted_actual_demand': predicted_actual_demand,
            # 'predicted_actual_variance': predicted_actual_variance,
            'actual_demand': total_actual_demand_sampled,
            'actual_bookings': actual_bookings,
            'actual_bookings_online': actual_bookings_online,  # 新增：线上成交量
            'actual_bookings_offline': actual_bookings_offline,  # 新增：线下成交量
            'demand_ratio': demand_ratio,
            'inventory_before': self.current_inventory + actual_bookings,
            'inventory_after': self.current_inventory,
            'revenue': total_revenue,
            'revenue_online': today_revenue_online,  # 新增：线上收益
            'revenue_offline': today_revenue_offline,  # 新增：线下收益
            # 'risk_penalty': risk_penalty,
            'reward': reward
        })
        
        # 获取新状态
        new_state = self._get_state()
        
        # 判断episode是否结束
        done = (self.day >= 90)  # 90天规划周期，不考虑库存耗尽

        # print(f"{'-'*15}一个episode结束{'-'*15}")
        
        return new_state, reward, done, {
            # 'predicted_booked_demand': predicted_booked_demand,
            # 'predicted_booked_variance': predicted_booked_variance,
            # 'predicted_actual_demand': predicted_actual_demand,
            # 'predicted_actual_variance': predicted_actual_variance,
            'actual_demand': total_actual_demand_sampled,
            'actual_bookings': actual_bookings,
            'actual_bookings_online': actual_bookings_online,
            'actual_bookings_offline': actual_bookings_offline,
            'demand_ratio': demand_ratio,
            'revenue': total_revenue,
            # 添加四个NGBoost预测器的详细预测结果
            'predicted_booked_demand_online': predicted_booked_demand_online,
            'predicted_actual_demand_online': predicted_actual_demand_online,
            'predicted_booked_demand_offline': predicted_booked_demand_offline,
            'predicted_actual_demand_offline': predicted_actual_demand_offline
        }
    
    def _update_inventory(self, bookings: int) -> None:
        """
        更新酒店库存状态
        
        根据方案文档的库存转移规则更新未来库存：
        inv_{原始, t+1+i} = inv_{原始, t+i} - β_i × d_t
        
        其中：
        - β系数表示入住1-n天的顾客比例分布
        - d_t是第t天的实际预订量
        - 库存数组向前滚动，模拟时间推移
        
        Args:
            bookings (int): 第t天的实际预订量
            
        Returns:
            None
            
        库存更新逻辑：
        1. 根据β系数从未来各天库存中扣除相应比例的预订
        2. 确保库存不会为负数（max(0, ...)保护）
        3. 滚动更新库存数组，模拟时间推移
        4. 新一天的库存初始化为初始库存值
            
        Note:
            - β系数分布表示不同入住天数的顾客比例
            - 从未来第i天的库存中扣除β_i × bookings
            - 库存数组向前滚动，最后一天初始化为初始库存
            - 支持多阶段库存管理和时间推移模拟
        """
        # 根据方案文档：β系数表示入住1-n天的顾客比例
        # 库存转移规则：inv_{原始, t+1+i} = inv_{原始, t+i} - β_i * d_t
        
        
        if self.future_inventory and len(self.future_inventory) >= len(self.beta_distribution):
            # 根据β系数从未来库存中扣除预订影响
            for i in range(len(self.beta_distribution)):
                if i < len(self.future_inventory):
                    # 从未来第i天的库存中扣除β_i * bookings
                    deduction = self.beta_distribution[i] * bookings
                    self.future_inventory[i] = max(0, self.future_inventory[i] - deduction)
            
            # 更新当前库存为future_inventory[0]（当天的原始库存）
            self.current_inventory = self.future_inventory[0]
            
            # 滚动更新库存数组：将数组向前移动一天
            # 新的最后一天库存等于初始库存
            self.future_inventory = self.future_inventory[1:] + [self.initial_inventory]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取酒店环境运行统计信息
        
        计算并返回酒店环境的运行统计信息，包括总天数、总收益、
        平均入住率、平均价格、需求满足率等关键指标。
        
        Returns:
            Dict[str, float]: 统计信息字典，包含以下字段：
                - total_days: 总运行天数
                - total_revenue: 总收益
                - total_bookings: 总预订数量
                - average_occupancy_rate: 平均入住率
                - average_daily_revenue: 平均每日收益
                - average_price: 平均价格
                - total_demand: 总需求
                - demand_satisfaction_rate: 需求满足率
                
        统计计算逻辑：
        1. 总天数：从daily_history中获取最大天数
        2. 总收益：累计所有天的收益
        3. 平均入住率：总预订量 / (初始库存 × 总天数)
        4. 平均价格：所有天价格的平均值
        5. 需求满足率：实际预订量 / 总需求量
                
        Note:
            - 基于daily_history数据计算统计信息
            - 入住率计算考虑初始库存和总天数
            - 需求满足率反映库存限制对需求的影响
            - 所有统计指标都基于历史运行数据
        """
        if not self.daily_history:
            return {}
        
        df_history = pd.DataFrame(self.daily_history)
        
        return {
            'total_days': self.day,
            'total_revenue': self.total_revenue,
            'total_bookings': self.total_bookings,
            'average_occupancy_rate': df_history['actual_bookings'].sum() / (self.initial_inventory * self.day) if self.day > 0 else 0,
            'average_daily_revenue': self.total_revenue / self.day if self.day > 0 else 0,
            'average_price': df_history['price'].mean(),
            'total_demand': df_history['actual_demand'].sum(),
            'demand_satisfaction_rate': df_history['actual_bookings'].sum() / df_history['actual_demand'].sum() if df_history['actual_demand'].sum() > 0 else 0
        }

class QLearningAgent:
    """
    Q-learning智能体
    
    实现Q-learning算法的智能体，用于酒店动态定价决策。
    支持ε-贪心探索策略、UCB探索增强、状态访问统计等功能。
    
    主要特性：
    - ε-贪心探索：平衡探索和利用
    - UCB增强：优先选择访问次数较少的状态-动作对
    - 状态离散化：将连续状态映射到离散状态空间
    - 访问统计：跟踪状态和动作访问次数
    - Q值更新：使用TD学习更新Q值
    
    状态空间：
    - 总状态数：30（库存等级5 × 季节3 × 日期类型2）
    - 状态编码：inventory_level × 6 + season × 2 + weekday
    
    动作空间：
    - 总动作数：36（线上6档 × 线下6档）
    - 动作映射：action_idx = online_idx * 6 + offline_idx
    - 线上价格档位：[80, 90, 100, 110, 120, 130]元
    - 线下价格档位：[90, 105, 120, 135, 150, 165]元
    
    学习参数：
    - 学习率：控制Q值更新速度
    - 折扣因子：权衡即时奖励和未来奖励
    - ε衰减：逐步减少探索概率
    
    Attributes:
        n_states (int): 状态数量（默认30）
        n_actions (int): 动作数量（默认6）
        learning_rate (float): 学习率
        discount_factor (float): 折扣因子
        epsilon_start (float): 初始探索概率
        epsilon_end (float): 最终探索概率
        epsilon_decay_steps (int): ε衰减步数
        q_table (Dict): Q值表，键为状态，值为动作Q值数组
        state_visit_count (Dict): 状态访问计数
        state_action_visit_count (Dict): 状态-动作访问计数
        training_history (List): 训练历史记录
        
    Note:
        - 使用defaultdict自动初始化Q值和访问计数
        - 支持UCB探索策略，优先探索访问次数少的状态-动作对
        - ε值随训练episode线性衰减
        - 状态离散化支持库存、季节、工作日类型组合
    """
    
    def __init__(self, n_states: int = 30, n_actions: int = None, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon_start: float = 0.8, epsilon_end: float = 0.01, epsilon_decay_steps: int = 50):
        
        # 如果未指定动作数，从配置文件读取
        if n_actions is None:
            from config import RL_CONFIG
            n_actions = RL_CONFIG['n_actions']  # 36个动作组合（线上6价格 × 线下6价格）
        
        self.n_states = n_states
        self.n_actions = n_actions  # 6×6=36个动作组合（线上6价格 × 线下6价格）
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        
        # Q表
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # 状态访问计数
        self.state_visit_count = defaultdict(int)
        
        # 状态-动作访问计数（用于UCB探索）
        self.state_action_visit_count = defaultdict(int)
        
        # 训练历史
        self.training_history = []
    
    def get_epsilon(self, episode: int) -> float:
        """获取当前的epsilon值 - 使用更快的指数衰减策略"""
        if episode >= self.epsilon_decay_steps:
            return self.epsilon_end
        else:
            # 使用更快的指数衰减策略，使探索率快速下降
            # epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-episode / decay_rate)
            decay_rate = self.epsilon_decay_steps / 2  # 进一步加快衰减速率
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode / decay_rate)
            return epsilon
    
    def discretize_state(self, state_info: Dict[str, Any], season: int, weekday: int) -> int:
        """离散化状态 - 基于当前库存、季节和日期类型"""
        inventory_level = state_info['inventory_level']
        
        # 计算状态索引
        # inventory_level: 0-4 (5个等级)
        # season: 0-2 (3个季节)
        # weekday: 0-1 (工作日/周末)
        state_index = inventory_level * 6 + season * 2 + weekday
        
        return min(state_index, self.n_states - 1)  # 防止越界
    
    def select_action(self, state: Union[List, np.ndarray, int], episode: int) -> int:
        """选择动作（epsilon-greedy + 增强UCB探索策略）"""
        epsilon = self.get_epsilon(episode)
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        q_values = self.q_table[state_key]
        
        # 36个动作组合：action_idx = online_idx * 6 + offline_idx
        if random.random() < epsilon:
            # 增强探索策略：结合UCB和随机探索
            visit_counts = np.array([self.state_action_visit_count.get((state_key, a), 0) for a in range(self.n_actions)])
            
            # 如果存在完全未探索的动作（访问次数为0），优先选择这些动作
            unvisited_actions = np.where(visit_counts == 0)[0]
            if len(unvisited_actions) > 0:
                # 如果有未探索的动作，随机选择一个
                return random.choice(unvisited_actions)
            
            # 否则使用UCB策略选择访问次数最少的动作
            min_visits = np.min(visit_counts)
            least_visited_actions = np.where(visit_counts == min_visits)[0]
            
            if len(least_visited_actions) > 1:
                # 如果有多个最少访问的动作，选择Q值较高的那个
                q_values_least = q_values[least_visited_actions]
                best_idx = np.argmax(q_values_least)
                return least_visited_actions[best_idx]
            else:
                return least_visited_actions[0]
        else:
            # 利用：选择Q值最大的动作
            # 如果有多个最大值，优先选择访问次数较少的
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            
            if len(best_actions) > 1:
                # 在最佳动作中选择访问次数最少的
                visit_counts = np.array([self.state_action_visit_count.get((state_key, a), 0) for a in best_actions])
                least_visited_idx = np.argmin(visit_counts)
                return best_actions[least_visited_idx]
            else:
                return best_actions[0]
    
    def update_q_table(self, state: Union[List, np.ndarray, int], action: int, reward: float, next_state: Union[List, np.ndarray, int], done: bool) -> float:
        """更新Q表"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
        
        # 更新访问计数
        self.state_visit_count[state_key] += 1
        self.state_action_visit_count[(state_key, action)] += 1
        
        # 当前Q值
        current_q = self.q_table[state_key][action]
        
        # 下一个状态的最大Q值
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # 更新Q表
        self.q_table[state_key][action] = new_q
        
        return new_q
    
    def train_episode(self, env: HotelEnvironment, online_booked_predictor: Optional[Any] = None, online_actual_predictor: Optional[Any] = None, offline_booked_predictor: Optional[Any] = None, offline_actual_predictor: Optional[Any] = None, date_features: Optional[pd.DataFrame] = None, episode: int = 0) -> Tuple[float, int]:
        """
        训练一个episode（完整的酒店定价周期）
        
        功能描述：
        执行完整的酒店定价决策周期，从环境重置到episode结束，记录完整的训练和交互过程。
        支持四个NGBoost预测器集成（双需求模型）、多阶段收益计算、详细日志记录等功能。
        
        参数:
            env (HotelEnvironment): 酒店环境实例
            online_booked_predictor (Optional[Any]): 线上用户预定需求NGBoost预测器包装器
            online_actual_predictor (Optional[Any]): 线上用户实际需求NGBoost预测器包装器
            offline_booked_predictor (Optional[Any]): 线下用户预定需求NGBoost预测器包装器
            offline_actual_predictor (Optional[Any]): 线下用户实际需求NGBoost预测器包装器
            date_features (Optional[pd.DataFrame]): 日期特征数据，包含季节、工作日等信息
            episode (int): 当前episode编号，用于控制探索率衰减
            
        返回值:
            Tuple[float, int]: (总奖励, 步数) - 当前episode的总奖励和执行的步数
            
        训练流程:
        1. 环境初始化：重置酒店环境到初始状态
        2. 状态获取：获取当前库存、季节、日期类型等状态信息
        3. 动作选择：基于ε-贪心策略选择定价动作
        4. 环境交互：执行定价决策，使用四个NGBoost预测器获取奖励和下一状态
        5. Q表更新：使用Q-learning算法更新价值函数
        6. 数据记录：记录每日决策、奖励、库存变化等信息
        7. 循环执行：重复步骤3-6直到episode结束
        8. 历史保存：将episode数据添加到训练历史中
        
        Note:
        - 支持最大200步限制，防止无限循环
        - 集成四个NGBoost预测器分别预测线上/线下用户的预定/实际需求
        - 记录详细的每日决策信息用于后续分析
        - 使用episode计数器控制探索率衰减
        - 支持多阶段收益计算（当日收益+未来预期收益）
        """
        state_info = env.reset()
        total_reward = 0.0  # 明确指定为float类型
        steps: int = 0
        
        # 初始化每日记录列表
        daily_rewards = []
        daily_inventory = []
        daily_prices = []
        
        # 获取季节和星期信息
        if date_features is not None:
            season = int(date_features['season'].iloc[0])
            weekday = int(date_features['is_weekend'].iloc[0])
        else:
            season = 0
            weekday = 0
        
        state = self.discretize_state(state_info, season, weekday)
        
        done = False
        while not done:
            # 选择动作
            action = self.select_action(state, episode)
            
            # 36个动作组合映射：action_idx = online_idx * 6 + offline_idx
            online_idx = action // 6  # 线上价格索引 (0-5)
            offline_idx = action % 6   # 线下价格索引 (0-5)
            
            online_prices = [80, 90, 100, 110, 120, 130]
            offline_prices = [90, 105, 120, 135, 150, 165]
            price_online = online_prices[online_idx]
            price_offline = offline_prices[offline_idx]
            price = price_online  # 使用线上价格进行历史记录
            
            # 获取当前库存（上一轮更新后的库存）
            current_inventory = state_info['inventory_raw']
            future_inventory = state_info.get('future_inventory', [])
            
            # 执行动作，使用四个NGBoost预测器
            next_state_info, reward, done, info = env.step(action, online_booked_predictor, online_actual_predictor, offline_booked_predictor, offline_actual_predictor, date_features)
            
            # 获取NGBoost预测的需求信息
            predicted_demand = info.get('predicted_demand', 0)
            predicted_variance = info.get('predicted_variance', 0)
            actual_bookings = info.get('actual_bookings', 0)
            
            # 获取收益分解信息 - 使用正确的计算逻辑
            beta_0 = env.beta_distribution[0] if env.beta_distribution else 1.0
            today_revenue = (price - env.cost_per_room) * actual_bookings * 1
            
            # 计算未来预期收益（从β₁开始）- 使用BNN预测需求，而不是实际成交量
            future_expected_revenue = 0
            if env.beta_distribution and len(env.beta_distribution) > 1:
                future_beta_sum = sum(env.beta_distribution[1:])  # β₁到β₄的和
                future_expected_revenue = (price - env.cost_per_room) * predicted_demand * future_beta_sum
            
            # # 减少详细输出，只在每10步或episode结束时显示
            # if steps % 10 == 0 or done:
                # print(f"第{steps+1}天 - 动作: {action}({price}元), 库存: {current_inventory}→{next_state_info['inventory_raw']}")
                # 计算总的预测需求
            #     total_predicted_booked_demand = info.get('predicted_booked_demand_online', 0) + info.get('predicted_booked_demand_offline', 0)
            #     # 计算线上和线下的实际预订
            #     total_predicted_actual_demand = info.get('predicted_actual_demand_online', 0) + info.get('predicted_actual_demand_offline', 0)
            #     print(f"        预测需求: {total_predicted_booked_demand:.1f}, 实际预订: {actual_bookings}, 奖励: {reward:.2f}")
            #     # 打印四个NGBoost预测器的详细预测结果
            #     print(f"        线上预定预测: {info.get('predicted_booked_demand_online', 0):.1f}, 线上实际预测: {info.get('predicted_actual_demand_online', 0):.1f}")
            #     print(f"        线下预定预测: {info.get('predicted_booked_demand_offline', 0):.1f}, 线下实际预测: {info.get('predicted_actual_demand_offline', 0):.1f}")
            #     # 打印实际预订的分解
            #     print(f"        线上实际预订: {info.get('actual_bookings_online', 0):.1f}, 线下实际预订: {info.get('actual_bookings_offline', 0):.1f}")
            
            # 离散化下一个状态
            next_state = self.discretize_state(next_state_info, season, weekday)
            
            # 更新Q表
            self.update_q_table(state, action, reward, next_state, done)
            
            # 记录每日信息
            daily_rewards.append(reward)
            daily_inventory.append(current_inventory)
            daily_prices.append(price)
            
            # 转移到下一个状态
            state = next_state
            state_info = next_state_info  # 更新状态信息，用于下一轮循环
            total_reward += reward
            steps += 1
            
            # 更新日期特征（模拟时间推移）
            if date_features is not None and steps < len(date_features) - 1:
                date_features = date_features.iloc[steps:steps+1].reset_index(drop=True)
                season = int(date_features['season'].iloc[0])
                weekday = int(date_features['is_weekend'].iloc[0])
            
            # 防止无限循环，设置最大步数限制
            if steps >= 200:  # 最多200步
                print(f"警告：达到最大步数限制({steps})，强制结束episode")
                break
        
        # 记录训练历史
        self.training_history.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.get_epsilon(episode),
            'final_inventory': next_state_info['inventory_raw'],  # 使用最终状态的实际库存
            'daily_rewards': daily_rewards,  # 记录每日奖励
            'daily_inventory': daily_inventory,  # 记录每日库存
            'daily_prices': daily_prices  # 记录每日价格
        })
        
        return total_reward, steps
    
    def get_policy(self) -> Dict[Any, int]:
        """
        获取当前策略（状态到动作的映射）
        
        功能描述：
        基于当前Q表生成确定性策略，为每个状态选择具有最高Q值的动作。
        
        返回值:
            Dict[Any, int]: 策略字典，键为状态，值为最优动作索引
            
        策略生成逻辑:
        - 遍历Q表中的所有状态
        - 对每个状态的Q值数组使用argmax获取最优动作
        - 返回状态到最优动作的映射字典
        
        Note:
        - 返回确定性策略（贪婪策略）
        - 如果Q表为空，返回空字典
        - 动作为0-7的整数，对应8个定价档位
        """
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = np.argmax(q_values)
        return policy
    
    def get_q_value_stats(self) -> Dict[str, float]:
        """
        获取Q值统计信息和学习进度指标
        
        功能描述：
        计算Q表的详细统计信息，包括Q值分布、探索覆盖率、学习进度等关键指标。
        
        返回值:
            Dict[str, float]: 统计信息字典，包含以下字段：
                - mean_q_value: Q值的平均值
                - std_q_value: Q值的标准差  
                - min_q_value: Q值的最小值
                - max_q_value: Q值的最大值
                - num_states: 已访问的状态数量
                - num_state_visits: 总状态访问次数
                - zero_q_percentage: 零值Q值所占百分比
                - exploration_coverage: 探索覆盖率（百分比）
                - explored_state_actions: 已探索的状态-动作对数量
                - total_state_actions: 总状态-动作对数量
                
        计算逻辑:
        1. 收集所有Q值并计算基本统计量（均值、标准差、极值）
        2. 统计零值Q值的数量和比例
        3. 计算探索覆盖率：已探索的状态-动作对 / 总状态-动作对
        4. 汇总状态访问和状态-动作访问计数
        
        Note:
        - 如果Q表为空，返回空字典
        - 探索覆盖率反映学习的完整性
        - 零值Q值比例可指示未充分探索的区域
        - 状态访问计数帮助分析学习重点
        """
        if not self.q_table:
            return {}
        
        all_q_values = []
        zero_q_count = 0
        total_q_entries = 0
        
        for q_values in self.q_table.values():
            all_q_values.extend(q_values)
            zero_q_count += np.sum(q_values == 0)
            total_q_entries += len(q_values)
        
        # 计算探索覆盖率
        explored_state_actions = sum(1 for count in self.state_action_visit_count.values() if count > 0)
        total_state_actions = len(self.q_table) * self.n_actions
        exploration_coverage = explored_state_actions / total_state_actions if total_state_actions > 0 else 0
        
        return {
            'mean_q_value': np.mean(all_q_values),
            'std_q_value': np.std(all_q_values),
            'min_q_value': np.min(all_q_values),
            'max_q_value': np.max(all_q_values),
            'num_states': len(self.q_table),
            'num_state_visits': sum(self.state_visit_count.values()),
            'zero_q_percentage': (zero_q_count / total_q_entries) * 100 if total_q_entries > 0 else 0,
            'exploration_coverage': exploration_coverage * 100,
            'explored_state_actions': explored_state_actions,
            'total_state_actions': total_state_actions
        }
    
    def save_agent(self, filepath: str) -> None:
        """
        保存智能体状态和训练历史到文件
        
        功能描述：
        将Q-learning智能体的完整状态保存到pickle文件，包括Q表、访问计数、训练历史、超参数等所有关键信息。
        
        参数:
            filepath (str): 保存文件的路径，应为.pkl文件
            
        保存内容:
        - q_table: Q值表，包含所有状态-动作对的Q值
        - state_visit_count: 状态访问计数统计
        - state_action_visit_count: 状态-动作对访问计数
        - training_history: 完整的训练历史记录
        - hyperparameters: 所有超参数设置
        
        文件格式:
        使用pickle格式保存，包含完整的智能体状态字典
        
        Note:
        - 自动将defaultdict转换为普通dict以便保存
        - 保存后打印确认信息
        - 文件可用于后续加载和继续训练
        - 包含所有必要的超参数信息
        """
        # 转换Q表为普通字典以便保存
        q_table_dict = dict(self.q_table)
        state_visit_dict = dict(self.state_visit_count)
        state_action_visit_dict = dict(self.state_action_visit_count)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'state_visit_count': state_visit_dict,
                'state_action_visit_count': state_action_visit_dict,
                'training_history': self.training_history,
                'hyperparameters': {
                    'n_states': self.n_states,
                    'n_actions': self.n_actions,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end,
                    'epsilon_decay_steps': self.epsilon_decay_steps
                }
            }, f)
        print(f"智能体已保存到：{filepath}")
    
    def load_agent(self, filepath: str) -> None:
        """
        从文件加载智能体状态和训练历史
        
        功能描述：
        从pickle文件恢复Q-learning智能体的完整状态，包括Q表、访问计数、训练历史等信息。
        
        参数:
            filepath (str): 加载文件的路径，应为之前保存的.pkl文件
            
        恢复内容:
        - q_table: Q值表，恢复所有状态-动作对的Q值
        - state_visit_count: 状态访问计数统计
        - state_action_visit_count: 状态-动作对访问计数  
        - training_history: 完整的训练历史记录
        - 超参数: 自动恢复保存时的超参数设置
        
        加载逻辑:
        1. 从pickle文件读取保存的数据字典
        2. 恢复Q表为defaultdict格式
        3. 恢复访问计数统计
        4. 恢复训练历史记录
        
        Note:
        - 自动将普通dict转换回defaultdict格式
        - 加载后打印确认信息
        - 可继续之前的训练过程
        - 保持与保存时相同的超参数设置
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 恢复Q表
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for state, q_values in data['q_table'].items():
            self.q_table[state] = q_values
        
        # 恢复其他属性
        self.state_visit_count = defaultdict(int, data['state_visit_count'])
        self.state_action_visit_count = defaultdict(int, data.get('state_action_visit_count', {}))
        self.training_history = data['training_history']
        
        print(f"智能体已从{filepath}加载")

class NGBoostPredictorWrapper:
    """
    NGBoost预测器包装器 - 自然梯度提升预测接口
    
    功能描述：
    为强化学习系统提供标准化的NGBoost预测接口，处理特征准备、预测调用、结果反标准化等流程。
    支持单次预测和批量预测，集成需求标准化器进行数据预处理和后处理。
    
    主要特性：
    - 特征标准化：自动处理输入特征的标准化
    - 预测包装：统一NGBoost预测调用接口  
    - 结果反标准化：将预测结果转换回原始尺度
    - 维度检查：确保输入输出维度正确性
    - 异常处理：处理预测过程中的各种边界情况
    
    Attributes:
        ngboost_trainer (Any): NGBoost训练器实例，包含预测功能
        preprocessor (Any): 预处理器，用于特征准备和标准化
        demand_scaler (Optional[Any]): 需求标准化器，用于反标准化预测结果
        
    Note:
        - 支持PyTorch张量操作和NumPy数组转换
        - 自动处理批次维度的添加和移除
        - 集成方差缩放保持预测不确定性
        - 提供简洁的__call__接口便于使用
    """
    
    def __init__(self, ngboost_trainer: Any, preprocessor: Any, demand_scaler: Optional[Any] = None):
        self.ngboost_trainer = ngboost_trainer
        self.preprocessor = preprocessor
        self.demand_scaler = demand_scaler
    
    def prepare_features_from_queues(self, date_features: pd.DataFrame, action: int, 
                                     demand_queues: dict, price_queue: deque, 
                                     user_type: str, demand_type: str) -> torch.Tensor:
        """
        从队列数据准备NGBoost模型的输入特征
        
        使用4个需求队列和1个价格队列计算6个核心特征：
        1. 价格（当前定价动作对应的价格）
        2. 是否周末（从date_features获取）
        3. 季节特征（从date_features获取）
        4. 价格变异系数（基于价格队列计算）
        5. 需求趋势（基于对应需求队列的7天数据线性回归斜率）
        6. 价格趋势（基于价格队列的7天数据线性回归斜率）
        
        Args:
            date_features (pd.DataFrame): 日期特征数据
            action (int): 定价动作索引
            demand_queues (dict): 4个需求队列的字典
            price_queue (deque): 价格队列
            user_type (str): 用户类型（'online'或'offline'）
            demand_type (str): 需求类型（'booked'或'actual'）
            
        Returns:
            torch.Tensor: 6维特征张量
        """
        # 1. 价格特征 - 使用当前定价动作对应的价格
        price_levels = [60, 80, 100, 110, 120, 130, 140, 150]  # 8个价格档位（基于真实数据分布优化）
        # 原配置: [60, 90, 120, 150, 180, 210] - 高价位缺乏数据支持
        # 新配置基于数据分析:
        # - 线上价格范围: 93.61-115.91元
        # - 线下价格范围: 99.96-152.80元  
        # - 90%的数据在150元以下
        price = float(price_levels[action])
        
        # 2. 是否周末特征
        if 'is_weekend' in date_features.columns:
            is_weekend = float(date_features['is_weekend'].iloc[0])
        else:
            is_weekend = 0.0  # 默认工作日
        
        # 3. 季节特征
        if 'season' in date_features.columns:
            season = float(date_features['season'].iloc[0])
        else:
            season = 1.0  # 默认平季
        
        # 4. 价格变异系数 - 基于价格队列计算
        if len(price_queue) >= 2:
            price_array = np.array(list(price_queue))
            price_cv = float(np.std(price_array) / np.mean(price_array) if np.mean(price_array) > 0 else 0.1)
        else:
            price_cv = 0.1  # 默认价格变异系数
        
        # 5. 需求趋势 - 基于对应需求队列的7天数据线性回归斜率
        queue_key = f"{user_type}_{demand_type}"
        if queue_key in demand_queues and len(demand_queues[queue_key]) >= 2:
            demand_array = np.array(list(demand_queues[queue_key]))
            x = np.arange(len(demand_array))
            slope, _ = np.polyfit(x, demand_array, 1)
            demand_trend = float(slope)
        else:
            demand_trend = 0.0  # 默认无趋势
        
        # 6. 价格趋势 - 基于价格队列的7天数据线性回归斜率
        if len(price_queue) >= 2:
            price_array = np.array(list(price_queue))
            x = np.arange(len(price_array))
            slope, _ = np.polyfit(x, price_array, 1)
            price_trend = float(slope)
        else:
            price_trend = 0.0  # 默认无趋势
        
        # 组合特征 [价格, 是否周末, 季节, 价格变异系数, 需求趋势, 价格趋势]
        features = [price, is_weekend, season, price_cv, demand_trend, price_trend]
        
        return torch.FloatTensor(features)
    
    def __call__(self, date_features: pd.DataFrame, action: int, record_env_changes: bool = False, 
                 user_type: str = None, demand_type: str = None, demand_queues: dict = None, 
                 price_queue: deque = None) -> Tuple[float, float]:
        """
        调用NGBoost预测器进行需求预测，支持环境变化记录和队列数据特征计算
        
        功能描述：
        执行完整的NGBoost预测流程：特征准备 → NGBoost预测 → 结果反标准化，返回预测需求的均值和方差。
        现在支持从队列数据计算特征，替代原有的特征准备方法。
        
        参数:
            date_features (pd.DataFrame): 日期特征数据，包含季节、工作日等信息
            action (int): 定价动作索引（0-5，对应6个价格档位）
            record_env_changes (bool): 是否记录环境变化，默认False
            user_type (str): 用户类型（'online'或'offline'）
            demand_type (str): 需求类型（'booked'或'actual'）
            demand_queues (dict): 4个需求队列的字典
            price_queue (deque): 价格队列
            
        返回值:
            Tuple[float, float]: (预测需求均值, 预测方差) - NGBoost的预测结果
            
        预测流程:
        1. 特征准备：使用队列数据准备NGBoost输入特征
        2. 维度检查：确保特征维度正确（6维：[季节, 是否周末, 平均价格, 价格变异系数, 需求趋势, 价格趋势]）
        3. 批次处理：为单样本预测添加批次维度
        4. NGBoost预测：调用训练器进行需求预测，支持环境变化记录
        5. 反标准化：如有标准化器，将结果转换回原始尺度
        
        Note:
        - 输入特征维度必须为6（季节、工作日类型、平均价格、价格变异系数、需求趋势、价格趋势）
        - 自动处理PyTorch张量的维度调整
        - 支持需求标准化器的反变换
        - 方差按标准化器尺度的平方进行缩放
        - 支持环境变化记录，记录预测结果作为环境状态变化
        """
        # 准备特征 - 使用队列数据计算6个核心特征
        if demand_queues is not None and price_queue is not None:
            features = self.prepare_features_from_queues(date_features, action, demand_queues, price_queue, user_type, demand_type)
        else:
            # 降级到原有方法（向后兼容）
            features = self.preprocessor.prepare_ngboost_features(date_features, action)
        
        # 确保特征维度正确（6维：[季节, 是否周末, 平均价格, 价格变异系数, 需求趋势, 价格趋势]）
        if hasattr(features, 'dim'):
            # PyTorch张量
            if features.dim() == 1 and features.shape[0] == 6:
                features = features.unsqueeze(0)  # 添加批次维度
        else:
            # NumPy数组
            if len(features.shape) == 1 and features.shape[0] == 6:
                features = features.reshape(1, -1)  # 添加批次维度
        
        # 预测，传递环境变化记录参数
        mean_pred, var_pred = self.ngboost_trainer.predict_single(
            features, record_env_changes=record_env_changes, 
            user_type=user_type, demand_type=demand_type
        )
        
        # 如果有标准化器，反标准化结果
        if self.demand_scaler is not None:
            mean_pred = self.demand_scaler.inverse_transform(np.array([[mean_pred]]))[0][0]
            var_pred = var_pred * (self.demand_scaler.scale_[0] ** 2)
        
        return mean_pred, var_pred

class HotelRLSystem:
    """
    酒店强化学习系统 - 集成NGBoost预测的完整RL解决方案
    
    功能描述：
    构建完整的酒店动态定价强化学习系统，集成自然梯度提升预测、Q-learning算法、
    在线学习等核心功能，提供从离线预训练到在线优化的端到端解决方案。
    
    系统架构：
    - NGBoost预测器：提供需求预测和不确定性估计
    - Q-learning智能体：学习最优定价策略
    - 酒店环境：模拟真实的酒店运营环境
    - 训练监控器：跟踪训练进度和性能指标
    
    主要特性：
    - 离线预训练：使用历史数据进行初始策略学习
    - 在线学习：根据新数据持续优化策略
    - 策略评估：评估学习到的定价策略性能
    - 模型持久化：支持模型保存和加载
    - 多阶段训练：支持渐进式学习策略
    
    Attributes:
        ngboost_trainer (Any): NGBoost训练器实例，提供需求预测功能
        preprocessor (Any): 数据预处理器，处理特征工程
        ngboost_predictor (NGBoostPredictorWrapper): NGBoost预测器包装器
        agent (QLearningAgent): Q-learning智能体
        env (HotelEnvironment): 酒店环境模拟器
        
    Note:
        - 系统集成NGBoost预测和强化学习
        - 支持增量学习和模型更新
        - 提供完整的训练监控和评估功能
        - 支持90天的完整定价周期模拟
    """
    
    def __init__(self, ngboost_trainers: Dict[str, Any], preprocessor: Any, 
                 demand_scalers: Dict[str, Any],
                 epsilon_start: float = 0.9, epsilon_end: float = 0.1, epsilon_decay_episodes: int = 400,
                 use_bayesian_rl: bool = False) -> None:
        """
        初始化酒店强化学习系统 - 支持双需求模型（预定需求和实际需求）
        
        Args:
            ngboost_trainers: 包含四个NGBoost训练器的字典
                - 'online_booked': 线上预定需求模型
                - 'online_actual': 线上实际需求模型
                - 'offline_booked': 线下预定需求模型
                - 'offline_actual': 线下实际需求模型
            preprocessor: 数据预处理器
            demand_scalers: 包含四个标准化器的字典
                - 'online_booked': 线上预定需求标准化器
                - 'online_actual': 线上实际需求标准化器
                - 'offline_booked': 线下预定需求标准化器
                - 'offline_actual': 线下实际需求标准化器
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay_episodes: 探索率衰减episode数
            use_bayesian_rl: 是否使用贝叶斯Q-learning
        """
        self.ngboost_trainers = ngboost_trainers
        self.preprocessor = preprocessor
        self.demand_scalers = demand_scalers
        
        # 创建四个NGBoost预测器包装器
        self.online_booked_predictor = NGBoostPredictorWrapper(ngboost_trainers['online_booked'], preprocessor, demand_scalers['online_booked'])
        self.online_actual_predictor = NGBoostPredictorWrapper(ngboost_trainers['online_actual'], preprocessor, demand_scalers['online_actual'])
        self.offline_booked_predictor = NGBoostPredictorWrapper(ngboost_trainers['offline_booked'], preprocessor, demand_scalers['offline_booked'])
        self.offline_actual_predictor = NGBoostPredictorWrapper(ngboost_trainers['offline_actual'], preprocessor, demand_scalers['offline_actual'])
        
        # 根据配置选择使用标准Q-learning还是贝叶斯Q-learning
        if use_bayesian_rl:
            print("使用贝叶斯Q-learning算法")
            self.agent = BayesianQLearning(
                n_states=30,  # 状态数：库存等级(5) × 季节(3) × 日期类型(2) = 30
                n_actions=8,  # 动作数：8个价格档位（与价格档位保持一致）
                discount_factor=BQL_CONFIG['discount_factor'],
                observation_noise_var=BQL_CONFIG['observation_noise_var'],
                prior_mean=BQL_CONFIG['prior_mean'],
                prior_var=BQL_CONFIG['prior_var']
            )
        else:
            print("使用标准Q-learning算法")
            self.agent = QLearningAgent(
                n_actions=36,  # 动作数：36个价格组合（线上6价格 × 线下6价格）
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay_steps=epsilon_decay_episodes
            )
        
        self.env = HotelEnvironment(initial_inventory=226)
    
    def offline_pretraining(self, features_df: pd.DataFrame, episodes: int = 1000) -> None:
        """
        离线预训练 - 使用历史数据训练初始定价策略（支持双需求模型）
        
        功能描述：
        使用历史酒店数据对Q-learning智能体进行离线预训练，通过大量episode学习基本的定价策略。
        集成训练监控器记录训练过程，支持随机日期选择和动态episode调整。
        现在支持四个NGBoost预测器：线上预定需求、线上实际需求、线下预定需求、线下实际需求。
        
        参数:
            features_df (pd.DataFrame): 历史特征数据，包含日期、季节、需求等特征
            episodes (int): 训练episode数量，默认1000个
            
        训练流程:
        1. 数据验证：检查特征数据是否满足最小天数要求
        2. Episode循环：运行指定数量的训练episode
        3. 随机采样：为每个episode随机选择起始日期
        4. 智能体训练：调用train_episode进行单episode训练（使用四个预测器）
        5. 指标记录：记录平均奖励、探索率、Q值统计等指标
        6. 进度显示：每10个episode显示训练进度
        7. 模型保存：训练完成后保存预训练模型
        8. 曲线绘制：生成训练过程可视化图表
        
        训练监控:
        - 平均奖励：每episode的平均收益
        - Episode长度：每个episode的步数
        - 探索率：当前的epsilon值
        - Q值统计：零值Q值比例、探索覆盖率
        - 每日详情：记录每日的奖励、库存、价格变化
        
        Note:
        - 自动调整episode数量以适应数据量
        - 支持最少90天的模拟周期
        - 集成详细的训练过程监控
        - 保存预训练模型供后续使用
        - 提供完整的训练可视化报告
        - 使用四个预测器：线上预定需求、线上实际需求、线下预定需求、线下实际需求
        """
        print("开始离线预训练...")
        
        # 获取训练监控器
        monitor = get_training_monitor()
        
        # 确保有足够的日期进行模拟
        min_days = 90
        if len(features_df) < min_days:
            print(f"警告：特征数据不足{min_days}天，将使用所有可用数据")
            episodes = min(episodes, len(features_df) // 2)  # 至少保证2天的数据
        
        # 随机选择一些日期进行模拟
        print(f"将运行 {episodes} 个episodes进行训练")
        
        for episode in range(episodes):
            # 随机选择一个起始日期
            max_start_idx = max(1, len(features_df) - min_days)
            start_idx = random.randint(0, max_start_idx)
            
            # 确保有足够的日期
            available_days = len(features_df) - start_idx
            episode_days = min(90, available_days)
            episode_features = features_df.iloc[start_idx:start_idx + episode_days].reset_index(drop=True)
            
            if len(episode_features) < 2:  # 至少需要2天数据
                continue
                
            # 训练一个episode，使用四个NGBoost预测器（双需求模型）
            total_reward, steps = self.agent.train_episode(
                env=self.env, 
                online_booked_predictor=self.online_booked_predictor,
                online_actual_predictor=self.online_actual_predictor,
                offline_booked_predictor=self.offline_booked_predictor,
                offline_actual_predictor=self.offline_actual_predictor,
                date_features=episode_features,
                episode=episode
            )
            
            # 记录训练指标
            avg_reward = total_reward / steps if steps > 0 else 0
            exploration_rate = self.agent.get_epsilon(episode)
            q_stats = self.agent.get_q_value_stats()
            
            # 每次episode都记录指标
            monitor.record_rl_episode(
                episode=episode, 
                avg_reward=avg_reward,
                episode_length=steps, 
                exploration_rate=exploration_rate,
                q_stats=q_stats
            )
            
            # 记录详细的每日训练信息
            if hasattr(self.agent, 'training_history') and self.agent.training_history:
                episode_history = self.agent.training_history[-1]  # 获取当前episode的历史
                if 'daily_rewards' in episode_history:
                    for day, daily_reward in enumerate(episode_history['daily_rewards']):
                        monitor.record_daily_training(
                            episode=episode,
                            day=day,
                            reward=daily_reward,
                            inventory=episode_history.get('daily_inventory', [0] * len(episode_history['daily_rewards']))[day] if 'daily_inventory' in episode_history else 0,
                            price=episode_history.get('daily_prices', [0] * len(episode_history['daily_rewards']))[day] if 'daily_prices' in episode_history else 0
                        )
            
            print(f"离线训练 Episode {episode + 1}/{episodes}, "
                  f"总奖励: {total_reward:.2f}, 步数: {steps}, "
                  f"平均奖励: {avg_reward:.2f}, 探索率: {exploration_rate:.3f}")
            
            if q_stats:
                print(f"  零值Q值占比: {q_stats['zero_q_percentage']:.1f}%, "
                      f"探索覆盖率: {q_stats['exploration_coverage']:.1f}%")
            
            # 每10个episode打印一次进度
            if (episode + 1) % 10 == 0:
                print(f"进度: {episode + 1}/{episodes} episodes 完成")
        
        print("离线预训练完成！")
        
        # 保存训练后的智能体
        self.agent.save_agent('../02_训练模型/q_agent_pretrained.pkl')
        
        # 训练完成后绘制训练曲线
        monitor.plot_training_curves()
    
    def online_learning(self, features_df: pd.DataFrame, days: int = 90, update_frequency: int = 7) -> Dict[str, float]:
        """
        在线学习 - 增量更新策略和BNN模型（支持双需求模型）
        
        功能描述：
        在真实环境中进行在线学习，根据实际交互数据持续优化Q-learning策略和BNN预测模型。
        现在支持四个NGBoost预测器：线上预定需求、线上实际需求、线下预定需求、线下实际需求。
        
        参数:
            features_df (pd.DataFrame): 在线特征数据，按天提供新的环境信息
            days (int): 在线学习天数，默认90天
            update_frequency (int): BNN模型更新频率（天），默认每7天更新一次
            
        返回值:
            Dict[str, float]: 在线学习统计信息，包含：
                - 总收益、入住率、平均价格等环境统计
                - incremental_data_count: 收集的增量数据条数
                - avg_incremental_reward: 增量数据的平均奖励
                
        在线学习流程:
        1. 环境初始化：重置酒店环境到初始状态
        2. 每日循环：对每一天执行定价决策和学习
        3. 状态获取：获取当前库存、季节、日期类型等状态
        4. 动作选择：基于当前策略选择定价动作（低探索率）
        5. 环境交互：执行定价决策，获取实际反馈（使用四个预测器）
        6. 数据收集：收集有意义的交互数据用于增量学习
        7. Q表更新：使用实际经验更新Q值函数
        8. BNN更新：定期使用新数据增量更新BNN模型（支持四个模型）
        9. 进度显示：定期显示学习进度和关键指标
        10. 模型保存：学习完成后保存最终模型
        
        Note:
        - 在线学习阶段主要利用已有知识，探索率较低
        - 支持增量数据收集和BNN模型的持续优化
        - 提供详细的学习进度监控和统计信息
        - 自动处理episode结束和环境重置
        - 支持长达90天的完整在线学习周期
        - 使用四个预测器：线上预定需求、线上实际需求、线下预定需求、线下实际需求
        """
        print("开始在线学习...")
        
        # 增量学习数据收集
        incremental_data = []
        episode_counter = 0  # 用于epsilon衰减的计数器
        
        for day in range(days):
            # 获取当天的特征
            day_features = features_df.iloc[day:day + 1].reset_index(drop=True)
            
            # 获取当前状态
            state_info = self.env._get_state()
            
            # 离散化状态
            season = int(day_features['season'].iloc[0])
            weekday = int(day_features['is_weekend'].iloc[0])
            state = self.agent.discretize_state(state_info, season, weekday)
            
            # 在线学习时主要利用已有知识，少量探索
            action = self.agent.select_action(state, episode_counter)
            
            # 执行动作并获取反馈，使用四个NGBoost预测器（双需求模型）
            next_state_info, reward, done, info = self.env.step(
                action, 
                self.online_booked_predictor, self.online_actual_predictor,
                self.offline_booked_predictor, self.offline_actual_predictor, 
                day_features
            )
            
            # 原条件: if info['actual_bookings'] > 0
            # 新条件: 只要有预测需求或实际预订就收集
            if info['predicted_demand'] > 0 or info['actual_bookings'] > 0:
                # 准备特征
                features = self.preprocessor.prepare_ngboost_features(day_features, action)
                
                incremental_data.append({
                    'features': features.unsqueeze(0).cpu().numpy().flatten(),
                    'target': info['actual_bookings'],
                    'predicted_demand': info['predicted_demand'],
                    'predicted_variance': info['predicted_variance'],
                    'day': day,
                    'action': action,
                    'price': info.get('price', [60, 80, 100, 110, 120, 130, 140, 150][action]),
                    'reward': reward
                })
            
            # 更新Q表
            next_state = self.agent.discretize_state(next_state_info, season, weekday)
            self.agent.update_q_table(state, action, reward, next_state, done)
            
            
            if (day + 1) % update_frequency == 0 and len(incremental_data) >= 10:
                print(f"第{day + 1}天：开始增量更新NGBoost...")
                
                # 使用最近的数据，确保有足够样本
                recent_data = incremental_data[-max(50, len(incremental_data)//2):]
                X_new = np.array([d['features'] for d in recent_data])
                y_new = np.array([d['target'] for d in recent_data])
                
                # 增量更新四个NGBoost模型
                print("增量更新线上预定需求模型...")
                self.ngboost_trainers['online_booked'].incremental_update(X_new, y_new, epochs=5)
                
                print("增量更新线上实际需求模型...")
                self.ngboost_trainers['online_actual'].incremental_update(X_new, y_new, epochs=5)
                
                print("增量更新线下预定需求模型...")
                self.ngboost_trainers['offline_booked'].incremental_update(X_new, y_new, epochs=5)
                
                print("增量更新线下实际需求模型...")
                self.ngboost_trainers['offline_actual'].incremental_update(X_new, y_new, epochs=5)
                
                print(f"增量更新完成，使用{len(X_new)}条新数据")
            
            
            if (day + 1) % 10 == 0:
                env_stats = self.env.get_statistics()
                recent_rewards = [d['reward'] for d in incremental_data[-10:]] if incremental_data else [0]
                
                print(f"第{day + 1}天：库存: {state_info['inventory_raw']}, "
                      f"价格: {[60, 80, 100, 110, 120, 130, 140, 150][action]}元, "
                      f"预订: {info['actual_bookings']}, "
                      f"预测需求: {info['predicted_demand']:.1f}, "
                      f"当日奖励: {reward:.2f}, "
                      f"总收益: {env_stats['total_revenue']:.2f}, "
                      f"收集数据: {len(incremental_data)}条")
            
            
            if done or self.env.day >= 90:  # 达到episode结束条件
                print(f"第{day + 1}天：Episode结束，重置环境...")
                self.env.reset()  # 重置环境开始新的episode
                episode_counter += 1  # 增加episode计数器
            else:
                episode_counter += 0.01  # 轻微增加计数器，保持epsilon衰减
        
        print(f"在线学习完成！共收集{len(incremental_data)}条增量数据")
        
        # 保存最终模型
        self.agent.save_agent('../02_训练模型/q_agent_final.pkl')
        
        # 保存四个NGBoost模型
        self.ngboost_trainers['online_booked'].save_model('../02_训练模型/ngboost_model_online_booked_final.pkl')
        self.ngboost_trainers['online_actual'].save_model('../02_训练模型/ngboost_model_online_actual_final.pkl')
        self.ngboost_trainers['offline_booked'].save_model('../02_训练模型/ngboost_model_offline_booked_final.pkl')
        self.ngboost_trainers['offline_actual'].save_model('../02_训练模型/ngboost_model_offline_actual_final.pkl')
        
        # 返回详细的统计信息
        stats = self.env.get_statistics()
        stats['incremental_data_count'] = len(incremental_data)
        stats['avg_incremental_reward'] = np.mean([d['reward'] for d in incremental_data]) if incremental_data else 0
        
        return stats
    
    def bnn_predictor(self, day_features: pd.DataFrame, action: int) -> Tuple[float, float]:
        """
        BNN预测器接口 - 用于模拟函数获取预测需求和方差
        
        功能描述：
        为run_simulation函数提供统一的预测接口，综合四个NGBoost预测器的结果
        返回总预测需求和预测方差。
        
        Args:
            day_features (pd.DataFrame): 当天的特征数据
            action (int): 价格动作索引 (0-5)
            
        Returns:
            Tuple[float, float]: (总预测需求, 总预测方差)
        """
        # 使用四个预测器获取预测结果
        online_booked_pred, online_booked_var = self.online_booked_predictor(day_features, action)
        online_actual_pred, online_actual_var = self.online_actual_predictor(day_features, action)
        offline_booked_pred, offline_booked_var = self.offline_booked_predictor(day_features, action)
        offline_actual_pred, offline_actual_var = self.offline_actual_predictor(day_features, action)
        
        # 计算总预测需求（预定需求 + 实际需求）
        total_predicted_demand = online_booked_pred + offline_booked_pred + online_actual_pred + offline_actual_pred
        
        # 计算总预测方差（方差相加）
        total_predicted_variance = online_booked_var + offline_booked_var + online_actual_var + offline_actual_var
        
        return total_predicted_demand, total_predicted_variance
    
    def evaluate_policy(self, features_df: pd.DataFrame, n_episodes: int = 10, verbose: bool = False) -> Tuple[pd.Series, List[Dict[str, float]]]:
        """
        评估策略 - 测试学习到的定价策略性能
        
        功能描述：
        对训练完成的Q-learning策略进行系统性评估，通过运行多个测试episode来评估策略的平均性能表现。
        使用贪婪策略（epsilon=0）进行决策，提供详细的统计指标。
        
        参数:
            features_df (pd.DataFrame): 测试特征数据，用于模拟评估环境
            n_episodes (int): 评估episode数量，默认10个episode
            verbose (bool): 是否显示详细评估信息，默认False
            
        返回值:
            Tuple[pd.Series, List[Dict[str, float]]]: 
                - avg_stats (pd.Series): 平均统计指标，包含总收益、入住率、平均价格等
                - all_stats (List[Dict]): 每个episode的详细统计信息列表
                
        评估指标:
        - total_revenue: 总收益
        - total_bookings: 总预订量  
        - average_occupancy_rate: 平均入住率
        - average_daily_revenue: 平均每日收益
        - average_price: 平均价格
        - total_demand: 总需求量
        - demand_satisfaction_rate: 需求满足率
        
        评估流程:
        1. Episode循环：运行指定数量的测试episode
        2. 环境重置：每个episode开始时重置酒店环境
        3. 状态获取：获取当前库存、季节、日期类型等状态信息
        4. 贪婪动作选择：基于Q表选择最优动作（epsilon=0）
        5. 环境交互：执行定价决策，获取奖励和下一状态
        6. 统计收集：记录每个episode的详细统计信息
        7. 结果汇总：计算所有episode的平均统计指标
        8. 报告输出：显示评估结果摘要（如verbose=True）
        
        Note:
        - 使用贪婪策略进行评估，无探索行为
        - 支持多episode评估以获得稳定的性能估计
        - 提供详细的统计指标用于策略比较
        - 可评估策略在不同环境条件下的鲁棒性
        - 评估结果可用于策略选择和超参数调优
        """
        if verbose:
            print("开始策略评估...")
        
        all_stats = []
        
        for episode in range(n_episodes):
            # 重置环境
            self.env.reset()
            
            # 运行一个episode
            for day in range(min(90, len(features_df))):
                day_features = features_df.iloc[day:day + 1].reset_index(drop=True)
                
                # 获取当前状态
                state_info = self.env._get_state()
                season = int(day_features['season'].iloc[0])
                weekday = int(day_features['is_weekend'].iloc[0])
                state = self.agent.discretize_state(state_info, season, weekday)
                
                # 选择动作（使用最优策略，epsilon=0）
                q_values = self.agent.q_table[state]
                action = np.argmax(q_values)
                
                # 执行动作，使用四个NGBoost预测器（双需求模型）
                _, reward, done, info = self.env.step(
                    action, 
                    self.online_booked_predictor, self.online_actual_predictor,
                    self.offline_booked_predictor, self.offline_actual_predictor, 
                    day_features
                )
                
                if done:
                    break
            
            # 记录统计信息
            stats = self.env.get_statistics()
            stats['episode'] = episode
            all_stats.append(stats)
        
        # 计算平均统计
        df_stats = pd.DataFrame(all_stats)
        avg_stats = df_stats.mean()
        
        if verbose:
            print("策略评估完成！")
            print(f"平均总收益: {avg_stats['total_revenue']:.2f}")
            print(f"平均入住率: {avg_stats['average_occupancy_rate']:.2%}")
            print(f"平均每日收益: {avg_stats['average_daily_revenue']:.2f}")
            print(f"平均价格: {avg_stats['average_price']:.2f}")
        
        return avg_stats, all_stats
    
    def is_standard_ql_agent(self) -> bool:
        """
        检查是否使用标准Q-learning算法
        
        Returns:
            bool: 如果是标准Q-learning算法返回True，否则返回False
        """
        return isinstance(self.agent, QLearningAgent)
    
    def is_bayesian_ql_agent(self) -> bool:
        """
        检查是否使用贝叶斯Q-learning算法
        
        Returns:
            bool: 如果是贝叶斯Q-learning算法返回True，否则返回False
        """
        return isinstance(self.agent, BayesianQLearning)

class BayesianQLearning:
    """
    贝叶斯Q-Learning (BQL) 实现
    
    在BQL中，我们维护Q(s,a)的概率分布信念，假设Q值服从高斯分布：
    Q(s,a) ~ N(μ_{s,a}, σ²_{s,a})
    
    更新过程使用贝叶斯推断，结合先验信念和观测证据来更新后验分布。
    """
    
    def __init__(self, n_states: int = 30, n_actions: int = 8, discount_factor: float = 0.9,
                 observation_noise_var: float = 1.0, prior_mean: float = 0.0, prior_var: float = 10.0,
                 q_value_max: float = 1000.0, reward_scale: float = 0.1):
        """
        初始化贝叶斯Q-Learning
        
        Args:
            n_states: 状态数量
            n_actions: 动作数量  
            discount_factor: 折扣因子γ
            observation_noise_var: 观测噪声方差σ²_r
            prior_mean: 先验均值
            prior_var: 先验方差
            q_value_max: Q值上限，防止极端值
            reward_scale: 奖励缩放因子，用于归一化奖励范围
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.observation_noise_var = observation_noise_var
        self.q_value_max = q_value_max
        self.reward_scale = reward_scale
        
        # Q值分布参数：每个状态-动作对的均值和方差
        # 使用字典存储，支持动态状态空间
        self.q_means = defaultdict(lambda: np.full(n_actions, prior_mean))  # μ_{s,a}
        self.q_vars = defaultdict(lambda: np.full(n_actions, prior_var))    # σ²_{s,a}
        
        # 状态访问计数
        self.state_visit_count = defaultdict(int)
        self.state_action_visit_count = defaultdict(int)
        
        # 训练历史
        self.training_history = []
        
        # 异常值检测参数
        self.q_value_history = defaultdict(list)  # 记录Q值历史用于异常检测
        self.max_q_value_change = 5.0  # 最大允许的Q值变化倍数
        self.min_variance = 0.1  # 最小方差，防止过度自信
    
    def get_state_distribution(self, state: Union[List, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray]:
        """获取状态的Q值分布（均值和方差）"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        return self.q_means[state_key].copy(), self.q_vars[state_key].copy()
    
    def select_action(self, state: Union[List, np.ndarray, int], episode: int, 
                     exploration_strategy: str = "ucb") -> int:
        """
        选择动作（基于贝叶斯探索策略）
        
        Args:
            state: 当前状态
            episode: 当前episode编号
            exploration_strategy: 探索策略 ("ucb", "thompson", "epsilon_greedy")
        
        Returns:
            选择的动作索引
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        means = self.q_means[state_key]
        vars = self.q_vars[state_key]
        
        if exploration_strategy == "ucb":
            # 基于不确定性的上置信界
            n_visits = np.array([self.state_action_visit_count.get((state_key, a), 0) for a in range(self.n_actions)])
            
            # 使用配置中的UCB参数
            ucb_c = BQL_CONFIG.get('ucb_c', 2.5)
            ucb_bonus_scale = BQL_CONFIG.get('ucb_bonus_scale', 2.0)
            
            # 改进的UCB计算，避免初始阶段探索不足
            total_visits = self.state_visit_count.get(state_key, 0)
            if total_visits == 0:
                # 初始阶段：综合考虑先验均值和不确定性，引入随机扰动
                exploration_bonus = np.sqrt(vars) / np.max(np.sqrt(vars))  # 标准化不确定性
                random_noise = np.random.normal(0, 0.1, self.n_actions)  # 小幅度随机噪声
                # 平衡先验信念和探索：均值 + 探索奖励 + 随机扰动
                ucb_values = means + 0.5 * exploration_bonus + random_noise
            else:
                # 正常UCB计算
                log_total = np.log(total_visits + 1)
                # 避免除零，给未访问的动作最大探索奖励
                ucb_bonus = ucb_c * np.sqrt(log_total / (n_visits + 1e-6))
                ucb_values = means + ucb_bonus_scale * ucb_bonus * np.sqrt(vars)
            
            return np.argmax(ucb_values)
            
        elif exploration_strategy == "thompson":
            # Thompson采样：从高斯分布中采样
            sampled_values = np.random.normal(means, np.sqrt(vars))
            return np.argmax(sampled_values)
            
        else:  # epsilon_greedy
            # ε-贪心策略，使用均值
            epsilon = max(0.1, 1.0 / (episode + 1))
            if np.random.random() < epsilon:
                return np.random.randint(self.n_actions)
            else:
                return np.argmax(means)
    
    def _normalize_reward(self, reward: float) -> float:
        """归一化奖励值，防止极端值"""
        # 使用tanh函数将奖励压缩到合理范围
        normalized = np.tanh(reward * self.reward_scale)
        # 然后缩放到与先验均值匹配的范围
        return normalized * 100.0  # 假设先验均值在50左右
    
    def _detect_anomalous_q_value(self, state_key: Union[tuple, int], action: int, 
                                  new_mean: float, new_var: float) -> bool:
        """检测异常Q值更新"""
        # 检查Q值是否超出合理范围
        if abs(new_mean) > self.q_value_max:
            return True
        
        # 检查方差是否过小（过度自信）
        if new_var < self.min_variance:
            return True
        
        # 检查Q值变化是否过于剧烈
        history = self.q_value_history.get((state_key, action), [])
        if len(history) >= 3:  # 需要至少3个历史值
            recent_mean = np.mean(history[-3:])
            if recent_mean != 0 and abs(new_mean - recent_mean) / abs(recent_mean) > self.max_q_value_change:
                return True
        
        return False
    
    def update_bayesian_q_table(self, state: Union[List, np.ndarray, int], action: int, 
                               reward: float, next_state: Union[List, np.ndarray, int], 
                               done: bool) -> Tuple[float, float]:
        """
        使用贝叶斯推断更新Q值分布 - 改进版
        
        根据贝叶斯定理，更新后验分布参数：
        σ²_new = (1/σ²_old + 1/σ²_r)^(-1)
        μ_new = σ²_new * (μ_old/σ²_old + y/σ²_r)
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            
        Returns:
            Tuple[float, float]: 更新后的(均值, 方差)
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
        
        # 更新访问计数
        self.state_visit_count[state_key] += 1
        self.state_action_visit_count[(state_key, action)] += 1
        
        # 获取当前分布参数
        current_mean = self.q_means[state_key][action]
        current_var = self.q_vars[state_key][action]
        
        # 归一化奖励，防止极端值
        normalized_reward = self._normalize_reward(reward)
        
        # 计算TD目标 y = r + γ * max_a' μ_{s',a'}
        if done:
            td_target = normalized_reward
        else:
            next_means = self.q_means[next_state_key]
            # 使用鲁棒的max估计，避免异常值影响
            max_next_mean = np.percentile(next_means, 90)  # 使用90分位数而非最大值
            td_target = normalized_reward + self.discount_factor * max_next_mean
        
        # 限制TD目标的范围
        td_target = np.clip(td_target, -self.q_value_max, self.q_value_max)
        
        # 改进的贝叶斯更新：考虑TD目标的不确定性
        if not done:
            # 计算下一状态Q值的最大值的不确定性
            next_vars = self.q_vars[next_state_key]
            # 同样使用90分位数对应的不确定性
            top_10_percent_count = max(1, int(np.ceil(len(next_means) * 0.1)))
            max_idx = np.argsort(next_means)[-top_10_percent_count:]
            max_next_var = np.mean([next_vars[i] for i in max_idx]) if len(max_idx) > 0 else np.mean(next_vars)
            # TD目标的总不确定性 = 奖励噪声 + 折扣后的下一状态不确定性
            td_target_var = self.observation_noise_var + (self.discount_factor ** 2) * max_next_var
        else:
            td_target_var = self.observation_noise_var
        
        # 确保方差在合理范围内
        td_target_var = max(td_target_var, self.min_variance)
        
        # 贝叶斯更新，使用TD目标的总不确定性
        new_var = 1.0 / (1.0 / current_var + 1.0 / td_target_var)
        new_mean = new_var * (current_mean / current_var + td_target / td_target_var)
        
        # 检测异常Q值更新
        if self._detect_anomalous_q_value(state_key, action, new_mean, new_var):
            # 如果检测到异常，使用保守的更新策略
            learning_rate = 0.1  # 使用较小的学习率
            new_mean = current_mean + learning_rate * (td_target - current_mean)
            new_var = max(current_var * 0.99, self.min_variance)  # 稍微减小方差
        
        # 确保方差不小于最小值
        new_var = max(new_var, self.min_variance)
        
        # 限制Q值范围
        new_mean = np.clip(new_mean, -self.q_value_max, self.q_value_max)
        
        # 记录Q值历史
        self.q_value_history[(state_key, action)].append(new_mean)
        # 只保留最近的历史
        if len(self.q_value_history[(state_key, action)]) > 10:
            self.q_value_history[(state_key, action)].pop(0)
        
        # 更新分布参数
        self.q_means[state_key][action] = new_mean
        self.q_vars[state_key][action] = new_var
        
        return new_mean, new_var
    
    def get_uncertainty(self, state: Union[List, np.ndarray, int], action: int = None) -> Union[float, np.ndarray]:
        """获取状态-动作对的不确定性（标准差）- 改进版"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        # 确保状态存在，如果不存在则返回先验不确定性
        if state_key not in self.q_vars:
            prior_var = BQL_CONFIG.get('prior_var', 15.0)
            if action is not None:
                return np.sqrt(prior_var)
            else:
                return np.full(self.n_actions, np.sqrt(prior_var))
        
        if action is not None:
            return np.sqrt(max(self.q_vars[state_key][action], self.min_variance))
        else:
            return np.sqrt(np.maximum(self.q_vars[state_key], self.min_variance))
    
    def get_epsilon(self, episode: int) -> float:
        """获取当前探索率 - 改进版，支持动态探索"""
        # 贝叶斯Q-learning使用UCB或Thompson采样，不直接使用epsilon
        # 但为了兼容性，返回一个基于不确定性的动态探索率
        
        # 计算平均不确定性
        if self.q_means:
            total_uncertainty = 0.0
            count = 0
            for state_key in self.q_means.keys():
                uncertainties = self.get_uncertainty(state_key)
                total_uncertainty += np.mean(uncertainties)
                count += 1
            
            avg_uncertainty = total_uncertainty / count if count > 0 else 1.0
            
            # 基于不确定性调整探索率：不确定性高时探索更多
            base_epsilon = max(0.1, 1.0 / (episode + 1))
            uncertainty_factor = min(2.0, 1.0 + avg_uncertainty / 10.0)  # 不确定性因子
            
            return min(0.5, base_epsilon * uncertainty_factor)
        else:
            return max(0.1, 1.0 / (episode + 1))
    
    def get_q_value_stats(self) -> Dict[str, float]:
        """获取Q值统计信息 - 改进版，包含异常检测"""
        if not self.q_means:
            return {
                'zero_q_percentage': 100.0, 
                'exploration_coverage': 0.0, 
                'mean_q_value': 0.0, 
                'num_state_visits': 0,
                'explored_state_actions': 0,
                'total_state_actions': 0,
                'mean_uncertainty': 0.0,
                'std_uncertainty': 0.0,
                'min_uncertainty': 0.0,
                'max_uncertainty': 0.0,
                'anomalous_q_percentage': 0.0,
                'high_uncertainty_percentage': 0.0,
                'unvisited_percentage': 100.0
            }
        
        # 计算零值Q值比例和异常Q值比例
        all_means = []
        all_uncertainties = []
        anomalous_count = 0
        high_uncertainty_count = 0
        
        for state_key in self.q_means.keys():
            state_means = self.q_means[state_key]
            state_vars = self.q_vars[state_key]
            state_uncertainties = np.sqrt(state_vars)
            
            all_means.extend(state_means)
            all_uncertainties.extend(state_uncertainties)
            
            # 检测异常Q值
            for action in range(self.n_actions):
                if self._detect_anomalous_q_value(state_key, action, state_means[action], state_vars[action]):
                    anomalous_count += 1
                
                # 检测高不确定性（标准差大于先验标准差）
                if state_uncertainties[action] > np.sqrt(BQL_CONFIG.get('prior_var', 15.0)):
                    high_uncertainty_count += 1
        
        if not all_means:
            return {
                'zero_q_percentage': 100.0, 
                'exploration_coverage': 0.0, 
                'mean_q_value': 0.0, 
                'num_state_visits': 0,
                'explored_state_actions': 0,
                'total_state_actions': 0,
                'mean_uncertainty': 0.0,
                'std_uncertainty': 0.0,
                'min_uncertainty': 0.0,
                'max_uncertainty': 0.0,
                'anomalous_q_percentage': 0.0,
                'high_uncertainty_percentage': 0.0,
                'unvisited_percentage': 100.0
            }
        
        zero_q_count = sum(bool(abs(mean) < 0.01) for mean in all_means)
        zero_q_percentage = (zero_q_count / len(all_means)) * 100
        
        anomalous_q_percentage = (anomalous_count / len(all_means)) * 100
        high_uncertainty_percentage = (high_uncertainty_count / len(all_means)) * 100
        
        # 计算平均Q值（排除异常值）
        normal_means = [mean for mean in all_means if abs(mean) <= self.q_value_max]
        mean_q_value = np.mean(normal_means) if normal_means else 0.0
        
        # 计算不确定性统计
        mean_uncertainty = np.mean(all_uncertainties)
        std_uncertainty = np.std(all_uncertainties)
        min_uncertainty = np.min(all_uncertainties)
        max_uncertainty = np.max(all_uncertainties)
        
        # 计算探索覆盖率（已访问的状态-动作对比例）
        total_state_action_pairs = len(self.q_means) * self.n_actions
        visited_state_action_pairs = len(self.state_action_visit_count)
        exploration_coverage = (visited_state_action_pairs / total_state_action_pairs) * 100 if total_state_action_pairs > 0 else 0.0
        
        # 计算未访问的状态-动作对比例
        unvisited_percentage = 100.0 - exploration_coverage
        
        # 计算总状态访问次数
        num_state_visits = sum(self.state_visit_count.values())
        
        return {
            'zero_q_percentage': zero_q_percentage,
            'exploration_coverage': exploration_coverage,
            'mean_q_value': mean_q_value,
            'num_state_visits': num_state_visits,
            'explored_state_actions': visited_state_action_pairs,
            'total_state_actions': total_state_action_pairs,
            # 贝叶斯Q-learning特有的不确定性统计
            'mean_uncertainty': mean_uncertainty,
            'std_uncertainty': std_uncertainty,
            'min_uncertainty': min_uncertainty,
            'max_uncertainty': max_uncertainty,
            # 新增异常检测统计
            'anomalous_q_percentage': anomalous_q_percentage,
            'high_uncertainty_percentage': high_uncertainty_percentage,
            'unvisited_percentage': unvisited_percentage
        }
    
    def discretize_state(self, state_info: Dict[str, Any], season: int, weekday: int) -> int:
        """离散化状态"""
        inventory_level = state_info['inventory_level']
        # 确保库存水平在合理范围内
        inventory_level = max(0, min(inventory_level, 4))  # 库存等级为0-4
        
        # 修正状态映射：库存(5) × 季节(3) × 星期(2) = 30种状态
        # 季节只有3个值：0=淡季，1=平季，2=旺季
        state_index = inventory_level * 6 + season * 2 + weekday
        
        # 确保状态索引在有效范围内
        return min(state_index, self.n_states - 1)
    
    def save_agent(self, filepath: str):
        """保存智能体状态"""
        agent_state = {
            'q_means': dict(self.q_means),
            'q_vars': dict(self.q_vars),
            'state_visit_count': dict(self.state_visit_count),
            'state_action_visit_count': dict(self.state_action_visit_count),
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'discount_factor': self.discount_factor,
            'observation_noise_var': self.observation_noise_var
        }
        with open(filepath, 'wb') as f:
            pickle.dump(agent_state, f)
    
    def load_agent(self, filepath: str):
        """加载智能体状态"""
        with open(filepath, 'rb') as f:
            agent_state = pickle.load(f)
        
        # 获取配置中的先验参数，确保加载时使用正确的默认值
        prior_mean = BQL_CONFIG.get('prior_mean', 50.0)
        prior_var = BQL_CONFIG.get('prior_var', 15.0)
        
        self.q_means = defaultdict(lambda: np.full(self.n_actions, prior_mean), agent_state['q_means'])
        self.q_vars = defaultdict(lambda: np.full(self.n_actions, prior_var), agent_state['q_vars'])
        self.state_visit_count = defaultdict(int, agent_state['state_visit_count'])
        self.state_action_visit_count = defaultdict(int, agent_state['state_action_visit_count'])
        self.n_states = agent_state['n_states']
        self.n_actions = agent_state['n_actions']
        self.discount_factor = agent_state['discount_factor']
        self.observation_noise_var = agent_state['observation_noise_var']
    
    def train_episode(self, env: HotelEnvironment, online_booked_predictor: Optional[Any] = None, 
                     online_actual_predictor: Optional[Any] = None, offline_booked_predictor: Optional[Any] = None, 
                     offline_actual_predictor: Optional[Any] = None, date_features: Optional[pd.DataFrame] = None, 
                     episode: int = 0, exploration_strategy: str = "ucb") -> Tuple[float, int]:
        """
        使用贝叶斯Q-Learning训练一个episode
        
        Args:
            env: 酒店环境实例
            online_booked_predictor: 线上用户预定需求NGBoost预测器
            online_actual_predictor: 线上用户实际需求NGBoost预测器
            offline_booked_predictor: 线下用户预定需求NGBoost预测器
            offline_actual_predictor: 线下用户实际需求NGBoost预测器
            date_features: 日期特征数据
            episode: 当前episode编号
            exploration_strategy: 探索策略
            
        Returns:
            Tuple[float, int]: (总奖励, 步数)
        """
        state_info = env.reset()
        total_reward = 0.0  # 明确指定为float类型
        steps: int = 0
        day_index = 0  # 添加日期索引，避免使用steps作为日期索引
        
        # 初始化每日记录
        daily_rewards: List[float] = []
        daily_uncertainties: List[float] = []  # 记录不确定性
        
        # 获取季节和星期信息
        if date_features is not None and len(date_features) > 0:
            season = int(date_features['season'].iloc[0])
            weekday = int(date_features['is_weekend'].iloc[0])
        else:
            season = 0
            weekday = 0
        
        state = self.discretize_state(state_info, season, weekday)
        
        done = False
        while not done:
            # 选择动作（使用贝叶斯探索策略）
            action = self.select_action(state, episode, exploration_strategy)
            
            # 获取价格信息
            prices = [60, 80, 100, 110, 120, 130, 140, 150]
            price = prices[action]
            
            # 执行动作，使用四个NGBoost预测器
            next_state_info, reward, done, info = env.step(action, online_booked_predictor, online_actual_predictor, offline_booked_predictor, offline_actual_predictor, date_features)
            
            # 获取当前状态的不确定性
            current_uncertainty = self.get_uncertainty(state, action)
            
            # 获取下一状态的season和weekday信息
            if date_features is not None and day_index + 1 < len(date_features):
                next_season = int(date_features['season'].iloc[day_index + 1])
                next_weekday = int(date_features['is_weekend'].iloc[day_index + 1])
                day_index += 1
            else:
                next_season = season
                next_weekday = weekday
            
            next_state = self.discretize_state(next_state_info, next_season, next_weekday)
            
            # 使用贝叶斯更新Q表
            new_mean, new_var = self.update_bayesian_q_table(state, action, reward, next_state, done)
            
            # 记录信息
            daily_rewards.append(float(reward))
            daily_uncertainties.append(float(current_uncertainty))
            
            # 打印训练信息（包含不确定性）
            if steps % 10 == 0:
                print(f"Episode {episode}, Step {steps}: 动作={action}({price}元), "
                      f"奖励={reward:.2f}, Q均值={new_mean:.2f}, Q方差={new_var:.2f}, "
                      f"不确定性={current_uncertainty:.2f}")
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if steps >= 200:  # 防止无限循环
                break
        
        # 记录训练历史
        episode_history = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'avg_reward': float(total_reward / steps) if steps > 0 else 0.0,
            'avg_uncertainty': float(np.mean(daily_uncertainties)) if daily_uncertainties else 0.0,
            'exploration_strategy': exploration_strategy
        }
        self.training_history.append(episode_history)
        
        return total_reward, steps


if __name__ == "__main__":
    print("正在测试强化学习系统...")
    
    # 设置随机种子 - 可取消注释以启用固定随机因子
    # np.random.seed(42)
    
    # 创建模拟的NGBoost训练器
    from ngboost_model import NGBoostTrainer
    
    # 创建模拟数据
    n_samples = 1000
    input_dim = 33
    
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.abs(2 * X[:, 0] + np.random.normal(0, 0.1, n_samples)) + 1
    
    # 训练NGBoost
    ngboost_trainer = NGBoostTrainer(input_dim=input_dim)
    ngboost_trainer.train(X, y, epochs=20, save_path='../02_训练模型/ngboost_model_rl_test.pkl')
    
    # 创建预处理器（模拟）
    class MockPreprocessor:
        def prepare_ngboost_features(self, date_features, action):
            return torch.FloatTensor(np.random.randn(1, input_dim).astype(np.float32))
    
    preprocessor = MockPreprocessor()
    
    # 加载需求标准化器
    import joblib
    demand_scaler = joblib.load('../02_训练模型/demand_scaler.pkl')

    # 创建RL系统
    rl_system = HotelRLSystem(ngboost_trainer, preprocessor, demand_scaler)
    
    # 创建模拟的特征数据
    dates = pd.date_range('2017-01-01', periods=100, freq='D')
    mock_features = pd.DataFrame({
        'date': dates,
        'season': np.random.randint(0, 3, 100),
        'is_weekend': np.random.randint(0, 2, 100),
        'month': dates.month,
        'weekday': dates.weekday,
        'demand_lag_1': np.random.randint(20, 50, 100),
        'noise': np.random.normal(0, 0.1, 100)
    })
    
    # 测试离线预训练
    print("\n测试离线预训练...")
    rl_system.offline_pretraining(mock_features, episodes=10)
    
    # 测试在线学习
    print("\n测试在线学习...")
    stats = rl_system.online_learning(mock_features, days=30)
    
    # 测试策略评估
    print("\n测试策略评估...")
    avg_stats, all_stats = rl_system.evaluate_policy(mock_features, n_episodes=3)
    
    print("\n强化学习系统测试完成！")
