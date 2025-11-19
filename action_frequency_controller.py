#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q表最优动作频数分析控制器
用于多次运行main.py并分析最优动作的频数分布
"""

import os
import sys
import subprocess
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
from datetime import datetime
import glob
import argparse
from collections import defaultdict, Counter
import warnings
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
import uuid
import pickle
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_PATH = os.path.join(CURRENT_DIR, '01_核心代码', 'main.py')
RESULTS_DIR = os.path.join(CURRENT_DIR, '05_分析报告')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'action_frequency_analysis')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用字典存储Q表信息，避免文件读取冲突
q_table_dict = {}

def check_ngboost_models():
    """
    检查NGBoost模型文件是否存在
    返回布尔值表示所有模型文件是否存在
    """
    model_paths = [
        '../02_训练模型/ngboost_model_online_booked.pkl',
        '../02_训练模型/ngboost_model_online_actual.pkl',
        '../02_训练模型/ngboost_model_offline_booked.pkl',
        '../02_训练模型/ngboost_model_offline_actual.pkl'
    ]
    
    # 检查所有模型文件是否存在
    models_exist = all(os.path.exists(os.path.join(CURRENT_DIR, '01_核心代码', path)) for path in model_paths)
    return models_exist

def train_ngboost_models():
    """
    训练NGBoost模型
    返回是否训练成功
    """
    try:
        cmd = [
            sys.executable, 
            MAIN_SCRIPT_PATH, 
            '--skip-hyperparameter-search',
            '--train-ngboost-only'
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        env['PYTHONUNBUFFERED'] = '1'
        
        print("开始训练NGBoost模型...")
        result = subprocess.run(
            cmd, 
            cwd=os.path.dirname(MAIN_SCRIPT_PATH),
            capture_output=True, 
            text=True,
            timeout=6400,
            env=env
        )
        
        if result.returncode != 0:
            print(f"NGBoost模型训练失败: {result.stderr}")
            return False
            
        print("NGBoost模型训练完成")
        return True
    except Exception as e:
        print(f"训练NGBoost模型时出现异常: {str(e)}")
        return False

def run_single_simulation(task_id):
    """
    运行单次模拟
    返回Q表数据的UUID
    """
    try:
        # 生成唯一的UUID用于标识这次运行
        run_uuid = str(uuid.uuid4())
        
        # 构建命令 - 总是跳过NGBoost训练，因为已经在主函数中统一处理了
        cmd = [
            sys.executable, 
            MAIN_SCRIPT_PATH, 
            '--skip-hyperparameter-search',
            '--skip-ngboost-training',
            f'--run-uuid={run_uuid}'
        ]
        
        # 设置环境变量，使matplotlib使用非交互式后端
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'  # 使用非交互式后端
        env['PYTHONUNBUFFERED'] = '1'  # 确保输出不被缓冲
        
        # 执行命令
        start_time = time.time()
        print(f"任务 {task_id} 开始执行Q-learning训练...")
        result = subprocess.run(
            cmd, 
            cwd=os.path.dirname(MAIN_SCRIPT_PATH), 
            capture_output=True, 
            text=True,
            timeout=6400,  # 60分钟超时，增加超时时间
            env=env  # 传递环境变量
        )
        end_time = time.time()
        
        # 检查是否成功
        if result.returncode != 0:
            print(f"任务 {task_id} 失败: {result.stderr}")
            return None, None, None, None
            
        # 检查Q表数据是否已存储到临时文件中
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"q_table_{run_uuid}.csv")
        
        if not os.path.exists(temp_file_path):
            print(f"任务 {task_id} 未找到Q表数据")
            return None, None, None, None
        
        print(f"任务 {task_id} 完成，耗时: {end_time - start_time:.2f} 秒")
        return run_uuid, None, None, end_time - start_time
        
    except subprocess.TimeoutExpired:
        print(f"任务 {task_id} 超时")
        return None, None, None, None
    except Exception as e:
        print(f"任务 {task_id} 出现异常: {str(e)}")
        return None, None, None, None

def analyze_q_table(q_table_uuid):
    """
    分析单个Q表，提取最优动作
    返回每个状态的最优动作
    """
    try:
        # 从临时文件中获取Q表数据
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"q_table_{q_table_uuid}.csv")
        
        if not os.path.exists(temp_file_path):
            print(f"未找到UUID为 {q_table_uuid} 的Q表临时文件")
            return {}
            
        # 从临时文件读取Q表数据
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            q_table_str = f.read()
        
        df = pd.read_csv(pd.io.common.StringIO(q_table_str))
        
        # 获取状态列
        state_col = 'state' if 'state' in df.columns else df.columns[0]
        
        # 提取最优动作
        best_actions = {}
        for _, row in df.iterrows():
            state = row[state_col]
            
            # 首先尝试使用Q表中已有的best_action列
            if 'best_action' in df.columns:
                best_action = row['best_action']
                best_actions[state] = int(best_action)
            else:
                # 如果没有best_action列，则通过Q值重新计算（兼容旧格式）
                # 获取所有动作列（排除状态列）
                action_cols = [col for col in df.columns if col != state_col and col.startswith('action_')]
                if not action_cols:
                    # 如果列名不是action_格式，则尝试其他方式
                    action_cols = [col for col in df.columns if col != state_col]
                
                # 获取Q值
                q_values = [row[col] for col in action_cols]
                
                # 找到最优动作
                best_action_idx = np.argmax(q_values)
                best_actions[state] = best_action_idx
            
        return best_actions
        
    except Exception as e:
        print(f"分析Q表 {q_table_path} 时出错: {str(e)}")
        return {}

def aggregate_best_actions(all_best_actions):
    """
    聚合所有运行的最优动作，计算频数分布
    """
    # 初始化动作计数器
    action_counts = Counter()
    state_action_counts = defaultdict(Counter)
    
    # 统计每个动作出现的次数
    for run_id, best_actions in enumerate(all_best_actions):
        for state, action in best_actions.items():
            action_counts[action] += 1
            state_action_counts[state][action] += 1
    
    return action_counts, state_action_counts

def generate_action_frequency_plot(action_counts, total_runs, output_path):
    """
    生成动作频数分布图
    """
    # 创建动作标签
    online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个动作）
    offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个动作）
    
    # 生成36个动作的标签（线上价格×线下价格组合）
    action_labels = []
    for online_idx in range(6):
        for offline_idx in range(6):
            online_price = online_prices[online_idx]
            offline_price = offline_prices[offline_idx]
            action_labels.append(f'线上¥{online_price}\n线下¥{offline_price}')
    
    # 转换为DataFrame
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    frequencies = [count / total_runs for count in counts]
    
    # 按动作索引排序
    sorted_indices = np.argsort(actions)
    sorted_actions = [actions[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    sorted_frequencies = [frequencies[i] for i in sorted_indices]
    sorted_labels = [action_labels[action] for action in sorted_actions]
    
    # 创建图表
    plt.figure(figsize=(20, 10))
    
    # 子图1: 频数分布
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(sorted_actions)), sorted_counts, color='skyblue', alpha=0.7)
    plt.xlabel('定价动作', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title(f'最优动作频数分布 (总运行次数: {total_runs})', fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_actions)), sorted_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, sorted_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom')
    
    # 子图2: 频率分布
    plt.subplot(2, 1, 2)
    bars = plt.bar(range(len(sorted_actions)), sorted_frequencies, color='lightcoral', alpha=0.7)
    plt.xlabel('定价动作', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.title(f'最优动作频率分布', fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_actions)), sorted_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, freq in zip(bars, sorted_frequencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'{freq:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_heatmap(state_action_counts, output_path):
    """
    生成状态-动作热力图，显示每个状态下最优动作的分布
    """
    # 创建动作标签
    online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个动作）
    offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个动作）
    
    # 生成36个动作的标签（线上价格×线下价格组合）
    action_labels = []
    for online_idx in range(6):
        for offline_idx in range(6):
            online_price = online_prices[online_idx]
            offline_price = offline_prices[offline_idx]
            action_labels.append(f'线上¥{online_price}\n线下¥{offline_price}')
    
    # 获取所有状态并排序
    states = sorted(state_action_counts.keys())
    
    # 创建矩阵
    matrix = np.zeros((len(states), 36))
    
    for i, state in enumerate(states):
        for action, count in state_action_counts[state].items():
            matrix[i, action] = count
    
    # 创建状态标签
    state_labels = []
    for state in states:
        # 状态编码：库存等级(0-4) × 3(季节) × 2(日期类型) = 30种状态
        state_value = state
        inventory_level = state_value // 6  # 5种库存等级 (0-4)
        remaining = state_value % 6
        season = remaining // 2  # 3种季节 (0-2)
        day_type = remaining % 2  # 2种日期类型 (0-1)
        
        # 库存等级描述
        inventory_descriptions = ['0-20间', '21-40间', '41-60间', '61-80间', '81-100间']
        # 季节描述
        season_descriptions = ['淡季', '平季', '旺季']
        # 日期类型描述
        day_type_descriptions = ['工作日', '周末']
        
        state_label = f"{inventory_descriptions[inventory_level]}\n{season_descriptions[season]}\n{day_type_descriptions[day_type]}"
        state_labels.append(state_label)
    
    # 创建热力图
    plt.figure(figsize=(20, 12))
    sns.heatmap(matrix, 
                xticklabels=action_labels, 
                yticklabels=state_labels,
                cmap='Blues', 
                annot=True, 
                fmt='.0f',
                cbar_kws={'label': '最优动作出现次数'})
    
    plt.title('状态-最优动作分布热力图', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('定价动作', fontsize=12, fontweight='bold')
    plt.ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def save_state_action_table(state_action_counts, output_dir):
    """
    保存状态-动作频数表，与热力图格式相同
    """
    # 创建动作标签
    online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个动作）
    offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个动作）
    
    # 生成36个动作的标签（线上价格×线下价格组合）
    action_labels = []
    for online_idx in range(6):
        for offline_idx in range(6):
            online_price = online_prices[online_idx]
            offline_price = offline_prices[offline_idx]
            action_labels.append(f'线上¥{online_price} 线下¥{offline_price}')
    
    # 获取所有状态并排序
    states = sorted(state_action_counts.keys())
    
    # 创建矩阵
    matrix = np.zeros((len(states), 36))
    
    for i, state in enumerate(states):
        for action, count in state_action_counts[state].items():
            matrix[i, action] = count
    
    # 创建状态标签
    state_labels = []
    for state in states:
        # 状态编码：库存等级(0-4) × 3(季节) × 2(日期类型) = 30种状态
        state_value = state
        inventory_level = state_value // 6  # 5种库存等级 (0-4)
        remaining = state_value % 6
        season = remaining // 2  # 3种季节 (0-2)
        day_type = remaining % 2  # 2种日期类型 (0-1)
        
        # 库存等级描述
        inventory_descriptions = ['0-20间', '21-40间', '41-60间', '61-80间', '81-100间']
        # 季节描述
        season_descriptions = ['淡季', '平季', '旺季']
        # 日期类型描述
        day_type_descriptions = ['工作日', '周末']
        
        state_label = f"{inventory_descriptions[inventory_level]}_{season_descriptions[season]}_{day_type_descriptions[day_type]}"
        state_labels.append(state_label)
    
    # 创建DataFrame
    df = pd.DataFrame(matrix, columns=action_labels, index=state_labels)
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, 'state_action_frequency_table.csv')
    df.to_csv(csv_path, encoding='utf-8-sig')
    
    # 保存为Excel格式，便于查看
    excel_path = os.path.join(output_dir, 'state_action_frequency_table.xlsx')
    df.to_excel(excel_path, engine='openpyxl')
    
    return csv_path, excel_path

def generate_state_2d_plot(state, state_action_counts, output_dir):
    """
    为单个状态生成二维的线上价格-线下价格频数图与CSV表
    """
    # 创建动作标签
    online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个动作）
    offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个动作）
    
    # 获取状态的动作计数
    action_counts = state_action_counts[state]
    
    # 创建6x6矩阵
    matrix = np.zeros((6, 6))
    for action, count in action_counts.items():
        online_idx = action // 6
        offline_idx = action % 6
        matrix[offline_idx, online_idx] = count
    
    # 生成二维热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, 
                xticklabels=[f'¥{p}' for p in online_prices], 
                yticklabels=[f'¥{p}' for p in offline_prices],
                cmap='Blues', 
                annot=True, 
                fmt='.0f',
                cbar_kws={'label': '频数'})
    
    plt.title(f'状态{state}价格策略频数分布', fontsize=16, fontweight='bold')
    plt.xlabel('线上价格', fontsize=12)
    plt.ylabel('线下价格', fontsize=12)
    
    # 保存二维热力图到临时位置
    heatmap_path = os.path.join(output_dir, f'state_{state}_2d_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建DataFrame
    df = pd.DataFrame(matrix, columns=[f'¥{p}' for p in online_prices], 
                      index=[f'¥{p}' for p in offline_prices])
    
    # 保存CSV到临时位置
    csv_path = os.path.join(output_dir, f'state_{state}_frequency_table.csv')
    df.to_csv(csv_path)
    
    return heatmap_path, csv_path

def generate_state_3d_plot(state, state_action_counts, output_dir):
    """
    为单个状态生成三维的线上价格-线下价格-频数图
    """
    # 创建动作标签
    online_prices = np.array([80, 90, 100, 110, 120, 130])  # X轴
    offline_prices = np.array([90, 105, 120, 135, 150, 165]) # Y轴
    
    # 获取状态的动作计数
    action_counts = state_action_counts[state]
    
    # 创建6x6矩阵
    frequencies = np.zeros(36)
    for action, count in action_counts.items():
        frequencies[action] = count
    
    # 创建网格点
    X, Y = np.meshgrid(online_prices, offline_prices)
    Z = frequencies.reshape(6, 6).T  # 转置以匹配X,Y维度
    
    # 创建更高分辨率的网格用于平滑插值
    xi = np.linspace(80, 130, 100)
    yi = np.linspace(90, 165, 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # 准备原始点坐标（用于插值，不显示）
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = Z.ravel()
    
    # 使用RBF插值进行平滑
    try:
        rbf = Rbf(points[:,0], points[:,1], values, function='multiquadric', smooth=0.1)
        zi_grid = rbf(xi_grid, yi_grid)
    except:
        # 如果插值失败，使用原始数据
        zi_grid = np.zeros_like(xi_grid)
        for i in range(len(xi)):
            for j in range(len(yi)):
                # 找到最近的原始数据点
                distances = np.sqrt((points[:,0] - xi[i])**2 + (points[:,1] - yi[j])**2)
                nearest_idx = np.argmin(distances)
                zi_grid[j,i] = values[nearest_idx]
    
    # 创建3D图形
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制平滑曲面
    surf = ax.plot_surface(xi_grid, yi_grid, zi_grid, 
                          cmap='viridis', 
                          alpha=0.9, 
                          linewidth=0, 
                          antialiased=True, 
                          rstride=1, 
                          cstride=1)
    
    # 设置坐标轴标签
    ax.set_xlabel('线上价格 (¥)', labelpad=15, fontsize=14)
    ax.set_ylabel('线下价格 (¥)', labelpad=15, fontsize=14)
    ax.set_zlabel('频数', labelpad=15, fontsize=14)
    ax.set_title(f'状态{state}价格策略频数分布 - Q-learning优化结果 (平滑曲面)', fontsize=18, pad=20)
    
    # 设置刻度
    ax.set_xticks([80, 90, 100, 110, 120, 130])
    ax.set_yticks([90, 105, 120, 135, 150, 165])
    ax.set_zticks(np.arange(0, max(15, np.max(values)+2), 2))
    
    # 调整视角
    ax.view_init(elev=30, azim=-45)
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('频数 (插值)', fontsize=12)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存3D图到临时位置
    plot_3d_path = os.path.join(output_dir, f'state_{state}_3d_surface.png')
    plt.savefig(plot_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_3d_path

def generate_state_descriptions(state):
    """
    生成状态的描述信息
    """
    # 状态编码：库存等级(0-4) × 3(季节) × 2(日期类型) = 30种状态
    state_value = state
    inventory_level = state_value // 6  # 5种库存等级 (0-4)
    remaining = state_value % 6
    season = remaining // 2  # 3种季节 (0-2)
    day_type = remaining % 2  # 2种日期类型 (0-1)
    
    # 库存等级描述
    inventory_descriptions = ['0-20间', '21-40间', '41-60间', '61-80间', '81-100间']
    # 季节描述
    season_descriptions = ['淡季', '平季', '旺季']
    # 日期类型描述
    day_type_descriptions = ['工作日', '周末']
    
    state_description = f"{inventory_descriptions[inventory_level]}_{season_descriptions[season]}_{day_type_descriptions[day_type]}"
    state_label = f"{inventory_descriptions[inventory_level]}\n{season_descriptions[season]}\n{day_type_descriptions[day_type]}"
    
    return state_description, state_label

def generate_all_state_plots(state_action_counts, output_dir):
    """
    为所有状态生成二维和三维图表
    """
    # 创建状态分析目录
    states_dir = os.path.join(output_dir, 'state_analysis')
    os.makedirs(states_dir, exist_ok=True)
    
    # 为每个状态生成图表
    for state in sorted(state_action_counts.keys()):
        print(f"生成状态 {state} 的图表...")
        
        # 生成状态描述
        state_description, state_label = generate_state_descriptions(state)
        
        # 创建状态子目录
        state_dir = os.path.join(states_dir, f"state_{state}_{state_description}")
        os.makedirs(state_dir, exist_ok=True)
        
        # 生成二维图表
        heatmap_path, csv_path = generate_state_2d_plot(state, state_action_counts, states_dir)
        
        # 移动文件到正确的状态目录
        final_heatmap_path = os.path.join(state_dir, f'state_{state}_2d_heatmap.png')
        final_csv_path = os.path.join(state_dir, f'state_{state}_frequency_table.csv')
        
        if os.path.exists(heatmap_path):
            os.rename(heatmap_path, final_heatmap_path)
        if os.path.exists(csv_path):
            os.rename(csv_path, final_csv_path)
        
        # 生成三维图表
        plot_3d_path = generate_state_3d_plot(state, state_action_counts, states_dir)
        
        # 移动3D图表到正确的状态目录
        final_3d_path = os.path.join(state_dir, f'state_{state}_3d_surface.png')
        if os.path.exists(plot_3d_path):
            os.rename(plot_3d_path, final_3d_path)
        
        # 创建状态描述文件
        desc_path = os.path.join(state_dir, 'state_description.txt')
        with open(desc_path, 'w', encoding='utf-8') as f:
            f.write(f"状态ID: {state}\n")
            f.write(f"状态描述: {state_label}\n")
            f.write(f"状态编码: 库存等级={state//6}, 季节={(state%6)//2}, 日期类型={(state%6)%2}\n")
        
        print(f"状态 {state} 的图表已保存到: {state_dir}")
    
    return states_dir

def save_results(action_counts, state_action_counts, total_runs, output_dir):
    """
    保存分析结果
    """
    # 创建结果字典
    results = {
        'total_runs': total_runs,
        'analysis_time': datetime.now().isoformat(),
        'action_frequency': {str(int(k)): int(v) for k, v in action_counts.items()},
        'state_action_distribution': {str(int(k)): {str(int(k2)): int(v2) for k2, v2 in v.items()} for k, v in state_action_counts.items()}
    }
    
    # 保存为JSON
    json_path = os.path.join(output_dir, 'action_frequency_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 创建动作频数表
    action_freq_df = pd.DataFrame([
        {'action': int(action), 'count': int(count), 'frequency': count/total_runs}
        for action, count in action_counts.items()
    ]).sort_values('action')
    
    # 添加动作描述
    online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个动作）
    offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个动作）
    
    action_descriptions = []
    for action in action_freq_df['action']:
        online_idx = action // 6
        offline_idx = action % 6
        online_price = online_prices[online_idx]
        offline_price = offline_prices[offline_idx]
        action_descriptions.append(f'线上¥{online_price} 线下¥{offline_price}')
    
    action_freq_df['description'] = action_descriptions
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, 'action_frequency_table.csv')
    action_freq_df.to_csv(csv_path, index=False)
    
    # 保存状态-动作频数表
    state_csv_path, state_excel_path = save_state_action_table(state_action_counts, output_dir)
    
    return json_path, csv_path, state_csv_path, state_excel_path

def main():
    parser = argparse.ArgumentParser(description='Q表最优动作频数分析控制器')
    parser.add_argument('--num-runs', type=int, default=10, 
                        help='运行main.py的次数 (默认: 10)')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='最大并行工作进程数 (默认: CPU核心数)')
    parser.add_argument('--episodes', type=int, default=300,
                        help='每次运行的训练轮数 (默认: 300)')
    
    args = parser.parse_args()
    
    # 设置最大工作进程数，限制最大并行数以避免资源竞争
    cpu_count = multiprocessing.cpu_count()
    if args.max_workers is None:
        # 默认使用CPU核心数的一半，避免资源竞争
        max_workers = max(1, cpu_count // 2)
    else:
        max_workers = min(args.max_workers, cpu_count)
    
    print("=" * 60)
    print("Q表最优动作频数分析控制器")
    print("=" * 60)
    print(f"运行次数: {args.num_runs}")
    print(f"CPU核心数: {cpu_count}")
    print(f"最大并行进程数: {max_workers}")
    print(f"每次训练轮数: {args.episodes}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 统一检测NGBoost模型
    print("\n检测NGBoost模型...")
    models_exist = check_ngboost_models()
    if models_exist:
        print("检测到NGBoost模型存在，将跳过NGBoost训练")
    else:
        print("未检测到NGBoost模型，将先训练NGBoost模型...")
        if not train_ngboost_models():
            print("NGBoost模型训练失败，退出。")
            return
        print("NGBoost模型训练完成，继续执行Q-learning训练")
    
    # 临时修改config.py中的episodes参数
    config_path = os.path.join(CURRENT_DIR, '01_核心代码', 'config.py')
    backup_config_path = os.path.join(CURRENT_DIR, '01_核心代码', 'config.py.backup')
    
    try:
        # 备份原始配置文件
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            with open(backup_config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
        
        # 修改episodes参数
        modified_config = config_content.replace(
            "'episodes': 300, # 离线预训练轮数（从150增加到300，确保充分训练）",
            f"'episodes': {args.episodes}, # 离线预训练轮数（从150增加到300，确保充分训练）"
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(modified_config)
        
        # 运行多次模拟
        print(f"\n开始运行 {args.num_runs} 次模拟...")
        start_time = time.time()
        
        # 使用进度条
        all_results = []
        successful_runs = 0
        failed_runs = 0
        
        # 使用进程池，但限制并发任务数
        with tqdm(total=args.num_runs, desc="运行进度") as pbar:
            # 使用较小的批次大小，避免同时启动太多进程
            batch_size = max_workers
            for batch_start in range(0, args.num_runs, batch_size):
                batch_end = min(batch_start + batch_size, args.num_runs)
                batch_tasks = list(range(batch_start, batch_end))
                
                # 使用进程池处理当前批次
                with multiprocessing.Pool(processes=max_workers) as pool:
                    # 提交当前批次的任务
                    results = pool.imap_unordered(run_single_simulation, batch_tasks)
                    
                    # 收集结果
                    for result in results:
                        all_results.append(result)
                        if result[0] is not None:
                            successful_runs += 1
                        else:
                            failed_runs += 1
                        pbar.update(1)
                
                # 批次间短暂休息，释放资源
                if batch_end < args.num_runs:
                    time.sleep(2)
        
        end_time = time.time()
        print(f"\n所有模拟完成，总耗时: {end_time - start_time:.2f} 秒")
        print(f"成功运行: {successful_runs}/{args.num_runs}, 失败: {failed_runs}")
        
        # 过滤有效结果
        valid_results = [r for r in all_results if r[0] is not None]
        print(f"有效运行次数: {len(valid_results)}/{args.num_runs}")
        
        if not valid_results:
            print("没有有效的运行结果，退出。")
            return
        
        # 分析所有Q表
        print("\n分析Q表...")
        all_best_actions = []
        
        for i, (q_table_uuid, stats_path, heatmap_path, run_time) in enumerate(valid_results):
            print(f"分析第 {i+1}/{len(valid_results)} 个Q表...")
            best_actions = analyze_q_table(q_table_uuid)
            all_best_actions.append(best_actions)
        
        # 聚合最优动作
        print("\n聚合最优动作...")
        action_counts, state_action_counts = aggregate_best_actions(all_best_actions)
        
        # 生成可视化
        print("\n生成可视化图表...")
        
        # 1. 动作频数分布图
        freq_plot_path = os.path.join(OUTPUT_DIR, 'action_frequency_distribution.png')
        generate_action_frequency_plot(action_counts, len(valid_results), freq_plot_path)
        print(f"动作频数分布图已保存到: {freq_plot_path}")
        
        # 2. 状态-动作热力图
        heatmap_path = os.path.join(OUTPUT_DIR, 'state_action_heatmap.png')
        generate_heatmap(state_action_counts, heatmap_path)
        print(f"状态-动作热力图已保存到: {heatmap_path}")
        
        # 3. 为每个状态生成单独的二维和三维图表
        print("\n为每个状态生成详细分析图表...")
        states_dir = generate_all_state_plots(state_action_counts, OUTPUT_DIR)
        print(f"所有状态分析图表已保存到: {states_dir}")
        
        # 保存结果
        print("\n保存分析结果...")
        json_path, csv_path, state_csv_path, state_excel_path = save_results(action_counts, state_action_counts, len(valid_results), OUTPUT_DIR)
        print(f"分析结果已保存到: {json_path}")
        print(f"动作频数表已保存到: {csv_path}")
        print(f"状态-动作频数表已保存到: {state_csv_path}")
        print(f"状态-动作频数Excel表已保存到: {state_excel_path}")
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("分析摘要")
        print("=" * 60)
        print(f"总运行次数: {len(valid_results)}")
        print(f"分析状态数: {len(state_action_counts)}")
        print(f"动作空间大小: 36")
        
        # 找出最常见的动作
        most_common_action, most_common_count = action_counts.most_common(1)[0]
        online_idx = most_common_action // 6
        offline_idx = most_common_action % 6
        online_price = [80, 90, 100, 110, 120, 130][online_idx]
        offline_price = [90, 105, 120, 135, 150, 165][offline_idx]
        
        print(f"最常见动作: 线上¥{online_price} 线下¥{offline_price} (出现次数: {most_common_count}/{len(valid_results)})")
        
        # 计算动作多样性
        unique_actions = len(action_counts)
        print(f"使用的不同动作数: {unique_actions}/36")
        
        print("\n分析完成!")
        
    finally:
        # 恢复原始配置文件
        if os.path.exists(backup_config_path):
            os.replace(backup_config_path, config_path)
            print("已恢复原始配置文件")

if __name__ == "__main__":
    main()