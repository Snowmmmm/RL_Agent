# -*- coding: utf-8 -*-
"""
训练监控器 - 记录和可视化训练过程中的关键指标
"""

# 标准库导入
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class TrainingMonitor:
    """
    训练过程监控器
    
    全面记录和可视化强化学习训练过程中的关键指标，包括BNN训练、RL训练、
    在线学习等各个阶段的性能指标。
    
    主要功能：
    - 记录BNN训练损失和验证损失
    - 记录RL训练的平均奖励、轮次长度、探索率
    - 记录在线学习的每日奖励、库存、价格变化
    - 记录Q值统计信息
    - 生成训练曲线和可视化图表
    - 保存训练指标到多种格式（JSON、CSV、Pickle）
    - 提供训练摘要和统计分析
    
    支持的指标类型：
    - BNN指标：训练损失、验证损失、训练轮次
    - RL指标：轮次奖励、轮次长度、探索率、Q值统计
    - 在线指标：每日奖励、库存变化、价格变化
    - 离线指标：每日训练数据记录
    
    Attributes:
        save_dir (str): 保存目录路径
        bnn_train_losses (List[float]): BNN训练损失历史
        bnn_val_losses (List[float]): BNN验证损失历史
        bnn_epochs (List[int]): BNN训练轮次
        rl_episode_rewards (List[float]): RL轮次平均奖励
        rl_episode_lengths (List[int]): RL轮次长度
        rl_exploration_rates (List[float]): 探索率变化
        rl_episodes (List[int]): RL训练轮次
        online_rewards (List[float]): 在线学习每日奖励
        online_days (List[int]): 在线学习天数
        online_inventory (List[int]): 库存变化
        online_prices (List[float]): 价格变化
        q_value_stats (List[Dict]): Q值统计信息
        start_time (datetime): 训练开始时间
    """
    
    def __init__(self, save_dir: str = '../05_分析报告'):
        """
        初始化训练监控器
        
        创建保存目录并初始化所有指标存储列表，用于记录训练过程中的各种性能指标。
        
        Args:
            save_dir (str): 保存目录路径，默认为'../05_分析报告'
            
        Note:
            - 自动创建保存目录（如果不存在）
            - 初始化所有指标存储列表为空
            - 记录训练开始时间
            - 支持多种指标类型的并行记录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练指标存储
        self.bnn_train_losses = []  # BNN训练损失
        self.bnn_val_losses = []    # BNN验证损失
        self.bnn_epochs = []        # BNN训练轮次
        
        self.rl_episode_rewards = []      # RL每轮平均奖励
        self.rl_episode_lengths = []      # RL每轮长度
        self.rl_exploration_rates = []    # 探索率变化
        self.rl_episodes = []             # RL训练轮次
        
        self.online_rewards = []      # 在线学习每日奖励
        self.online_days = []         # 在线学习天数
        self.online_inventory = []    # 库存变化
        self.online_prices = []       # 价格变化
        
        # Q值统计
        self.q_value_stats = []       # Q值统计信息
        
        # 开始时间
        self.start_time = datetime.now()
        
    def record_bnn_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None) -> None:
        """
        记录BNN训练轮次指标
        
        将BNN训练的轮次信息、训练损失和可选的验证损失记录到对应列表中。
        
        Args:
            epoch (int): 训练轮次编号
            train_loss (float): 训练损失值
            val_loss (Optional[float]): 验证损失值，可选参数
            
        Note:
            - 验证损失可以为None，表示该轮次没有验证
            - 所有指标按时间顺序追加到列表末尾
            - 支持训练和验证损失的并行记录
        """
        self.bnn_epochs.append(epoch)
        self.bnn_train_losses.append(train_loss)
        self.bnn_val_losses.append(val_loss)
        
    def record_rl_episode(self, episode: int, avg_reward: float, episode_length: int, 
                         exploration_rate: float, q_stats: Optional[Dict[str, Any]] = None) -> None:
        """
        记录RL训练轮次指标
        
        记录强化学习训练的轮次信息，包括平均奖励、轮次长度、探索率和Q值统计。
        
        Args:
            episode (int): 训练轮次编号
            avg_reward (float): 该轮次的平均奖励
            episode_length (int): 轮次长度（步数）
            exploration_rate (float): 探索率（epsilon值）
            q_stats (Optional[Dict[str, Any]]): Q值统计信息，可选参数
            
        Note:
            - Q值统计信息包含Q值的最大值、最小值、平均值等
            - 探索率用于监控epsilon衰减策略
            - 所有指标按时间顺序追加到列表末尾
        """
        self.rl_episodes.append(episode)
        self.rl_episode_rewards.append(avg_reward)
        self.rl_episode_lengths.append(episode_length)
        self.rl_exploration_rates.append(exploration_rate)
        
        if q_stats:
            q_stats['episode'] = episode
            self.q_value_stats.append(q_stats)
            
    def record_online_day(self, day: int, reward: float, inventory: int, price: float) -> None:
        """
        记录在线学习每日指标
        
        记录在线学习阶段的每日表现指标，包括奖励、库存水平和定价决策。
        
        Args:
            day (int): 天数编号
            reward (float): 当日奖励值
            inventory (int): 当日库存水平
            price (float): 当日定价决策
            
        Note:
            - 用于监控在线学习阶段的性能表现
            - 支持库存和定价策略的可视化分析
            - 所有指标按时间顺序追加到列表末尾
        """
        self.online_days.append(day)
        self.online_rewards.append(reward)
        self.online_inventory.append(inventory)
        self.online_prices.append(price)
    
    def record_daily_training(self, episode: int, day: int, reward: float, inventory: int, price: float) -> None:
        """记录离线训练每日数据"""
        # 初始化数据结构
        if not hasattr(self, 'daily_training_data'):
            self.daily_training_data = []
        
        self.daily_training_data.append({
            'episode': episode,
            'day': day,
            'reward': reward,
            'inventory': inventory,
            'price': price
        })
        
    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要
        
        计算并返回训练过程的统计摘要，包括训练时长、最终性能指标、
        最佳性能指标等关键信息。
        
        Returns:
            Dict[str, Any]: 训练摘要字典，包含以下字段：
                - training_duration (str): 训练总时长
                - bnn_epochs (int): BNN训练轮次数
                - rl_episodes (int): RL训练轮次数
                - online_days (int): 在线学习天数
                - final_bnn_train_loss (float): 最终BNN训练损失
                - final_bnn_val_loss (float): 最终BNN验证损失
                - final_rl_avg_reward (float): 最终RL平均奖励
                - final_exploration_rate (float): 最终探索率
                - best_rl_avg_reward (float): 最佳RL平均奖励
                - avg_online_reward (float): 在线学习平均奖励
                
        Note:
            - 所有指标基于已记录的数据计算
            - 对于空列表返回None值
            - 训练时长格式为"HH:MM:SS"
        """
        summary = {
            'training_duration': str(datetime.now() - self.start_time),
            'bnn_epochs': len(self.bnn_epochs),
            'rl_episodes': len(self.rl_episodes),
            'online_days': len(self.online_days),
            'final_bnn_train_loss': self.bnn_train_losses[-1] if self.bnn_train_losses else None,
            'final_bnn_val_loss': self.bnn_val_losses[-1] if self.bnn_val_losses and self.bnn_val_losses[-1] is not None else None,
            'final_rl_avg_reward': self.rl_episode_rewards[-1] if self.rl_episode_rewards else None,
            'final_exploration_rate': self.rl_exploration_rates[-1] if self.rl_exploration_rates else None,
            'best_rl_avg_reward': max(self.rl_episode_rewards) if self.rl_episode_rewards else None,
            'avg_online_reward': np.mean(self.online_rewards) if self.online_rewards else None,
        }
        return summary
        
    def save_metrics(self, filename_prefix: str = 'training_metrics') -> str:
        """
        保存训练指标到文件
        
        将训练过程中的所有指标保存到多种格式的文件中，包括JSON、CSV和Pickle格式。
        支持BNN、RL、在线学习等不同阶段的指标保存。
        
        Args:
            filename_prefix (str): 文件名前缀，默认为'training_metrics'
            
        Returns:
            str: JSON文件的完整路径
            
        Note:
            - 自动生成时间戳避免文件名冲突
            - JSON格式包含完整的训练数据
            - CSV格式便于数据分析和可视化
            - Pickle格式保存完整的对象状态
            - 支持离线训练每日数据的单独保存
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存为JSON格式
        metrics_data = {
            'bnn': {
                'epochs': self.bnn_epochs,
                'train_losses': self.bnn_train_losses,
                'val_losses': self.bnn_val_losses
            },
            'rl': {
                'episodes': self.rl_episodes,
                'episode_rewards': self.rl_episode_rewards,
                'episode_lengths': self.rl_episode_lengths,
                'exploration_rates': self.rl_exploration_rates
            },
            'online': {
                'days': self.online_days,
                'rewards': self.online_rewards,
                'inventory': self.online_inventory,
                'prices': self.online_prices
            },
            'daily_training': self.daily_training_data if hasattr(self, 'daily_training_data') else [],
            'q_stats': self.q_value_stats,
            'summary': self.get_training_summary()
        }
        
        json_file = os.path.join(self.save_dir, f'{filename_prefix}_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2, default=str)
            
        # 保存为CSV格式（便于分析）
        if self.rl_episodes:
            df_rl = pd.DataFrame({
                'episode': self.rl_episodes,
                'avg_reward': self.rl_episode_rewards,
                'episode_length': self.rl_episode_lengths,
                'exploration_rate': self.rl_exploration_rates
            })
            csv_file = os.path.join(self.save_dir, f'rl_training_{timestamp}.csv')
            df_rl.to_csv(csv_file, index=False, encoding='utf-8')
            
        if self.online_days:
            df_online = pd.DataFrame({
                'day': self.online_days,
                'reward': self.online_rewards,
                'inventory': self.online_inventory,
                'price': self.online_prices
            })
            csv_file = os.path.join(self.save_dir, f'online_learning_{timestamp}.csv')
            df_online.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存离线训练每日数据为CSV
        if hasattr(self, 'daily_training_data') and self.daily_training_data:
            df_daily = pd.DataFrame(self.daily_training_data)
            csv_file = os.path.join(self.save_dir, f'daily_training_{timestamp}.csv')
            df_daily.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存pickle文件（完整对象状态）
        pickle_file = os.path.join(self.save_dir, f'{filename_prefix}_{timestamp}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)
            
        print(f"训练指标已保存到: {json_file}")
        print(f"  - Pickle: {pickle_file}")
        return json_file
    
    def load_from_pickle(self, pickle_file: str) -> bool:
        """
        从pickle文件加载训练监控器状态
        
        从之前保存的pickle文件中加载完整的训练监控器状态，包括所有记录的指标和历史数据。
        
        Args:
            pickle_file (str): pickle文件路径
            
        Returns:
            bool: 加载成功返回True，失败返回False
            
        Note:
            - 会覆盖当前实例的所有属性
            - 加载失败时会打印错误信息
            - 支持跨会话的状态恢复
            - 保持所有历史数据和指标完整性
        """
        try:
            with open(pickle_file, 'rb') as f:
                loaded_monitor = pickle.load(f)
            
            # 复制所有属性到当前实例
            for attr_name in dir(loaded_monitor):
                if not attr_name.startswith('_'):
                    attr_value = getattr(loaded_monitor, attr_name)
                    setattr(self, attr_name, attr_value)
            
            print(f"训练监控器状态已从 {pickle_file} 加载")
            return True
        except Exception as e:
            print(f"加载pickle文件失败: {e}")
            return False
        
    def plot_training_curves(self, save_plots: bool = True) -> List[str]:
        """
        绘制训练曲线
        
        生成训练过程的可视化图表，包括BNN损失曲线、RL奖励曲线、在线学习指标等。
        使用符合学术期刊标准的图表样式和字体设置。
        
        Args:
            save_plots (bool): 是否保存图表到文件，默认为True
            
        Returns:
            List[str]: 保存的图表文件路径列表
            
        Note:
            - 设置符合学术期刊的字体和样式
            - 支持多种图表类型的组合显示
            - 自动保存高分辨率图表
            - 包含完整的图例和标签
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_saved = []
        
        # 设置符合学术期刊的字体和样式
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴宽度
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.5
        plt.rcParams['xtick.minor.width'] = 1.0
        plt.rcParams['ytick.minor.width'] = 1.0
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
        # 1. BNN损失曲线
        if self.bnn_epochs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 更紧凑的尺寸
            
            # Training loss
            ax1.plot(self.bnn_epochs, self.bnn_train_losses, 'b-', linewidth=2, label='Training Loss')
            ax1.set_xlabel('Training Epoch')
            ax1.set_ylabel('Loss Value')
            ax1.set_title('BNN Training Loss')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Validation loss if available
            val_losses_filtered = [loss for loss in self.bnn_val_losses if loss is not None]
            if val_losses_filtered:
                val_epochs = [self.bnn_epochs[i] for i, loss in enumerate(self.bnn_val_losses) if loss is not None]
                ax1.plot(val_epochs, val_losses_filtered, 'r-', linewidth=2, label='Validation Loss')
                ax1.legend(frameon=True, framealpha=0.9)
            
            # Loss difference
            if len(self.bnn_train_losses) > 1:
                loss_diff = np.diff(self.bnn_train_losses)
                ax2.plot(self.bnn_epochs[1:], loss_diff, 'k-', linewidth=2)  # 使用更专业的黑色
                ax2.set_xlabel('Training Epoch')
                ax2.set_ylabel('Loss Decrease')
                ax2.set_title('Loss Decrease Rate')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            if save_plots:
                plot_file = os.path.join(self.save_dir, f'bnn_training_curves_{timestamp}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plots_saved.append(plot_file)
                plt.close()
            else:
                plt.show()
        
        # 2. RL训练曲线（按训练轮次）
        if self.rl_episodes:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))  # 更紧凑的尺寸
            
            # Average reward change (by training episodes)
            episodes_90_days = [ep * 90 for ep in self.rl_episodes]  # Convert to days
            
            # 在同一张图上显示原始数据和平滑数据
            ax1.plot(episodes_90_days, self.rl_episode_rewards, 'k-', linewidth=1, alpha=0.3, label='Raw Rewards')
            
            # 添加移动平均线
            if len(self.rl_episode_rewards) > 10:
                moving_avg = np.convolve(self.rl_episode_rewards, np.ones(10)/10, mode='valid')
                ax1.plot(episodes_90_days[9:], moving_avg, 'b-', linewidth=2, label='10-Episode Moving Average')
                ax1.legend(frameon=True, framealpha=0.9)
            
            ax1.set_xlabel('Training Days')
            ax1.set_ylabel('Average Reward')
            ax1.set_title('Average Reward')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Exploration rate change
            ax2.plot(episodes_90_days, self.rl_exploration_rates, 'r-', linewidth=2)
            ax2.set_xlabel('Training Days')
            ax2.set_ylabel('Exploration Rate (ε)')
            ax2.set_title('Exploration Rate Decay')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(0, 1)
            
            # Episode length
            ax3.plot(episodes_90_days, self.rl_episode_lengths, 'g-', linewidth=2)
            ax3.set_xlabel('Training Days')
            ax3.set_ylabel('Episode Length')
            ax3.set_title('Episode Length')
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            # Reward distribution
            ax4.hist(self.rl_episode_rewards, bins=20, alpha=0.7, color='gray', edgecolor='black')
            ax4.set_xlabel('Average Reward')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Reward Distribution')
            ax4.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            if save_plots:
                plot_file = os.path.join(self.save_dir, f'rl_training_curves_{timestamp}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plots_saved.append(plot_file)
                plt.close()
            else:
                plt.show()
        
        # 3. 在线学习曲线（按天）
        if self.online_days:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))  # 更紧凑的尺寸
            
            # Daily rewards
            ax1.plot(self.online_days, self.online_rewards, 'k-', linewidth=1, alpha=0.3)
            
            # Add moving average line
            if len(self.online_rewards) > 7:
                moving_avg = np.convolve(self.online_rewards, np.ones(7)/7, mode='valid')
                ax1.plot(self.online_days[6:], moving_avg, 'b-', linewidth=2, label='7-Day Moving Average')
                ax1.legend(frameon=True, framealpha=0.9)
            
            ax1.set_xlabel('Day')
            ax1.set_ylabel('Reward Value')
            ax1.set_title('Daily Rewards')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Inventory changes
            ax2.plot(self.online_days, self.online_inventory, 'g-', linewidth=2)
            ax2.set_xlabel('Day')
            ax2.set_ylabel('Inventory Count')
            ax2.set_title('Inventory Trend')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(0, 100)
            
            # Price changes
            ax3.plot(self.online_days, self.online_prices, 'r-', linewidth=2)
            ax3.set_xlabel('Day')
            ax3.set_ylabel('Price')
            ax3.set_title('Pricing Strategy')
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            # Reward vs Inventory relationship
            scatter = ax4.scatter(self.online_inventory, self.online_rewards, alpha=0.6, c=self.online_days, 
                       cmap='coolwarm', s=50, edgecolor='k', linewidth=0.5)  # 更专业的颜色映射和边框
            ax4.set_xlabel('Inventory Count')
            ax4.set_ylabel('Reward Value')
            ax4.set_title('Reward-Inventory Relationship')
            ax4.grid(True, alpha=0.3, linestyle='--')
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Day')
            
            plt.tight_layout()
            if save_plots:
                plot_file = os.path.join(self.save_dir, f'online_learning_curves_{timestamp}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plots_saved.append(plot_file)
                plt.close()
            else:
                plt.show()
        
        # 4. 离线训练每日数据可视化
        if hasattr(self, 'daily_training_data') and self.daily_training_data:
            # 使用英文显示，移除中文字体设置
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            plt.style.use('seaborn-v0_8-whitegrid')  # 使用美观的样式
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 将数据转换为DataFrame以便分析
            df_daily = pd.DataFrame(self.daily_training_data)
            
            # 统计每个episode的平均奖励和最终库存
            episode_stats = df_daily.groupby('episode').agg({
                'reward': ['mean', 'sum'],
                'inventory': 'last',
                'price': 'mean'
            }).reset_index()
            episode_stats.columns = ['episode', 'avg_reward', 'total_reward', 'final_inventory', 'avg_price']
            
            # 1. 每个episode的平均奖励和总奖励（条形图）
            ax1.bar(episode_stats['episode'] - 0.2, episode_stats['avg_reward'], width=0.4, label='Average Daily Reward')
            ax1.bar(episode_stats['episode'] + 0.2, episode_stats['total_reward'], width=0.4, label='Total Reward')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward Value')
            ax1.set_title('Reward Comparison Across Episodes')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. 选择最近的5个episode绘制每日奖励变化趋势
            recent_episodes = df_daily['episode'].unique()[-5:]
            for episode in recent_episodes:
                episode_data = df_daily[df_daily['episode'] == episode]
                color = plt.cm.viridis(float(episode) / max(df_daily['episode'].unique()))
                ax2.plot(episode_data['day'], episode_data['reward'], 
                        alpha=0.9, linewidth=2, label=f'Episode {episode}')
            
            # 添加整体平均奖励曲线
            daily_avg = df_daily.groupby('day')['reward'].mean().reset_index()
            ax2.plot(daily_avg['day'], daily_avg['reward'], 'k-', linewidth=3, alpha=0.8, label='Average Across All Episodes')
            
            ax2.set_xlabel('Day in Episode')
            ax2.set_ylabel('Daily Reward')
            ax2.set_title('Daily Reward Trends in Last 5 Episodes')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # 3. 库存与价格的关系（散点图）
            scatter = ax3.scatter(df_daily['inventory'], df_daily['price'], 
                                c=df_daily['day'], cmap='plasma', alpha=0.6, s=50)
            ax3.set_xlabel('Inventory Count')
            ax3.set_ylabel('Price (RMB)')
            ax3.set_title('Inventory vs Price Relationship (Colored by Day)')
            ax3.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Day in Episode')
            
            # 4. 奖励分布热力图（按库存和天数）
            # 创建库存-天数的网格，计算平均奖励
            heatmap_data = df_daily.pivot_table(values='reward', index='inventory', columns='day', aggfunc='mean')
            # 用最近一个episode的数据填充缺失值
            latest_episode = df_daily[df_daily['episode'] == max(df_daily['episode'])]
            latest_heatmap = latest_episode.pivot_table(values='reward', index='inventory', columns='day', aggfunc='mean')
            heatmap_data = heatmap_data.fillna(latest_heatmap)
            
            im = ax4.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
            ax4.set_xlabel('Day in Episode')
            ax4.set_ylabel('Inventory Count')
            ax4.set_title('Inventory-Day Reward Heatmap')
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Average Reward Value')
            
            # 添加标题和美化
            fig.suptitle('Offline Training Daily Data Analysis', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
            
            if save_plots:
                plot_file = os.path.join(self.save_dir, f'daily_training_curves_{timestamp}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plots_saved.append(plot_file)
                plt.close()
            else:
                plt.show()
        
        # 5. Q值统计变化
        if self.q_value_stats:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))  # 更紧凑的尺寸
            
            episodes = [stat['episode'] * 90 for stat in self.q_value_stats]  # 转换为天数
            
            # Zero Q-value percentage
            zero_q_percentages = [stat['zero_q_percentage'] for stat in self.q_value_stats]
            ax1.plot(episodes, zero_q_percentages, 'r-', linewidth=2)
            ax1.set_xlabel('Training Days')
            ax1.set_ylabel('Zero Q-value Percentage (%)')
            ax1.set_title('Q-value Exploration Progress')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylim(0, 100)
            
            # Exploration coverage
            exploration_coverages = [stat['exploration_coverage'] for stat in self.q_value_stats]
            ax2.plot(episodes, exploration_coverages, 'g-', linewidth=2)
            ax2.set_xlabel('Training Days')
            ax2.set_ylabel('Exploration Coverage (%)')
            ax2.set_title('State-Action Space Coverage')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(0, 100)
            
            # Average Q-values
            mean_q_values = [stat['mean_q_value'] for stat in self.q_value_stats]
            ax3.plot(episodes, mean_q_values, 'b-', linewidth=2)
            ax3.set_xlabel('Training Days')
            ax3.set_ylabel('Average Q-value')
            ax3.set_title('Average Q-value Evolution')
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            # State visit count
            state_visits = [stat['num_state_visits'] for stat in self.q_value_stats]
            ax4.plot(episodes, state_visits, 'k-', linewidth=2)  # 使用更专业的黑色
            ax4.set_xlabel('Training Days')
            ax4.set_ylabel('State Visit Count')
            ax4.set_title('State Visit Frequency')
            ax4.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            if save_plots:
                plot_file = os.path.join(self.save_dir, f'q_value_stats_{timestamp}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plots_saved.append(plot_file)
                plt.close()
            else:
                plt.show()
        
        print(f"训练曲线图已保存: {len(plots_saved)} 张")
        for plot_file in plots_saved:
            print(f"  - {plot_file}")
            
        return plots_saved
        
    def load_metrics(self, json_file: str) -> None:
        """从文件加载训练指标"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 恢复数据
        if 'bnn' in data:
            self.bnn_epochs = data['bnn']['epochs']
            self.bnn_train_losses = data['bnn']['train_losses']
            self.bnn_val_losses = data['bnn']['val_losses']
            
        if 'rl' in data:
            self.rl_episodes = data['rl']['episodes']
            self.rl_episode_rewards = data['rl']['episode_rewards']
            self.rl_episode_lengths = data['rl']['episode_lengths']
            self.rl_exploration_rates = data['rl']['exploration_rates']
            
        if 'online' in data:
            self.online_days = data['online']['days']
            self.online_rewards = data['online']['rewards']
            self.online_inventory = data['online']['inventory']
            self.online_prices = data['online']['prices']
            
        if 'q_stats' in data:
            self.q_value_stats = data['q_stats']
        
        if 'daily_training' in data:
            self.daily_training_data = data['daily_training']
            
        print(f"训练指标已从 {json_file} 加载")
        
    def generate_training_report(self) -> str:
        """Generate training report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("REINFORCEMENT LEARNING TRAINING REPORT")
            report.append("=" * 60)
            
            # Training summary
            report.append("\n[TRAINING SUMMARY]")
            report.append(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Total Training Days: {len(self.rl_episodes) * 90}")
            report.append(f"Total Simulation Days: {len(self.online_days)}")
            
            if self.rl_episode_rewards:
                avg_reward = np.mean(self.rl_episode_rewards)
                final_reward = self.rl_episode_rewards[-1]
                report.append(f"Average Reward: {avg_reward:.2f}")
                report.append(f"Final Reward: {final_reward:.2f}")
            
            # Key metrics
            report.append("\n[KEY METRICS]")
            if self.q_value_stats:
                latest_stats = self.q_value_stats[-1]
                report.append(f"Zero Q-value Percentage: {latest_stats['zero_q_percentage']:.2f}%")
                report.append(f"Exploration Coverage: {latest_stats['exploration_coverage']:.2f}%")
                report.append(f"Average Q-value: {latest_stats['mean_q_value']:.4f}")
                report.append(f"State Visit Count: {latest_stats['num_state_visits']}")
            
            # Training effectiveness analysis
            report.append("\n[TRAINING EFFECTIVENESS ANALYSIS]")
            if len(self.q_value_stats) > 1:
                first_stats = self.q_value_stats[0]
                last_stats = self.q_value_stats[-1]
                
                zero_q_improvement = first_stats['zero_q_percentage'] - last_stats['zero_q_percentage']
                coverage_improvement = last_stats['exploration_coverage'] - first_stats['exploration_coverage']
                
                report.append(f"Zero Q-value Percentage Improvement: {zero_q_improvement:.2f}%")
                report.append(f"Exploration Coverage Improvement: {coverage_improvement:.2f}%")
                
                if zero_q_improvement > 10:
                    report.append("[OK] Q-value exploration effectiveness good")
                else:
                    report.append("[WARNING] Q-value exploration may need more training")
                    
                if coverage_improvement > 20:
                    report.append("[OK] State space exploration sufficient")
                else:
                    report.append("[WARNING] State space exploration may be insufficient")
            
            report.append("\n" + "=" * 60)
            
            # Save report
            report_filename = f"../05_分析报告/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            print(f"Training report saved: {report_filename}")
            return '\n'.join(report)
            
        except Exception as e:
            print(f"Error when generating training report: {e}")
            return "Training report generation failed"


# 全局监控器实例
_training_monitor = None

def get_training_monitor() -> TrainingMonitor:
    """获取全局训练监控器实例"""
    global _training_monitor
    if _training_monitor is None:
        _training_monitor = TrainingMonitor()
    return _training_monitor