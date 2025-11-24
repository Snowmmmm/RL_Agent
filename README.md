### 1. 环境准备
运行环境 Python 3.12.0
```bash
cd RL_Agent
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保数据文件位于正确位置：
```
03_数据文件/
└── hotel_bookings.csv  # 酒店预订数据
```

### 3. 运行系统

#### 完整训练模式
```bash
cd 01_核心代码
# 完整训练流程（包含NGBoost超参数搜索和Q-learning训练）
python main.py
```

#### 仅训练NGBoost模型
```bash
cd 01_核心代码
# 只训练NGBoost模型，跳过Q-learning训练
python main.py --train-ngboost-only
```

#### 跳过超参数搜索模式
```bash
cd 01_核心代码
# 跳过NGBoost超参数搜索，使用预设最佳参数
python main.py --skip-hyperparameter-search
```

#### 跳过超参数搜索并仅训练NGBoost
```bash
cd 01_核心代码
# 跳过超参数搜索，仅使用预设参数训练NGBoost模型
python main.py --skip-hyperparameter-search --train-ngboost-only
```

#### 跳过训练模式
```bash
cd 01_核心代码
# 使用已有模型，跳过训练过程
python main.py --skip-training
```

#### 强制重新训练
```bash
cd 01_核心代码
# 强制重新训练所有模型（忽略已有模型）
python main.py --force-retrain
```

#### 贝叶斯Q-learning模式
```bash
cd 01_核心代码
# 仅运行贝叶斯Q-learning算法
python main.py --use-bayesian-rl
# 从头开始训练(含NGBoost+BQL)
python main.py --use-bayesian-rl --force-retrain
```

#### 训练方式组合说明

| 命令 | NGBoost超参数搜索 | NGBoost训练 | Q-learning训练 | 适用场景 |
|------|-------------------|------------|---------------|----------|
| `python main.py` | ✓ | ✓ | ✓ | 完整训练流程 |
| `python main.py --train-ngboost-only` | ✓ | ✓ | ✗ | 仅优化NGBoost模型 |
| `python main.py --skip-hyperparameter-search` | ✗ | ✓ | ✓ | 快速训练，使用预设参数 |
| `python main.py --skip-hyperparameter-search --train-ngboost-only` | ✗ | ✓ | ✗ | 仅NGBoost训练，使用预设参数 |
| `python main.py --skip-training` | ✗ | ✗ | ✗ | 使用已有模型进行预测 |
| `python main.py --force-retrain` | ✓ | ✓ | ✓ | 忽略已有模型，重新训练 |
| `python main.py --use-bayesian-rl` | ✓ | ✓ | 贝叶斯Q-learning | 使用贝叶斯方法 |


## 📁 项目结构

```
RL_Agent/
├── 01_核心代码/               # 核心代码目录
│   ├── main.py               # 主程序入口
│   ├── config.py             # 配置文件
│   ├── ngboost_model.py      # NGBoost模型
│   ├── rl_system.py          # 强化学习系统
│   ├── data_preprocessing.py # 数据预处理
│   ├── training_monitor.py   # 训练监控
├── 02_训练模型/               # 训练好的模型
├── 03_数据文件/               # 数据文件
│   └── hotel_bookings.csv    # 酒店预订数据
├── 04_结果输出/               # 结果输出
├── 05_分析报告/               # 分析报告
├── 06_临时文件/               # 临时文件
└── 07_备份文件/               # 备份文件
```


## 🎛️ 命令行参数

```bash
cd 01_核心代码
python main.py [选项]

选项:
  --skip-training          跳过训练，使用已有模型
  --train-ngboost-only     仅训练NGBoost模型，跳过Q-learning训练
  --skip-hyperparameter-search  跳过NGBoost超参数搜索，使用预设参数
  --force-retrain          强制重新训练所有模型
  --use-bayesian-rl        使用贝叶斯Q-learning算法（默认使用标准Q-learning）
  --help                   显示帮助信息
```

## 📊 Q表最优动作频数分析

### action_frequency_controller.py 使用说明

`action_frequency_controller.py` 是一个用于多次运行main.py并分析Q表最优动作频数分布的工具。它可以帮助评估强化学习算法的稳定性和收敛性。

#### 主要功能

1. **统一模型检测**：在开始所有任务前统一检测NGBoost模型是否存在，避免重复训练
2. **智能模型管理**：如果模型不存在，先统一训练NGBoost模型，然后所有任务跳过NGBoost训练
3. **多次运行模拟**：自动多次运行main.py，收集每次运行生成的Q表
4. **最优动作分析**：分析每个状态下最优动作的选择
5. **频数统计**：统计各动作被选为最优动作的频数和频率
6. **可视化分析**：生成动作频数分布图、状态-动作热力图等
7. **详细状态分析**：为每个状态生成单独的二维和三维价格策略分布图

#### 使用方法

```bash
# 基本用法（默认运行10次）
python action_frequency_controller.py

# 自定义运行次数
python action_frequency_controller.py --num-runs 20

# 设置最大并行进程数
python action_frequency_controller.py --max-workers 4

# 设置每次运行的训练轮数
python action_frequency_controller.py --episodes 500

# 组合使用
python action_frequency_controller.py --num-runs 30 --max-workers 6 --episodes 400
```

#### 参数说明

- `--num-runs`：运行main.py的次数（默认：10）
- `--max-workers`：最大并行工作进程数（默认：CPU逻辑处理器数量的一半）
- `--episodes`：每次运行的训练轮数（默认：300）

#### 工作流程

1. **模型检测阶段**：
   - 检查NGBoost模型文件是否存在
   - 如果不存在，统一训练NGBoost模型（所有任务共享）
   - 如果存在，跳过NGBoost训练

2. **并行执行阶段**：
   - 所有任务并行执行Q-learning训练
   - 每个任务使用相同的NGBoost模型，但独立进行Q-learning训练
   - 每个任务生成唯一UUID标识，并将Q表数据保存到临时文件
   - 收集每次运行生成的Q表

3. **分析阶段**：
   - 从临时文件中读取所有Q表数据
   - 分析所有Q表的最优动作
   - 统计动作频数分布
   - 生成可视化图表

#### 技术实现细节

1. **UUID机制**：
   - 每次运行生成唯一UUID标识
   - 通过UUID确保Q表数据的唯一性和可追溯性
   - 避免多进程环境下的数据冲突

2. **临时文件存储**：
   - 使用系统临时目录存储Q表数据
   - 文件命名格式：`q_table_{UUID}.csv`
   - 解决多进程环境下的内存共享问题
   - 自动清理机制，避免临时文件累积

3. **多进程优化**：
   - 批次处理机制，避免同时启动过多进程
   - 智能资源管理，根据CPU核心数调整并行度
   - 超时控制，防止单个任务阻塞整体进度

#### 输出结果

分析完成后，会在`action_frequency_analysis`目录下生成以下文件：

1. **action_frequency_distribution.png**：动作频数分布图
2. **state_action_heatmap.png**：状态-动作热力图
3. **action_frequency_analysis.json**：详细分析结果（JSON格式）
4. **action_frequency_table.csv**：动作频数表
5. **state_action_frequency_table.csv/excel**：状态-动作频数表
6. **state_analysis/**：包含每个状态的详细分析图表
   - 每个状态有单独的子目录，包含二维热力图、三维曲面图和频数表

#### 应用场景

1. **算法稳定性评估**：通过多次运行评估Q-learning算法的收敛稳定性
2. **策略一致性分析**：分析不同运行中最优策略的一致性
3. **参数调优**：评估不同参数设置对策略稳定性的影响
4. **学术研究**：为强化学习算法研究提供统计分析工具
5. **批量实验**：高效进行多次实验，避免重复NGBoost训练

#### 优化说明

最新版本优化了模型检测逻辑：
- **统一检测**：在开始所有任务前统一检测NGBoost模型
- **避免重复训练**：确保NGBoost模型只训练一次，所有任务共享
- **提高效率**：大幅减少总体运行时间，特别是在多次运行时
- **资源节约**：减少计算资源消耗，避免不必要的模型训练

## 💾 模型保存格式说明

### NGBoost模型保存格式

NGBoost模型使用`joblib`格式保存，文件扩展名为`.pkl`，保存路径为`02_训练模型/`目录。模型数据包含以下字段：

```python
model_data = {
    'model': self.model,                    # NGBoost模型实例
    'feature_scaler': self.feature_scaler,  # 特征标准化器
    'train_losses': self.train_losses,      # 训练损失历史
    'val_losses': self.val_losses,          # 验证损失历史
    'config': {                             # 模型配置参数
        'distribution': self.distribution,  # 分布类型
        'score': self.score,                # 评分函数
        'n_estimators': self.n_estimators,  # 估计器数量
        'learning_rate': self.learning_rate,# 学习率
        'max_depth': self.max_depth,      # 最大深度
        'min_samples_leaf': self.min_samples_leaf,  # 叶节点最小样本数
        'subsample': self.subsample,        # 子采样比例
        'random_state': self.random_state   # 随机种子
    }
}
```

**模型文件命名格式：**
- `ngboost_model_{customer_type}_{demand_type}.pkl`
- 例如：`ngboost_model_online_booked.pkl`、`ngboost_model_offline_actual.pkl`

### Q-Learning智能体保存格式

Q-Learning智能体使用`pickle`格式保存，文件扩展名为`.pkl`，保存路径为`02_训练模型/`目录。智能体数据包含以下字段：

```python
agent_data = {
    'q_table': q_table_dict,                    # Q值表（状态-动作对的Q值）
    'state_visit_count': state_visit_dict,      # 状态访问计数
    'state_action_visit_count': state_action_visit_dict,  # 状态-动作对访问计数
    'training_history': self.training_history,  # 训练历史记录
    'hyperparameters': {                        # 超参数设置
        'n_states': self.n_states,              # 状态数量
        'n_actions': self.n_actions,            # 动作数量
        'learning_rate': self.learning_rate,    # 学习率
        'discount_factor': self.discount_factor,# 折扣因子
        'epsilon_start': self.epsilon_start,  # 初始探索率
        'epsilon_end': self.epsilon_end,      # 最终探索率
        'epsilon_decay_steps': self.epsilon_decay_steps  # 探索率衰减步数
    }
}
```

**智能体文件命名格式：**
- `q_agent_{type}.pkl`
- 例如：`q_agent_pretrained.pkl`（预训练模型）、`q_agent_final.pkl`（最终模型）


### 需求标准化器保存格式

需求标准化器使用`joblib`格式保存，用于将预测结果转换回原始尺度：

```python
# 保存路径格式
demand_scaler_path = f'../02_训练模型/demand_scaler_{customer_type}_{demand_type}.pkl'
# 例如：demand_scaler_线上用户_booked.pkl、demand_scaler_线下用户_actual.pkl
```

### 数据预处理器保存格式

数据预处理器使用`pickle`格式保存，包含完整的`HotelDataPreprocessor`实例状态，确保数据预处理的一致性和可重现性。

**保存内容：**

```python
# 保存路径
preprocessor_path = '../02_训练模型/preprocessor.pkl'

# 预处理器保存的完整状态包括：
preprocessor_state = {
    'feature_columns': List[str],           # 特征列名列表（共50+个特征）
    'scaler': Optional[Any],                # 数据预处理器中的特征标准化器（StandardScaler）
    'categorical_encoders': Dict[str, Any], # 分类变量编码器字典
    
    # 完整的处理流水线状态：
    # - 数据清洗规则（ADR范围0-500元，成人数最多4人等）
    # - 客户分类逻辑（线上用户：market_segment='Online TA' 或 distribution_channel='TA/TO'）
    # - 特征工程参数（滞后窗口、滚动统计周期等）
    # - 缺失值处理策略（前后向填充）
}
```

**核心功能模块：**

1. **数据清洗模块**
   - 缺失值处理：children、agent、company字段填充0
   - 异常值截断：ADR>500元截断为500元，成人数>4人截断为4人
   - 客户分类：自动区分线上用户与线下用户

2. **需求标签构造模块**
   - 构造每日需求标签（区分预定需求和实际需求）
   - 处理住宿顺延：将多晚住宿分布到对应日期
   - 价格统计：基于实际入住订单计算平均价格、标准差等

3. **特征工程模块**
   - **时间特征**：年、月、日、星期、季度、周数
   - **滞后特征**：1、2、3、7、14、30天滞后值
   - **滚动统计**：3、7、14、30天移动平均和标准差
   - **节假日特征**：周末标识、月初月末标识、季节编码
   - **需求-价格关系**：价格比率、价格弹性、取消率
   - **趋势特征**：基于7天窗口的线性趋势

4. **特征列表（50+个特征）**
   ```python
   # 基础时间特征（6个）
   ['year', 'month', 'day', 'dayofweek', 'quarter', 'weekofyear']
   
   # 双需求滞后特征（12个）
   ['booked_demand_lag_1/2/3/7/14/30', 'actual_demand_lag_1/2/3/7/14/30']
   
   # 双需求滚动统计（16个）
   ['booked_demand_ma_3/7/14/30', 'booked_demand_std_3/7/14/30',
    'actual_demand_ma_3/7/14/30', 'actual_demand_std_3/7/14/30']
   
   # 价格相关特征（18个+）
   ['price_lag_1/2/3/7/14/30', 'price_ma_3/7/14/30', 'price_std_3/7/14/30',
    'price_range', 'price_cv', 'price_trend', 'cancellation_rate_*']
   
   # 节假日和季节性特征（4个）
   ['is_weekend', 'is_month_start', 'is_month_end', 'season']
   ```

**使用场景：**
- **训练阶段**：首次处理原始数据，保存预处理参数
- **推理阶段**：加载保存的预处理器，确保新数据使用相同的处理规则
- **模型更新**：保证新旧数据处理方式一致，避免数据漂移

**保存/加载方法：**
```python
# 保存预处理器
preprocessor.save_preprocessor('../02_训练模型/preprocessor.pkl')

# 加载预处理器
preprocessor = HotelDataPreprocessor.load_preprocessor('../02_训练模型/preprocessor.pkl')
```

**标准化器说明：**
- **NGBoost模型中的`feature_scaler`**：专门用于NGBoost模型特征的标准化，在模型训练时fit，在预测时transform
- **数据预处理器中的`scaler`**：用于整体数据预处理流程的特征标准化，处理更广泛的特征工程pipeline

**注:**  ```rl_system.py``` 中的在线学习,策略评估没有开启使用,为占位函数

