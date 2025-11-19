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
- `--max-workers`：最大并行工作进程数（默认：CPU核心数的一半）
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

**注:**  ```rl_system.py``` 中的在线学习,策略评估没有开启使用,为占位函数

