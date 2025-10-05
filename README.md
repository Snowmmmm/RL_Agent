### 1. 环境准备
运行环境 python3.12
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
# 只训练RL模型,不训练BNN模型
python main.py
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

## 📁 项目结构

```
RL_Agent/
├── 01_核心代码/               # 核心代码目录
│   ├── main.py               # 主程序入口
│   ├── config.py             # 配置文件
│   ├── bnn_model.py          # 贝叶斯神经网络
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
  --force-retrain          强制重新训练所有模型
  --help                   显示帮助信息
```

**注:**  ```rl_system.py``` 中的在线学习,策略评估没有开启使用,为占位函数
