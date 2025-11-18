import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

def analyze_actual_demand():
    """
    分析实际需求（线上线下相加）的统计特征，包括最大值和95%分位值
    """
    print("正在分析实际需求（线上线下相加）...")
    
    # ----------------------
    # 1. 读取并处理actual_demand数据
    # ----------------------
    try:
        # 读取线上和线下数据
        online_df = pd.read_csv('../03_数据文件/online_features.csv')
        offline_df = pd.read_csv('../03_数据文件/offline_features.csv')
        
        # 转换日期格式
        online_df['date'] = pd.to_datetime(online_df['date'])
        offline_df['date'] = pd.to_datetime(offline_df['date'])
        
        # 合并线上线下actual_demand
        merged_df = pd.merge(online_df[['date', 'actual_demand']], 
                           offline_df[['date', 'actual_demand']], 
                           on='date', suffixes=('_online', '_offline'))
        merged_df['total_actual_demand'] = merged_df['actual_demand_online'] + merged_df['actual_demand_offline']
        
        print(f"成功读取actual_demand数据，日期范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
        print(f"总天数: {len(merged_df)}")
    except Exception as e:
        print(f"读取actual_demand数据时出错: {e}")
        return None
    
    # ----------------------
    # 2. 计算统计指标
    # ----------------------
    total_demand = merged_df['total_actual_demand']
    
    # 计算基本统计量
    mean_demand = total_demand.mean()
    max_demand = total_demand.max()
    min_demand = total_demand.min()
    std_demand = total_demand.std()
    
    # 计算分位数
    p25 = total_demand.quantile(0.25)
    p50 = total_demand.quantile(0.5)  # 中位数
    p75 = total_demand.quantile(0.75)
    p90 = total_demand.quantile(0.90)
    p95 = total_demand.quantile(0.95)
    p99 = total_demand.quantile(0.99)
    
    # 找出最大值和95%分位值对应的日期
    max_date = merged_df.loc[total_demand.idxmax(), 'date']
    p95_date = merged_df[total_demand >= p95]['date'].min()  # 第一个达到95%分位值的日期
    
    print("\n=== 实际需求统计结果 ===")
    print(f"平均值: {mean_demand:.2f}")
    print(f"标准差: {std_demand:.2f}")
    print(f"最小值: {min_demand:.2f}")
    print(f"25%分位数: {p25:.2f}")
    print(f"50%分位数(中位数): {p50:.2f}")
    print(f"75%分位数: {p75:.2f}")
    print(f"90%分位数: {p90:.2f}")
    print(f"95%分位数: {p95:.2f} (首次出现日期: {p95_date.strftime('%Y-%m-%d')})")
    print(f"99%分位数: {p99:.2f}")
    print(f"最大值: {max_demand:.2f} (日期: {max_date.strftime('%Y-%m-%d')})")
    
    # ----------------------
    # 3. 可视化
    # ----------------------
    plt.figure(figsize=(15, 8))
    
    # 绘制实际需求曲线
    plt.plot(merged_df['date'], merged_df['total_actual_demand'], 'b-', linewidth=1.5, label='实际需求 (线上+线下)')
    
    # 添加统计线
    plt.axhline(y=mean_demand, color='green', linestyle='--', linewidth=1, label=f'平均值 ({mean_demand:.2f})')
    plt.axhline(y=p95, color='orange', linestyle='--', linewidth=1, label=f'95%分位值 ({p95:.2f})')
    plt.axhline(y=max_demand, color='red', linestyle='--', linewidth=1, label=f'最大值 ({max_demand:.2f})')
    
    # 标记最大值和95%分位值点
    plt.scatter([max_date], [max_demand], color='red', s=80, zorder=3)
    plt.scatter([p95_date], [p95], color='orange', s=80, zorder=3)
    
    # 设置标题和标签
    plt.title('实际需求（线上线下相加）时间序列与统计特征', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('实际需求量')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('实际需求_线上线下相加_统计分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ----------------------
    # 4. 保存结果
    # ----------------------
    import json
    result_data = {
        'statistics': {
            'mean': float(mean_demand),
            'std': float(std_demand),
            'min': float(min_demand),
            'max': float(max_demand),
            'max_date': max_date.strftime('%Y-%m-%d'),
            'p25': float(p25),
            'p50': float(p50),
            'p75': float(p75),
            'p90': float(p90),
            'p95': float(p95),
            'p95_date': p95_date.strftime('%Y-%m-%d'),
            'p99': float(p99)
        },
        'data_points': len(merged_df),
        'date_range': {
            'start': merged_df['date'].min().strftime('%Y-%m-%d'),
            'end': merged_df['date'].max().strftime('%Y-%m-%d')
        }
    }
    
    with open('实际需求_统计分析结果.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存至: 实际需求_统计分析结果.json")
    print("可视化图表已保存为: 实际需求_线上线下相加_统计分析.png")
    
    return result_data

if __name__ == "__main__":
    analyze_actual_demand()