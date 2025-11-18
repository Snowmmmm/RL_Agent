import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_max_inventory_with_second(hotel_type='City Hotel'):
    """
    分析最大库存数、第二大库存数及对应日期，含详细分布说明
    """
    print(f"正在分析{hotel_type}库存峰值（含第二大）...")
    
    # ----------------------
    # 1. 数据处理与库存计算（复用核心逻辑）
    # ----------------------
    try:
        df = pd.read_csv('../03_数据文件/hotel_bookings.csv')
    except FileNotFoundError:
        print("错误：未找到数据文件，请检查路径")
        return None
    
    hotel_df = df[df['hotel'] == hotel_type].copy()
    print(f"{hotel_type}总订单数（含取消）: {len(hotel_df)}")
    
    # 日期处理
    hotel_df['arrival_date'] = pd.to_datetime(
        hotel_df['arrival_date_year'].astype(str) + '-' + 
        hotel_df['arrival_date_month'] + '-' + 
        hotel_df['arrival_date_day_of_month'].astype(str)
    )
    hotel_df['total_nights'] = hotel_df['stays_in_weekend_nights'] + hotel_df['stays_in_week_nights']
    hotel_df['departure_date'] = hotel_df['arrival_date'] + pd.to_timedelta(hotel_df['total_nights'], unit='D')
    hotel_df['booking_date'] = hotel_df['arrival_date'] - pd.to_timedelta(hotel_df['lead_time'], unit='D')
    hotel_df['reservation_status_date'] = pd.to_datetime(hotel_df['reservation_status_date'])
    hotel_df['cancel_date'] = np.where(
        hotel_df['is_canceled'] == 1,
        hotel_df['reservation_status_date'],
        hotel_df['departure_date']
    )
    
    # 过滤无效订单
    valid_mask = (
        (hotel_df['total_nights'] >= 1) &
        (hotel_df['arrival_date'] < hotel_df['departure_date']) &
        (hotel_df['booking_date'] <= hotel_df['cancel_date'])
    )
    hotel_df = hotel_df[valid_mask].copy()
    print(f"过滤后有效订单数: {len(hotel_df)}")
    
    # 拆分连续入住订单
    split_orders = []
    for idx, order in hotel_df.iterrows():
        stay_dates = pd.date_range(order['arrival_date'], order['departure_date'] - timedelta(days=1), freq='D')
        for stay_date in stay_dates:
            split_orders.append({
                'order_id': idx,
                'stay_date': stay_date,
                'booking_date': order['booking_date'],
                'cancel_date': order['cancel_date']
            })
    split_df = pd.DataFrame(split_orders)
    print(f"拆分后单日订单数: {len(split_df)}")
    
    # ----------------------
    # 2. 计算所有日期的最大库存数
    # ----------------------
    all_stay_dates = sorted(split_df['stay_date'].unique())
    stay_date_max = {}  # 键：入住日期（date），值：该日期的最大库存数
    for target_stay_date in all_stay_dates:
        target_orders = split_df[split_df['stay_date'] == target_stay_date].copy()
        if len(target_orders) == 0:
            stay_date_max[target_stay_date.date()] = 0
            continue
        
        min_track = target_orders['booking_date'].min()
        max_track = target_stay_date
        track_dates = pd.date_range(min_track, max_track, freq='D')
        target_orders = target_orders.sort_values('booking_date')
        
        current = 0
        max_cnt = 0
        for date in track_dates:
            current += len(target_orders[target_orders['booking_date'] == date])
            current -= len(target_orders[target_orders['cancel_date'] == date])
            current = max(current, 0)
            max_cnt = max(max_cnt, current)
        stay_date_max[target_stay_date.date()] = max_cnt
    
    # ----------------------
    # 3. 提取最大、第二大库存数及对应日期
    # ----------------------
    # 所有库存值（去重并排序）
    all_values = sorted(list(set(stay_date_max.values())), reverse=True)
    if len(all_values) < 2:
        print("警告：数据中库存值种类不足2种，无法计算第二大")
        return None
    
    # 最大库存
    max1 = all_values[0]
    max1_dates = sorted([d for d, v in stay_date_max.items() if v == max1])
    max1_days = len(max1_dates)
    
    # 第二大库存（排除最大后的值）
    max2 = all_values[1]
    max2_dates = sorted([d for d, v in stay_date_max.items() if v == max2])
    max2_days = len(max2_dates)
    
    # ----------------------
    # 4. 结果输出
    # ----------------------
    print(f"\n=== {hotel_type}库存峰值分析 ===")
    print(f"1. 最大库存数: {max1} 间")
    print(f"   - 出现天数: {max1_days} 天")
    print(f"   - 具体日期: {[str(d) for d in max1_dates]}")
    
    print(f"\n2. 第二大库存数: {max2} 间")
    print(f"   - 与最大库存的差距: {max1 - max2} 间")
    print(f"   - 出现天数: {max2_days} 天")
    print(f"   - 具体日期（前5个，共{max2_days}天）: {[str(d) for d in max2_dates[:5]]}")
    
    # 最大库存仅1天的可能原因
    if max1_days == 1:
        print(f"\n=== 为什么最大库存仅在1天出现？ ===")
        print(f"可能原因：")
        print(f"1. {max1_dates[0]}当天可能有大型活动（如展会、节日），导致预订量骤增；")
        print(f"2. 该日期处于旅游旺季顶峰，其他日期预订量未超过此值；")
        print(f"3. 数据时间范围有限，仅出现一次极端峰值。")
    
    # ----------------------
    # 5. 读取并处理actual_demand数据
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
        
        # 创建日期到actual_demand的映射
        actual_demand_map = {row['date'].date(): row['total_actual_demand'] for _, row in merged_df.iterrows()}
        print(f"成功读取actual_demand数据，日期范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
    except Exception as e:
        print(f"读取actual_demand数据时出错: {e}")
        actual_demand_map = {}
    
    # ----------------------
    # 6. 可视化（标记最大和第二大，添加actual_demand）
    # ----------------------
    plt.figure(figsize=(15, 8))
    sorted_dates = sorted(stay_date_max.keys())
    sorted_values = [stay_date_max[d] for d in sorted_dates]
    
    # 绘制所有日期的库存曲线
    plt.plot(sorted_dates, sorted_values, 'b-', alpha=0.5, linewidth=1, label='每日最大库存')
    
    # 绘制actual_demand曲线（如果有数据）
    if actual_demand_map:
        actual_dates = sorted(actual_demand_map.keys())
        actual_values = [actual_demand_map.get(d, 0) for d in actual_dates]
        plt.plot(actual_dates, actual_values, 'g-', alpha=0.7, linewidth=2, label='实际需求 (线上+线下)')
    
    # 标记最大库存日期
    plt.scatter(max1_dates, [max1]*max1_days, color='red', s=80, zorder=3, label=f'最大库存 ({max1}间)')
    # 标记第二大库存日期
    plt.scatter(max2_dates, [max2]*max2_days, color='orange', s=50, zorder=3, label=f'第二大库存 ({max2}间)')
    # 参考线
    plt.axhline(y=max1, color='r', linestyle='--', linewidth=1)
    plt.axhline(y=max2, color='orange', linestyle='--', linewidth=1)
    
    plt.title(f'{hotel_type}库存峰值分布（含最大和第二大）与实际需求对比', fontsize=14)
    plt.xlabel('入住日期')
    plt.ylabel('房间数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{hotel_type}_max2_inventory_with_actual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存结果
    import json
    result_data = {
        'max1': {'value': max1, 'days': max1_days, 'dates': [str(d) for d in max1_dates]},
        'max2': {'value': max2, 'days': max2_days, 'dates': [str(d) for d in max2_dates]},
        'gap': max1 - max2
    }
    
    # 添加actual_demand统计信息
    if actual_demand_map:
        actual_values = list(actual_demand_map.values())
        result_data['actual_demand_stats'] = {
            'mean': sum(actual_values) / len(actual_values),
            'max': max(actual_values),
            'min': min(actual_values)
        }
    
    with open(f'{hotel_type}_max2_inventory_with_actual_details.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    return {'max1': max1, 'max2': max2, 'max1_dates': max1_dates, 'max2_dates': max2_dates}

if __name__ == "__main__":
    result = analyze_max_inventory_with_second(hotel_type='City Hotel')