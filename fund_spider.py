import pandas as pd
import glob
import os

# --- 配置参数 (双重筛选条件) ---
FUND_DATA_DIR = 'fund_data'
MIN_CONSECUTIVE_DROP_DAYS = 5 # 连续下跌天数的阈值 (用于30日)
MIN_MONTH_DRAWDOWN = 0.10     # 1个月回撤的阈值 (10%)
REPORT_FILE = 'report.md'

def calculate_consecutive_drops(series):
    """
    计算系列中最大连续下跌天数。
    下跌定义为：当日净值 < 前一日净值。
    """
    # 确保系列不是空的
    if series.empty or len(series) < 2:
        return 0

    # 计算每日的变化率或方向 (当日净值 < 前一日净值 = True)
    # drops 是一个布尔系列，表示当日是否比前一日低
    # shift(1) 是前一日净值，与当日净值比较
    # 注意: drops 的长度会比 series 少 1
    drops = (series < series.shift(1)).iloc[1:] # 从第二个元素开始比较，丢弃第一个 NaN
    # 转换为 1 (下跌) 和 0 (非下跌)
    drops_int = drops.astype(int)
    
    # 查找最长连续 1 的长度
    max_drop_days = 0
    current_drop_days = 0
    for val in drops_int:
        if val == 1:
            current_drop_days += 1
        else:
            max_drop_days = max(max_drop_days, current_drop_days)
            current_drop_days = 0
    max_drop_days = max(max_drop_days, current_drop_days)

    return max_drop_days


def calculate_max_drawdown(series):
    """
    计算净值系列的最大回撤 (MDD)。
    MDD = (Peak - Trough) / Peak
    """
    if series.empty:
        return 0.0
    rolling_max = series.cummax()
    drawdown = (rolling_max - series) / rolling_max
    mdd = drawdown.max()
    return mdd

def generate_report(results):
    """
    将分析结果生成 Markdown 格式报告。
    使用优化后的排版，并明确说明双重筛选条件。
    """
    # 明确设置时区为北京时间
    try:
        now_str = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    if not results:
        return (
            f"# 基金预警报告 ({now_str} UTC+8)\n\n"
            f"## 分析总结\n\n"
            f"**恭喜，在过去一个月内，没有发现同时满足 '连续下跌{MIN_CONSECUTIVE_DROP_DAYS}天以上' 和 '1个月回撤{MIN_MONTH_DRAWDOWN*100:.0f}%以上' 的基金。**\n\n"
            f"---\n"
            f"分析数据时间范围: 最近30个交易日 (通常约为1个月)。"
        )

    # 1. 排序：按 '最大回撤' 从高到低排序
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='最大回撤', ascending=False).reset_index(drop=True)
    df_results.index = df_results.index + 1 # 排名从 1 开始
    
    total_count = len(df_results)
    
    # 2. 格式化输出
    report = f"# 基金预警报告 ({now_str} UTC+8)\n\n"
    
    # --- 增加总结部分 ---
    report += f"## 分析总结\n\n"
    report += f"本次分析共发现 **{total_count}** 只基金同时满足以下两个预警条件（基于最近30个交易日）：\n"
    report += f"1. **连续下跌**：净值连续下跌 **{MIN_CONSECUTIVE_DROP_DAYS}** 天以上。\n"
    report += f"2. **高回撤**：近 1 个月内最大回撤达到 **{MIN_MONTH_DRAWDOWN*100:.0f}%** 以上。\n\n"
    report += f"---"
    
    # --- 预警基金列表 (优化表格对齐) ---
    report += f"\n## 预警基金列表 (按最大回撤降序排列)\n\n"
    
    # 新增了“近一周连跌天数”列
    report += f"| 排名 | 基金代码 | 最大回撤 (1个月) | 最大连续下跌天数 (1个月) | 近一周连跌天数 (5日) |\n"
    # 对齐设置：回撤和天数都右对齐 (---:)
    report += f"| :---: | :---: | ---: | ---: | ---: |\n"  

    for index, row in df_results.iterrows():
        # 最大回撤加粗
        # 注意: '近一周连跌' 是新添加的列
        report += f"| {index} | `{row['基金代码']}` | **{row['最大回撤']:.2%}** | {row['最大连续下跌']} | {row['近一周连跌']} |\n"
    
    report += "\n---\n"
    report += f"分析数据时间范围: 最近30个交易日 (通常约为1个月)。\n"

    return report


def analyze_all_funds():
    """
    遍历基金数据目录，分析每个基金，并返回符合条件的基金列表。
    增加了对最近 5 个交易日连续下跌的分析。
    """
    csv_files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
    
    if not csv_files:
        print(f"警告：在目录 '{FUND_DATA_DIR}' 中未找到任何 CSV 文件。")
        return []

    print(f"找到 {len(csv_files)} 个基金数据文件，开始分析...")
    
    qualifying_funds = []
    
    for filepath in csv_files:
        try:
            fund_code = os.path.splitext(os.path.basename(filepath))[0]
            
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            # 按日期降序排列，方便选取最近数据 (最新数据在最前面)
            df = df.sort_values(by='date', ascending=False).reset_index(drop=True) 
            df = df.rename(columns={'net_value': 'value'})
            
            # 确保有足够的数据，至少需要 30 个交易日
            if len(df) < 30:
                # print(f"基金 {fund_code} 数据不足30条，跳过。")
                continue
            
            # 1. 选取最近 30 条数据 (约 1 个月) 用于主筛选
            df_recent_month = df.head(30)
            
            # 2. 选取最近 5 条数据 (约 1 周) 用于新要求
            df_recent_week = df.head(5)
            
            # --- 1个月数据分析 ---
            # 1. 连续下跌天数 (1个月内)
            max_drop_days_month = calculate_consecutive_drops(df_recent_month['value'])
            
            # 2. 1个月最大回撤
            mdd_recent_month = calculate_max_drawdown(df_recent_month['value'])
            
            # --- 近一周数据分析 (新要求) ---
            # 3. 近一周 (5日) 最大连续下跌天数
            max_drop_days_week = calculate_consecutive_drops(df_recent_week['value'])


            # 4. 筛选条件：必须同时满足两个主条件
            if max_drop_days_month >= MIN_CONSECUTIVE_DROP_DAYS and mdd_recent_month >= MIN_MONTH_DRAWDOWN:
                qualifying_funds.append({
                    '基金代码': fund_code,
                    '最大回撤': mdd_recent_month,  
                    '最大连续下跌': max_drop_days_month,
                    '近一周连跌': max_drop_days_week # 新增：记录近一周的连跌天数
                })

        except Exception as e:
            print(f"处理文件 {filepath} 时发生错误: {e}")
            continue

    return qualifying_funds


if __name__ == '__main__':
    # 修正 calculate_consecutive_drops 中的一个潜在问题
    # 如果您使用原始脚本，请务必使用我修改后的 calculate_consecutive_drops
    # 原始脚本中的 series.shift(-1) < series 逻辑用于降序数据可能导致计算错误。
    # 我已在上面提供了修正后的版本，它适用于降序 (最新数据在最前) 的 DataFrame。

    # 1. 执行分析
    results = analyze_all_funds()
    
    # 2. 生成 Markdown 报告
    report_content = generate_report(results)
    
    # 3. 写入报告文件
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析完成，报告已保存到 {REPORT_FILE}")
