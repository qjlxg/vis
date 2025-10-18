import os
import datetime
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 定义指数列表
indices = {
    '上证指数': '000001',
    '深成指数': '399001',
    '创业板指': '399006',
    '沪深300': '000300'
}

def get_historical_volume(code, start_date, end_date):
    """获取指定指数的历史成交量数据"""
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date)
    if not df.empty:
        return df[['日期', '成交量']], df['成交量'].mean()  # 返回日期和成交量，以及均值
    return None, None

def get_latest_trading_day_data(code, today, max_backtrack=3):
    """尝试获取最近交易日的数据，最多回溯 max_backtrack 天"""
    for i in range(max_backtrack + 1):
        check_date = today - datetime.timedelta(days=i)
        check_date_str = check_date.strftime('%Y%m%d')
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=check_date_str, end_date=check_date_str)
        if not df.empty:
            return df, check_date
    return None, None

def plot_volume_chart(code, name, df, year_month, timestamp):
    """生成并保存成交量曲线图"""
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(df['日期']), df['成交量'], marker='o', linestyle='-', color='#1f77b4', label=name)
    plt.title(f'{name} ({code}) 每日成交量', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('成交量 (手)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    filename = f"index_volume_chart_{code}_{timestamp}.png"
    filepath = os.path.join(year_month, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"图表保存到: {filepath}")

def get_daily_volume():
    data_list = []
    today = datetime.date.today()
    today_str = today.strftime('%Y%m%d')
    # 计算过去30天的日期范围（包括最近交易日，用于曲线图）
    past_30_days_start = (today - datetime.timedelta(days=30)).strftime('%Y%m%d')
    past_7_days_end = (today - datetime.timedelta(days=1)).strftime('%Y%m%d')
    past_7_days_start = (today - datetime.timedelta(days=8)).strftime('%Y%m%d')

    # 用于保存历史数据以生成图表
    year_month = today.strftime('%Y%m')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for name, code in indices.items():
        # 尝试获取最近交易日数据
        df_daily, latest_date = get_latest_trading_day_data(code, today)
        if df_daily is not None and not df_daily.empty:
            volume = df_daily['成交量'].iloc[0]
            # 获取过去7天平均成交量（基于最新交易日前的7天）
            past_7_days_end = (latest_date - datetime.timedelta(days=1)).strftime('%Y%m%d')
            past_7_days_start = (latest_date - datetime.timedelta(days=8)).strftime('%Y%m%d')
            _, hist_volume = get_historical_volume(code, past_7_days_start, past_7_days_end)
            if hist_volume:
                volume_change_pct = ((volume - hist_volume) / hist_volume * 100) if hist_volume != 0 else 0
            else:
                volume_change_pct = None
            data_list.append({
                '指数名称': name,
                '指数代码': code,
                '日期': df_daily['日期'].iloc[0],
                '成交量': volume,
                '历史7天均值': hist_volume,
                '历史均值对比(%)': volume_change_pct
            })

            # 获取最近30天数据并生成曲线图
            df_hist, _ = get_historical_volume(code, past_30_days_start, latest_date.strftime('%Y%m%d'))
            if df_hist is not None and not df_hist.empty:
                plot_volume_chart(code, name, df_hist, year_month, timestamp)
        else:
            print(f"{name} ({code}) 无最近交易日数据，可能连续非交易日。")
    
    if data_list:
        return pd.DataFrame(data_list), year_month, timestamp
    else:
        return None, None, None

def save_to_file(df, year_month, timestamp):
    if df is None:
        print("无数据可保存。")
        return
    
    # 创建目录如果不存在
    os.makedirs(year_month, exist_ok=True)
    
    # 保存CSV文件
    filename = f"index_volume_{timestamp}.csv"
    filepath = os.path.join(year_month, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据保存到: {filepath}")

if __name__ == "__main__":
    df, year_month, timestamp = get_daily_volume()
    save_to_file(df, year_month, timestamp)
