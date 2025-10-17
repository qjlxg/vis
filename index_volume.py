import os
import datetime
import akshare as ak
import pandas as pd

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
        return df['成交量'].mean()  # 返回平均成交量
    return None

def get_daily_volume():
    data_list = []
    today = datetime.date.today()
    today_str = today.strftime('%Y%m%d')
    # 计算过去7天的日期范围（不包括今天）
    past_7_days_end = (today - datetime.timedelta(days=1)).strftime('%Y%m%d')
    past_7_days_start = (today - datetime.timedelta(days=8)).strftime('%Y%m%d')

    for name, code in indices.items():
        # 获取当日数据
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=today_str, end_date=today_str)
        if not df.empty:
            volume = df['成交量'].iloc[0]
            # 获取过去7天平均成交量
            hist_volume = get_historical_volume(code, past_7_days_start, past_7_days_end)
            if hist_volume:
                # 计算与历史均值的百分比变化
                volume_change_pct = ((volume - hist_volume) / hist_volume * 100) if hist_volume != 0 else 0
            else:
                volume_change_pct = None  # 无历史数据
            data_list.append({
                '指数名称': name,
                '指数代码': code,
                '日期': df['日期'].iloc[0],
                '成交量': volume,
                '历史7天均值': hist_volume,
                '历史均值对比(%)': volume_change_pct
            })
        else:
            print(f"{name} ({code}) 无当日数据，可能非交易日。")
    
    if data_list:
        return pd.DataFrame(data_list)
    else:
        return None

def save_to_file(df):
    if df is None:
        print("无数据可保存。")
        return
    
    now = datetime.datetime.now()
    year_month = now.strftime('%Y/%m')
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    filename = f"index_volume_{timestamp}.csv"
    
    # 创建目录如果不存在
    os.makedirs(year_month, exist_ok=True)
    
    filepath = os.path.join(year_month, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据保存到: {filepath}")

if __name__ == "__main__":
    df = get_daily_volume()
    save_to_file(df)
