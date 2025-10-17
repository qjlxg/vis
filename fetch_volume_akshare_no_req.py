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

def get_daily_volume():
    data_list = []
    today = datetime.date.today().strftime('%Y%m%d')
    
    for name, code in indices.items():
        # 使用 akshare 获取指数日线数据（包含成交量）
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=today, end_date=today)
        if not df.empty:
            # 提取成交量（单位：手）
            volume = df['成交量'].iloc[0]
            data_list.append({
                '指数名称': name,
                '指数代码': code,
                '日期': df['日期'].iloc[0],
                '成交量': volume
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
