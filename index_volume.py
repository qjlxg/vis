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
    """获取指定指数的历史成交量和价格数据"""
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date)
    if not df.empty:
        return df[['日期', '收盘', '成交量']], df['成交量'].mean(), df['收盘'].mean()  # 返回日期、收盘价、成交量，以及均值
    return None, None, None

def get_latest_trading_day_data(code, today, max_backtrack=3):
    """尝试获取最近交易日的数据，最多回溯 max_backtrack 天"""
    for i in range(max_backtrack + 1):
        check_date = today - datetime.timedelta(days=i)
        check_date_str = check_date.strftime('%Y%m%d')
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=check_date_str, end_date=check_date_str)
        if not df.empty:
            return df, check_date
    return None, None

def get_previous_trading_day_data(code, latest_date, max_backtrack=3):
    """获取前一交易日的数据，基于最近交易日"""
    for i in range(1, max_backtrack + 1):
        check_date = latest_date - datetime.timedelta(days=i)
        check_date_str = check_date.strftime('%Y%m%d')
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=check_date_str, end_date=check_date_str)
        if not df.empty:
            return df['成交量'].iloc[0], df['收盘'].iloc[0]
    return None, None

def analyze_volume_price(price_change_pct, volume_change_pct, hist_price_change_pct):
    """简单量价关系分析，并给出市场情绪参考"""
    if price_change_pct > 0 and volume_change_pct > 0:
        vp_relation = '量增价涨'
        sentiment = '强势，建议持仓或小幅加仓'
    elif price_change_pct > 0 and volume_change_pct < 0:
        vp_relation = '量缩价涨'
        sentiment = '警惕背离，建议观望'
    elif price_change_pct < 0 and volume_change_pct > 0:
        vp_relation = '量增价跌'
        sentiment = '弱势，建议减仓'
    elif price_change_pct < 0 and volume_change_pct < 0:
        vp_relation = '量缩价跌'
        sentiment = '可能筑底，建议补仓'
    else:
        vp_relation = '量价平稳'
        sentiment = '中性，持仓观望'
    
    # 考虑7日均价对比加强建议（阈值可调整）
    if hist_price_change_pct < -5:  # 价格低于7日均价超5%
        sentiment += '（价格低估，中期补仓机会）'
    elif hist_price_change_pct > 5:  # 价格高于7日均价超5%
        sentiment += '（价格高估，谨慎加仓）'
    
    return vp_relation, sentiment

def save_history_to_file(code, name, df, year_month, timestamp):
    """保存最近30天的成交量和价格数据到CSV"""
    if df is None or df.empty:
        print(f"{name} ({code}) 无历史数据可保存。")
        return
    
    filename = f"index_volume_history_{code}_{timestamp}.csv"
    filepath = os.path.join(year_month, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"历史数据保存到: {filepath}")

def get_daily_volume():
    data_list = []
    today = datetime.date.today()
    today_str = today.strftime('%Y%m%d')
    # 计算过去30天的日期范围（包括最近交易日，用于历史数据）
    past_30_days_start = (today - datetime.timedelta(days=30)).strftime('%Y%m%d')
    past_7_days_end = (today - datetime.timedelta(days=1)).strftime('%Y%m%d')
    past_7_days_start = (today - datetime.timedelta(days=8)).strftime('%Y%m%d')

    # 用于保存文件的目录和时间戳
    year_month = today.strftime('%Y%m')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for name, code in indices.items():
        # 尝试获取最近交易日数据
        df_daily, latest_date = get_latest_trading_day_data(code, today)
        if df_daily is not None and not df_daily.empty:
            volume = df_daily['成交量'].iloc[0]
            close_price = df_daily['收盘'].iloc[0]
            # 获取过去7天平均成交量和平均价格（基于最新交易日前的7天）
            past_7_days_end = (latest_date - datetime.timedelta(days=1)).strftime('%Y%m%d')
            past_7_days_start = (latest_date - datetime.timedelta(days=8)).strftime('%Y%m%d')
            _, hist_volume, hist_price = get_historical_volume(code, past_7_days_start, past_7_days_end)
            # 获取前一交易日成交量和收盘价
            prev_volume, prev_price = get_previous_trading_day_data(code, latest_date)
            if hist_volume:
                volume_change_pct = ((volume - hist_volume) / hist_volume * 100) if hist_volume != 0 else 0
            else:
                volume_change_pct = None
            if hist_price:
                hist_price_change_pct = ((close_price - hist_price) / hist_price * 100) if hist_price != 0 else 0
            else:
                hist_price_change_pct = None
            if prev_volume:
                prev_volume_change_pct = ((volume - prev_volume) / prev_volume * 100) if prev_volume != 0 else 0
            else:
                prev_volume_change_pct = None
            if prev_price:
                price_change_pct = ((close_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
            else:
                price_change_pct = None
            # 量价关系分析和市场情绪参考
            vp_relation, sentiment = analyze_volume_price(price_change_pct, prev_volume_change_pct, hist_price_change_pct) if price_change_pct is not None and prev_volume_change_pct is not None and hist_price_change_pct is not None else (None, None)
            data_list.append({
                '指数名称': name,
                '指数代码': code,
                '日期': df_daily['日期'].iloc[0],
                '收盘价': close_price,
                '成交量': volume,
                '历史7天均价': hist_price,
                '历史均值对比(价格%)': hist_price_change_pct,
                '历史7天均值(成交量)': hist_volume,
                '历史均值对比(成交量%)': volume_change_pct,
                '前日对比(成交量%)': prev_volume_change_pct,
                '前日对比(价格%)': price_change_pct,
                '量价关系': vp_relation,
                '市场情绪参考': sentiment
            })

            # 获取并保存最近30天历史数据（包括收盘价和成交量）
            df_hist, _, _ = get_historical_volume(code, past_30_days_start, latest_date.strftime('%Y%m%d'))
            save_history_to_file(code, name, df_hist, year_month, timestamp)
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
    
    # 保存每日成交量和价格分析CSV文件
    filename = f"index_volume_price_{timestamp}.csv"
    filepath = os.path.join(year_month, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据保存到: {filepath}")

if __name__ == "__main__":
    df, year_month, timestamp = get_daily_volume()
    save_to_file(df, year_month, timestamp)
