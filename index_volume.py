import os
import datetime
import akshare as ak
import pandas as pd
import pandas_ta as ta

# 定义指数列表
indices = {
    '上证指数': '000001',
    '深成指数': '399001',
    '创业板指': '399006',
    '沪深300': '000300'
}

# --- 辅助函数 ---

def get_historical_volume(code, start_date, end_date):
    """获取指定指数的历史成交量和价格数据，并计算 RSI"""
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date)
    if not df.empty:
        # 计算14天 RSI（需要至少15天数据）
        if len(df) >= 15:
            df['RSI_14'] = ta.rsi(df['收盘'], length=14)
        else:
            df['RSI_14'] = None  # 数据不足，无法计算 RSI
        return df[['日期', '收盘', '成交量', 'RSI_14']], df['成交量'].mean(), df['收盘'].mean()
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

def analyze_volume_price(price_change_pct, volume_hist_change_pct, price_hist_change_pct, rsi):
    """量价关系分析，结合 RSI 给出市场情绪参考"""
    vol_up = volume_hist_change_pct > 10
    vol_down = volume_hist_change_pct < -10
    price_up = price_change_pct > 0
    price_down = price_change_pct < 0
    rsi_overbought = rsi > 70 if rsi is not None else False
    rsi_oversold = rsi < 30 if rsi is not None else False

    vp_relation = '量价平稳'
    sentiment = '中性，持仓观望'

    if price_up and vol_up:
        vp_relation = '量增价涨'
        sentiment = '强势，趋势健康'
    elif price_up and vol_down:
        vp_relation = '量缩价涨'
        sentiment = '警惕背离，上涨动能不足'
    elif price_down and vol_up:
        vp_relation = '量增价跌'
        sentiment = '恐慌抛售或主力出货，弱势'
    elif price_down and vol_down:
        vp_relation = '量缩价跌'
        sentiment = '抛压减轻，可能筑底'

    if rsi_overbought:
        sentiment += '（RSI 超买，谨慎加仓）'
    elif rsi_oversold:
        sentiment += '（RSI 超卖，补仓机会）'
    if price_hist_change_pct < -3:
        sentiment += '（价格低估，中期补仓机会）'
    elif price_hist_change_pct > 3:
        sentiment += '（价格高估，谨慎加仓）'

    return vp_relation, sentiment, rsi

def save_history_to_file(code, name, df, year_month, timestamp):
    """保存最近30天的成交量、价格和 RSI 数据到CSV"""
    if df is None or df.empty:
        print(f"{name} ({code}) 无历史数据可保存。")
        return
    
    filename = f"index_volume_history_{code}_{timestamp}.csv"
    filepath = os.path.join(year_month, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"历史数据保存到: {filepath}")

# --- 主逻辑函数 ---

def get_daily_volume():
    data_list = []
    today = datetime.date.today()
    
    # 计算过去45天的日期范围（覆盖30天历史、15天 RSI、7天均值）
    past_45_days_start = (today - datetime.timedelta(days=45)).strftime('%Y%m%d')
    year_month = today.strftime('%Y%m')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for name, code in indices.items():
        # 尝试获取最近交易日数据
        df_daily, latest_date = get_latest_trading_day_data(code, today)
        
        if df_daily is not None and not df_daily.empty:
            volume = df_daily['成交量'].iloc[0]
            close_price = df_daily['收盘'].iloc[0]
            
            # 获取45天数据，覆盖所有需求
            latest_date_str = latest_date.strftime('%Y%m%d')
            df_hist, hist_volume, hist_price = get_historical_volume(code, past_45_days_start, latest_date_str)
            
            if df_hist is not None and not df_hist.empty:
                # 提取7天均值数据（最近7天，基于最新交易日前）
                past_7_days_end = latest_date - datetime.timedelta(days=1)
                past_7_days_start = latest_date - datetime.timedelta(days=8)
                df_7_days = df_hist[(df_hist['日期'] >= past_7_days_start.strftime('%Y-%m-%d')) & 
                                  (df_hist['日期'] <= past_7_days_end.strftime('%Y-%m-%d'))]
                hist_volume = df_7_days['成交量'].mean() if not df_7_days.empty else None
                hist_price = df_7_days['收盘'].mean() if not df_7_days.empty else None
                
                # 提取30天历史数据
                past_30_days_start = (latest_date - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                df_30_days = df_hist[df_hist['日期'] >= past_30_days_start]
                
                # 提取 RSI（最新值）
                rsi = df_hist['RSI_14'].iloc[-1] if 'RSI_14' in df_hist.columns and df_hist['RSI_14'].notnull().any() else None
                
                # 获取前一交易日数据
                prev_volume, prev_price = get_previous_trading_day_data(code, latest_date)
                
                # 计算百分比指标
                volume_hist_change_pct = ((volume - hist_volume) / hist_volume * 100) if hist_volume else None
                price_hist_change_pct = ((close_price - hist_price) / hist_price * 100) if hist_price else None
                volume_prev_change_pct = ((volume - prev_volume) / prev_volume * 100) if prev_volume else None
                price_prev_change_pct = ((close_price - prev_price) / prev_price * 100) if prev_price else None
                
                # 量价关系分析和市场情绪参考
                if all(v is not None for v in [price_prev_change_pct, volume_hist_change_pct, price_hist_change_pct, rsi]):
                    vp_relation, sentiment, rsi_value = analyze_volume_price(
                        price_prev_change_pct, 
                        volume_hist_change_pct, 
                        price_hist_change_pct,
                        rsi
                    )
                else:
                    vp_relation, sentiment, rsi_value = None, "数据不足，无法分析", None

                # 整理数据
                data_list.append({
                    '指数名称': name,
                    '指数代码': code,
                    '日期': df_daily['日期'].iloc[0],
                    '收盘价': close_price,
                    '前日对比(价格%)': price_prev_change_pct,
                    '成交量': volume,
                    '历史7天均值(成交量)': hist_volume,
                    '历史均值对比(成交量%)': volume_hist_change_pct,
                    '前日对比(成交量%)': volume_prev_change_pct,
                    '历史7天均价': hist_price,
                    '历史均值对比(价格%)': price_hist_change_pct,
                    'RSI_14': rsi_value,
                    '量价关系': vp_relation,
                    '市场情绪参考': sentiment
                })

                # 保存30天历史数据
                save_history_to_file(code, name, df_30_days, year_month, timestamp)
            else:
                print(f"{name} ({code}) 无历史数据，可能数据接口问题。")
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
