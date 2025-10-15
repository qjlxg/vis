import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# 加载配置文件
try:
    with open('holdings_config.json', 'r', encoding='utf-8') as f:
        holdings_config = json.load(f)
except FileNotFoundError:
    print("错误: holdings_config.json 未找到，请确保文件存在。")
    # 使用备用配置，防止脚本崩溃
    holdings_config = {
        "parameters": {
            "rsi_window": 14, "ma_window": 50, "bb_window": 20, "rsi_overbought_threshold": 80,
            "consecutive_days_threshold": 3, "profit_lock_days": 14, "volatility_window": 7,
            "volatility_threshold": 0.03, "decline_days_threshold": 5, "trailing_stop_loss_pct": 0.08,
            "macd_divergence_window": 60, "adx_window": 14, "adx_threshold": 30
        },
        "017423": 1.0580 # 示例基金
    }
except json.JSONDecodeError:
    print("错误: holdings_config.json 格式错误，无法解析。")
    exit()

# 获取可配置参数
params = holdings_config.get('parameters', {})
rsi_window = params.get('rsi_window', 14)
ma_window = params.get('ma_window', 50)
bb_window = params.get('bb_window', 20)
rsi_overbought_threshold = params.get('rsi_overbought_threshold', 80)
consecutive_days_threshold = params.get('consecutive_days_threshold', 3)
profit_lock_days = params.get('profit_lock_days', 14)
volatility_window = params.get('volatility_window', 7)
volatility_threshold = params.get('volatility_threshold', 0.03)
decline_days_threshold = params.get('decline_days_threshold', 5)
trailing_stop_loss_pct = params.get('trailing_stop_loss_pct', 0.08)
macd_divergence_window = params.get('macd_divergence_window', 60)
adx_window = params.get('adx_window', 14)
adx_threshold = params.get('adx_threshold', 30)

# 数据路径
big_market_path = 'index_data/000300.csv'
fund_data_dir = 'fund_data/'

# 加载大盘数据
if os.path.exists(big_market_path):
    big_market = pd.read_csv(big_market_path, parse_dates=['date'])
    big_market = big_market.sort_values('date').reset_index(drop=True)
    if big_market['net_value'].max() > 10 or big_market['net_value'].min() < 0:
        print("警告: 大盘数据净值异常，请检查 index_data/000300.csv")
else:
    raise FileNotFoundError("大盘数据文件 index_data/000300.csv 未找到")

# 计算ADX指标（已加入除零保护）
def calculate_adx(df, window):
    df = df.copy()
    high = df['net_value']
    low = df['net_value']
    close = df['net_value']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['+dm'] = (high - high.shift(1)).apply(lambda x: x if x > 0 else 0)
    df['-dm'] = (low.shift(1) - low).apply(lambda x: x if x > 0 else 0)
    # 修正：将 +DM/-DM 仅保留正向的逻辑移除，交给 ATR/PDI 计算
    atr = df['tr'].ewm(span=window, adjust=False, min_periods=window).mean()
    pdm = df['+dm'].ewm(span=window, adjust=False, min_periods=window).mean()
    mdm = df['-dm'].ewm(span=window, adjust=False, min_periods=window).mean()
    
    # 除零保护
    pdi = np.where(atr != 0, (pdm / atr) * 100, 0)
    mdi = np.where(atr != 0, (mdm / atr) * 100, 0)
    
    pdi_plus_mdi = pdi + mdi
    # 除零保护
    dx = np.where(pdi_plus_mdi != 0, (abs(pdi - mdi) / pdi_plus_mdi) * 100, 0)
    
    df['adx'] = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean()
    return df['adx']

# 计算指标（已加入趋势延续计数和更健壮的RSI计算）
def calculate_indicators(df, rsi_win, ma_win, bb_win, adx_win):
    df = df.copy()
    delta = df['net_value'].diff()
    # 使用ewm平滑RSI计算
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    avg_up = up.ewm(com=rsi_win - 1, adjust=False, min_periods=rsi_win).mean()
    avg_down = down.ewm(com=rsi_win - 1, adjust=False, min_periods=rsi_win).mean()
    # 避免除零，用 1e-10 替代 0
    rs = avg_up / avg_down.replace(0, np.nan).fillna(1e-10) 
    df['rsi'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(100)))

    df['ma50'] = df['net_value'].rolling(window=ma_win, min_periods=1).mean()
    exp12 = df['net_value'].ewm(span=12, adjust=False).mean()
    exp26 = df['net_value'].ewm(span=26, adjust=False).mean()
    df['macd'] = 2 * (exp12 - exp26) 
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['bb_mid'] = df['net_value'].rolling(window=bb_win, min_periods=1).mean()
    df['bb_std'] = df['net_value'].rolling(window=bb_win, min_periods=1).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['daily_return'] = df['net_value'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=bb_win).std()
    
    # 新增：布林带上轨突破标志
    df['bb_break_upper'] = df['net_value'] > df['bb_upper']
    
    # 新增：趋势延续计数 (MACD 持续同向柱状体计数，用于加强信号)
    df['hist'] = df['macd'] - df['signal']
    df['trend_days'] = 0
    
    if len(df) > 1:
        for i in range(1, len(df)):
            if (df['hist'].iloc[i] > 0 and df['hist'].iloc[i-1] > 0) or \
               (df['hist'].iloc[i] < 0 and df['hist'].iloc[i-1] < 0):
                df['trend_days'].iloc[i] = df['trend_days'].iloc[i-1] + 1
            else:
                df['trend_days'].iloc[i] = 0

    if len(df) > adx_win:
        df['adx'] = calculate_adx(df, adx_win)
    else:
        df['adx'] = np.nan
    return df

# 加载大盘指标
big_market_data = calculate_indicators(big_market, rsi_window, ma_window, bb_window, adx_window)
big_market_latest = big_market_data.iloc[-1]

# 动态调整波动率
volatility_window_adjusted = volatility_window
big_nav = big_market_latest['net_value']
big_ma50 = big_market_latest['ma50']

if big_market_latest['rsi'] > 50 and big_nav > big_ma50:
    volatility_threshold = 0.03
    big_trend = '强势'
elif big_market_latest['rsi'] < 50 and big_nav < big_ma50:
    volatility_threshold = 0.025
    volatility_window_adjusted = 10
    big_trend = '弱势'
else:
    volatility_threshold = 0.03
    big_trend = '中性'

# 加载基金数据
fund_nav_data = {}
holdings = {}
for code, cost_nav in holdings_config.items():
    if code == 'parameters' or code == 'estimated_net_values': # 排除新增的预估净值配置项
        continue
    fund_file = os.path.join(fund_data_dir, f"{code}.csv")
    if os.path.exists(fund_file):
        fund_df = pd.read_csv(fund_file, parse_dates=['date'])
        if fund_df['net_value'].max() > 10 or fund_df['net_value'].min() < 0:
            print(f"警告: 基金 {code} 数据净值异常，请检查 {fund_file}")
        
        full_fund_data = calculate_indicators(fund_df, rsi_window, ma_window, volatility_window_adjusted, adx_window)
        fund_nav_data[code] = full_fund_data
        latest_nav_data = full_fund_data.iloc[-1]
        latest_net_value = float(latest_nav_data['net_value'])
        shares = 1
        value = shares * latest_net_value
        cost = shares * cost_nav
        profit = value - cost
        profit_rate = (profit / cost) * 100 if cost > 0 else 0
        full_fund_data['rolling_peak'] = full_fund_data['net_value'].cummax()
        current_peak = full_fund_data['rolling_peak'].iloc[-1]
        
        holdings[code] = {
            'value': value,
            'cost_nav': cost_nav,
            'shares': shares,
            'latest_net_value': latest_net_value,
            'profit': profit,
            'profit_rate': profit_rate,
            'current_peak': current_peak,
            'estimated_net_value': holdings_config.get('estimated_net_values', {}).get(code) # 整合您的预估净值
        }
    else:
        print(f"警告: 基金数据文件 {fund_file} 未找到，跳过 {code}")


# 决策函数（结合预估净值和趋势延续计数）
def decide_sell(code, holding, full_fund_data, big_market_latest, big_market_data, big_trend):
    print(f"\n处理基金: {code}")
    
    # T+1 延迟应对：如果存在预估净值，则使用预估值进行当前轮次的决策
    decision_nav = holding['latest_net_value']
    if holding['estimated_net_value'] is not None:
        decision_nav = holding['estimated_net_value']
        print(f"*** 警告: 使用T日预估净值({decision_nav})进行决策，实际成交以T日净值确认。 ***")

    # 基于当前净值（实际或预估）重新计算收益率和回撤
    current_value = holding['shares'] * decision_nav
    current_profit = current_value - (holding['shares'] * holding['cost_nav'])
    profit_rate = (current_profit / (holding['shares'] * holding['cost_nav'])) * 100
    
    fund_latest = full_fund_data.iloc[-1]
    rsi = fund_latest['rsi']
    macd = fund_latest['macd']
    signal = fund_latest['signal']
    ma50 = fund_latest['ma50']
    adx = fund_latest['adx']
    trend_days = fund_latest['trend_days'] # V3.0：趋势延续计数
    bb_pos = '中轨'
    sell_reasons = []

    # MACD信号和布林位置更新
    if len(full_fund_data) >= 2:
        recent_data = full_fund_data.tail(2)
        if (recent_data['net_value'] > recent_data['bb_upper']).all():
            bb_pos = '上轨'
        elif (recent_data['net_value'] < recent_data['bb_lower']).all():
             bb_pos = '下轨'
        else:
            bb_pos = '中轨'
            
    macd_signal = '金叉'
    macd_zero_dead_cross = False
    if len(full_fund_data) >= 2:
        recent_macd = full_fund_data.tail(2)
        if (recent_macd['macd'] < recent_macd['signal']).all():
            macd_signal = '死叉'
            if (recent_macd.iloc[-1]['macd'] < 0 and recent_macd.iloc[-1]['signal'] < 0):
                macd_zero_dead_cross = True

    # --- 优先级最高：绝对止损/分级止损 ---
    
    if profit_rate < -20:
        sell_reasons.append('绝对止损（亏损>20%）触发，卖出100%')
        return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因绝对止损（亏损>20%）卖出100%'}
    elif profit_rate < -15:
        sell_reasons.append('亏损>15%触发，减仓50%')
        return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因亏损>15%减仓50%'}
    elif profit_rate < -10:
        sell_reasons.append('亏损>10%触发，暂停定投')
        return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '暂停定投'}

    # --- 高优先级规则（趋势反转/止盈） ---

    # 移动止盈 (使用当前净值而非旧净值计算回撤)
    drawdown = (holding['current_peak'] - decision_nav) / holding['current_peak']
    if drawdown > trailing_stop_loss_pct:
        sell_reasons.append(f'移动止盈触发 (回撤>{int(trailing_stop_loss_pct * 100)}%)')
        return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因移动止盈卖出'}

    # MACD 顶背离 (V3.0：增加 trend_days < 2 的确认)
    if len(full_fund_data) >= macd_divergence_window:
        recent_data = full_fund_data.tail(macd_divergence_window)
        if not recent_data.empty:
            peak_nav_idx = recent_data['net_value'].idxmax()
            peak_macd_idx = recent_data['macd'].idxmax()
            # 顶背离：净值创新高，MACD未创新高，且MACD已开始收缩（trend_days < 2），且已经死叉
            if peak_nav_idx == len(recent_data) - 1 and peak_macd_idx != peak_nav_idx and trend_days < 2 and macd_signal == '死叉':
                sell_reasons.append('MACD顶背离触发')
                return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因MACD顶背离减仓70%'}

    # ADX 趋势转弱 (V3.0：增加 trend_days < 2 的确认)
    if not np.isnan(adx) and adx >= adx_threshold and macd_zero_dead_cross and trend_days < 2:
        sell_reasons.append('ADX趋势转弱触发')
        return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因ADX趋势转弱减仓50%'}

    # 布林带上轨突破计数
    if len(full_fund_data) >= profit_lock_days:
        recent_data = full_fund_data.tail(profit_lock_days)
        break_count = recent_data['bb_break_upper'].sum()
        if break_count >= 2 and rsi > 75 and macd_signal == '死叉':
            sell_reasons.append('布林带上轨突破≥2次且RSI>75且死叉，减仓50%锁定利润')
            return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因布林带上轨突破≥2次且RSI>75减仓50%'}

    # 最大回撤止损
    if len(full_fund_data) >= profit_lock_days:
        recent_data = full_fund_data.tail(profit_lock_days)
        if not recent_data.empty:
            peak_nav = recent_data['net_value'].max()
            current_nav = decision_nav # 使用当前净值（或预估净值）
            drawdown_pct = (peak_nav - current_nav) / peak_nav
            if drawdown_pct > 0.10:
                sell_reasons.append('14天内最大回撤>10%触发，止损20%')
                return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因最大回撤止损20%'}
    
    # 连续回吐日（保留原规则）
    if len(full_fund_data) >= decline_days_threshold:
        recent_returns = full_fund_data['daily_return'].tail(decline_days_threshold)
        if not recent_returns.empty and (recent_returns < 0).all():
            sell_reasons.append(f'连续{decline_days_threshold}天回吐')
            
    # 波动率卖出（保留原规则）
    if len(full_fund_data) >= volatility_window_adjusted and fund_latest['volatility'] is not np.nan and fund_latest['volatility'] > volatility_threshold:
        if rsi > 80 and macd_signal == '死叉' and decision_nav < ma50:
            sell_reasons.append('波动率过高且指标过热')
            return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '因波动率过高卖出'}

    # 超规则（指标钝化） - 暂停卖出信号
    is_overbought_consecutive = False
    if len(full_fund_data) >= consecutive_days_threshold:
        recent_rsi = full_fund_data.tail(consecutive_days_threshold)['rsi']
        if (recent_rsi > rsi_overbought_threshold).all():
             is_overbought_consecutive = True
             
    big_market_recent = big_market_data.iloc[-2:]
    big_macd_dead_cross_today = False
    if len(big_market_recent) == 2:
        if big_market_latest['macd'] < big_market_latest['signal'] and big_market_recent.iloc[0]['macd'] >= big_market_recent.iloc[0]['signal']:
            big_macd_dead_cross_today = True

    if is_overbought_consecutive:
        if (big_market_latest['macd'] > big_market_latest['signal']) and not big_macd_dead_cross_today:
             sell_reasons.append(f'持续强势，RSI>{rsi_overbought_threshold}，暂停卖出')
             return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '持续强势，暂停卖出'}

    # RSI和布林带锁定利润（保留原规则，作为次级保障）
    if len(full_fund_data) >= profit_lock_days:
        recent_data = full_fund_data.tail(profit_lock_days)
        recent_rsi = recent_data['rsi']
        bb_break = False
        if len(recent_data) >= 2:
            bb_break = recent_data.tail(2)['bb_break_upper'].all() 

        if (recent_rsi > 75).any() and bb_break:
            sell_reasons.append('RSI>75且连续突破布林带上轨，减仓50%锁定利润')
            return {'code': code, 'profit_rate': round(profit_rate, 2), 'rsi': round(rsi, 2), 'macd_signal': macd_signal, 'bb_pos': bb_pos, 'big_trend': big_trend, 'decision': '减仓50%锁定利润'}
    
    # --- 三要素综合决策（最低优先级） ---

    # 收益率要素
    if profit_rate > 50:
        sell_profit = '卖50%'
    elif profit_rate > 40:
        sell_profit = '卖30%'
    elif profit_rate > 30:
        sell_profit = '卖20%'
    elif profit_rate > 20:
        sell_profit = '卖10%'
    elif profit_rate < -10:
        sell_profit = '暂停定投'
    else:
        sell_profit = '持仓'

    # 指标要素
    indicator_sell = '持仓'
    if rsi > 85 or bb_pos == '上轨' :
        indicator_sell = '卖30%'
    elif rsi > 75 or macd_signal == '死叉':
        indicator_sell = '卖20%'

    # 大盘要素
    if big_trend == '弱势':
        market_sell = '卖10%'
    else:
        market_sell = '持仓'

    # 综合决策
    if '卖' in sell_profit and '卖' in indicator_sell and '卖' in market_sell:
        decision = '卖30%'
    elif '卖' in sell_profit and '卖' in indicator_sell:
        decision = '卖20%'
    elif '卖' in sell_profit and '卖' in market_sell:
        decision = '卖10%'
    elif '卖' in indicator_sell and '卖' in market_sell:
        decision = '卖10%'
    elif '暂停' in sell_profit:
        decision = '暂停定投'
    else:
        decision = '持仓'
        
    if sell_reasons:
        print(f"辅助信息: 触发了次级信号 {sell_reasons}")
    print(f"收益率: {round(profit_rate, 2)}%")
    print(f"RSI: {rsi}, MACD信号: {macd_signal}, 布林带位置: {bb_pos}, 50天均线: {ma50}, ADX: {adx}")

    return {
        'code': code,
        'profit_rate': round(profit_rate, 2),
        'rsi': round(rsi, 2),
        'macd_signal': macd_signal,
        'bb_pos': bb_pos,
        'big_trend': big_trend,
        'decision': decision
    }

# 生成决策
decisions = []
for code, holding in holdings.items():
    decisions.append(decide_sell(code, holding, fund_nav_data[code], big_market_latest, big_market_data, big_trend))

# 报告生成
current_time = datetime.now()
md_content = f"""
# 基金卖出决策报告 (最终整合版 - 预估净值+趋势强化 V3.0)

## 报告总览
生成时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')} (上海时间)
**重要提示:** **决策已优先使用 `estimated_net_values` (预估净值)**。如果配置中包含预估值，决策结果更及时。

## 大盘趋势
沪深300 RSI: {round(big_market_latest['rsi'], 2)} | MACD: {round(big_market_latest['macd'], 4)} | 趋势: **{big_trend}** (RSI和MA50综合判断)
