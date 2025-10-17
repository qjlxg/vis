import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
# import matplotlib.pyplot as plt # <--- 不再需要，但保留了，因为脚本中仍有字体配置，注释掉以减少不必要的依赖
# import matplotlib.font_manager as fm # <--- 不再需要

# --- 配置参数 ---
FUND_DATA_DIR = 'fund_data'  # 基金数据目录
INDEX_NAME = 'Simple_BuySignal_Index'  # 指数名称
STARTING_NAV = 1000  # 起始净值
MA_WINDOW = 50  # MA窗口
RISK_FREE_RATE_DAILY = 0.03 / 252  # 无风险利率（年化3%日化）

# 信号阈值
RSI_STRONG_BUY = 30
RSI_WEAK_BUY = 40
NAV_MA50_STRONG_SELL = 0.95
NAV_MA50_STRONG_BUY_MAX = 1.00
RSI_STRONG_SELL_MAX = 75
MAX_MISSING_DAYS = 20  # 最大连续缺失天数

# --- 辅助函数 ---

def validate_data(df, filepath):
    """验证数据完整性"""
    required_columns = ['date', 'net_value']
    if not all(col in df.columns for col in required_columns):
        print(f"错误: 文件 {filepath} 缺少必需列 {required_columns}")
        return False
    df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
    df.dropna(subset=['net_value'], inplace=True)
    if (df['net_value'] <= 0).any():
        print(f"错误: 文件 {filepath} 包含无效净值（负值或零）")
        return False
    return True

def calculate_technical_indicators(df, ma_window):
    """计算RSI, MACD, MA"""
    if len(df) < ma_window:
        print(f"警告: 数据少于 {ma_window} 天，跳过指标计算")
        return df

    # RSI (14 days)
    delta = df['net_value'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = df['net_value'].ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = df['net_value'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

    # MA
    df['MA'] = df['net_value'].rolling(window=ma_window, min_periods=ma_window).mean()
    df['NAV_MA50'] = df['net_value'] / df['MA']

    # 前一日MACD用于金叉判断
    df['Prev_MACD'] = df['MACD'].shift(1)
    df['Prev_Signal'] = df['MACD_Signal'].shift(1)

    return df

def generate_action_signal(df):
    """生成信号，向量化"""
    signals = pd.Series('持有/观察', index=df.index)

    is_macd_golden_cross = (df['MACD'] > df['MACD_Signal']) & (df['Prev_MACD'] < df['Prev_Signal'])

    # 强卖出
    is_strong_sell = (df['NAV_MA50'] < NAV_MA50_STRONG_SELL) | (df['RSI'] > RSI_STRONG_SELL_MAX)
    signals[is_strong_sell] = '强卖出'

    # 强买入
    strong_buy_combo = (
        (df['RSI'] < RSI_STRONG_BUY) &
        (df['NAV_MA50'] < NAV_MA50_STRONG_BUY_MAX) &
        is_macd_golden_cross
    )
    is_super_strong_buy = (df['RSI'] < (RSI_STRONG_BUY - 5))
    strong_buy_condition = strong_buy_combo | is_super_strong_buy
    signals[(signals == '持有/观察') & strong_buy_condition] = '强买入'

    # 弱买入
    is_weak_buy_base = (df['RSI'] < RSI_WEAK_BUY) | is_macd_golden_cross
    signals[(signals == '持有/观察') & is_weak_buy_base] = '弱买入'

    return signals

# --- 核心指数构建 ---

class SimpleIndexBuilder:
    def __init__(self):
        self.all_data = {}
        self.common_dates = None

    def load_and_preprocess_data(self):
        if not os.path.exists(FUND_DATA_DIR):
            print(f"错误: 目录 '{FUND_DATA_DIR}' 不存在")
            return False

        csv_files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
        all_dates = []

        for filepath in csv_files:
            fund_code = os.path.splitext(os.path.basename(filepath))[0]
            try:
                df = pd.read_csv(filepath)
                if not validate_data(df, filepath):
                    continue
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date').set_index('date')

                # 检查缺失
                is_na_series = df['net_value'].isna()
                max_consecutive_na = is_na_series.rolling(window=MAX_MISSING_DAYS).sum().max()
                missing_ratio = is_na_series.sum() / len(is_na_series)
                if max_consecutive_na >= MAX_MISSING_DAYS or missing_ratio > 0.5:
                    print(f"警告: 基金 {fund_code} 缺失数据过多，跳过")
                    continue

                df['net_value'] = df['net_value'].interpolate(method='linear').ffill().bfill()
                df = calculate_technical_indicators(df, MA_WINDOW)
                self.all_data[fund_code] = df
                all_dates.append(df.index)
            except Exception as e:
                print(f"错误处理 {filepath}: {e}")
                continue

        if not self.all_data:
            print("错误: 没有加载任何基金数据")
            return False

        # 公共日期
        full_index = pd.Index([])
        for idx in all_dates:
            full_index = full_index.union(idx)
        min_start = max(df.index.min() for df in self.all_data.values())
        max_end = min(df.index.max() for df in self.all_data.values())
        self.common_dates = full_index[(full_index >= min_start) & (full_index <= max_end)].sort_values()

        min_indicator_start = self.common_dates.min() + pd.Timedelta(days=MA_WINDOW)
        self.common_dates = self.common_dates[self.common_dates >= min_indicator_start]

        if len(self.common_dates) < MA_WINDOW:
            print(f"错误: 公共日期少于 {MA_WINDOW} 天")
            return False

        self._precalculate_signals_and_returns()
        return True

    def _precalculate_signals_and_returns(self):
        signals = {}
        returns = {}
        for code, df in self.all_data.items():
            signals[code] = generate_action_signal(df)
            returns[code] = df['net_value'].pct_change()
        self.signals_df = pd.DataFrame(signals).reindex(self.common_dates)
        self.returns_df = pd.DataFrame(returns).reindex(self.common_dates)

    def build_index(self):
        index_nav = [STARTING_NAV]
        current_holdings = []

        for i, date in enumerate(self.common_dates):
            if i == 0:
                continue

            prev_date = self.common_dates[i-1]
            prev_signals = self.signals_df.loc[prev_date]
            buy_codes = prev_signals[prev_signals.isin(['强买入', '弱买入'])].index.tolist()

            strategy_return = RISK_FREE_RATE_DAILY  # 默认无风险利率
            if buy_codes:
                current_holdings = buy_codes
                holdings_returns = self.returns_df.loc[date, current_holdings].dropna()
                if not holdings_returns.empty:
                    strategy_return = holdings_returns.mean()

            prev_nav = index_nav[-1]
            current_nav = prev_nav * (1 + strategy_return)
            index_nav.append(current_nav)

        index_df = pd.DataFrame({'NAV': index_nav}, index=self.common_dates)
        return index_df

    # 以下 plot_nav_curve 函数已注释/移除，不再生成图片
    # def plot_nav_curve(self, index_df):
    #     ...

    def save_nav_csv(self, index_df):
        """保存 NAV 数据为 CSV"""
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f'plots/{INDEX_NAME}_NAV_{now}.csv'
        os.makedirs('plots', exist_ok=True)
        index_df.to_csv(csv_file, encoding='utf-8')
        print(f"NAV 数据已保存到: {csv_file}")
        return csv_file

    def build_signal_count_index(self):
        """统计信号数量作为指数值"""
        index_values = {'强买入': [], '弱买入': [], '强卖出': [], '持有/观察': []}
        for date in self.common_dates:
            signals = self.signals_df.loc[date]
            index_values['强买入'].append(signals[signals == '强买入'].count())
            index_values['弱买入'].append(signals[signals == '弱买入'].count())
            index_values['强卖出'].append(signals[signals == '强卖出'].count())
            index_values['持有/观察'].append(signals[signals == '持有/观察'].count())
        return pd.DataFrame(index_values, index=self.common_dates)

    # 以下 plot_signal_count_curve 函数已注释/移除，不再生成图片
    # def plot_signal_count_curve(self, index_df):
    #     ...

    def save_signal_count_csv(self, index_df):
        """保存信号数量数据为 CSV"""
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f'plots/{INDEX_NAME}_Signal_Count_{now}.csv'
        os.makedirs('plots', exist_ok=True)
        index_df.to_csv(csv_file, encoding='utf-8')
        print(f"信号数量数据已保存到: {csv_file}")
        return csv_file

    def run(self):
        if not self.load_and_preprocess_data():
            return
            
        # 1. NAV 指数
        nav_index_df = self.build_index()
        # self.plot_nav_curve(nav_index_df) # <--- 注释掉绘图
        self.save_nav_csv(nav_index_df) # <--- 保留保存 CSV

        # 2. 信号数量指数
        signal_count_df = self.build_signal_count_index()
        # self.plot_signal_count_curve(signal_count_df) # <--- 注释掉绘图
        self.save_signal_count_csv(signal_count_df) # <--- 保留保存 CSV

if __name__ == '__main__':
    builder = SimpleIndexBuilder()
    builder.run()
