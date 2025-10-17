import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 【新增】绘图库字体管理器，用于支持中文
import matplotlib.font_manager as fm

# --- 配置参数 (不变) ---
FUND_DATA_DIR = 'fund_data'
INDEX_DATA_DIR = 'index_data'
INDEX_REPORT_BASE_NAME = 'quant_strategy_index_report'
INDEX_NAME = 'MarketMonitor_BuySignal_Index' # 策略指数名称
CSI300_CODE = '000300' 
CSI300_FILENAME = f'{CSI300_CODE}.csv' 
RISK_FREE_RATE_FILENAME = 'risk_free_rate.csv' 

STARTING_NAV = 1000
RISK_FREE_RATE_ANNUAL = 0.03 
TRANSACTION_COST = 0.001  
MAX_MISSING_DAYS = 20 

# 配置日志 (不变)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 辅助函数 (保持不变) ---
def validate_data(df, filepath, required_columns=['date', 'net_value']):
    """验证DataFrame的基本完整性和数据合理性。"""
    if not all(col in df.columns for col in required_columns):
        logger.error(f"❌ 数据验证失败: 文件 {filepath} 缺少必需列: {required_columns}")
        return False
    
    df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
    df.dropna(subset=['net_value'], inplace=True)
    
    if (df['net_value'] <= 0).any():
        logger.error(f"❌ 数据验证失败: 文件 {filepath} 包含无效净值（负值或零）")
        return False
        
    return True

def calculate_mdd(nav_series):
    """计算最大回撤 (Maximum Drawdown)"""
    if nav_series.empty:
        return 0.0
    rolling_max = nav_series.expanding().max()
    drawdown = (nav_series / rolling_max) - 1.0
    return abs(drawdown.min())

def calculate_sharpe_ratio(return_series, index_df, risk_free_rate_series):
    """计算年化夏普比率，使用动态年化因子和动态无风险利率。"""
    if return_series.empty or len(return_series) < 2:
        return np.nan
    
    daily_returns = return_series.dropna()
    total_trading_days = len(index_df)
    time_span_days = (index_df.index.max() - index_df.index.min()).days
    trading_days_per_year = total_trading_days / (time_span_days / 365.25) if time_span_days > 0 else 252 
        
    aligned_returns = daily_returns.reindex(risk_free_rate_series.index)
    valid_dates = aligned_returns.index.intersection(risk_free_rate_series.index)
    aligned_returns = aligned_returns.loc[valid_dates]
    rfr_aligned = risk_free_rate_series.loc[valid_dates]
    
    excess_returns = aligned_returns - rfr_aligned
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return np.nan 

    return (mean_excess_return / std_excess_return) * np.sqrt(trading_days_per_year)

def calculate_technical_indicators_for_day(df, ma_window):
    """计算关键技术指标 (RSI, MACD, MA) 的历史序列。"""
    if 'net_value' not in df.columns or len(df) < ma_window:
        for col in ['RSI', 'MACD', 'MACD_Signal', 'NAV_MA50', 'Prev_MACD', 'Prev_Signal']:
             df[col] = np.nan
        return df

    # 1. RSI (14 days)
    delta = df['net_value'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9)
    ema_12 = df['net_value'].ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = df['net_value'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

    # 3. 动态移动平均线 (MA)
    df['MA'] = df['net_value'].rolling(window=ma_window, min_periods=ma_window).mean()
    df['NAV_MA50'] = df['net_value'] / df['MA']
    
    # 4. 前一日信号 (用于金叉死叉判断)
    df['Prev_MACD'] = df['MACD'].shift(1)
    df['Prev_Signal'] = df['MACD_Signal'].shift(1)
    
    return df

def generate_action_signal_vectorized(df, rsi_strong_buy, rsi_weak_buy, nav_ma50_strong_sell, nav_ma50_strong_buy_max, rsi_strong_sell_max):
    """
    【完全向量化】根据多因子共振逻辑生成行动信号。
    """
    signals = pd.Series('持有/观察', index=df.index)
    is_macd_golden_cross = (df['MACD'] > df['MACD_Signal']) & (df['Prev_MACD'] < df['Prev_Signal'])
    
    # 1. 强卖出/规避 (最高优先级)
    is_strong_sell = (df['NAV_MA50'] < nav_ma50_strong_sell) | (df['RSI'] > rsi_strong_sell_max)
    signals[is_strong_sell] = '强卖出/规避'
    
    # 2. 强买入 (第二优先级)
    strong_buy_combo = (
        (df['RSI'] < rsi_strong_buy) & 
        (df['NAV_MA50'] < nav_ma50_strong_buy_max) & 
        is_macd_golden_cross
    )
    is_super_strong_buy = (df['RSI'] < (rsi_strong_buy - 5)) 
    strong_buy_condition = strong_buy_combo | is_super_strong_buy
    signals[(signals == '持有/观察') & strong_buy_condition] = '强买入'

    # 3. 弱买入 (最低优先级)
    is_weak_buy_base = (df['RSI'] < rsi_weak_buy) | is_macd_golden_cross
    signals.mask((signals == '持有/观察') & is_weak_buy_base, '弱买入', inplace=True)

    return signals
    
def _calculate_turnover_ratio(prev_holdings_set, new_holdings_set):
    """
    计算实际换仓比例 (Total Turnover Ratio)。
    采用 (买入权重 + 卖出权重) 的更精细模型。
    """
    if not prev_holdings_set and not new_holdings_set:
        return 0.0
    
    sell_count = len(prev_holdings_set - new_holdings_set)
    sell_weight = sell_count / max(len(prev_holdings_set), 1)
    
    buy_count = len(new_holdings_set - prev_holdings_set)
    buy_weight = buy_count / max(len(new_holdings_set), 1)
    
    turnover_ratio = sell_weight + buy_weight
    
    return min(turnover_ratio, 1.0) 

# --- 核心指数构建类 ---

class IndexBuilder:
    def __init__(self, fund_data_dir=FUND_DATA_DIR, index_data_dir=INDEX_DATA_DIR, index_name=INDEX_NAME, starting_nav=STARTING_NAV,
                 rsi_strong_buy=30, rsi_weak_buy=40, nav_ma50_strong_sell=0.95, nav_ma50_strong_buy_max=1.00, rsi_strong_sell_max=75,
                 ma_window=50): 
        
        self.fund_data_dir = fund_data_dir
        self.index_data_dir = index_data_dir
        self.index_name = index_name
        self.starting_nav = starting_nav
        self.all_data = {}
        self.csi300_data = None
        self.common_dates = None
        
        self.transaction_cost = TRANSACTION_COST 
        
        self.rsi_strong_buy = rsi_strong_buy
        self.rsi_weak_buy = rsi_weak_buy
        self.nav_ma50_strong_sell = nav_ma50_strong_sell
        self.nav_ma50_strong_buy_max = nav_ma50_strong_buy_max
        self.rsi_strong_sell_max = rsi_strong_sell_max
        self.ma_window = ma_window 
        self.max_missing_days = MAX_MISSING_DAYS 
        
        self.default_risk_free_daily = RISK_FREE_RATE_ANNUAL / 252
        self.risk_free_rate_df = self._load_risk_free_rate()

    def _load_risk_free_rate(self):
        """加载动态无风险利率，若不存在则返回 None，并增强鲁棒性。"""
        rfr_file = os.path.join(self.index_data_dir, RISK_FREE_RATE_FILENAME)
        if not os.path.exists(rfr_file):
            logger.warning(f"⚠️ 未找到动态无风险利率文件 '{rfr_file}'。将使用固定年化 {RISK_FREE_RATE_ANNUAL:.1%}。")
            return None
        
        try:
            df = pd.read_csv(rfr_file)
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'rate': 'risk_free_rate_daily'}, inplace=True)
            df.dropna(subset=['risk_free_rate_daily', 'date'], inplace=True)
            df = df.sort_values(by='date').set_index('date')
            
            if (df['risk_free_rate_daily'] <= 0).any():
                logger.error(f"❌ 动态无风险利率文件 {rfr_file} 包含无效值（负值或零）。使用固定值。")
                return None
                
            df['risk_free_rate_daily'] = df['risk_free_rate_daily'] / 252 
            
            logger.info("✅ 动态无风险利率数据加载成功。")
            return df['risk_free_rate_daily']
        except Exception as e:
            logger.error(f"❌ 加载动态无风险利率数据时发生错误: {e}")
            return None

    def _get_csi300_data(self):
        """加载沪深300指数数据。"""
        csi300_file = os.path.join(self.index_data_dir, CSI300_FILENAME)
        if not os.path.exists(csi300_file):
            logger.warning(f"⚠️ 未找到沪深300数据文件 '{csi300_file}'。指数对比功能将被禁用。")
            return None
        
        try:
            df = pd.read_csv(csi300_file)
            if not validate_data(df.copy(), csi300_file): return None 
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date').set_index('date')
            return df
        except Exception as e:
            logger.error(f"❌ 加载沪深300数据时发生错误: {e}")
            return None


    def load_and_preprocess_data(self):
        """加载所有基金和基准指数数据，计算指标，并查找公共日期。"""
        if not os.path.exists(self.fund_data_dir):
            logger.error(f"❌ 基金数据目录 '{self.fund_data_dir}' 不存在。")
            return False
            
        self.csi300_data = self._get_csi300_data()
        
        csv_files = glob.glob(os.path.join(self.fund_data_dir, '*.csv'))
        all_dates_indices = []
        if self.csi300_data is not None:
             all_dates_indices.append(self.csi300_data.index)
        
        for filepath in csv_files:
            fund_code = os.path.splitext(os.path.basename(filepath))[0]
            try:
                df = pd.read_csv(filepath)
                df = df.rename(columns={'net_value': 'net_value', 'date': 'date'}) 
                
                if not validate_data(df.copy(), filepath):
                    continue
                    
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date').set_index('date')
                
                # 检查长期缺失数据 (停牌)
                initial_na_series = pd.read_csv(filepath)['net_value'].isna()
                max_consecutive_na = initial_na_series.rolling(window=self.max_missing_days).sum().max() if len(initial_na_series) >= self.max_missing_days else 0
                missing_ratio = initial_na_series.sum() / len(initial_na_series) if len(initial_na_series) > 0 else 0
                     
                if max_consecutive_na >= self.max_missing_days or missing_ratio > 0.5:
                    logger.warning(f"⚠️ 基金 {fund_code} 长期缺失数据（最大连缺 {max_consecutive_na} 天或缺失率 {missing_ratio:.1%}），跳过处理。")
                    continue
                
                # 数据缺失处理：插值和填充
                df['net_value'] = df['net_value'].interpolate(method='linear').ffill().bfill()
                
                # 历史计算技术指标
                df = calculate_technical_indicators_for_day(df.copy(), self.ma_window)
                
                self.all_data[fund_code] = df
                all_dates_indices.append(df.index)
            except Exception as e:
                logger.warning(f"❌ 处理基金文件 {filepath} 时发生错误: {e}")
                continue

        if not self.all_data:
            logger.error("❌ 没有成功加载任何基金数据。")
            return False

        # 确定公共日期范围
        full_index = all_dates_indices[0]
        for index in all_dates_indices[1:]:
            full_index = full_index.union(index)
            
        min_start_date = max(df.index.min() for df in self.all_data.values())
        if self.csi300_data is not None:
            min_start_date = max(min_start_date, self.csi300_data.index.min())

        max_end_date = min(df.index.max() for df in self.all_data.values())
        if self.csi300_data is not None:
             max_end_date = min(max_end_date, self.csi300_data.index.max())

        self.common_dates = full_index[
            (full_index >= min_start_date) & 
            (full_index <= max_end_date)
        ].sort_values()
        
        # 剔除无法计算指标的初期数据 (需要 MA 窗口天数的数据)
        min_indicator_start_date = self.common_dates.min() + pd.Timedelta(days=self.ma_window)
        self.common_dates = self.common_dates[self.common_dates >= min_indicator_start_date]

        if len(self.common_dates) < self.ma_window:
             logger.error(f"❌ 警告：公共数据日期少于 {self.ma_window} 天 (找到 {len(self.common_dates)} 天)。停止构建。")
             return False

        self._precalculate_signals_and_returns()
        
        logger.info(f"✅ 数据预处理完成。公共日期范围: {self.common_dates.min().strftime('%Y-%m-%d')} - {self.common_dates.max().strftime('%Y-%m-%d')}")
        return True

    def _precalculate_signals_and_returns(self):
        """预先计算所有基金在所有 common_dates 上的行动信号和日收益率。"""
        signals = {}
        returns = {}
        
        for code, df in self.all_data.items():
            signals[code] = generate_action_signal_vectorized(
                df, 
                self.rsi_strong_buy, 
                self.rsi_weak_buy, 
                self.nav_ma50_strong_sell, 
                self.nav_ma50_strong_buy_max,
                self.rsi_strong_sell_max
            )
            
            returns[code] = df['net_value'].pct_change()

        self.signals_df = pd.DataFrame(signals).reindex(self.common_dates)
        self.returns_df = pd.DataFrame(returns).reindex(self.common_dates)
        
        if self.csi300_data is not None:
            self.csi300_returns = self.csi300_data['net_value'].pct_change().reindex(self.common_dates)
        else:
            self.csi300_returns = pd.Series(0.0, index=self.common_dates)

    def _calculate_turnover_ratio(self, prev_holdings_set, new_holdings_set):
        """
        计算实际换仓比例 (Total Turnover Ratio)。
        采用 (买入权重 + 卖出权重) 的更精细模型。
        """
        if not prev_holdings_set and not new_holdings_set:
            return 0.0
        
        sell_count = len(prev_holdings_set - new_holdings_set)
        sell_weight = sell_count / max(len(prev_holdings_set), 1)
        
        buy_count = len(new_holdings_set - prev_holdings_set)
        buy_weight = buy_count / max(len(new_holdings_set), 1)
        
        turnover_ratio = sell_weight + buy_weight
        
        return min(turnover_ratio, 1.0) 


    def build_index(self):
        """计算策略指数和基准指数的每日净值 (NAV)，并记录每日信号数量。"""
        
        index_data = pd.DataFrame(index=self.common_dates)
        index_nav = [self.starting_nav]
        csi300_nav = [self.starting_nav]
        current_holdings = [] 
        
        # 处理动态无风险利率
        if self.risk_free_rate_df is not None:
            rfr_series = self.risk_free_rate_df.reindex(self.common_dates).ffill().fillna(self.default_risk_free_daily)
        else:
            rfr_series = pd.Series(self.default_risk_free_daily, index=self.common_dates)
            
        # 从第二个日期开始计算指数
        for i, date in enumerate(self.common_dates):
            if i == 0:
                index_data.loc[date, 'Strategy_Return'] = 0.0
                index_data.loc[date, 'CSI300_Return'] = 0.0
                index_data.loc[date, 'Signal_Funds_Count'] = 0 # 记录信号数量
                index_data.loc[date, 'Turnover_Ratio'] = 0.0
                continue
                
            prev_date = self.common_dates[i-1]
            prev_signals = self.signals_df.loc[prev_date]
            buy_signal_codes = prev_signals[prev_signals.isin(['强买入', '弱买入'])].index.tolist()

            strategy_return = 0.0
            daily_rfr = rfr_series.loc[date]
            
            is_rebalance = bool(buy_signal_codes) 
            
            turnover_ratio = 0.0
            prev_holdings_set = set(current_holdings)
            new_holdings_set = set(buy_signal_codes)
            
            # 判断是否需要计算换仓和扣除成本
            if is_rebalance or (prev_holdings_set != new_holdings_set):
                
                turnover_ratio = self._calculate_turnover_ratio(prev_holdings_set, new_holdings_set)
                
                current_holdings = buy_signal_codes
                signal_count = len(current_holdings)
                
                holdings_returns = self.returns_df.loc[date, current_holdings].dropna()
                
                if not holdings_returns.empty and len(current_holdings) > 0:
                    strategy_return = holdings_returns.mean()
                else:
                    strategy_return = daily_rfr
                
                strategy_return -= self.transaction_cost * turnover_ratio
                    
            elif current_holdings:
                # 保持前日的持仓组合
                signal_count = len(current_holdings)
                
                holdings_returns = self.returns_df.loc[date, current_holdings].dropna()
                        
                if not holdings_returns.empty and len(current_holdings) > 0:
                    strategy_return = holdings_returns.mean()
                else:
                    strategy_return = daily_rfr
                    
            else:
                # 空仓
                signal_count = 0
                strategy_return = daily_rfr 

            # 基准指数计算
            csi300_return = self.csi300_returns.loc[date] if date in self.csi300_returns.index and not pd.isna(self.csi300_returns.loc[date]) else 0.0

            # 更新 NAV
            prev_strategy_nav = index_nav[-1]
            current_strategy_nav = prev_strategy_nav * (1 + strategy_return)
            index_nav.append(current_strategy_nav)
            
            prev_csi300_nav = csi300_nav[-1]
            current_csi300_nav = prev_csi300_nav * (1 + csi300_return)
            csi300_nav.append(current_csi300_nav)

            # 记录数据
            index_data.loc[date, 'Strategy_Return'] = strategy_return
            index_data.loc[date, 'CSI300_Return'] = csi300_return
            index_data.loc[date, 'Signal_Funds_Count'] = signal_count # 核心数据：每日买入信号基金数
            index_data.loc[date, 'Turnover_Ratio'] = turnover_ratio

        index_data['Strategy_NAV'] = index_nav
        index_data['CSI300_NAV'] = csi300_nav
        index_data.index.name = 'Date'
        
        strategy_mdd = calculate_mdd(index_data['Strategy_NAV'])
        csi300_mdd = calculate_mdd(index_data['CSI300_NAV'])
        strategy_sharpe = calculate_sharpe_ratio(index_data['Strategy_Return'], index_data, rfr_series)
        csi300_sharpe = calculate_sharpe_ratio(index_data['CSI300_Return'], index_data, rfr_series)

        return index_data, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe

    def _configure_chinese_font(self):
        """配置中文支持，优先使用 Noto Sans CJK SC"""
        # 尝试查找 Noto Sans CJK SC 或其他常用中文字体
        font_names = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        
        found_font = None
        for font_name in font_names:
            try:
                # 检查字体是否可用
                fm.findfont(font_name, fallback_to_default=False)
                found_font = font_name
                break
            except:
                continue

        if found_font:
            plt.rcParams['font.sans-serif'] = [found_font]
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        else:
            # 如果是 CI 环境 (如 GitHub Actions)，可能会找不到字体，但仍应尝试使用默认配置
            logger.warning("⚠️ 未找到合适的中文字体，图表中的中文可能显示为方块。请确保系统安装了 Noto Sans CJK SC 或 SimHei。")

    # 【原有功能】绘制累计净值曲线图
    def _plot_index_nav(self, index_df, output_path):
        """生成并保存指数净值曲线图。"""
        self._configure_chinese_font() # 调用字体配置
        plt.style.use('ggplot') 
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制策略指数
        ax.plot(index_df.index, index_df['Strategy_NAV'], label=self.index_name, color='blue', linewidth=2)
        
        # 绘制基准指数 (如果存在)
        if self.csi300_data is not None:
            ax.plot(index_df.index, index_df['CSI300_NAV'], label='沪深300 (基准)', color='red', linestyle='--', linewidth=1.5)
        
        # 格式化日期轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=45)

        # 设置标题和标签
        ax.set_title(f'{self.index_name} vs 沪深300 净值走势 ({index_df.index.min().strftime("%Y-%m-%d")} to {index_df.index.max().strftime("%Y-%m-%d")})', fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('累计净值 (基值={})'.format(self.starting_nav), fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图片
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"✅ 累计净值曲线图已保存到 {output_path}")

    # 【新增功能】绘制信号数量指数图
    def _plot_signal_count(self, index_df, output_path):
        """生成并保存每天满足买入信号的基金数量的统计图（信号强度指数）。"""
        
        if 'Signal_Funds_Count' not in index_df.columns:
            logger.error("❌ 无法绘制信号基金数量图：缺少 'Signal_Funds_Count' 列。")
            return
            
        self._configure_chinese_font() # 调用字体配置
        plt.style.use('ggplot') 
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制每日满足买入信号的基金数量
        total_funds_count = len(self.all_data)
        ax.bar(index_df.index, index_df['Signal_Funds_Count'], label='每日买入信号基金数量', color='green', alpha=0.7)
        
        # 绘制数量的滚动平均线 (20日均值)
        rolling_mean = index_df['Signal_Funds_Count'].rolling(window=20).mean()
        ax.plot(index_df.index, rolling_mean, label='20日信号平均数', color='darkorange', linewidth=2)
        
        # 添加总基金数横线作为参考
        ax.axhline(y=total_funds_count, color='red', linestyle='--', alpha=0.8, label=f'总基金数量 ({total_funds_count})')
        
        # 格式化日期轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=45)

        # 设置标题和标签
        ax.set_title(f'量化策略信号强度指数 (Signal Strength Index) - 每日买入信号基金数量', fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('基金数量', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图片
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"✅ 信号强度指数图已保存到 {output_path}")


    def generate_report(self, index_df, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe):
        """生成 Markdown 报告，并调用绘图函数（包含两个图）。"""
        now = datetime.now()
        timestamp_for_filename = now.strftime('%Y%m%d_%H%M%S')
        DIR_NAME = now.strftime('%Y%m')
        os.makedirs(DIR_NAME, exist_ok=True)
        
        REPORT_FILE = os.path.join(DIR_NAME, f"{INDEX_REPORT_BASE_NAME}_{timestamp_for_filename}.md")
        PLOT_NAV_FILE = os.path.join(DIR_NAME, f"{INDEX_REPORT_BASE_NAME}_NAV_{timestamp_for_filename}.png") # 净值图路径
        PLOT_COUNT_FILE = os.path.join(DIR_NAME, f"{INDEX_REPORT_BASE_NAME}_COUNT_{timestamp_for_filename}.png") # 信号数量图路径
        
        # 1. 生成两张图片并保存
        self._plot_index_nav(index_df, PLOT_NAV_FILE)
        self._plot_signal_count(index_df, PLOT_COUNT_FILE) 
        
        # 2. 生成 Markdown 报告内容
        start_date = index_df.index.min().strftime('%Y-%m-%d')
        end_date = index_df.index.max().strftime('%Y-%m-%d')
        strategy_nav_end = index_df['Strategy_NAV'].iloc[-1]
        csi300_nav_end = index_df['CSI300_NAV'].iloc[-1]
        total_return_strategy = (strategy_nav_end / self.starting_nav) - 1
        total_return_csi300 = (csi300_nav_end / self.starting_nav) - 1
        excess_return = total_return_strategy - total_return_csi300
        
        report = f"# 量化策略指数报告 - {self.index_name}\n\n"
        report += f"生成日期: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"数据周期: {start_date} 至 {end_date} (共 {len(index_df)} 个交易日)\n"
        
        # 【更新报告】图片引用部分，包含两个图
        report += f"\n## 1. 策略累计净值走势图 (传统指数功能)\n\n![累计净值曲线图]({os.path.basename(PLOT_NAV_FILE)})\n"
        report += f"\n## 2. 信号强度指数 (Signal Strength Index - 市场机会热度)\n\n![信号基金数量图]({os.path.basename(PLOT_COUNT_FILE)})\n"
        
        report += "### **策略与模型最终改进总结：**\n"
        report += f"- **性能优化:** 信号计算已完全采用**向量化**操作，并修复了无风险利率 `NoneType` 错误。\n"
        report += f"- **信号逻辑:** 采用多因子共振，MA窗口调整为 **{self.ma_window}** 日，**信号优先级严格化** (强卖 > 强买 > 弱买)。\n"
        report += f"- **交易成本:** 每次换仓扣除 **{self.transaction_cost * 100:.2f}%** 成本，并根据**实际换仓比例**（买入权重+卖出权重）动态调整。\n"
        report += f"- **数据鲁棒性:** 增强了数据验证，并剔除了有**超过 {self.max_missing_days} 天连续缺失净值**的基金。\n"
        report += f"- **无风险利率:** 采用动态无风险利率 ({'已加载动态数据' if self.risk_free_rate_df is not None else '使用固定年化 3.0%'})。\n"
        
        report += f"## **策略指数表现总结**\n"
        report += f"**指数名称:** {self.index_name}\n"
        report += f"**起始净值:** {self.starting_nav:.0f}\n"
        
        report += "| 指数 | 最终净值 | 总回报率 | 超额收益 | 夏普比率 | 最大回撤 |\n"
        report += "| :--- | ---: | ---: | :---: | :---: | :---: |\n"
        report += (f"| **{self.index_name}** | **{strategy_nav_end:.4f}** | **{total_return_strategy:.2%}** "
                   f"| **{excess_return:.2%}** | **{strategy_sharpe:.2f}** | **{strategy_mdd:.2%}** |\n")
        report += (f"| **沪深300 (基准)** | {csi300_nav_end:.4f} | {total_return_csi300:.2%} "
                   f"| - | {csi300_sharpe:.2f} | {csi300_mdd:.2%} |\n\n")
        
        report += "## 指数净值历史走势 (最新 60 天)\n\n"
        
        display_df = index_df.tail(60).copy()
        display_df['Strategy_NAV'] = display_df['Strategy_NAV'].apply(lambda x: f"{x:.4f}")
        display_df['CSI300_NAV'] = display_df['CSI300_NAV'].apply(lambda x: f"{x:.4f}")
        display_df['Signal_Funds_Count'] = display_df['Signal_Funds_Count'].astype(int)
        
        display_df = display_df.rename(columns={
            'Strategy_NAV': '策略指数净值',
            'CSI300_NAV': '沪深300净值',
            'Strategy_Return': '策略日收益',
            'CSI300_Return': '300日收益',
            'Signal_Funds_Count': '持仓基金数',
            'Turnover_Ratio': '换手率'
        })
        
        display_df['策略日收益'] = display_df['策略日收益'].apply(lambda x: f"{x * 100:.2%}")
        display_df['300日收益'] = display_df['300日收益'].apply(lambda x: f"{x * 100:.2%}")
        display_df['换手率'] = display_df['换手率'].apply(lambda x: f"{x * 100:.1f}%")

        markdown_table = display_df[['策略指数净值', '沪深300净值', '策略日收益', '300日收益', '持仓基金数', '换手率']].to_markdown(index=True)
        
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write(markdown_table)

        logger.info(f"✅ 指数报告已保存到 {REPORT_FILE}")
        return REPORT_FILE

    def run(self):
        """主执行流程。"""
        logger.info("🚀 开始执行量化策略指数构建...")
            
        if not self.load_and_preprocess_data():
            logger.warning("🚫 数据加载失败或数据量不足，停止指数构建。")
            return None
        
        result = self.build_index()
        
        if result is not None:
            index_df, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe = result
            self.generate_report(index_df, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe)


if __name__ == '__main__':
    # 确保在运行前，您的 fund_data/ 和 index_data/ 目录下有基金净值文件和 000300.csv 等文件
    builder = IndexBuilder()
    builder.run()
