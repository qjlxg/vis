import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging
import io

# --- 配置参数 ---
FUND_DATA_DIR = 'fund_data'
INDEX_DATA_DIR = 'index_data'
INDEX_REPORT_BASE_NAME = 'quant_strategy_index_report'
INDEX_NAME = 'MarketMonitor_BuySignal_Index'
CSI300_CODE = '000300' 
CSI300_FILENAME = f'{CSI300_CODE}.csv' 
RISK_FREE_RATE_FILENAME = 'risk_free_rate.csv' 

STARTING_NAV = 1000
RISK_FREE_RATE_ANNUAL = 0.03 
TRANSACTION_COST = 0.001  # 买卖单边成本 0.1%

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 辅助函数 ---

def validate_data(df, filepath, required_columns=['date', 'net_value']):
    """验证DataFrame的基本完整性和数据合理性。"""
    if not all(col in df.columns for col in required_columns):
        logger.error(f"❌ 数据验证失败: 文件 {filepath} 缺少必需列: {required_columns}")
        return False
    
    df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
    df.dropna(subset=['net_value'], inplace=True)
    
    # 检查非正净值
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
    """
    计算年化夏普比率，使用动态年化因子和动态无风险利率。
    """
    if return_series.empty or len(return_series) < 2:
        return np.nan
    
    daily_returns = return_series.dropna()
    
    total_trading_days = len(index_df)
    time_span_days = (index_df.index.max() - index_df.index.min()).days
    
    trading_days_per_year = total_trading_days / (time_span_days / 365.25) if time_span_days > 0 else 252 
        
    aligned_returns = daily_returns.reindex(risk_free_rate_series.index)
    # 仅使用有数据的日期进行计算
    valid_dates = aligned_returns.index.intersection(risk_free_rate_series.index)
    aligned_returns = aligned_returns.loc[valid_dates]
    rfr_aligned = risk_free_rate_series.loc[valid_dates]
    
    excess_returns = aligned_returns - rfr_aligned
    
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return np.nan 

    return (mean_excess_return / std_excess_return) * np.sqrt(trading_days_per_year)

# --- 性能优化: 向量化信号计算 ---

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
    
    注：Pandas布尔索引赋值是按顺序覆盖的，但为了清晰和安全，我们从最激进/最需要回避的信号开始赋值。
    """
    signals = pd.Series('持有/观察', index=df.index)
    
    # --- 信号布尔条件 ---
    
    # MACD 金叉
    is_macd_golden_cross = (df['MACD'] > df['MACD_Signal']) & (df['Prev_MACD'] < df['Prev_Signal'])
    
    # 1. 强卖出/规避 (绝对最高优先级，覆盖所有买入)
    is_strong_sell = (df['NAV_MA50'] < nav_ma50_strong_sell) | (df['RSI'] > rsi_strong_sell_max)
    signals[is_strong_sell] = '强卖出/规避'
    
    # 2. 强买入：多因子共振 (优先级高于弱买入)
    is_deep_oversold = (df['RSI'] < rsi_strong_buy)
    is_below_ma = (df['NAV_MA50'] < nav_ma50_strong_buy_max)
    is_strong_buy_combo = is_deep_oversold & is_below_ma & is_macd_golden_cross
    is_super_strong_buy = (df['RSI'] < (rsi_strong_buy - 5)) # 例如 RSI < 25
    
    signals[is_strong_buy_combo | is_super_strong_buy] = '强买入'

    # 3. 弱买入：RSI 超卖或 MACD 金叉 (不能覆盖强卖出，但可以被强买入覆盖)
    is_weak_buy = (df['RSI'] < rsi_weak_buy) | is_macd_golden_cross
    
    # 只对目前还是 '持有/观察' 的位置赋值为 '弱买入'
    signals.mask((signals == '持有/观察') & is_weak_buy, '弱买入', inplace=True)
    
    # 4. 再次确保强卖出覆盖一切（安全）：
    signals[is_strong_sell] = '强卖出/规避'

    return signals

# --- 核心指数构建类 ---

class IndexBuilder:
    def __init__(self, fund_data_dir=FUND_DATA_DIR, index_data_dir=INDEX_DATA_DIR, index_name=INDEX_NAME, starting_nav=STARTING_NAV,
                 rsi_strong_buy=30, rsi_weak_buy=40, nav_ma50_strong_sell=0.95, nav_ma50_strong_buy_max=1.00, rsi_strong_sell_max=75,
                 ma_window=50): # 【改进】动态MA窗口
        
        self.fund_data_dir = fund_data_dir
        self.index_data_dir = index_data_dir
        self.index_name = index_name
        self.starting_nav = starting_nav
        self.all_data = {}
        self.csi300_data = None
        self.common_dates = None
        
        # 交易和风险配置
        self.transaction_cost = TRANSACTION_COST 
        
        # 信号阈值配置
        self.rsi_strong_buy = rsi_strong_buy
        self.rsi_weak_buy = rsi_weak_buy
        self.nav_ma50_strong_sell = nav_ma50_strong_sell
        self.nav_ma50_strong_buy_max = nav_ma50_strong_buy_max
        self.rsi_strong_sell_max = rsi_strong_sell_max
        self.ma_window = ma_window # 【改进】动态MA窗口
        
        # 动态无风险利率
        self.risk_free_rate_df = self._load_risk_free_rate()
        self.default_risk_free_daily = RISK_FREE_RATE_ANNUAL / 252


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
            
            # 【改进】鲁棒性检查：检查利率是否为正
            if (df['risk_free_rate_daily'] <= 0).any():
                logger.error(f"❌ 动态无风险利率文件 {rfr_file} 包含无效值（负值或零）。使用固定值。")
                return None
                
            # 转换为日利率
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
            return None
        
        try:
            df = pd.read_csv(csi300_file)
            if not validate_data(df.copy(), csi300_file): return None # 验证数据
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
                
                # 【改进】数据验证
                if not validate_data(df.copy(), filepath):
                    continue
                    
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date').set_index('date')
                
                # --- 数据缺失处理改进：插值和填充 ---
                df['net_value'] = df['net_value'].interpolate(method='linear').ffill().bfill()
                
                # 历史计算技术指标 (传入动态 MA 窗口)
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

        # --- 性能优化：预计算信号和收益率 ---
        self._precalculate_signals_and_returns()
        
        logger.info(f"✅ 数据预处理完成。公共日期范围: {self.common_dates.min().strftime('%Y-%m-%d')} - {self.common_dates.max().strftime('%Y-%m-%d')}")
        return True

    def _precalculate_signals_and_returns(self):
        """
        预先计算所有基金在所有 common_dates 上的行动信号和日收益率。
        【改进】使用向量化信号函数。
        """
        signals = {}
        returns = {}
        
        for code, df in self.all_data.items():
            # 1. 预计算信号 (使用向量化函数)
            signals[code] = generate_action_signal_vectorized(
                df, 
                self.rsi_strong_buy, 
                self.rsi_weak_buy, 
                self.nav_ma50_strong_sell, 
                self.nav_ma50_strong_buy_max,
                self.rsi_strong_sell_max
            )
            
            # 2. 预计算日收益率
            returns[code] = df['net_value'].pct_change()

        # 转换为 DataFrame 并对齐到公共日期
        self.signals_df = pd.DataFrame(signals).reindex(self.common_dates)
        self.returns_df = pd.DataFrame(returns).reindex(self.common_dates)
        
        # 沪深300收益率
        if self.csi300_data is not None:
            self.csi300_returns = self.csi300_data['net_value'].pct_change().reindex(self.common_dates)
        else:
            self.csi300_returns = pd.Series(0.0, index=self.common_dates)

    def _calculate_turnover_ratio(self, prev_holdings_set, new_holdings_set):
        """
        计算实际换仓比例。
        换手率定义近似为：(需要调整的资产份额) / 总资产
        假设等权：换手率 = (买入份额 + 卖出份额) / 总资产
        
        """
        # 如果新旧持仓完全一致，换手率为 0
        if prev_holdings_set == new_holdings_set:
            return 0.0
            
        # 换入换出基金总数 (需要调整的基金数)
        total_share_to_adjust = len(prev_holdings_set.symmetric_difference(new_holdings_set))
        
        # 总资产的归一化分母：调整前后持仓数量的最大值（因为是等权，分母代表了总资产）
        max_holdings = max(len(prev_holdings_set), len(new_holdings_set), 1)
        
        # 换仓比例 (Total adjusted share / Max Holdings)
        # 例如：prev={A,B,C}, new={A,D,E}. total_share_to_adjust=4 ({B,C,D,E}). max_holdings=3. Ratio = 4/3. (非标准)
        
        # 采用更直观的换手率计算：(卖出数量/旧持仓数) + (买入数量/新持仓数)
        
        # 需要卖出的份额 (占旧持仓的比例)
        sell_count = len(prev_holdings_set - new_holdings_set)
        sell_turnover = sell_count / max(len(prev_holdings_set), 1)
        
        # 需要买入的份额 (占新持仓的比例)
        buy_count = len(new_holdings_set - prev_holdings_set)
        buy_turnover = buy_count / max(len(new_holdings_set), 1)
        
        # 实际调整的换手率：(买入份额 + 卖出份额) / 2
        turnover_ratio = (sell_turnover + buy_turnover) / 2
        
        return turnover_ratio


    def build_index(self):
        """
        计算策略指数和基准指数的每日净值 (NAV)。
        【改进】：使用预计算的 DataFrame，并加入动态交易成本（基于实际换仓比例）。
        """
        
        index_data = pd.DataFrame(index=self.common_dates)
        index_nav = [self.starting_nav]
        csi300_nav = [self.starting_nav]
        current_holdings = [] # 当前持仓基金代码列表
        
        # 动态无风险利率系列
        rfr_series = self.risk_free_rate_df.reindex(self.common_dates).ffill().fillna(self.default_risk_free_daily)

        # 从第二个日期开始计算指数
        for i, date in enumerate(self.common_dates):
            if i == 0:
                index_data.loc[date, 'Strategy_Return'] = 0.0
                index_data.loc[date, 'CSI300_Return'] = 0.0
                index_data.loc[date, 'Signal_Funds_Count'] = 0
                continue
                
            prev_date = self.common_dates[i-1]
            
            # 获取前一日的信号 (即换仓依据)
            prev_signals = self.signals_df.loc[prev_date]
            buy_signal_codes = prev_signals[prev_signals.isin(['强买入', '弱买入'])].index.tolist()

            strategy_return = 0.0
            daily_rfr = rfr_series.loc[date]
            
            is_rebalance = bool(buy_signal_codes) 
            
            # 交易成本模型计算
            turnover_ratio = 0.0
            prev_holdings_set = set(current_holdings)
            new_holdings_set = set(buy_signal_codes)
            
            if is_rebalance:
                # 出现新信号：清仓旧持仓，买入新的信号组合 (换仓/再平衡)
                
                # 【改进】计算换仓比例
                turnover_ratio = self._calculate_turnover_ratio(prev_holdings_set, new_holdings_set)
                
                # 更新持仓
                current_holdings = buy_signal_codes
                signal_count = len(current_holdings)
                
                # 计算当日收益：新组合的日收益率平均值
                holdings_returns = self.returns_df.loc[date, current_holdings].dropna()
                
                if not holdings_returns.empty:
                    strategy_return = holdings_returns.mean()
                else:
                    strategy_return = daily_rfr
                
                # 扣除交易成本: 成本 * 换手率
                strategy_return -= self.transaction_cost * turnover_ratio
                    
            elif current_holdings:
                # 无新信号，但有持仓：保持前日的持仓组合 (持仓等待)
                signal_count = len(current_holdings)
                
                # 计算当日收益：旧组合的日收益率平均值
                holdings_returns = self.returns_df.loc[date, current_holdings].dropna()
                        
                if not holdings_returns.empty:
                    strategy_return = holdings_returns.mean()
                else:
                    strategy_return = daily_rfr
                    
            else:
                # 无新信号，且无持仓 
                signal_count = 0
                strategy_return = daily_rfr 

            # 基准指数计算 (CSI300)
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
            index_data.loc[date, 'Signal_Funds_Count'] = signal_count 
            index_data.loc[date, 'Turnover_Ratio'] = turnover_ratio

        # 最终数据整理
        index_data['Strategy_NAV'] = index_nav
        index_data['CSI300_NAV'] = csi300_nav
        index_data.index.name = 'Date'
        
        # 计算核心绩效指标
        strategy_mdd = calculate_mdd(index_data['Strategy_NAV'])
        csi300_mdd = calculate_mdd(index_data['CSI300_NAV'])
        
        strategy_sharpe = calculate_sharpe_ratio(index_data['Strategy_Return'], index_data, rfr_series)
        csi300_sharpe = calculate_sharpe_ratio(index_data['CSI300_Return'], index_data, rfr_series)

        return index_data, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe

    def generate_report(self, index_df, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe):
        """生成 Markdown 报告，更新策略描述以反映所有改进。"""
        now = datetime.now()
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
        report += "### **策略与模型最终改进总结：**\n"
        report += f"- **性能优化:** 信号计算已完全采用**向量化**操作 (无需 `apply` 循环)。\n"
        report += f"- **信号逻辑:** 采用多因子共振，MA窗口调整为 **{self.ma_window}** 日。\n"
        report += f"- **交易成本:** 每次换仓扣除 **{self.transaction_cost * 100:.2f}%** 成本，并根据**实际换仓比例**动态调整，模型更精确。\n"
        report += "- **数据鲁棒性:** 增强了对非正净值和动态无风险利率的**验证和错误处理**。\n"
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

        timestamp_for_filename = now.strftime('%Y%m%d_%H%M%S')
        DIR_NAME = now.strftime('%Y%m')
        os.makedirs(DIR_NAME, exist_ok=True)
        
        REPORT_FILE = os.path.join(DIR_NAME, f"{INDEX_REPORT_BASE_NAME}_{timestamp_for_filename}.md")
        
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
    # 示例运行：可在此处配置不同的参数
    # 例如：builder = IndexBuilder(ma_window=20, rsi_strong_buy=25)
    builder = IndexBuilder()
    builder.run()
