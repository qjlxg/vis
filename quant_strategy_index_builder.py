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
STARTING_NAV = 1000
RISK_FREE_RATE_ANNUAL = 0.03 # 假设年化无风险利率 3%
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / 252 # 每日无风险收益率

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 关键绩效指标计算函数 ---

def calculate_mdd(nav_series):
    """计算最大回撤 (Maximum Drawdown)"""
    if nav_series.empty:
        return 0.0
    # 计算累计最大值 (Previous Peak)
    rolling_max = nav_series.expanding().max()
    # 计算回撤
    drawdown = (nav_series / rolling_max) - 1.0
    return abs(drawdown.min())

def calculate_sharpe_ratio(return_series, risk_free_rate_daily):
    """计算年化夏普比率 (假设 252 个交易日)"""
    if return_series.empty or len(return_series) < 2:
        return np.nan
    
    # 转换为日收益率 (假设 index_data['Strategy_Return'] 已经是日收益率)
    daily_returns = return_series.dropna()
    
    # 计算超额收益
    excess_returns = daily_returns - risk_free_rate_daily
    
    # 年化夏普比率
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return np.nan # 波动率为零时，夏普比率无意义

    # 年化
    return (mean_excess_return / std_excess_return) * np.sqrt(252)

# --- 信号计算函数 (已根据用户要求调整阈值) ---

def calculate_technical_indicators_for_day(df):
    """
    计算关键技术指标 (RSI, MACD, MA50) 的历史序列。
    DataFrame 必须按日期升序 (ASC) 排列。
    """
    if 'net_value' not in df.columns or len(df) < 50:
        df['RSI'] = np.nan
        df['MACD'] = np.nan
        df['MACD_Signal'] = np.nan
        df['NAV_MA50'] = np.nan
        return df

    # 1. RSI (14 days)
    delta = df['net_value'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # 使用 EWM 平滑计算 RSI (更贴近真实应用)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9)
    ema_12 = df['net_value'].ewm(span=12, adjust=False).mean()
    ema_26 = df['net_value'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3. MA50
    df['MA50'] = df['net_value'].rolling(window=50).mean()
    df['NAV_MA50'] = df['net_value'] / df['MA50']
    
    return df

def generate_action_signal(row):
    """
    根据调整后的逻辑生成行动信号。
    信号：强买入/弱买入/持有观察/强卖出规避
    """
    rsi = row['RSI']
    macd = row.get('MACD')
    signal = row.get('MACD_Signal')
    prev_macd = row.get('Prev_MACD', -99)
    prev_signal = row.get('Prev_Signal', -99)
    nav_ma50 = row.get('NAV_MA50')
    
    if pd.isna(rsi) or pd.isna(nav_ma50) or pd.isna(macd) or pd.isna(signal):
        return '持有/观察'
        
    # --- 信号规则 (已调整阈值) ---
    
    # 强卖出/规避：趋势严重恶化或过度超买
    if nav_ma50 < 0.95 or rsi > 75: 
        return '强卖出/规避'
    
    # 强买入：深度超卖且接近底部
    # RSI < 30 (原 25/30) + 价格低于长期均线 + MACD金叉
    if rsi < 30 and nav_ma50 < 1.00:
        # MACD金叉
        if macd > signal and prev_macd < prev_signal:
             return '强买入'
        # 深度超卖（无需金叉）
        if rsi < 25:
             return '强买入'
    
    # 弱买入：超卖或 MACD 转向
    # RSI < 40 (原 30) + MACD 金叉 (修正：使用金叉作为独立信号)
    if rsi < 40:
        return '弱买入'
    
    # MACD金叉作为弱买入补充信号
    if macd > signal and prev_macd < prev_signal:
        return '弱买入'

    return '持有/观察'

# --- 核心指数构建类 ---

class IndexBuilder:
    def __init__(self, fund_data_dir=FUND_DATA_DIR, index_data_dir=INDEX_DATA_DIR, index_name=INDEX_NAME, starting_nav=STARTING_NAV):
        self.fund_data_dir = fund_data_dir
        self.index_data_dir = index_data_dir
        self.index_name = index_name
        self.starting_nav = starting_nav
        self.all_data = {}
        self.csi300_data = None
        self.common_dates = None
        self.risk_free_rate_daily = RISK_FREE_RATE_DAILY

    def _get_csi300_data(self):
        """从 index_data 目录加载沪深300指数数据。"""
        csi300_file = os.path.join(self.index_data_dir, CSI300_FILENAME)
        if not os.path.exists(csi300_file):
            logger.warning(f"警告：未找到沪深300数据文件 '{csi300_file}'。指数对比功能将被禁用。")
            return None
        
        try:
            df = pd.read_csv(csi300_file)
            df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df.dropna(subset=['net_value', 'date'], inplace=True)
            df = df.sort_values(by='date', ascending=True).set_index('date')
            
            logger.info(f"沪深300数据加载成功 (数据点: {len(df)})。")
            return df
        except Exception as e:
            logger.error(f"加载沪深300数据时发生错误: {e}")
            return None


    def load_and_preprocess_data(self):
        """加载所有基金和基准指数数据，计算指标，并查找公共日期。"""
        # 1. 检查目录是否存在
        if not os.path.exists(self.fund_data_dir):
            logger.error(f"错误: 基金数据目录 '{self.fund_data_dir}' 不存在。")
            return False
            
        # 2. 加载沪深300数据 (从 INDEX_DATA_DIR)
        self.csi300_data = self._get_csi300_data()
        
        # 3. 加载基金数据 (从 FUND_DATA_DIR)
        csv_files = glob.glob(os.path.join(self.fund_data_dir, '*.csv'))
        
        all_dates_indices = []
        if self.csi300_data is not None:
             all_dates_indices.append(self.csi300_data.index)
        
        for filepath in csv_files:
            fund_code = os.path.splitext(os.path.basename(filepath))[0]
            try:
                # 假设文件只有 date, net_value 两列，并确保加载到足够的历史数据
                df = pd.read_csv(filepath)
                df = df.rename(columns={'net_value': 'net_value', 'date': 'date'}) 
                df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df.dropna(subset=['net_value', 'date'], inplace=True)
                df = df.sort_values(by='date', ascending=True).set_index('date')
                
                # 历史计算技术指标
                df = calculate_technical_indicators_for_day(df.copy())
                # 用于MACD金叉判断
                df['Prev_MACD'] = df['MACD'].shift(1)
                df['Prev_Signal'] = df['MACD_Signal'].shift(1)
                
                self.all_data[fund_code] = df
                all_dates_indices.append(df.index)
            except Exception as e:
                logger.warning(f"处理基金文件 {filepath} 时发生错误: {e}")
                continue

        if not self.all_data:
            logger.error("没有成功加载任何基金数据。")
            return False

        # 4. 确定公共日期范围
        if not all_dates_indices:
            logger.error("无法确定日期范围，all_dates_indices为空。")
            return False

        # 使用 union 方法合并所有 DatetimeIndex 对象
        full_index = all_dates_indices[0]
        for index in all_dates_indices[1:]:
            full_index = full_index.union(index)
            
        full_date_range = full_index
        
        # 确定最早的可计算指标的日期（通常是所有基金中最晚的开始日期）
        min_start_date = max(df.index.min() for df in self.all_data.values())
        if self.csi300_data is not None:
            min_start_date = max(min_start_date, self.csi300_data.index.min())

        # 确定最晚的结束日期
        max_end_date = min(df.index.max() for df in self.all_data.values())
        if self.csi300_data is not None:
             max_end_date = min(max_end_date, self.csi300_data.index.max())

        # 限制公共日期范围
        self.common_dates = full_date_range[
            (full_date_range >= min_start_date) & 
            (full_date_range <= max_end_date)
        ].sort_values()
        
        # 剔除无法计算指标的初期数据 (例如：需要50天数据才能计算MA50)
        self.common_dates = self.common_dates[self.common_dates >= self.common_dates.min() + pd.Timedelta(days=50)]

        if len(self.common_dates) < 50:
             logger.error(f"警告：公共数据日期少于50天 (找到 {len(self.common_dates)} 天)。指数可能不稳定。")
             return False # 至少需要50天数据
        
        logger.info(f"数据预处理完成。公共日期范围: {self.common_dates.min().strftime('%Y-%m-%d')} - {self.common_dates.max().strftime('%Y-%m-%d')}")
        return True

    def build_index(self):
        """
        计算策略指数和基准指数的每日净值 (NAV)。
        策略改进：当无买入信号时，保持前一交易日的持仓组合，而不是收益为 0。
        """
        
        index_data = pd.DataFrame(index=self.common_dates)
        
        index_nav = [self.starting_nav]
        csi300_nav = [self.starting_nav]
        
        # 当前持仓基金代码列表 (用于“持仓等待新信号”的逻辑)
        current_holdings = [] 
        
        # 预先计算所有基金和沪深300的日收益率
        daily_returns = {}
        for code, df in self.all_data.items():
            daily_returns[code] = df['net_value'].pct_change()
        
        csi300_returns = self.csi300_data['net_value'].pct_change() if self.csi300_data is not None else None
        
        # 从第二个日期开始计算指数
        for i, date in enumerate(self.common_dates):
            if i == 0:
                index_data.loc[date, 'Strategy_Return'] = 0.0
                index_data.loc[date, 'CSI300_Return'] = 0.0
                index_data.loc[date, 'Signal_Funds_Count'] = 0
                continue
                
            prev_date = self.common_dates[i-1]
            buy_signal_codes = []
            
            # 1. 检查前一日是否出现新的买入信号 (Rebalance/Switch day)
            for code, df in self.all_data.items():
                if prev_date in df.index:
                    signal = generate_action_signal(df.loc[prev_date])
                    
                    if signal in ['强买入', '弱买入']:
                        buy_signal_codes.append(code)

            strategy_return = 0.0
            
            if buy_signal_codes:
                # 2a. 出现新信号：清仓旧持仓，买入新的信号组合 (换仓/再平衡)
                current_holdings = buy_signal_codes
                signal_count = len(current_holdings)
                
                # 计算当日收益：新组合的日收益率平均值
                holdings_returns = []
                for code in current_holdings:
                    if date in daily_returns[code].index and not pd.isna(daily_returns[code].loc[date]):
                        holdings_returns.append(daily_returns[code].loc[date])
                
                if holdings_returns:
                    strategy_return = np.mean(holdings_returns)
                else:
                    # 极端情况：新信号基金当日无数据，按无风险利率计算
                    strategy_return = self.risk_free_rate_daily
                    
            elif current_holdings:
                # 2b. 无新信号，但有持仓：保持前日的持仓组合 (持仓等待)
                signal_count = len(current_holdings)
                
                # 计算当日收益：旧组合的日收益率平均值
                holdings_returns = []
                for code in current_holdings:
                    if date in daily_returns[code].index and not pd.isna(daily_returns[code].loc[date]):
                        holdings_returns.append(daily_returns[code].loc[date])
                        
                if holdings_returns:
                    strategy_return = np.mean(holdings_returns)
                else:
                    # 极端情况：所有持仓基金当日无数据/停牌，按无风险利率计算
                    strategy_return = self.risk_free_rate_daily
                    
            else:
                # 2c. 无新信号，且无持仓 (仅发生在数据期初或极端清仓后)
                signal_count = 0
                strategy_return = self.risk_free_rate_daily # 按无风险利率计算

            # 基准指数计算 (CSI300)
            if csi300_returns is not None and date in csi300_returns.index and not pd.isna(csi300_returns.loc[date]):
                csi300_return = csi300_returns.loc[date]
            else:
                csi300_return = 0.0

            # 更新策略指数 NAV
            prev_strategy_nav = index_nav[-1]
            current_strategy_nav = prev_strategy_nav * (1 + strategy_return)
            index_nav.append(current_strategy_nav)
            
            # 更新基准指数 NAV
            prev_csi300_nav = csi300_nav[-1]
            current_csi300_nav = prev_csi300_nav * (1 + csi300_return)
            csi300_nav.append(current_csi300_nav)

            # 记录数据
            index_data.loc[date, 'Strategy_Return'] = strategy_return
            index_data.loc[date, 'CSI300_Return'] = csi300_return
            index_data.loc[date, 'Signal_Funds_Count'] = signal_count # 记录的是当日持仓的基金数
            # index_data.loc[date, 'Holding_Codes'] = ','.join(current_holdings) # 可选：记录持仓代码

        # 最终数据整理
        index_data['Strategy_NAV'] = index_nav
        index_data['CSI300_NAV'] = csi300_nav
        index_data.index.name = 'Date'
        
        # 计算核心绩效指标
        strategy_mdd = calculate_mdd(index_data['Strategy_NAV'])
        csi300_mdd = calculate_mdd(index_data['CSI300_NAV'])
        
        strategy_sharpe = calculate_sharpe_ratio(index_data['Strategy_Return'], self.risk_free_rate_daily)
        csi300_sharpe = calculate_sharpe_ratio(index_data['CSI300_Return'], self.risk_free_rate_daily)

        return index_data, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe

    def generate_report(self, index_df, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe):
        """生成 Markdown 报告，增强量化指标输出。"""
        now = datetime.now()
        start_date = index_df.index.min().strftime('%Y-%m-%d')
        end_date = index_df.index.max().strftime('%Y-%m-%d')
        
        # 总结数据
        strategy_nav_end = index_df['Strategy_NAV'].iloc[-1]
        csi300_nav_end = index_df['CSI300_NAV'].iloc[-1]
        
        total_return_strategy = (strategy_nav_end / self.starting_nav) - 1
        total_return_csi300 = (csi300_nav_end / self.starting_nav) - 1
        
        excess_return = total_return_strategy - total_return_csi300
        
        report = f"# 量化策略指数报告 - {self.index_name}\n\n"
        report += f"生成日期: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"数据周期: {start_date} 至 {end_date} (共 {len(index_df)} 个交易日)\n"
        report += f"策略逻辑: 仅在出现'强买入'或'弱买入'信号时换仓，等权持有信号基金，否则保持现有持仓。\n"
        report += f"无持仓时按年化 {RISK_FREE_RATE_ANNUAL:.1%} (每日 {RISK_FREE_RATE_DAILY:.4f}) 计算收益。\n\n"
        
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
        
        # 显示最新 60 天
        display_df = index_df.tail(60).copy()
        display_df['Strategy_NAV'] = display_df['Strategy_NAV'].apply(lambda x: f"{x:.4f}")
        display_df['CSI300_NAV'] = display_df['CSI300_NAV'].apply(lambda x: f"{x:.4f}")
        display_df['Signal_Funds_Count'] = display_df['Signal_Funds_Count'].astype(int)
        
        display_df = display_df.rename(columns={
            'Strategy_NAV': '策略指数净值',
            'CSI300_NAV': '沪深300净值',
            'Strategy_Return': '策略日收益',
            'CSI300_Return': '300日收益',
            'Signal_Funds_Count': '持仓基金数'
        })
        
        # 格式化收益率 (只在输出时格式化)
        display_df['策略日收益'] = display_df['策略日收益'].apply(lambda x: f"{x * 100:.2%}")
        display_df['300日收益'] = display_df['300日收益'].apply(lambda x: f"{x * 100:.2%}")

        # 写入文件
        timestamp_for_filename = now.strftime('%Y%m%d_%H%M%S')
        DIR_NAME = now.strftime('%Y%m')
        os.makedirs(DIR_NAME, exist_ok=True)
        
        REPORT_FILE = os.path.join(DIR_NAME, f"{INDEX_REPORT_BASE_NAME}_{timestamp_for_filename}.md")
        
        # 使用 to_markdown (需要 tabulate 库)
        markdown_table = display_df[['策略指数净值', '沪深300净值', '策略日收益', '300日收益', '持仓基金数']].to_markdown(index=True)
        
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write(markdown_table)

        logger.info(f"指数报告已保存到 {REPORT_FILE}")
        return REPORT_FILE

    def run(self):
        """主执行流程。"""
        logger.info("开始执行量化策略指数构建...")
            
        if not self.load_and_preprocess_data():
            logger.warning("数据加载失败或数据量不足，停止指数构建。")
            return None
        
        # 获取增强后的回测结果
        result = self.build_index()
        
        if result is not None:
            index_df, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe = result
            self.generate_report(index_df, strategy_mdd, csi300_mdd, strategy_sharpe, csi300_sharpe)


if __name__ == '__main__':
    builder = IndexBuilder()
    builder.run()
