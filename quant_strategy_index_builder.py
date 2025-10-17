import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

# --- 配置参数 ---
# 明确区分基金数据目录和指数数据目录
FUND_DATA_DIR = 'fund_data'
INDEX_DATA_DIR = 'index_data'
INDEX_REPORT_BASE_NAME = 'quant_strategy_index_report'
INDEX_NAME = 'MarketMonitor_BuySignal_Index'
CSI300_CODE = '000300' 
CSI300_FILENAME = f'{CSI300_CODE}.csv' 
STARTING_NAV = 1000

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 关键技术指标计算函数 (与 market_monitor.py 保持一致) ---

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
    根据 market_monitor.py 的逻辑 (假设的简化规则) 生成行动信号。
    信号：强买入/弱买入/持有观察/强卖出规避
    """
    rsi = row['RSI']
    macd = row.get('MACD')
    signal = row.get('MACD_Signal')
    nav_ma50 = row.get('NAV_MA50')
    
    if pd.isna(rsi) or pd.isna(nav_ma50):
        return '持有/观察'
        
    # --- 信号规则 ---
    if rsi > 70 or nav_ma50 > 1.2:
        return '强卖出/规避'
    
    if rsi < 25 and nav_ma50 < 1.00:
        return '强买入'
    
    if rsi < 30:
        return '弱买入'
    
    if not pd.isna(macd) and not pd.isna(signal):
        # MACD金叉
        if macd > signal and row.get('Prev_MACD', -99) < row.get('Prev_Signal', -99):
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

    def _get_csi300_data(self):
        """
        从 index_data 目录加载沪深300指数数据。
        """
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
        # 1. 加载沪深300数据 (从 INDEX_DATA_DIR)
        self.csi300_data = self._get_csi300_data()
        
        # 2. 加载基金数据 (从 FUND_DATA_DIR)
        csv_files = glob.glob(os.path.join(self.fund_data_dir, '*.csv'))
        
        all_dates_indices = []
        if self.csi300_data is not None:
             all_dates_indices.append(self.csi300_data.index)
        
        for filepath in csv_files:
            fund_code = os.path.splitext(os.path.basename(filepath))[0]
            try:
                df = pd.read_csv(filepath)
                df = df.rename(columns={'net_value': 'net_value', 'date': 'date'}) # 保持列名统一
                df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df.dropna(subset=['net_value', 'date'], inplace=True)
                df = df.sort_values(by='date', ascending=True).set_index('date')
                
                # 历史计算技术指标
                df = calculate_technical_indicators_for_day(df.copy())
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

        # 3. 确定公共日期范围
        full_date_range = pd.to_datetime(pd.concat(all_dates_indices).unique())
        
        # 确定最晚的起始日期和最早的结束日期
        min_start_date = max(df.index.min() for df in self.all_data.values())
        if self.csi300_data is not None:
            min_start_date = max(min_start_date, self.csi300_data.index.min())

        max_end_date = min(df.index.max() for df in self.all_data.values())
        if self.csi300_data is not None:
             max_end_date = min(max_end_date, self.csi300_data.index.max())

        
        self.common_dates = full_date_range[
            (full_date_range >= min_start_date) & 
            (full_date_range <= max_end_date)
        ].sort_values()

        if len(self.common_dates) < 50:
             logger.error(f"警告：公共数据日期少于50天 (找到 {len(self.common_dates)} 天)。指数可能不稳定。")
        
        logger.info(f"数据预处理完成。公共日期范围: {self.common_dates.min().strftime('%Y-%m-%d')} - {self.common_dates.max().strftime('%Y-%m-%d')}")
        return True

    def build_index(self):
        """计算策略指数和基准指数的每日净值 (NAV)。"""
        
        index_data = pd.DataFrame(index=self.common_dates)
        
        index_nav = [self.starting_nav]
        csi300_nav = [self.starting_nav]

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
            buy_signal_returns = []
            
            # 策略指数计算 (Strategy Index)
            for code, df in self.all_data.items():
                if prev_date in df.index:
                    signal = generate_action_signal(df.loc[prev_date])
                    
                    if signal in ['强买入', '弱买入']:
                        if date in daily_returns[code].index and not pd.isna(daily_returns[code].loc[date]):
                             buy_signal_returns.append(daily_returns[code].loc[date])

            if buy_signal_returns:
                strategy_return = np.mean(buy_signal_returns)
                signal_count = len(buy_signal_returns)
            else:
                strategy_return = 0.0 # 无信号基金时，当日收益为 0 (持币)
                signal_count = 0
            
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
            index_data.loc[date, 'Signal_Funds_Count'] = signal_count

        # 最终数据整理
        index_data['Strategy_NAV'] = index_nav
        index_data['CSI300_NAV'] = csi300_nav
        index_data.index.name = 'Date'
        
        return index_data

    def generate_report(self, index_df):
        """生成 Markdown 报告，输出到年月目录中。"""
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
        report += f"数据周期: {start_date} 至 {end_date} (共 {len(index_df)} 个交易日)\n\n"
        
        report += f"## **策略指数表现总结**\n"
        report += f"**指数名称:** {self.index_name} (基于 market_monitor.py 强/弱买入信号等权)\n"
        report += f"**起始净值:** {self.starting_nav:.0f}\n"
        report += f"| 指数 | 最终净值 | 总回报率 | 超额收益 (对比沪深300) |\n"
        report += f"| :--- | ---: | ---: | :---: |\n"
        report += f"| **{self.index_name}** | **{strategy_nav_end:.4f}** | **{total_return_strategy:.2%}** | **{excess_return:.2%}** |\n"
        report += f"| **沪深300 (基准)** | {csi300_nav_end:.4f} | {total_return_csi300:.2%} | - |\n\n"
        
        report += "## 指数净值历史走势 (最新 60 天)\n\n"
        report += "**注：** 指数假设在信号日后的第一个交易日等权买入所有信号基金。无信号时持币 (收益为 0)。\n\n"
        
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
            'Signal_Funds_Count': '信号基金数'
        })
        
        # 格式化收益率 (只在输出时格式化)
        display_df['策略日收益'] = display_df['策略日收益'].apply(lambda x: f"{x * 100:.2%}")
        display_df['300日收益'] = display_df['300日收益'].apply(lambda x: f"{x * 100:.2%}")

        # 写入文件
        timestamp_for_filename = now.strftime('%Y%m%d_%H%M%S')
        DIR_NAME = now.strftime('%Y%m')
        os.makedirs(DIR_NAME, exist_ok=True)
        
        REPORT_FILE = os.path.join(DIR_NAME, f"{INDEX_REPORT_BASE_NAME}_{timestamp_for_filename}.md")
        
        markdown_table = display_df[['策略指数净值', '沪深300净值', '策略日收益', '300日收益', '信号基金数']].to_markdown(index=True)
        
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write(markdown_table)

        logger.info(f"指数报告已保存到 {REPORT_FILE}")
        return REPORT_FILE

    def run(self):
        """主执行流程。"""
        logger.info("开始执行量化策略指数构建...")
        
        # 检查数据目录是否存在
        if not os.path.exists(self.fund_data_dir):
            logger.error(f"错误: 基金数据目录 '{self.fund_data_dir}' 不存在。")
            return None
        if not os.path.exists(self.index_data_dir):
            logger.error(f"错误: 指数数据目录 '{self.index_data_dir}' 不存在。")
            return None
            
        if not self.load_and_preprocess_data():
            logger.warning("数据加载失败或数据量不足，停止指数构建。")
            return None
        
        index_df = self.build_index()
        
        if index_df is not None:
            self.generate_report(index_df)


if __name__ == '__main__':
    builder = IndexBuilder()
    builder.run()
