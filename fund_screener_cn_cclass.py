import pandas as pd
import numpy as np
import akshare as ak
import requests
import re
from datetime import datetime, timedelta
import yfinance as yf
import asyncio
import aiohttp
import logging
import random
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', filename='cn_fund_screener.log')

def randHeader() -> Dict[str, str]:
    """随机生成User-Agent，防反爬"""
    head_user_agent = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    ]
    return {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'User-Agent': random.choice(head_user_agent),
        'Referer': 'http://fund.eastmoney.com/'
    }

async def get_url_async(session: aiohttp.ClientSession, url: str, tries_num: int = 5) -> Optional[str]:
    """异步获取URL内容，带重试和更细致的异常处理"""
    for i in range(tries_num):
        try:
            async with session.get(url, headers=randHeader(), timeout=15) as response:
                response.raise_for_status()
                return await response.text(encoding='utf-8')
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            await asyncio.sleep(2 ** i + random.uniform(0, 1))
            logging.warning(f"请求 {url} 失败，第 {i+1} 次重试: {e}")
        except Exception as e:
            logging.error(f"请求 {url} 发生未知错误: {e}")
            break
    logging.error(f"请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_list() -> pd.DataFrame:
    """步骤1: 基础过滤 - 获取C类基金，规模>5亿，费用<0.8%"""
    try:
        logging.info("开始获取所有开放式基金列表...")
        fund_info = ak.fund_em_open_fund_info(fund="全部", indicator="基本信息")
        
        if '基金代码' not in fund_info.columns:
            logging.error("基金数据缺少'基金代码'列。")
            return pd.DataFrame()
        
        # 数据清洗和过滤
        fund_info['基金规模_亿'] = fund_info['基金规模'].str.replace('亿元', '').astype(float)
        fund_info['总费率'] = fund_info['管理费率'].str.replace('%', '').astype(float) / 100 + \
                            fund_info['托管费率'].str.replace('%', '').astype(float) / 100
        
        filtered_funds = fund_info[
            (fund_info['基金代码'].str.len() == 6) &
            (fund_info['基金简称'].str.contains('C$|C类', regex=True, na=False)) &
            (~fund_info['类型'].str.contains('ETF|LOF|场内|QDII', na=False, regex=True)) &
            (fund_info['基金规模_亿'] > 5) &
            (fund_info['总费率'] < 0.008)
        ].copy()

        logging.info(f"基础过滤后剩余基金数量: {len(filtered_funds)}")
        return filtered_funds[['基金代码', '基金简称', '类型']]

    except Exception as e:
        logging.error(f"获取基金列表失败: {e}")
        return pd.DataFrame()

async def get_all_fund_data(fund_codes: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """一次性获取所有基金的历史排名和每日净值数据"""
    logging.info("开始获取历史排名和每日净值数据...")
    data = {}
    
    try:
        rank_df = ak.fund_open_fund_rank_em()
        data['rankings'] = rank_df[rank_df['基金代码'].isin(fund_codes)]
        logging.info(f"获取到 {len(data['rankings'])} 只基金的排名数据。")
    except Exception as e:
        logging.error(f"获取基金排名数据失败: {e}")
        data['rankings'] = pd.DataFrame()

    try:
        daily_nav_df = ak.fund_em_nav_history(fund_code="全部", start_date=start_date, end_date=end_date)
        daily_nav_df['基金代码'] = daily_nav_df['基金代码'].astype(str)
        data['daily_nav'] = daily_nav_df[daily_nav_df['基金代码'].isin(fund_codes)].copy()
        logging.info(f"获取到 {len(data['daily_nav']['基金代码'].unique())} 只基金的每日净值数据。")
    except Exception as e:
        logging.error(f"获取每日净值数据失败: {e}")
        data['daily_nav'] = pd.DataFrame()

    return data

def filter_by_rankings(df: pd.DataFrame, rank_df: pd.DataFrame) -> pd.DataFrame:
    """根据历史排名进行筛选"""
    if rank_df.empty:
        logging.warning("排名数据为空，跳过排名筛选。")
        return df.copy()

    # 将百分比排名转化为相对排名
    rank_df['rank_r(3y)'] = rank_df['近3年排名'].str.split('/').str[0].astype(float) / rank_df['近3年排名'].str.split('/').str[1].astype(float)
    rank_df['rank_r(1y)'] = rank_df['近1年排名'].str.split('/').str[0].astype(float) / rank_df['近1年排名'].str.split('/').str[1].astype(float)
    rank_df['rank_r(6m)'] = rank_df['近6月排名'].str.split('/').str[0].astype(float) / rank_df['近6月排名'].str.split('/').str[1].astype(float)
    rank_df['rank_r(3m)'] = rank_df['近3月排名'].str.split('/').str[0].astype(float) / rank_df['近3月排名'].str.split('/').str[1].astype(float)

    ranked_filtered = rank_df[
        (rank_df['rank_r(3y)'] <= 0.3) &
        (rank_df['rank_r(1y)'] <= 0.3) &
        (rank_df['rank_r(6m)'] <= 0.4) &
        (rank_df['rank_r(3m)'] <= 0.4)
    ]
    
    filtered_codes = ranked_filtered['基金代码'].tolist()
    logging.info(f"排名筛选后剩余 {len(filtered_codes)} 只基金。")
    return df[df['基金代码'].isin(filtered_codes)].copy()

def calculate_metrics(fund_codes: List[str], daily_nav_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """步骤2: 风险评估 - 计算Sharpe、回撤、换手率"""
    results = []
    
    for code in fund_codes:
        df = daily_nav_df[daily_nav_df['基金代码'] == code].copy()
        
        if df.empty or len(df) < 2:
            logging.warning(f"基金 {code} 净值数据不足，跳过计算。")
            continue
            
        try:
            df['净值日期'] = pd.to_datetime(df['净值日期'])
            df = df.sort_values('净值日期')
            returns = df['单位净值'].pct_change().dropna()
            
            # Sharpe比率
            annual_return = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.03) / volatility if volatility != 0 else 0
            
            # 最大回撤
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 换手率（假设，需替换为真实数据源）
            turnover = np.random.uniform(0.3, 0.6)
            
            results.append({
                'fund_code': code,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'turnover': turnover,
                'annual_return': annual_return
            })
        except Exception as e:
            logging.warning(f"计算基金 {code} 指标失败: {e}")
            continue

    return pd.DataFrame(results)

async def get_holdings_async(session: aiohttp.ClientSession, fund_code: str) -> Dict[str, Any]:
    """异步获取持仓，检查科技、消费等行业占比"""
    url = f'http://fundf10.eastmoney.com/ccmx_{fund_code}.html'
    try:
        response_text = await get_url_async(session, url)
        if not response_text:
            return {'fund_code': fund_code, 'tech_ratio': 0, 'consumer_ratio': 0, 'medical_ratio': 0}
        
        df = pd.read_html(response_text, header=0)[0]
        df.columns = ['排名', '股票代码', '股票名称', '持仓占比', '涨跌幅', '持仓市值']
        
        tech_keywords = ['科技', '信息', '半导体', '软件', '互联网']
        consumer_keywords = ['消费', '食品饮料', '家用电器', '医药生物']
        medical_keywords = ['医药', '医疗']
        
        tech_ratio = df[df['股票名称'].str.contains('|'.join(tech_keywords), na=False)]['持仓占比'].sum()
        consumer_ratio = df[df['股票名称'].str.contains('|'.join(consumer_keywords), na=False)]['持仓占比'].sum()
        medical_ratio = df[df['股票名称'].str.contains('|'.join(medical_keywords), na=False)]['持仓占比'].sum()

        return {
            'fund_code': fund_code,
            'tech_ratio': tech_ratio,
            'consumer_ratio': consumer_ratio,
            'medical_ratio': medical_ratio
        }
    except Exception as e:
        logging.warning(f"获取 {fund_code} 持仓失败: {e}")
        return {'fund_code': fund_code, 'tech_ratio': 0, 'consumer_ratio': 0, 'medical_ratio': 0}

async def momentum_optimize(metrics_df: pd.DataFrame, daily_nav_df: pd.DataFrame, session: aiohttp.ClientSession) -> pd.DataFrame:
    """步骤3: 前瞻优化 - 6月动量 + 持仓检查"""
    optimized = metrics_df[(metrics_df['sharpe'] > 0.8) & 
                           (metrics_df['max_drawdown'] > -0.25) & 
                           (metrics_df['turnover'] < 0.6)].copy()
    
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # 异步计算6个月动量
    for idx, row in optimized.iterrows():
        code = row['fund_code']
        df = daily_nav_df[daily_nav_df['基金代码'] == code].copy()
        
        if df.empty or len(df) < 2:
            optimized.at[idx, 'momentum_6m'] = 0
            continue
            
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df[df['净值日期'] >= six_months_ago].sort_values('净值日期')
        
        if len(df) > 1:
            momentum = (df['单位净值'].iloc[-1] / df['单位净值'].iloc[0] - 1)
            optimized.at[idx, 'momentum_6m'] = momentum
        else:
            optimized.at[idx, 'momentum_6m'] = 0

    # 异步获取持仓
    tasks = [get_holdings_async(session, code) for code in optimized['fund_code'].tolist()]
    holdings_results = await asyncio.gather(*tasks)
    holdings_df = pd.DataFrame(holdings_results).set_index('fund_code')
    optimized = optimized.join(holdings_df)
    
    # 过滤动量和持仓
    optimized = optimized[
        (optimized['momentum_6m'] > 0.03) & 
        (optimized['momentum_6m'] < 0.12) &
        (optimized['tech_ratio'] + optimized['consumer_ratio'] > 30)
    ]
    
    # 综合评分，引入新的持仓权重
    optimized['composite_score'] = (
        optimized['sharpe'] * 0.4 + 
        optimized['momentum_6m'] * 0.3 + 
        (optimized['tech_ratio'] + optimized['consumer_ratio'] + optimized['medical_ratio']) / 100 * 0.3
    )

    return optimized.sort_values('composite_score', ascending=False).head(5)

async def main():
    """主函数：筛选中国场外C类基金"""
    logging.info("开始筛选场外C类基金...")
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    fund_info = get_fund_list()
    if fund_info.empty:
        logging.error("无符合条件的基金。")
        return
    fund_codes = fund_info['基金代码'].tolist()
    
    # 一次性获取所有所需数据
    async with aiohttp.ClientSession() as session:
        all_data = await get_all_fund_data(fund_codes, start_date, end_date)
    
    # 步骤2: 历史排名筛选
    filtered_funds_after_rank = filter_by_rankings(fund_info, all_data.get('rankings', pd.DataFrame()))
    if filtered_funds_after_rank.empty:
        logging.info("无基金通过排名筛选，程序结束。")
        return
    fund_codes_after_rank = filtered_funds_after_rank['基金代码'].tolist()
    
    # 步骤3: 风险评估
    metrics_df = calculate_metrics(fund_codes_after_rank, all_data.get('daily_nav', pd.DataFrame()), start_date, end_date)
    if metrics_df.empty:
        logging.error("风险评估无数据。")
        return
    
    # 步骤4: 前瞻优化
    async with aiohttp.ClientSession() as session:
        recommended = await momentum_optimize(metrics_df, all_data.get('daily_nav', pd.DataFrame()), session)
    
    if recommended.empty:
        logging.info("无基金通过前瞻优化，没有推荐结果。")
        return

    # 输出
    final_output = recommended.merge(fund_info[['基金代码', '基金简称']], on='基金代码', how='left')
    output_path = 'recommended_cn_cclass_funds_2025.csv'
    final_output.to_csv(output_path, encoding='gbk', index=False)
    
    logging.info(f"推荐基金保存至 {output_path}，共 {len(final_output)} 只")
    print("推荐基金：")
    print(final_output[['基金代码', '基金简称', 'sharpe', 'momentum_6m', 'composite_score']])

if __name__ == '__main__':
    asyncio.run(main())
