import os
import json
import time
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import yfinance as yf
import akshare as ak
import random
from datetime import datetime
import traceback
import csv

# --- 辅助函数：从 integrated_fund_screener.py 整合而来 ---
def randHeader():
    head_user_agent = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
    ]
    return {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'User-Agent': random.choice(head_user_agent),
        'Referer': 'http://fund.eastmoney.com/'
    }

def getURL(url, tries_num=5, sleep_time=1, time_out=10, proxies=None):
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.5, sleep_time))
            res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            res.raise_for_status()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 5)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_basic_info():
    print("开始获取基金基本信息...")
    try:
        url = 'http://fund.eastmoney.com/js/fundcode_search.js'
        response = getURL(url)
        if response:
            text = re.findall(r'"(\d*?)","(.*?)","(.*?)","(.*?)","(.*?)"', response.text)
            fund_codes = [item[0] for item in text]
            fund_names = [item[2] for item in text]
            fund_types = [item[3] for item in text]
        else:
            print("天天基金基本信息获取失败，尝试 akshare...")
            fund_info = ak.fund_open_fund_info_em()
            fund_codes = fund_info['基金代码'].tolist()
            fund_names = fund_info['基金简称'].tolist()
            fund_types = fund_info['类型'].tolist()
    except Exception as e:
        print(f"获取基金基本信息失败: {e}")
        fund_info = ak.fund_open_fund_info_em()
        fund_codes = fund_info['基金代码'].tolist()
        fund_names = fund_info['基金简称'].tolist()
        fund_types = fund_info['类型'].tolist()
    
    fund_info_df = pd.DataFrame({
        '代码': fund_codes,
        '名称': fund_names,
        '类型': fund_types
    })
    
    codes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_fund_codes.csv')
    fund_codes_df = pd.DataFrame({'基金代码': fund_codes})
    fund_codes_df.to_csv(codes_path, index=False, encoding='utf-8')
    print(f"全量基金代码列表已保存至 '{codes_path}'（{len(fund_codes)} 只基金）")
    print("基金基本信息获取完成。")
    return fund_info_df

def get_fund_rankings(fund_type='hh', proxies=None):
    print("开始获取基金排名...")
    periods = {
        '3y': '近3年', '2y': '近2年', '1y': '近1年',
        '6m': '近6月', '3m': '近3月'
    }
    
    try:
        full_rank_df = ak.fund_open_fund_rank_em()
        if not full_rank_df.empty:
            print("成功使用 akshare 获取全量基金排名数据。")
            full_rank_df['基金代码'] = full_rank_df['基金代码'].astype(str)
            full_rank_df.set_index('基金代码', inplace=True)
            df_final = pd.DataFrame()
            
            total_records = len(full_rank_df)
            
            for period, col_name in periods.items():
                if col_name in full_rank_df.columns:
                    period_df = full_rank_df[['基金简称', col_name]].copy()
                    period_df.rename(columns={'基金简称': 'name', col_name: f'rose({period})'}, inplace=True)
                    period_df[f'rose({period})'] = pd.to_numeric(period_df[f'rose({period})'], errors='coerce') / 100
                    period_df[f'rank({period})'] = period_df[f'rose({period})'].rank(method='min', ascending=False)
                    period_df[f'rank_r({period})'] = period_df[f'rank({period})'] / total_records
                    
                    if df_final.empty:
                        df_final = period_df
                    else:
                        df_final = df_final.join(period_df.drop('name', axis=1, errors='ignore'), how='outer')
            
            if not df_final.empty:
                df_final.to_csv('fund_rankings.csv', encoding='gbk')
                print(f"排名数据已保存至 'fund_rankings.csv'")
                print("基金排名获取完成。")
                return df_final
    except Exception as e:
        print(f"akshare 获取排名失败: {e}")
    
    print("排名数据获取失败，将返回空 DataFrame。")
    return pd.DataFrame()

def apply_4433_rule(df, total_records):
    print("正在应用四四三三法则进行筛选...")
    thresholds = {
        '3y': 0.25, '2y': 0.25, '1y': 0.25,
        '6m': 1/3, '3m': 1/3
    }
    filtered_df = df.copy()
    for period in thresholds:
        rank_col = f'rank_r({period})'
        if rank_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[rank_col] <= thresholds[period]]
    print(f"四四三三法则筛选出 {len(filtered_df)} 只基金")
    return filtered_df

def get_fund_details(code, proxies=None):
    print(f"开始获取基金 {code} 详情...")
    try:
        url = f'http://fund.eastmoney.com/f10/{code}.html'
        tables = pd.read_html(url)
        if len(tables) < 2:
            raise ValueError("表格数量不足")
        df = tables[1]
        df1 = df[[0, 1]].set_index(0).T
        df2 = df[[2, 3]].set_index(2).T
        df1['code'] = code
        df2['code'] = code
        df1.set_index('code', inplace=True)
        df2.set_index('code', inplace=True)
        df_details = pd.concat([df1, df2], axis=1)
        
        url2 = f'http://fund.eastmoney.com/f10/tsdata_{code}.html'
        tables2 = pd.read_html(url2)
        if len(tables2) < 2:
            raise ValueError("风险表格数量不足")
        df_sharpe = tables2[1]
        df_sharpe['code'] = code
        df_sharpe.set_index('code', inplace=True)
        df_sharpe.drop('基金风险指标', axis='columns', inplace=True, errors='ignore')
        df_sharpe = df_sharpe[1:]
        df_sharpe.columns = [f'夏普比率(近{c})' for c in df_sharpe.columns]
        df_sharpe = df_sharpe.apply(pd.to_numeric, errors='coerce')
        
        df_final = df_details.combine_first(df_sharpe)
        print(f"成功获取基金 {code} 详情。")
        return df_final
    except Exception as e:
        print(f"获取基金 {code} 详情失败: {e}")
        return pd.DataFrame()

def get_fund_data(code, sdate='', edate=''):
    print(f"开始获取基金 {code} 历史净值数据...")
    url = f'https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page=1&per=65535&sdate={sdate}&edate={edate}'
    try:
        response = getURL(url)
        if not response:
            raise ValueError("无法获取响应")
            
        content_match = re.search(r'content:"(.*?)"', response.text)
        
        if not content_match:
            raise ValueError("未找到净值表格内容")
            
        html_content = content_match.group(1).encode('utf-8').decode('unicode_escape')
        
        if "净值日期单位净值" in html_content:
            print("识别为纯文本净值数据，使用新方法解析...")
            rows = re.findall(r'(\d{4}-\d{2}-\d{2})([\d.]+)([\d.]+)([-+]?\d+\.\d+%)', html_content)
            if not rows:
                raise ValueError("纯文本解析数据为空")
                
            data = []
            for row in rows:
                data.append({
                    '净值日期': row[0],
                    '单位净值': row[1],
                    '累计净值': row[2],
                    '日增长率': row[3]
                })
            
            df = pd.DataFrame(data)
            df['申购状态'] = '开放申购'
            df['赎回状态'] = '开放赎回'
            df['分红送配'] = ''
        else:
            print("识别为HTML表格数据，尝试lxml解析...")
            tree = etree.HTML(html_content)
            rows = tree.xpath("//tbody/tr")
            if not rows:
                raise ValueError("未找到净值表格")
            
            data = []
            for row in rows:
                cols = row.xpath("./td/text()")
                if len(cols) >= 7:
                    data.append({
                        '净值日期': cols[0].strip(),
                        '单位净值': cols[1].strip(),
                        '累计净值': cols[2].strip(),
                        '日增长率': cols[3].strip(),
                        '申购状态': cols[4].strip(),
                        '赎回状态': cols[5].strip(),
                        '分红送配': cols[6].strip()
                    })
            df = pd.DataFrame(data)

        if df.empty:
            raise ValueError("解析数据为空")
        
        df['净值日期'] = pd.to_datetime(df['净值日期'], format='mixed', errors='coerce')
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        df['累计净值'] = pd.to_numeric(df['累计净值'], errors='coerce')
        df['日增长率'] = pd.to_numeric(df['日增长率'].str.strip('%'), errors='coerce') / 100
        df = df.dropna(subset=['净值日期', '单位净值'])
        df = df[(df['净值日期'] >= sdate) & (df['净值日期'] <= edate)] if sdate and edate else df
        
        print(f"成功获取 {code} 的 {len(df)} 条净值数据")
        return df

    except Exception as e:
        print(f"解析净值数据失败 ({e})，尝试 akshare...")
        try:
            df = ak.fund_open_fund_daily_em(symbol=code, start_date=sdate, end_date=edate)
            if df.empty:
                raise ValueError("akshare 数据为空")
            df['净值日期'] = pd.to_datetime(df['净值日期'], format='mixed', errors='coerce')
            df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
            df['累计净值'] = pd.to_numeric(df.get('累计净值', df['单位净值']), errors='coerce')
            df['日增长率'] = pd.to_numeric(df.get('日增长率', 0), errors='coerce')
            print(f"akshare 获取 {code} 的 {len(df)} 条净值数据")
            return df
        except Exception as e:
            print(f"akshare 解析失败: {e}")
            return pd.DataFrame()


def get_fund_holdings_with_selenium(fund_code):
    print(f"开始使用 Selenium 获取基金 {fund_code} 持仓数据...")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument(f'user-agent={randHeader()["User-Agent"]}')
    
    print("正在安装 ChromeDriver...")
    service = Service(ChromeDriverManager().install())
    
    driver = None
    try:
        print("正在启动 Chrome 浏览器...")
        driver = webdriver.Chrome(service=service, options=options)
        url = f'http://fundf10.eastmoney.com/ccmx_{fund_code}.html'
        print(f"正在访问 {url}")
        driver.get(url)
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, 'lxml')
        tables = soup.find_all('table')

        if not tables:
            print(f"× 未在页面找到表格。URL: {url}")
            return []

        holdings_table = None
        for table in tables:
            if table.find('th', string='股票名称'):
                holdings_table = table
                break
        
        if not holdings_table:
            print(f"× 未在页面找到股票持仓表格。URL: {url}")
            return []

        holdings_data = []
        rows = holdings_table.find_all('tr')[1:]
        if not rows:
            print(f"× 股票持仓表格为空。URL: {url}")
            return []

        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 8:
                try:
                    stock_code = cols[1].text.strip()
                    stock_name = cols[2].text.strip()
                    proportion = cols[6].text.strip().replace('%', '')
                    shares = cols[7].text.strip()
                    value = cols[8].text.strip()

                    if proportion == '-' or not proportion:
                        proportion = '0'
                    if shares == '-' or not shares:
                        shares = '0'
                    if value == '-' or not value:
                        value = '0'
                    
                    holdings_data.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'proportion': float(proportion),
                        'shares': float(re.sub(r'[^\d.]', '', shares)),
                        'value': float(re.sub(r'[^\d.]', '', value))
                    })
                except (IndexError, ValueError) as e:
                    print(f"解析持仓数据时发生错误：{e}")
                    traceback.print_exc()
                    continue
        
        print(f"成功提取 {len(holdings_data)} 条持仓数据。")
        return holdings_data
    
    except Exception as e:
        print(f"在获取基金持仓数据时发生错误: {e}")
        traceback.print_exc()
        return []
    finally:
        if driver:
            driver.quit()
        print("Chrome 浏览器已关闭。")

def get_fund_managers(fund_code, proxies=None):
    print(f"开始获取基金 {fund_code} 经理数据...")
    fund_url = f'http://fund.eastmoney.com/f10/jjjl_{fund_code}.html'
    try:
        res = getURL(fund_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        tables = soup.find_all("table")
        if len(tables) < 2:
            raise ValueError("表格数量不足")
        tab = tables[1]
        result = []
        for tr in tab.find_all('tr'):
            if tr.find_all('td'):
                try:
                    manager_data = {
                        'fund_code': fund_code,
                        'start_date': tr.select('td:nth-of-type(1)')[0].get_text().strip(),
                        'end_date': tr.select('td:nth-of-type(2)')[0].get_text().strip(),
                        'fund_managers': tr.select('td:nth-of-type(3)')[0].get_text().strip(),
                        'term': tr.select('td:nth-of-type(4)')[0].get_text().strip(),
                        'management_return': tr.select('td:nth-of-type(5)')[0].get_text().strip(),
                        'management_rank': tr.select('td:nth-of-type(6)')[0].get_text().strip()
                    }
                    result.append(manager_data)
                except IndexError:
                    continue
        print(f"成功获取基金经理数据。")
        return result
    except Exception as e:
        print(f"获取基金经理数据失败: {e}")
        return []

def analyze_fund(fund_code, start_date, end_date):
    print(f"开始分析基金 {fund_code} 风险指标...")
    try:
        df = get_fund_data(fund_code, sdate=start_date, edate=end_date)
        if df.empty:
            raise ValueError("净值数据为空，无法进行分析")
        
        returns = df['单位净值'].pct_change().dropna()
        if returns.empty:
            raise ValueError("没有足够的回报数据进行分析")

        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_returns - 0.03) / annual_volatility if annual_volatility != 0 else 0
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        result = {
            "fund_code": fund_code,
            "annual_returns": float(annual_returns),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "data_source": "akshare"
        }
        print(f"基金 {fund_code} 风险分析完成。")
        return result
    except Exception as e:
        print(f"分析基金 {fund_code} 风险参数失败: {e}")
        return {"error": "风险参数计算失败"}

# --- 指数估值函数：从 index_valuation_scraper.py 整合而来 ---
INDEX_API_URL = 'https://danjuanfunds.com/djapi/index_eva/dj'
def get_index_data():
    print("开始获取指数估值数据...")
    try:
        response = requests.get(INDEX_API_URL, headers=randHeader(), timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('data') and data['data'].get('items'):
            print("成功获取指数估值数据。")
            return data['data']['items']
        else:
            print("错误：API返回的数据结构不正确或无数据。")
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求指数估值数据失败：{e}")
        return None
    except json.JSONDecodeError:
        print("无法解析API响应，返回的不是有效的JSON。")
        return None

def comprehensive_filter_indices(data):
    print("正在筛选低估指数...")
    selected_indices = []
    pe_percentile_threshold = 0.20
    yeild_threshold = 0.03
    
    for item in data:
        pe_percentile = item.get('pe_percentile')
        yeild = item.get('yeild')
        if pe_percentile is not None and yeild is not None:
            if pe_percentile < pe_percentile_threshold and yeild > yeild_threshold:
                selected_indices.append(item)
    print("低估指数筛选完成。")
    return selected_indices

def save_to_csv(data, filename, fieldnames):
    print(f"正在保存数据到 {filename}...")
    if not data:
        print(f"没有数据可保存到 {filename}。")
        return
        
    today_date = datetime.now().strftime('%Y-%m-%d')
    csv_file_path = os.path.join(os.getcwd(), filename)
    
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in data:
                row = {
                    '日期': today_date,
                    '指数名称': item.get('name', ''),
                    '指数代码': item.get('index_code', ''),
                    'PE': item.get('pe', ''),
                    'PE百分位': item.get('pe_percentile', ''),
                    'PB': item.get('pb', ''),
                    'PB百分位': item.get('pb_percentile', ''),
                    '股息率': item.get('yeild', ''),
                    'ROE': item.get('roe', '')
                }
                writer.writerow(row)
        print(f"数据已成功保存到 {csv_file_path}")
    except IOError as e:
        print(f"写入文件时出错：{e}")

# --- 主函数：整合两个脚本的工作流 ---
def main_analyzer():
    # 第一步：获取和筛选低估指数
    print("--- 第一步：开始获取和筛选指数估值数据 ---")
    index_data = get_index_data()
    if index_data:
        all_index_fields = ['日期', '指数名称', '指数代码', 'PE', 'PE百分位', 'PB', 'PB百分位', '股息率', 'ROE']
        save_to_csv(index_data, filename='all_index_valuation.csv', fieldnames=all_index_fields)
        
        selected_indices = comprehensive_filter_indices(index_data)
        if selected_indices:
            print("\n根据综合筛选策略，发现以下值得关注的低估指数：")
            selected_index_fields = ['日期', '指数名称', '指数代码', 'PE', 'PE百分位', 'PB', 'PB百分位', '股息率', 'ROE']
            save_to_csv(selected_indices, filename='selected_low_valuation_indices.csv', fieldnames=selected_index_fields)
            for index in selected_indices:
                print(f"- {index.get('name')} (PE百分位: {index.get('pe_percentile')*100:.2f}%, PB百分位: {index.get('pb_percentile')*100:.2f}%)")
        else:
            print("没有发现符合筛选条件的低估指数。")

    # 第二步：获取和筛选基金
    print("\n--- 第二步：开始获取全量基金信息并进行筛选 ---")
    fund_info = get_fund_basic_info()
    
    fund_info = fund_info[fund_info['代码'].str.len() == 6]
    fund_info = fund_info[~fund_info['类型'].str.contains('ETF|LOF|场内', na=False, regex=True)]
    fund_info = fund_info[fund_info['名称'].str.contains('C$|C类', na=False, regex=True)]
    fund_codes = fund_info['代码'].tolist()
    print(f"过滤后只保留场外C类基金：{len(fund_info)} 只")

    rankings_df = get_fund_rankings(fund_type='hh')
    
    if not rankings_df.empty:
        total_records = len(rankings_df)
        recommended_df = apply_4433_rule(rankings_df, total_records)
        
        recommended_df = pd.merge(recommended_df, fund_info[['代码', '名称']], left_index=True, right_on='代码', how='inner')
        recommended_df = recommended_df.set_index('代码')
        
        recommended_path = 'recommended_cn_funds.csv'
        recommended_df.to_csv(recommended_path, encoding='gbk')
        print(f"推荐场外C类基金列表已保存至 '{recommended_path}'（{len(recommended_df)} 只基金）")
        fund_codes_to_analyze = recommended_df.index.tolist()[:20]
    else:
        print("排名数据为空，使用前 10 只场外C类基金继续处理")
        fund_codes_to_analyze = fund_codes[:10]
    
    # 第三步：对筛选出的基金进行详细分析
    print("\n--- 第三步：开始对筛选出的基金进行详细分析 ---")
    all_fund_details = []
    
    # 获取日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    
    for i, fund_code in enumerate(fund_codes_to_analyze, 1):
        print(f"[{i}/{len(fund_codes_to_analyze)}] 处理场外C类基金 {fund_code}...")
        
        fund_data = {
            "fund_code": fund_code,
            "fund_details": get_fund_details(fund_code).to_dict('records') if not get_fund_details(fund_code).empty else {},
            "fund_holdings": get_fund_holdings_with_selenium(fund_code),
            "fund_managers": get_fund_managers(fund_code),
            "risk_metrics": analyze_fund(fund_code, start_date, end_date)
        }
        
        all_fund_details.append(fund_data)
        
    comprehensive_analysis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comprehensive_fund_analysis.json')
    with open(comprehensive_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(all_fund_details, f, indent=4, ensure_ascii=False)
    print(f"\n所有基金的详细分析结果已整合并保存至 '{comprehensive_analysis_path}'。")
    print("脚本运行完成。")

if __name__ == '__main__':
    main_analyzer()
