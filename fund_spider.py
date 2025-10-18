import pandas as pd
import requests
import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
import concurrent.futures
import json
import ast  # 用于安全地解析 Python/JS 对象字面量

# 定义文件路径和目录
INPUT_FILE = 'C类.txt'
OUTPUT_DIR = 'fund_data'
BASE_URL_NET_VALUE = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
BASE_URL_INFO_JS = "http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
INFO_CACHE_FILE = 'fund_info.csv'

# 设置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 3.5
MAX_CONCURRENT = 5
MAX_FUNDS_PER_RUN = 0 
PAGE_SIZE = 20

# --------------------------------------------------------------------------------------
# 辅助函数：JS变量提取和解析
# --------------------------------------------------------------------------------------

def extract_simple_var(text, var_name):
    """提取简单的字符串或数字变量，如 fS_name, fund_Rate"""
    match = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*(.*?);', text, re.DOTALL)
    if match:
        value = match.group(1).strip()
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1].lstrip('\ufeff')
        return value
    return None

def extract_json_var(text, var_name):
    """提取复杂的 JSON/JS Object Literal 变量，如 apidata, Data_fund_info"""
    # 匹配 var var_name = [{...}] ; 或 var var_name = {...} ;
    # 非贪婪匹配直到遇到结束的 ]} 之一，后面跟着分号
    match = re.search(r'var\s+' + re.escape(var_name) + r'\s*=\s*([\[\{].*?[\]\}])\s*;', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_js_object_literal(js_text):
    """
    尝试将 JS 对象字面量（键没有引号）转换为 Python 字典。
    处理对象 { } 或单元素数组 [ { } ] 结构，并抽取基金经理名称。
    """
    if not js_text:
        return None

    # 1. 如果是 [ {...} ] 结构，移除外层数组中括号
    if js_text.startswith('[') and js_text.endswith(']'):
        js_text = js_text.strip()[1:-1].strip()
        if not js_text:
            return None

    try:
        # 2. 替换 JS 的 true/false/null 为 Python 的 True/False/None
        text = js_text.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        
        # 3. 将无引号的键名 (key:) 替换为带引号的键名 ("key":)
        # 此处使用更精确的正则，匹配字母数字下划线开头的键名
        text = re.sub(r'([\{\,]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', text)
        
        # 4. 尝试用 ast.literal_eval 解析 (最安全的方法)
        data = ast.literal_eval(text)

        # 5. 抽取基金经理名称，使 FundManager 字段变成字符串
        if isinstance(data, dict):
            if 'FundManager' in data and isinstance(data['FundManager'], list) and data['FundManager']:
                # 提取第一个基金经理的名称
                manager_name = data['FundManager'][0].get('name', '未知')
                data['FundManager'] = manager_name
        
        return data
        
    except Exception as e:
        # 失败时，尝试用标准 JSON 解析器（如果它接近标准 JSON）
        try:
             # 清理可能残留的单引号或 JS 注释
             clean_text = js_text.replace("'", '"')
             clean_text = re.sub(r'//.*?\n|/\*.*?\*/', '', clean_text, flags=re.DOTALL)
             return json.loads(clean_text)
        except:
             print(f"    [解析错误] 尝试使用 ast/json 解析 JS 对象字面量失败: {e}. 原始文本: {js_text[:50]}...")
             return None


# --------------------------------------------------------------------------------------
# 基金信息抓取核心逻辑 (已更新)
# --------------------------------------------------------------------------------------

async def fetch_fund_info(fund_code, session, semaphore):
    """异步抓取基金基本信息，包含增强的变量提取逻辑"""
    print(f"    -> 开始抓取基金 {fund_code} 的基本信息")
    url = BASE_URL_INFO_JS.format(fund_code=fund_code)
    
    # 默认值
    default_result = {
        '代码': fund_code, '名称': '抓取失败', '类型': '未知', '成立日期': '未知', '基金经理': '未知', 
        '公司名称': '未知', '基金简称': '未知', '基金类型': '未知', '发行时间': '未知', '估值日期': '未知', 
        '估值涨幅': '未知', '累计净值': '未知', '单位净值': '未知', '申购净值': '未知', '申购报价': '未知', 
        '净值日期': '未知', '申购状态': '未知', '赎回状态': '未知', '费率': '未知', '申购步长': '未知'
    }
    
    async with semaphore:
        try:
            await asyncio.sleep(REQUEST_DELAY * 0.5)
            js_text = await fetch_js_page(session, url)
            
            # 1. 移除 BOM 字符 (U+FEFF)
            js_text = js_text.lstrip('\ufeff')
            
            # 2. 提取简单变量
            fund_name = extract_simple_var(js_text, 'fS_name') or '未知名称'
            fund_code_in_data = extract_simple_var(js_text, 'fS_code') or fund_code
            fund_type_raw = extract_simple_var(js_text, 'FTyp')
            fund_rate = extract_simple_var(js_text, 'fund_Rate')
            fund_minsg = extract_simple_var(js_text, 'fund_minsg')
            
            # 3. 提取并解析 apidata (大型 JSON 块，包含估值和最新净值)
            data_main = {}
            data_main_str = extract_json_var(js_text, 'apidata')
            if data_main_str:
                 # apidata 通常是标准 JSON，但可能包含注释或格式问题，先简单清洗
                 corrected_json_str = data_main_str.replace("'", '"')
                 corrected_json_str = re.sub(r'//.*?\n|/\*.*?\*/', '', corrected_json_str, flags=re.DOTALL)
                 try:
                     data_main = json.loads(corrected_json_str)
                 except json.JSONDecodeError:
                     # 如果标准 JSON 失败，尝试用 JS Object Literal 方式解析
                     data_main = parse_js_object_literal(data_main_str) or {}
            
            # 4. 提取并解析 Data_fund_info (基金概况信息，包含经理、公司、成立日期)
            data_info = {}
            data_info_str = extract_json_var(js_text, 'Data_fund_info')
            if data_info_str:
                 # Data_fund_info 是典型的 JS Object Literal，使用新的通用解析函数
                 data_info = parse_js_object_literal(data_info_str) or {}


            # 5. 整理最终结果
            # 注意：所有字段都优先从 Data_fund_info 或 data_main 中获取，fallback 到 '未知'
            result = {
                '代码': fund_code_in_data,
                '名称': fund_name,
                
                # 核心信息提取
                '基金经理': data_info.get('FundManager', '未知'),
                '公司名称': data_info.get('FundCompany', data_main.get('jjgs', '未知')),
                '成立日期': data_info.get('EstablishDate', data_main.get('qjdate', '未知')),
                '基金简称': data_info.get('jjjc', fund_name.split('(')[0].strip() if '(' in fund_name else fund_name.strip()),
                '基金类型': data_info.get('FundType', fund_type_raw or data_main.get('FTyp', '未知')),
                '发行时间': data_info.get('IssueDate', '未知'),
                '类型': data_info.get('FundType', fund_type_raw or data_main.get('FTyp', '未知')),
                
                # 净值/估值信息
                '估值日期': data_main.get('gzrq', '未知'),
                '估值涨幅': data_main.get('gszzl', '未知'),
                '累计净值': data_main.get('ljjz', '未知'),
                '单位净值': data_main.get('dwjz', '未知'),
                '净值日期': data_main.get('jzrq', '未知'),
                '申购状态': data_main.get('sgzt', '未知'),
                '赎回状态': data_main.get('shzt', '未知'),
                
                # 费率/步长
                '费率': fund_rate if fund_rate else '未知',
                '申购步长': fund_minsg if fund_minsg else '未知',
                
                # 冗余字段
                '申购净值': '未知',
                '申购报价': '未知',
            }

            print(f"    -> 基金 {fund_code} 基本信息抓取成功 (名称: {result['名称']} | 公司: {result['公司名称']} | 经理: {result['基金经理']})")
            return fund_code, result

        except Exception as e:
            error_message = str(e)
            print(f"    -> 基金 {fund_code} [信息抓取失败]: {error_message}")
            return fund_code, default_result

# --------------------------------------------------------------------------------------
# 文件读取、缓存和动态数据抓取逻辑 (保持不变)
# --------------------------------------------------------------------------------------

def get_all_fund_codes(file_path):
    """从 C类.txt 文件中读取基金代码"""
    print(f"尝试读取基金代码文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"[错误] 文件 {file_path} 不存在。")
        return []

    if os.path.getsize(file_path) == 0:
        print(f"[错误] 文件 {file_path} 为空。")
        return []

    encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
    df = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, header=None, dtype=str)
            df.columns = ['code']
            print(f"  -> 成功使用 {encoding} 编码读取文件，找到 {len(df)} 个基金代码。")
            break
        except UnicodeDecodeError as e:
            print(f"  -> 使用 {encoding} 编码读取失败: {e}")
            continue
        except Exception as e:
            print(f"  -> 读取文件时发生错误: {e}")
            continue

    if df is None:
        print("[错误] 无法读取文件，请检查文件格式和编码。")
        return []

    codes = df['code'].dropna().astype(str).str.strip().unique().tolist()
    valid_codes = [code for code in codes if re.match(r'^\d{6}$', code)]
    if not valid_codes:
        print("[错误] 没有找到有效的6位基金代码。")
        return []
    print(f"  -> 找到 {len(valid_codes)} 个有效基金代码。")
    return valid_codes

def load_info_cache():
    """加载基金基本信息缓存 (从 CSV 文件)"""
    if not os.path.exists(INFO_CACHE_FILE):
        print(f"[信息] 缓存文件 {INFO_CACHE_FILE} 不存在，将创建新缓存。")
        return {}

    try:
        df = pd.read_csv(INFO_CACHE_FILE, dtype={'代码': str}, encoding='utf-8')
        if df.empty:
            print(f"[警告] 缓存文件 {INFO_CACHE_FILE} 为空。")
            return {}
        df.set_index('代码', inplace=True)
        info_cache = df.to_dict('index')
        print(f"  -> 成功从 {INFO_CACHE_FILE} 加载 {len(info_cache)} 条缓存记录。")
        return info_cache
    except Exception as e:
        print(f"[警告] 加载基本信息缓存失败: {e}。将从头抓取。")
        return {}

def save_info_cache(cache):
    """保存基金基本信息缓存 (到 CSV 文件)"""
    if not cache:
        print("[警告] 基本信息缓存为空，不保存。")
        return

    try:
        df = pd.DataFrame.from_dict(cache, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': '代码'}, inplace=True)
        
        # 固定列顺序，使用中文列名
        cols = ['代码', '名称', '类型', '成立日期', '基金经理', '公司名称', '基金简称', '基金类型', '发行时间', '估值日期', '估值涨幅', '累计净值', '单位净值', '申购净值', '申购报价', '净值日期', '申购状态', '赎回状态', '费率', '申购步长']
        for col in cols:
            if col not in df.columns:
                df[col] = '未知'
        df = df[cols]
        
        df['代码'] = df['代码'].astype(str)
        df.to_csv(INFO_CACHE_FILE, index=False, encoding='utf-8')
        print(f"  -> 成功保存基本信息缓存到 {INFO_CACHE_FILE}")
    except Exception as e:
        print(f"[错误] 保存基本信息缓存失败: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_message(match='Frequency Capped')
)
async def fetch_js_page(session, url):
    """异步请求 JS 页面，带重试机制"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status == 514:
            raise aiohttp.ClientError("Frequency Capped")
        if response.status != 200:
            print(f"    -> HTTP 状态码: {response.status}")
            text = await response.text()
            print(f"    -> 响应内容（前100字符）: {text[:100]}")
            raise aiohttp.ClientError(f"HTTP 错误: {response.status}")
        return await response.text()

async def fetch_and_cache_fund_info(fund_codes):
    """主函数：检查缓存，对未缓存的基金进行并发抓取"""
    print("\n======== 开始抓取基金基本信息（静态数据）========\n")
    
    loop = asyncio.get_event_loop()
    info_cache = await loop.run_in_executor(None, load_info_cache)
    
    # 重新抓取条件：代码缺失或名称/经理/公司为 '未知' 或 '抓取失败'
    codes_to_fetch = [
        code for code in fund_codes if code not in info_cache or 
        info_cache[code].get('名称') in ['未知名称', '抓取失败'] or 
        info_cache[code].get('基金经理') == '未知' or 
        info_cache[code].get('公司名称') == '未知'
    ]
    
    if not codes_to_fetch:
        print("所有基金的基本信息都已在缓存中。")
        return info_cache

    print(f"发现 {len(codes_to_fetch)} 个基金信息缺失/需要重新抓取，开始抓取...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async with ClientSession() as session:
        fetch_tasks = [fetch_fund_info(code, session, semaphore) for code in codes_to_fetch]
        
        for future in asyncio.as_completed(fetch_tasks):
            code, result = await future
            info_cache[code] = result
            
    await loop.run_in_executor(None, save_info_cache, info_cache)
    print("\n======== 基金基本信息抓取完成 ========\n")
    return info_cache


# --------------------------------------------------------------------------------------
# 基金净值 (动态数据) 抓取和保存逻辑 (保持不变)
# --------------------------------------------------------------------------------------

def load_latest_date(fund_code):
    """从本地 CSV 文件中读取现有最新日期"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            if not df.empty and 'date' in df.columns:
                latest_date = df['date'].max().to_pydatetime().date()
                print(f"  -> 基金 {fund_code} 现有最新日期: {latest_date.strftime('%Y-%m-%d')}")
                return latest_date
        except Exception as e:
            print(f"  -> 加载 {fund_code} CSV 失败: {e}")
    return None

async def fetch_net_values(fund_code, session, semaphore):
    """使用“最新日期”作为停止条件，实现智能增量更新"""
    print(f"-> [START] 基金代码 {fund_code}")
    
    async with semaphore:
        all_records = []
        page_index = 1
        total_pages = 1
        first_run = True
        dynamic_delay = REQUEST_DELAY
        
        loop = asyncio.get_event_loop()
        latest_date = await loop.run_in_executor(None, load_latest_date, fund_code)

        while page_index <= total_pages:
            url = BASE_URL_NET_VALUE.format(fund_code=fund_code, page_index=page_index)
            
            try:
                if page_index > 1:
                    await asyncio.sleep(dynamic_delay)
                
                text = await fetch_page(session, url)
                
                if first_run:
                    total_pages_match = re.search(r'pages:(\d+)', text)
                    total_pages = int(total_pages_match.group(1)) if total_pages_match else 1
                    records_match = re.search(r'records:(\d+)', text)
                    total_records = int(records_match.group(1)) if records_match else '未知'
                    print(f"    基金 {fund_code} 信息：总页数 {total_pages}，总记录数 {total_records}。")
                    
                    if total_records == '未知' or int(total_records) == 0:
                        print(f"    基金 {fund_code} [跳过]：API 返回总记录数为 0 或未知。")
                        return fund_code, "API返回记录数为0或代码无效"
                        
                    first_run = False
                
                soup = BeautifulSoup(text, 'lxml')
                table = soup.find('table')
                if not table:
                    print(f"    基金 {fund_code} [警告]：页面 {page_index} 无表格数据。提前停止。")
                    break

                rows = table.find_all('tr')[1:]
                if not rows:
                    print(f"    基金 {fund_code} 第 {page_index} 页无数据行。停止抓取。")
                    break

                page_records = []
                stop_fetch = False
                latest_api_date = None
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 7:
                        continue
                        
                    date_str = cols[0].text.strip()
                    net_value_str = cols[1].text.strip()
                    cumulative_net_value = cols[2].text.strip()
                    daily_growth_rate = cols[3].text.strip()
                    purchase_status = cols[4].text.strip()
                    redemption_status = cols[5].text.strip()
                    dividend = cols[6].text.strip()
                    
                    if not date_str or not net_value_str:
                        continue
                        
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        if not latest_api_date or date > latest_api_date:
                            latest_api_date = date
                        
                        if latest_date and date <= latest_date:
                            stop_fetch = True
                            break
                            
                        page_records.append({
                            'date': date_str,
                            'net_value': net_value_str,
                            'cumulative_net_value': cumulative_net_value,
                            'daily_growth_rate': daily_growth_rate,
                            'purchase_status': purchase_status,
                            'redemption_status': redemption_status,
                            'dividend': dividend
                        })
                    except ValueError:
                        continue
                
                if latest_api_date:
                    print(f"    基金 {fund_code} API 返回最新日期: {latest_api_date.strftime('%Y-%m-%d')}")
                
                all_records.extend(page_records)
                
                if stop_fetch:
                    print(f"    基金 {fund_code} [增量停止]：页面 {page_index} 遇到旧数据 ({latest_date.strftime('%Y-%m-%d')})，停止抓取。")
                    break
                
                page_index += 1
                dynamic_delay = max(REQUEST_DELAY, dynamic_delay * 0.9)

            except aiohttp.ClientError as e:
                if "Frequency Capped" in str(e):
                    dynamic_delay = min(dynamic_delay * 2, 5.0)
                    print(f"    基金 {fund_code} [警告]：频率限制，延迟调整为 {dynamic_delay} 秒，重试第 {page_index} 页")
                    continue
                print(f"    基金 {fund_code} [错误]：请求 API 时发生网络错误 (超时/连接) 在第 {page_index} 页: {e}")
                return fund_code, f"网络错误: {e}"
            except Exception as e:
                print(f"    基金 {fund_code} [错误]：处理数据时发生意外错误在第 {page_index} 页: {e}")
                return fund_code, f"数据处理错误: {e}"
            
        print(f"-> [COMPLETE] 基金 {fund_code} 数据抓取完毕，共获取 {len(all_records)} 条新记录。")
        if not all_records:
            return fund_code, "数据已是最新，无新数据"
        return fund_code, all_records

def save_to_csv(fund_code, data):
    """将历史净值数据以增量更新方式保存为 CSV 文件"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if not isinstance(data, list) or not data:
        print(f"    基金 {fund_code} 无新数据可保存。")
        return False, 0

    new_df = pd.DataFrame(data)

    try:
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['cumulative_net_value'] = pd.to_numeric(new_df['cumulative_net_value'], errors='coerce').round(4)
        new_df['daily_growth_rate'] = new_df['daily_growth_rate'].replace('--', '0').str.rstrip('%').astype(float) / 100.0
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df.dropna(subset=['date', 'net_value'], inplace=True)
        if new_df.empty:
            print(f"    基金 {fund_code} 数据无效或为空，跳过保存。")
            return False, 0
    except Exception as e:
        print(f"    基金 {fund_code} 数据转换失败: {e}")
        return False, 0
    
    old_record_count = 0
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            old_record_count = len(existing_df)
            combined_df = pd.concat([new_df, existing_df])
        except Exception as e:
            print(f"    读取现有 CSV 文件 {output_path} 失败: {e}。仅保存新数据。")
            combined_df = new_df
    else:
        combined_df = new_df
        
    final_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    final_df = final_df.sort_values(by='date', ascending=False)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        new_record_count = len(final_df)
        newly_added = new_record_count - old_record_count
        print(f"    -> 基金 {fund_code} [保存完成]：总记录数 {new_record_count} (新增 {max(0, newly_added)} 条)。")
        return True, max(0, newly_added)
    except Exception as e:
        print(f"    基金 {fund_code} 保存 CSV 文件 {output_path} 失败: {e}")
        return False, 0

async def fetch_all_funds(fund_codes):
    """异步获取所有基金数据，并在任务完成时立即保存数据"""
    print("\n======== 开始基金净值数据抓取（动态数据）========\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    loop = asyncio.get_event_loop()

    async with ClientSession() as session:
        fetch_tasks = [fetch_net_values(fund_code, session, semaphore) for fund_code in fund_codes]
        
        success_count = 0
        total_new_records = 0
        failed_codes = []
        
        for future in asyncio.as_completed(fetch_tasks):
            print("-" * 30)
            fund_code = "UNKNOWN"
            try:
                fund_code, net_values = await future
            except Exception as e:
                print(f"[错误] 处理基金数据时发生顶级异步错误: {e}")
                failed_codes.append("UNKNOWN")
                continue

            if isinstance(net_values, list):
                try:
                    success, new_records = await loop.run_in_executor(None, save_to_csv, fund_code, net_values)
                    if success:
                        success_count += 1
                        total_new_records += new_records
                    else:
                        failed_codes.append(fund_code)
                except Exception as e:
                    print(f"[错误] 基金 {fund_code} 的保存任务在线程中发生错误: {e}")
                    failed_codes.append(fund_code)
            else:
                print(f"    基金 {fund_code} [抓取失败/跳过]：{net_values}")
                if not str(net_values).startswith('数据已是最新'):
                    failed_codes.append(fund_code)

        return success_count, total_new_records, failed_codes

def main():
    """主函数：集成静态信息抓取和动态净值抓取"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"确保输出目录存在: {OUTPUT_DIR}")

    fund_codes = get_all_fund_codes(INPUT_FILE)
    if not fund_codes:
        print("[错误] 没有可处理的基金代码，脚本结束。")
        return

    if MAX_FUNDS_PER_RUN > 0 and len(fund_codes) > MAX_FUNDS_PER_RUN:
        print(f"限制本次运行最多处理 {MAX_FUNDS_PER_RUN} 个基金。")
        processed_codes = fund_codes[:MAX_FUNDS_PER_RUN]
    else:
        processed_codes = fund_codes
    print(f"本次处理 {len(processed_codes)} 个基金。")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(fetch_and_cache_fund_info(processed_codes))
    success_count, total_new_records, failed_codes = loop.run_until_complete(fetch_all_funds(processed_codes))
    
    print(f"\n======== 本次更新总结 ========")
    print(f"成功处理 {success_count} 个基金，新增/更新 {total_new_records} 条记录，失败 {len(failed_codes)} 个基金。")
    if failed_codes:
        print(f"失败的基金代码: {', '.join(set(failed_codes))}")
    if total_new_records == 0:
        print("[警告] 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")

if __name__ == "__main__":
    main()
