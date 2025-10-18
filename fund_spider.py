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
import json5 # *** 新增：用于解析非严格JSON格式的JS对象 ***

# 定义文件路径和目录
INPUT_FILE = 'C类.txt'
OUTPUT_DIR = 'fund_data'
# 基金净值数据 API (不变)
BASE_URL_NET_VALUE = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
# 基金基本信息 JS 接口 (新来源)
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
MAX_FUNDS_PER_RUN = 0  # 默认不限制
PAGE_SIZE = 20

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
    valid_codes = [code for code in codes if re.match(r'^\d{6}$', code)]  # 确保是6位数字
    if not valid_codes:
        print("[错误] 没有找到有效的6位基金代码。")
        return []
    print(f"  -> 找到 {len(valid_codes)} 个有效基金代码。")
    return valid_codes

# --------------------------------------------------------------------------------------
# 基金基本信息 (静态数据) 抓取和缓存逻辑
# --------------------------------------------------------------------------------------

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
        
        # *** 修正后的固定列顺序，新增 '基金规模' ***
        cols = [
            '代码', '名称', '类型', '成立日期', '基金经理', '公司名称', '基金简称', '基金类型', 
            '发行时间', '基金规模', '估值日期', '估值涨幅', '累计净值', '单位净值', 
            '申购净值', '申购报价', '净值日期', '申购状态', '赎回状态', '费率', '申购步长'
        ]
        
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
        response.raise_for_status()
        return await response.text()

async def fetch_fund_info(fund_code, session, semaphore):
    """异步抓取基金基本信息 (从 JS 接口)"""
    print(f"   -> 开始抓取基金 {fund_code} 的基本信息")
    url = BASE_URL_INFO_JS.format(fund_code=fund_code)
    
    # 默认值，包含所有用户期望的字段
    default_result = {
        '代码': fund_code, '名称': '抓取失败', '类型': '未知', '成立日期': '未知', 
        '基金经理': '未知', '公司名称': '未知', '基金简称': '未知', '基金类型': '未知', 
        '发行时间': '未知', '基金规模': '未知', '估值日期': '未知', '估值涨幅': '未知', 
        '累计净值': '未知', '单位净值': '未知', '申购净值': '未知', '申购报价': '未知', 
        '净值日期': '未知', '申购状态': '未知', '赎回状态': '未知', '费率': '未知', 
        '申购步长': '未知'
    }

    async with semaphore:
        try:
            await asyncio.sleep(REQUEST_DELAY * 0.5)
            js_text = await fetch_js_page(session, url)
            
            # --- 1. 提取 fS_json (基金基本信息、净值信息) ---
            fs_json_match = re.search(r'var fS_json = ({.*?});', js_text, re.DOTALL)
            f_s_data = json5.loads(fs_json_match.group(1)) if fs_json_match else {}
            
            # --- 2. 提取 fInfo (基金经理、公司、费率、状态) ---
            finfo_match = re.search(r'var fInfo = ({.*?});', js_text, re.DOTALL)
            f_info_data = json5.loads(finfo_match.group(1)) if finfo_match else {}
            
            # --- 3. 提取 Data_fund規模 (最新基金规模) ---
            scale_match = re.search(r'Data_fund规模 = (\[.*?\]);', js_text, re.DOTALL)
            scale = '未知'
            if scale_match:
                # 规模数据是严格 JSON 数组
                scale_data = json.loads(scale_match.group(1))
                if scale_data and scale_data[-1][1] is not None:
                    # 取最新一个季度的数据
                    scale = f"{scale_data[-1][1]} 亿元"
            
            # --- 4. 提取 Data_buyRate (申购费率，可选，主要用 fInfo['rate']) ---
            buy_rate_display = f_info_data.get('rate', '未知') # 默认使用 fInfo 中的费率
            if str(buy_rate_display).replace('.', '', 1).isdigit():
                 buy_rate_display = f"{float(buy_rate_display) * 100:.2f}%"

            # --- 5. 组合字段 ---
            
            # 基金经理
            manager_list = f_info_data.get('fundManager', [])
            manager = ', '.join([m.get('name', '未知') for m in manager_list]) if manager_list else '未知'
            
            # 申购/赎回状态
            buy_status_code = str(f_info_data.get('buy', '0'))
            buy_status = "开放申购" if buy_status_code == '1' else ("暂停申购" if buy_status_code == '0' else buy_status_code)
            
            sell_status_code = str(f_info_data.get('sell', '0'))
            sell_status = "开放赎回" if sell_status_code == '1' else ("暂停赎回" if sell_status_code == '0' else sell_status_code)


            result = {
                '代码': fund_code,
                '名称': f_s_data.get('name', '未知'),
                '类型': f_s_data.get('fundtype', '未知'),
                '成立日期': f_s_data.get('startdate', '未知'),
                '基金经理': manager,
                '公司名称': f_info_data.get('compname', '未知'),
                '基金简称': f_s_data.get('shortname', '未知'),
                '基金类型': f_info_data.get('fundType2', f_s_data.get('fundtype', '未知')),
                '发行时间': f_s_data.get('startdate', '未知'), # 通常与成立日期相同
                '基金规模': scale, # 新增字段
                '估值日期': f_s_data.get('gzdate', '未知'), # 估值日期 (如2024-01-01)
                '估值涨幅': f_s_data.get('gszzl', '未知'), # 估算涨幅 (如-0.01%)
                '累计净值': f_s_data.get('ljjz', '未知'),
                '单位净值': f_s_data.get('dwjz', '未知'),
                '申购净值': f_s_data.get('dwjz', '未知'), # 申购净值通常是单位净值
                '申购报价': f_s_data.get('gzdate', '未知'), # 申购报价字段不明确，暂时使用估值日期
                '净值日期': f_s_data.get('jzrq', '未知'),
                '申购状态': buy_status,
                '赎回状态': sell_status,
                '费率': buy_rate_display,
                '申购步长': f_s_data.get('sgbs', '未知')
            }
            
            if result.get('名称') in ['未知', 'undefined']:
                 raise ValueError("基金信息不完整或无效")

            print(f"   -> 基金 {fund_code} 基本信息抓取成功: {result.get('名称')}")
            return fund_code, result

        except Exception as e:
            print(f"   -> 基金 {fund_code} [信息抓取失败]: {e}")
            return fund_code, default_result

async def fetch_and_cache_fund_info(fund_codes):
    """主函数：检查缓存，对未缓存的基金进行并发抓取"""
    print("\n======== 开始抓取基金基本信息（静态数据）========\n")
    
    loop = asyncio.get_event_loop()
    info_cache = await loop.run_in_executor(None, load_info_cache)
    
    # 检查缓存中是否有缺失或失败的记录
    codes_to_fetch = [code for code in fund_codes if code not in info_cache or info_cache[code].get('名称') in ['未知', '抓取失败']]
    
    if not codes_to_fetch:
        print("所有基金的基本信息都已在缓存中。")
        return info_cache

    print(f"发现 {len(codes_to_fetch)} 个基金信息缺失，开始抓取...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async with ClientSession() as session:
        fetch_tasks = [fetch_fund_info(code, session, semaphore) for code in codes_to_fetch]
        
        # 使用 asyncio.as_completed 实时处理已完成的任务
        for future in asyncio.as_completed(fetch_tasks):
            try:
                code, result = await future
                info_cache[code] = result
            except Exception as e:
                print(f"[错误] 处理基本信息任务时发生错误: {e}")
                # 确保失败时也使用完整的默认字典，避免保存时缺少字段
                info_cache[code] = {
                    '代码': code, '名称': '抓取失败', '类型': '未知', '成立日期': '未知', 
                    '基金经理': '未知', '公司名称': '未知', '基金简称': '未知', '基金类型': '未知', 
                    '发行时间': '未知', '基金规模': '未知', '估值日期': '未知', '估值涨幅': '未知', 
                    '累计净值': '未知', '单位净值': '未知', '申购净值': '未知', '申购报价': '未知', 
                    '净值日期': '未知', '申购状态': '未知', '赎回状态': '未知', '费率': '未知', 
                    '申购步长': '未知'
                }
                
    # 保存更新后的缓存
    await loop.run_in_executor(None, save_info_cache, info_cache)
    print("\n======== 基金基本信息抓取完成 ========\n")
    return info_cache

# --------------------------------------------------------------------------------------
# 基金净值 (动态数据) 抓取和保存逻辑 - 保持不变
# --------------------------------------------------------------------------------------

def load_latest_date(fund_code):
    """从本地 CSV 文件中读取现有最新日期"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            if not df.empty and 'date' in df.columns:
                latest_date = df['date'].max().date()
                print(f"  -> 基金 {fund_code} 现有最新日期: {latest_date.strftime('%Y-%m-%d')}")
                return latest_date
        except Exception as e:
            print(f"  -> 加载 {fund_code} CSV 失败: {e}")
    return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_message(match='Frequency Capped')
)
async def fetch_page(session, url):
    """异步请求单页数据，带重试机制"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status == 514:
            raise aiohttp.ClientError("Frequency Capped")
        response.raise_for_status()
        return await response.text()

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
                soup = BeautifulSoup(text, 'lxml')
                
                if first_run:
                    total_pages_match = re.search(r'pages:(\d+)', text)
                    total_pages = int(total_pages_match.group(1)) if total_pages_match else 1
                    records_match = re.search(r'records:(\d+)', text)
                    total_records = int(records_match.group(1)) if records_match else '未知'
                    print(f"   基金 {fund_code} 信息：总页数 {total_pages}，总记录数 {total_records}。")
                    
                    if total_records == '未知' or int(total_records) == 0:
                        print(f"   基金 {fund_code} [跳过]：API 返回总记录数为 0 或未知。")
                        return fund_code, "API返回记录数为0或代码无效"
                        
                    first_run = False
                
                table = soup.find('table')
                if not table:
                    print(f"   基金 {fund_code} [警告]：页面 {page_index} 无表格数据。提前停止。")
                    break

                rows = table.find_all('tr')[1:]
                if not rows:
                    print(f"   基金 {fund_code} 第 {page_index} 页无数据行。停止抓取。")
                    break

                page_records = []
                stop_fetch = False
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 2:
                        continue
                    date_str = cols[0].text.strip()
                    net_value_str = cols[1].text.strip()
                    
                    if not date_str or not net_value_str or net_value_str == '-':
                        continue
                        
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        if latest_date and date <= latest_date:
                            stop_fetch = True
                            break
                            
                        page_records.append({'date': date_str, 'net_value': net_value_str})
                    except ValueError:
                        continue
                
                all_records.extend(page_records)
                
                if stop_fetch:
                    date_display = latest_date.strftime('%Y-%m-%d') if latest_date else '未知日期'
                    print(f"   基金 {fund_code} [增量停止]：页面 {page_index} 遇到旧数据 ({date_display})，停止抓取。")
                    break
                
                page_index += 1
                dynamic_delay = max(REQUEST_DELAY, dynamic_delay * 0.9)

            except aiohttp.ClientError as e:
                if "Frequency Capped" in str(e):
                    dynamic_delay = min(dynamic_delay * 2, 5.0)
                    print(f"   基金 {fund_code} [警告]：频率限制，延迟调整为 {dynamic_delay} 秒，重试第 {page_index} 页")
                    continue
                print(f"   基金 {fund_code} [错误]：请求 API 时发生网络错误 (超时/连接) 在第 {page_index} 页: {e}")
                return fund_code, f"网络错误: {e}"
            except Exception as e:
                print(f"   基金 {fund_code} [错误]：处理数据时发生意外错误在第 {page_index} 页: {e}")
                return fund_code, f"数据处理错误: {e}"
            
        print(f"-> [COMPLETE] 基金 {fund_code} 数据抓取完毕，共获取 {len(all_records)} 条新记录。")
        if not all_records:
            return fund_code, "数据已是最新，无新数据"
        return fund_code, all_records

def save_to_csv(fund_code, data):
    """将历史净值数据以增量更新方式保存为 CSV 文件，格式为 date,net_value"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if not isinstance(data, list) or not data:
        print(f"   基金 {fund_code} 无新数据可保存。")
        return False, 0

    new_df = pd.DataFrame(data)

    try:
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df.dropna(subset=['date', 'net_value'], inplace=True)
        if new_df.empty:
            print(f"   基金 {fund_code} 数据无效或为空，跳过保存。")
            return False, 0
    except Exception as e:
        print(f"   基金 {fund_code} 数据转换失败: {e}")
        return False, 0
    
    old_record_count = 0
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['date'], dtype={'net_value': float}, encoding='utf-8')
            old_record_count = len(existing_df)
            combined_df = pd.concat([new_df, existing_df])
        except Exception as e:
            print(f"   读取现有 CSV 文件 {output_path} 失败: {e}。仅保存新数据。")
            combined_df = new_df
    else:
        combined_df = new_df
        
    final_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    final_df = final_df.sort_values(by='date', ascending=False)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保目录存在
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        new_record_count = len(final_df)
        newly_added = new_record_count - old_record_count
        print(f"   -> 基金 {fund_code} [保存完成]：总记录数 {new_record_count} (新增 {max(0, newly_added)} 条)。")
        return True, max(0, newly_added)
    except Exception as e:
        print(f"   基金 {fund_code} 保存 CSV 文件 {output_path} 失败: {e}")
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
            try:
                fund_code, net_values = await future
            except Exception as e:
                print(f"[错误] 处理基金数据时发生顶级异步错误: {e}")
                failed_codes.append("未知基金代码")
                continue

            if isinstance(net_values, list):
                try:
                    # 使用 run_in_executor 在线程池中执行同步的 pandas/文件I/O操作
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
                print(f"   基金 {fund_code} [抓取失败/跳过]：{net_values}")
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
        print(f"失败的基金代码: {', '.join(failed_codes)}")
    if total_new_records == 0:
        print("[警告] 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")

if __name__ == "__main__":
    main()
