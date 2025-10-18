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

# 定义文件路径和目录
INPUT_FILE = 'C类.txt'
OUTPUT_DIR = 'fund_data'
# 基金净值 API
BASE_URL_NET_VALUE = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
# 基金详情页 URL
BASE_URL_INFO = "http://fund.eastmoney.com/{fund_code}.html"
# 基本信息缓存文件
INFO_CACHE_FILE = 'fund_info.json'


# 设置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/533.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30 
REQUEST_DELAY = 0.5  # 初始延迟，动态调整
MAX_CONCURRENT = 5  # 最大并发基金数量

# 调试修复后的代码，暂时限制处理数量 (请根据需要改回 0)
MAX_FUNDS_PER_RUN = 10  

PAGE_SIZE = 20 # 每页记录数

def get_all_fund_codes(file_path):
    """从 C类.txt 文件中读取基金代码（单列无标题，UTF-8 编码）"""
    print(f"尝试读取基金代码文件: {file_path}")
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
        print("  -> 无法读取文件，请检查文件格式和编码。")
        return []

    codes = df['code'].dropna().astype(str).unique().tolist()
    valid_codes = [code for code in codes if code.isdigit() and len(code) >= 3]
    print(f"  -> 找到 {len(valid_codes)} 个有效基金代码。")
    return valid_codes

# --------------------------------------------------------------------------------------
# 基金基本信息 (静态数据) 抓取和缓存逻辑
# --------------------------------------------------------------------------------------

def load_info_cache():
    """加载基金基本信息缓存"""
    if os.path.exists(INFO_CACHE_FILE):
        try:
            with open(INFO_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[警告] 加载基本信息缓存失败: {e}。将从头抓取。")
            return {}
    return {}

def save_info_cache(cache):
    """保存基金基本信息缓存"""
    try:
        with open(INFO_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
        print(f"  -> 成功保存基本信息缓存到 {INFO_CACHE_FILE}")
    except Exception as e:
        print(f"[错误] 保存基本信息缓存失败: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_message(match='Frequency Capped')
)
async def fetch_html_page(session, url):
    """异步请求 HTML 页面，带重试机制"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status == 514:
            raise aiohttp.ClientError("Frequency Capped")
        response.raise_for_status()
        return await response.text()

async def fetch_fund_info(fund_code, session, semaphore):
    """异步抓取基金基本信息"""
    url = BASE_URL_INFO.format(fund_code=fund_code)
    async with semaphore:
        try:
            # 抓取页面
            html = await fetch_html_page(session, url)
            soup = BeautifulSoup(html, 'lxml')
            
            # 1. 基金名称
            name_tag = soup.find('div', class_='fundDetail-tit')
            fund_name = name_tag.find('div').text.strip() if name_tag else '未知名称'
            
            # 2. 详情表格 (基金类型、成立日期、管理人等)
            info_table = soup.find('table', class_='info w790')
            info = {}
            if info_table:
                # 提取表格中键值对
                for row in info_table.find_all('tr'):
                    cells = row.find_all(['th', 'td'])
                    if len(cells) >= 2:
                        key = cells[0].text.strip().replace('：', '').replace(':', '')
                        value = cells[1].text.strip()
                        info[key] = value

            # 解析关键字段
            fund_type = info.get('基金类型', '未知')
            establish_date = info.get('成 立 日', '未知')
            manager = info.get('基金管理人', '未知')
            
            result = {
                'code': fund_code,
                'name': fund_name,
                'type': fund_type,
                'establish_date': establish_date,
                'manager': manager
            }
            return fund_code, result

        except Exception as e:
            print(f"   基金 {fund_code} [信息抓取失败]：{e}")
            return fund_code, f"信息抓取失败: {e}"

async def fetch_and_cache_fund_info(fund_codes):
    """主函数：检查缓存，对未缓存的基金进行并发抓取"""
    print("\n======== 开始抓取基金基本信息（静态数据）========\n")
    
    loop = asyncio.get_event_loop()
    # 同步加载缓存
    info_cache = await loop.run_in_executor(None, load_info_cache)
    
    codes_to_fetch = []
    
    # 检查哪些基金信息缺失或需要更新 (对于静态信息，只抓取一次)
    for code in fund_codes:
        if code not in info_cache or info_cache.get(code, {}).get('name') in ['未知名称', '']:
            codes_to_fetch.append(code)

    if not codes_to_fetch:
        print("所有基金的基本信息都已在缓存中。")
        return info_cache

    print(f"发现 {len(codes_to_fetch)} 个基金信息缺失，开始抓取...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async with ClientSession() as session:
        fetch_tasks = [fetch_fund_info(code, session, semaphore) for code in codes_to_fetch]
        
        for future in asyncio.as_completed(fetch_tasks):
            try:
                code, result = await future
                if isinstance(result, dict):
                    info_cache[code] = result
                else:
                    # 抓取失败，仍然将代码添加到缓存中，防止下次重复尝试（可以设置标记）
                    info_cache[code] = {"code": code, "name": "抓取失败"}
            except Exception as e:
                print(f"处理基本信息任务时发生错误: {e}")
                
    # 同步保存更新后的缓存
    await loop.run_in_executor(None, save_info_cache, info_cache)
    print("\n======== 基金基本信息抓取完成 ========\n")
    return info_cache

# --------------------------------------------------------------------------------------
# 基金净值 (动态数据) 抓取和保存逻辑
# --------------------------------------------------------------------------------------

def load_latest_date(fund_code):
    """从本地 CSV 文件中读取现有最新日期，并返回纯 Python 的 datetime.date 对象"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            if not df.empty:
                # 关键修复：使用 .date() 获取纯 Python 的 date 对象，而不是 .normalize()
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
    """
    使用“最新日期”作为停止条件，实现智能增量更新。
    从 Page 1 开始抓取，遇到已有数据即停止。
    """
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
            url = BASE_URL_NET_VALUE.format(fund_code=fund_code, page_index=page_index, per=PAGE_SIZE)
            
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
                    if len(cols) < 2: continue
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
                    print(f"   基金 {fund_code} [增量停止]：页面 {page_index} 遇到旧数据 ({latest_date.strftime('%Y-%m-%d')})，停止抓取。")
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
                result = await future 
            except Exception as e:
                print(f"处理基金数据时发生顶级异步错误: {e}")
                failed_codes.append("未知基金代码")
                continue

            if isinstance(result, tuple) and len(result) == 2:
                fund_code, net_values = result
                
                if isinstance(net_values, list):
                    try:
                        success, new_records = await loop.run_in_executor(
                            None, 
                            save_to_csv, 
                            fund_code, 
                            net_values
                        )
                        
                        if success:
                            success_count += 1
                            total_new_records += new_records
                        else:
                            failed_codes.append(fund_code)
                            
                    except Exception as e:
                        print(f"基金 {fund_code} 的保存任务在线程中发生错误: {e}")
                        failed_codes.append(fund_code)
                        
                else:
                    print(f"   基金 {fund_code} [抓取失败/跳过]：{net_values}")
                    if not str(net_values).startswith('数据已是最新'):
                         failed_codes.append(fund_code)

        return success_count, total_new_records, failed_codes

def main():
    """主函数：集成静态信息抓取和动态净值抓取"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    fund_codes = get_all_fund_codes(INPUT_FILE)
    if not fund_codes:
        print("没有可处理的基金代码，脚本结束。")
        return

    # 限制本次运行处理的基金数量
    if MAX_FUNDS_PER_RUN > 0 and len(fund_codes) > MAX_FUNDS_PER_RUN:
        print(f"限制本次运行最多处理 {MAX_FUNDS_PER_RUN} 个基金。")
        processed_codes = fund_codes[:MAX_FUNDS_PER_RUN]
        print(f"本次实际处理的基金数量: {len(processed_codes)}")
    else:
        processed_codes = fund_codes
        print(f"本次处理所有 {len(processed_codes)} 个基金。")
        
    print(f"找到 {len(processed_codes)} 个基金代码，开始获取数据...")
    
    # 1. 抓取并缓存静态信息 (fund_info.json)
    # asyncio.run(fetch_and_cache_fund_info(processed_codes)) # 可以在单独的 run 中执行

    # 由于 main 是同步的，为了方便，将静态信息抓取也集成到同步环境中
    # 为了避免嵌套 asyncio.run，我们将 fetch_and_cache_fund_info 放在 fetch_all_funds 的前面
    loop = asyncio.get_event_loop()
    loop.run_until_complete(fetch_and_cache_fund_info(processed_codes))

    # 2. 抓取动态净值数据 (.csv)
    success_count, total_new_records, failed_codes = loop.run_until_complete(fetch_all_funds(processed_codes))
    
    # 打印总结
    print(f"\n======== 本次更新总结 ========")
    print(f"本次基金历史净值数据获取和保存完成。")
    print(f"总结: 成功处理 {success_count} 个基金，新增/更新 {total_new_records} 条记录，失败 {len(failed_codes)} 个基金。")
    if failed_codes:
        print(f"失败的基金代码: {', '.join(failed_codes)}")
    if total_new_records == 0:
        print("警告: 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")


if __name__ == "__main__":
    main()
