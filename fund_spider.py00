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
BASE_URL = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"

# 设置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/533.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30 
REQUEST_DELAY = 0.5  # 初始延迟，动态调整
MAX_CONCURRENT = 5  # 最大并发基金数量
FORCE_UPDATE = False

# 调试修复后的代码，暂时限制处理数量
MAX_FUNDS_PER_RUN = 0  

# 检查基金数据新鲜度的阈值（不再用于强制从头开始，逻辑已简化）
FRESHNESS_CHECK_DAYS = 5 
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

# ****** 核心修改：新的增量起始点判断逻辑 ******

def load_latest_date(fund_code):
    """
    [修复点 A]：从本地 CSV 文件中读取现有最新日期，并返回纯 Python 的 datetime.date 对象
    """
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
    return None  # 无缓存，或加载失败，从头抓取


# --------------------------------------------------------------------------------------


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
        page_index = 1  # 永远从第 1 页开始抓取
        total_pages = 1
        first_run = True
        dynamic_delay = REQUEST_DELAY
        
        loop = asyncio.get_event_loop()
        # 在线程中同步读取 CSV，latest_date 此时是 datetime.date 对象
        latest_date = await loop.run_in_executor(None, load_latest_date, fund_code) 


        while page_index <= total_pages:
            url = BASE_URL.format(fund_code=fund_code, page_index=page_index, per=PAGE_SIZE)
            
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
                        # [修复点 B]：转换日期，并获取纯 Python 的 date 对象进行比较
                        date = datetime.strptime(date_str, '%Y-%m-%d').date() 
                        
                        # ****** 核心停止条件 ******
                        # 此时 date 和 latest_date 都是 datetime.date 对象，可以安全比较
                        if latest_date and date <= latest_date: 
                            stop_fetch = True
                            break # 退出 for row 循环
                            
                        # 如果是新数据，添加到列表 (注意：添加到列表时使用原始的 date_str 字符串，方便 DataFrame 处理)
                        page_records.append({'date': date_str, 'net_value': net_value_str})
                    except ValueError:
                        continue # 日期或净值格式错误，跳过该行
                
                all_records.extend(page_records)
                
                if stop_fetch:
                    print(f"   基金 {fund_code} [增量停止]：页面 {page_index} 遇到旧数据 ({latest_date.strftime('%Y-%m-%d')})，停止抓取。")
                    break # 退出 while page_index 循环
                
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
                # 捕获其他所有错误，包括我们修复的日期处理错误
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
        
    # 去重：以 date 为准，保留最新的净值记录 (keep='first' 是关键)
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
    
    print("\n======== 开始基金数据抓取（基于最新日期实现智能增量更新） ========\n")

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
                            
                    except concurrent.futures.CancelledError:
                        print(f"基金 {fund_code} 的保存任务被取消。")
                        failed_codes.append(fund_code)
                    except Exception as e:
                        print(f"基金 {fund_code} 的保存任务在线程中发生错误: {e}")
                        failed_codes.append(fund_code)
                        
                else:
                    print(f"   基金 {fund_code} [抓取失败/跳过]：{net_values}")
                    # 如果不是提示“数据已是最新”的信息，则计入失败
                    if not str(net_values).startswith('数据已是最新'):
                         failed_codes.append(fund_code)

        return success_count, total_new_records, failed_codes

def main():
    """主函数"""
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

    print(f"找到 {len(processed_codes)} 个基金代码，开始获取历史净值...")
    
    success_count, total_new_records, failed_codes = asyncio.run(fetch_all_funds(processed_codes))
    
    # 打印总结
    print(f"\n======== 本次更新总结 ========")
    print(f"本次基金历史净值数据获取和保存完成。")
    print(f"总结: 成功处理 {success_count} 个基金，新增/更新 {total_new_records} 条记录，失败 {len(failed_codes)} 个基金。")
    if failed_codes:
        print(f"失败的基金代码 (本次调试只尝试处理前 {MAX_FUNDS_PER_RUN} 个，如果错误在此，请检查网络或代码有效性): {', '.join(failed_codes)}")
    if total_new_records == 0:
        print("警告: 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")


if __name__ == "__main__":
    main()
