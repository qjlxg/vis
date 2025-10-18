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
FORCE_UPDATE = False  # 是否强制重新抓取

# 每次运行脚本最多处理的基金代码数量 (0 表示不限制)
MAX_FUNDS_PER_RUN = 0  

# 检查基金数据新鲜度的阈值（如果最新日期比今天早 X 天，就强制从头（page=1）开始爬取）
# 调整为 5 天，更好地覆盖周末和节假日。
FRESHNESS_CHECK_DAYS = 5 
PAGE_SIZE = 20 # 每页记录数

def get_all_fund_codes(file_path):
    """从 C类.txt 文件中读取基金代码（单列无标题，UTF-8 编码） - 保持不变"""
    # ... (与之前代码保持一致)
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
# 核心优化：直接计算起始页并进行新鲜度检查
# --------------------------------------------------------------------------------------

def calculate_start_page(fund_code):
    """
    计算基金数据的起始抓取页码。
    返回: int (1 或基于本地记录数计算的增量页数)
    """
    csv_file = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    
    if FORCE_UPDATE or not os.path.exists(csv_file):
        print(f"  -> [START_CALC] 基金 {fund_code}：强制更新或无本地文件，从第 1 页开始抓取。")
        return 1

    try:
        # 1. 检查新鲜度
        local_df = pd.read_csv(csv_file, parse_dates=['date'], encoding='utf-8')
        
        if local_df.empty:
            print(f"  -> [START_CALC] 基金 {fund_code}：本地 CSV 为空，从第 1 页开始抓取。")
            return 1
            
        local_latest_date = local_df['date'].max()
        today = datetime.now().date()
        date_threshold = today - timedelta(days=FRESHNESS_CHECK_DAYS)

        if local_latest_date.date() < date_threshold:
            print(f"  -> [START_CALC] 基金 {fund_code} 判定为【过期】：最新日期 {local_latest_date.date()} 早于阈值 {date_threshold}。从第 1 页开始抓取（利用去重机制高效更新）。")
            return 1 # 数据过期，从头开始，让 save_to_csv 处理增量和去重
        else:
            # 2. 数据新鲜，计算增量更新的起始页码
            total_records = len(local_df)
            
            # 计算总页数。Python 的 // 是向下取整
            # 100条记录：100/20=5页。 101条记录：101/20=5.05 -> 5+1=6页
            total_pages_local = (total_records + PAGE_SIZE - 1) // PAGE_SIZE
            
            # 回退 2 页以确保获取最新变动，且至少从第 1 页开始
            start_page = max(1, total_pages_local - 1) 
            
            # 打印增量更新信息，使用新的起始页码
            print(f"  -> [START_CALC] 基金 {fund_code}：数据新鲜 (最新 {local_latest_date.date()})，本地 {total_records} 条记录（约 {total_pages_local} 页）。从第 {start_page} 页开始抓取（增量更新，回退 2 页）。")
            return start_page

    except Exception as e:
        print(f"  -> [START_CALC] 基金 {fund_code} 检查失败/文件损坏: {e}。从第 1 页开始抓取。")
        return 1
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
    """异步获取指定基金代码的【所有】历史净值数据 (分页，异步)"""
    print(f"-> [START] 基金代码 {fund_code}")
    
    # 核心修改：在抓取任务开始前，计算起始页码
    # 使用 loop.run_in_executor 确保文件 I/O 不阻塞异步循环
    loop = asyncio.get_event_loop()
    page_index = await loop.run_in_executor(None, calculate_start_page, fund_code)

    async with semaphore:
        all_records = []
        total_pages = 1
        first_run = True
        dynamic_delay = REQUEST_DELAY

        # 如果计算出来的起始页大于 API 总页数，将导致跳过，但通常 API 首次响应会校正 total_pages

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
                    
                    if page_index > total_pages:
                        print(f"   基金 {fund_code} [跳过]：计算的起始页 ({page_index}) > API总页数 ({total_pages})。")
                        return fund_code, f"增量跳过: 起始页 {page_index} > 总页数 {total_pages}"
                        
                    first_run = False
                
                table = soup.find('table') 
                if not table:
                    # 如果当前页不是 API 报告的最后一页，则可能是错误
                    if page_index < total_pages:
                         print(f"   基金 {fund_code} 第 {page_index} 页未找到表格数据，提前停止。")
                    break

                rows = table.find_all('tr')[1:] 
                if not rows and page_index < total_pages:
                    print(f"   基金 {fund_code} 第 {page_index} 页无数据行，提前停止。")
                    break

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        date_str = cols[0].text.strip()
                        net_value_str = cols[1].text.strip() 
                        
                        if date_str and net_value_str and net_value_str != '-':
                            all_records.append({'date': date_str, 'net_value': net_value_str})
                
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
        return fund_code, all_records

def save_to_csv(fund_code, data):
    """将历史净值数据以增量更新方式保存为 CSV 文件，格式为 date,net_value - 保持不变"""
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
            # 确保读取时也用正确的格式，防止去重逻辑失败
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
        # 如果 old_record_count 不准确（比如文件读取失败），newly_added 可能出错，但总数是准确的。
        print(f"   -> 基金 {fund_code} [保存完成]：总记录数 {new_record_count} (新增/更新 {newly_added} 条)。")
        return True, newly_added
    except Exception as e:
        print(f"   基金 {fund_code} 保存 CSV 文件 {output_path} 失败: {e}")
        return False, 0

async def fetch_all_funds(fund_codes):
    """异步获取所有基金数据，并在任务完成时立即保存数据"""
    
    # 移除原有的新鲜度检查阶段，因为计算已移入 fetch_net_values 内部
    print("\n======== 开始基金数据抓取（自动计算起始页，无需预处理） ========\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    loop = asyncio.get_event_loop()

    async with ClientSession() as session:
        # 创建所有异步抓取任务 (fetch_net_values 会在内部计算起始页)
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
                    if not str(net_values).startswith('增量跳过'):
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
        print(f"失败的基金代码: {', '.join(failed_codes)}")
    if total_new_records == 0:
        print("警告: 未新增任何记录，可能是数据已是最新，或 API 无新数据。")
    print(f"==============================")


if __name__ == "__main__":
    main()
