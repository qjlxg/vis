import requests
import re
import json
import time
import os

def fetch_fund_data():
    """
    从东方财富获取最新的基金代码数据。
    该数据以JavaScript变量r = [...]的形式存在。
    """
    url = "http://fund.eastmoney.com/js/fundcode_search.js"
    headers = {
        # 模拟浏览器访问，防止被拒绝
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 打印提示信息
    print(f"--- 正在从 {url} 获取基金代码数据... ---")
    
    try:
        # 发起HTTP请求，设置超时
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # 如果状态码不是200，则抛出异常
        content = response.text
        
        # 使用正则表达式匹配并提取JavaScript变量 r 的内容，即 JSON 数组
        # r = [[...], [...]];
        match = re.search(r'var r = (\[.*?\]);', content, re.DOTALL)
        
        if not match:
            print("错误: 无法在响应内容中找到基金数据数组 'r'。")
            return None
            
        json_string = match.group(1)
        
        # 将提取的 JSON 字符串解析为 Python 列表
        fund_data = json.loads(json_string)
        return fund_data

    except requests.exceptions.RequestException as e:
        print(f"请求数据时发生网络错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"解析 JSON 数据时发生格式错误: {e}")
        print("提示: 数据格式可能已改变，请检查原始文件内容。")
        return None

def filter_c_funds(fund_data):
    """
    过滤基金列表，找出名称以 'C' 结尾的 C 类基金代码。
    基金数据格式通常为: [代码(0), 拼音缩写(1), 基金名称(2), 基金类型(3), 拼音全称(4), ...]
    """
    if not fund_data:
        return []
        
    c_fund_codes = []
    
    # 基金名称位于索引 2
    NAME_INDEX = 2
    
    for row in fund_data:
        # 确保行中至少有3个元素，并且基金名称是字符串类型
        if len(row) > NAME_INDEX and isinstance(row[NAME_INDEX], str):
            fund_name = row[NAME_INDEX].strip()
            # 检查基金名称是否以 'C' 结尾
            if fund_name.endswith('C'):
                # 基金代码始终是第一个元素 (索引 0)
                fund_code = row[0]
                c_fund_codes.append(fund_code)
            
    return c_fund_codes

def save_codes_to_file(codes, filename="C类.txt"):
    """将基金代码列表保存到文本文件中。"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for code in codes:
                f.write(f"{code}\n")
        print(f"--- 成功筛选出 {len(codes)} 个 C 类基金代码，并保存到文件: {filename} ---")
    except IOError as e:
        print(f"写入文件时发生错误: {e}")

if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 获取所有基金数据
    all_funds = fetch_fund_data()
    
    if all_funds:
        # 2. 筛选 C 类基金代码
        c_codes = filter_c_funds(all_funds)
        
        # 3. 保存结果
        save_codes_to_file(c_codes, "C0类.txt")
    
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")
