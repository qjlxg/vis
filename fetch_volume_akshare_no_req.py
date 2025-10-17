import requests
import json
import pandas as pd
import datetime
import os
import time

# 需要获取的指数代码及其在东方财富接口中的标识 (市场代码.指数代码)
# 接口的fs参数定义了数据来源，这里使用 m:1+t:2 (A股指数) 和 m:1+t:23 (A股指数)
# 指数代码 (f12) 和 市场代码 (f13) 决定了数据请求
INDEX_CODES = {
    "000001": {"name": "上证指数", "market": "1"},  # 上交所 1.000001
    "399001": {"name": "深证成指", "market": "0"},  # 深交所 0.399001
    "399006": {"name": "创业板指", "market": "0"},  # 深交所 0.399006
    "000300": {"name": "沪深300",  "market": "1"}   # 上交所 1.000300
}

def fetch_index_volume(code, market, name):
    """
    通过东方财富指数接口获取指定指数的最新成交量。
    """
    try:
        # 东方财富实时行情接口 (与你提供的接口类似，但针对指数)
        url = (
            "http://push2.eastmoney.com/api/qt/ulist.np/get?"
            "fltt=2&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f20,f21"
            "&secids={0}.{1}"  # 格式: 市场代码.指数代码
            "&_={2}"
        ).format(market, code, int(time.time() * 1000))
        
        r = requests.get(url, timeout=10)
        r.encoding = 'utf-8'
        content_dict = r.json()

        data = content_dict.get('data', {}).get('diff', [])
        
        if not data:
            print(f"  -> 接口未返回 {name} ({code}) 的数据。")
            return None
        
        # 假设返回列表的第一个元素就是所需数据
        index_data = data[0]
        
        # f5: 成交量 (手)
        volume = index_data.get('f5')
        
        # f1: 最新时间戳 (通常是 Unix 时间戳，但可能不稳定，这里不取)
        # 简单使用运行脚本的日期作为数据的交易日期（需在收盘后运行）
        today = datetime.datetime.now().strftime('%Y%m%d')

        if volume is None:
             print(f"  -> 成功获取数据但缺少成交量 (f5) 字段。")
             return None
        
        return {
            '交易日期': today,
            '指数代码': f"{market}.{code}",
            '指数名称': name,
            '成交量(手)': volume
        }
    except Exception as e:
        print(f"获取 {name} ({code}) 数据时出错: {e}")
        return None

def main():
    """主函数，获取所有指数数据并保存到指定路径"""
    print("开始获取指数成交量数据...")
    
    now = datetime.datetime.now()
    timestamp_str = now.strftime('%Y%m%d_%H%M%S')
    
    # 构造保存目录 (年月)
    year_month = now.strftime('%Y%m')
    output_dir = os.path.join(os.getcwd(), year_month)
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造文件名
    output_filename = f"index_volume_{timestamp_str}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    results = []
    
    for code, info in INDEX_CODES.items():
        name = info['name']
        market = info['market']
        print(f"正在获取 {name} ({market}.{code})...")
        
        # 调用爬虫函数
        data = fetch_index_volume(code, market, name)
        
        if data:
            results.append(data)
        
        # 增加延迟以避免对接口造成过大压力
        time.sleep(1) 
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results[['交易日期', '指数代码', '指数名称', '成交量(手)']]

        # 保存到 CSV 文件
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n数据已成功保存到: {output_path}")
    else:
        print("\n所有指数数据获取失败，未生成文件。")

if __name__ == "__main__":
    main()
