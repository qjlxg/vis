import requests
import re
import json
import pandas as pd
import os

# --- 配置 ---
# 东方财富网基金代码列表URL
url = 'http://fund.eastmoney.com/js/fundcode_search.js'
# 设置 User-Agent 避免被拒绝
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
# 输出文件名
output_file = 'C类.txt'

def fetch_and_filter_funds():
    """
    从东方财富获取所有基金代码，筛选出场外 C 类基金，并保存到文件。
    """
    try:
        print(f"1. 正在从 {url} 获取基金代码数据...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 检查 HTTP 错误

        text = response.text

        # 提取 JSON 数据（去除 js 包装）
        # 匹配格式如：var r = [["000001","HXCZ","华夏成长","混合型","H","huaxiacaizhi"],...]
        match = re.search(r'var r = (\[.*?\]);', text, re.DOTALL)
        if not match:
            print("错误：未找到基金代码列表数据结构。")
            return

        funds = json.loads(match.group(1))

        # 转换为 DataFrame
        # 字段含义: [代码, 拼音简称, 基金名称, 类型, 拼音首字母, 全拼]
        df = pd.DataFrame(funds, columns=['code', 'pinyin', 'name', 'type', 'initial', 'full_pinyin'])
        
        # 确保代码是纯数字
        df['code'] = df['code'].str.extract(r'(\d+)').astype(str)

        print("2. 正在筛选场外 C 类基金...")
        
        # 筛选逻辑：
        # 1. 基金名称包含 'C'（通常代表C类份额）
        # 2. 基金类型不包含 '场内'、'ETF'、'LOF' 或 '联接'（排除场内交易、联接基金以聚焦场外申购的C类）
        c_df = df[
            df['name'].str.contains('C', na=False) & 
            ~df['type'].str.contains('场内|ETF|LOF|联接', na=False, case=False)
        ]

        # 提取代码列表
        codes = c_df['code'].tolist()
        
        # 移除重复代码
        codes = sorted(list(set(codes)))

        # 检查 codes 列表是否为空
        if not codes:
            print("警告：筛选结果为空，没有找到匹配的场外 C 类基金代码。")
            return

        # 保存 TXT
        print(f"3. 筛选完成，共找到 {len(codes)} 个场外 C 类基金代码。")
        
        # 确保目录存在（如果需要保存到子目录，但这里保存到根目录）
        # os.makedirs(os.path.dirname(output_file), exist_ok=True) 

        with open(output_file, 'w', encoding='utf-8') as f:
            for code in codes:
                f.write(code + '\n')

        print(f"4. 成功保存代码到 {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except json.JSONDecodeError:
        print("错误：JSON 数据解析失败。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == '__main__':
    fetch_and_filter_funds()
