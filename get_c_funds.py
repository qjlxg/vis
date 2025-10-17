import requests
import re
import json
import pandas as pd

url = 'http://fund.eastmoney.com/js/fundcode_search.js'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
text = response.text

# 提取JSON数据（去除js包装）
data_str = re.findall(r'var (.*)=$$ (.*?) $$;', text)[0][1]
funds = json.loads('[' + data_str + ']')

# 转换为DataFrame
df = pd.DataFrame(funds, columns=['code', 'name', 'type', 'other'])
df['code'] = df['code'].str.extract(r'(\d+)').astype(str)

# 筛选场外C类（name含'C'，type非'场内'如etf/lof）
c_df = df[df['name'].str.contains('C', na=False) & ~df['type'].str.contains('场内|etf|lof', na=False)]

# 保存TXT
codes = c_df['code'].tolist()
with open('C0类.txt', 'w', encoding='utf-8') as f:
    for code in codes:
        f.write(code + '\n')

print(f"获取{len(codes)}个场外C类基金代码。")
