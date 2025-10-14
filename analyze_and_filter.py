import pandas as pd
import json
import os
import chardet

def detect_file_encoding(file_path):
    """检测文件的编码"""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            print(f"检测到文件 {file_path} 的编码：{result['encoding']}（置信度：{result['confidence']})")
            return result['encoding']
    except Exception as e:
        print(f"检测文件编码时发生错误：{e}")
        return 'gb18030'  # 默认回退到 gb18030

def analyze_and_recommend_funds():
    """
    该脚本读取本地文件，整合数据并筛选出符合条件的基金。
    """
    print("--- 正在加载并分析文件，为您筛选最终结果 ---")

    try:
        # 1. 读取所有文件
        selected_indices_df = pd.read_csv('selected_low_valuation_indices.csv', encoding='utf-8')
        
        # 检测 recommended_cn_funds.csv 的编码
        csv_file = 'recommended_cn_funds.csv'
        encoding = detect_file_encoding(csv_file)
        try:
            recommended_funds_df = pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError:
            print(f"使用编码 {encoding} 失败，尝试使用 'gb18030'")
            recommended_funds_df = pd.read_csv(csv_file, encoding='gb18030')
        
        with open('comprehensive_fund_analysis.json', 'r', encoding='utf-8') as f:
            comprehensive_analysis_json = json.load(f)

        # 调试：打印列名以确认
        print("recommended_cn_funds.csv 的列名：", recommended_funds_df.columns.tolist())

        # 2. 从低估指数列表中提取行业关键词
        low_valuation_indices = selected_indices_df['指数名称'].tolist()
        keywords = []
        for index_name in low_valuation_indices:
            if '白酒' in index_name:
                keywords.append('白酒')
            if '食品' in index_name:
                keywords.append('食品')
            if '消费' in index_name:
                keywords.append('消费')
        keywords = list(set(keywords))

        if not keywords:
            print("未在低估指数文件中找到相关行业关键词（白酒、食品、消费）。")
            return

        print(f"✅ 已识别出低估行业关键词：{', '.join(keywords)}")

        # 3. 筛选出同时符合低估行业和四四三三法则的基金
        matching_funds = []
        for index, row in recommended_funds_df.iterrows():
            if 'name' not in row or 'code' not in row:
                print(f"警告：第 {index} 行缺少 'name' 或 'code' 列，跳过此行。")
                continue
            fund_name = row['name']
            fund_code = row['code']
            # 将基金代码格式化为6位数
            fund_code_str = str(fund_code).zfill(6)
            if any(k in fund_name for k in keywords):
                matching_funds.append({
                    'code': fund_code_str,
                    'name': fund_name,
                    '排名信息': row.to_dict()
                })

        if not matching_funds:
            print("❌ 未在推荐基金列表中找到任何与低估行业匹配的基金。")
            return

        print(f"✅ 找到 {len(matching_funds)} 只同时符合双重标准的基金。")
        # 调试：打印匹配的基金代码
        print("初步筛选的基金代码：", [fund['code'] for fund in matching_funds])

        # 4. 从综合分析文件中提取详细信息并整合
        analysis_data_dict = {item['fund_code']: item for item in comprehensive_analysis_json}
        # 调试：打印 JSON 文件中的基金代码
        print("JSON 文件中的基金代码：", list(analysis_data_dict.keys()))

        final_recommendations = []
        for fund in matching_funds:
            fund_code = fund['code']
            if fund_code in analysis_data_dict:
                details = analysis_data_dict[fund_code]

                # 提取前五大持仓
                top_holdings = [
                    {'股票名称': h.get('stock_name'), '持仓占比': h.get('proportion')}
                    for h in details.get('fund_holdings', [])[:5]
                ]

                # 提取风险指标
                risk_metrics = details.get('risk_metrics', {})

                # 确保风险指标是数字类型以便格式化，否则使用'N/A'
                sharpe_ratio_formatted = f"{risk_metrics.get('sharpe_ratio', 'N/A'):.2f}" if isinstance(risk_metrics.get('sharpe_ratio'), (int, float)) else 'N/A'
                max_drawdown_formatted = f"{risk_metrics.get('max_drawdown', 'N/A') * 100:.2f}%" if isinstance(risk_metrics.get('max_drawdown'), (int, float)) else 'N/A'
                
                # 提取基金经理信息，增加健壮性检查
                fund_manager = 'N/A'
                if details.get('fund_managers') and isinstance(details['fund_managers'], list) and details['fund_managers'] and 'fund_managers' in details['fund_managers'][0]:
                    fund_manager = details['fund_managers'][0]['fund_managers']

                final_recommendations.append({
                    '基金代码': fund_code,
                    '基金名称': fund['name'],
                    '匹配行业': ', '.join([k for k in keywords if k in fund['name']]),
                    '前五大持仓': top_holdings,
                    '年化夏普比率': sharpe_ratio_formatted,
                    '最大回撤': max_drawdown_formatted,
                    '基金经理': fund_manager
                })
            else:
                print(f"警告：基金代码 {fund_code} 未在 JSON 文件中找到匹配数据。")

        # 5. 打印最终结果
        print("\n--- 最终筛选结果：可供参考和购买的基金 ---")
        if not final_recommendations:
            print("没有找到符合所有条件的基金。")
            print("可能原因：筛选出的基金代码与 JSON 文件中的 fund_code 不匹配。")
        else:
            for item in final_recommendations:
                print("\n" + "="*50)
                print(f"基金名称：{item['基金名称']} ({item['基金代码']})")
                print(f"匹配行业：{item['匹配行业']}")
                print(f"基金经理：{item['基金经理']}")
                print("核心风险指标：")
                print(f"  - 年化夏普比率：{item['年化夏普比率']}")
                print(f"  - 最大回撤：{item['最大回撤']}")
                print("前五大持仓：")
                if item['前五大持仓']:
                    for holding in item['前五大持仓']:
                        proportion_str = f"{holding.get('持仓占比')}%" if holding.get('持仓占比') is not None else 'N/A'
                        print(f"  - {holding.get('股票名称', 'N/A')} (占比: {proportion_str})")
                else:
                    print("  - 暂无持仓数据")

    except FileNotFoundError as e:
        print(f"错误：缺少必要的数据文件。请确保以下文件存在于同一目录下：{e.filename}")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    analyze_and_recommend_funds()
