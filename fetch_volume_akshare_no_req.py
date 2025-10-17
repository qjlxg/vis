import akshare as ak
import pandas as pd
import datetime
import os
import time

# 需要获取的指数代码及其在 akshare 中的代码
INDEX_CODES = {
    "sh000001": "上证指数",     # 000001
    "sz399001": "深证成指",     # 399001
    "sz399006": "创业板指",     # 399006
    "sh000300": "沪深300"      # 000300
}

def fetch_index_volume(symbol_code, index_name):
    """
    使用 akshare 获取指定指数代码最近一个交易日的成交量数据
    """
    try:
        # 修正：移除 'adjust="qfq"' 参数，因为它在新版本中不再被支持。
        # 仅保留 symbol, period, start_date, end_date 参数。
        # start_date="" 和 end_date="" 表示获取所有历史数据（即最近数据）
        df = ak.index_zh_a_hist(symbol=symbol_code, period="daily", start_date="", end_date="")
        
        if df.empty:
            print(f"未找到 {index_name} ({symbol_code}) 的数据。")
            return None
        
        # 筛选出最新的一个交易日数据
        # akshare返回的日期列名为 '日期'，成交量列名为 '成交量'
        latest_data = df.iloc[-1]
        
        return {
            '指数代码': symbol_code,
            # 确保日期格式化正确
            '交易日期': latest_data['日期'].strftime('%Y%m%d'),
            '指数名称': index_name,
            '成交量': latest_data['成交量']
        }
    except Exception as e:
        print(f"获取 {index_name} ({symbol_code}) 数据时出错: {e}")
        return None

def main():
    """主函数，获取所有指数数据并保存到指定路径"""
    print("开始获取指数成交量数据...")
    
    # 获取当前日期和时间
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
    
    for code, name in INDEX_CODES.items():
        print(f"正在获取 {name} ({code})...")
        data = fetch_index_volume(code, name)
        if data:
            results.append(data)
        
        # 增加延迟以避免对数据源接口造成过大压力
        # 在 GitHub Actions 中建议保留
        time.sleep(1) 
    
    if results:
        # 转换为 DataFrame 
        df_results = pd.DataFrame(results)
        
        # akshare的成交量单位通常是股，转换为“手”需要除以100
        df_results['成交量(手)'] = df_results['成交量'] / 100
        df_results = df_results.drop(columns=['成交量']) # 删除原始成交量列
        
        # 调整最终列顺序
        df_results = df_results[['交易日期', '指数代码', '指数名称', '成交量(手)']]

        df_results.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n数据已成功保存到: {output_path}")
    else:
        print("\n所有指数数据获取失败，未生成文件。")

if __name__ == "__main__":
    main()
