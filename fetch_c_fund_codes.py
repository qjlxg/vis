import akshare as ak
import pandas as pd
import datetime
import os
import time

# 需要获取的指数代码及其在 akshare 中的代码
# akshare 获取指数数据时，通常需要完整的市场标识
INDEX_CODES = {
    "sh000001": "上证指数",     # 000001
    "sz399001": "深证成指",     # 399001
    "sz399006": "创业板指",     # 399006
    "sh000300": "沪深300"      # 000300
}

def fetch_index_volume(symbol_code, index_name):
    """
    使用 akshare 获取指定指数代码最近一个交易日的成交量数据
    symbol_code: akshare使用的指数代码 (例如: 'sh000001')
    index_name: 指数名称 (例如: '上证指数')
    """
    try:
        # 获取指数日线历史数据
        # adjust="qfq" 参数可选，对指数意义不大，但为保持一致性或方便未来扩展
        df = ak.index_zh_a_hist(symbol=symbol_code, period="daily", start_date="", end_date="", adjust="qfq")
        
        if df.empty:
            print(f"未找到 {index_name} ({symbol_code}) 的数据。")
            return None
        
        # 筛选出最新的一个交易日数据
        # akshare返回的日期列名为 '日期'，成交量列名为 '成交量'
        latest_data = df.iloc[-1]
        
        # akshare的成交量单位通常是手或股，具体取决于数据源，这里标注为 '成交量'
        return {
            '指数代码': symbol_code,
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
        time.sleep(1)
    
    if results:
        # 转换为 DataFrame 并排序
        df_results = pd.DataFrame(results)
        df_results = df_results[['交易日期', '指数代码', '指数名称', '成交量']]
        
        # 保存到 CSV 文件
        # 注意：akshare返回的成交量单位，如果需要转换为“手”，可能需要除以100
        # 如果 akshare 返回的是“股”，则：成交量(手) = 成交量 / 100
        # 为保持数据源的原始性，这里直接保存为 '成交量'，用户可自行在分析时转换。
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
