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

# 设定获取历史数据的天数（交易日）
DAYS_TO_FETCH = 10 

def fetch_index_historical_volume(symbol_code, index_name):
    """
    使用 akshare 获取指定指数代码最近 N 个交易日的成交量数据。
    """
    # 计算开始日期：从今天往前推 N 个交易日，使用一个足够长的跨度（例如 20 天）来确保获取到至少 10 个交易日的数据。
    end_date_str = datetime.datetime.now().strftime('%Y%m%d')
    start_date = datetime.datetime.now() - datetime.timedelta(days=DAYS_TO_FETCH * 2) 
    start_date_str = start_date.strftime('%Y%m%d')
    
    try:
        # 使用 index_zh_a_hist 接口获取历史数据
        # 移除 'adjust' 参数，并使用明确的日期范围
        df = ak.index_zh_a_hist(
            symbol=symbol_code, 
            period="daily", 
            start_date=start_date_str, 
            end_date=end_date_str
        )
        
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"  -> 未找到 {index_name} ({symbol_code}) 的数据。")
            return None
        
        # 确保数据按日期升序排列，并取最新的 N 条记录
        df = df.sort_values(by='日期', ascending=True).tail(DAYS_TO_FETCH)
        
        # 提取所需列，并重命名
        df_result = df.rename(columns={'日期': '交易日期', '成交量': '成交量(股)'})
        
        # 添加指数名称和代码
        df_result['指数代码'] = symbol_code
        df_result['指数名称'] = index_name
        
        # 计算成交量(手)
        # akshare 的成交量通常是股，需要除以 100 转换为手
        df_result['成交量(手)'] = df_result['成交量(股)'] / 100
        
        # 格式化日期，并选择最终列
        df_result['交易日期'] = df_result['交易日期'].dt.strftime('%Y%m%d')
        df_result = df_result[['交易日期', '指数代码', '指数名称', '成交量(手)']]
        
        return df_result
        
    except Exception as e:
        print(f"获取 {index_name} ({symbol_code}) 历史数据时出错: {e}")
        return None

def main():
    """主函数，获取所有指数的历史数据并合并保存"""
    print(f"开始获取最近 {DAYS_TO_FETCH} 个交易日的指数成交量数据...")
    
    now = datetime.datetime.now()
    timestamp_str = now.strftime('%Y%m%d_%H%M%S')
    
    # 构造保存目录 (年月)
    year_month = now.strftime('%Y%m')
    output_dir = os.path.join(os.getcwd(), year_month)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造文件名
    output_filename = f"index_volume_history_{timestamp_str}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    all_results = []
    
    for code, name in INDEX_CODES.items():
        print(f"正在获取 {name} ({code}) 历史数据...")
        
        # 调用获取历史数据的函数
        df_data = fetch_index_historical_volume(code, name)
        
        if df_data is not None and not df_data.empty:
            all_results.append(df_data)
        
        # 增加延迟
        time.sleep(2) 
    
    if all_results:
        # 合并所有指数的历史数据到一个 DataFrame
        final_df = pd.concat(all_results, ignore_index=True)
        
        # 按交易日期排序，便于查看
        final_df = final_df.sort_values(by=['交易日期', '指数代码'])
        
        # 保存到 CSV 文件
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n数据已成功保存到: {output_path}")
        print(f"总共获取了 {len(final_df)} 条历史数据记录。")
    else:
        print("\n所有指数历史数据获取失败，未生成文件。")

if __name__ == "__main__":
    main()
