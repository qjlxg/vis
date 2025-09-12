import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import akshare as ak

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_real_data_from_list(fund_codes, benchmark_code, start_date, end_date):
    all_data = pd.DataFrame()
    
    # 获取基准指数数据
    try:
        index_data = ak.stock_zh_index_daily_em(symbol=benchmark_code)
        print("指数数据列名：", index_data.columns)  # 调试：打印列名
        if index_data.empty:
            print("❌ 指数数据为空")
            return None
        if 'date' not in index_data.columns:
            print(f"❌ 指数数据缺少 'date' 列，实际列名：{index_data.columns}")
            return None
        index_data['date'] = pd.to_datetime(index_data['date'])
        index_data = index_data.set_index('date')['close'].rename('沪深300')
        all_data = pd.DataFrame(index_data)
        print("✅ 已获取基准指数 沪深300 的数据")
    except Exception as e:
        print(f"❌ 获取指数 沪深300 数据失败：{e}")
        return None

    # 获取基金净值数据
    for code in fund_codes:
        try:
            fund_data = ak.fund_open_fund_info_em(fund_code=str(code).zfill(6), start_date=start_date, end_date=end_date)
            if fund_data.empty:
                print(f"❌ 基金 {code} 数据为空")
                continue
            fund_data['净值日期'] = pd.to_datetime(fund_data['净值日期'])
            fund_data = fund_data.set_index('净值日期')['单位净值'].rename(code)
            all_data = pd.concat([all_data, fund_data], axis=1)
            print(f"✅ 已获取基金 {code} 的数据")
        except Exception as e:
            print(f"❌ 获取基金 {code} 数据失败：{e}")

    # 清理和处理数据
    if all_data.empty:
        print("❌ 所有数据为空，无法进行分析")
        return None
    all_data = all_data.dropna().sort_index()
    all_data_normalized = all_data / all_data.iloc[0]
    
    return all_data_normalized

def plot_net_value(df_normalized):
    plt.figure(figsize=(12, 6))
    for col in df_normalized.columns:
        plt.plot(df_normalized.index, df_normalized[col], label=col)
    plt.title('基金与基准指数净值走势对比', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('标准化净值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('net_value_chart.png')
    print("📊 净值走势图已保存到 net_value_chart.png")

def plot_drawdown(df_normalized):
    plt.figure(figsize=(12, 6))
    for col in df_normalized.columns:
        cumulative_returns = df_normalized[col]
        drawdown = (cumulative_returns / cumulative_returns.cummax() - 1)
        plt.plot(drawdown.index, drawdown, label=col)
    plt.title('基金与基准指数回撤走势对比', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('回撤')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('drawdown_chart.png')
    print("📉 回撤走势图已保存到 drawdown_chart.png")

def main():
    csv_url = 'https://github.com/qjlxg/rep/raw/refs/heads/main/recommended_cn_funds.csv'
    
    print("--- 1. 从CSV文件获取基金代码列表 ---")
    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        df_list = pd.read_csv(io.StringIO(response.text), encoding='utf-8')
        df_list.columns = df_list.columns.str.strip()
        fund_codes = df_list['代码'].tolist()
        selected_fund_codes = fund_codes[:5]
        print(f"✅ 成功获取基金代码列表: {selected_fund_codes}")
    except Exception as e:
        print(f"❌ 获取或处理CSV文件失败：{e}")
        return
        
    print("\n--- 2. 开始从 akshare 获取真实数据 ---")
    
    end_date = pd.to_datetime('today').strftime('%Y%m%d')
    start_date = (pd.to_datetime('today') - pd.DateOffset(years=2)).strftime('%Y%m%d')
    
    df_normalized = get_real_data_from_list(
        fund_codes=selected_fund_codes,
        benchmark_code='sh000300',  # 修改为沪深300的正确代码
        start_date=start_date,
        end_date=end_date
    )
    
    if df_normalized is None or df_normalized.empty:
        print("最终数据为空，无法进行分析。")
        return

    print("\n--- 3. 绘制分析图表 ---")
    plot_net_value(df_normalized)
    plot_drawdown(df_normalized)

if __name__ == "__main__":
    main()
