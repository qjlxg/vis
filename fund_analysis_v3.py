import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import warnings

warnings.filterwarnings('ignore')

# --- 1. 数据加载与准备 ---

def load_data(url):
    """
    从提供的URL加载CSV数据并进行预处理。
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        csv_content = response.text
        
        df = pd.read_csv(io.StringIO(csv_content), encoding='utf-8')
        df.columns = df.columns.str.strip()  # 清除列名中的空格
        df['代码'] = df['代码'].astype(str)
        
        # 将收益率数据转换为数值类型
        for col in ['rose(3y)', 'rose(2y)', 'rose(1y)', 'rose(6m)', 'rose(3m)']:
            # 强制转换，遇到非数值用NaN填充
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from URL: {e}")
        return None

# --- 2. 业绩指标分析 ---

def analyze_performance(df):
    """
    打印主要时间维度的业绩排名和表现。
    """
    print("--- 基金业绩表现概览 ---")
    for time_frame in ['3y', '2y', '1y', '6m', '3m']:
        rose_col = f'rose({time_frame})'
        rank_col = f'rank({time_frame})'
        
        if rose_col in df.columns and rank_col in df.columns:
            # 过滤掉NaN值，确保排名正确
            df_filtered = df.dropna(subset=[rose_col]).copy()
            if not df_filtered.empty:
                print(f"\n--- 过去 {time_frame} 业绩 ---")
                
                # 找到表现最好和最差的基金
                best_fund = df_filtered.loc[df_filtered[rose_col].idxmax()]
                worst_fund = df_filtered.loc[df_filtered[rose_col].idxmin()]
                
                print(f"收益率最高基金: {best_fund['名称']} (收益率: {best_fund[rose_col]:.2f}%)")
                print(f"收益率最低基金: {worst_fund['名称']} (收益率: {worst_fund[rose_col]:.2f}%)")
                
                # 打印排名前5的基金
                top_5 = df_filtered.sort_values(by=rose_col, ascending=False).head(5)
                print("\n排名前5的基金:")
                print(top_5[['名称', rose_col, rank_col]].to_string(index=False))


# --- 3. 散点图分析 ---

def plot_scatter(df, x_col, y_col, title, filename):
    """
    绘制基金收益率 vs 排名散点图。
    """
    df_filtered = df.dropna(subset=[x_col, y_col])
    if df_filtered.empty:
        print(f"No valid data to plot for {title}.")
        return

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df_filtered[x_col], y=df_filtered[y_col])
    
    # 标注排名前10的基金
    top_10 = df_filtered.sort_values(by=y_col, ascending=False).head(10)
    for i, row in top_10.iterrows():
        plt.text(row[x_col], row[y_col], f"{row['名称']}", 
                 ha='right', va='bottom', fontsize=9)
    
    plt.title(title, fontsize=16)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"{title}散点图已保存到 {filename}")

def main():
    file_url = 'https://github.com/qjlxg/rep/raw/refs/heads/main/recommended_cn_funds.csv'
    
    # 1. 加载数据
    df = load_data(file_url)
    if df is None:
        print("无法继续，数据加载失败。")
        return
        
    print("--- 数据预览 ---")
    print(df.head())
    
    # 2. 业绩分析
    analyze_performance(df)
    
    # 3. 散点图分析
    plot_scatter(df, 'rank(3y)', 'rose(3y)', '3年期基金收益率 vs 排名', 'performance_scatter_3y.png')
    plot_scatter(df, 'rank(1y)', 'rose(1y)', '1年期基金收益率 vs 排名', 'performance_scatter_1y.png')

if __name__ == "__main__":
    main()
