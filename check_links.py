import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor

# 禁用不安全请求的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def check_url_logic(url):
    """底层探测逻辑"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        # 使用 allow_redirects=True 处理跳转
        response = requests.get(url, timeout=(5, 10), headers=headers, verify=False, allow_redirects=True)
        if response.status_code < 400:
            return True
    except:
        pass
    return False

def check_url(line):
    line = line.strip()
    if not line:
        return None

    # 1. 如果有协议头，直接测试
    if line.startswith(('http://', 'https://')):
        if check_url_logic(line):
            print(f"[SUCCESS] {line}")
            return line
    else:
        # 2. 如果没有协议头，先测 http 再测 https
        domain = line.lstrip('/')
        
        # 测试 HTTP (通常会自动跳转到 HTTPS)
        http_url = f"http://{domain}"
        if check_url_logic(http_url):
            print(f"[SUCCESS] {http_url}")
            return http_url
            
        # 测试 HTTPS
        https_url = f"https://{domain}"
        if check_url_logic(https_url):
            print(f"[SUCCESS] {https_url}")
            return https_url
    
    print(f"[FAILED] {line}")
    return None

def main():
    file_path = 'urls.txt'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取所有行并过滤掉空行
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    if not urls:
        print("No URLs found in the file.")
        return

    # 去重
    urls = list(dict.fromkeys(urls))
    print(f"Total entries to check: {len(urls)}")

    # 并发测试 (max_workers 可根据网络情况调整)
    with ThreadPoolExecutor(max_workers=85) as executor:
        results = list(executor.map(check_url, urls))

    # 过滤掉 None 结果
    valid_urls = [url for url in results if url is not None]

    # 保存回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for url in valid_urls:
            f.write(url + '\n')
    
    print(f"Done! Saved {len(valid_urls)} valid links.")

if __name__ == "__main__":
    main()
