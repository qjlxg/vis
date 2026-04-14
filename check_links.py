import requests
import urllib3
import os
from concurrent.futures import ThreadPoolExecutor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = os.getenv('DATA_PATH', '.')
FILE_NAME = 'urls.txt'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

def check_url_logic(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
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

    if line.startswith(('http://', 'https://')):
        if check_url_logic(line):
            return line
    else:
        domain = line.lstrip('/')
        http_url = f"http://{domain}"
        if check_url_logic(http_url):
            return http_url
            
        https_url = f"https://{domain}"
        if check_url_logic(https_url):
            return https_url
    return None

def main():
    if not os.path.exists(FILE_PATH):
        return

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except:
        return

    if not urls:
        return

    urls = list(dict.fromkeys(urls))
    
    with ThreadPoolExecutor(max_workers=85) as executor:
        results = list(executor.map(check_url, urls))

    valid_urls = [url for url in results if url is not None]

    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        for url in valid_urls:
            f.write(url + '\n')
            
    print(f"--- Done | Valid: {len(valid_urls)}/{len(urls)} ---")

if __name__ == "__main__":
    main()
