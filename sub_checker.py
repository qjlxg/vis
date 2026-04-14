import asyncio
import httpx
import re
import os
from datetime import datetime


DATA_DIR = os.getenv('DATA_PATH', 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


INPUT_FILE = os.path.join(DATA_DIR, 'subscribes.txt')
OUTPUT_FILE = os.path.join(DATA_DIR, 'valid_subs.txt')
LINKS_FILE = os.path.join(DATA_DIR, 'sub_links.txt')


CONCURRENT_LIMIT = 500 
TIMEOUT = 15.0

# --- 排除过滤名单 ---
BLACKLIST_KEYWORDS = [
    "ly.ba000.cc", "wocao.su7.me", "jiasu01.vip", "louwangzhiyu", "mojie", "lyly.649844.xyz", "multiserver", "shahramv1",
    "yywhale", "nxxbbf", "slianvpn", "cloudaddy", "quickbeevpn", 
    "tianmiao", "cokecloud", "boluoidc", "gpket", "fast8888", "ykxqn",
    'baidu.com', 'google.com', 'github.com', 'zhihu.com', 'xueqiu.com', 'ripwall', 'thugiping', 'xpanel.shoptnetz.com', 'yywhale.com', 'ro3shop.ir',
    'yandex.com', 'yamcode.com', 'wikipedia.org', 'microsoft.com', '15.204.191.53', 'argo.onl','51.83.8.191', 'ripwall.men',
    'apple.com', 'cloudflare.com', 'douban.com', 'weibo.com', 'qq.com',
    'csdn.net', 'juejin.cn', 'v2ex.com', 'bilibili.com', 'youtube.com',
    'twitter.com', 'facebook.com', 'instagram.com', 'telegram.org',
    'speedtest.net', 'fast.com', 'ip138.com', 'ip.skk.moe', 'gitee.com',
    'xueshu', 'research', 'edu', 'gov', 'amazon', 'bing', 'outlook', 'mail'
]

def format_bytes(size):
    if size is None or size <= 0: return "0B"
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if size < 1024.0:
            return f"{int(size)}{unit}" if size % 1 == 0 else f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}PB"

async def check_sub(client, url, semaphore):
    url = url.strip()
    if not url or not url.startswith('http'):
        return None
    
    if any(keyword.lower() in url.lower() for keyword in BLACKLIST_KEYWORDS):
        return None

    async with semaphore:
        try:
            headers = {
                'User-Agent': 'ClashMeta/1.18.0 (Windows NT 10.0; Win64; x64)',
                'Accept': '*/*'
            }
            resp = await client.get(url, headers=headers, follow_redirects=True)
            
            info_header = resp.headers.get('subscription-userinfo')
            if not info_header:
                return None

            data = {k: int(v) for k, v in re.findall(r'(\w+)=([^; ]+)', info_header)}
            
            total = data.get('total', 0)
            used = data.get('upload', 0) + data.get('download', 0)
            remain = total - used
            expire = data.get('expire', 0)
            
            now = datetime.now()
            now_ts = int(now.timestamp())
            
         
            if total > 0 and remain > 0 and (expire == 0 or expire > now_ts):
                is_premium = False
               
                if expire == 0 or (expire - now_ts) >= 172800:
                    is_premium = True

                if expire > 0:
                    expire_dt = datetime.fromtimestamp(expire)
                    expire_date_str = expire_dt.strftime('%Y-%m-%d %H:%M:%S+08:00')
                    delta = expire_dt - now
                    hours, remainder = divmod(int(delta.total_seconds()), 3600)
                    remain_time_detail = f"{delta.days} days, {hours % 24:02}:{remainder // 60:02}"
                else:
                    expire_date_str = "永不过期"
                    remain_time_detail = "Infinite"

                used_str = format_bytes(used).replace('GB', 'G').replace('MB', 'M')
                total_str = format_bytes(total).replace('GB', 'G').replace('MB', 'M')
                remain_str = format_bytes(remain).replace('GB', 'G').replace('MB', 'M')
                check_time = now.strftime('%Y-%m-%d %H:%M:%S+08:00')

                res_info = (
                    f"sub_info  {used_str}  {total_str}  {expire_date_str}  (剩余 {remain_str} {remain_time_detail})\n"
                    f"sub_url  {url}\n"
                    f"time  {check_time}\n"
                )
                
              
                return (res_info, is_premium, url)
            
            return None
        except:
            return None

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found"); return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_urls = list(set([line.strip() for line in f if line.strip().startswith('http')]))

    if not raw_urls:
        print("No URLs to check."); return

    print(f"--- 正在检测 {len(raw_urls)} 个订阅源 ---")
    
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=CONCURRENT_LIMIT)
    
    async with httpx.AsyncClient(verify=False, http2=True, timeout=TIMEOUT, limits=limits) as client:
        tasks = [check_sub(client, url, semaphore) for url in raw_urls]
        results = await asyncio.gather(*tasks)

    valid_data = [r for r in results if r]
    
 
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join([item[0] for item in valid_data]))
    
   
    premium_links = [item[2] for item in valid_data if item[1]]
    with open(LINKS_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(premium_links))
    
    print(f"--- 任务完成 ---")
    print(f"✅ 有效/总数: {len(valid_data)}/{len(raw_urls)}")
    print(f"🚀 优质链接已存入: {LINKS_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
