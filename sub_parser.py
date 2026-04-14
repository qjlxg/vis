import asyncio
import aiohttp
import base64
import re
import csv
import os
import socket
import json
import time
import ssl
import hashlib
from datetime import datetime
from urllib.parse import urlparse, quote, unquote
import geoip2.database

# --- 基础配置 ---
DATA_DIR = os.getenv('DATA_PATH', 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

INPUT_FILE = os.path.join(DATA_DIR, "sub_links.txt")
OUTPUT_TXT = os.path.join(DATA_DIR, "sub_parser.txt")
OUTPUT_B64 = os.path.join(DATA_DIR, "sub_parser_base64.txt")
OUTPUT_CSV = os.path.join(DATA_DIR, "sub_parser.csv")
OUTPUT_YAML = os.path.join(DATA_DIR, "sub_parser.yaml")
GEOIP_DB = os.path.join(DATA_DIR, "GeoLite2-Country.mmdb")

MAX_CONCURRENT_TASKS = 500 
MAX_RETRIES = 1

# --- 排除过滤名单 (包含网址及其变种关键词) ---
BLACKLIST_KEYWORDS = [
    "ly.ba000.cc", "wocao.su7.me", "jiasu01.vip", "louwangzhiyu", "mojie",  "lyly.649844.xyz", "multiserver", "shahramv1",
    "yywhale", "nxxbbf", "slianvpn", "cloudaddy", "quickbeevpn", 
    "tianmiao", "cokecloud", "boluoidc", "gpket", "fast8888", "ykxqn"
]

# --- 工具函数 ---
def decode_base64(data):
    if not data: return ""
    try:
        data = data.replace("-", "+").replace("_", "/")
        clean_data = re.sub(r'[^A-Za-z0-9+/=]', '', data.strip())
        missing_padding = len(clean_data) % 4
        if missing_padding: clean_data += '=' * (4 - missing_padding)
        return base64.b64decode(clean_data).decode('utf-8', errors='ignore')
    except: return ""

def encode_base64(data):
    try: return base64.b64encode(data.encode('utf-8')).decode('utf-8')
    except: return ""

def get_md5_short(text):
    return hashlib.md5(text.encode()).hexdigest()[:4]

def get_geo_info(host, reader):
    if not host or not reader: return "🌐", "未知地区"
    ip = host
    if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        try: ip = socket.gethostbyname(host)
        except: return "🌐", "未知地区"
    try:
        res = reader.country(ip)
        code = res.country.iso_code
        flag = "".join(chr(ord(c) + 127397) for c in code.upper()) if code else "🌐"
        country_name = res.country.names.get('zh-CN') or res.country.name or "未知国家"
        return flag, country_name
    except:
        return "🌐", "未知地区"

def get_node_details(line, protocol):
    try:
        if protocol == 'vmess':
            v = json.loads(decode_base64(line.split("://")[1]))
            return {"server": v.get('add'), "port": int(v.get('port', 443)), "uuid": v.get('id'), "tls": v.get('tls') == "tls"}
        u = urlparse(line)
        return {"server": u.hostname, "port": int(u.port or 443)}
    except: return None

def parse_nodes(content, reader):
    if "://" not in content[:50] and len(content) > 20:
        content = decode_base64(content)
    protocols = ['vmess', 'vless', 'trojan', 'anytls', 'hysteria', 'hysteria2', 'hy2', 'tuic', 'ss', 'ssr']
    pattern = r'(?:' + '|'.join(protocols) + r')://[^\s\"\'<>#]+(?:#[^\s\"\'<>]*)?'
    found_links = re.findall(pattern, content, re.IGNORECASE)
    nodes = []
    for link in found_links:
        if link.lower().startswith(('http://', 'https://')): continue
        protocol = link.split("://")[0].lower()
        try:
            # 提取服务器地址
            if protocol == 'vmess':
                host = json.loads(decode_base64(link.split("://")[1])).get('add')
            else:
                host_part = urlparse(link).hostname
                if not host_part:
                    host_part = re.search(r'@([^:/?#\s]+)', link).group(1).split(':')[0]
                host = host_part
            
            if not host: continue
            
            # --- 核心过滤逻辑：排除黑名单域名或变种 ---
            host_lower = host.lower()
            if any(keyword in host_lower for keyword in BLACKLIST_KEYWORDS):
                continue

            flag, country = get_geo_info(host, reader)
            nodes.append({"protocol": protocol, "flag": flag, "country": country, "line": link})
        except: continue
    return nodes

async def fetch_with_retry(session, url, reader, semaphore):
    async with semaphore:
        for _ in range(MAX_RETRIES + 1):
            try:
                async with session.get(url, timeout=15, ssl=False) as res:
                    if res.status != 200: return url, [], 0
                    text = await res.text()
                    nodes = parse_nodes(text, reader)
                    if nodes:
                        print(f"[+] 成功 ({len(nodes)} 节点): {url}")
                        return url, nodes, len(nodes)
            except: pass
        return url, [], 0

async def main():
    all_urls = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_urls = re.findall(r'https?://[^\s<>\"\'\u4e00-\u9fa5]+', f.read())

    unique_urls = list(dict.fromkeys(all_urls))
    # 同时也排除包含黑名单关键词的订阅链接本身
    unique_urls = [u for u in unique_urls if not any(k in u.lower() for k in BLACKLIST_KEYWORDS)]
    
    if not unique_urls: return
    if not os.path.exists(GEOIP_DB):
        print(f"缺失 {GEOIP_DB} 库文件"); return

    print(f"--- 正在处理 {len(unique_urls)} 个源 ---")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    with geoip2.database.Reader(GEOIP_DB) as reader:
        connector = aiohttp.TCPConnector(limit=50, ssl=False)
        async with aiohttp.ClientSession(headers={'User-Agent': 'v2rayN/6.23'}, connector=connector) as session:
            tasks = [fetch_with_retry(session, url, reader, semaphore) for url in unique_urls]
            results = await asyncio.gather(*tasks)
            raw_node_objs = []
            stats = []
            for url, nodes, count in results:
                raw_node_objs.extend(nodes); stats.append([url, count])

    final_links = []
    yaml_proxies = []
    seen_lines = set()
    
    for obj in raw_node_objs:
        line, protocol, flag, country = obj["line"], obj["protocol"], obj["flag"], obj["country"]
        base_link = line.split('#')[0] if protocol != 'vmess' else line
        if base_link in seen_lines: continue
        seen_lines.add(base_link)

        short_id = get_md5_short(base_link)
        new_name = f"{flag} {country} 打倒美帝国主义及其一切走狗_{short_id}"
        
        try:
            if protocol == 'vmess':
                v_json = json.loads(decode_base64(line.split("://")[1]))
                v_json['ps'] = new_name
                final_links.append(f"vmess://{encode_base64(json.dumps(v_json))}")
            elif protocol == 'ssr':
                ssr_body = decode_base64(line.split("://")[1])
                main_part = ssr_body.split('&remarks=')[0]
                new_rem = encode_base64(new_name).replace('=', '').replace('+', '-').replace('/', '_')
                final_links.append(f"ssr://{encode_base64(main_part + '&remarks=' + new_rem)}")
            else:
                final_links.append(f"{base_link}#{quote(new_name)}")

            d = get_node_details(line, protocol)
            if d:
                p_type = "trojan" if protocol == 'anytls' else protocol
                proxy_item = f"  - {{ name: \"{new_name}\", type: {p_type}, server: {d['server']}, port: {d['port']}"
                if protocol == 'vmess': proxy_item += f", uuid: {d['uuid']}, cipher: auto, tls: {str(d['tls']).lower()}"
                proxy_item += ", udp: true }"
                yaml_proxies.append(proxy_item)
        except: continue

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(OUTPUT_TXT, "a", encoding="utf-8") as f: f.write("\n".join(final_links))
    with open(OUTPUT_B64, "w", encoding="utf-8") as f: f.write(encode_base64("\n".join(final_links)))
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(["订阅链接", "节点数量"]); writer.writerows(stats)

    yaml_header = f"""# 美帝国主义是纸老虎
# Updated: {now_str}
# Total: {len(final_links)}

port: 7890
mode: Rule
dns:
  enable: true
  nameserver: [119.29.29.29, 223.5.5.5]

proxies:
"""
    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n".join(yaml_proxies))

    print(f"--- 任务完成！已生成 4 个文件，总计节点: {len(final_links)} ---")

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
