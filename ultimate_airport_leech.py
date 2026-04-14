# coding=utf-8
import json, re, base64, time, random, string, os, socket, threading, datetime, sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from random import choice
from threading import RLock, Thread
from time import sleep, time as stime
from urllib.parse import (parse_qsl, unquote_plus, urlencode, urljoin,
                        urlsplit, urlunsplit, quote, parse_qs)

import json5, urllib3, requests
from bs4 import BeautifulSoup

# --- 核心引擎：过墙级伪装 ---
try:
    from curl_cffi import requests as crequests 
except ImportError:
    crequests = requests

# --- 核心识别：OCR 验证码 ---
try:
    import ddddocr
    ocr = ddddocr.DdddOcr(show_ad=False)
    ocr_lock = threading.Lock()
except ImportError:
    ocr = None

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== 配置与参数 ====================
INPUT_FILE = "urls.txt"
CACHE_FILE = "airport_master.cache"      
ERROR_FILE = "airport_error.cache"      
SUB_FILE = "subscribes.txt"
NODES_FILE = "nodes.txt"
MAX_WORKERS = 150
SH_TZ = datetime.timezone(datetime.timedelta(hours=8))

# ==================== 增强版黑名单系统 ====================
DOMAIN_BLACKLIST = {
    'baidu.com', 'google.com', 'github.com', 'zhihu.com', 'xueqiu.com', 'ripwall', 'thugiping', 'xpanel.shoptnetz.com', 'yywhale.com', 'ro3shop.ir', 'wocao.su7.me',
    'yandex.com', 'yamcode.com', 'wikipedia.org', 'microsoft.com', '15.204.191.53', 'argo.onl','51.83.8.191', 'ripwall.men',
    'apple.com', 'cloudflare.com', 'douban.com', 'weibo.com', 'qq.com',
    'csdn.net', 'juejin.cn', 'v2ex.com', 'bilibili.com', 'youtube.com',
    'twitter.com', 'facebook.com', 'instagram.com', 'telegram.org',
    'speedtest.net', 'fast.com', 'ip138.com', 'ip.skk.moe', 'gitee.com',
    'xueshu', 'research', 'edu', 'gov', 'amazon', 'bing', 'outlook', 'mail'
}

SUFFIX_BLACKLIST = ('.gov', '.edu', '.mil', '.org', '.gov.cn', '.edu.cn')
io_lock = threading.Lock()

# ==================== 基础工具函数 ====================
def fast_log(msg):
    now = datetime.datetime.now(SH_TZ).strftime('%H:%M:%S')
    print(f"[{now}] {msg}", flush=True)

def format_size(size):
    try:
        s = float(size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if s < 1024: return f"{s:.2f}{unit}"
            s /= 1024
        return f"{s:.2f}PB"
    except: return "0B"

def format_time(ts):
    if not ts or ts == 0 or ts == "0" or ts == "": return "永久"
    try:
        ts = float(ts)
        if ts > 2147483647: ts = ts / 1000
        return datetime.datetime.fromtimestamp(ts, SH_TZ).strftime('%Y-%m-%d')
    except: return "未知"

def cached(func):
    cache = {}
    def wrapper(*args):
        if args not in cache: cache[args] = func(*args)
        return cache[args]
    return wrapper

# ==================== 响应与 Session 包装 ====================
class Response:
    def __init__(self, r, url=""):
        self.__content = getattr(r, 'content', b'')
        self.__headers = getattr(r, 'headers', {})
        self.__status_code = getattr(r, 'status_code', 500)
        self.__url = getattr(r, 'url', url)

    @property
    def content(self): return self.__content
    
    @property
    def status_code(self): return self.__status_code 
    
    @property
    def ok(self): return 200 <= self.__status_code < 300
    
    @property
    @cached
    def text(self):
        try: return self.__content.decode('utf-8', errors='ignore').replace('\t', '    ')
        except: return ""

    @cached
    def json(self):
        try:
            jt = self.text.strip()
            if not (jt.startswith('{') or jt.startswith('[')): return {}
            return json.loads(jt)
        except: return {}

    @cached
    def bs(self): return BeautifulSoup(self.text, 'html.parser')

class Session:
    def __init__(self, base=None):
        self.session = crequests.Session(impersonate="chrome120", verify=False)
        self.headers = self.session.headers
        self.__base = base.rstrip('/') if base else None

    @property
    def base(self): return self.__base

    def request(self, method, url='', data=None, **kwargs):
        full_url = url if url.startswith('http') else urljoin(self.__base + '/', url.lstrip('/'))
        try:
            r = self.session.request(method, full_url, data=data, timeout=20, **kwargs)
            return Response(r, full_url)
        except Exception as e:
            class Fake: pass
            f = Fake(); f.content = f"Error: {type(e).__name__}".encode(); f.status_code = 599; f.headers = {}
            return Response(f, full_url)

    def get(self, url='', **kwargs): return self.request('GET', url, **kwargs)
    def post(self, url='', data=None, **kwargs): return self.request('POST', url, data, **kwargs)

# ==================== 面板 Session 逻辑 ====================
class V2BoardSession(Session):
    def register(self, email, password):
        paths = ['api/v1/passport/auth/register', 'api/v1/guest/passport/auth/register']
        payload = {'email': email, 'password': password, 'repassword': password, 'invite_code': ''}
        last_msg = "API NotFound"
        for path in paths:
            res_obj = self.post(path, payload)
            if res_obj.status_code >= 500 or res_obj.status_code == 404: continue
            
            res = res_obj.json()
            if 'captcha' in str(res.get('message','')).lower() and ocr:
                for cp in ['api/v1/passport/comm/captcha', 'api/v1/guest/passport/comm/captcha']:
                    c_res = self.get(cp).json()
                    if c_res.get('data'):
                        try:
                            img = base64.b64decode(c_res['data'].split(',')[-1])
                            with ocr_lock: payload['captcha_code'] = ocr.classification(img)
                            res_obj = self.post(path, payload)
                            res = res_obj.json()
                            break
                        except: pass
            
            if res.get('data') and isinstance(res['data'], dict):
                token = res['data'].get('token') or res['data'].get('auth_data')
                if token: self.headers['authorization'] = token; return True, "操作成功"
            
            last_msg = res.get('message') or str(res_obj.status_code)
            if any(x in str(last_msg) for x in ["已经", "存在"]): return True, f"账户已存在:{last_msg}"
        return False, last_msg

    def buy(self):
        try:
            r = self.get('api/v1/user/plan/fetch').json()
            plans = r.get('data', [])
            for p in plans:
                if any(p.get(k) == 0 for k in ['month_price', 'onetime_price', 'year_price']):
                    period = 'month_price' if p.get('month_price') == 0 else 'onetime_price'
                    order = self.post('api/v1/user/order/save', {'period': period, 'plan_id': p['id']}).json()
                    if order.get('data'):
                        self.post('api/v1/user/order/checkout', {'trade_no': order['data']})
                        return f"FreePlan({p['id']})"
        except: pass
        return "NoFreePlan"

    def get_sub_url(self):
        tk = self.headers.get('authorization')
        try:
            res = self.get('api/v1/user/getSubscribe').json()
            if res.get('data') and isinstance(res['data'], dict): return res['data'].get('subscribe_url')
        except: pass
        return f"{self.base}/api/v1/client/subscribe?token={tk}" if tk else None

class SSPanelSession(Session):
    def register(self, email, password):
        payload = {'email': email, 'passwd': password, 'repasswd': password, 'agreeterm': 1}
        res_obj = self.post('auth/register', payload)
        res = res_obj.json()
        msg = res.get('msg', 'Reg Fail')
        if res.get('ret') or "成功" in str(msg): return True, "操作成功"
        return False, msg

    def get_sub_url(self):
        try:
            r = self.get('user').bs()
            tag = r.find(attrs={'data-clipboard-text': re.compile(r'https?://')})
            return tag['data-clipboard-text'] if tag else None
        except: return None

# ==================== 核心处理器 ====================
def check_sub(url):
    """
    加强版订阅检测：
    1. 解析 Header 获取流量和过期时间
    2. 如果没有 Header，则下载内容检测是否包含节点关键字
    3. 判定流量 > 0 且包含节点才算真正成功
    """
    try:
        r = crequests.get(url, headers={'User-Agent': 'Clash.meta'}, timeout=15, verify=False)
        if not r.ok: return "HTTP_Error", False

        info_h = r.headers.get('subscription-userinfo', '')
        traffic_info = ""
        total_traffic = 0
        
        # 1. 尝试从 Header 获取信息
        if info_h:
            try:
                p = {i.split('=')[0].strip(): i.split('=')[1].strip() for i in info_h.split(';') if '=' in i}
                total_traffic = int(p.get('total', 0))
                used = int(p.get('upload', 0)) + int(p.get('download', 0))
                expire = p.get('expire', 0)
                traffic_info = f"{format_size(used)}/{format_size(total_traffic)} ({format_time(expire)})"
            except: pass

        # 2. 深度校检内容：检查是否包含节点关键字
        content = r.text
        # 尝试 Base64 解码检测内容
        try:
            decoded_content = base64.b64decode(content).decode('utf-8', errors='ignore')
        except:
            decoded_content = content
            
        # 节点关键字匹配 (支持常见格式)
        node_keywords = ['vmess://', 'ssr://', 'ss://', 'trojan://', 'vless://', 'proxies:', 'Proxy:', 'SERVER=']
        has_nodes = any(k in decoded_content for k in node_keywords)

        # 3. 最终成功判定
        if total_traffic > 0 and has_nodes:
            return traffic_info, True
        elif total_traffic > 0 and not has_nodes:
            return f"TrafficOnly({traffic_info})", False # 有流量但没节点，通常是空订阅
        elif has_nodes:
            return "Active(NoTrafficHeader)", True # 没有流量头但有节点，视为成功
        else:
            return "EmptySubscription", False # 既没流量又没节点

    except Exception as e:
        return f"CheckFailed({type(e).__name__})", False

def process_worker(url):
    clean_dom = urlsplit(url).netloc.lower() or url.split('/')[0].lower()
    result = {
        "domain": clean_dom, "reg_info": "未开始", "buy": "N/A", "email": "N/A",
        "pass": "N/A", "sub_info": "N/A", "sub_url": "N/A", "type": "unknown",
        "time": datetime.datetime.now(SH_TZ).isoformat()
    }

    if any(black in clean_dom for black in DOMAIN_BLACKLIST) or clean_dom.endswith(SUFFIX_BLACKLIST):
        result["reg_info"] = "[跳过] 黑名单关键词/后缀"
        return result

    base_url = url if url.startswith('http') else 'https://' + url
    session = None
    try:
        test_s = Session(base_url)
        conf_res = test_s.get('api/v1/guest/comm/config')
        if conf_res.ok or "v2board" in test_s.get('env.js').text.lower():
            session = V2BoardSession(base_url); result["type"] = "v2board"
        else:
            login_text = test_s.get('auth/login').text
            if any(x in login_text for x in ["SSPanel", "staff", "checkin"]):
                session = SSPanelSession(base_url); result["type"] = "sspanel"
    except Exception as e:
        result["reg_info"] = f"[失败] 面板识别异常: {type(e).__name__}"
        return result

    if not session:
        result["reg_info"] = "[失败] 无法识别面板或非机场网站"
        return result

    email = f"{''.join(random.choices(string.ascii_lowercase + string.digits, k=10))}@gmail.com"
    password = "".join(random.choices(string.ascii_letters + string.digits, k=12))
    result["email"], result["pass"] = email, password

    try:
        ok, reg_msg = session.register(email, password)
        result["reg_info"] = reg_msg
        if not ok: return result

        if result["type"] == "v2board": result["buy"] = session.buy()
        else: result["buy"] = "Default"

        sub_url = session.get_sub_url()
        if sub_url:
            result["sub_url"] = sub_url
            info, is_ok = check_sub(sub_url)
            result["sub_info"] = info
            if is_ok:
                fast_log(f" [+] {clean_dom} | {info} | {result['buy']}")
                with io_lock:
                    with open(SUB_FILE, 'a') as f: f.write(sub_url + "\n")
                    with open(NODES_FILE, 'a') as f: f.write(sub_url + "\n")
            else:
                # 如果检测不通过，强行把 reg_info 改为失败原因，方便主程序分流
                result["reg_info"] = f"[判定失败] {info}" 
        else:
            result["sub_info"] = "无法获取订阅地址"
    except Exception as e:
        result["reg_info"] = f"[报错] 运行异常: {str(e)}"

    return result

def main():
    if not os.path.exists(INPUT_FILE):
        fast_log(f"错误: 找不到输入文件 {INPUT_FILE}")
        return
        
    urls = list(set([u.strip() for u in open(INPUT_FILE).readlines() if "." in u]))
    fast_log(f"=== 启动加强校验版引擎(过滤空订阅) === 任务数: {len(urls)}")
    
    success_logs = []
    error_logs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(process_worker, u): u for u in urls}
        for f in as_completed(futures):
            try:
                res = f.result()
                if not res: continue
                
                log_block = (
                    f"[{res['domain']}]\n"
                    f"reg_info  {res['reg_info']}\n"
                    f"buy       {res['buy']}\n"
                    f"email     {res['email']}\n"
                    f"pass      {res['pass']}\n"
                    f"sub_info  {res['sub_info']}\n"
                    f"sub_url   {res['sub_url']}\n"
                    f"time      {res['time']}\n"
                    f"type      {res['type']}\n\n"
                )
                
                # 分流判断：只有真正通过 check_sub 验证的才进 master
                if "操作成功" in res['reg_info'] and res['sub_url'] != "N/A" and "判定失败" not in res['reg_info']:
                    success_logs.append(log_block)
                else:
                    error_logs.append(log_block)
            except: pass 

    with open(CACHE_FILE, 'w', encoding='utf-8') as f: f.writelines(success_logs)
    with open(ERROR_FILE, 'w', encoding='utf-8') as f: f.writelines(error_logs)
    
    fast_log(f"任务结束 | 真实有效: {len(success_logs)} | 无效/跳过: {len(error_logs)}")

if __name__ == "__main__":
    main()
