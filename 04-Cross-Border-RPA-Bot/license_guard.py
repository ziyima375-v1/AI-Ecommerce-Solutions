# -*- coding: utf-8 -*-
"""
极简本地授权（不采指纹）：
- 首次激活：校验 GitHub JSON 中的 key 条目（days、used）
- 通过后在本机落一个令牌（随机 uuid + 到期日），以后优先本地验票
- 你把 key 的 used 改为 true -> 拒绝新的机器再次激活；已领票机器照常用到期
- 可选硬撤销：在 JSON 里加 "revoked_keys": ["KEY1","KEY2"]，上线即失效（默认不启用）
"""
import os, json, uuid, time, datetime, requests

PRODUCT_CODE = os.getenv("PRODUCT_CODE", "DEMO-PRODUCT").strip()  # 对外发布请改为你的产品代号     # 必须与云端 JSON 中的 product 一致
GITHUB_JSON_URL = os.getenv("LICENSE_JSON_URL", "").strip()  # 对外发布请在环境变量里配置  # ← 改成你的
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def _desktop_data_dir():
    base = os.path.join(os.path.expanduser("~"), "Desktop", os.getenv("APP_DATA_DIRNAME", "Ecom_Excel_Tool"))
    os.makedirs(base, exist_ok=True)
    return base

DATA_DIR = _desktop_data_dir()
TOKEN_FILE = os.path.join(DATA_DIR, "license_token.json")
ETAG_FILE  = os.path.join(DATA_DIR, "license_etag.txt")

DEFAULT_TIMEOUT = 8

class LicenseError(Exception): pass

def _now_utc():
    return datetime.datetime.utcnow()

def _date_add_days(days:int):
    return (_now_utc() + datetime.timedelta(days=days))

def _load_local_token():
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            t = json.load(f)
        # 基本字段检查
        if t.get("product") != PRODUCT_CODE: return None
        # 过期检查
        exp = datetime.datetime.fromisoformat(t.get("expires_at"))
        if _now_utc() > exp:
            return None
        return t
    except Exception:
        return None

def _save_local_token(key:str, days:int):
    tok = {
        "product": PRODUCT_CODE,
        "license_key_tail": (key[-4:] if key else ""),
        "issued_at": _now_utc().isoformat(),
        "expires_at": _date_add_days(days).isoformat(),
        "local_id": str(uuid.uuid4()),  # 仅本机随机号，不采集硬件指纹
        "version": 1
    }
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)
    return tok

def _load_cloud_json():
    if not GITHUB_JSON_URL:
        raise LicenseError("LICENSE_JSON_URL 未配置（对外发布请设置环境变量）")
    headers = {}
    if os.path.exists(ETAG_FILE):
        try:
            etag = open(ETAG_FILE, "r", encoding="utf-8").read().strip()
            if etag:
                headers["If-None-Match"] = etag
        except Exception:
            pass
    r = requests.get(GITHUB_JSON_URL, timeout=DEFAULT_TIMEOUT, headers=headers)
    if r.status_code == 304:
        # 未变更，从缓存落地读取（若你愿意可本地存一份快照）
        # 为简洁，这里直接再拉一次不带 If-None-Match
        r = requests.get(GITHUB_JSON_URL, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    etag = r.headers.get("ETag", "")
    if etag:
        try:
            with open(ETAG_FILE, "w", encoding="utf-8") as f:
                f.write(etag)
        except Exception:
            pass
    return r.json()

def _find_key_entry(cloud, key):
    if not isinstance(cloud, dict): raise LicenseError("云端格式错误")
    if cloud.get("product") != PRODUCT_CODE:
        raise LicenseError("产品不匹配")
    keys = cloud.get("keys", [])
    for item in keys:
        if (item or {}).get("key") == key:
            return item
    return None

def _cloud_revoked(cloud, key):
    rev = cloud.get("revoked_keys")
    if not rev: return False
    return key in set(rev)

def check_local_first():
    """优先本地验票，离线也能用"""
    tok = _load_local_token()
    if tok:  # 有票且未过期
        return True, tok
    return False, None

def activate_with_key(user_input_key:str):
    """首次激活：联网校验 JSON，发放本地票"""
    if not user_input_key or len(user_input_key) < 8:
        raise LicenseError("密钥格式无效")
    try:
        cloud = _load_cloud_json()
    except Exception as e:
        raise LicenseError(f"无法连接授权服务器：{e}")
    if _cloud_revoked(cloud, user_input_key):
        raise LicenseError("该密钥已被撤销")
    entry = _find_key_entry(cloud, user_input_key)
    if not entry:
        raise LicenseError("密钥不存在")
    if entry.get("used", False):
        # 你人为置为 true 后，任何新机器都不能再激活
        raise LicenseError("该密钥已被使用")
    days = int(entry.get("days", 30))
    tok = _save_local_token(user_input_key, days)
    # 注意：不回写 used_by，因为你明确要求不做绑定和不上报；你自己去 GitHub 把 used 改为 true
    return tok

def require_license(interactive_input_func=None, notify_func=None):
    """
    入口守卫。放在主程序启动前调用。
    interactive_input_func: 回调函数，用来获取用户输入的 key（例如弹窗输入）
    notify_func: 回调函数，通知消息（成功/失败）
    """
    ok, tok = check_local_first()
    if ok:
        if notify_func: notify_func("授权通过（本地令牌）")
        return True
    # 本地无票 -> 请求用户输入
    if interactive_input_func is None:
        raise LicenseError("缺少输入回调")
    user_key = (interactive_input_func() or "").strip().upper()
    if not user_key:
        if notify_func: notify_func("未输入密钥")
        return False
    try:
        activate_with_key(user_key)
        if notify_func: notify_func("授权成功")
        return True
    except LicenseError as e:
        if notify_func: notify_func(f"授权失败：{e}")
        return False
