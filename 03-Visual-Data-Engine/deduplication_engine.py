# -*- coding: utf-8 -*-
"""
图片去重工具（极速增强版）
- 多哈希（a/d/p/w）+ 过滤候选 + （可选）SSIM 精核
- 主库索引：SQLite（WAL），增量维护
- 新图处理：并发判重，重复默认“隔离”到新图文件夹/Quarantine（可选直接删除）
"""

import os
import io
import math
import time
import hashlib
import sqlite3
import threading
import queue
import shutil
import multiprocessing
from dataclasses import dataclass
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from PIL import Image
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ============== 配置 =================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DB_FILENAME = "imgdedup.sqlite"

# 最大并发线程（过高会让磁盘/CPU打满，默认按CPU核数）
MAX_WORKERS = max(4, min(32, (multiprocessing.cpu_count() or 8)))


# ============== 工具函数 =================

def walk_images(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, fn))
    return out


def sha1_of_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pil_open_rgb(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def img_hashes(path: str) -> Optional[Tuple[int, int, int, int, int, int, int]]:
    """
    返回 (size, width, height, ahash, dhash, phash, whash)；哈希为 64bit int
    """
    try:
        st = os.stat(path)
        size = int(st.st_size)
        img = pil_open_rgb(path)
        if img is None:
            return None
        w, h = img.size

        ah = int(str(imagehash.average_hash(img, hash_size=8)), 16)
        dh = int(str(imagehash.dhash(img, hash_size=8)), 16)
        ph = int(str(imagehash.phash(img, hash_size=8)), 16)
        wh = int(str(imagehash.whash(img, hash_size=8)), 16)

        try:
            img.close()
        except Exception:
            pass

        return size, w, h, ah, dh, ph, wh
    except Exception:
        return None


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def int_to_hex64(x: int) -> str:
    return f"{x:016x}"


def hex64_to_int(s: str) -> int:
    return int(s, 16)


def compute_ssim(path1: str, path2: str, side: int = 320) -> float:
    im1 = pil_open_rgb(path1)
    im2 = pil_open_rgb(path2)
    if im1 is None or im2 is None:
        return 0.0
    try:
        im1 = im1.resize((side, side), Image.LANCZOS).convert("L")
        im2 = im2.resize((side, side), Image.LANCZOS).convert("L")
        a1 = np.array(im1, dtype=np.uint8)
        a2 = np.array(im2, dtype=np.uint8)
        return float(ssim(a1, a2, data_range=255, gaussian_weights=True, sigma=1.5))
    except Exception:
        return 0.0
    finally:
        try:
            im1.close()
            im2.close()
        except Exception:
            pass


# ============== 数据库存取层 =================

class DB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            # 建表（目标结构：四个哈希字段 TEXT）
            self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS images(
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                size INTEGER,
                width INTEGER,
                height INTEGER,
                mtime REAL,
                sha1 TEXT,
                ahash TEXT,
                dhash TEXT,
                phash TEXT,
                whash TEXT,
                phash_prefix INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_size ON images(size);
            CREATE INDEX IF NOT EXISTS idx_dims ON images(width, height);
            CREATE INDEX IF NOT EXISTS idx_sha1 ON images(sha1);
            CREATE INDEX IF NOT EXISTS idx_phash_prefix ON images(phash_prefix);
            """)
            self.conn.commit()

            # 检查是否老表（哈希列是 INTEGER），需要自动迁移
            cur = self.conn.execute("PRAGMA table_info(images)")
            cols = {r[1]: r[2].upper() for r in cur.fetchall()}
            # 如果哈希字段是 INTEGER，迁移到 TEXT
            need_migrate = False
            for c in ("ahash", "dhash", "phash", "whash"):
                if c in cols and cols[c] == "INTEGER":
                    need_migrate = True
                    break
            if need_migrate:
                # 迁移：重命名旧表 -> 新建 -> 拷贝 -> 删除旧表
                self.conn.executescript("""
                ALTER TABLE images RENAME TO images_old;
                CREATE TABLE IF NOT EXISTS images(
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    mtime REAL,
                    sha1 TEXT,
                    ahash TEXT,
                    dhash TEXT,
                    phash TEXT,
                    whash TEXT,
                    phash_prefix INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_size ON images(size);
                CREATE INDEX IF NOT EXISTS idx_dims ON images(width, height);
                CREATE INDEX IF NOT EXISTS idx_sha1 ON images(sha1);
                CREATE INDEX IF NOT EXISTS idx_phash_prefix ON images(phash_prefix);
                """)
                cur = self.conn.execute("SELECT path,size,width,height,mtime,sha1,ahash,dhash,phash,whash,phash_prefix FROM images_old")
                rows = cur.fetchall()
                for r in rows:
                    path, size, width, height, mtime, sha1, ah, dh, ph, wh, pp = r
                    # 老库里哈希可能是整数，这里统一转 hex
                    ah = int_to_hex64(ah) if isinstance(ah, int) else ah
                    dh = int_to_hex64(dh) if isinstance(dh, int) else dh
                    ph = int_to_hex64(ph) if isinstance(ph, int) else ph
                    wh = int_to_hex64(wh) if isinstance(wh, int) else wh
                    self.conn.execute(
                        "INSERT OR REPLACE INTO images(path,size,width,height,mtime,sha1,ahash,dhash,phash,whash,phash_prefix) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (path, size, width, height, mtime, sha1, ah, dh, ph, wh, pp)
                    )
                self.conn.executescript("DROP TABLE images_old;")
                self.conn.commit()

    def upsert_image(self, feat: "ImageFeature"):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO images(path,size,width,height,mtime,sha1,ahash,dhash,phash,whash,phash_prefix) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    feat.path,
                    feat.size,
                    feat.width,
                    feat.height,
                    feat.mtime,
                    feat.sha1,
                    int_to_hex64(feat.ahash),
                    int_to_hex64(feat.dhash),
                    int_to_hex64(feat.phash),
                    int_to_hex64(feat.whash),
                    feat.phash_prefix,
                ),
            )

    def commit(self):
        with self._lock:
            self.conn.commit()

    def delete_by_path(self, path: str):
        with self._lock:
            self.conn.execute("DELETE FROM images WHERE path=?", (path,))

    def need_update(self, path: str, mtime: float) -> bool:
        with self._lock:
            cur = self.conn.execute("SELECT mtime FROM images WHERE path=? LIMIT 1", (path,))
            row = cur.fetchone()
            return (row is None) or (abs(float(row[0]) - mtime) > 1e-6)

    def find_exact_by_sha1(self, size: int, sha1: str) -> Optional[str]:
        with self._lock:
            cur = self.conn.execute("SELECT path FROM images WHERE size=? AND sha1=? LIMIT 1", (size, sha1))
            r = cur.fetchone()
            return r[0] if r else None

    def candidates_by_filters(
        self,
        size: int,
        width: int,
        height: int,
        phash_prefix: int,
        tol_ratio: float = 0.05
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        返回候选：[(path, ahash(int), dhash(int), phash(int), whash(int)), ...]
        先用 size/width/height ±tol 以及 phash_prefix 做粗筛
        """
        with self._lock:
            min_size = int(size * (1 - tol_ratio))
            max_size = int(size * (1 + tol_ratio))
            min_w = int(width * (1 - tol_ratio))
            max_w = int(width * (1 + tol_ratio))
            min_h = int(height * (1 - tol_ratio))
            max_h = int(height * (1 + tol_ratio))

            cur = self.conn.execute(
                "SELECT path, ahash, dhash, phash, whash FROM images "
                "WHERE size BETWEEN ? AND ? AND width BETWEEN ? AND ? AND height BETWEEN ? AND ? "
                "AND phash_prefix=?",
                (min_size, max_size, min_w, max_w, min_h, max_h, phash_prefix),
            )
            rows = cur.fetchall()
            out = []
            for p, ah, dh, ph, wh in rows:
                try:
                    out.append((p, hex64_to_int(ah), hex64_to_int(dh), hex64_to_int(ph), hex64_to_int(wh)))
                except Exception:
                    # 兼容性兜底
                    try:
                        out.append((p, int(ah, 16), int(dh, 16), int(ph, 16), int(wh, 16)))
                    except Exception:
                        continue
            return out


# ============== 业务引擎 =================

@dataclass
class ImageFeature:
    path: str
    size: int
    width: int
    height: int
    mtime: float
    sha1: str
    ahash: int
    dhash: int
    phash: int
    whash: int
    phash_prefix: int


def compute_feature_for_path(path: str, prefix_bits: int = 16) -> Optional[ImageFeature]:
    try:
        st = os.stat(path)
        mtime = float(st.st_mtime)
        size = int(st.st_size)

        hs = img_hashes(path)
        if not hs:
            return None
        size2, w, h, ah, dh, ph, wh = hs

        # size 用 stat 的更可靠
        sha1 = sha1_of_file(path)

        p = ph
        phash_prefix = p & ((1 << prefix_bits) - 1)

        return ImageFeature(
            path=path,
            size=size,
            width=w,
            height=h,
            mtime=mtime,
            sha1=sha1,
            ahash=ah,
            dhash=dh,
            phash=ph,
            whash=wh,
            phash_prefix=phash_prefix,
        )
    except Exception:
        return None


class DedupEngine:
    def __init__(self, main_folder: str, hash_threshold: int, use_ssim: bool, ssim_threshold: float, log_func):
        self.main_folder = main_folder
        self.hash_threshold = hash_threshold
        self.use_ssim = use_ssim
        self.ssim_threshold = ssim_threshold
        self.log = log_func
        self.db = DB(os.path.join(main_folder, DB_FILENAME))

    def build_or_update_index(self):
        paths = walk_images(self.main_folder)
        self.log(f"主库索引：发现 {len(paths)} 张图片，开始增量更新…")

        need = []
        for p in paths:
            try:
                st = os.stat(p)
                mt = float(st.st_mtime)
                if self.db.need_update(p, mt):
                    need.append(p)
            except Exception:
                continue

        self.log(f"需更新/新增 {len(need)} 张（其余已在索引中）")

        cnt = 0
        for p in need:
            feat = compute_feature_for_path(p)
            if feat:
                self.db.upsert_image(feat)
                cnt += 1
                if cnt % 200 == 0:
                    self.db.commit()
                    self.log(f"已入库 {cnt}/{len(need)}…")
        self.db.commit()
        self.log("主库索引已就绪。")

    def judge_duplicate(self, new_path: str) -> Tuple[bool, Optional[str]]:
        feat = compute_feature_for_path(new_path)
        if not feat:
            return False, None

        # 先做精确：size+sha1
        exact = self.db.find_exact_by_sha1(feat.size, feat.sha1)
        if exact:
            return True, exact

        # 候选粗筛
        cands = self.db.candidates_by_filters(
            feat.size, feat.width, feat.height, feat.phash_prefix, tol_ratio=0.06
        )

        best = None
        for path, A, D, P, W in cands:
            # 四哈希都不过阈值，才判定为不相似
            if (
                hamming64(feat.ahash, A) > self.hash_threshold
                and hamming64(feat.dhash, D) > self.hash_threshold
                and hamming64(feat.phash, P) > self.hash_threshold
                and hamming64(feat.whash, W) > self.hash_threshold
            ):
                continue

            ah_ok = hamming64(feat.ahash, A) <= self.hash_threshold
            dh_ok = hamming64(feat.dhash, D) <= self.hash_threshold
            ph_ok = hamming64(feat.phash, P) <= self.hash_threshold
            wh_ok = hamming64(feat.whash, W) <= self.hash_threshold

            # 至少3个哈希命中更稳
            hit = int(ah_ok) + int(dh_ok) + int(ph_ok) + int(wh_ok)
            if hit < 3:
                continue

            best = path
            # 可选 SSIM 精核
            if self.use_ssim:
                s = compute_ssim(new_path, path)
                if s >= self.ssim_threshold:
                    return True, path
                else:
                    continue
            else:
                return True, path

        return (best is not None), best

    def add_to_main_and_index(self, src_path: str) -> str:
        base = os.path.basename(src_path)
        name, ext = os.path.splitext(base)
        dst = os.path.join(self.main_folder, base)
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(self.main_folder, f"{name}_{i}{ext}")
            i += 1
        shutil.copy2(src_path, dst)

        feat = compute_feature_for_path(dst)
        if feat:
            self.db.upsert_image(feat)
            self.db.commit()
        return dst


# ============== GUI =================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("图片去重工具（极速增强版）")
        self.root.geometry("900x640")

        self.main_folder = tk.StringVar()
        self.new_folder = tk.StringVar()
        self.hash_threshold = tk.IntVar(value=5)
        self.ssim_threshold = tk.DoubleVar(value=0.85)
        self.use_ssim = tk.BooleanVar(value=True)

        self.dup_action = tk.StringVar(value="move")  # move: 移到隔离区；delete: 直接删除

        # UI 线程安全队列（后台线程只往队列写，主线程负责刷新界面）
        self._ui_queue = queue.Queue()
        self._ui_total = 0
        self._ui_done = 0
        self._ui_dup = 0
        self._ui_added = 0
        self._ui_pump_job = None

        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="请选择主库与新图片文件夹")
        self.total_label = None
        self.dup_label = None
        self.new_label = None

        self._op_running = False
        self._build_ui()

        # 启动主线程 UI 刷新泵
        self._ui_pump_job = self.root.after(50, self._pump_ui)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=16)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="主库（数据库）：").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(frm, textvariable=self.main_folder, width=70).grid(row=0, column=1, pady=6, sticky="we")
        ttk.Button(frm, text="浏览…", command=self._pick_main).grid(row=0, column=2, padx=8)

        ttk.Label(frm, text="新图片文件夹：").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(frm, textvariable=self.new_folder, width=70).grid(row=1, column=1, pady=6, sticky="we")
        ttk.Button(frm, text="浏览…", command=self._pick_new).grid(row=1, column=2, padx=8)

        lf = ttk.LabelFrame(frm, text="参数", padding=10)
        lf.grid(row=2, column=0, columnspan=3, sticky="we", pady=8)

        ttk.Label(lf, text="哈希汉明距离阈值：").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            lf,
            from_=1,
            to=10,
            variable=self.hash_threshold,
            command=lambda v: self.hash_threshold.set(int(float(v))),
        ).grid(row=0, column=1, sticky="we", padx=8)
        self._hash_label = ttk.Label(lf, text=str(self.hash_threshold.get()))
        self._hash_label.grid(row=0, column=2, padx=6)
        self.hash_threshold.trace_add(
            "write", lambda *args: self._hash_label.config(text=str(self.hash_threshold.get()))
        )

        ttk.Checkbutton(lf, text="启用 SSIM 精核（更准，稍慢）", variable=self.use_ssim).grid(
            row=0, column=3, padx=16
        )

        ttk.Label(lf, text="SSIM 阈值：").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Scale(
            lf,
            from_=0.70,
            to=1.00,
            variable=self.ssim_threshold,
            command=lambda v: self.ssim_threshold.set(round(float(v), 2)),
        ).grid(row=1, column=1, sticky="we", padx=8)
        self._ssim_label = ttk.Label(lf, text=str(self.ssim_threshold.get()))
        self._ssim_label.grid(row=1, column=2, padx=6)
        self.ssim_threshold.trace_add(
            "write", lambda *args: self._ssim_label.config(text=str(round(self.ssim_threshold.get(), 2)))
        )

        ttk.Label(lf, text="重复处理：").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Radiobutton(lf, text="移到隔离区（推荐）", variable=self.dup_action, value="move").grid(
            row=2, column=1, sticky="w", padx=8
        )
        ttk.Radiobutton(lf, text="直接删除（不可恢复）", variable=self.dup_action, value="delete").grid(
            row=2, column=2, sticky="w", padx=8
        )
        ttk.Label(lf, text="隔离区位置：新图片文件夹/Quarantine", foreground="#666").grid(
            row=2, column=3, sticky="w", padx=8
        )

        ttk.Button(frm, text="开始比对与处理", command=self._start).grid(row=3, column=0, columnspan=3, pady=10)

        ttk.Label(frm, text="进度：").grid(row=4, column=0, sticky="w")
        ttk.Progressbar(frm, variable=self.progress_var, length=600).grid(row=4, column=1, sticky="we", pady=4)
        ttk.Label(frm, textvariable=self.status_var).grid(row=5, column=0, columnspan=3, sticky="w", pady=4)

        ttk.Label(frm, text="日志：").grid(row=6, column=0, sticky="nw", pady=4)
        self.log_box = tk.Text(frm, height=16, width=90)
        self.log_box.grid(row=6, column=1, columnspan=2, sticky="nsew")
        sbar = ttk.Scrollbar(frm, command=self.log_box.yview)
        sbar.grid(row=6, column=3, sticky="ns")
        self.log_box.config(yscrollcommand=sbar.set)

        stats = ttk.LabelFrame(frm, text="统计", padding=10)
        stats.grid(row=7, column=0, columnspan=3, sticky="we", pady=8)
        self.total_label = ttk.Label(stats, text="总图片数：0")
        self.total_label.grid(row=0, column=0, padx=16)
        self.dup_label = ttk.Label(stats, text="重复处理：0")
        self.dup_label.grid(row=0, column=1, padx=16)
        self.new_label = ttk.Label(stats, text="新增入库：0")
        self.new_label.grid(row=0, column=2, padx=16)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(6, weight=1)

    def _mask_path(self, path: str, base: str) -> str:
        """日志脱敏：不输出完整磁盘路径，只显示相对路径末尾片段。"""
        try:
            path = os.path.normpath(path)
            base = os.path.normpath(base)
            rel = os.path.relpath(path, base)
            if rel.startswith(".."):
                return os.path.basename(path)
            parts = rel.split(os.sep)
            if len(parts) <= 2:
                return rel
            return os.path.join("...", parts[-2], parts[-1])
        except Exception:
            return os.path.basename(str(path))

    def _move_to_quarantine(self, src_path: str, new_root: str, quarantine_root: str) -> str:
        """将重复图片移动到隔离区（保留相对目录结构，避免误删不可恢复）。"""
        os.makedirs(quarantine_root, exist_ok=True)
        try:
            rel = os.path.relpath(src_path, new_root)
        except Exception:
            rel = os.path.basename(src_path)

        rel_dir = os.path.dirname(rel)
        dst_dir = os.path.join(quarantine_root, rel_dir) if rel_dir else quarantine_root
        os.makedirs(dst_dir, exist_ok=True)

        base = os.path.basename(rel)
        dst_path = os.path.join(dst_dir, base)

        root_name, ext = os.path.splitext(base)
        i = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(dst_dir, f"{root_name}_{i}{ext}")
            i += 1

        shutil.move(src_path, dst_path)
        return dst_path

    def _pump_ui(self):
        """主线程 UI 刷新泵：集中处理日志、进度、统计与完成弹窗，避免 Tk 线程问题。"""
        try:
            for _ in range(200):
                try:
                    item = self._ui_queue.get_nowait()
                except queue.Empty:
                    break

                typ = item[0]

                if typ == "log":
                    msg = item[1]
                    self.log_box.insert(tk.END, msg + "\n")
                    self.log_box.see(tk.END)

                elif typ == "status":
                    self.status_var.set(item[1])

                elif typ == "total":
                    self._ui_total = int(item[1])
                    self._ui_done = 0
                    self._ui_dup = 0
                    self._ui_added = 0
                    self.total_label.config(text=f"总图片数：{self._ui_total}")
                    self.dup_label.config(text=f"重复处理：{self._ui_dup}")
                    self.new_label.config(text=f"新增入库：{self._ui_added}")
                    self.progress_var.set(0)

                elif typ == "dup":
                    src, ref = item[1], item[2]
                    self._ui_done += 1
                    self._ui_dup += 1
                    s_src = self._mask_path(src, self.new_folder.get())
                    s_ref = self._mask_path(ref, self.main_folder.get()) if isinstance(ref, str) else str(ref)
                    self.log_box.insert(tk.END, f"重复（已删除）：{s_src}\n  匹配：{s_ref}\n")
                    self.log_box.see(tk.END)

                elif typ == "dup_move":
                    src, ref, dst = item[1], item[2], item[3]
                    self._ui_done += 1
                    self._ui_dup += 1
                    s_src = self._mask_path(src, self.new_folder.get())
                    s_dst = self._mask_path(dst, self.new_folder.get())
                    s_ref = self._mask_path(ref, self.main_folder.get()) if isinstance(ref, str) else str(ref)
                    self.log_box.insert(tk.END, f"重复（已隔离）：{s_src}\n  匹配：{s_ref}\n  移动到：{s_dst}\n")
                    self.log_box.see(tk.END)

                elif typ == "new":
                    dst = item[2]
                    self._ui_done += 1
                    self._ui_added += 1
                    s_dst = self._mask_path(dst, self.main_folder.get())
                    self.log_box.insert(tk.END, f"新增入库：{s_dst}\n")
                    self.log_box.see(tk.END)

                elif typ == "err":
                    src, err = item[1], item[2]
                    self._ui_done += 1
                    s_src = self._mask_path(src, self.new_folder.get())
                    self.log_box.insert(tk.END, f"错误：{s_src} | {err}\n")
                    self.log_box.see(tk.END)

                elif typ == "finish":
                    self.progress_var.set(100.0)
                    self.status_var.set("完成")
                    self.dup_label.config(text=f"重复处理：{self._ui_dup}")
                    self.new_label.config(text=f"新增入库：{self._ui_added}")
                    messagebox.showinfo(
                        "完成",
                        f"处理完成：\n总图片：{self._ui_total}\n重复处理：{self._ui_dup}\n新增入库：{self._ui_added}",
                    )

                elif typ == "op_end":
                    pass

            if self._ui_total > 0:
                self.progress_var.set(100.0 * self._ui_done / self._ui_total)
                self.dup_label.config(text=f"重复处理：{self._ui_dup}")
                self.new_label.config(text=f"新增入库：{self._ui_added}")
                if self._ui_done < self._ui_total:
                    self.status_var.set(f"处理中 ({self._ui_done}/{self._ui_total})…")

        finally:
            self._ui_pump_job = self.root.after(50, self._pump_ui)

    def _pick_main(self):
        d = filedialog.askdirectory(title="选择主库文件夹")
        if d:
            self.main_folder.set(d)

    def _pick_new(self):
        d = filedialog.askdirectory(title="选择新图片文件夹")
        if d:
            self.new_folder.set(d)

    def _log(self, msg: str):
        # 后台线程安全：只写入队列，由主线程统一刷新
        try:
            self._ui_queue.put(("log", str(msg)))
        except Exception:
            pass

    def _start(self):
        if self._op_running:
            messagebox.showwarning("提示", "当前任务进行中，请等待完成。")
            return
        m = self.main_folder.get().strip()
        n = self.new_folder.get().strip()
        if not m or not n:
            messagebox.showerror("错误", "请先选择主库和新图片文件夹。")
            return
        if m == n:
            messagebox.showerror("错误", "主库与新图片文件夹不能相同。")
            return

        if self.dup_action.get() == "delete":
            ok = messagebox.askyesno(
                "确认操作",
                "你选择了【直接删除重复图片】。\n\n这将永久删除新图片文件夹中的重复图片，无法恢复。\n\n是否继续？",
            )
            if not ok:
                return

        self.log_box.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.total_label.config(text="总图片数：0")
        self.dup_label.config(text="重复处理：0")
        self.new_label.config(text="新增入库：0")

        self._op_running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        """后台线程入口：只做计算与文件操作，所有 UI 更新通过队列交给主线程。"""
        try:
            engine = DedupEngine(
                self.main_folder.get(),
                hash_threshold=self.hash_threshold.get(),
                use_ssim=self.use_ssim.get(),
                ssim_threshold=self.ssim_threshold.get(),
                log_func=self._log,
            )

            self._ui_queue.put(("status", "建立/更新主库索引…"))
            engine.build_or_update_index()

            new_root = self.new_folder.get()
            new_paths = walk_images(new_root)
            total = len(new_paths)
            if total == 0:
                self._ui_queue.put(("status", "新图片文件夹无图片。"))
                return

            self._ui_queue.put(("total", total))
            self._ui_queue.put(("log", f"开始处理新文件夹，共 {total} 张…"))
            self._ui_queue.put(("status", "处理中…"))

            action = self.dup_action.get()
            quarantine_root = os.path.join(new_root, "Quarantine")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def handle_one(pth: str):
                try:
                    is_dup, ref = engine.judge_duplicate(pth)
                    if is_dup:
                        if action == "delete":
                            os.remove(pth)
                            self._ui_queue.put(("dup", pth, ref))
                        else:
                            dst = self._move_to_quarantine(pth, new_root, quarantine_root)
                            self._ui_queue.put(("dup_move", pth, ref, dst))
                    else:
                        dst = engine.add_to_main_and_index(pth)
                        self._ui_queue.put(("new", pth, dst))
                except Exception as e:
                    self._ui_queue.put(("err", pth, str(e)))

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(handle_one, p) for p in new_paths]
                for _ in as_completed(futs):
                    pass

            self._ui_queue.put(("finish",))
        finally:
            self._ui_queue.put(("op_end",))
            self._op_running = False


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.configure("TLabel", font=("SimHei", 10))
        style.configure("TButton", font=("SimHei", 10))
        style.configure("TEntry", font=("SimHei", 10))
        style.configure("TCheckbutton", font=("SimHei", 10))
    except Exception:
        pass
    app = App(root)
    root.mainloop()
