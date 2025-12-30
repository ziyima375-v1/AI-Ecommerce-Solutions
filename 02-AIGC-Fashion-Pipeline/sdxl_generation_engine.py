import os
import sys

# ================= ⚠️ 环境变量配置 ⚠️ =================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

# 如需镜像/加速，可自行设置环境变量 HF_ENDPOINT（例如指向镜像站）。
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# =======================================================

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import torch
import numpy as np
import cv2
import random
import time
from PIL import Image, ImageOps, ImageEnhance, ImageChops, ImageFilter, ImageStat
from datetime import datetime

# ================= 🛡️ 依赖库自检 =================
try:
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
except ImportError as e:
    class StableDiffusionXLPipeline:
        pass


    class DPMSolverMultistepScheduler:
        pass


    print(f"环境错误: 缺少库！{e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("硬件错误: 未检测到 NVIDIA 显卡！")
    sys.exit(1)

# ================= 0. 配置区域 =================
GEN_WIDTH = 1024
GEN_HEIGHT = 1024
BATCH_SIZE = 1000
INFERENCE_STEPS = 35

# 🚀 提示词配置
QUALITY_BOOSTERS = "vector art, screen print style, flat color, solid vivid colors, sharp edges, no anti-aliasing, clean lines, 4k, best quality"

# 🛠️ 【配置】定义需要抠图的软性品类关键词
# 只有这些品类会走：抠图 -> 光影融合 -> 强制3:4裁剪
NEED_CUTOUT_KEYWORDS = ["T恤", "t-shirt", "shirt", "毛毯", "blanket", "卫衣", "hoodie", "衣服", "cloth", "fabric"]

# 模版坐标配置
TEMPLATE_COORDS = {}

# 说明：原本这里可以针对“特定模板文件名”做精确贴图坐标（避免错位）。
# 对外分发时不建议把你私有模板文件名/坐标写死在源码里。
# 如需精确坐标，可自行在此处补充： {"模板文件名.jpg": (x, y, w, h), ...}


# ================= 1. 基因库 =================
PROMPT_GENES = {
    "STYLES": ["flat vector illustration", "vintage comic book style", "pop art style", "minimalist bauhaus design",
               "linocut print style", "risograph print style", "japanese ukiyo-e style", "art deco poster style",
               "propaganda poster style", "stained glass art", "paper cut-out layer art", "graffiti street art",
               "sticker art with white border", "blueprint schematic style", "tarot card style", "low poly 3d render",
               "claymation plasticine style", "isometric 3d view", "unreal engine 5 render", "octane render",
               "zbrush sculpt style", "knolling photography", "lego brick style", "voxel art style",
               "origami paper craft", "porcelain ceramic style", "glass blowing art", "cyberpunk style",
               "steampunk mechanical style", "vaporwave aesthetic", "impressionist oil painting",
               "watercolor splash art", "ink wash painting", "charcoal sketch", "pencil drawing", "pointillism dot art",
               "gothic horror style", "surrealist dreamscape", "psychedelic trance art", "glitch art datamosh",
               "thermal imaging style", "x-ray aesthetic", "double exposure photography", "bioluminescent fantasy",
               "post-apocalyptic style"],
    "MATERIALS": ["made of liquid metal", "made of transparent glass", "made of fluffy yarn", "made of carved wood",
                  "made of colorful candy", "made of smoke and fire", "made of circuit boards",
                  "made of flowers and leaves", "gold plated texture", "holographic texture", "marble stone texture",
                  "denim fabric texture", "leather texture", "ceramic porcelain", "jelly gummy texture", "rusty metal",
                  "obsidian black stone", "diamond crystal", "made of clouds and mist", "made of flowing water",
                  "made of neon lights", "made of plastic wrap", "made of crumpled paper", "made of mosaic tiles",
                  "made of feathers", "made of wires and cables", "made of ice and snow", "made of chocolate",
                  "made of soap bubbles", "made of lace fabric"],
    "VIEWS": ["side profile view", "front symmetric view", "isometric 3d view", "wide angle fish-eye lens",
              "macro close-up shot", "dynamic action pose", "floating in zero gravity", "worm's eye view",
              "bird's eye view", "exploded view diagram", "looking over shoulder", "selfie angle", "drone aerial view",
              "through a keyhole view"],
    "LIGHTING": ["cinematic lighting", "studio soft lighting", "neon volumetric fog", "natural sunlight",
                 "dramatic rim lighting", "bioluminescent glow", "rembrandt lighting", "colorful lens flare",
                 "dark moody lighting", "god rays", "ultraviolet blacklight", "candle light warm glow",
                 "underwater caustic lighting", "disco strobe lights", "fireplace glow", "cyberpunk city reflection",
                 "golden hour sunset"],
    "BACKGROUNDS": ["on clean white background"],
    "ARTISTS": ["style of Van Gogh", "style of Picasso", "style of Hokusai", "style of Studio Ghibli",
                "style of Makoto Shinkai", "style of Da Vinci", "style of Andy Warhol", "style of Banksy",
                "style of Miyazaki", "style of HR Giger", "style of Dali", "style of Monet", "style of Keith Haring",
                "style of Basquiat", "style of Mucha", "style of Tim Burton", "style of Pixar", "style of Disney"]
}


# ================= 2. 核心引擎类 =================
class FissionEngine:
    def __init__(self, log_func):
        self.pipe = None
        self.log = log_func

    def load_model(self):
        if self.pipe: return
        self.log("⏳ 正在加载 SDXL...")
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.log("⏳ 加载 IP-Adapter...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors"
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
            self.log("✅ 引擎就绪！")
        except Exception as e:
            self.log(f"模型加载失败: {str(e)}")
            print(f"模型加载失败: {str(e)}")
            raise RuntimeError(str(e))

    def generate(self, original_image, prompt_combine, seed):
        full_prompt = prompt_combine
        neg_prompt = "photo, realistic, soft edges, blurry, shadow, glow, 3d render, gradient, dirty, noise, grain, low res, jpeg artifacts, complex details"

        self.pipe.set_ip_adapter_scale(random.uniform(0.55, 0.75))
        try:
            images = self.pipe(
                prompt=full_prompt,
                negative_prompt=neg_prompt,
                ip_adapter_image=original_image,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=8.0,
                width=GEN_WIDTH,
                height=GEN_HEIGHT,
                generator=torch.Generator("cpu").manual_seed(seed)
            ).images
            return images[0]
        except Exception as e:
            self.log(f"⚠️ 生成失败: {e}")
            return Image.new("RGB", (GEN_WIDTH, GEN_HEIGHT), "black")


# ================= 3. 模版管理器 (3:4裁剪+分流) =================
class MockupManager:
    def __init__(self, root_dir):
        self.categories = {}
        self.coords_config = TEMPLATE_COORDS
        self.image_cache = {}

        if root_dir and os.path.exists(root_dir):
            for entry in os.listdir(root_dir):
                full_path = os.path.join(root_dir, entry)
                if os.path.isdir(full_path):
                    tpls = [os.path.join(full_path, f) for f in os.listdir(full_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if tpls:
                        self.categories[entry] = sorted(tpls)

    def get_template_from_category(self, cat_name):
        if cat_name in self.categories and self.categories[cat_name]:
            return random.choice(self.categories[cat_name])
        return None

    # 👇 3:4 比例裁剪
    def crop_to_3_4(self, img):
        w, h = img.size
        target_ratio = 3.0 / 4.0
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            return img.crop((left, 0, left + new_w, h))
        elif current_ratio < target_ratio:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            return img.crop((0, top, w, top + new_h))
        return img

    def smart_floodfill_bg(self, image):
        try:
            orig_w, orig_h = image.size
            upscale_w, upscale_h = orig_w * 3, orig_h * 3
            img_pil = image.resize((upscale_w, upscale_h), Image.LANCZOS).convert("RGB")
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]

            mask = np.zeros((h + 2, w + 2), np.uint8)
            loDiff = (25, 25, 25)
            upDiff = (25, 25, 25)
            seeds = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]

            for seed in seeds:
                cv2.floodFill(img_bgr, mask, seed, 0, loDiff, upDiff,
                              flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)

            mask = mask[1:-1, 1:-1]
            mask_inv = cv2.bitwise_not(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_inv = cv2.erode(mask_inv, kernel, iterations=1)
            _, mask_hard = cv2.threshold(mask_inv, 127, 255, cv2.THRESH_BINARY)
            b, g, r = cv2.split(img_bgr)
            img_bgra = cv2.merge([b, g, r, mask_hard])
            result_large = Image.fromarray(cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA))
            return result_large.resize((orig_w, orig_h), Image.LANCZOS)
        except Exception as e:
            print(f"智能去底失败: {e}")
            return image.convert("RGBA")

    def get_cached_image(self, tpl_path):
        if tpl_path not in self.image_cache:
            try:
                img = Image.open(tpl_path).convert("RGBA")
                self.image_cache[tpl_path] = img
            except:
                return None
        return self.image_cache[tpl_path].copy()

    def blend(self, design_img, tpl_path, cat_name):
        try:
            bg = self.get_cached_image(tpl_path)
            if bg is None: return design_img, None

            bg_w, bg_h = bg.size
            filename = os.path.basename(tpl_path)

            if filename in self.coords_config:
                x, y, w, h = self.coords_config[filename]
            else:
                target_w, target_h = int(bg_w * 0.55), int(bg_h * 0.55)
                x = (bg_w - target_w) // 2
                y = (bg_h - target_h) // 2
                w, h = target_w, target_h

            is_wrinkle_item = False
            for kw in NEED_CUTOUT_KEYWORDS:
                if kw.lower() in cat_name.lower():
                    is_wrinkle_item = True
                    break

            # 🅰️ 硬材质 -> 直贴
            if not is_wrinkle_item:
                design_final = ImageOps.fit(design_img, (w, h), method=Image.LANCZOS)
                bg_copy = bg.copy()
                bg_copy.paste(design_final, (x, y))
                final_w, final_h = bg_copy.size
                final_comp_large = bg_copy.resize((int(final_w * 1.5), int(final_h * 1.5)), Image.LANCZOS).convert(
                    "RGB")
                return final_comp_large, None

            # 🅱️ 软材质 -> 抠图+3:4裁剪
            else:
                design_clean = self.smart_floodfill_bg(design_img)
                cutout_png = design_clean.copy()

                design_final = ImageOps.fit(design_clean, (w, h), method=Image.LANCZOS)
                design_final = design_final.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))
                design_final = ImageEnhance.Color(design_final).enhance(1.05)

                bg_crop = bg.crop((x, y, x + w, y + h)).convert("RGB")
                bg_gray = bg_crop.convert("L")
                stat = ImageStat.Stat(bg_gray)
                avg_bg_light = stat.mean[0]
                gain = 230.0 / (avg_bg_light + 1.0)
                if gain < 1.0: gain = 1.0
                illumination_mask = bg_gray.point(lambda p: min(255, int(p * gain)))
                illumination_mask = illumination_mask.filter(ImageFilter.GaussianBlur(1.5))
                illumination_layer = illumination_mask.convert("RGB")

                blurred = bg_gray.filter(ImageFilter.GaussianBlur(radius=3))
                texture_diff = ImageChops.difference(bg_gray, blurred)
                texture_diff = ImageOps.invert(texture_diff)
                texture_diff = ImageEnhance.Contrast(texture_diff).enhance(3.0)
                texture_layer = texture_diff.convert("RGB")

                design_rgb = design_final.convert("RGB")
                design_shaded = ImageChops.multiply(design_rgb, illumination_layer)
                design_final_look = Image.blend(design_shaded, ImageChops.multiply(design_shaded, texture_layer), 0.08)

                _, _, _, final_alpha = design_final.split()
                final_alpha_np = np.array(final_alpha)
                final_alpha_np = np.where(final_alpha_np > 100, 255, 0).astype(np.uint8)
                final_alpha = Image.fromarray(final_alpha_np)

                r, g, b = design_final_look.split()
                overlay_content = Image.merge('RGBA', (r, g, b, final_alpha))

                overlay_layer = Image.new('RGBA', (bg_w, bg_h), (0, 0, 0, 0))
                overlay_layer.paste(overlay_content, (x, y))

                final_comp = Image.alpha_composite(bg, overlay_layer)
                final_w, final_h = final_comp.size
                final_comp_large = final_comp.resize((int(final_w * 1.5), int(final_h * 1.5)), Image.LANCZOS).convert(
                    "RGB")

                final_comp_large = self.crop_to_3_4(final_comp_large)
                return final_comp_large, cutout_png

        except Exception as e:
            print(f"融合异常 ({tpl_path}): {e}")
            import traceback
            traceback.print_exc()
            return design_img, None


# ================= GUI 主程序 =================
class SDXLFissionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SDXL 批量生成工具")
        self.root.geometry("1000x850")

        self.input_dir = tk.StringVar()
        self.template_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.name_prefix = tk.StringVar(value="A")
        self.name_start_num = tk.StringVar(value="000001")

        self.progress_val = tk.DoubleVar(value=0)
        self.is_running = False
        self.engine = None
        self.mockup_mgr = None
        self.quota_entries = {}

        self.setup_ui()

    def setup_ui(self):
        ttk.Label(self.root, text="🏭 SDXL 批量生成工具", font=("微软雅黑", 16, "bold")).pack(
            pady=15)

        # 1. 文件夹
        frame_p = ttk.LabelFrame(self.root, text="第一步：选择文件夹", padding=10)
        frame_p.pack(fill="x", padx=15)
        self.mk_entry(frame_p, "原图文件夹:", self.input_dir, 0)
        self.mk_entry(frame_p, "模板库文件夹:", self.template_dir, 1, cmd=self.on_template_selected)
        self.mk_entry(frame_p, "输出文件夹:", self.output_dir, 2)

        # 1.5 命名设置
        frame_n = ttk.LabelFrame(self.root, text="命名设置 (字母+6位数字)", padding=10)
        frame_n.pack(fill="x", padx=15, pady=5)
        ttk.Label(frame_n, text="前缀字母 (如 A):").pack(side="left")
        ttk.Entry(frame_n, textvariable=self.name_prefix, width=10).pack(side="left", padx=5)
        ttk.Label(frame_n, text="起始数字 (如 000001):").pack(side="left")
        ttk.Entry(frame_n, textvariable=self.name_start_num, width=15).pack(side="left", padx=5)

        # 2. 配额
        self.frame_quota = ttk.LabelFrame(self.root, text="第二步：设置各品类生产数量", padding=10)
        self.frame_quota.pack(fill="both", expand=True, padx=15, pady=10)
        self.lbl_tip = ttk.Label(self.frame_quota, text="👉 请先选择【模板库文件夹】...", foreground="gray")
        self.lbl_tip.pack(pady=20)

        # 3. 底部
        frame_act = ttk.Frame(self.root, padding=10)
        frame_act.pack(fill="x", padx=15)
        self.progressbar = ttk.Progressbar(frame_act, variable=self.progress_val, maximum=100)
        self.progressbar.pack(fill="x", pady=5)
        self.btn_run = ttk.Button(frame_act, text="🔥 开始极速生产", command=self.start_thread, state="disabled")
        self.btn_run.pack(fill="x", ipady=10)

        # 日志
        self.log_text = scrolledtext.ScrolledText(self.root, height=10, bg="#1e1e1e", fg="#00ff00",
                                                  font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=15, pady=5)

    def mk_entry(self, p, l, v, r, cmd=None):
        ttk.Label(p, text=l).grid(row=r, column=0, sticky="w")
        ttk.Entry(p, textvariable=v, width=60).grid(row=r, column=1, padx=5)
        if cmd:
            ttk.Button(p, text="📂 浏览", command=lambda: self.select_path(v, cmd)).grid(row=r, column=2)
        else:
            ttk.Button(p, text="📂 浏览", command=lambda: self.select_path(v)).grid(row=r, column=2)

    def select_path(self, var, callback=None):
        p = filedialog.askdirectory()
        if p:
            var.set(p)
            if callback: callback()

    def on_template_selected(self):
        tpl_dir = self.template_dir.get()
        if not tpl_dir or not os.path.exists(tpl_dir): return
        for widget in self.frame_quota.winfo_children(): widget.destroy()
        self.quota_entries = {}
        self.mockup_mgr = MockupManager(tpl_dir)
        categories = list(self.mockup_mgr.categories.keys())

        if not categories:
            ttk.Label(self.frame_quota, text="⚠️ 未在模板库中发现子文件夹！", foreground="red").pack()
            self.btn_run.config(state="disabled")
            return

        canvas = tk.Canvas(self.frame_quota, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.frame_quota, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        r = 0
        ttk.Label(scrollable_frame, text="检测到的品类", font=("", 10, "bold")).grid(row=r, column=0, padx=10, pady=5)
        ttk.Label(scrollable_frame, text="计划生成数量", font=("", 10, "bold")).grid(row=r, column=1, padx=10, pady=5)
        r += 1

        for cat in categories:
            ttk.Label(scrollable_frame, text=cat).grid(row=r, column=0, sticky="e", padx=10, pady=2)
            ent = ttk.Spinbox(scrollable_frame, from_=0, to=100000, width=10)
            ent.set(100)
            ent.grid(row=r, column=1, sticky="w", padx=10, pady=2)
            self.quota_entries[cat] = ent
            r += 1
        self.btn_run.config(state="normal")
        self.log(f"✅ 已识别 {len(categories)} 个品类。")

    def log(self, msg):
        self.root.after(0, lambda: self.log_text.insert(tk.END, msg + "\n") or self.log_text.see(tk.END))

    def save_smart_size(self, img, path):
        target_min = 1.4 * 1024 * 1024
        target_max = 1.9 * 1024 * 1024
        q = 95
        try:
            img.save(path, quality=q, subsampling=0)
            file_size = os.path.getsize(path)
            for _ in range(5):
                if file_size > target_max:
                    if q <= 80: break
                    q -= 3
                    img.save(path, quality=q, subsampling=0)
                    file_size = os.path.getsize(path)
                elif file_size < target_min:
                    if q >= 100: break
                    q += 2
                    img.save(path, quality=q, subsampling=0)
                    file_size = os.path.getsize(path)
                else:
                    break
        except Exception as e:
            print(f"保存失败，尝试普通保存: {e}")
            img.save(path, quality=90)

    def start_thread(self):
        if self.is_running: return
        self.production_plan = {}
        total_count = 0
        for cat, ent in self.quota_entries.items():
            try:
                count = int(ent.get())
                if count > 0:
                    self.production_plan[cat] = count
                    total_count += count
            except:
                pass

        if total_count == 0:
            messagebox.showwarning("提示", "请至少设置一个数量！")
            return

        inp = self.input_dir.get()
        out = self.output_dir.get()
        if not inp or not out:
            messagebox.showerror("错误", "请选择文件夹！")
            return

        self.is_running = True
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        try:
            inp = self.input_dir.get()
            out = self.output_dir.get()

            # 🆕 获取命名设置
            file_prefix = self.name_prefix.get()
            try:
                current_serial = int(self.name_start_num.get())
            except ValueError:
                self.log("⚠️ 起始数字无效，默认从1开始")
                current_serial = 1

            date_str = datetime.now().strftime("%y%m%d")
            date_root_dir = os.path.join(out, date_str)
            if not os.path.exists(date_root_dir): os.makedirs(date_root_dir)

            if self.engine is None:
                self.engine = FissionEngine(self.log)
                self.engine.load_model()

            files = [f for f in os.listdir(inp) if f.lower().endswith(('.png', '.jpg'))]
            total_ops = len(files) * sum(self.production_plan.values())
            done_ops = 0

            for f in files:
                subj_name = os.path.splitext(f)[0]
                self.log(f"🎬 处理主体: {subj_name}")
                try:
                    orig_img = Image.open(os.path.join(inp, f)).convert("RGB")
                except:
                    continue

                for cat_name, count in self.production_plan.items():
                    self.log(f"  👉 {cat_name}: {count}张")
                    cat_root_dir = os.path.join(date_root_dir, subj_name, cat_name)

                    for i in range(count):
                        batch_idx = (i // BATCH_SIZE) + 1
                        batch_name = f"batch_{batch_idx:03d}"
                        current_batch_dir = os.path.join(cat_root_dir, batch_name)
                        dir_raw = os.path.join(current_batch_dir, "raw")
                        dir_mockup = os.path.join(current_batch_dir, "mockup")

                        if not os.path.exists(dir_raw): os.makedirs(dir_raw)
                        if not os.path.exists(dir_mockup): os.makedirs(dir_mockup)

                        style = random.choice(PROMPT_GENES["STYLES"])
                        mat = random.choice(PROMPT_GENES["MATERIALS"])
                        view = random.choice(PROMPT_GENES["VIEWS"])
                        light = random.choice(PROMPT_GENES["LIGHTING"])

                        bg_prompt = "white background, clean negative space, sticker"
                        base_prompt = f"{QUALITY_BOOSTERS}, {style}, {mat}, {view}, {light}, {bg_prompt}"

                        if random.random() > 0.7:
                            artist = random.choice(PROMPT_GENES["ARTISTS"])
                            combined_prompt = f"{base_prompt}, {artist}"
                        else:
                            combined_prompt = base_prompt

                        seed = random.randint(0, 2 ** 32 - 1)
                        gen_img = self.engine.generate(orig_img, combined_prompt, seed)
                        if np.mean(np.array(gen_img)) == 0: continue

                        # ✅ 核心修改：使用 :06d 强制补零为6位
                        # 如果 current_serial 是 1，变成 000001
                        # 如果 current_serial 是 100，变成 000100
                        filename_base = f"{file_prefix}{current_serial:06d}"

                        # 保存Raw
                        self.save_smart_size(gen_img, os.path.join(dir_raw, f"{filename_base}_raw.jpg"))

                        tpl_path = self.mockup_mgr.get_template_from_category(cat_name)
                        if tpl_path:
                            final_img, cutout_png = self.mockup_mgr.blend(gen_img, tpl_path, cat_name)

                            self.save_smart_size(final_img, os.path.join(dir_mockup, f"{filename_base}.jpg"))

                            if cutout_png is not None:
                                dir_png = os.path.join(current_batch_dir, "png_cutout")
                                if not os.path.exists(dir_png): os.makedirs(dir_png)
                                cutout_png.save(os.path.join(dir_png, f"{filename_base}_cutout.png"))
                        else:
                            self.save_smart_size(gen_img, os.path.join(dir_mockup, f"{filename_base}.jpg"))

                        done_ops += 1
                        current_serial += 1  # 递增序列号
                        self.progress_val.set((done_ops / total_ops) * 100)

                        if i % 50 == 0: self.log(f"    ⚡ {cat_name}: {i}/{count}")

                        if (i + 1) % BATCH_SIZE == 0 and (i + 1) < count:
                            wait_m = random.randint(10, 15)
                            self.log(f"☕ 显卡散热 {wait_m} 分钟...")
                            for _ in range(wait_m * 60):
                                if not self.is_running: break
                                time.sleep(1)

            self.log("🏁 任务达成！")
            messagebox.showinfo("成功", f"归档完成！\n文件夹: {date_root_dir}")

        except Exception as e:
            self.log(f"❌ 运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self.btn_run.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = SDXLFissionApp(root)
    root.mainloop()