````md
# SDXL 批量裂变生成工具（sdxl_generation_engine.py）

本工具是一个本地桌面程序，用于：
- 以**参考图（原图）**为主体，通过 **SDXL + IP-Adapter** 批量生成多风格变体；
- 将生成图自动套入指定**模板库（按品类分类）**，输出 mockup 成品图；
- 对“衣服/毛毯等软材质品类”自动做去底与光影融合，并按 **3:4** 比例裁剪；
- 支持按品类设置产量、批次归档、统一命名与导出目录结构。

> 适用场景：电商图案裂变、批量产图、按品类输出 mockup。

---

## 1. 运行环境要求

### 硬件
- **NVIDIA 显卡（必须）**，并已正确安装驱动
- 建议显存：**12GB+**（越大越稳）

程序启动时会检测 CUDA：未检测到 NVIDIA 显卡会直接退出。

### 软件
- Windows / macOS / Linux（更推荐 Windows）
- Python 3.10+（推荐 3.11）

---

## 2. 安装依赖

建议使用虚拟环境（可选但推荐）：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
````

安装基础依赖：

```bash
pip install diffusers accelerate transformers safetensors
pip install pillow opencv-python numpy psutil certifi
```

安装 **PyTorch（CUDA 版本）**：

* 请按你的 CUDA/驱动版本，从 PyTorch 官方安装页选择对应命令安装（最稳）。
* 目标是让 `torch.cuda.is_available()` 返回 True。

---

## 3. 文件夹准备

### 3.1 原图文件夹（input）

放入你要裂变的参考图，支持：

* `.png`
* `.jpg` / `.jpeg`

示例：

```
input/
  cat.png
  dog.jpg
  logo.png
```

> 每张原图会被当作一个“主体”，输出目录会按主体名分开归档。

### 3.2 模板库文件夹（templates）

模板库目录必须是「**一级子文件夹 = 品类**」的结构，每个品类子文件夹里放模板图：

```
templates/
  T恤/
    001.jpg
    002.jpg
  卫衣/
    a.png
    b.png
  毛毯/
    x.jpg
    y.jpg
```

程序会自动识别这些品类，并在界面里让你为每个品类设置生成数量。

### 3.3 输出文件夹（output）

选择一个空目录或已有目录，用于保存生成结果。

---

## 4. 启动程序

在脚本所在目录运行：

```bash
python sdxl_generation_engine.py
```

启动后会弹出桌面窗口。

---

## 5. 使用步骤（GUI）

### 第一步：选择文件夹

依次选择：

* **原图文件夹**
* **模板库文件夹**
* **输出文件夹**

选择完模板库文件夹后，程序会在界面中自动列出识别到的品类。

### 第二步：设置命名规则

* **前缀字母**：例如 `A`
* **起始数字**：例如 `000001`

最终文件名形如：

```
A000001.jpg
A000002.jpg
...
```

### 第三步：设置各品类产量

在“设置各品类生产数量”区域，对每个品类输入要生成的数量（默认 100）。

### 第四步：开始生成

点击 **开始极速生产**。

程序会：

1. 逐张读取原图作为参考图；
2. 按品类与数量生成变体；
3. 保存 raw 图与 mockup 成品图；
4. 对特定品类自动执行去底/融合/3:4 裁剪（见下文）。

---

## 6. 输出目录结构说明

输出根目录会按当天日期自动建文件夹（格式：`YYMMDD`），并按“主体名/品类/批次”归档。

示例：

```
output/
  251214/
    cat/
      T恤/
        batch_001/
          raw/
            A000001_raw.jpg
            A000002_raw.jpg
          mockup/
            A000001.jpg
            A000002.jpg
          png_cutout/                # 仅部分品类会生成
            A000001_cutout.png
    dog/
      卫衣/
        batch_001/
          raw/
          mockup/
```

说明：

* `raw/`：模型直接生成的原始图
* `mockup/`：套入模板后的成品图
* `png_cutout/`：仅在需要去底的品类时输出（透明底 PNG）

---

## 7. 品类处理规则（重要）

程序会根据品类名（子文件夹名称）判断是否走“软材质流程”。

默认判定为软材质的关键词包括（可在源码里调整）：

* `T恤`, `t-shirt`, `shirt`
* `毛毯`, `blanket`
* `卫衣`, `hoodie`
* `衣服`, `cloth`, `fabric`

### 软材质品类会执行：

* 智能去底（生成透明 alpha）
* 光影/纹理融合（更贴合褶皱质感）
* 输出 `png_cutout/`
* 最终成品自动裁剪为 **3:4**

### 非软材质品类会执行：

* 直接贴图合成（不去底）
* 不强制 3:4 裁剪

---

## 8. 常见问题

### 8.1 启动提示未检测到 NVIDIA 显卡

* 确认你使用的是带 NVIDIA 独显的机器
* 安装/更新显卡驱动
* 确认 PyTorch 安装的是 CUDA 版本，并且：

  * 在 Python 里执行 `import torch; print(torch.cuda.is_available())` 为 True

### 8.2 生成很慢/显存爆了（OOM）

* 关闭占用显存的软件（游戏、浏览器大量标签、其他推理程序）
* 适当降低推理参数（需要改源码）：

  * `INFERENCE_STEPS`
  * `GEN_WIDTH / GEN_HEIGHT`
* 减少单次任务规模（降低各品类数量）

### 8.3 没识别到品类

* 检查模板库目录结构是否为“模板库/品类子文件夹/模板图片”
* 品类子文件夹中必须包含 `.png/.jpg/.jpeg`

### 8.4 导出图片大小不稳定

程序保存时会尝试把 JPG 文件体积控制在约 **1.4MB ~ 1.9MB** 之间（通过动态调节质量参数）。
如需更大/更小体积，可在源码中调整目标范围。

---

## 9. 可调参数（需要改源码）

在脚本顶部可调整：

* `GEN_WIDTH / GEN_HEIGHT`：生成分辨率（默认 1024）
* `INFERENCE_STEPS`：推理步数（默认 35）
* `BATCH_SIZE`：批次分组（影响目录归档与散热等待逻辑）
* `NEED_CUTOUT_KEYWORDS`：哪些品类走去底/融合/3:4

---

## 10. 入口文件

* 主入口：`sdxl_generation_engine.py`

```

[下载对外展示 README](sandbox:/mnt/data/README_SDXL.md)
```
