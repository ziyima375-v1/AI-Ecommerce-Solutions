[下载对外展示 README](sandbox:/mnt/data/README_web_suite.md)

```md
# 电商 Listing 生成工具（桌面版）

该目录是一套可直接运行的桌面程序：打开可视化界面，按步骤生成 Listing 所需的数据文件（例如 Excel）与相关输出结果。程序首次运行会进行授权校验。

---

## 目录结构

```

web/
index_public.html          # 前端界面文件（Eel 会加载它）
temu_listing_generator.py  # 主程序入口（运行它启动界面）
license_guard.py           # 授权校验模块

````

---

## 运行环境

- Windows / macOS / Linux
- Python 3.10+（推荐 3.11）

---

## 安装依赖

在 `web/` 目录内执行（建议先创建虚拟环境）：

```bash
pip install eel requests pandas openpyxl pillow
````

如果运行时报缺库（ImportError），按报错提示继续 `pip install xxx` 补齐即可。

---

## 配置（必做）

### 1) 大模型 API Key（如程序需要调用模型）

推荐使用环境变量提供，不要写在代码里。

**Windows（PowerShell）**

```powershell
$env:DASHSCOPE_API_KEY="你的Key"
```

**Windows（CMD）**

```bat
set DASHSCOPE_API_KEY=你的Key
```

**macOS / Linux**

```bash
export DASHSCOPE_API_KEY="你的Key"
```

> 如果你使用的是其他 Key 名称，请以程序实际读取的环境变量为准。

### 2) 授权

* 首次启动会弹窗要求输入授权码（由提供方发放）
* 授权成功后本机会保存授权票据，下次启动会自动校验

---

## 启动方式

进入 `web/` 目录，运行：

```bash
python temu_listing_generator.py
```

启动后会自动打开桌面窗口（或自动拉起浏览器界面，取决于你的实现方式）。

---

## 使用流程（界面）

界面按钮名称以实际显示为准，一般流程为：

1. 选择/导入需要处理的素材或模板
2. 按界面提示填写必要字段（例如标题/描述/属性等）
3. 点击生成/导出
4. 在输出目录中获得生成结果（例如 `.xlsx` 文件）

---

## 常见问题

### 1) 打开后白屏/页面不显示

* 确认 `index_public.html` 与 `temu_listing_generator.py` 在同一目录（或程序配置的 web 目录正确）
* 检查控制台是否提示端口占用（常见是 8081 被占用）

  * 解决：关闭占用端口的软件，或在主程序里把 `eel.start(..., port=xxxx)` 改成未占用端口

### 2) 提示缺少模块

按提示安装依赖即可，例如：

```bash
pip install eel
pip install openpyxl
```

### 3) 授权失败

* 确认网络可用（授权校验通常需要联网）
* 确认授权码输入无误（注意大小写与前后空格）
* 如仍失败，联系提供方刷新/更换授权码

---

## 入口文件

* 主入口：`temu_listing_generator.py`

```
```
