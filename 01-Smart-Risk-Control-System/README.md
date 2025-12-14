````md
# TemuTRO 图片风控扫描工具（tro_risk_analyzer.py）

本项目提供一个本地桌面程序，用于对图片进行批量风险扫描（TRO/版权/商标/文字敏感等方向），输出每张图片的风险等级与建议，并支持将高风险清单导出为 Excel，便于运营/设计复核处理。

---

## 1. 运行环境

- Windows / macOS / Linux（建议 Windows）
- Python 3.10+（推荐 3.11）
- 依赖库（按程序实际导入安装）：

```bash
pip install PyQt6 pandas pillow requests psutil certifi openpyxl
````

建议使用虚拟环境（可选）：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## 2. API Key 配置

程序需要 API Key 才能调用模型接口。支持两种方式提供：

### 方式 A：环境变量（推荐）

* Windows（PowerShell）：

```powershell
$env:DASHSCOPE_API_KEY="你的Key"
```

* Windows（CMD）：

```bat
set DASHSCOPE_API_KEY=你的Key
```

* macOS / Linux：

```bash
export DASHSCOPE_API_KEY="你的Key"
```

也支持备用变量名：

* `TRO_API_KEY`

### 方式 B：运行时输入

如果未设置环境变量，程序会弹窗提示输入 API Key（密码框输入）。

---

## 3. 启动程序

在项目目录执行：

```bash
python tro_risk_analyzer.py
```

启动后会打开桌面窗口。

---

## 4. 使用流程（GUI）

界面按钮名称以程序实际显示为准，常用流程如下：

### 4.1 导入图片

* **上传图片**：选择单张/多张图片导入列表
* **上传文件夹**：选择一个目录，程序将扫描该目录下的图片并加入列表
* **导入**：将已选图片/文件夹内容加入待分析列表（如果你的界面存在该步骤）

### 4.2 开始分析

* 点击 **开始分析**
* 程序会对列表中的图片逐张调用模型进行扫描
* 分析过程中会更新进度与状态提示

### 4.3 查看与筛选结果

* 结果会在表格中展示（例如：文件名、风险等级、建议、详细说明、评分等）
* 对于标记为高风险的图片，会进入“高风险”列表区域（若界面有该区域）

### 4.4 导出高风险清单（Excel）

* 点击 **导出高风险图片**
* 选择保存位置
* 输出文件为 `.xlsx`（内容为高风险图片清单，便于二次处理/分发给同事）

### 4.5 清理与删除

* **清空**：清空当前导入列表/结果（以实际按钮行为为准）
* **删除选中**：删除表格中选中的条目

### 4.6 预览

* **点击查看**：打开图片预览窗口（若界面提供该功能）

---

## 5. 常见问题

### 5.1 运行报错：缺少模块 / ImportError

按提示安装对应依赖即可，通常是：

```bash
pip install PyQt6 pandas pillow requests psutil certifi openpyxl
```

### 5.2 导出 Excel 报错

请确认安装了 `openpyxl`：

```bash
pip install openpyxl
```

### 5.3 提示限流/429 或调用失败

* 可能是接口限流、Key 配额不足或网络不稳定
* 程序会做一定的重试；若持续失败，建议降低并发/稍后再试或更换可用 Key

---

## 6. 入口文件

* 主入口：`tro_risk_analyzer.py`

```

[下载 README（对外使用说明版）](sandbox:/mnt/data/README_Usage.md)
```
