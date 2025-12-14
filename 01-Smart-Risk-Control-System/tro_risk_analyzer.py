import os
import sys
import time
import json
import base64
import re
import requests
import io
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any, Set
import multiprocessing
import psutil
import os, sys, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()  # 保证 HTTPS 正常工作

def auto_concurrency(min_workers: int = 2,
                     max_workers: int = 16,
                     io_ratio: float = 0.7) -> int:
    """
    根据本机 CPU 核心数自动给出合适的线程池大小。
    min_workers : 最小并发数
    max_workers : 最大并发数（防止宿主机核数过大）
    io_ratio    : IO 密集型任务时，认为多少比例的时间在等待 IO
    """
    cores = psutil.cpu_count(logical=True) or 4  # 兜底 4
    # IO 密集任务常见经验公式：核心数 * (1 + io_ratio)
    workers = max(min_workers, int(cores * (1 + io_ratio) + 0.5))
    return min(workers, max_workers)

# GUI库导入
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QInputDialog, QFileDialog, QTextEdit,
    QScrollArea, QFrame, QSplitter, QMessageBox, QStatusBar, QGridLayout,
    QGroupBox, QTabWidget, QProgressBar, QGraphicsDropShadowEffect,
    QTableWidget, QTableWidgetItem, QCheckBox, QDialog, QHeaderView,
    QSpacerItem, QSizePolicy
)

from PyQt6.QtGui import (
    QIcon, QColor, QBrush, QFont, QClipboard, QPixmap
)

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QEvent, QPoint, QMutex, QMutexLocker  # 新增QMutex
)

# 常量定义
MAX_IMAGE_SIZE: int = 512  # 图片最大压缩尺寸（像素）
BATCH_SIZE: int = 10  # 批量处理大小
DEFAULT_MAX_TOKENS: int = 800  # 默认API调用tokens
CACHE_EXPIRY_DAYS: int = 7  # 缓存过期天数
PROGRESS_UPDATE_INTERVAL: int = 1  # 进度更新间隔（秒）


class CustomMessageBox(QMessageBox):
    """自定义消息框，用于显示上传成功或失败的提示"""

    def __init__(self, message, is_success=True, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 设置消息框样式
        self.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                padding: 20px;
            }
            QLabel {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 16px;
                color: #1e293b;
                margin: 10px 20px;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                padding: 8px 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 15px;
                font-weight: 500;
                border: none;
                transition: all 0.2s ease;
                min-width: 80px;
                margin: 0 10px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)

        # 根据成功或失败设置图标和样式
        if is_success:
            self.setIconPixmap(QPixmap(
                "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEwIDE2LjE3TDE1LjU5IDEwLjU4TDE2Ljk5IDExLjk5TDEwIDIwTDMgMTNMMDQuNDEgMTEuNDFMMTAgMTYuMTd6IiBmaWxsPSIjMTBiOThhIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMyIgZmlsbD0iIzEwYjI4YSIvPjxzdmc+"))
            self.setWindowTitle("上传成功")
            self.setText(f"<span style='color:#10b981;'>{message}</span>")
        else:
            self.setIconPixmap(QPixmap(
                "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMTRjLTEuMTEgMC0yLS44OS0yLTJzLjg5LTIgMi0yIDIgLjg5IDIgMi0uODkgMi0yIDJ6bTAtOGMtMS4xMSAwLTIuLS44OS0yLTJzLjg5LTIgMi0yIDIgLjg5IDIgMi0uODkgMi0yIDJ6Ii8+PC9zdmc+"))
            self.setWindowTitle("上传失败")
            self.setText(f"<span style='color:#ef4444;'>{message}</span>")

        self.addButton("确定", QMessageBox.ButtonRole.AcceptRole)


class APIThrottledError(Exception):
    """API限流异常类"""

    def __init__(self, message: str, retry_after: int):  # 补充注解
        super().__init__(message)
        self.retry_after = retry_after


class ImageAnalysisThread(QThread):
    progress_updated = pyqtSignal(str, int)
    result_added = pyqtSignal(str, str, str, str)
    analysis_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str, int)
    batch_completed = pyqtSignal()

    def __init__(self, images: List[str], api_key: str, base_url: str, prompt_text: str,
                 parent: 'ImageAnalyzerApp') -> None:
        super().__init__()
        self.images: List[str] = images
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.prompt_text: str = prompt_text
        self.analysis_results: Dict[str, dict] = {}  # 存储完整分析结果
        self.processing_times: List[float] = []
        self.is_running: bool = True  # 运行标志，用于安全停止线程
        self.total_tokens_used: int = 0
        self.parent: 'ImageAnalyzerApp' = parent  # 保存父窗口引用，用于访问缓存
        self.cache_mutex: QMutex = QMutex()  # 缓存锁，确保线程安全

    def _get_cached_result(self, image_path: str) -> Optional[Tuple[dict, int]]:
        """从缓存中获取分析结果（带过期检查）"""
        self.cache_mutex.lock()
        try:
            if image_path in self.parent.analysis_cache:
                # 缓存结构: (结果, tokens, 时间戳)
                result, tokens_used, timestamp = self.parent.analysis_cache[image_path]
                # 检查是否过期
                if time.time() - timestamp < CACHE_EXPIRY_DAYS * 86400:  # 转换为秒
                    return (result, tokens_used)
                else:
                    # 过期缓存自动清理
                    del self.parent.analysis_cache[image_path]
                    self.parent.save_cache()
            return None
        finally:
            self.cache_mutex.unlock()

    def _cache_result(self, image_path: str, result: dict, tokens_used: int) -> None:
        """将分析结果写入缓存（带时间戳），确保线程安全"""
        self.cache_mutex.lock()
        try:
            timestamp = time.time()  # 记录缓存时间
            self.parent.analysis_cache[image_path] = (result, tokens_used, timestamp)
            # 在这里调用 save_cache，确保它在锁的保护下
            self.parent.save_cache()
        finally:
            self.cache_mutex.unlock()

    def run(self) -> None:
        """使用线程池控制并发数，只更新总进度"""
        total: int = len(self.images)
        start_time: float = time.time()
        self.processed_count = 0  # 已处理数量

        # 动态调整并发数（根据CPU核心数）
        max_workers: int = auto_concurrency()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index: Dict = {
                executor.submit(self._process_image, i, img_data): i
                for i, img_data in enumerate(self.images)
            }

            for future in as_completed(future_to_index):
                if not self.is_running:
                    break

                try:
                    future.result()  # 获取任务结果，可能抛出异常
                    self.processed_count += 1
                    # 计算并更新总进度
                    total_progress = int((self.processed_count / total) * 100)
                    self.progress_updated.emit(
                        f"已完成 {self.processed_count}/{total} 张图片分析",
                        total_progress
                    )
                except Exception as e:
                    self.error_occurred.emit(f"分析图片时出错: {str(e)}", 500)

        if self.is_running:
            end_time = time.time()
            total_time = end_time - start_time
            mins, secs = divmod(total_time, 60)
            self.analysis_completed.emit(f"总耗时: {int(mins)}分钟{int(secs)}秒")
            self.batch_completed.emit()

    def _process_image(self, index, img_data):
        """处理单个图片的分析任务，不再更新单图进度"""
        display_path = img_data
        start_processing = time.time()
        file_name = os.path.basename(display_path)
        total = len(self.images)

        # 检查缓存
        cached_result = self._get_cached_result(display_path)
        if cached_result:
            result, tokens_used = cached_result
            self.result_added.emit(display_path, result["风险等级"], result["建议"], result["分析内容"])
            return  # 跳过API调用

        try:
            # 分析图片
            result, tokens_used = self._analyze_image_with_retry(display_path)
        except Exception as e:
            # logging.error(f"处理图片 {display_path} 时出错: {str(e)}") # 移除logging，使用emit
            self.error_occurred.emit(f"处理图片 {display_path} 时出错: {str(e)}", 500)
            self.result_added.emit(display_path, "错误", f"分析失败: {str(e)}", "")
            return

        # 保存分析结果
        self.analysis_results[display_path] = result
        self.total_tokens_used += tokens_used

        # 直接使用模型返回的风险等级和建议
        risk_level = result.get("风险等级", "未知")
        suggestion = result.get("建议", "无")
        analysis_content = result.get("分析内容", "")

        self.result_added.emit(display_path, risk_level, suggestion, analysis_content)

        # 分析完成后存入缓存
        self._cache_result(display_path, result, tokens_used)

        end_processing = time.time()
        self.processing_times.append(end_processing - start_processing)

    def _analyze_image_with_retry(self, display_path: str) -> Tuple[dict, int]:
        """带重试机制的图片分析方法"""
        max_retries: int = 3
        retry_delay: int = 2  # 初始重试延迟（秒）

        # 检查文件是否存在
        if not os.path.isfile(display_path):
            raise FileNotFoundError(f"文件不存在: {display_path}")

        # 检查文件是否可读
        if not os.access(display_path, os.R_OK):
            raise PermissionError(f"文件无法读取: {display_path}")

        for attempt in range(max_retries):
            try:
                result, tokens_used = self._analyze_image(display_path)
                return result, tokens_used

            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    self.progress_updated.emit(
                        f"网络连接失败，{retry_delay}秒后重试（{attempt + 1}/{max_retries}）", 0)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    raise ConnectionError("网络连接持续失败，请检查网络")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    self.progress_updated.emit(
                        f"请求超时，{retry_delay}秒后重试（{attempt + 1}/{max_retries}）", 0)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise TimeoutError("API请求多次超时")

            except APIThrottledError as e:
                # 限流错误使用固定延迟（由API返回）
                if attempt < max_retries - 1:
                    self.progress_updated.emit(
                        f"请求被限流，等待 {e.retry_after} 秒后重试", 0)
                    time.sleep(e.retry_after)
                else:
                    raise

            except requests.exceptions.RequestException as re:
                # 其他网络错误或5xx错误，重试
                if attempt < max_retries - 1:
                    self.progress_updated.emit(
                        f"请求失败，正在重试 ({attempt + 1}/{max_retries}): {str(re)}", 0)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    raise

    def _analyze_image(self, img_path):
        """调用API分析图片，获取结构化结果"""
        try:
            # 打开图片并压缩
            with Image.open(img_path) as img:
                # 设定最大边长（如640px）
                max_size = MAX_IMAGE_SIZE  # 使用常量
                width, height = img.size
                # 计算压缩比例
                if width > max_size or height > max_size:
                    ratio = max_size / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    # 压缩图片（保持比例）
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 处理RGBA转RGB（核心修改部分）
                if img.mode in ('RGBA', 'LA'):
                    # 创建白色背景
                    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                    # 粘贴原图并应用alpha通道
                    background.paste(img, img.split()[-1])
                    img = background
                elif img.mode == 'P':
                    # 处理调色板模式图片
                    img = img.convert('RGB')

                # 保存压缩后的图片到内存
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                image_data = img_byte_arr.getvalue()

                # 检查图片数据是否有效（新增）
                if len(image_data) == 0:
                    raise ValueError(f"图片处理失败，生成空数据: {img_path}")

            # 生成base64编码
            base64_image = base64.b64encode(image_data).decode('utf-8').replace('\n', '').replace('\r', '')
            if not base64_image:
                raise ValueError(f"图片base64编码失败: {img_path}")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            structured_prompt = f"""{self.prompt_text} 
            请严格按照以下要求返回结果：
            1. 必须是完整JSON，包含"风险等级"、"建议"、"分析内容"。
            2. 若满足"2项高权重维度≥7分"，分析内容简要说明（限100字内），减少输出长度。
            """

            # 动态计算max_tokens（提示词越长，预留越少）
            prompt_length = len(self.prompt_text)
            base_tokens = DEFAULT_MAX_TOKENS
            # 提示词每增加500字符，减少100tokens（最低保留300）
            reduce_tokens = max(0, min(int(prompt_length / 500) * 100, base_tokens - 300))
            max_tokens = base_tokens - reduce_tokens

            payload = {
                "model": "qwen-vl-plus",
                "messages": [
                    {"role": "system", "content": structured_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请分析以下图片："},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens  # 使用动态计算的值
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )

            # 检查响应头中的RateLimit信息
            if 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers['X-RateLimit-Remaining'])
                if remaining < 5:  # 如果剩余请求数少于5，减慢请求速度
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 1))
                    time.sleep(reset_time / 1000)  # 等待重置时间的一部分

            if response.status_code == 200:
                result = response.json()

                # 提取token使用量
                tokens_used = result.get("usage", {}).get("total_tokens", 0)

                # 解析模型返回的内容（应该是JSON格式）
                analysis = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                try:
                    structured_result = json.loads(analysis)
                    # 只保留需要的字段，过滤多余内容
                    filtered_result = {
                        "风险等级": structured_result.get("风险等级", "未知"),
                        "建议": structured_result.get("建议", ""),
                        "分析内容": structured_result.get("分析内容", ""),
                        "总评分": structured_result.get("总评分", 0)  # 新增：保留总分
                    }
                    return filtered_result, tokens_used
                except json.JSONDecodeError:
                    # 后备处理不变，但提示更明确的格式错误
                    return {
                        "风险等级": self._extract_risk_level(analysis),
                        "建议": self._extract_suggestion(analysis),
                        "分析内容": f"格式错误：{analysis[:100]}..."  # 截断过长的错误内容
                    }, tokens_used
            else:
                # 处理各种HTTP错误
                if response.status_code == 401:
                    raise Exception("认证失败：API密钥无效", 401)
                elif response.status_code == 403:
                    raise Exception("权限不足：API密钥没有访问该资源的权限", 403)
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    raise APIThrottledError(f"请求频率过高，请等待 {retry_after} 秒", retry_after)
                else:
                    raise Exception(f"API请求失败: HTTP {response.status_code}", response.status_code)

        except APIThrottledError:
            raise  # 特殊错误，由上层处理重试
        except Exception as e:
            # 其他错误，记录并返回错误结果
            error_code = getattr(e, 'args', (None,))[1] if len(getattr(e, 'args', [])) > 1 else 500
            return {
                "分析内容": f"分析出错: {str(e)}",
                "风险等级": "未知",
                "建议": f"分析出错: {str(e)}"
            }, 0

    def _extract_total_score(self, analysis_text: str) -> int:
        """从分析文本中提取总分数"""
        patterns = [
            r'总分[：:]\s*(\d+)',
            r'总评分[：:]\s*(\d+)',
            r'总计[：:]\s*(\d+)',
            r'合计[：:]\s*(\d+)',
            r'总分数[：:]\s*(\d+)',
            r'总分为\s*(\d+)',
            r'总分是\s*(\d+)',
            r'总共\s*(\d+)\s*分',
            r'共计\s*(\d+)\s*分',
            r'(\d+)\s*分\s*(?:总分|总计|合计)',
            r'总分.*?(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 0 <= score <= 120:  # 分数合理性校验
                        return score
                except ValueError:
                    continue
        return 0  # 未提取到分数时返回0

    def _extract_risk_level(self, analysis_text: str) -> str:
        """从分析文本中提取风险等级"""
        # 优先匹配明确的风险等级描述
        patterns = [
            r'风险等级[：:]\s*([高高低]风险)',
            r'风险级别[：:]\s*([高高低]风险)',
            r'属于\s*([高高低]风险)',
            r'判定为\s*([高高低]风险)',
            r'([高高低]风险)'  # 最后匹配单独出现的风险等级
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                return match.group(1)

        # 若未提取到明确风险等级，根据总分推断（仅作为备选方案）
        score = self._extract_total_score(analysis_text)
        if score >= 46:
            return "高风险"
        elif score > 0:
            return "低风险"
        return "未知"

    def _extract_suggestion(self, analysis_text: str) -> str:
        """从分析文本中提取建议（优化格式）"""
        # 清洗文本，先去除所有特殊符号
        cleaned_text = re.sub(r'[^\w\s]', '', analysis_text)

        # 精准匹配目标建议
        if '建议删除' in cleaned_text:
            return "建议删除"
        elif '无需处理' in cleaned_text:
            return "无需处理"

        # 若未匹配到，根据风险等级推断
        risk_level = self._extract_risk_level(analysis_text)
        if risk_level == "高风险":
            return "建议删除"
        else:
            return "无需处理"

    def stop(self):
        """停止线程运行"""
        self.is_running = False


class DetailDialog(QDialog):
    """分析详情对话框（优化排版）"""

    def __init__(self, analysis_content: str, image_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("分析详情")
        self.setMinimumSize(800, 650)  # 增大窗口尺寸
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.image_path = image_path
        self.init_ui(analysis_content)

    def init_ui(self, analysis_content):
        main_widget = QWidget()
        main_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            }
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(main_widget)

        content_layout = QVBoxLayout(main_widget)
        content_layout.setContentsMargins(32, 32, 32, 32)
        content_layout.setSpacing(24)

        # 标题区域
        header_layout = QHBoxLayout()

        title_label = QLabel("分析详细内容")
        title_label.setStyleSheet("""
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 22px;
            font-weight: 600;
            color: #1e293b;
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn = QPushButton("×")
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #6b7280;
                font-size: 20px;
                font-weight: bold;
                padding: 5px 10px;
                border: none;
                border-radius: 50%;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

        content_layout.addLayout(header_layout)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("border-color: #e5e7eb;")
        content_layout.addWidget(line)

        # 图片预览区域
        image_preview = QLabel()
        image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_preview.setStyleSheet("""
            QLabel {
                background-color: #e0e7ff;
                border: 1px dashed #93c5fd;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        image_preview.setFixedSize(600, 400)  # 固定大小
        content_layout.addWidget(image_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # 加载图片预览
        try:
            if os.path.exists(self.image_path):  # 使用传入的图片路径
                pixmap = QPixmap(self.image_path)
                scaled_pixmap = pixmap.scaled(
                    600, 400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                image_preview.setPixmap(scaled_pixmap)
            else:
                image_preview.setText("图片文件不存在")
        except Exception as e:
            image_preview.setText(f"图片加载失败: {e}")

        # 详情文本区域
        detail_text = QTextEdit()
        detail_text.setReadOnly(True)
        detail_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 15px;
                color: #4b5563;
                line-height: 1.7;
                min-height: 150px;
            }
        """)
        detail_text.setText(analysis_content)
        content_layout.addWidget(detail_text)
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f3f4f6;
                color: #4b5563;
                border-radius: 8px;
                padding: 10px 24px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 15px;
                font-weight: 500;
                border: none;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
            }
        """)
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        content_layout.addLayout(button_layout)


# 在ImageAnalyzerApp类的样式常量定义中添加滚动条样式
class ImageAnalyzerApp(QMainWindow):
    """图像分析应用主窗口"""
    # 样式常量定义（增加滚动条样式）
    TABLE_STYLE = """
        QTableWidget {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            gridline-color: #e5e7eb;
        }
        QTableWidget::item {
            padding: 10px;
            border-radius: 4px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            word-wrap: break-word;
        }
        QTableWidget::item:selected {
            background-color: #dbeafe;
            color: #1e40af;
            outline: none;
            border: none;
        }
        QHeaderView::section {
            background-color: #f9fafb;
            padding: 8px;
            border: 1px solid #e5e7eb;
            font-weight: bold;
        }
        QTableWidget::indicator {
            width: 20px;
            height: 20px;
            border-radius: 10px;
            border: 2px solid #d1d5db;
            background-color: white;
        }
        QTableWidget::indicator:checked {
            background-color: #3b82f6;
            border-color: #3b82f6;
            image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEwIDE2LjE3TDE1LjU5IDEwLjU4TDE2LjY5IDEwLjU4TDE2Ljk5IDExLjk5TDEwIDIwTDMgMTNMMDQuNDEgMTEuNDFMMTAgMTYuMTd6IiBmaWxsPSJ3aGl0ZSIvPjxzdmc+);
        }
        /* 新增风险等级和建议的样式 */
        .high-risk {
            color: #dc2626; /* 红色 */
            font-weight: bold;
        }
        .low-risk {
            color: #059669; /* 绿色 */
            font-weight: bold;
        }
        .high-risk-suggestion {
            color: #dc2626; /* 红色 */
            font-weight: bold;
        }
        .low-risk-suggestion {
            color: #059669; /* 绿色 */
            font-weight: bold;
        }

        /* 优化滚动条样式 */
        QScrollBar:vertical {
            border: none;
            background: #f9fafb;
            width: 8px;
            margin: 0px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #d1d5db;
            min-height: 20px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover {
            background: #9ca3af;
        }
        QScrollBar::add-line:vertical {
            background: none;
            height: 0px;
        }
        QScrollBar::sub-line:vertical {
            background: none;
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
    """
    CHECKBOX_STYLE = """
        QCheckBox {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 15px;
            color: #4b5563;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 3px;
            border: 2px solid #d1d5db;
            background-color: white;
        }
        QCheckBox::indicator:checked {
            background-color: #3b82f6;
            border-color: #3b82f6;
            image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEwIDE2LjE3TDE1LjU5IDEwLjU4TDE2LjY5IDEwLjU4TDE2Ljk5IDExLjk5TDEwIDIwTDMgMTNMMDQuNDEgMTEuNDFMMTAgMTYuMTd6IiBmaWxsPSJ3aGl0ZSIvPjxzdmc+);
        }
    """
    # 风险等级常量定义
    RISK_LEVEL_HIGH = "高风险"
    RISK_LEVEL_LOW = "低风险"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片分析工具")
        self.setWindowIcon(QIcon())
        self.setMinimumSize(1050, 750)
        self.resize(1200, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 窗口阴影效果
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 4)
        self.shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(self.shadow)

        # 初始化变量
        self.images = []  # 存储图片路径 (display_path, original_path)
        self.analysis_results = {}  # 存储分析结果
        self.api_key = (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("TRO_API_KEY") or "").strip()
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.prompt_text = self._load_prompt()
        self.folder_path = ""
        self.analysis_thread = None
        self.timer = QTimer(self)
        self.dragging = False
        self.drag_position = None
        self.analysis_cache = {}  # 缓存格式：{图片路径: (分析结果, tokens_used, timestamp)}
        self.current_preview_window = None  # 单例：当前唯一的预览窗口
        self.cache_mutex = QMutex()

        # 最近路径记录
        self.recent_image_paths = []
        self.recent_folder_paths = []
        self.config_file = "image_analyzer_config.json"
        self._cleanup_sensitive_files()
        self.load_config()
        self.load_cache()  # 加载缓存

        # 选择状态管理
        self.selected_images = set()
        self.all_checked = False
        self.upload_status_label = None
        self.init_ui()

        # 信号连接
        self.all_results_table.itemChanged.connect(self._on_item_changed)
        self.high_risk_table.itemChanged.connect(self._on_item_changed)
        self.low_risk_table.itemChanged.connect(self._on_item_changed)
        self.result_tabs.currentChanged.connect(self._on_tab_changed)

        # 窗口拖动事件
        self.header_widget.installEventFilter(self)

        # 不再用定时器清理，改为退出时统一清理
        self.cache_clean_timer = None

    def load_config(self):
        """加载保存的配置文件（已禁用落盘以隐藏隐私信息）"""
        return

    def save_config(self):
        """保存配置到文件（已禁用落盘以隐藏隐私信息）"""
        return

    def save_cache(self):
        """保存缓存到文件（已禁用落盘以隐藏隐私信息）"""
        return

    def load_cache(self):
        """加载缓存（已禁用落盘以隐藏隐私信息）"""
        return

    def _cleanup_sensitive_files(self):
        """清理可能包含隐私信息的落盘文件（静默）"""
        for fp in (self.config_file, "analysis_cache.json", "processed_images.json"):
            try:
                if fp and os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass

    def _prompt_api_key(self) -> str:
        """运行时输入API Key（不会保存到本地）"""
        try:
            key, ok = QInputDialog.getText(
                self,
                "输入API Key",
                "请输入API Key（不会保存到本地）：",
                QLineEdit.EchoMode.Password,
            )
            if ok:
                return (key or "").strip()
        except Exception:
            pass
        return ""

    def clean_expired_cache(self) -> None:
        self.cache_mutex.lock()
        try:
            current_time = time.time()
            expired_keys = []

            # 收集过期键
            for path, value in list(self.analysis_cache.items()):  # 使用list创建副本
                if isinstance(value, (list, tuple)) and len(value) == 3:
                    _, _, timestamp = value
                    if current_time - timestamp >= CACHE_EXPIRY_DAYS * 86400:
                        expired_keys.append(path)
                else:
                    expired_keys.append(path)

            # 删除过期键
            for key in expired_keys:
                self.analysis_cache.pop(key, None)

            if expired_keys:
                try:
                    self.save_cache()  # save_cache 现在是无锁版本，这里调用前已持有锁
                    self.upload_status_label.setText(f"已清理 {len(expired_keys)} 条过期缓存")
                except Exception as e:
                    self.upload_status_label.setText(f"清理缓存后写入失败: {str(e)}")
        finally:
            self.cache_mutex.unlock()

    def init_ui(self):
        """初始化用户界面（主入口）"""
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: white; border-radius: 16px;")
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 头部区域
        self._init_header(main_layout)

        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(24)
        main_layout.addWidget(content_widget)

        # 工具按钮区
        self._init_toolbar(content_layout)

        # 选择控件区
        self._init_selection_controls(content_layout)

        # 上传状态区
        self.upload_status_label = QLabel()
        self.upload_status_label.setStyleSheet("""
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 15px;
            color: #10b981;
        """)
        content_layout.addWidget(self.upload_status_label)

        # 结果展示区
        self._init_result_area(content_layout)

        # 底部状态栏
        self._init_status_bar(main_layout)

    def _init_header(self, parent_layout):
        """初始化头部区域"""
        self.header_widget = QWidget()
        self.header_widget.setStyleSheet("""
            QWidget {
                background-color: #f9fafb;
                border-radius: 16px 16px 0 0;
            }
        """)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(24, 16, 24, 16)
        header_layout.setSpacing(16)
        parent_layout.addWidget(self.header_widget)

        # 标题居中
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("图片分析工具")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 22px;
            font-weight: 600;
            color: #1e293b;
        """)
        title_layout.addWidget(title_label)

        header_layout.addWidget(title_widget, 1)
        header_layout.addStretch()

        # 窗口控制按钮
        minimize_btn = QPushButton("−")
        minimize_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #6b7280;
                font-size: 16px;
                padding: 5px 10px;
                border: none;
                border-radius: 50%;
            }
            QPushButton:hover {
                background-color: #f59e0b;
                color: white;
            }
        """)
        minimize_btn.clicked.connect(self.showMinimized)
        header_layout.addWidget(minimize_btn)

        close_btn = QPushButton("×")
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #6b7280;
                font-size: 16px;
                padding: 5px 10px;
                border: none;
                border-radius: 50%;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: white;
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

    def _init_toolbar(self, parent_layout):
        """初始化工具按钮区域"""
        tools_widget = QWidget()
        tools_layout = QHBoxLayout(tools_widget)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(12)
        parent_layout.addWidget(tools_widget)

        # 上传图片按钮
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.setMinimumHeight(44)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                border: none;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
                color: #d1d5db;
            }
        """)
        self.upload_btn.clicked.connect(self._upload_images)
        tools_layout.addWidget(self.upload_btn)

        # 上传文件夹按钮
        self.upload_folder_btn = QPushButton("上传文件夹")
        self.upload_folder_btn.setMinimumHeight(44)
        self.upload_folder_btn.setStyleSheet(self.upload_btn.styleSheet())
        self.upload_folder_btn.clicked.connect(self._upload_folder)
        tools_layout.addWidget(self.upload_folder_btn)

        # 路径输入框和导入按钮
        path_group = QWidget()
        path_group.setStyleSheet("""
            QWidget {
                background-color: #f9fafb;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        path_layout = QHBoxLayout(path_group)
        path_layout.setSpacing(8)
        path_layout.setContentsMargins(0, 0, 0, 0)

        self.path_input = QLineEdit()
        self.path_input.setMinimumHeight(32)
        self.path_input.setStyleSheet("""
            QLineEdit {
                background-color: white;
                color: #1e293b;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #93c5fd;
                outline: none;
            }
        """)
        path_layout.addWidget(self.path_input)

        self.upload_path_btn = QPushButton("导入")
        self.upload_path_btn.setMinimumHeight(32)
        self.upload_path_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 14px;
                border: none;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.upload_path_btn.clicked.connect(self._upload_path_folder)
        path_layout.addWidget(self.upload_path_btn)

        tools_layout.addWidget(path_group, 1)

        # 开始分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setMinimumHeight(44)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                border: none;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
                color: #d1d5db;
            }
        """)
        self.analyze_btn.clicked.connect(self._start_analysis)
        tools_layout.addWidget(self.analyze_btn)

        # 清空按钮
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setMinimumHeight(44)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                border: none;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
                color: #d1d5db;
            }
        """)
        self.clear_btn.clicked.connect(self._clear_all)
        tools_layout.addWidget(self.clear_btn)

        # 导出高风险按钮
        self.export_high_risk_btn = QPushButton("导出高风险图片")
        self.export_high_risk_btn.setMinimumHeight(44)
        self.export_high_risk_btn.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                border: none;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #d97706;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
                color: #d1d5db;
            }
        """)
        self.export_high_risk_btn.clicked.connect(self._export_high_risk_images)
        self.export_high_risk_btn.setEnabled(False)  # 初始禁用
        tools_layout.addWidget(self.export_high_risk_btn)

    def _init_selection_controls(self, parent_layout):
        """初始化选择控件（全选、删除按钮）"""
        selection_widget = QWidget()
        selection_layout = QHBoxLayout(selection_widget)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(16)
        parent_layout.addWidget(selection_widget)

        # 全选复选框
        self.select_all_checkbox = QCheckBox("全选")
        self.select_all_checkbox.setStyleSheet(self.CHECKBOX_STYLE)
        self.select_all_checkbox.stateChanged.connect(self._toggle_select_all)
        selection_layout.addWidget(self.select_all_checkbox)

        # 删除选中按钮
        self.delete_selected_btn = QPushButton("删除选中")
        self.delete_selected_btn.setMinimumHeight(40)
        self.delete_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 15px;
                font-weight: 500;
                border: none;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
                color: #d1d5db;
            }
        """)
        self.delete_selected_btn.clicked.connect(self._delete_selected)
        self.delete_selected_btn.setEnabled(False)
        selection_layout.addWidget(self.delete_selected_btn)

        selection_layout.addStretch()

    def _init_result_area(self, parent_layout):
        """初始化结果展示区域（选项卡+表格）"""
        results_group = QGroupBox("")
        results_group.setStyleSheet("""
            QGroupBox {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 18px;
                font-weight: 600;
                color: #1e293b;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin: 0;
                padding: 0;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top center;
                padding: 0 10px;
                margin-top: -5px;
                background-color: white;
            }
        """)
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(16, 10, 16, 16)
        results_layout.setSpacing(16)
        parent_layout.addWidget(results_group)

        # 结果选项卡
        self.result_tabs = QTabWidget()
        self.result_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 5px;
                background-color: white;
            }
            QTabBar::tab {
                font-size: 14px;
                font-weight: 500;
                color: #4b5563;
                padding: 8px 16px;
                min-width: 70px;
                height: 18px;
                border: 1px solid #e5e7eb;
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                margin-right: 2px;
                background-color: #f9fafb;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #1e40af;
                border-color: #93c5fd;
                border-bottom: 1px solid white;
            }
        """)
        results_layout.addWidget(self.result_tabs)

        # 创建三个表格（统一通过工厂方法创建）
        self.all_results_table = self._create_result_table()
        self.high_risk_table = self._create_result_table()
        self.low_risk_table = self._create_result_table()

        # 为每个表格创建带表头的容器
        self.result_tabs.addTab(self._create_table_container(self.all_results_table), "所有结果")
        self.result_tabs.addTab(self._create_table_container(self.high_risk_table), "高风险")
        self.result_tabs.addTab(self._create_table_container(self.low_risk_table), "低风险")

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #f3f4f6;
                border-radius: 6px;
                height: 8px;
                text-align: center;
                font-size: 12px;
                color: #4b5563;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 6px;
            }
        """)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        results_layout.addWidget(self.progress_bar)

    def _init_status_bar(self, parent_layout):
        """初始化底部状态栏（上市公司风格 UI）"""

        # 创建底部状态栏容器
        status_widget = QWidget()
        status_widget.setStyleSheet("""
            QWidget {
                background-color: #f4f6f8;  /* 稳重灰 */
                border-bottom-left-radius: 16px;
                border-bottom-right-radius: 16px;
            }
        """)
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(24, 12, 24, 12)
        status_layout.setSpacing(0)

        # 左侧 spacer（使版权文字居中）
        status_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # 版权声明标签
        copyright_label = QLabel(
            "版权所有\n"
            "Copyright © 2025. All Rights Reserved."
        )
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont("Microsoft YaHei", 9)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)  # 抗锯齿更平滑
        copyright_label.setFont(font)
        copyright_label.setStyleSheet("""
            color: #999999;
        """)

        # 加入状态栏
        status_layout.addWidget(copyright_label)

        # 右侧 spacer
        status_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # 加入主布局
        parent_layout.addWidget(status_widget)

    def _create_result_table(self) -> QTableWidget:
        """表格工厂方法：统一创建表格实例（添加预览列）"""
        table = QTableWidget()
        table.setColumnCount(6)  # 增加一列用于预览
        table.setHorizontalHeaderLabels(["选择", "预览", "图片名称", "风险等级", "建议", "详情"])

        # 列宽设置
        table.setColumnWidth(0, 60)
        table.setColumnWidth(1, 80)  # 预览列宽度
        table.setColumnWidth(2, 100)
        table.setColumnWidth(3, 120)
        table.setColumnWidth(4, 100)

        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)

        # 行为设置
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(self.TABLE_STYLE)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._show_detail_dialog)

        # 为表格添加垂直滚动条样式（显式启用）
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        return table

    # ✅ 修改 _create_table_container 方法
    def _create_table_container(self, table):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ✅ 每次新建 QLabel，避免复用
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(0)

        select_header = QLabel("选择")
        select_header.setFixedWidth(60)
        select_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_header = QLabel("预览")
        preview_header.setFixedWidth(80)
        preview_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        path_header = QLabel("图片名称（点击可复制）")
        path_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        risk_header = QLabel("风险等级")
        risk_header.setFixedWidth(120)
        risk_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        suggestion_header = QLabel("处理建议")
        suggestion_header.setFixedWidth(100)
        suggestion_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        detail_header = QLabel("详情")
        detail_header.setFixedWidth(80)
        detail_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_style = """
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: #6b7280;
            padding: 8px 0;
            background-color: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
        """
        for label in [select_header, preview_header, path_header, risk_header, suggestion_header, detail_header]:
            label.setStyleSheet(header_style)
            header_layout.addWidget(label)

        layout.addLayout(header_layout)
        layout.addWidget(table)

        return container

    def _load_prompt(self):
        """内置提示词，不再依赖外部文件"""
        PROMPT_TEXT = r"""
{
  "角色": "你是跨境电商知识产权风险控制专家，专注于识别美国市场中商品图片是否存在 TRO（临时限制令）诉讼风险，需严格按照分析维度，基于图像内容、视觉特征与数据库比对结果进行量化评分和风险判断，禁止输出默认值或模板化内容。",

  "任务": "根据以下9个维度，从商标/IP/视觉等角度判断商品图片在美国是否易被品牌方发起 TRO 诉讼，需输出“总评分（0~65）”“风险等级（高/低）”“处理建议”和“风险说明”，并满足输出规则。禁止跳过分析或输出固定值（如45分）。",

  "分析维度": {
    "知名IP形象": [
      "是否含知名影视、动画、游戏、音乐、漫画等 IP 形象（角色、场景、文字、字体、视觉元素等），须通过 IP 权利数据库或著作权图像库进行匹配确认；包括但不限于 Disney、Marvel、Pixar、US Copyright 登记项目等；",
      "支持模糊匹配新兴IP关键词池（例如：《Five Nights at Freddy's》、《Wednesday》、《Squid Game》等），以提升识别准确率。"
    ],
    "商标隐含性": [
      "是否含注册商标词语、商标图案、字体特征，基于 USPTO 和 WIPO 注册信息库判断；",
      "包含知名品牌的核心商业标语、口号、标识等文字元素（如 ‘LIFE IS GOOD’、‘I BELIEVE IN THE GREAT PUMPKIN’），因其商标或权利属性应纳入严格识别范围。"
    ],
    "视觉混淆风险": [
      "图案是否与在案商标或LOGO构成相似性，评估消费者混淆概率，参考TTABVUE判例；",
      "特别关注图案布局、符号组合是否可能构成外观设计专利侵权。"
    ],
    "地域法律风险": [
      "图像是否触发美国法律限制（如宗教/军事标识、机构用语），包括对特定组织标识、名人肖像权、机构专有标志的保护；",
      "支持结合 EUIPO 等国际数据补充说明。"
    ],
    "类目敏感性": [
      "该类商品在平台是否易受投诉（如服饰类IP周边高发区），结合品牌维权记录及投诉案例。"
    ],
    "历史相似度": [
      "图像与历史TRO/诉讼图案的相似度，参考 Markify、Google Patents 图形比对及 PACER 案例库；",
      "如匹配已知案例，应在对应维度加分并注明‘参考历史案例’。"
    ],
    "平台识别敏感度": [
      "图像是否容易被 Amazon Brand Registry、eBay VeRO、Facebook AI 等平台识别并封禁；",
      "支持调用品牌黑名单数据库与平台判例，如某图已被亚马逊封禁，需加大该维度评分权重。"
    ],
    "商品用途偏移（可选）": [
      "图像是否存在误导性用途，如军事标志、宗教符号等被不当用于玩具、服饰等非授权类目，构成伪装使用型侵权风险。"
    ],
    "虚假宣传及其他违规": [
      "图像或文字内容是否存在暗示与知名品牌、组织、名人存在虚假关联的风险，可能构成商业信誉侵害或虚假宣传。"
    ]
  },

  "评分说明": {
    "评分范围": "每维度0~10分，必须基于实际图像内容判断；如识别出Disney角色，‘知名IP’维度得分≥8，否则≤3；禁止随意或平均赋分。",

    "评分细化指导": [
      "请避免仅使用固定分值（如3、5、7、9），根据证据和侵权程度细分评分，评分可带小数（如7.5、6.2）以反映具体风险差异。",
      "在评分时应考虑侵权元素的明显程度、数量、商业用途及历史判例影响，综合量化风险。",
      "风险说明中应详细解释评分依据，明确指出为何赋予该具体分数，避免空洞描述。"
    ],

    "权重规则": "高权重（前5个维度）直接计入总分，中权重（后3个维度）×0.5后计入总分，‘商品用途偏移’和‘虚假宣传及其他违规’维度为可选，视具体业务决定是否纳入总分。总分上限为65分（不计入可选维度时为55分）。",

    "缺失信息处理": "维度信息模糊时，仅可在以下范围赋值：高权重维度=5~7分，中权重维度=3~5分，并在‘风险说明’中说明赋值理由。",

    "图像文字规则": "图中若包含文字，需OCR识别（优先英文）；如含‘I BELIEVE IN THE GREAT PUMPKIN’，则著作权/商标维度需提高评分。",

    "历史案例规则": "如图像/文字匹配已知TRO案例（PACER数据库），对应维度加2分，最多不超过10分，并注明‘参考历史案例’。",

    "提前终止规则": "若2个高权重维度得分均≥7分，可触发‘提前终止’，但其他维度仍需评分并按公式计算总分；提前终止发生时，风险说明中需详细标明触发维度及分数。",

    "总分计算要求": "严格遵守公式：高权重维度总和 + 中权重维度总和×0.5；若纳入可选维度，需另行说明计算方法，禁止简化计算或直接写出45分。"
  },

  "执行要求": [
    "必须先识别图像的核心要素（IP角色、标志、图形、文字等），再依次对各维度进行评分。",
    "风险等级划分：总评分≥46分为‘高风险’，＜46分为‘低风险’，不得模糊判定或错配风险等级。",
    "输出格式必须完整包含以下字段：‘总评分’（整数0~65）、‘风险等级’、‘处理建议’、‘风险说明’（至少2个关键维度详细说明），不得缺失或用描述性语言替代结构字段。"
  ],

  "输出格式": {
    "总评分": "例如：47.5（因知名IP=9.2、视觉混淆=8.3触发提前终止）",
    "风险等级": "高风险 / 低风险（必须与评分匹配）",
    "处理建议": "建议删除（高风险） / 无需处理（低风险）",
    "风险说明": [
      "高风险示例：知名IP=9.2分（含Disney角色Mickey Mouse），视觉混淆=8.3分（图形高度相似已注册LOGO），历史相似度=6.5，类目敏感=7.1，总评分=47.5，建议删除。",
      "低风险示例：知名IP=2.4分（无显著IP元素），商标隐含性=3.7分（无商标文字），视觉混淆=4.1，总评分=28.0，图像为通用设计，无侵权特征。"
    ]
  },

  "错误禁止": [
    "禁止所有维度赋7分（如5个高权重7分 + 3个中权重7分，总分=54，应为异常值）",
    "禁止输出固定总评分45分（重复出现视为无效结果）",
    "禁止评分逻辑与说明矛盾（如评分为3但说明中描述为‘明显存在Disney元素’）",
    "禁止随意或机械赋分，必须体现评分细致差异"
  ]
}

        """
        return PROMPT_TEXT

    def _upload_images(self):
        """上传多张图片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.gif *.bmp)"
        )

        if file_paths:
            try:
                self.images.extend([(f, f) for f in file_paths])
                self.folder_path = ""
                self.path_input.setText("")

                # 更新最近路径
                for path in file_paths:
                    if path not in self.recent_image_paths:
                        self.recent_image_paths.insert(0, path)
                        if len(self.recent_image_paths) > 10:
                            self.recent_image_paths.pop()
                self.save_config()

                upload_count = len(file_paths)
                self.upload_status_label.setText(f"已成功上传 {upload_count} 张图片")
            except Exception as e:
                self.upload_status_label.setText(f"上传图片失败: {str(e)}")
                self.upload_status_label.setStyleSheet("color: #ef4444;")

    def _upload_folder(self):
        """上传图片文件夹"""
        self.folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")

        if self.folder_path:
            try:
                self.path_input.setText(self.folder_path)
                self._load_folder_images()

                # 更新最近路径
                if self.folder_path not in self.recent_folder_paths:
                    self.recent_folder_paths.insert(0, self.folder_path)
                    if len(self.recent_folder_paths) > 10:
                        self.recent_folder_paths.pop()
                self.save_config()

                image_count = len(self.images)
                self.upload_status_label.setText(f"已成功从文件夹加载 {image_count} 张图片")
            except Exception as e:
                self.upload_status_label.setText(f"上传文件夹失败: {str(e)}")
                self.upload_status_label.setStyleSheet("color: #ef4444;")

    def _upload_path_folder(self):
        """通过路径输入上传文件夹"""
        folder_path = self.path_input.text()
        if os.path.isdir(folder_path):
            try:
                self.folder_path = folder_path
                self._load_folder_images()

                # 更新最近路径
                if self.folder_path not in self.recent_folder_paths:
                    self.recent_folder_paths.insert(0, self.folder_path)
                    if len(self.recent_folder_paths) > 10:
                        self.recent_folder_paths.pop()
                self.save_config()

                image_count = len(self.images)
                self.upload_status_label.setText(f"已成功从文件夹加载 {image_count} 张图片")
            except Exception as e:
                self.upload_status_label.setText(f"上传文件夹失败: {str(e)}")
                self.upload_status_label.setStyleSheet("color: #ef4444;")
        else:
            self.upload_status_label.setText("请输入有效的文件夹路径")
            self.upload_status_label.setStyleSheet("color: #ef4444;")

    def _load_folder_images(self):
        """从文件夹加载图片"""
        try:
            image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
            self.images = []
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        full_path = os.path.join(root, file)
                        self.images.append((full_path, full_path))
        except Exception as e:
            self.upload_status_label.setText(f"加载文件夹图片时出错: {str(e)}")
            self.upload_status_label.setStyleSheet("color: #ef4444;")

    def _start_analysis(self):
        """开始分析图片，分批处理，每批最多BATCH_SIZE张，并支持断点续传"""
        if not self.images:
            self.upload_status_label.setText("请先上传图片")
            self.upload_status_label.setStyleSheet("color: #ef4444;")
            return

        # API Key（不落盘；可从环境变量 DASHSCOPE_API_KEY / TRO_API_KEY 读取，或运行时输入）
        if not self.api_key:
            self.api_key = self._prompt_api_key()
        if not self.api_key:
            self.upload_status_label.setText("请先设置API Key（不会保存到本地）")
            self.upload_status_label.setStyleSheet("color: #ef4444;")
            return


        # 已处理记录（已禁用落盘以隐藏本地路径）
        processed_images: Set[str] = set()

        # 过滤已处理图片
        display_paths = [
            img[0] for img in self.images
            if img[0] not in processed_images
        ]

        total_images_to_process = len(display_paths)

        if not display_paths and self.images:
            self.upload_status_label.setText("所有图片已分析，无需重复分析。")
            self.upload_status_label.setStyleSheet("color: #10b981;")
            self._set_buttons_enabled(True)
            return
        elif not display_paths and not self.images:
            self.upload_status_label.setText("请先上传图片")
            self.upload_status_label.setStyleSheet("color: #ef4444;")
            return

        # 重置进度条
        self.progress_bar.setValue(0)
        self.upload_status_label.setText(f"准备分析 {total_images_to_process} 张图片...")

        # 停止现有线程
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.analysis_thread.wait(2000)

        # 禁用按钮
        self._set_buttons_enabled(False)

        # 分批处理
        self.current_batch = 0
        self.total_batches = (total_images_to_process + BATCH_SIZE - 1) // BATCH_SIZE
        self.processed_images_in_session = set()  # 本次会话中已处理的图片

        # 处理完成后的回调（已禁用落盘以隐藏本地路径）
        def save_processed_callback():
            return

        self._process_next_batch(display_paths, save_processed_callback)

    def _process_next_batch(self, display_paths, save_processed_callback):
        """处理下一批图片"""
        start_idx = self.current_batch * BATCH_SIZE
        end_idx = min((self.current_batch + 1) * BATCH_SIZE, len(display_paths))

        if start_idx >= len(display_paths):
            # 所有批次处理完成
            self._set_buttons_enabled(True)
            save_processed_callback()  # 调用保存已处理记录的回调
            return

        batch_paths = display_paths[start_idx:end_idx]
        self.upload_status_label.setText(f"正在处理第 {self.current_batch + 1}/{self.total_batches} 批图片")

        # 启动分析线程
        self.analysis_thread = ImageAnalysisThread(
            batch_paths, self.api_key, self.base_url, self.prompt_text, self
        )
        self.analysis_thread.progress_updated.connect(self._update_status)
        self.analysis_thread.result_added.connect(self._add_result)
        self.analysis_thread.batch_completed.connect(self._batch_completed)
        self.analysis_thread.error_occurred.connect(self._handle_error)
        self.analysis_thread.start()

    def _batch_completed(self):
        """批次处理完成回调"""
        # 记录本次批次处理的图片，用于断点续传
        if self.analysis_thread:
            for img_path in self.analysis_thread.images:
                self.processed_images_in_session.add(img_path)

        self.current_batch += 1
        display_paths = [img[0] for img in self.images]
        # 已禁用落盘记录，直接使用本次会话已处理集合
        processed_images: Set[str] = set(self.processed_images_in_session)

        remaining_display_paths = [
            img_path for img_path in display_paths
            if img_path not in processed_images
        ]

        self._process_next_batch(remaining_display_paths, lambda: self._analysis_completed("本次分析完成"))  # 传递回调函数

    def _set_buttons_enabled(self, enabled):
        self.upload_btn.setEnabled(enabled)
        self.upload_folder_btn.setEnabled(enabled)
        self.upload_path_btn.setEnabled(enabled)
        self.analyze_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
        self.select_all_checkbox.setEnabled(enabled)
        self.delete_selected_btn.setEnabled(enabled and len(self.selected_images) > 0)
        if hasattr(self, 'export_high_risk_btn'):
            self.export_high_risk_btn.setEnabled(enabled and self.high_risk_table.rowCount() > 0)

    def _update_status(self, message, progress):
        """更新状态栏和进度条"""
        self.progress_bar.setValue(progress)
        self.upload_status_label.setText(message)

    def _add_result(self, display_path, risk_level, suggestion, analysis_content):
        with QMutexLocker(self.cache_mutex):
            self.analysis_results[display_path] = {
                "分析内容": analysis_content,
                "风险等级": risk_level,
                "建议": suggestion
            }
            self.save_cache()
        # 准备行数据
        file_name = os.path.basename(display_path)
        row_data = [
            display_path,  # 复选框关联的路径
            file_name,  # 显示的文件名
            risk_level,  # 风险等级（使用模型提取的结果）
            suggestion,  # 建议（使用模型提取的结果）
            analysis_content  # 分析内容
        ]

        # 添加到全量表格
        self._add_row_to_table(self.all_results_table, row_data, risk_level)

        # 根据风险等级添加到对应表格
        if risk_level == self.RISK_LEVEL_HIGH:
            self._add_row_to_table(self.high_risk_table, row_data, risk_level)
        elif risk_level == self.RISK_LEVEL_LOW:
            self._add_row_to_table(self.low_risk_table, row_data, risk_level)

        # 更新删除按钮状态
        self.delete_selected_btn.setEnabled(len(self.selected_images) > 0)

        # 更新导出按钮状态
        has_high_risk = self.high_risk_table.rowCount() > 0
        if hasattr(self, 'export_high_risk_btn'):
            self.export_high_risk_btn.setEnabled(has_high_risk)

    def _add_row_to_table(self, table, row_data, risk_level):
        """向指定表格添加一行数据（优化图片预览与交互）"""
        display_path, file_name, risk_level, suggestion, analysis_content = row_data
        row = table.rowCount()
        table.insertRow(row)

        # 复选框列
        checkbox_item = QTableWidgetItem()
        checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        checkbox_item.setCheckState(Qt.CheckState.Checked if self.all_checked else Qt.CheckState.Unchecked)
        checkbox_item.setData(Qt.ItemDataRole.UserRole, display_path)
        table.setItem(row, 0, checkbox_item)

        # 预览列：优化图片显示，固定大小并保持比例，点击可查看原图
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(2, 2, 2, 2)

        preview_label = QLabel()
        preview_label.setFixedSize(60, 60)  # 固定预览框大小
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                background-color: #f9fafb;
            }
        """)

        # 加载并缩放图片
        try:
            if os.path.exists(display_path):
                pixmap = QPixmap(display_path)
                # 保持比例缩放，确保图片完全显示在预览框内
                scaled_pixmap = pixmap.scaled(
                    56, 56,  # 略小于容器大小，留边距
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                preview_label.setPixmap(scaled_pixmap)
            else:
                preview_label.setText("图片不存在")
        except Exception as e:
            preview_label.setText("加载失败")

        # 添加点击事件
        preview_label.mousePressEvent = lambda event, p=display_path: self._show_original_image(p)
        preview_label.setCursor(Qt.CursorShape.PointingHandCursor)

        preview_layout.addWidget(preview_label)
        table.setCellWidget(row, 1, preview_container)

        # 文件名列（点击复制）
        path_item = QTableWidgetItem(file_name)
        path_item.setFlags(path_item.flags() | Qt.ItemFlag.ItemIsSelectable)
        path_item.setToolTip(f"点击复制文件名: {file_name}")
        path_item.setData(Qt.ItemDataRole.UserRole, display_path)
        path_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        path_item.setForeground(QBrush(QColor("#1e293b")))  # ✅ 显式设置字体颜色
        table.setItem(row, 2, path_item)

        # 风险等级列（带颜色）
        risk_item = QTableWidgetItem(risk_level)
        if risk_level == self.RISK_LEVEL_HIGH:
            risk_item.setForeground(QBrush(QColor("#dc2626")))  # 红色
            risk_item.setData(Qt.ItemDataRole.UserRole, "high-risk")
        else:
            risk_item.setForeground(QBrush(QColor("#059669")))  # 绿色
            risk_item.setData(Qt.ItemDataRole.UserRole, "low-risk")
        risk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(row, 3, risk_item)

        # 建议列（直接使用模型返回的建议）
        suggestion_item = QTableWidgetItem(suggestion)
        # 根据风险等级设置建议颜色（保持视觉一致性）
        if risk_level == self.RISK_LEVEL_HIGH:
            suggestion_item.setForeground(QBrush(QColor("#dc2626")))  # 红色
        else:
            suggestion_item.setForeground(QBrush(QColor("#059669")))  # 绿色
        suggestion_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(row, 4, suggestion_item)

        # 详情按钮列
        detail_btn = QPushButton("点击查看")
        detail_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e7ff;
                color: #3b82f6;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c7d2fe;
            }
        """)
        detail_btn.clicked.connect(lambda: self._show_detail_dialog(display_path))
        table.setCellWidget(row, 5, detail_btn)

        # 设置整行背景色
        bg_color = QColor("#fecaca") if risk_level == self.RISK_LEVEL_HIGH else QColor("#d1fae5")
        for col in range(6):
            item = table.item(row, col) or QTableWidgetItem()
            if col not in [0, 1, 5]:  # 跳过复选框、预览和按钮列
                item.setBackground(QBrush(bg_color))
            table.setItem(row, col, item)

        # 绑定点击事件（复制文件名）
        table.cellClicked.connect(self._on_cell_clicked)

    def _on_cell_clicked(self, row, column):
        """处理表格单元格点击事件：仅保留复制文件名"""
        sender = self.sender()
        item = sender.item(row, 0)
        if not item:
            return
        display_path = item.data(Qt.ItemDataRole.UserRole)

        # 仅处理“文件名”列点击复制
        if column == 2:  # 文件名列
            name_item = sender.item(row, column)
            if name_item:
                file_name = name_item.text()
                clipboard = QApplication.clipboard()
                clipboard.setText(file_name)
                self.upload_status_label.setText(f"已复制文件名: {file_name}")
                self.upload_status_label.setStyleSheet("color: #10b981;")

    def _show_original_image(self, image_path):
        if not image_path or not os.path.exists(image_path):
            return

        # 关闭并清理旧窗口
        try:
            if self.current_preview_window is not None:
                self.current_preview_window.close()
                self.current_preview_window.deleteLater()
        except Exception:
            pass
        self.current_preview_window = None

        dialog = QDialog(self)
        dialog.setWindowTitle("原图预览")
        dialog.setModal(False)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setFixedSize(800, 600)

        layout = QVBoxLayout(dialog)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setScaledContents(False)

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                780, 500,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            img_label.setPixmap(scaled_pixmap)
        else:
            img_label.setText("图片加载失败")

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        close_btn.setStyleSheet("""
            QPushButton{background:#3b82f6;color:white;border-radius:6px;padding:8px 16px;font-size:14px;}
        """)
        layout.addWidget(img_label)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.current_preview_window = dialog
        dialog.show()

    def _toggle_select_all(self, state):
        """全选/取消全选当前标签页的内容"""
        self.all_checked = state == Qt.CheckState.Checked.value
        current_table = self._get_current_table(self.result_tabs.currentIndex())

        if not current_table:
            return

        # 阻止信号递归
        current_table.blockSignals(True)

        # 更新当前表格的选择状态
        selected_in_tab = set()
        for i in range(current_table.rowCount()):
            checkbox_item = current_table.item(i, 0)
            if checkbox_item and checkbox_item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                checkbox_item.setCheckState(Qt.CheckState.Checked if self.all_checked else Qt.CheckState.Unchecked)
                img_path = checkbox_item.data(Qt.ItemDataRole.UserRole)
                if self.all_checked and img_path:
                    selected_in_tab.add(img_path)

        current_table.blockSignals(False)
        self.selected_images = selected_in_tab
        self.delete_selected_btn.setEnabled(len(self.selected_images) > 0)

    def _get_current_table(self, tab_index):
        """根据标签页索引获取对应的表格"""
        if tab_index == 0:
            return self.all_results_table
        elif tab_index == 1:
            return self.high_risk_table
        elif tab_index == 2:
            return self.low_risk_table
        return None

    def _on_item_changed(self, item):
        """处理表格项变化（主要是复选框状态）"""
        if item.column() != 0:  # 只处理复选框列
            return

        img_path = item.data(Qt.ItemDataRole.UserRole)
        if not img_path:
            return

        # 更新选中集合
        if item.checkState() == Qt.CheckState.Checked:
            self.selected_images.add(img_path)
        else:
            self.selected_images.discard(img_path)

        # 更新全选状态
        self._update_all_tabs_check_state()
        self.delete_selected_btn.setEnabled(len(self.selected_images) > 0)

    def _on_tab_changed(self, index):
        """标签页切换时更新选择状态"""
        self.selected_images = set()
        self.all_checked = False
        self.select_all_checkbox.setChecked(False)

        # 检查当前标签页是否有已选中项
        current_table = self._get_current_table(index)
        if current_table:
            for i in range(current_table.rowCount()):
                item = current_table.item(i, 0)
                if item and item.checkState() == Qt.CheckState.Checked:
                    img_path = item.data(Qt.ItemDataRole.UserRole)
                    if img_path:
                        self.selected_images.add(img_path)

        self.delete_selected_btn.setEnabled(len(self.selected_images) > 0)

    def _update_all_tabs_check_state(self):
        """更新所有标签页的全选状态"""
        all_checked = True
        has_items = False

        for table in [self.all_results_table, self.high_risk_table, self.low_risk_table]:
            if table.rowCount() == 0:
                continue
            has_items = True

            for i in range(table.rowCount()):
                item = table.item(i, 0)
                if item and item.checkState() != Qt.CheckState.Checked:
                    all_checked = False
                    break
            if not all_checked:
                break

        # 更新全选复选框
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(all_checked and has_items)
        self.select_all_checkbox.blockSignals(False)
        self.all_checked = all_checked

    def _delete_selected(self):
        """删除选中的图片（实际文件+表格行+分析结果）"""
        if not self.selected_images:
            return

        # 确认对话框
        count = len(self.selected_images)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除选中的 {count} 张图片吗？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # 1. 收集要删除的实际文件路径和保留的图片
        deleted_original_paths = []
        remaining_images = []
        for display_path, original_path in self.images:
            if display_path in self.selected_images:
                deleted_original_paths.append(original_path)
            else:
                remaining_images.append((display_path, original_path))
        self.images = remaining_images

        # 2. 删除实际文件
        self._delete_files(deleted_original_paths)

        # 3. 删除分析结果
        for path in self.selected_images:
            if path in self.analysis_results:
                del self.analysis_results[path]
            # 同时从缓存中删除
            if path in self.analysis_cache:
                del self.analysis_cache[path]
        self.save_cache()  # 保存更新后的缓存

        # 4. 从表格中移除行
        self._remove_selected_rows_from_tables()

        # 5. 重置选择状态
        self.selected_images = set()
        self.all_checked = False
        self.select_all_checkbox.setChecked(False)
        self.delete_selected_btn.setEnabled(False)

        # 6. 显示结果
        self.upload_status_label.setText(f"已删除 {len(deleted_original_paths)} 张图片")

    def _delete_files(self, file_paths):
        """删除指定路径的文件，带错误处理"""
        if not file_paths:
            return

        failed_files = []
        for path in file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    # 更新最近路径记录
                    if path in self.recent_image_paths:
                        self.recent_image_paths.remove(path)
                    folder = os.path.dirname(path)
                    if folder in self.recent_folder_paths:
                        self.recent_folder_paths.remove(folder)
                    self.save_config()
                except Exception as e:
                    failed_files.append((path, str(e)))

        # 提示删除失败的文件
        if failed_files:
            msg = "以下文件删除失败:\n\n"
            for path, err in failed_files:
                msg += f"文件: {path}\n错误: {err}\n\n"
            QMessageBox.warning(self, "删除警告", msg)

    def _remove_selected_rows_from_tables(self):
        """从所有表格中移除选中行"""
        for table in [self.all_results_table, self.high_risk_table, self.low_risk_table]:
            # 倒序删除避免索引问题
            for i in range(table.rowCount() - 1, -1, -1):
                item = table.item(i, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) in self.selected_images:
                    table.removeRow(i)

    def _show_detail_dialog(self, position=None):
        """显示分析详情对话框"""
        # 处理右键菜单触发的情况（点击表格右键）
        if isinstance(position, QPoint):
            sender = self.sender()
            if not sender:
                return
            index = sender.indexAt(position)
            if not index.isValid():
                return
            row = index.row()
            item = sender.item(row, 0)  # 获取复选框列的item
            if not item:
                return
            display_path = item.data(Qt.ItemDataRole.UserRole)  # 从item中获取图片路径
        else:
            # 处理按钮直接触发的情况（点击"详情"按钮）
            display_path = position

        # 获取分析内容和图片路径并显示对话框
        analysis = self.analysis_results.get(display_path, {})
        analysis_content = analysis.get("分析内容", "")
        if analysis_content:
            dialog = DetailDialog(analysis_content, display_path, self)  # 传递display_path作为image_path
            dialog.exec()

    def _analysis_completed(self, total_time_str):
        """分析完成回调"""
        self._set_buttons_enabled(True)

        # 统计结果
        total = self.all_results_table.rowCount()
        high_risk = self.high_risk_table.rowCount()
        low_risk = self.low_risk_table.rowCount()

        self.upload_status_label.setText(
            f"分析完成！共分析 {total} 张图片，其中高风险 {high_risk} 张，低风险 {low_risk} 张。{total_time_str}"
        )

    def _handle_error(self, error_msg, error_code=500):
        """处理错误"""
        self._set_buttons_enabled(True)
        self.upload_status_label.setText(f"分析过程中出现错误: {error_msg}")
        self.upload_status_label.setStyleSheet("color: #ef4444;")
        CustomMessageBox(error_msg, is_success=False, parent=self).exec()

    def _clear_all(self):
        """清空所有图片和结果"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有图片和结果吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 清空数据
            self.images = []
            self.analysis_results = {}
            self.folder_path = ""
            self.path_input.setText("")

            # 清空表格
            self.all_results_table.setRowCount(0)
            self.high_risk_table.setRowCount(0)
            self.low_risk_table.setRowCount(0)

            # 已禁用落盘记录，无需清理文件

            # 重置状态
            self.progress_bar.setValue(0)
            self.selected_images = set()
            self.all_checked = False
            self.select_all_checkbox.setChecked(False)
            self.delete_selected_btn.setEnabled(False)
            self.upload_status_label.setText("已清空所有图片和结果")

    def _export_high_risk_images(self):
        """导出高风险图片的名称到Excel文件"""
        # 检查是否有高风险图片
        if self.high_risk_table.rowCount() == 0:
            self.upload_status_label.setText("没有高风险图片可导出")
            self.upload_status_label.setStyleSheet("color: #ef4444;")
            return

        # 获取保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"高风险图片列表_{timestamp}.xlsx"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出高风险图片", default_filename, "Excel文件 (*.xlsx)"
        )

        if not file_path:  # 用户取消操作
            return

        try:
            # 收集高风险图片名称
            high_risk_names = []
            for row in range(self.high_risk_table.rowCount()):
                # 获取文件名（第二列）
                name_item = self.high_risk_table.item(row, 2)  # 修改为第2列
                if name_item:
                    high_risk_names.append({
                        "序号": row + 1,
                        "图片名称": name_item.text()
                    })

            # 创建DataFrame并保存为Excel
            df = pd.DataFrame(high_risk_names)
            df.to_excel(file_path, index=False, engine="openpyxl")

            # 显示成功信息
            self.upload_status_label.setText(f"已成功导出 {len(high_risk_names)} 个高风险图片名称到 {file_path}")
            self.upload_status_label.setStyleSheet("color: #10b981;")

            # 新增导出成功提示框
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("导出成功")
            msg_box.setText(f"已成功导出 {len(high_risk_names)} 个高风险图片名称")
            msg_box.setInformativeText(f"文件保存路径:\n{file_path}")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.exec()

        except Exception as e:
            error_msg = f"导出失败: {str(e)}"
            self.upload_status_label.setText(error_msg)
            self.upload_status_label.setStyleSheet("color: #ef4444;")
            CustomMessageBox(error_msg, is_success=False, parent=self).exec()

    def eventFilter(self, obj, event):
        # 处理图片预览对话框的事件
        if hasattr(obj, 'image_path') and obj.image_path:
            # 1. 键盘ESC关闭（保持不变）
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                obj.close()
                return True

            # 2. 鼠标释放事件（替代按下事件，确保按钮点击完整触发）
            if event.type() == QEvent.Type.MouseButtonRelease:  # 改为Release事件
                # 找到关闭按钮（通过ObjectName精确匹配）
                close_btn = obj.findChild(QPushButton, "previewCloseBtn")
                if close_btn and close_btn.underMouse():
                    # 点击在关闭按钮上，触发按钮点击事件（释放时触发）
                    close_btn.click()  # 主动触发点击信号
                    return True  # 已处理，阻止事件扩散
                else:
                    # 点击在图片区域或标题栏X按钮，直接关闭窗口
                    obj.close()
                    return True

        # 窗口拖动事件处理（保持不变）
        if obj == self.header_widget:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging = True
                    self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if self.dragging and (event.buttons() & Qt.MouseButton.LeftButton):
                    self.move(event.globalPosition().toPoint() - self.drag_position)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self.dragging = False
                return True

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.save_config()
        self.clean_expired_cache()
        self.cache_mutex.lock()
        try:
            self.save_cache()
        finally:
            self.cache_mutex.unlock()

        # 安全关闭预览窗口
        try:
            if self.current_preview_window is not None:
                self.current_preview_window.close()
                self.current_preview_window.deleteLater()
        except Exception:
            pass

        event.accept()

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    # 如果是带 --splash 打包的 exe，会内置 pyi_splash
    try:
        import pyi_splash
        pyi_splash.update_text("Loading UI...")
    except Exception:
        pyi_splash = None

    app = QApplication(sys.argv)

    # 启动主窗口
    window = ImageAnalyzerApp()
    window.show()

    # 主窗体显示后，主动关闭 PyInstaller 的 splash
    if pyi_splash:
        try:
            pyi_splash.close()
        except Exception:
            pass

    sys.exit(app.exec())