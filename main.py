"""
AudioTransfer - 内网音频实时传输工具
发送端：麦克风 + 系统音频混音 → UDP
接收端：UDP → 扬声器播放
"""

import sys
import socket
import queue
import threading
import ipaddress

import numpy as np
import soundcard as sc
import sounddevice as sd
from config import (
    APP_NAME,
    APP_VERSION,
    RECEIVER_IP as DEFAULT_TARGET_IP,
    PORT as DEFAULT_PORT,
    SAMPLE_RATE,
    CHANNELS,
    CHUNK,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSlider, QCheckBox,
    QPushButton, QMessageBox, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QPen, QFont, QIcon, QPixmap, QAction
import math
import os

CONFIG_FILE = "config.ini"


def _resource_path(filename: str) -> str:
    """兼容开发模式和 PyInstaller 打包模式的资源路径。"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def _app_icon() -> QIcon:
    """加载应用图标，优先使用 .ico 文件。"""
    path = _resource_path("AudioTransfer.ico")
    if os.path.exists(path):
        return QIcon(path)
    # 回退：动态绘制
    px = QPixmap(32, 32)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(137, 180, 250))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, 30, 30)
    p.setBrush(QColor(30, 30, 46))
    for i, h in enumerate([8, 14, 22, 14, 8]):
        bw, x = 3, 7 + i * 5
        p.drawRoundedRect(x, (32 - h) // 2, bw, h, 1, 1)
    p.end()
    return QIcon(px)

# 传输协议参数从 config.py 读取，下面是 GUI 本地播放缓冲策略
PLAYBACK_QUEUE_FRAMES = 8
STALE_AUDIO_BLOCKS = 3
FADE_OUT_BLOCKS = 6


def _parse_port(text: str) -> int:
    port = int(text.strip())
    if not 1 <= port <= 65535:
        raise ValueError("port out of range")
    return port


def _parse_target_ip(text: str) -> str:
    target = text.strip()
    ipaddress.ip_address(target)
    return target


def _resolve_sounddevice_device(device_name: str, *, is_input: bool) -> int:
    channel_key = "max_input_channels" if is_input else "max_output_channels"
    exact_matches = []
    partial_matches = []

    for idx, device in enumerate(sd.query_devices()):
        if device[channel_key] <= 0:
            continue
        current_name = device["name"]
        if current_name == device_name:
            exact_matches.append(idx)
        elif device_name in current_name or current_name in device_name:
            partial_matches.append(idx)

    if exact_matches:
        return exact_matches[0]
    if partial_matches:
        return partial_matches[0]

    device_type = "input" if is_input else "output"
    raise RuntimeError(f"未找到匹配的{device_type}设备: {device_name}")

# ─── 电平指示条 + dB 刻度 ─────────────────────────────────────
class LevelMeter(QWidget):
    BAR_H   = 13
    SCALE_H = 17
    DB_MIN  = -60
    DB_MAX  =   0
    # 每 5dB 一个刻度，整 10 才显示数字
    DB_STEPS = list(range(-60, 1, 5))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db         = self.DB_MIN   # 当前电平（dB）
        self._peak_db    = self.DB_MIN   # 峰值（dB）
        self._peak_hold  = 0
        total = self.BAR_H + self.SCALE_H
        self.setMinimumHeight(total)
        self.setMaximumHeight(total)

    def _db_to_x(self, db: float, width: int) -> int:
        """dB 均匀映射到像素 x（-60dB=左端，0dB=右端）"""
        ratio = (db - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        return int(width * max(0.0, min(1.0, ratio)))

    def set_level(self, level: float):
        """接收线性 RMS（0~1），转换为 dB 后更新"""
        if level > 0:
            db = 20 * math.log10(level)
        else:
            db = self.DB_MIN
        self._db = max(self.DB_MIN, min(self.DB_MAX, db))

        if self._db >= self._peak_db:
            self._peak_db   = self._db
            self._peak_hold = 25
        else:
            if self._peak_hold > 0:
                self._peak_hold -= 1
            else:
                self._peak_db = max(self._peak_db - 1.5, self._db)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w  = self.width()
        bh = self.BAR_H

        # ── 电平条背景 ──
        p.fillRect(0, 0, w, bh, QColor(30, 30, 46))

        # ── 电平条（绿→黄→红，按 dB 位置） ──
        fill_w = self._db_to_x(self._db, w)
        if fill_w > 0:
            g = QLinearGradient(0, 0, w, 0)
            g.setColorAt(0.0, QColor(166, 227, 161))
            g.setColorAt(0.7, QColor(249, 226, 175))
            g.setColorAt(1.0, QColor(243, 139, 168))
            p.fillRect(0, 2, fill_w, bh - 4, g)

        # ── 峰值白线 ──
        peak_x = self._db_to_x(self._peak_db, w)
        if peak_x > 1:
            p.setPen(QPen(QColor(255, 255, 255, 200), 2))
            p.drawLine(peak_x, 2, peak_x, bh - 2)

        # ── dB 刻度尺 ──
        p.fillRect(0, bh, w, self.SCALE_H, QColor(20, 20, 32))
        font = QFont("Segoe UI", 7)
        p.setFont(font)
        fm = p.fontMetrics()

        for db in self.DB_STEPS:
            x = self._db_to_x(db, w)
            show_label = (db % 10 == 0)   # 整 10 才显示数字

            # 刻度竖线（整10稍长）
            tick_h = 6 if show_label else 3
            p.setPen(QPen(QColor(88, 91, 112), 1))
            p.drawLine(x, bh, x, bh + tick_h)

            if show_label:
                label = str(db)
                lw = fm.horizontalAdvance(label)
                tx = max(0, min(x - lw // 2, w - lw))
                p.setPen(QColor(108, 112, 134))
                p.drawText(tx, bh + self.SCALE_H - 2, label)

        p.end()


# ─── 发送端后台线程 ───────────────────────────────────────────
class SenderWorker(QThread):
    level_mic   = pyqtSignal(float)
    level_sys   = pyqtSignal(float)
    level_mixed = pyqtSignal(float)
    status      = pyqtSignal(str)
    error       = pyqtSignal(str)

    def __init__(self, target_ip, port, mic_device, speaker_device,
                 use_mic, use_sys, mic_vol, sys_vol):
        super().__init__()
        self.target_ip      = target_ip
        self.port           = port
        self.mic_device     = mic_device
        self.speaker_device = speaker_device
        self.use_mic        = use_mic
        self.use_sys        = use_sys
        self.mic_vol        = mic_vol
        self.sys_vol        = sys_vol
        self._stop          = threading.Event()
        self._mic_q         = queue.Queue(maxsize=4)
        self._sys_q         = queue.Queue(maxsize=4)
        self._capture_threads = []

    def stop(self):
        self._stop.set()

    def _put(self, q: queue.Queue, chunk: np.ndarray):
        try:
            q.put_nowait(chunk)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(chunk)

    def _mic_capture(self):
        try:
            dev_idx = _resolve_sounddevice_device(self.mic_device.name, is_input=True)

            def callback(indata, frames, time, status):
                self._put(self._mic_q, indata[:, 0].copy())

            with sd.InputStream(
                device=dev_idx,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=CHUNK,
                callback=callback,
            ):
                self._stop.wait()   # 阻塞直到 stop() 被调用
        except Exception as e:
            self.error.emit(f"麦克风采集错误: {e}")
            self._stop.set()

    def _sys_capture(self):
        try:
            loopback = sc.get_microphone(self.speaker_device.id, include_loopback=True)
            with loopback.recorder(
                samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=CHUNK
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=CHUNK)
                    self._put(self._sys_q, data[:, 0].astype(np.float32))
        except Exception as e:
            self.error.emit(f"系统音频采集错误: {e}")
            self._stop.set()

    def run(self):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            silence = np.zeros(CHUNK, dtype=np.float32)

            if self.use_mic:
                mic_thread = threading.Thread(target=self._mic_capture, daemon=True)
                self._capture_threads.append(mic_thread)
                mic_thread.start()
            if self.use_sys:
                sys_thread = threading.Thread(target=self._sys_capture, daemon=True)
                self._capture_threads.append(sys_thread)
                sys_thread.start()

            self.status.emit(f"正在发送到 {self.target_ip}:{self.port}")

            # 主驱动队列：优先用麦克风节奏，否则用系统音频
            primary_q    = self._mic_q if self.use_mic else self._sys_q
            primary_is_mic = self.use_mic

            frame_count = 0
            while not self._stop.is_set():
                mic_frame = silence
                sys_frame = silence

                try:
                    frame = primary_q.get(timeout=0.5)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue

                if primary_is_mic:
                    mic_frame = frame
                    if self.use_sys:
                        try:
                            sys_frame = self._sys_q.get_nowait()
                        except queue.Empty:
                            pass
                else:
                    sys_frame = frame

                mic_scaled = mic_frame * self.mic_vol
                sys_scaled = sys_frame * self.sys_vol
                mixed      = np.clip(mic_scaled + sys_scaled, -1.0, 1.0)

                sock.sendto(
                    (mixed * 32767).astype(np.int16).tobytes(),
                    (self.target_ip, self.port)
                )

                # 每 4 帧更新一次电平，降低 GUI 刷新压力
                frame_count += 1
                if frame_count % 4 == 0:
                    self.level_mic.emit(float(np.sqrt(np.mean(mic_scaled ** 2))))
                    self.level_sys.emit(float(np.sqrt(np.mean(sys_scaled ** 2))))
                    self.level_mixed.emit(float(np.sqrt(np.mean(mixed ** 2))))

            sock.close()
            self.status.emit("已停止")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._stop.set()
            if sock is not None:
                sock.close()
            for thread in self._capture_threads:
                thread.join(timeout=1.0)


# ─── 接收端后台线程 ───────────────────────────────────────────
class ReceiverWorker(QThread):
    level  = pyqtSignal(float)
    status = pyqtSignal(str)
    error  = pyqtSignal(str)

    def __init__(self, port, output_device):
        super().__init__()
        self.port          = port
        self.output_device = output_device
        self._stop         = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        sock = None
        try:
            # 用设备名匹配 sounddevice 输出设备
            dev_idx = _resolve_sounddevice_device(self.output_device.name, is_input=False)

            packet_bytes = CHUNK * 2
            play_q = queue.Queue(maxsize=PLAYBACK_QUEUE_FRAMES)

            silence = np.zeros(CHUNK, dtype=np.float32)
            _last_chunk = silence
            missing_blocks = STALE_AUDIO_BLOCKS + FADE_OUT_BLOCKS

            def audio_callback(outdata, frames, time, status):
                nonlocal _last_chunk, missing_blocks
                try:
                    chunk = play_q.get_nowait()
                    _last_chunk = chunk
                    missing_blocks = 0
                except queue.Empty:
                    missing_blocks += 1
                    if missing_blocks <= STALE_AUDIO_BLOCKS:
                        chunk = _last_chunk
                    elif missing_blocks <= STALE_AUDIO_BLOCKS + FADE_OUT_BLOCKS:
                        remaining = STALE_AUDIO_BLOCKS + FADE_OUT_BLOCKS - missing_blocks + 1
                        chunk = _last_chunk * (remaining / FADE_OUT_BLOCKS)
                    else:
                        chunk = silence
                        _last_chunk = silence

                outdata[:, 0] = 0.0
                n = min(len(chunk), frames)
                outdata[:n, 0] = chunk[:n]

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", self.port))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            sock.settimeout(0.5)

            self.status.emit(f"监听端口 {self.port}，等待连接...")

            frame_count = 0
            with sd.OutputStream(
                device=dev_idx,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=CHUNK,
                callback=audio_callback,
            ):
                while not self._stop.is_set():
                    try:
                        data, _ = sock.recvfrom(packet_bytes + 64)
                    except socket.timeout:
                        continue

                    if len(data) < packet_bytes:
                        continue

                    audio_f32 = (np.frombuffer(data[:packet_bytes], dtype=np.int16)
                                 .astype(np.float32) / 32767.0)

                    # 队列满时丢弃最旧帧，保持低延迟
                    try:
                        play_q.put_nowait(audio_f32)
                    except queue.Full:
                        try:
                            play_q.get_nowait()
                        except queue.Empty:
                            pass
                        play_q.put_nowait(audio_f32)

                    frame_count += 1
                    if frame_count % 4 == 0:
                        self.level.emit(float(np.sqrt(np.mean(audio_f32 ** 2))))

            sock.close()
            self.status.emit("已停止")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._stop.set()
            if sock is not None:
                sock.close()


# ─── 发送端 Tab ───────────────────────────────────────────────
class SenderTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker   = None
        self._mics     = sc.all_microphones(include_loopback=False)
        self._speakers = sc.all_speakers()
        self._build_ui()
        self._load_settings()
        self._update_device_availability()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 网络 ──
        net = QGroupBox("网络目标")
        nl  = QHBoxLayout(net)
        nl.addWidget(QLabel("IP 地址:"))
        self.ip_edit = QLineEdit(DEFAULT_TARGET_IP)
        self.ip_edit.setMaximumWidth(160)
        nl.addWidget(self.ip_edit)
        nl.addSpacing(16)
        nl.addWidget(QLabel("端口:"))
        self.port_edit = QLineEdit(str(DEFAULT_PORT))
        self.port_edit.setMaximumWidth(70)
        nl.addWidget(self.port_edit)
        nl.addStretch()
        root.addWidget(net)

        # ── 麦克风 ──
        mic_grp = QGroupBox("麦克风")
        ml = QVBoxLayout(mic_grp)

        r1 = QHBoxLayout()
        self.mic_chk = QCheckBox("启用")
        self.mic_chk.setChecked(True)
        r1.addWidget(self.mic_chk)
        self.mic_combo = QComboBox()
        for m in self._mics:
            self.mic_combo.addItem(m.name)
        self._select_default_mic()
        r1.addWidget(self.mic_combo, 1)
        ml.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("音量:"))
        self.mic_vol = QSlider(Qt.Orientation.Horizontal)
        self.mic_vol.setRange(0, 100)
        self.mic_vol.setValue(100)
        r2.addWidget(self.mic_vol)
        self.mic_vol_lbl = QLabel("100%")
        self.mic_vol_lbl.setMinimumWidth(38)
        r2.addWidget(self.mic_vol_lbl)
        ml.addLayout(r2)
        self.mic_vol.valueChanged.connect(lambda v: self.mic_vol_lbl.setText(f"{v}%"))

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("电平:"))
        self.mic_meter = LevelMeter()
        r3.addWidget(self.mic_meter)
        ml.addLayout(r3)

        root.addWidget(mic_grp)

        # ── 系统音频 ──
        sys_grp = QGroupBox("系统音频  (Loopback)")
        sl = QVBoxLayout(sys_grp)

        s1 = QHBoxLayout()
        self.sys_chk = QCheckBox("启用")
        self.sys_chk.setChecked(False)
        s1.addWidget(self.sys_chk)
        self.sys_combo = QComboBox()
        for spk in self._speakers:
            self.sys_combo.addItem(spk.name)
        self._select_default_speaker()
        s1.addWidget(self.sys_combo, 1)
        sl.addLayout(s1)

        s2 = QHBoxLayout()
        s2.addWidget(QLabel("音量:"))
        self.sys_vol = QSlider(Qt.Orientation.Horizontal)
        self.sys_vol.setRange(0, 100)
        self.sys_vol.setValue(100)
        s2.addWidget(self.sys_vol)
        self.sys_vol_lbl = QLabel("100%")
        self.sys_vol_lbl.setMinimumWidth(38)
        s2.addWidget(self.sys_vol_lbl)
        sl.addLayout(s2)
        self.sys_vol.valueChanged.connect(lambda v: self.sys_vol_lbl.setText(f"{v}%"))

        s3 = QHBoxLayout()
        s3.addWidget(QLabel("电平:"))
        self.sys_meter = LevelMeter()
        s3.addWidget(self.sys_meter)
        sl.addLayout(s3)

        root.addWidget(sys_grp)

        # ── 混音输出 ──
        out_grp = QGroupBox("混音输出")
        ol = QHBoxLayout(out_grp)
        ol.addWidget(QLabel("电平:"))
        self.out_meter = LevelMeter()
        ol.addWidget(self.out_meter)
        root.addWidget(out_grp)

        root.addStretch()

        # ── 状态栏 + 按钮 ──
        btm = QHBoxLayout()
        self.status_lbl = QLabel("就绪")
        self.status_lbl.setObjectName("status")
        btm.addWidget(self.status_lbl, 1)
        self.start_btn = QPushButton("▶  开始发送")
        self.start_btn.setMinimumWidth(130)
        self.start_btn.clicked.connect(self._toggle)
        btm.addWidget(self.start_btn)
        root.addLayout(btm)

    def _select_default_mic(self):
        try:
            default = sc.default_microphone()
            for i, m in enumerate(self._mics):
                if m.id == default.id:
                    self.mic_combo.setCurrentIndex(i)
                    return
        except Exception:
            pass

    def _select_default_speaker(self):
        try:
            default = sc.default_speaker()
            for i, s in enumerate(self._speakers):
                if s.id == default.id:
                    self.sys_combo.setCurrentIndex(i)
                    return
        except Exception:
            pass

    def _load_settings(self):
        cfg = QSettings(CONFIG_FILE, QSettings.Format.IniFormat)
        self.ip_edit.setText(cfg.value("sender/ip", self.ip_edit.text()))
        self.port_edit.setText(cfg.value("sender/port", self.port_edit.text()))
        self.mic_chk.setChecked(cfg.value("sender/use_mic", True, type=bool))
        self.sys_chk.setChecked(cfg.value("sender/use_sys", False, type=bool))
        self.mic_vol.setValue(cfg.value("sender/mic_vol", 100, type=int))
        self.sys_vol.setValue(cfg.value("sender/sys_vol", 100, type=int))
        mic_name = cfg.value("sender/mic_name", "")
        if mic_name:
            for i, m in enumerate(self._mics):
                if m.name == mic_name:
                    self.mic_combo.setCurrentIndex(i)
                    break
        sys_name = cfg.value("sender/sys_name", "")
        if sys_name:
            for i, s in enumerate(self._speakers):
                if s.name == sys_name:
                    self.sys_combo.setCurrentIndex(i)
                    break

    def _update_device_availability(self):
        missing = []

        if not self._mics:
            self.mic_chk.setChecked(False)
            self.mic_chk.setEnabled(False)
            self.mic_combo.addItem("未检测到输入设备")
            self.mic_combo.setEnabled(False)
            self.mic_vol.setEnabled(False)
            missing.append("麦克风")

        if not self._speakers:
            self.sys_chk.setChecked(False)
            self.sys_chk.setEnabled(False)
            self.sys_combo.addItem("未检测到输出设备")
            self.sys_combo.setEnabled(False)
            self.sys_vol.setEnabled(False)
            missing.append("输出设备")

        if missing:
            self.status_lbl.setText(f"未检测到{' / '.join(missing)}")

    def save_settings(self):
        cfg = QSettings(CONFIG_FILE, QSettings.Format.IniFormat)
        cfg.setValue("sender/ip",       self.ip_edit.text())
        cfg.setValue("sender/port",     self.port_edit.text())
        cfg.setValue("sender/use_mic",  self.mic_chk.isChecked())
        cfg.setValue("sender/use_sys",  self.sys_chk.isChecked())
        cfg.setValue("sender/mic_vol",  self.mic_vol.value())
        cfg.setValue("sender/sys_vol",  self.sys_vol.value())
        cfg.setValue("sender/mic_name", self.mic_combo.currentText())
        cfg.setValue("sender/sys_name", self.sys_combo.currentText())

    def _toggle(self):
        if self._worker and self._worker.isRunning():
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.mic_chk.isChecked() and not self.sys_chk.isChecked():
            QMessageBox.warning(self, "提示", "请至少启用一个音频源（麦克风或系统音频）")
            return

        try:
            port = _parse_port(self.port_edit.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "端口号无效")
            return

        try:
            target_ip = _parse_target_ip(self.ip_edit.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "IP 地址无效")
            return

        if self.mic_chk.isChecked():
            if not self._mics or self.mic_combo.currentIndex() < 0:
                QMessageBox.warning(self, "错误", "当前没有可用的麦克风设备")
                return
            mic_dev = self._mics[self.mic_combo.currentIndex()]
        else:
            mic_dev = None

        if self.sys_chk.isChecked():
            if not self._speakers or self.sys_combo.currentIndex() < 0:
                QMessageBox.warning(self, "错误", "当前没有可用的系统音频输出设备")
                return
            spk_dev = self._speakers[self.sys_combo.currentIndex()]
        else:
            spk_dev = None

        self._worker = SenderWorker(
            target_ip      = target_ip,
            port           = port,
            mic_device     = mic_dev,
            speaker_device = spk_dev,
            use_mic        = self.mic_chk.isChecked(),
            use_sys        = self.sys_chk.isChecked(),
            mic_vol        = self.mic_vol.value() / 100.0,
            sys_vol        = self.sys_vol.value() / 100.0,
        )
        self._worker.level_mic.connect(self.mic_meter.set_level)
        self._worker.level_sys.connect(self.sys_meter.set_level)
        self._worker.level_mixed.connect(self.out_meter.set_level)
        self._worker.status.connect(self.status_lbl.setText)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self.start_btn.setText("■  停止")
        self.start_btn.setProperty("danger", True)
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self._set_inputs(False)
        if hasattr(self.window(), "_update_tray_menu"):
            self.window()._update_tray_menu()

    def _stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        self.start_btn.setText("▶  开始发送")
        self.start_btn.setProperty("danger", False)
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.status_lbl.setText("已停止")
        self._set_inputs(True)
        for m in (self.mic_meter, self.sys_meter, self.out_meter):
            m.set_level(0)
        if hasattr(self.window(), "_update_tray_menu"):
            self.window()._update_tray_menu()

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "错误", msg)
        self._stop()

    def _set_inputs(self, enabled: bool):
        self.ip_edit.setEnabled(enabled)
        self.port_edit.setEnabled(enabled)
        self.mic_chk.setEnabled(enabled and bool(self._mics))
        self.mic_combo.setEnabled(enabled and bool(self._mics))
        self.sys_chk.setEnabled(enabled and bool(self._speakers))
        self.sys_combo.setEnabled(enabled and bool(self._speakers))


# ─── 接收端 Tab ───────────────────────────────────────────────
class ReceiverTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker   = None
        self._speakers = sc.all_speakers()
        self._build_ui()
        self._load_settings()
        self._update_device_availability()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 网络 ──
        net = QGroupBox("监听设置")
        nl  = QHBoxLayout(net)
        nl.addWidget(QLabel("监听端口:"))
        self.port_edit = QLineEdit(str(DEFAULT_PORT))
        self.port_edit.setMaximumWidth(80)
        nl.addWidget(self.port_edit)
        nl.addStretch()
        root.addWidget(net)

        # ── 播放设备 ──
        spk_grp = QGroupBox("播放设备")
        sl = QVBoxLayout(spk_grp)
        self.spk_combo = QComboBox()
        for s in self._speakers:
            self.spk_combo.addItem(s.name)
        self._select_default_speaker()
        sl.addWidget(self.spk_combo)

        mr = QHBoxLayout()
        mr.addWidget(QLabel("接收电平:"))
        self.in_meter = LevelMeter()
        mr.addWidget(self.in_meter)
        sl.addLayout(mr)
        root.addWidget(spk_grp)

        root.addStretch()

        # ── 状态栏 + 按钮 ──
        btm = QHBoxLayout()
        self.status_lbl = QLabel("就绪")
        self.status_lbl.setObjectName("status")
        btm.addWidget(self.status_lbl, 1)
        self.start_btn = QPushButton("▶  开始接收")
        self.start_btn.setMinimumWidth(130)
        self.start_btn.clicked.connect(self._toggle)
        btm.addWidget(self.start_btn)
        root.addLayout(btm)

    def _select_default_speaker(self):
        try:
            default = sc.default_speaker()
            for i, s in enumerate(self._speakers):
                if s.id == default.id:
                    self.spk_combo.setCurrentIndex(i)
                    return
        except Exception:
            pass

    def _load_settings(self):
        cfg = QSettings(CONFIG_FILE, QSettings.Format.IniFormat)
        self.port_edit.setText(cfg.value("receiver/port", self.port_edit.text()))
        spk_name = cfg.value("receiver/spk_name", "")
        if spk_name:
            for i, s in enumerate(self._speakers):
                if s.name == spk_name:
                    self.spk_combo.setCurrentIndex(i)
                    break

    def _update_device_availability(self):
        if self._speakers:
            return

        self.spk_combo.addItem("未检测到输出设备")
        self.spk_combo.setEnabled(False)
        self.status_lbl.setText("未检测到输出设备")

    def save_settings(self):
        cfg = QSettings(CONFIG_FILE, QSettings.Format.IniFormat)
        cfg.setValue("receiver/port",     self.port_edit.text())
        cfg.setValue("receiver/spk_name", self.spk_combo.currentText())

    def _toggle(self):
        if self._worker and self._worker.isRunning():
            self._stop()
        else:
            self._start()

    def _start(self):
        try:
            port = _parse_port(self.port_edit.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "端口号无效")
            return

        if not self._speakers or self.spk_combo.currentIndex() < 0:
            QMessageBox.warning(self, "错误", "当前没有可用的播放设备")
            return

        spk = self._speakers[self.spk_combo.currentIndex()]
        self._worker = ReceiverWorker(port=port, output_device=spk)
        self._worker.level.connect(self.in_meter.set_level)
        self._worker.status.connect(self.status_lbl.setText)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self.start_btn.setText("■  停止")
        self.start_btn.setProperty("danger", True)
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self._set_inputs(False)
        if hasattr(self.window(), "_update_tray_menu"):
            self.window()._update_tray_menu()

    def _stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        self.start_btn.setText("▶  开始接收")
        self.start_btn.setProperty("danger", False)
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.status_lbl.setText("已停止")
        self._set_inputs(True)
        self.in_meter.set_level(0)
        if hasattr(self.window(), "_update_tray_menu"):
            self.window()._update_tray_menu()

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "错误", msg)
        self._stop()

    def _set_inputs(self, enabled: bool):
        self.port_edit.setEnabled(enabled)
        self.spk_combo.setEnabled(enabled and bool(self._speakers))


# ─── 主窗口 ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumWidth(480)
        self.setMinimumHeight(520)
        self._tray = None
        self._quit_requested = False

        self.sender_tab   = SenderTab()
        self.receiver_tab = ReceiverTab()
        tabs = QTabWidget()
        tabs.addTab(self.sender_tab,   "  发送端  ")
        tabs.addTab(self.receiver_tab, "  接收端  ")
        self.setCentralWidget(tabs)
        self._apply_style()
        icon = _app_icon()
        self.setWindowIcon(icon)
        self._setup_tray(icon)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background: #313244;
                color: #a6adc8;
                padding: 8px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected { background: #45475a; }
            QLineEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px 8px;
                color: #cdd6f4;
            }
            QLineEdit:focus { border-color: #89b4fa; }
            QComboBox {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px 8px;
                color: #cdd6f4;
            }
            QComboBox:focus { border-color: #89b4fa; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: none; }
            QComboBox QAbstractItemView {
                background: #313244;
                border: 1px solid #45475a;
                selection-background-color: #89b4fa;
                selection-color: #1e1e2e;
                outline: none;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #45475a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 14px; height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border-radius: 2px;
            }
            QPushButton {
                background: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover  { background: #b4befe; }
            QPushButton:pressed { background: #74c7ec; }
            QPushButton[danger="true"] {
                background: #f38ba8;
                color: #1e1e2e;
            }
            QPushButton[danger="true"]:hover { background: #eba0ac; }
            QCheckBox { spacing: 6px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #45475a;
                border-radius: 3px;
                background: #313244;
            }
            QCheckBox::indicator:checked  { background: #89b4fa; border-color: #89b4fa; }
            QCheckBox::indicator:hover    { border-color: #89b4fa; }
            QLabel#status { color: #6c7086; font-size: 12px; }
        """)

    def _setup_tray(self, icon: QIcon):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(icon)
        self._tray.setToolTip(f"{APP_NAME} v{APP_VERSION}")

        menu = QMenu()

        self._act_send = QAction("▶  开始发送", self)
        self._act_send.triggered.connect(self.sender_tab._toggle)
        self._act_send.triggered.connect(self._update_tray_menu)
        menu.addAction(self._act_send)

        self._act_recv = QAction("▶  开始接收", self)
        self._act_recv.triggered.connect(self.receiver_tab._toggle)
        self._act_recv.triggered.connect(self._update_tray_menu)
        menu.addAction(self._act_recv)

        menu.addSeparator()

        act_show = QAction("显示窗口", self)
        act_show.triggered.connect(self._show_window)
        menu.addAction(act_show)

        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self._quit)
        menu.addAction(act_quit)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(
            lambda reason: self._show_window()
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )
        # 同步按钮状态 → 托盘菜单文字
        self.sender_tab.start_btn.clicked.connect(self._update_tray_menu)
        self.receiver_tab.start_btn.clicked.connect(self._update_tray_menu)

        self._tray.show()

    def _update_tray_menu(self):
        if self._tray is None:
            return

        sending  = self.sender_tab._worker   is not None and self.sender_tab._worker.isRunning()
        receiving = self.receiver_tab._worker is not None and self.receiver_tab._worker.isRunning()
        self._act_send.setText("■  停止发送" if sending  else "▶  开始发送")
        self._act_recv.setText("■  停止接收" if receiving else "▶  开始接收")

    def _show_window(self):
        self.showNormal()
        self.activateWindow()

    def _stop_workers(self):
        self.sender_tab._stop()
        self.receiver_tab._stop()

    def _quit(self):
        self._quit_requested = True
        self._stop_workers()
        self._save_all()
        if self._tray is not None:
            self._tray.hide()
        QApplication.quit()

    def _save_all(self):
        self.sender_tab.save_settings()
        self.receiver_tab.save_settings()

    def closeEvent(self, event):
        """点 X 最小化到托盘，不退出"""
        if self._tray is None or self._quit_requested:
            self._save_all()
            self._stop_workers()
            event.accept()
            return

        event.ignore()
        self._save_all()
        self.hide()
        self._tray.showMessage(
            APP_NAME,
            "程序已最小化到托盘，右键图标可开始发送/接收",
            QSystemTrayIcon.MessageIcon.Information,
            2000
        )


# ─── 入口 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Windows 任务栏图标：设置 AppUserModelID 让 Windows 正确关联图标
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"{APP_NAME}.App.{APP_VERSION}"
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    app.setWindowIcon(_app_icon())   # 所有窗口默认图标

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
