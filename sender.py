"""
发送端：采集麦克风 + 系统音频（WASAPI Loopback），混音后通过 UDP 发送。

用法：
  python sender.py                    # 使用默认设备
  python sender.py --list             # 列出所有可用设备
  python sender.py --mic 1 --sys 2   # 指定麦克风和系统音频设备索引
"""

import socket
import threading
import queue
import argparse

import numpy as np
import soundcard as sc

from config import RECEIVER_IP, PORT, SAMPLE_RATE, CHANNELS, CHUNK, MIC_VOLUME, SYS_VOLUME

# 队列最大深度：超过则丢弃旧帧，避免积压导致延迟升高
QUEUE_MAXSIZE = 4

mic_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
sys_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
stop_event = threading.Event()


def _select_device(devices, index, default_getter, label):
    if not devices:
        raise RuntimeError(f"未检测到可用的{label}设备")

    if index is not None:
        if index < 0 or index >= len(devices):
            raise ValueError(f"{label}设备索引超出范围: {index}")
        return devices[index]

    device = default_getter()
    if device is None:
        raise RuntimeError(f"未找到默认{label}设备")
    return device


def list_devices():
    print("\n=== 麦克风设备 ===")
    for i, m in enumerate(sc.all_microphones(include_loopback=False)):
        print(f"  [{i}] {m.name}")

    print("\n=== 扬声器设备（用于 Loopback 系统音频）===")
    for i, s in enumerate(sc.all_speakers()):
        print(f"  [{i}] {s.name}")
    print()


def mic_capture_thread(mic_device):
    """采集麦克风，转为 float32 放入队列。"""
    try:
        print(f"[麦克风] 使用设备: {mic_device.name}")
        with mic_device.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=CHUNK) as rec:
            while not stop_event.is_set():
                data = rec.record(numframes=CHUNK)          # shape: (CHUNK, channels)
                chunk = data[:, 0].astype(np.float32)       # 取第一声道
                try:
                    mic_queue.put_nowait(chunk)
                except queue.Full:
                    mic_queue.get_nowait()                  # 丢弃最旧帧
                    mic_queue.put_nowait(chunk)
    except Exception as e:
        print(f"[麦克风] 错误: {e}")
        stop_event.set()


def sys_capture_thread(loopback_device):
    """采集系统音频（WASAPI Loopback），转为 float32 放入队列。"""
    try:
        print(f"[系统音频] 使用设备: {loopback_device.name}")
        with loopback_device.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=CHUNK) as rec:
            while not stop_event.is_set():
                data = rec.record(numframes=CHUNK)
                chunk = data[:, 0].astype(np.float32)
                try:
                    sys_queue.put_nowait(chunk)
                except queue.Full:
                    sys_queue.get_nowait()
                    sys_queue.put_nowait(chunk)
    except Exception as e:
        print(f"[系统音频] 错误: {e}")
        stop_event.set()


def mix_and_send(sock):
    """从两个队列取帧，混音后通过 UDP 发送。"""
    silence = np.zeros(CHUNK, dtype=np.float32)

    while not stop_event.is_set():
        # 获取麦克风帧（阻塞等待，驱动发送节奏）
        try:
            mic_frame = mic_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # 获取系统音频帧（非阻塞，没有则用静音）
        try:
            sys_frame = sys_queue.get_nowait()
        except queue.Empty:
            sys_frame = silence

        # 混音：加权相加，clip 防止溢出
        mixed_f32 = mic_frame * MIC_VOLUME + sys_frame * SYS_VOLUME
        mixed_f32 = np.clip(mixed_f32, -1.0, 1.0)

        # 转为 int16 字节发送
        mixed_i16 = (mixed_f32 * 32767).astype(np.int16)
        sock.sendto(mixed_i16.tobytes(), (RECEIVER_IP, PORT))


def main():
    parser = argparse.ArgumentParser(description="AudioTransfer 发送端")
    parser.add_argument("--list", action="store_true", help="列出所有设备")
    parser.add_argument("--mic", type=int, default=None, help="麦克风设备索引")
    parser.add_argument("--sys", type=int, default=None, help="系统音频（扬声器 Loopback）设备索引")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    stop_event.clear()

    try:
        mics = sc.all_microphones(include_loopback=False)
        mic_device = _select_device(mics, args.mic, sc.default_microphone, "麦克风")

        speakers = sc.all_speakers()
        speaker = _select_device(speakers, args.sys, sc.default_speaker, "扬声器")
        loopback_device = sc.get_microphone(speaker.id, include_loopback=True)
    except (RuntimeError, ValueError) as exc:
        print(f"启动失败: {exc}")
        return

    # 创建 UDP Socket（提升发送缓冲区）
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

    print(f"\n发送目标: {RECEIVER_IP}:{PORT}")
    print(f"采样率: {SAMPLE_RATE}Hz | 帧大小: {CHUNK} 样本 (~{CHUNK/SAMPLE_RATE*1000:.1f}ms)\n")
    print("按 Ctrl+C 停止\n")

    # 启动两个采集线程
    t_mic = threading.Thread(target=mic_capture_thread, args=(mic_device,), daemon=True)
    t_sys = threading.Thread(target=sys_capture_thread, args=(loopback_device,), daemon=True)
    t_mic.start()
    t_sys.start()

    try:
        mix_and_send(sock)
    except KeyboardInterrupt:
        print("\n停止发送。")
    finally:
        stop_event.set()
        sock.close()


if __name__ == "__main__":
    main()
