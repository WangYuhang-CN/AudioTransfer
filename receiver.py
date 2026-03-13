"""
接收端：接收 UDP 音频包并实时播放。

用法：
  python receiver.py                  # 使用默认扬声器
  python receiver.py --list           # 列出所有可用扬声器
  python receiver.py --device 1       # 指定扬声器设备索引
"""

import socket
import argparse

import numpy as np
import soundcard as sc

from config import PORT, SAMPLE_RATE, CHANNELS, CHUNK


def _select_speaker(speakers, index):
    if not speakers:
        raise RuntimeError("未检测到可用的扬声器设备")

    if index is not None:
        if index < 0 or index >= len(speakers):
            raise ValueError(f"扬声器设备索引超出范围: {index}")
        return speakers[index]

    speaker = sc.default_speaker()
    if speaker is None:
        raise RuntimeError("未找到默认扬声器设备")
    return speaker


def list_devices():
    print("\n=== 扬声器设备 ===")
    for i, s in enumerate(sc.all_speakers()):
        print(f"  [{i}] {s.name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="AudioTransfer 接收端")
    parser.add_argument("--list", action="store_true", help="列出所有扬声器设备")
    parser.add_argument("--device", type=int, default=None, help="扬声器设备索引")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    try:
        speakers = sc.all_speakers()
        speaker = _select_speaker(speakers, args.device)
    except (RuntimeError, ValueError) as exc:
        print(f"启动失败: {exc}")
        return

    print(f"\n[扬声器] 使用设备: {speaker.name}")

    # 创建 UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    # 接收缓冲区尽量小，避免系统积压旧包
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)

    packet_bytes = CHUNK * 2  # int16 = 2 bytes/sample

    print(f"监听端口: {PORT}")
    print(f"采样率: {SAMPLE_RATE}Hz | 帧大小: {CHUNK} 样本 (~{CHUNK/SAMPLE_RATE*1000:.1f}ms)\n")
    print("按 Ctrl+C 停止\n")

    try:
        with speaker.player(samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=CHUNK) as player:
            while True:
                data, _ = sock.recvfrom(packet_bytes + 64)  # +64 容错
                if len(data) < packet_bytes:
                    continue  # 丢弃残包

                audio_i16 = np.frombuffer(data[:packet_bytes], dtype=np.int16)
                audio_f32 = audio_i16.astype(np.float32) / 32767.0
                # soundcard player 需要 shape: (frames, channels)
                player.play(audio_f32.reshape(-1, CHANNELS))

    except KeyboardInterrupt:
        print("\n停止接收。")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
