import serial
import argparse
import re
import threading
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--history", type=int, default=2000)
    return parser.parse_args()


# ===== 数据 =====
rssi_all = deque()
index_all = deque()

lock = threading.Lock()
sample_index = 0


def serial_worker(port, baud, history):
    global sample_index

    ser = serial.Serial(port, baud, timeout=1)
    pattern = re.compile(r"rssi\s*:\s*(-?\d+)")

    print(f"[INFO] Serial started: {port}")

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            match = pattern.search(line)

            if match:
                rssi = int(match.group(1))

                with lock:
                    rssi_all.append(rssi)
                    index_all.append(sample_index)
                    sample_index += 1

                    if len(rssi_all) > history:
                        rssi_all.popleft()
                        index_all.popleft()

        except Exception as e:
            print("[WARN]", e)


def main():
    args = parse_args()

    t = threading.Thread(
        target=serial_worker,
        args=(args.port, args.baud, args.history),
        daemon=True
    )
    t.start()

    fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(10, 6))

    # ===== 原始 =====
    line_raw, = ax_raw.plot([], [], lw=1)
    ax_raw.set_title("Raw RSSI")
    ax_raw.set_ylabel("RSSI")

    # ===== 去均值 =====
    line_proc, = ax_proc.plot([], [], lw=2)
    ax_proc.set_title("Zero-Mean RSSI (Motion Enhanced)")
    ax_proc.set_xlabel("Sample Index")
    ax_proc.set_ylabel("RSSI (Centered)")

    def update(frame):
        with lock:
            if not index_all:
                return line_raw, line_proc

            x_all = list(index_all)
            y_all = list(rssi_all)

        # ===== 滑动窗口 =====
        x_win = x_all[-args.window:]
        y_win = y_all[-args.window:]

        # ===== 去均值（核心）=====
        mean = sum(y_win) / len(y_win)
        y_proc = [v - mean for v in y_win]

        # ===== 更新 raw =====
        line_raw.set_data(x_all, y_all)
        ax_raw.set_xlim(x_all[0], x_all[-1])
        ax_raw.set_ylim(min(y_all) - 5, max(y_all) + 5)

        # ===== 更新处理后 =====
        line_proc.set_data(x_win, y_proc)
        ax_proc.set_xlim(x_win[0], x_win[-1])

        # 去均值后围绕0对称更合理
        max_abs = max(abs(min(y_proc)), abs(max(y_proc)))
        ax_proc.set_ylim(-max_abs - 1, max_abs + 1)

        return line_raw, line_proc

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=50,
        blit=True
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()