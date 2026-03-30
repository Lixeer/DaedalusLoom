import argparse
import serial
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3", help="串口号")
    parser.add_argument("--baud", type=int, default=115200, help="波特率")
    parser.add_argument("--window", type=int, default=128, help="滑动窗口长度（子载波数量）")
    parser.add_argument("--history", type=int, default=2000, help="最大历史帧数")
    return parser.parse_args()

def csi_parse_frame(line):
    """
    将一行字符串解析为 CSI 复数数组
    - 自动提取 data:[...] 部分
    - 忽略非法字符和非整数
    - 奇数长度丢掉最后一个
    - 返回 np.complex64，如果无法解析返回空数组
    """
    import re
    line = line.strip()
    print(f"[csi_parse_frame] 解析行: {line}")
    
    # 尝试提取 data:[...] 部分
    match = re.search(r'data:\[([^\]]+)\]', line)
    if not match:
        # 没找到 data 部分，直接返回空
        print("[csi_parse_frame] 无法找到 data 字段")
        return None
    
    data_str = match.group(1)
    
    # 尝试解析成整数列表，忽略无法解析的部分
    data_list = []
    for x in data_str.split(','):
        x = x.strip()
        try:
            data_list.append(int(x))
        except ValueError:
            # 非整数直接忽略
            continue
    
    if len(data_list) < 2:
        return np.array([], dtype=np.complex64)
    
    # 奇数长度丢掉最后一个元素
    if len(data_list) % 2 != 0:
        data_list = data_list[:-1]
    
    # 转换为 Nx2 数组
    csi_np = np.array(data_list, dtype=np.int16).reshape(-1,2)
    
    # 实部 + j*虚部
    csi_complex = csi_np[:,1] + 1j*csi_np[:,0]
    
    return csi_complex
def serial_thread_func(port, baud, q):
    """
    串口线程：不断读取串口数据，放入队列
    """
    with serial.Serial(port, baud, timeout=1) as ser:
        print(f"[串口线程] 打开 {port} @ {baud}bps")
        while True:
            try:
                line = ser.readline().decode('utf-8')
                if not line:
                    continue
                csi_frame = csi_parse_frame(line)
                if csi_frame is not None:
                    q.put(csi_frame)
                    
                
            except Exception as e:
                print("[串口线程] 错误:", e)

def main():
    args = parse_args()
    q = queue.Queue()
    
    # 启动串口线程
    t = threading.Thread(target=serial_thread_func, args=(args.port, args.baud, q), daemon=True)
    t.start()

    # 滑动窗口历史缓存
    history_frames = []

    fig, ax = plt.subplots(figsize=(10,4))
    line1, = ax.plot([], [], label="Raw CSI")
    line2, = ax.plot([], [], label="SLIDE window")
    ax.set_xlim(0, args.window)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Real-time CSI")
    ax.grid(True)
    ax.legend()

    def update(frame):
        nonlocal history_frames
        # 从队列获取最新数据
        while not q.empty():
            csi = q.get()
            mag = np.abs(csi)
            mag_demean = mag - np.mean(mag)
            history_frames.append(mag_demean)
            if len(history_frames) > args.history:
                history_frames = history_frames[-args.history:]
        
        if not history_frames:
            return line1, line2

        # 最新一帧
        latest = history_frames[-1]
        # 滑动窗口取最后 window 个子载波
        window_data = latest[-args.window:] if len(latest) > args.window else latest
        line1.set_data(np.arange(len(window_data)), window_data)

        # 计算滑动平均作为平滑曲线
        smooth = np.convolve(window_data, np.ones(5)/5, mode='same')
        line2.set_data(np.arange(len(window_data)), smooth)

        return line1, line2

    ani = FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    main()