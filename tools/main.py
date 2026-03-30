import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import re

# =============================
# 串口配置
# =============================
SERIAL_PORT = "COM3"
BAUD_RATE = 921600

# =============================
# 参数
# =============================
CSI_LEN = 256
SUBCARRIERS = CSI_LEN // 2
WINDOW_SIZE = 100  # 用于瀑布图

# =============================
# 初始化
# =============================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# 瀑布图缓存
csi_buffer = deque(maxlen=WINDOW_SIZE)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# =============================
# 解析函数
# =============================
def parse_csi(line):
    """
    从串口文本中提取 CSI 数组
    """
    match = re.search(r"\[(.*?)\]", line)
    if not match:
        return None

    data_str = match.group(1)
    data = list(map(int, data_str.split(',')))

    if len(data) != CSI_LEN:
        return None

    return data

# =============================
# 主循环
# =============================
while True:
    try:
        line = ser.readline().decode(errors='ignore').strip()

        if "CSI_DATA" not in line and "data:[" not in line:
            continue

        raw = parse_csi(line)
        if raw is None:
            continue

        # =============================
        # 转复数
        # =============================
        imag = np.array(raw[0::2])
        real = np.array(raw[1::2])
        csi = real + 1j * imag

        # =============================
        # 幅度 & 相位
        # =============================
        amplitude = np.abs(csi)
        phase = np.angle(csi)

        # 去掉全0区域（简单裁剪）
        valid = amplitude > 0
        amplitude = amplitude[valid]

        # =============================
        # 更新瀑布图缓存
        # =============================
        if len(amplitude) > 10:
            csi_buffer.append(amplitude)

        # =============================
        # 绘图
        # =============================
        ax1.clear()
        ax1.set_title("CSI Amplitude (Subcarriers)")
        ax1.plot(amplitude)
        ax1.set_xlabel("Subcarrier")
        ax1.set_ylabel("Amplitude")

        ax2.clear()
        ax2.set_title("CSI Waterfall (Time Series)")
        if len(csi_buffer) > 10:
            data_2d = np.array(csi_buffer)
            ax2.imshow(data_2d, aspect='auto', origin='lower')
            ax2.set_xlabel("Subcarrier")
            ax2.set_ylabel("Time")

        plt.pause(0.01)

    except KeyboardInterrupt:
        break

ser.close()