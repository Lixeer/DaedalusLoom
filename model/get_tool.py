# -*- coding: utf-8 -*-
import os
import sys
import serial
import threading
import queue
import re
from collections import deque
import numpy as np
import torch
import torch.nn as nn

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

LABEL_MAP = {'draw': 0, 'stand-up': 1, 'wave': 2}
INV_LABEL_MAP = {
    0: '画圈',
    1: '起立',
    2: '挥手'
}

# ----------------- Model Architecture -----------------
class Advanced1DCNN(nn.Module):
    def __init__(self, n_subcarriers=114, num_classes=3):
        super(Advanced1DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.AdaptiveAvgPool1d(4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# ----------------- Serial Receiver Thread -----------------
class SerialReceiver(QtCore.QThread):
    frame_received = QtCore.pyqtSignal(list)
    log_message = QtCore.pyqtSignal(str)

    def __init__(self, port, baud):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = False
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.5)
            self.running = True
            self.log_message.emit(f"Successfully connected to {self.port} at {self.baud} bps.")
        except Exception as e:
            self.log_message.emit(f"Error opening serial port: {e}")
            return

        # Pattern to capture data:[...] array
        pattern = re.compile(r'data:\s*\[([^\]]+)\]')
        fallback_pattern = re.compile(r'\[([^\]]+)\]')

        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    match = pattern.search(line) or fallback_pattern.search(line)
                    if not match:
                        continue

                    data_str = match.group(1)
                    data_list = []
                    for x in data_str.split(','):
                        x = x.strip()
                        try:
                            data_list.append(int(x))
                        except ValueError:
                            continue

                    if len(data_list) > 2:
                        self.frame_received.emit(data_list)
            except Exception as e:
                self.log_message.emit(f"Serial read error: {e}")
                self.msleep(100)

        if self.ser and self.ser.is_open:
            self.ser.close()
            self.log_message.emit("Serial port closed.")

    def stop(self):
        self.running = False
        self.wait()

# ----------------- Main GUI Window -----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WiFi CSI Real-time Gesture Inference (CNN1D)")
        self.resize(1100, 700)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.receiver = None

        # CSI Buffer: sliding window of size 50
        self.csi_window = deque(maxlen=50)
        self.motion_threshold = 2.5 # Threshold for motion detection (std dev)
        
        # Load model weights
        self.load_model()
        
        # Build UI layout
        self.init_ui()

        # Update Timer (approx. 60 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.data_queue = queue.Queue(maxsize=2000)

    def load_model(self):
        # Look for weights in both muti-model folder and root folder
        model_path = os.path.join("muti-model", "best_cnn1d.pth")
        if not os.path.exists(model_path):
            model_path = "best_cnn1d.pth"

        self.model = Advanced1DCNN(n_subcarriers=114, num_classes=3).to(self.device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"Model loaded successfully from '{model_path}' on {self.device}")
            except Exception as e:
                print(f"Error loading weight parameters: {e}")
        else:
            print(f"Warning: CNN1D weight file not found at '{model_path}'. Running with random weights.")

    def init_ui(self):
        # Central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # 1. Left side: Plots
        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=7)

        # Plot 1: CSI Amplitude per carrier
        self.amp_plot = pg.PlotWidget(title="Real-time CSI Amplitude (114 Subcarriers)")
        self.amp_plot.setLabel('left', 'Amplitude')
        self.amp_plot.setLabel('bottom', 'Subcarrier Index')
        self.amp_curve = self.amp_plot.plot(pen=pg.mkPen('y', width=2))
        left_layout.addWidget(self.amp_plot)

        # Plot 2: CSI Waterfall (Time series)
        self.waterfall_plot = pg.PlotWidget(title="Sliding Window CSI Waterfall (50 frames)")
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        # Apply colormap to look like a heat map/waterfall
        colormap = pg.colormap.get('viridis')
        self.waterfall_image.setLookupTable(colormap.getLookupTable())
        left_layout.addWidget(self.waterfall_plot)

        # 2. Right side: Controls & Predictions
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=3)

        # Group 1: Serial connection settings
        conn_group = QtWidgets.QGroupBox("Serial Settings")
        conn_layout = QtWidgets.QFormLayout(conn_group)
        
        self.port_input = QtWidgets.QLineEdit("COM3")
        self.baud_input = QtWidgets.QComboBox()
        self.baud_input.addItems(["921600", "115200", "57600", "9600"])
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        conn_layout.addRow("Port:", self.port_input)
        conn_layout.addRow("Baud Rate:", self.baud_input)
        conn_layout.addRow(self.connect_btn)
        right_layout.addWidget(conn_group)

        # Group 2: Inference Results panel
        infer_group = QtWidgets.QGroupBox("Real-time Inference")
        infer_layout = QtWidgets.QVBoxLayout(infer_group)

        # Large prediction display
        self.pred_label = QtWidgets.QLabel("IDLE")
        self.pred_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pred_label.setFont(QtGui.QFont("Microsoft YaHei", 36, QtGui.QFont.Weight.Bold))
        self.pred_label.setStyleSheet("color: #7f8c8d; background-color: #2c3e50; padding: 20px; border-radius: 10px;")
        infer_layout.addWidget(self.pred_label)

        # Motion Level progress bar and text
        self.motion_label = QtWidgets.QLabel("Motion Level: 0.00 / Threshold: 2.50")
        self.motion_label.setFont(QtGui.QFont("Microsoft YaHei", 11))
        infer_layout.addWidget(self.motion_label)

        self.motion_bar = QtWidgets.QProgressBar()
        self.motion_bar.setRange(0, 100)
        self.motion_bar.setValue(0)
        infer_layout.addWidget(self.motion_bar)

        self.motion_status = QtWidgets.QLabel("⚪ Motion Status: Standby")
        self.motion_status.setFont(QtGui.QFont("Microsoft YaHei", 12))
        infer_layout.addWidget(self.motion_status)

        right_layout.addWidget(infer_group)

        # Group 3: Connection logs
        log_group = QtWidgets.QGroupBox("System Logs")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

    def append_log(self, text):
        self.log_text.appendPlainText(text)

    def toggle_connection(self):
        if self.receiver is None or not self.receiver.isRunning():
            # Start connection
            port = self.port_input.text().strip()
            baud = int(self.baud_input.currentText())
            self.receiver = SerialReceiver(port, baud)
            self.receiver.frame_received.connect(self.enqueue_frame)
            self.receiver.log_message.connect(self.append_log)
            self.receiver.start()
            self.connect_btn.setText("Disconnect")
            self.timer.start(15)
        else:
            # Stop connection
            self.receiver.stop()
            self.receiver = None
            self.connect_btn.setText("Connect")
            self.timer.stop()
            self.pred_label.setText("IDLE")
            self.pred_label.setStyleSheet("color: #7f8c8d; background-color: #2c3e50; padding: 20px; border-radius: 10px;")
            self.motion_status.setText("⚪ Motion Status: Disconnected")

    def enqueue_frame(self, csi_frame):
        # Prevent queue overflow
        if self.data_queue.full():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                pass
        self.data_queue.put(csi_frame)

    def process_queue(self):
        # Read all available items in the queue
        new_frame_added = False
        latest_amp = None

        while not self.data_queue.empty():
            raw_list = self.data_queue.get()
            
            # 1. Parse complex number amplitude
            if len(raw_list) % 2 != 0:
                raw_list = raw_list[:-1]
                
            imag = np.array(raw_list[0::2])
            real = np.array(raw_list[1::2])
            csi = real + 1j * imag
            amplitude = np.abs(csi)
            
            # Filter non-zero subcarriers (ESP32 CSI active subcarriers)
            amplitude = amplitude[amplitude > 0]
            
            # Ensure shape is exactly 114
            if len(amplitude) > 114:
                amplitude = amplitude[:114]
            elif len(amplitude) < 114:
                amplitude = np.pad(amplitude, (0, 114 - len(amplitude)), 'edge')

            self.csi_window.append(amplitude)
            latest_amp = amplitude
            new_frame_added = True

        if not new_frame_added:
            return

        # 2. Update CSI Line Plot
        self.amp_curve.setData(np.arange(114), latest_amp)

        # 3. Handle inference and sliding window logic
        if len(self.csi_window) == 50:
            window_matrix = np.array(self.csi_window) # Shape: (50, 114)
            
            # Update waterfall plot (transpose for vertical time scroll)
            self.waterfall_image.setImage(window_matrix.T, autoLevels=True)

            # Calculate Motion Level using variance/standard deviation
            # We measure the variance over the temporal dimension (axis 0) across all subcarriers
            subcarrier_stds = np.std(window_matrix, axis=0)
            motion_val = float(subcarrier_stds.mean())

            # Update motion labels and progress bar
            self.motion_label.setText(f"Motion Level: {motion_val:.2f} / Threshold: {self.motion_threshold:.2f}")
            bar_val = min(100, int(motion_val * 20)) # scale for UI
            self.motion_bar.setValue(bar_val)

            # Run inference if motion level exceeds the trigger threshold
            if motion_val >= self.motion_threshold:
                self.motion_status.setText("🔴 Motion Status: ACTIVE MOTION")
                
                # Apply SR-Std Standardization ($\epsilon=2.0$)
                mean = window_matrix.mean(axis=0, keepdims=True)
                std = window_matrix.std(axis=0, keepdims=True)
                window_norm = (window_matrix - mean) / (std + 2.0)

                # Reshape to (batch, subcarriers, time) = (1, 114, 50)
                tensor_in = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
                tensor_in = tensor_in.to(self.device)

                # Predict
                with torch.no_grad():
                    logits = self.model(tensor_in)
                    _, predicted = torch.max(logits, 1)
                    pred_idx = predicted.item()
                    label_name = INV_LABEL_MAP[pred_idx]

                # Update UI Label color based on gesture prediction
                color_map = {
                    '画圈': "#3498db",       # Blue
                    '起立': "#2ecc71",   # Green
                    '挥手': "#e67e22"        # Orange/Red
                }
                color = color_map.get(label_name, "#ffffff")
                self.pred_label.setText(label_name)
                self.pred_label.setStyleSheet(f"color: white; background-color: {color}; padding: 20px; border-radius: 10px;")
            else:
                self.motion_status.setText("⚪ Motion Status: Standby")
                self.pred_label.setText("NO MOTION")
                self.pred_label.setStyleSheet("color: #7f8c8d; background-color: #2c3e50; padding: 20px; border-radius: 10px;")

    def closeEvent(self, event):
        if self.receiver and self.receiver.isRunning():
            self.receiver.stop()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
