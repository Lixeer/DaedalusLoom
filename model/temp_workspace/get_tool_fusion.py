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

# Label mappings for the new dataset
LABEL_MAP = {'cut': 0, 'grip': 1, 'draw_o': 2}
INV_LABEL_MAP = {
    0: '挥手',
    1: '抓握',
    2: '画圈',
    3: '未知'
}

# ----------------- Model Architecture -----------------
class Advanced1DCNN(nn.Module):
    def __init__(self, n_subcarriers=228, num_classes=4):
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
        self.setWindowTitle("WiFi CSI Complex Amplitude & Phase Real-time Inference (CNN1D)")
        self.resize(1100, 700)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.receiver = None

        # CSI Buffer: sliding window of size 50 complex frames
        self.csi_window = deque(maxlen=50)
        
        # Tuning parameters
        self.motion_threshold = 2.5
        self.confidence_threshold = 0.80
        self.smoothing_frames = 7
        self.pred_history = deque(maxlen=7)
        
        # Event detection states
        self.is_moving = False
        self.event_probabilities = []
        self.idle_counter = 0
        self.debounce_frames = 15
        self.motion_trigger_counter = 0
        self.required_trigger_frames = 3
        
        # Load model weights
        self.load_model()
        
        # Build UI layout
        self.init_ui()

        # Update Timer (approx. 60 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.data_queue = queue.Queue(maxsize=2000)

    def load_model(self):
        # Look for weights in possible directories
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_optimized_cnn.pth"),
            os.path.join("temp_workspace", "models", "best_optimized_cnn.pth"),
            os.path.join("models", "best_optimized_cnn.pth"),
            "best_optimized_cnn.pth"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        self.model = Advanced1DCNN(n_subcarriers=228, num_classes=4).to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"Model loaded successfully from '{model_path}' on {self.device}")
            except Exception as e:
                print(f"Error loading weight parameters: {e}")
        else:
            print("Warning: Optimized CNN1D (best_optimized_cnn.pth) weight file not found. Running with random weights.")

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

        # Sliders for parameters tuning
        ctrl_group = QtWidgets.QGroupBox("Parameter Tuning (实时调优)")
        ctrl_layout = QtWidgets.QFormLayout(ctrl_group)
        
        self.motion_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.motion_slider.setRange(10, 100)
        self.motion_slider.setValue(25) # 2.5
        self.motion_slider.valueChanged.connect(self.update_parameters)
        self.motion_val_lbl = QtWidgets.QLabel("2.5")
        
        motion_row = QtWidgets.QHBoxLayout()
        motion_row.addWidget(self.motion_slider)
        motion_row.addWidget(self.motion_val_lbl)
        ctrl_layout.addRow("Motion Thresh:", motion_row)
        
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(50, 100)
        self.conf_slider.setValue(80) # 80%
        self.conf_slider.valueChanged.connect(self.update_parameters)
        self.conf_val_lbl = QtWidgets.QLabel("0.80")
        
        conf_row = QtWidgets.QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_val_lbl)
        ctrl_layout.addRow("Min Confidence:", conf_row)
        
        self.vote_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.vote_slider.setRange(1, 21)
        self.vote_slider.setValue(7)
        self.vote_slider.valueChanged.connect(self.update_parameters)
        self.vote_val_lbl = QtWidgets.QLabel("7")
        
        vote_row = QtWidgets.QHBoxLayout()
        vote_row.addWidget(self.vote_slider)
        vote_row.addWidget(self.vote_val_lbl)
        ctrl_layout.addRow("Smoothing Frames:", vote_row)

        self.debounce_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.debounce_slider.setRange(5, 30)
        self.debounce_slider.setValue(15)
        self.debounce_slider.valueChanged.connect(self.update_parameters)
        self.debounce_val_lbl = QtWidgets.QLabel("15")
        
        debounce_row = QtWidgets.QHBoxLayout()
        debounce_row.addWidget(self.debounce_slider)
        debounce_row.addWidget(self.debounce_val_lbl)
        ctrl_layout.addRow("Event Debounce:", debounce_row)
        
        infer_layout.addWidget(ctrl_group)

        right_layout.addWidget(infer_group)

        # Group 3: Connection logs
        log_group = QtWidgets.QGroupBox("System Logs")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

    def update_parameters(self):
        self.motion_threshold = self.motion_slider.value() / 10.0
        self.motion_val_lbl.setText(f"{self.motion_threshold:.1f}")
        
        self.confidence_threshold = self.conf_slider.value() / 100.0
        self.conf_val_lbl.setText(f"{self.confidence_threshold:.2f}")
        
        self.smoothing_frames = self.vote_slider.value()
        self.vote_val_lbl.setText(str(self.smoothing_frames))
        
        self.debounce_frames = self.debounce_slider.value()
        self.debounce_val_lbl.setText(str(self.debounce_frames))
        
        current_history = list(self.pred_history)
        self.pred_history = deque(current_history, maxlen=self.smoothing_frames)
        
        self.motion_label.setText(f"Motion Level: 0.00 / Threshold: {self.motion_threshold:.2f}")

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
            self.is_moving = False
            self.event_probabilities = []
            self.motion_trigger_counter = 0
            self.idle_counter = 0

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
        latest_csi = None

        while not self.data_queue.empty():
            raw_list = self.data_queue.get()
            
            # Parse complex numbers (interleaved real and imaginary)
            if len(raw_list) % 2 != 0:
                raw_list = raw_list[:-1]
                
            imag = np.array(raw_list[0::2])
            real = np.array(raw_list[1::2])
            csi = real + 1j * imag
            
            # Filter non-zero subcarriers (based on amplitude)
            non_zero_mask = np.abs(csi) > 0
            csi_filtered = csi[non_zero_mask]
            
            # Ensure shape is exactly 114
            if len(csi_filtered) > 114:
                csi_filtered = csi_filtered[:114]
            elif len(csi_filtered) < 114:
                # pad with edge values
                padding_len = 114 - len(csi_filtered)
                if len(csi_filtered) > 0:
                    csi_filtered = np.pad(csi_filtered, (0, padding_len), 'edge')
                else:
                    csi_filtered = np.zeros(114, dtype=np.complex128)

            self.csi_window.append(csi_filtered)
            latest_csi = csi_filtered
            new_frame_added = True

        if not new_frame_added or latest_csi is None:
            return

        # 2. Update CSI Line Plot (Amplitude)
        self.amp_curve.setData(np.arange(114), np.abs(latest_csi))

        # 3. Handle inference and sliding window logic
        if len(self.csi_window) == 50:
            window_complex = np.array(self.csi_window) # Shape: (50, 114) complex
            
            # Extract amplitude and update waterfall plot
            amplitude_matrix = np.abs(window_complex)
            self.waterfall_image.setImage(amplitude_matrix.T, autoLevels=True)

            # Calculate Motion Level using variance/standard deviation
            subcarrier_stds = np.std(amplitude_matrix, axis=0)
            motion_val = float(subcarrier_stds.mean())

            # Update motion labels and progress bar
            self.motion_label.setText(f"Motion Level: {motion_val:.2f} / Threshold: {self.motion_threshold:.2f}")
            bar_val = min(100, int(motion_val * 20))
            self.motion_bar.setValue(bar_val)

            # Run inference if motion level exceeds the trigger threshold
            if motion_val >= self.motion_threshold:
                self.idle_counter = 0
                if not self.is_moving:
                    self.motion_trigger_counter += 1
                    if self.motion_trigger_counter >= self.required_trigger_frames:
                        self.is_moving = True
                        self.event_probabilities = []
                        self.pred_label.setText("DETECTING...")
                        self.pred_label.setStyleSheet("color: white; background-color: #f1c40f; padding: 20px; border-radius: 10px;")
                        self.append_log(f"Motion triggered ({self.required_trigger_frames} consecutive frames). Gesture event started.")
                
                if self.is_moving:
                    # Preprocessing pipeline:
                    # 1. Amplitude and Phase extraction
                    X_amp = np.abs(window_complex)
                    X_phase = np.angle(window_complex)

                    # 2. Vectorized Linear Phase Calibration (unwrap + CFO/SFO subtraction)
                    unwrapped = np.unwrap(X_phase, axis=1) # unwrap along subcarriers
                    x_idx = np.arange(114)
                    x_mean = x_idx.mean()
                    x_dev = x_idx - x_mean
                    D = np.sum(x_dev**2)

                    Y_mean = unwrapped.mean(axis=1, keepdims=True)
                    y_dev = unwrapped - Y_mean
                    a = np.sum(y_dev * x_dev, axis=1, keepdims=True) / D # Slope per frame
                    X_phase_cal = y_dev - a * x_dev

                    # 3. SR-Std Standardization (eps=2.0)
                    eps = 2.0
                    mean_amp = X_amp.mean(axis=0, keepdims=True)
                    std_amp = X_amp.std(axis=0, keepdims=True)
                    X_amp_norm = (X_amp - mean_amp) / (std_amp + eps)

                    mean_phase = X_phase_cal.mean(axis=0, keepdims=True)
                    std_phase = X_phase_cal.std(axis=0, keepdims=True)
                    X_phase_norm = (X_phase_cal - mean_phase) / (std_phase + eps)

                    # 4. Feature Fusion: (50, 114) + (50, 114) -> (50, 228)
                    X_combined = np.concatenate([X_amp_norm, X_phase_norm], axis=1)

                    # Reshape to (batch, channels, time) = (1, 228, 50)
                    tensor_in = torch.tensor(X_combined, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
                    tensor_in = tensor_in.to(self.device)

                    # Predict
                    with torch.no_grad():
                        logits = self.model(tensor_in)
                        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                        pred_idx = np.argmax(probabilities)
                        confidence = float(probabilities[pred_idx])
                        
                    raw_label = INV_LABEL_MAP[pred_idx]
                    
                    # Calculate dynamic weight based on motion level
                    weight = (motion_val - self.motion_threshold) ** 2
                    
                    # Accumulate only when confidence is >= 0.50 and class is not 'unknown' (index 3)
                    if confidence >= 0.50 and pred_idx != 3:
                        self.event_probabilities.append((probabilities, weight))
                    
                    # Check confidence threshold before pushing to smoothing queue (for real-time GUI feedback)
                    if confidence >= self.confidence_threshold:
                        self.pred_history.append(raw_label)
                    
                    # Resolve smoothed label using majority vote for active display
                    if len(self.pred_history) > 0:
                        from collections import Counter
                        counter = Counter(self.pred_history)
                        smoothed_label = counter.most_common(1)[0][0]
                    else:
                        smoothed_label = "UNCERTAIN"

                    # Update UI Label color based on smoothed gesture prediction during action
                    if smoothed_label != "UNCERTAIN":
                        self.motion_status.setText(f"🔴 Motion Status: ACTIVE ({smoothed_label} {confidence*100:.1f}%)")
                    else:
                        self.motion_status.setText(f"🔴 Motion Status: ACTIVE (UNCERTAIN)")
            else:
                if self.is_moving:
                    self.motion_trigger_counter = 0 # reset trigger counter
                    self.idle_counter += 1
                    self.motion_status.setText(f"🟡 Motion Status: Debouncing ({self.idle_counter}/{self.debounce_frames})")
                    
                    if self.idle_counter >= self.debounce_frames:
                        # Event completed! Calculate the final classification using weighted probabilities
                        self.is_moving = False
                        self.pred_history.clear()
                        
                        if len(self.event_probabilities) > 0:
                            # Sum weighted probabilities over the active period, ignoring the 'unknown' dimension
                            probs_sum = np.zeros(4)
                            total_weight = 0.0
                            for probs, w in self.event_probabilities:
                                probs_sum += probs * w
                                total_weight += w
                            
                            if total_weight > 0:
                                probs_sum /= total_weight
                            else:
                                probs_sum = np.sum([p for p, _ in self.event_probabilities], axis=0)
                                
                            pred_idx = np.argmax(probs_sum[:3]) # Only predict among cut, grip, draw_o
                            final_label = INV_LABEL_MAP[pred_idx]
                            
                            color_map = {
                                '挥手': "#3498db",       # Blue
                                '抓握': "#e74c3c",      # Red
                                '画圈': "#9b59b6"     # Purple
                            }
                            color = color_map.get(final_label, "#ffffff")
                            self.pred_label.setText(final_label)
                            self.pred_label.setStyleSheet(f"color: white; background-color: {color}; padding: 20px; border-radius: 10px; border: 3px solid white;")
                            
                            self.motion_status.setText(f"⚪ Motion Status: Standby")
                            self.append_log(f"Gesture event completed. Final prediction: {final_label} (weighted prob: {probs_sum[pred_idx]*100:.1f}%)")
                        else:
                            self.pred_label.setText("UNCERTAIN")
                            self.pred_label.setStyleSheet("color: #7f8c8d; background-color: #2c3e50; padding: 20px; border-radius: 10px;")
                            self.motion_status.setText("⚪ Motion Status: Standby")
                            self.append_log("Gesture event completed, but no high-confidence frames were accumulated.")
                else:
                    self.motion_trigger_counter = 0
                    self.motion_status.setText("⚪ Motion Status: Standby")

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
