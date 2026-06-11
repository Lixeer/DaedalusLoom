import os
import sys
import time
import numpy as np
import torch
from pyqtgraph.Qt import QtWidgets, QtCore

# Add temp_workspace to path so we can import get_tool_fusion
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_tool_fusion import MainWindow

def run_test():
    print("Initializing headless PyQt Application...")
    app = QtWidgets.QApplication(sys.argv)
    
    print("Instantiating MainWindow...")
    win = MainWindow()
    
    # 1. Verify initialization states
    assert win.is_moving is False
    assert len(win.event_probabilities) == 0
    assert win.motion_trigger_counter == 0
    assert win.required_trigger_frames == 3
    print("[OK] Initialization states verified.")
    
    # 2. Simulate frames. We need to feed CSI data into the sliding window.
    # Each CSI frame needs to be a 114-element complex array.
    # To trigger different motion levels, we can manually fill win.csi_window.
    # Since csi_window size must be 50 to run inference:
    
    print("\n--- Test 1: Transient noise spike (1 frame of motion) ---")
    # Fill window with quiet frames (standard deviation = 0.0)
    for _ in range(50):
        win.csi_window.append(np.ones(114, dtype=np.complex128))
    
    # Process one cycle with quiet frames
    win.data_queue.put([1]*228) # dummy data to trigger new_frame_added in process_queue
    win.process_queue()
    assert win.is_moving is False
    assert win.motion_trigger_counter == 0
    
    # Inject 1 high-motion frame (e.g. modify the last frame in window to have high amplitude)
    # This will cause subcarrier_stds to have a non-zero mean.
    # Let's check how win.process_queue calculates motion:
    # subcarrier_stds = np.std(amplitude_matrix, axis=0)
    # motion_val = float(subcarrier_stds.mean())
    # So if we make amplitude of a frame larger:
    temp_window = [np.ones(114, dtype=np.complex128) for _ in range(49)]
    temp_window.append(np.ones(114, dtype=np.complex128) * 100.0) # huge variation
    for frame in temp_window:
        win.csi_window.append(frame)
        
    win.data_queue.put([1]*228)
    win.process_queue()
    
    # motion_val should be high. Since it's only 1 frame:
    print(f"Trigger counter after 1 frame of high motion: {win.motion_trigger_counter}")
    assert win.motion_trigger_counter == 1
    assert win.is_moving is False
    print("[OK] Transient spike correctly ignored by trigger guard.")
    
    # 3. Simulate Drop back to quiet
    # The next frame is quiet, trigger counter should reset to 0.
    for _ in range(50):
        win.csi_window.append(np.ones(114, dtype=np.complex128))
    win.data_queue.put([1]*228)
    win.process_queue()
    assert win.motion_trigger_counter == 0
    assert win.is_moving is False
    print("[OK] Trigger counter correctly reset to 0 after motion dropped.")
    
    # 4. Simulate a real gesture (3 consecutive frames of high motion)
    print("\n--- Test 2: Valid gesture start (3 consecutive frames) ---")
    for step in range(3):
        temp_window = [np.ones(114, dtype=np.complex128) for _ in range(49)]
        temp_window.append(np.ones(114, dtype=np.complex128) * 100.0)
        for frame in temp_window:
            win.csi_window.append(frame)
        win.data_queue.put([1]*228)
        win.process_queue()
        
    assert win.is_moving is True
    print(f"[OK] State is_moving: {win.is_moving}")
    print(f"[OK] Event probabilities accumulated: {len(win.event_probabilities)} frames.")
    assert len(win.event_probabilities) == 1 # Triggered on the 3rd frame, so 1 frame accumulated
    
    # 5. Continue gesture with 5 more active frames
    print("\n--- Test 3: Gesture continuation and weighted probability accumulation ---")
    for _ in range(5):
        temp_window = [np.ones(114, dtype=np.complex128) for _ in range(49)]
        temp_window.append(np.ones(114, dtype=np.complex128) * 100.0)
        for frame in temp_window:
            win.csi_window.append(frame)
        win.data_queue.put([1]*228)
        win.process_queue()
        
    assert win.is_moving is True
    print(f"[OK] Accumulated probability frames: {len(win.event_probabilities)}")
    # We should have accumulated a total of 1 (from step 3) + 5 = 6 frames
    assert len(win.event_probabilities) == 6
    
    # Check that probabilities are pairs of (prob_vector, weight)
    for probs, w in win.event_probabilities:
        assert len(probs) == 4
        assert w > 0.0
    print("[OK] Probability and weight structure verified.")
    
    # 6. Simulate debounce and event completion
    print("\n--- Test 4: Gesture debouncing and final decision ---")
    # Feed 15 quiet frames (debounce_frames = 15)
    for d in range(1, 16):
        for _ in range(50):
            win.csi_window.append(np.ones(114, dtype=np.complex128))
        win.data_queue.put([1]*228)
        win.process_queue()
        if d < 15:
            assert win.is_moving is True
            assert "Debouncing" in win.motion_status.text()
        else:
            assert win.is_moving is False
            assert "Standby" in win.motion_status.text()
            
    print("[OK] Event successfully debounced and finalized.")
    assert win.pred_label.text() == "挥手"
    print("[OK] Final prediction displayed on label successfully.")
    
    print("\n================ All Tests Passed! ================")

if __name__ == "__main__":
    run_test()
