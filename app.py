from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import math
import time
import subprocess
import threading
import serial
from picamera2 import Picamera2

app = Flask(__name__)

# ==========================================
# SINGLE SOURCE OF TRUTH: MAPPING DICTIONARY
# (Updated from True Spreadsheet Mapping)
# ==========================================
CV_TO_PCB_MAP = {
    # --- OUTER RING ---
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
    10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19,
    20: 38, 21: 39, 22: 40, 23: 41, 24: 42, 25: 43, 26: 44, 27: 45, 28: 46, 29: 47,
    30: 48, 31: 49, 32: 50, 33: 51, 34: 52, 35: 53, 36: 54, 37: 55, 38: 56, 39: 57,
    40: 76, 41: 77, 42: 78, 43: 79, 44: 80, 45: 81, 46: 82, 47: 83, 48: 84, 49: 85,
    50: 86, 51: 87, 52: 88, 53: 89, 54: 90, 55: 91, 56: 92, 57: 93,
    
    # --- INNER RING ---
    58: 37, 59: 36, 60: 35, 61: 34, 62: 33, 63: 32, 64: 31, 65: 30, 66: 29, 67: 28,
    68: 27, 69: 26, 70: 25, 71: 24, 72: 23, 73: 22, 74: 21, 75: 20,
    76: 75, 77: 74, 78: 73, 79: 72, 80: 71, 81: 70, 82: 69, 83: 68, 84: 67, 85: 66,
    86: 65, 87: 64, 88: 63, 89: 62, 90: 61, 91: 60, 92: 59, 93: 58,
    94: 113, 95: 112, 96: 111, 97: 110, 98: 109, 99: 108, 100: 107, 101: 106, 102: 105, 103: 104,
    104: 103, 105: 102, 106: 101, 107: 100, 108: 99, 109: 98, 110: 97, 111: 96
}

class HeadlessQAServer:
    def __init__(self):
        self.sim_size = 800
        self.binary_threshold = 180 
        print("Connecting to Arduino...")
        try:
            self.arduino = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)
            time.sleep(2) 
            print("SUCCESS: Arduino connected on ttyUSB0!")
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            self.arduino = None

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (self.sim_size, self.sim_size)})
        self.picam2.configure(config)
        self.picam2.start()

        self.dynamic_map = [] 
        self.calibrated_outer_map = []
        self.calibrated_inner_map = []
        self.qa_errors = [] 
        self.calibration_warning = "" 
        self.show_calibration_labels = False

    def send_to_arduino(self, command_string):
        if self.arduino:
            full_command = f"{command_string}\n"
            self.arduino.write(full_command.encode('utf-8'))
            time.sleep(0.1) 

    def set_geometric_state(self, geometric_indices):
        physical_indices = [str(CV_TO_PCB_MAP[i]) for i in geometric_indices]
        batch_str = ",".join(physical_indices)
        self.send_to_arduino(f"BATCH:{batch_str}")

    def grab_live_camera(self):
        try:
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return cv2.resize(frame, (self.sim_size, self.sim_size))
        except Exception:
            return np.zeros((self.sim_size, self.sim_size, 3), dtype=np.uint8)

    def render_overlays(self, frame):
        if self.show_calibration_labels:
            for i, (gx, gy) in enumerate(self.calibrated_outer_map):
                cv2.circle(frame, (gx, gy), 14, (0, 200, 0), 2)
                cv2.putText(frame, str(i), (gx - 10, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            for i, (gx, gy) in enumerate(self.calibrated_inner_map):
                cv2.circle(frame, (gx, gy), 14, (200, 100, 0), 2)
                label_idx = i + len(self.calibrated_outer_map)
                cv2.putText(frame, str(label_idx), (gx - 10, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if self.qa_errors:
            for error in self.qa_errors:
                if len(error) == 6 and error[1] != "UNMAPPED": 
                    cv_index, ex, ey, ax, ay, hole_idx = error
                    cv2.line(frame, (ex, ey), (ax, ay), (0, 0, 255), 3)
                    cv2.circle(frame, (ax, ay), 18, (0, 255, 255), 2) 
            cv2.putText(frame, "QA FAILED: CHECK ROUTING", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if self.calibration_warning:
            cv2.putText(frame, self.calibration_warning, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

        return frame

    def run_calibration(self):
        self.qa_errors = []
        self.calibration_warning = ""
        self.show_calibration_labels = False

        states = [
            ("Outer Evens", list(range(0, 58, 2))),
            ("Outer Odds", list(range(1, 58, 2))),
            ("Inner Evens", list(range(58, 112, 2))),
            ("Inner Odds", list(range(59, 112, 2)))
        ]

        merged_image = np.zeros((self.sim_size, self.sim_size, 3), dtype=np.uint8)

        for name, geometric_list in states:
            self.set_geometric_state(geometric_list)
            time.sleep(0.3) 
            frame = self.grab_live_camera()
            merged_image = cv2.bitwise_or(merged_image, frame)

        self.send_to_arduino("CLEAR")

        gray = cv2.cvtColor(merged_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_blobs = []
        for c in contours:
            if cv2.contourArea(c) > 15:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    raw_blobs.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        if len(raw_blobs) < 3:
            self.calibration_warning = f"ERROR: Mapped {len(raw_blobs)}/112. Too dim/unfocused!"
            return "Calibration Failed. Not enough light detected. Check focus and try again."

        web_log = []
        if len(raw_blobs) != 112:
            self.calibration_warning = f"WARNING: Mapped only {len(raw_blobs)}/112 fibers!"
            web_log.append(f"WARNING: Found {len(raw_blobs)} physical dots instead of 112.")

        global_cx = sum([b[0] for b in raw_blobs]) / len(raw_blobs)
        global_cy = sum([b[1] for b in raw_blobs]) / len(raw_blobs)
        radiuses = [math.sqrt((bx - global_cx)**2 + (by - global_cy)**2) for bx, by in raw_blobs]
        r_threshold = (max(radiuses) + min(radiuses)) / 2

        outer_blobs = [(bx, by) for (bx, by), r in zip(raw_blobs, radiuses) if r > r_threshold]
        inner_blobs = [(bx, by) for (bx, by), r in zip(raw_blobs, radiuses) if r <= r_threshold]

        def align_outer_ring(blobs):
            if len(blobs) < 3: return blobs, 0
            angles = [math.degrees(math.atan2(by - global_cy, bx - global_cx)) % 360 for bx, by in blobs]
            sorted_blobs = [b for _, b in sorted(zip(angles, blobs))]
            sorted_angles = sorted(angles)
            diffs = []
            for i in range(len(sorted_angles)):
                next_i = (i+1) % len(sorted_angles)
                gap = (sorted_angles[next_i] - sorted_angles[i]) % 360
                diffs.append((gap, next_i, sorted_angles[next_i]))
            largest_gap = max(diffs, key=lambda x: x[0])
            index_zero = largest_gap[1]
            return sorted_blobs[index_zero:] + sorted_blobs[:index_zero], largest_gap[2]

        def align_inner_ring(blobs, target_angle):
            if len(blobs) < 3: return blobs
            angles = [math.degrees(math.atan2(by - global_cy, bx - global_cx)) % 360 for bx, by in blobs]
            sorted_blobs = [b for _, b in sorted(zip(angles, blobs))]
            sorted_angles = sorted(angles)
            diffs = []
            for i in range(len(sorted_angles)):
                next_i = (i+1) % len(sorted_angles)
                gap = (sorted_angles[next_i] - sorted_angles[i]) % 360
                diffs.append((gap, next_i, sorted_angles[next_i]))
            diffs.sort(key=lambda x: x[0], reverse=True)
            top_3 = diffs[:min(3, len(diffs))]
            def angle_dist(a, target):
                d = abs(a - target) % 360
                return 360 - d if d > 180 else d
            true_gap = min(top_3, key=lambda x: angle_dist(x[2], target_angle))
            index_zero = true_gap[1]
            return sorted_blobs[index_zero:] + sorted_blobs[:index_zero]

        self.calibrated_outer_map, master_angle = align_outer_ring(outer_blobs)
        self.calibrated_inner_map = align_inner_ring(inner_blobs, master_angle)
        self.dynamic_map = self.calibrated_outer_map + self.calibrated_inner_map

        self.show_calibration_labels = True
        aiming_list = list(range(0, 58, 2)) + list(range(58, 112, 2))
        self.set_geometric_state(aiming_list)

        web_log.append(f"Jig Calibrated successfully. Mapped {len(self.dynamic_map)} fibers.")
        return "<br>".join(web_log)

    def run_auto_calibrate_and_sweep(self):
        self.qa_errors = []
        self.calibration_warning = ""
        self.show_calibration_labels = False

        states = [
            ("Outer Evens", list(range(0, 58, 2))),
            ("Outer Odds", list(range(1, 58, 2))),
            ("Inner Evens", list(range(58, 112, 2))),
            ("Inner Odds", list(range(59, 112, 2)))
        ]

        merged_image = np.zeros((self.sim_size, self.sim_size, 3), dtype=np.uint8)

        for name, geometric_list in states:
            self.set_geometric_state(geometric_list)
            time.sleep(0.3) 
            frame = self.grab_live_camera()
            merged_image = cv2.bitwise_or(merged_image, frame)

        self.send_to_arduino("CLEAR")

        gray = cv2.cvtColor(merged_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- THE UPGRADED "CENTER-FINDING" IN-MEMORY SWEEP ---
        # Test every threshold from 240 down to 40 in steps of 5
        search_space = list(range(240, 39, -5))
        working_thresholds = []
        blobs_at_threshold = {}

        for test_thresh in search_space:
            _, thresh = cv2.threshold(gray, test_thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_blobs = []
            for c in contours:
                if cv2.contourArea(c) > 15:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        current_blobs.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
            
            # Save the results for this threshold
            blobs_at_threshold[test_thresh] = current_blobs
            
            # If it perfectly sees 112 fibers, log it as a working candidate
            if len(current_blobs) == 112:
                working_thresholds.append(test_thresh)

        # Evaluate the results to find the safest center point
        if not working_thresholds:
            # It failed completely. Find the attempt that got closest to 112 for the error log.
            best_attempt_thresh = max(blobs_at_threshold.keys(), key=lambda t: len(blobs_at_threshold[t]))
            raw_blobs = blobs_at_threshold[best_attempt_thresh]
            self.calibration_warning = f"AUTO-TUNE FAILED: Best was {len(raw_blobs)}/112."
            return False, f"<span style='color:red;'>Auto-Tuning Failed. Could not find all 112 fibers at any setting. Please check physical alignment and camera focus.</span>", self.binary_threshold

        # Success! Calculate the exact middle of the working range
        safe_max = max(working_thresholds)
        safe_min = min(working_thresholds)
        best_threshold = int((safe_max + safe_min) / 2)
        
        raw_blobs = blobs_at_threshold[best_threshold] # Grab the specific blobs for the center threshold
        self.binary_threshold = best_threshold 
        # ---------------------------------------------------------

        global_cx = sum([b[0] for b in raw_blobs]) / len(raw_blobs)
        global_cy = sum([b[1] for b in raw_blobs]) / len(raw_blobs)
        radiuses = [math.sqrt((bx - global_cx)**2 + (by - global_cy)**2) for bx, by in raw_blobs]
        r_threshold = (max(radiuses) + min(radiuses)) / 2

        outer_blobs = [(bx, by) for (bx, by), r in zip(raw_blobs, radiuses) if r > r_threshold]
        inner_blobs = [(bx, by) for (bx, by), r in zip(raw_blobs, radiuses) if r <= r_threshold]

        def align_outer_ring(blobs):
            if len(blobs) < 3: return blobs, 0
            angles = [math.degrees(math.atan2(by - global_cy, bx - global_cx)) % 360 for bx, by in blobs]
            sorted_blobs = [b for _, b in sorted(zip(angles, blobs))]
            sorted_angles = sorted(angles)
            diffs = []
            for i in range(len(sorted_angles)):
                next_i = (i+1) % len(sorted_angles)
                gap = (sorted_angles[next_i] - sorted_angles[i]) % 360
                diffs.append((gap, next_i, sorted_angles[next_i]))
            largest_gap = max(diffs, key=lambda x: x[0])
            index_zero = largest_gap[1]
            return sorted_blobs[index_zero:] + sorted_blobs[:index_zero], largest_gap[2]

        def align_inner_ring(blobs, target_angle):
            if len(blobs) < 3: return blobs
            angles = [math.degrees(math.atan2(by - global_cy, bx - global_cx)) % 360 for bx, by in blobs]
            sorted_blobs = [b for _, b in sorted(zip(angles, blobs))]
            sorted_angles = sorted(angles)
            diffs = []
            for i in range(len(sorted_angles)):
                next_i = (i+1) % len(sorted_angles)
                gap = (sorted_angles[next_i] - sorted_angles[i]) % 360
                diffs.append((gap, next_i, sorted_angles[next_i]))
            diffs.sort(key=lambda x: x[0], reverse=True)
            top_3 = diffs[:min(3, len(diffs))]
            def angle_dist(a, target):
                d = abs(a - target) % 360
                return 360 - d if d > 180 else d
            true_gap = min(top_3, key=lambda x: angle_dist(x[2], target_angle))
            index_zero = true_gap[1]
            return sorted_blobs[index_zero:] + sorted_blobs[:index_zero]

        self.calibrated_outer_map, master_angle = align_outer_ring(outer_blobs)
        self.calibrated_inner_map = align_inner_ring(inner_blobs, master_angle)
        self.dynamic_map = self.calibrated_outer_map + self.calibrated_inner_map

        self.show_calibration_labels = True
        aiming_list = list(range(0, 58, 2)) + list(range(58, 112, 2))
        self.set_geometric_state(aiming_list)

        calib_msg = f"Auto-Tuned to Threshold {best_threshold} (Range: {safe_min}-{safe_max}). Mapped 112 fibers."
        sweep_msg = self.run_sweep()
        
        return True, f"<span style='color:cyan;'>{calib_msg}</span><br>{sweep_msg}", best_threshold

    def run_sweep(self):
        if not self.dynamic_map:
            return "ERROR: You must CALIBRATE JIG first!"

        self.qa_errors = []
        detailed_web_logs = [] 

        for cv_index in range(112):
            pcb_index = CV_TO_PCB_MAP[cv_index]
            self.send_to_arduino(f"LED:{pcb_index}")

            # --- ANTI-LAG OPTIMIZATION ---
            self.grab_live_camera() # 1. Flush the camera hardware buffer 
            
            frame = None
            valid_contours = []
            
            # 2. Smart Wait: Loop until we actually see the light turn on
            for attempt in range(4):
                frame = self.grab_live_camera()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = [c for c in contours if cv2.contourArea(c) > 15]
                
                # Found the dot! Break the loop instantly to run super fast.
                if valid_contours:
                    break 
            # ------------------------------

            if not valid_contours:
                self.qa_errors.append((cv_index, "DEAD", -1, -1, -1))
                error_msg = f"<span style='color:red;'>FAIL: CV Hole #{cv_index} is completely dead to the camera.</span>"
                detailed_web_logs.append(error_msg)
                continue

            largest_c = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_c)
            if M["m00"] == 0: continue

            actual_x = int(M["m10"] / M["m00"])
            actual_y = int(M["m01"] / M["m00"])

            closest_hole_index = -1
            closest_dist = float('inf')

            for hole_idx, (hx, hy) in enumerate(self.dynamic_map):
                dist = math.sqrt((actual_x - hx)**2 + (actual_y - hy)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_hole_index = hole_idx

            if closest_hole_index != cv_index:
                if cv_index < len(self.dynamic_map):
                    expected_x, expected_y = self.dynamic_map[cv_index]
                    self.qa_errors.append((cv_index, expected_x, expected_y, actual_x, actual_y, closest_hole_index))
                else:
                    self.qa_errors.append((cv_index, "UNMAPPED", -1, actual_x, actual_y, closest_hole_index))

                error_msg = f"<span style='color:orange;'>FAIL: CV Hole #{cv_index} is misrouted into hole #{closest_hole_index}.</span>"
                detailed_web_logs.append(error_msg)

        aiming_list = list(range(0, 58, 2)) + list(range(58, 112, 2))
        self.set_geometric_state(aiming_list)
        self.show_calibration_labels = True

        if not self.qa_errors:
            return "<span style='color:lime;'>UNIT PASSED: 100% Routing Accuracy.</span>"
        else:
            header = f"<span style='color:red; font-weight:bold;'>UNIT FAILED: Found {len(self.qa_errors)} routing errors.</span>"
            return header + "<br>" + "<br>".join(detailed_web_logs)

qa_engine = HeadlessQAServer()

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_stream():
    while True:
        frame = qa_engine.grab_live_camera()
        frame = qa_engine.render_overlays(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command/<cmd>')
def handle_command(cmd):
    msg = "Unknown command."
    if cmd == 'aiming':
        aiming_list = list(range(0, 58, 2)) + list(range(58, 112, 2))
        qa_engine.set_geometric_state(aiming_list)
        msg = "Aiming Mode (50% LEDs ON)."
    elif cmd == 'clear':
        qa_engine.show_calibration_labels = False
        qa_engine.qa_errors = []
        qa_engine.calibration_warning = ""
        qa_engine.send_to_arduino("CLEAR")
        msg = "Display cleared. All Lights OFF."
    elif cmd == 'calibrate':
        msg = qa_engine.run_calibration()
    elif cmd == 'qa_sweep':
        msg = qa_engine.run_sweep()
    return jsonify({"message": msg})

@app.route('/command/auto_sweep')
def auto_sweep():
    success, html_msg, new_thresh = qa_engine.run_auto_calibrate_and_sweep()
    return jsonify({
        "message": html_msg,
        "new_threshold": new_thresh,
        "success": success
    })

@app.route('/set_threshold/<int:val>')
def set_threshold(val):
    qa_engine.binary_threshold = val
    return jsonify({"message": f"CV Light Threshold updated to {val} / 255."})

@app.route('/command/restart_server')
def restart_server():
    def delayed_restart():
        time.sleep(1) 
        subprocess.Popen(["sudo", "systemctl", "restart", "qa_server.service"])

    threading.Thread(target=delayed_restart).start()
    return jsonify({"message": "REBOOTING BACKEND SERVICE..."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False, threaded=True)
