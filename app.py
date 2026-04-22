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
# SINGLE SOURCE OF TRUTH: MAPPING DICTIONARIES
# ==========================================
CV_TO_PCB_MAP = {
    # --- OUTER RING (CV 0 to 57) ---
    0: 0,
    1: 93, 2: 92, 3: 91, 4: 90, 5: 89, 6: 88, 7: 87, 8: 86, 9: 85, 
    10: 84, 11: 83, 12: 82, 13: 81, 14: 80, 15: 79, 16: 78, 17: 77, 18: 76,
    19: 57, 20: 56, 21: 55, 22: 54, 23: 53, 24: 52, 25: 51, 26: 50, 27: 49, 
    28: 48, 29: 47, 30: 46, 31: 45, 32: 44, 33: 43, 34: 42, 35: 41, 36: 40, 37: 39, 38: 38,
    39: 19, 40: 18, 41: 17, 42: 16, 43: 15, 44: 14, 45: 13, 46: 12, 47: 11, 
    48: 10, 49: 9, 50: 8, 51: 7, 52: 6, 53: 5, 54: 4, 55: 3, 56: 2, 57: 1,

    # --- INNER RING (CV 58 to 111) ---
    58: 96, 59: 97, 60: 98, 61: 99, 62: 100, 63: 101, 64: 102, 65: 103, 66: 104, 
    67: 105, 68: 106, 69: 107, 70: 108, 71: 109, 72: 110, 73: 111, 74: 112, 75: 113,
    76: 58, 77: 59, 78: 60, 79: 61, 80: 62, 81: 63, 82: 64, 83: 65, 84: 66, 
    85: 67, 86: 68, 87: 69, 88: 70, 89: 71, 90: 72, 91: 73, 92: 74, 93: 75,
    94: 20, 95: 21, 96: 22, 97: 23, 98: 24, 99: 25, 100: 26, 101: 27, 102: 28, 
    103: 29, 104: 30, 105: 31, 106: 32, 107: 33, 108: 34, 109: 35, 110: 36, 111: 37
}

LEFT_TO_CV_MAP = {
    0: 0, 1: 58, 2: 1, 3: 59, 4: 2, 5: 60, 6: 3, 7: 61, 8: 4, 9: 62,
    10: 5, 11: 63, 12: 6, 13: 64, 14: 7, 15: 65, 16: 8, 17: 66, 18: 9, 19: 67,
    20: 10, 21: 68, 22: 11, 23: 69, 24: 12, 25: 70, 26: 13, 27: 71, 28: 14, 29: 72,
    30: 15, 31: 73, 32: 16, 33: 74, 34: 17, 35: 75, 36: 18, 37: 19, 38: 20, 39: 76,
    40: 21, 41: 77, 42: 22, 43: 78, 44: 23, 45: 79, 46: 24, 47: 80, 48: 25, 49: 81,
    50: 26, 51: 82, 52: 27, 53: 83, 54: 28, 55: 84, 56: 29, 57: 85, 58: 30, 59: 86,
    60: 31, 61: 87, 62: 32, 63: 88, 64: 33, 65: 89, 66: 34, 67: 90, 68: 35, 69: 91,
    70: 36, 71: 92, 72: 37, 73: 93, 74: 38, 75: 39, 76: 40, 77: 94, 78: 41, 79: 95,
    80: 42, 81: 96, 82: 43, 83: 97, 84: 44, 85: 98, 86: 45, 87: 99, 88: 46, 89: 100,
    90: 47, 91: 101, 92: 48, 93: 102, 94: 49, 95: 103, 96: 50, 97: 104, 98: 51, 99: 105,
    100: 52, 101: 106, 102: 53, 103: 107, 104: 54, 105: 108, 106: 55, 107: 109, 108: 56, 109: 110,
    110: 57, 111: 111
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
            
            blobs_at_threshold[test_thresh] = current_blobs
            
            if len(current_blobs) == 112:
                working_thresholds.append(test_thresh)

        if not working_thresholds:
            best_attempt_thresh = max(blobs_at_threshold.keys(), key=lambda t: len(blobs_at_threshold[t]))
            raw_blobs = blobs_at_threshold[best_attempt_thresh]
            
            self.calibrated_outer_map = raw_blobs 
            self.calibrated_inner_map = [] 
            self.dynamic_map = raw_blobs
            self.show_calibration_labels = True
            
            aiming_list = list(range(0, 58, 2)) + list(range(58, 112, 2))
            self.set_geometric_state(aiming_list)
            
            self.calibration_warning = f"AUTO-TUNE FAILED: Best was {len(raw_blobs)}/112."
            return False, f"<span style='color:orange;'>Auto-Tuning Failed. Best threshold ({best_attempt_thresh}) found {len(raw_blobs)} fibers. Showing partial map on video feed so you can locate the dead fiber.</span>", best_attempt_thresh

        # --- LOWEST THRESHOLD SELECTION ---
        safe_max = max(working_thresholds)
        safe_min = min(working_thresholds)
        
        best_threshold = safe_min
        
        raw_blobs = blobs_at_threshold[best_threshold] 
        self.binary_threshold = int(best_threshold) 

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

        calib_msg = f"Auto-Tuned to Lowest Threshold {best_threshold} (Max safe was {safe_max}). Mapped 112 fibers."
        sweep_msg = self.run_sweep()
        
        return True, f"<span style='color:cyan;'>{calib_msg}</span><br>{sweep_msg}", int(best_threshold)

    def run_sweep(self):
        if not self.dynamic_map:
            return "ERROR: You must CALIBRATE JIG first!"

        self.qa_errors = []
        detailed_web_logs = [] 

        for cv_index in range(112):
            pcb_index = CV_TO_PCB_MAP[cv_index]
            self.send_to_arduino(f"LED:{pcb_index}")

            self.grab_live_camera() # Flush the camera hardware buffer 
            
            frame = None
            valid_contours = []
            
            for attempt in range(4):
                frame = self.grab_live_camera()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = [c for c in contours if cv2.contourArea(c) > 15]
                
                if valid_contours:
                    break 

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

# --- UPDATED MANUAL OVERRIDE (Now takes Left Index) ---
@app.route('/command/manual/<cmd_val>')
def manual_override(cmd_val):
    if cmd_val.lower() == 'clear':
        qa_engine.send_to_arduino("CLEAR")
        return jsonify({"message": "<span style='color:var(--accent-blue);'>[MANUAL] All LEDs cleared.</span>"})
    try:
        left_index = int(cmd_val)
        if 0 <= left_index <= 111:
            # Shield the user from the internal array math: Left -> CV -> PCB
            cv_index = LEFT_TO_CV_MAP[left_index]
            pcb_index = CV_TO_PCB_MAP[cv_index]
            qa_engine.send_to_arduino(f"LED:{pcb_index}")
            return jsonify({"message": f"<span style='color:var(--accent-blue);'>[MANUAL] True Left #{left_index} (CV: {cv_index}, PCB: {pcb_index}) powered ON.</span>"})
        else:
            return jsonify({"message": "<span style='color:red;'>[ERROR] Index must be between 0 and 111.</span>"})
    except ValueError:
        return jsonify({"message": "<span style='color:red;'>[ERROR] Invalid command. Enter a number 0-111 or 'clear'.</span>"})

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
