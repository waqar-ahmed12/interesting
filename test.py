import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import math
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import webbrowser
import os

class GestureServer(socketserver.TCPServer):
    """Custom server that allows address reuse"""
    allow_reuse_address = True

class GestureHandler(SimpleHTTPRequestHandler):
    """Handler for serving the webpage and handling gesture data"""
    
    # Class variable to store latest gesture data
    latest_gesture = {"gesture": "None", "data": {}}
    
    def do_GET(self):
        if self.path == '/gestures':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(self.latest_gesture).encode())
        else:
            # Serve files from the current directory
            return SimpleHTTPRequestHandler.do_GET(self)
    
    @classmethod
    def set_gesture(cls, gesture_name, data=None):
        """Set the latest gesture data"""
        if data is None:
            data = {}
        cls.latest_gesture = {"gesture": gesture_name, "data": data}

class HandGestureRecognizer:
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Tracking history for each hand
        self.tracking_history = [[] for _ in range(max_hands)]
        
        # Gesture states
        self.gesture_history = []
        self.last_gesture = "None"
        self.gesture_cooldown = 0
        
        # Movement tracking
        self.prev_hand_center = None
        self.movement_buffer = []
        self.swipe_threshold = 0.15  # Percentage of screen width/height
        
        # Hand identification
        self.left_hand_id = None
        self.right_hand_id = None
        
        # Gesture tracking
        self.palm_center_history = []  # For pan gesture
        self.pinch_distance_history = {}  # For zoom gestures
        self.fist_state = False  # For center network gesture
        self.index_positions = []  # For rotation gesture
        
    def find_hands(self, img, draw=True):
        """Process image and find hands"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        self.img_height, self.img_width, _ = img.shape
        
        if self.results.multi_hand_landmarks:
            # Reset hand IDs if we have new detection
            if len(self.results.multi_handedness) > 0:
                self.left_hand_id = None
                self.right_hand_id = None
                
                # Identify left and right hands
                for idx, hand_info in enumerate(self.results.multi_handedness):
                    hand_type = hand_info.classification[0].label
                    if hand_type == "Left":
                        self.right_hand_id = idx  # Inverted because camera image is mirrored
                    else:
                        self.left_hand_id = idx   # Inverted because camera image is mirrored
            
            for hand_id, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                if draw:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Track wrist movement (landmark 0)
                    if hand_id < len(self.tracking_history):
                        wrist = hand_landmarks.landmark[0]
                        cx, cy = int(wrist.x * self.img_width), int(wrist.y * self.img_height)
                        
                        # Store tracking points (limit to 20 recent points)
                        self.tracking_history[hand_id].append((cx, cy))
                        if len(self.tracking_history[hand_id]) > 20:
                            self.tracking_history[hand_id].pop(0)
                        
                        # Draw tracking path
                        for i in range(1, len(self.tracking_history[hand_id])):
                            if i > 0:
                                cv2.line(img, self.tracking_history[hand_id][i-1], 
                                        self.tracking_history[hand_id][i], (0, 255, 0), 2)
                
                # Add text to identify right/left hand
                hand_type = "Unknown"
                if hand_id == self.left_hand_id:
                    hand_type = "Left Hand"
                elif hand_id == self.right_hand_id:
                    hand_type = "Right Hand"
                
                # Draw hand label
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * self.img_width), int(wrist.y * self.img_height)
                cv2.putText(img, hand_type, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return img
    
    def find_positions(self, img, hand_no=0):
        """Return list of landmark positions for specified hand"""
        landmark_list = []
        
        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * self.img_width), int(lm.y * self.img_height)
                    landmark_list.append((id, cx, cy, lm.z))
    
        return landmark_list
    
    def calc_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    
    def detect_thumb_index_pinch(self, hand_landmarks):
        """Detect pinch between thumb and index finger"""
        if not hand_landmarks:
            return False, 0
        
        # Get thumb tip (4) and index finger tip (8)
        thumb_tip = None
        index_tip = None
        
        for lm in hand_landmarks:
            if lm[0] == 4:  # Thumb tip
                thumb_tip = lm
            elif lm[0] == 8:  # Index tip
                index_tip = lm
        
        if thumb_tip and index_tip:
            # Calculate distance between thumb and index tips
            distance = self.calc_distance(thumb_tip, index_tip)
            
            # Normalize by hand size (distance between wrist and middle finger MCP)
            wrist = None
            middle_mcp = None
            
            for lm in hand_landmarks:
                if lm[0] == 0:  # Wrist
                    wrist = lm
                elif lm[0] == 9:  # Middle finger MCP
                    middle_mcp = lm
            
            if wrist and middle_mcp:
                hand_size = self.calc_distance(wrist, middle_mcp)
                normalized_distance = distance / hand_size if hand_size > 0 else 0
                
                # Return True if pinching (distance is small) and the normalized distance
                return normalized_distance < 0.3, normalized_distance
        
        return False, 0
    
    def detect_palm_position(self, hand_landmarks):
        """Detect palm center position for pan gestures"""
        if not hand_landmarks:
            return None
        
        # Calculate palm center using wrist and finger MCPs
        palm_points = []
        for lm in hand_landmarks:
            if lm[0] in [0, 5, 9, 13, 17]:  # Wrist and finger MCPs
                palm_points.append((lm[1], lm[2]))
        
        if palm_points:
            # Calculate average position
            cx = sum(p[0] for p in palm_points) / len(palm_points)
            cy = sum(p[1] for p in palm_points) / len(palm_points)
            return (cx, cy)
        
        return None
    
    def detect_index_finger_position(self, hand_landmarks):
        """Detect index finger tip position for rotation control"""
        if not hand_landmarks:
            return None
        
        # Get index finger tip (8)
        for lm in hand_landmarks:
            if lm[0] == 8:  # Index tip
                return (lm[1], lm[2])
        
        return None
    
    def detect_fist(self, hand_landmarks):
        """Detect if hand is in a fist position"""
        if not hand_landmarks:
            return False
            
        # Get finger tips and middle joints
        finger_tips = []  # 8, 12, 16, 20
        pip_joints = []   # 6, 10, 14, 18 (Proximal Interphalangeal joints)
        
        for lm in hand_landmarks:
            if lm[0] in [8, 12, 16, 20]:  # Finger tips
                finger_tips.append(lm)
            elif lm[0] in [6, 10, 14, 18]:  # PIP joints
                pip_joints.append(lm)
        
        # Sort by landmark ID
        finger_tips.sort(key=lambda x: x[0])
        pip_joints.sort(key=lambda x: x[0])
        
        # Check if all fingertips are below their PIP joints (fingers are curled)
        if len(finger_tips) == 4 and len(pip_joints) == 4:
            # For a fist, all fingertips should be below their PIP joints
            # Note: Y coordinates increase downward in image space
            all_fingers_curled = all(tip[2] > pip[2] for tip, pip in zip(finger_tips, pip_joints))
            
            # Also check thumb position (tucked in for a fist)
            thumb_tip = None
            thumb_mcp = None
            
            for lm in hand_landmarks:
                if lm[0] == 4:  # Thumb tip
                    thumb_tip = lm
                elif lm[0] == 2:  # Thumb MCP
                    thumb_mcp = lm
            
            thumb_curled = False
            if thumb_tip and thumb_mcp:
                # For a fist, thumb should be curled inward
                thumb_curled = thumb_tip[1] > thumb_mcp[1]  # X coordinate is greater (more to the right/inside)
            
            return all_fingers_curled and thumb_curled
        
        return False
    
    def detect_open_hand(self, hand_landmarks):
        """Detect if hand is open (fingers extended)"""
        if not hand_landmarks:
            return False
            
        # Get finger tips and middle joints
        finger_tips = []  # 8, 12, 16, 20
        pip_joints = []   # 6, 10, 14, 18
        
        for lm in hand_landmarks:
            if lm[0] in [8, 12, 16, 20]:  # Finger tips
                finger_tips.append(lm)
            elif lm[0] in [6, 10, 14, 18]:  # PIP joints
                pip_joints.append(lm)
        
        # Sort by landmark ID
        finger_tips.sort(key=lambda x: x[0])
        pip_joints.sort(key=lambda x: x[0])
        
        # Check if all fingertips are above their PIP joints (fingers are extended)
        if len(finger_tips) == 4 and len(pip_joints) == 4:
            all_fingers_extended = all(tip[2] < pip[2] for tip, pip in zip(finger_tips, pip_joints))
            
            # Also check thumb position (extended for open hand)
            thumb_tip = None
            wrist = None
            
            for lm in hand_landmarks:
                if lm[0] == 4:  # Thumb tip
                    thumb_tip = lm
                elif lm[0] == 0:  # Wrist
                    wrist = lm
            
            thumb_extended = False
            if thumb_tip and wrist:
                # For an open hand, thumb should be extended away from wrist
                thumb_wrist_dist = self.calc_distance(thumb_tip, wrist)
                
                # Get middle finger MCP for hand size normalization
                middle_mcp = None
                for lm in hand_landmarks:
                    if lm[0] == 9:
                        middle_mcp = lm
                        break
                
                if middle_mcp:
                    hand_size = self.calc_distance(wrist, middle_mcp)
                    if hand_size > 0:
                        normalized_thumb_dist = thumb_wrist_dist / hand_size
                        thumb_extended = normalized_thumb_dist > 1.2  # Threshold for extended thumb
            
            return all_fingers_extended and thumb_extended
        
        return False
    
    def detect_pinch_zoom(self):
        """Detect pinch-to-zoom gestures with both hands"""
        # Need both hands detected
        if self.left_hand_id is None or self.right_hand_id is None:
            return None, {}
        
        # Get positions for both hands
        left_positions = self.find_positions(None, self.left_hand_id)
        right_positions = self.find_positions(None, self.right_hand_id)
        
        # Check for thumb-index pinch on both hands
        left_pinch, left_dist = self.detect_thumb_index_pinch(left_positions)
        right_pinch, right_dist = self.detect_thumb_index_pinch(right_positions)
        
        # Update pinch distance history
        if len(self.pinch_distance_history.get('left', [])) > 10:
            self.pinch_distance_history['left'] = self.pinch_distance_history['left'][-10:]
        if len(self.pinch_distance_history.get('right', [])) > 10:
            self.pinch_distance_history['right'] = self.pinch_distance_history['right'][-10:]
            
        # Store current pinch distances
        if 'left' not in self.pinch_distance_history:
            self.pinch_distance_history['left'] = []
        if 'right' not in self.pinch_distance_history:
            self.pinch_distance_history['right'] = []
            
        if left_pinch:
            self.pinch_distance_history['left'].append(left_dist)
        if right_pinch:
            self.pinch_distance_history['right'].append(right_dist)
        
        # Detect zoom-in (left hand pinch decreasing)
        if left_pinch and len(self.pinch_distance_history['left']) > 3:
            # Check if distance is decreasing (pinching in)
            if self.pinch_distance_history['left'][-1] < self.pinch_distance_history['left'][-3] - 0.02:
                return "Zoom In", {"amount": 0.05}
        
        # Detect zoom-out (right hand pinch decreasing)
        if right_pinch and len(self.pinch_distance_history['right']) > 3:
            # Check if distance is decreasing (pinching in)
            if self.pinch_distance_history['right'][-1] < self.pinch_distance_history['right'][-3] - 0.02:
                return "Zoom Out", {"amount": 0.05}
                
        return None, {}
    
    def detect_palm_pan(self):
        """Detect palm position for pan gesture"""
        # Use left hand for panning
        if self.left_hand_id is not None:
            left_positions = self.find_positions(None, self.left_hand_id)
            palm_pos = self.detect_palm_position(left_positions)
            
            if palm_pos:
                # Track palm movement
                self.palm_center_history.append(palm_pos)
                if len(self.palm_center_history) > 10:
                    self.palm_center_history.pop(0)
                
                # Need enough history to detect movement
                if len(self.palm_center_history) > 5:
                    # Calculate movement delta from 5 frames ago
                    dx = (palm_pos[0] - self.palm_center_history[-5][0]) / self.img_width
                    dy = (palm_pos[1] - self.palm_center_history[-5][1]) / self.img_height
                    
                    # If significant movement detected
                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        return "Pan", {"dx": dx * 100, "dy": dy * 100}
        
        return None, {}
    
    def detect_index_rotation(self):
        """Detect index finger movement for rotation"""
        # Use right hand for rotation control
        if self.right_hand_id is not None:
            right_positions = self.find_positions(None, self.right_hand_id)
            index_pos = self.detect_index_finger_position(right_positions)
            
            if index_pos:
                # Track index finger movement
                self.index_positions.append(index_pos)
                if len(self.index_positions) > 10:
                    self.index_positions.pop(0)
                
                # Need enough history to detect movement
                if len(self.index_positions) > 5:
                    # Calculate movement delta from 5 frames ago
                    dx = (index_pos[0] - self.index_positions[-5][0]) / self.img_width
                    dy = (index_pos[1] - self.index_positions[-5][1]) / self.img_height
                    
                    # If significant movement detected
                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        return "Rotate", {"dx": dx * 5, "dy": dy * 5}
        
        return None, {}
    
    def detect_fist_to_open(self):
        """Detect transition from fist to open hand (for centering network)"""
        if self.right_hand_id is not None:
            right_positions = self.find_positions(None, self.right_hand_id)
            
            # Check current state
            is_fist = self.detect_fist(right_positions)
            is_open = self.detect_open_hand(right_positions)
            
            # Detect transition from fist to open
            if self.fist_state and is_open:
                self.fist_state = False
                return "Reset View", {}
            
            # Update fist state
            self.fist_state = is_fist
        
        return None, {}
    
    def recognize_gesture(self, img):
        """Recognize hand gestures for network control"""
        # Default to no gesture for this frame
        current_gesture = None
        
        # Process the current image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        # If no hands detected, return None
        if not self.results.multi_hand_landmarks:
            # Report no gesture detected
            GestureHandler.set_gesture("None", {})
            return None
        
        # Check for fist to open transition (center view)
        reset_gesture, reset_data = self.detect_fist_to_open()
        if reset_gesture:
            GestureHandler.set_gesture(reset_gesture, reset_data)
            return reset_gesture
        
        # Check for pinch zoom gestures
        zoom_gesture, zoom_data = self.detect_pinch_zoom()
        if zoom_gesture:
            GestureHandler.set_gesture(zoom_gesture, zoom_data)
            return zoom_gesture
        
        # Check for palm pan gesture
        pan_gesture, pan_data = self.detect_palm_pan()
        if pan_gesture:
            GestureHandler.set_gesture(pan_gesture, pan_data)
            return pan_gesture
        
        # Check for index finger rotation
        rotate_gesture, rotate_data = self.detect_index_rotation()
        if rotate_gesture:
            GestureHandler.set_gesture(rotate_gesture, rotate_data)
            return rotate_gesture
        
        # No specific gesture detected
        GestureHandler.set_gesture("None", {})
        return None

def start_server(port=8000):
    """Start HTTP server for gesture communication"""
    server_address = ('', port)
    httpd = GestureServer(server_address, GestureHandler)
    print(f"Starting gesture server on port {port}...")
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return httpd

def process_video(source=0, port=8000):
    """Process video source (0=webcam, or file path)"""
    # Start HTTP server
    httpd = start_server(port)
    
    # Copy the HTML file to the serving directory (create if needed)
    target_html = "network_visualization.html"
    source_html = "networkWithLessSensTransform.html"
    
    # Copy the HTML file if the names don't match
    if source_html != target_html and os.path.exists(source_html):
        print(f"Copying {source_html} to {target_html}...")
        with open(source_html, "r") as src:
            with open(target_html, "w") as dst:
                dst.write(src.read())
    
    # Open the HTML file in the default browser
    webbrowser.open(f'http://localhost:{port}/test.html')
    
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        httpd.shutdown()
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle invalid fps
    if fps <= 0 or math.isnan(fps):
        fps = 30  # Default to 30fps if not available
    
    print(f"Video opened: {width}x{height} at {fps}fps")
    
    # Initialize hand recognizer
    recognizer = HandGestureRecognizer()
    
    # Performance metrics
    prev_time = 0
    
    # Recent gestures display
    recent_gestures = []
    
    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("End of video stream or frame read error.")
                break
            
            # Handle resize if image is too large
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
                width, height = new_width, new_height
            
            # Store image dimensions for landmark calculations
            recognizer.img_width, recognizer.img_height = width, height
            
            # Process frame and find hands
            img = recognizer.find_hands(img)
            
            # Recognize gestures
            gesture = recognizer.recognize_gesture(img)
            if gesture:
                # Add to recent gestures with timestamp
                recent_gestures.append((gesture, time.time()))
                # Keep only recent gestures (last 3 seconds)
                recent_gestures = [g for g in recent_gestures if time.time() - g[1] < 3]
                recognizer.last_gesture = gesture
            
            # Display current gesture state
            cv2.putText(img, f"Current: {recognizer.last_gesture}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display recent gestures
            y_pos = 70
            cv2.putText(img, "Recent Gestures:", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for idx, (g, t) in enumerate(reversed(recent_gestures[-5:])):  # Show last 5 gestures
                y_pos += 30
                seconds_ago = round(time.time() - t, 1)
                cv2.putText(img, f"{g} ({seconds_ago}s ago)", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Calculate and display FPS
            current_time = time.time()
            fps_val = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            cv2.putText(img, f"FPS: {int(fps_val)}", 
                       (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display help text
            cv2.putText(img, "Left Hand: Pan with palm, Zoom In with pinch", 
                      (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img, "Right Hand: Rotate with index finger, Zoom Out with pinch, Fist->Open to reset view", 
                      (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display result
            cv2.imshow("Hand Gesture Network Control", img)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        httpd.shutdown()



def choose_input_source():
    """Prompt user to choose input source"""
    print("\n=== Hand Gesture Network Control ===")
    print("1: Use webcam")
    print("2: Use video file")
    choice = input("Choose input source (1/2): ")
    
    if choice == "1":
        return 0  # Default webcam
    elif choice == "2":
        video_path = input("Enter video file path: ")
        return video_path
    else:
        print("Invalid choice. Using webcam.")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand Gesture Network Control')
    parser.add_argument('--source', default=None, help='Video source (0 for webcam, or file path)')
    parser.add_argument('--port', type=int, default=8000, help='Port for the HTTP server')
    
    args = parser.parse_args()
      # Uncomment and run this alone to test
    try:

        source = args.source
        if source is None:
            source = choose_input_source()
        
        # Convert source to int if it's a number (for webcam)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        process_video(source, args.port)
    except Exception as e:
        print(f"Error: {e}")