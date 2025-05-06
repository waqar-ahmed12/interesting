import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import math

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
        
    def find_hands(self, img, draw=True):
        """Process image and find hands"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        self.img_height, self.img_width, _ = img.shape
        
        if self.results.multi_hand_landmarks:
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
        
        return img
    
    def find_positions(self, img, hand_no=0):
        """Return list of landmark positions for specified hand"""
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
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
            return False
        
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
                
                # Return True if pinching (distance is small)
                return normalized_distance < 0.3
        
        return False
    
    def detect_all_fingers_pinch(self, hand_landmarks):
        """Detect pinch with all fingers to thumb"""
        if not hand_landmarks:
            return False
        
        # Get thumb tip (4) and other finger tips (8, 12, 16, 20)
        thumb_tip = None
        finger_tips = []
        
        for lm in hand_landmarks:
            if lm[0] == 4:  # Thumb tip
                thumb_tip = lm
            elif lm[0] in [8, 12, 16, 20]:  # Finger tips
                finger_tips.append(lm)
        
        if thumb_tip and len(finger_tips) == 4:  # We need all 4 finger tips
            # Calculate distance from thumb to each finger
            distances = [self.calc_distance(thumb_tip, tip) for tip in finger_tips]
            
            # Normalize by hand size
            wrist = None
            middle_mcp = None
            
            for lm in hand_landmarks:
                if lm[0] == 0:  # Wrist
                    wrist = lm
                elif lm[0] == 9:  # Middle finger MCP
                    middle_mcp = lm
            
            if wrist and middle_mcp:
                hand_size = self.calc_distance(wrist, middle_mcp)
                if hand_size > 0:
                    normalized_distances = [d / hand_size for d in distances]
                    
                    # Return True if all fingers are close to thumb
                    return all(d < 0.5 for d in normalized_distances)
        
        return False
    
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
    
    def detect_swipe(self, hand_landmarks):
        """Detect hand swipe gestures (left, right, up, down)"""
        if not hand_landmarks or len(hand_landmarks) == 0:
            # Reset tracking when no hand is detected
            self.prev_hand_center = None
            self.movement_buffer = []
            return None
        
        # Calculate hand center (average of all landmarks)
        cx_sum = cy_sum = 0
        for lm in hand_landmarks:
            cx_sum += lm[1]
            cy_sum += lm[2]
        
        cx_avg = cx_sum / len(hand_landmarks)
        cy_avg = cy_sum / len(hand_landmarks)
        current_center = (cx_avg, cy_avg)
        
        if self.prev_hand_center is None:
            self.prev_hand_center = current_center
            return None
        
        # Calculate movement delta
        dx = (current_center[0] - self.prev_hand_center[0]) / self.img_width
        dy = (current_center[1] - self.prev_hand_center[1]) / self.img_height
        
        # Update previous center
        self.prev_hand_center = current_center
        
        # Add to movement buffer
        self.movement_buffer.append((dx, dy))
        if len(self.movement_buffer) > 5:  # Keep last 5 movements
            self.movement_buffer.pop(0)
        
        # Check for consistent movement direction
        if len(self.movement_buffer) < 3:  # Need at least 3 samples
            return None
        
        # Calculate average movement
        avg_dx = sum(m[0] for m in self.movement_buffer) / len(self.movement_buffer)
        avg_dy = sum(m[1] for m in self.movement_buffer) / len(self.movement_buffer)
        
        # Check if movement exceeds threshold
        if abs(avg_dx) > self.swipe_threshold or abs(avg_dy) > self.swipe_threshold:
            # Determine primary direction
            if abs(avg_dx) > abs(avg_dy):
                if avg_dx > 0:
                    return "Swipe Right"
                else:
                    return "Swipe Left"
            else:
                if avg_dy > 0:
                    return "Swipe Down"
                else:
                    return "Swipe Up"
        
        return None
    
    def recognize_gesture(self, img):
        """Recognize hand gestures"""
        landmark_list = self.find_positions(img)
        
        # Detect gestures
        gesture = None
        
        # Swipe detection
        swipe = self.detect_swipe(landmark_list)
        if swipe:
            gesture = swipe
            # Reset movement buffer after a swipe is detected
            self.movement_buffer = []
        
        # Check for thumb-index pinch
        elif self.detect_thumb_index_pinch(landmark_list):
            gesture = "Thumb-Index Pinch"
        
        # Check for all fingers pinch
        elif self.detect_all_fingers_pinch(landmark_list):
            gesture = "All Fingers Pinch"
        
        # Check for fist vs open hand
        elif self.detect_fist(landmark_list):
            gesture = "Fist Closed"
        elif self.detect_open_hand(landmark_list):
            gesture = "Hand Open"
        
        # Apply cooldown to avoid rapid gesture changes
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
        elif gesture and gesture != self.last_gesture:
            self.last_gesture = gesture
            self.gesture_cooldown = 10  # Adjust cooldown frames as needed
            # Add to gesture history
            self.gesture_history.append((gesture, time.time()))
            if len(self.gesture_history) > 10:
                self.gesture_history.pop(0)
            return gesture
        
        return None

def process_video(source=0, output=None):
    """Process video source (0=webcam, or file path)"""
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output specified
    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
    
    # Initialize hand recognizer
    recognizer = HandGestureRecognizer()
    
    # Performance metrics
    prev_time = 0
    
    # Recent gestures display
    recent_gestures = []
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("End of video stream.")
            break
        
        # Process frame and find hands
        img = recognizer.find_hands(img)
        
        # Recognize gestures
        gesture = recognizer.recognize_gesture(img)
        if gesture:
            # Add to recent gestures with timestamp
            recent_gestures.append((gesture, time.time()))
            # Keep only recent gestures (last 3 seconds)
            recent_gestures = [g for g in recent_gestures if time.time() - g[1] < 3]
        
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
        
        # Write to output file if specified
        if writer:
            writer.write(img)
        
        # Display result
        cv2.imshow("Hand Gesture Recognition", img)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand Gesture Recognition')
    parser.add_argument('--source', default=0, help='Video source (0 for webcam, or file path)')
    parser.add_argument('--output', default=None, help='Output video file path (optional)')
    
    args = parser.parse_args()
    
    try:
        # Convert source to int if it's a number (for webcam)
        source = int(args.source) if args.source.isdigit() else args.source
        process_video(source, args.output)
    except Exception as e:
        print(f"Error: {e}")