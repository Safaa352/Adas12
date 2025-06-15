import cv2
import numpy as np
import pygame
import time
import math
from collections import deque

# Initialize pygame for sound
pygame.init()
pygame.mixer.init()

# Generate beep sound
def generate_beep(freq=880, duration=0.1):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    buf = np.zeros((n_samples, 2), dtype=np.float32)
    for s in range(n_samples):
        t = float(s) / sample_rate
        buf[s][0] = math.sin(2 * math.pi * freq * t) * 0.5
        buf[s][1] = math.sin(2 * math.pi * freq * t) * 0.5
    return pygame.sndarray.make_sound(buf * 32767)

# Constants
WARNING_DISTANCE = 1.5  # meters (distance to start warning)
DANGER_DISTANCE = 0.8   # meters (distance to stop)
WARNING_COLOR = (0, 165, 255)  # Orange
DANGER_COLOR = (0, 0, 255)     # Red
SAFE_COLOR = (0, 255, 0)       # Green
PARKING_GUIDE_COLOR = (0, 255, 255)  # Yellow

class ParkingAssistant:
    def __init__(self):
        # Try to open camera (try multiple indices)
        self.cap = None
        for i in range(3):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            print("Error: Could not open camera. Using simulated feed.")
            self.simulated = True
            self.frame_size = (640, 480)
        else:
            self.simulated = False
            self.frame_size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        
        # Generate beep sound
        self.beep_sound = generate_beep()
        
        # Parking state
        self.parking_mode = False
        self.last_warning_time = 0
        self.distance_history = deque(maxlen=5)
        
        # Known object width (for distance estimation)
        self.known_width = 1.8  # meters (typical car width)
        self.focal_length = 700  # Needs calibration for your camera
        
        # Parking guide lines
        self.guide_lines = [
            (0.3, 0.7),  # Left line position (x ratios)
            (0.7, 0.7)   # Right line position
        ]

    def calculate_distance(self, width_in_pixels):
        """Estimate distance to object based on pixel width"""
        if width_in_pixels <= 0:
            return float('inf')
        return (self.known_width * self.focal_length) / width_in_pixels

    def detect_obstacles(self, frame):
        """Detect obstacles using simple contour detection"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for obstacles (dark colors)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Noise reduction
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate distance (using width as reference)
            distance = self.calculate_distance(w)
            
            # Only consider objects in the bottom half of the frame
            if y > frame.shape[0] * 0.4:
                detected_objects.append({
                    'position': (x, y, w, h),
                    'distance': distance
                })
        
        return detected_objects

    def generate_instructions(self, obstacles, frame_width):
        """Generate parking instructions based on obstacle positions"""
        if not self.parking_mode:
            return "Press 'P' to activate parking assist"
            
        if not obstacles:
            return "Proceed slowly - no obstacles detected"
            
        # Find closest obstacle
        closest = min(obstacles, key=lambda o: o['distance'])
        x, y, w, h = closest['position']
        center_x = x + w/2
        
        # Determine obstacle position relative to center
        if closest['distance'] < DANGER_DISTANCE:
            return "STOP! Too close to obstacle"
        elif closest['distance'] < WARNING_DISTANCE:
            if center_x < frame_width * 0.4:
                return "Turn right slightly - obstacle on left"
            elif center_x > frame_width * 0.6:
                return "Turn left slightly - obstacle on right"
            else:
                return "Caution! Obstacle directly behind"
        else:
            if center_x < frame_width * 0.4:
                return "Slight right - obstacle on left"
            elif center_x > frame_width * 0.6:
                return "Slight left - obstacle on right"
            else:
                return "Proceed slowly - obstacle centered"

    def visualize_display(self, frame, obstacles, instructions):
        """Create visual display with warnings and instructions"""
        # Create overlay
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Add status bar
        cv2.rectangle(overlay, (0, 0), (width, 70), (40, 40, 40), -1)
        
        # Add instructions
        cv2.putText(overlay, instructions, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add mode indicator
        mode_text = "PARKING ASSIST: ACTIVE" if self.parking_mode else "PARKING ASSIST: INACTIVE"
        mode_color = (0, 255, 0) if self.parking_mode else (0, 0, 255)
        cv2.putText(overlay, mode_text, (width - 350, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Draw parking guide lines
        if self.parking_mode:
            for line_pos in self.guide_lines:
                x1 = int(width * line_pos[0])
                x2 = int(width * line_pos[1])
                cv2.line(frame, (x1, 0), (x1, height), PARKING_GUIDE_COLOR, 2)
                cv2.line(frame, (x2, 0), (x2, height), PARKING_GUIDE_COLOR, 2)
            
            # Draw center line
            cv2.line(frame, (width//2, 0), (width//2, height), (255, 0, 255), 1)
            
            # Draw distance markers
            for i in range(1, 4):
                y_pos = height - int(height * (i/5))
                cv2.line(frame, (width//2 - 20, y_pos), (width//2 + 20, y_pos), (0, 255, 255), 2)
                cv2.putText(frame, f"{i}m", (width//2 + 30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Process obstacles
        min_distance = float('inf')
        for obstacle in obstacles:
            x, y, w, h = obstacle['position']
            distance = obstacle['distance']
            min_distance = min(min_distance, distance)
            
            # Determine color based on distance
            if distance < DANGER_DISTANCE:
                color = DANGER_COLOR
                label = f"DANGER: {distance:.1f}m"
            elif distance < WARNING_DISTANCE:
                color = WARNING_COLOR
                label = f"WARNING: {distance:.1f}m"
            else:
                color = SAFE_COLOR
                label = f"SAFE: {distance:.1f}m"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add distance bar at bottom
        for i, dist in enumerate(np.linspace(0, 3, 30)):
            color = DANGER_COLOR if dist < DANGER_DISTANCE else (
                WARNING_COLOR if dist < WARNING_DISTANCE else SAFE_COLOR)
            cv2.rectangle(frame, 
                         (int(i*width/30), height-20),
                         (int((i+1)*width/30), height),
                         color, -1)
        
        # Add distance labels
        cv2.putText(frame, "0m", (10, height-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"{DANGER_DISTANCE}m", 
                   (int(DANGER_DISTANCE/3*width), height-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"{WARNING_DISTANCE}m", 
                   (int(WARNING_DISTANCE/3*width), height-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "3m", (width-30, height-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Combine overlay with frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame, min_distance

    def sound_warning(self, min_distance):
        """Generate beep warnings based on distance"""
        if not self.parking_mode:
            return
            
        current_time = time.time()
        
        if min_distance < DANGER_DISTANCE:
            # Continuous beep for danger zone
            if not pygame.mixer.get_busy():
                self.beep_sound.play(-1)  # Continuous play
        elif min_distance < WARNING_DISTANCE:
            # Intermittent beeps for warning zone
            beep_interval = max(0.1, min_distance / 2)
            if current_time - self.last_warning_time > beep_interval:
                self.beep_sound.play()
                self.last_warning_time = current_time
        else:
            # Stop any playing beeps
            if pygame.mixer.get_busy():
                self.beep_sound.stop()

    def get_simulated_frame(self):
        """Generate simulated camera feed"""
        frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Add some random "obstacles"
        num_obstacles = np.random.choice([0, 1, 2])
        obstacles = []
        for _ in range(num_obstacles):
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 200)
            x = np.random.randint(0, self.frame_size[0]-w)
            y = np.random.randint(self.frame_size[1]//2, self.frame_size[1]-h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            obstacles.append({
                'position': (x, y, w, h),
                'distance': np.random.uniform(0.5, 3.0)
            })
        
        # Add guide lines
        for line_pos in self.guide_lines:
            x_pos = int(self.frame_size[0] * line_pos[0])
            cv2.line(frame, (x_pos, 0), (x_pos, self.frame_size[1]), PARKING_GUIDE_COLOR, 2)
        
        return frame, obstacles

    def run(self):
        while True:
            if self.simulated:
                frame, obstacles = self.get_simulated_frame()
                min_distance = min([o['distance'] for o in obstacles]) if obstacles else float('inf')
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)  # Mirror effect for rear camera
                obstacles = self.detect_obstacles(frame)
                min_distance = min([o['distance'] for o in obstacles]) if obstacles else float('inf')
            
            # Generate instructions
            instructions = self.generate_instructions(obstacles, frame.shape[1])
            
            # Visualize results
            display_frame, _ = self.visualize_display(frame, obstacles, instructions)
            
            # Sound warnings
            self.sound_warning(min_distance)
            
            # Show frame
            cv2.imshow('Parking Assist System', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.parking_mode = not self.parking_mode
                if not self.parking_mode and pygame.mixer.get_busy():
                    self.beep_sound.stop()
                print(f"Parking mode {'activated' if self.parking_mode else 'deactivated'}")
            elif key == ord('q'):
                break
            elif key == ord('a'):  # Move guide lines left
                self.guide_lines = [(max(0.1, p-0.05), max(0.1, q-0.05)) for p, q in self.guide_lines]
            elif key == ord('d'):  # Move guide lines right
                self.guide_lines = [(min(0.9, p+0.05), min(0.9, q+0.05)) for p, q in self.guide_lines]

        if not self.simulated:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    print("Starting Parking Assist System")
    print("Controls:")
    print("  'P' - Toggle parking assist mode")
    print("  'A' - Move guide lines left")
    print("  'D' - Move guide lines right")
    print("  'Q' - Quit program")
    
    assistant = ParkingAssistant()
    assistant.run()
