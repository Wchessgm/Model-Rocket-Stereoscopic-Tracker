import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import math
import os

class RocketTracker:
    def __init__(self, baseline_m, fov_degrees_x, fov_degrees_y, hsv_lower, hsv_upper):
        self.baseline = baseline_m
        self.fov_x = math.radians(fov_degrees_x)
        self.fov_y = math.radians(fov_degrees_y)
        
        # Custom tracking colors passed from user configuration
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        
    def extract_trajectory(self, video_path, window_name="Tracking"):
        print(f"Processing {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        trajectory = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to HSV and threshold based on custom limits
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            
            # Reduce background noise
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Find contours of the target color
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroid = (np.nan, np.nan)
            if contours:
                # Target the largest matching color blob
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 10:  # Noise threshold filter
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroid = (cx, cy)
                        cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            
            trajectory.append(centroid)
            
            # Display tracking masks for debugging
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Handle temporary dropouts via linear interpolation
        df = pd.DataFrame(trajectory, columns=['x', 'y'])
        df = df.interpolate(method='linear', limit_direction='both')
        
        return df.to_numpy(), fps, width, height

    def synchronize_videos(self, traj_a, traj_b):
        print("Synchronizing videos via launch motion detection...")
        
        def find_launch_frame(traj):
            y_coords = traj[:, 1]
            for i in range(5, len(y_coords)):
                # Detect initial upward movement (Y decreases in pixel coordinates)
                if y_coords[i-5] - y_coords[i] > 5:
                    return i
            return 0
            
        launch_a = find_launch_frame(traj_a)
        launch_b = find_launch_frame(traj_b)
        
        print(f"Launch found at frame {launch_a} (Vid A) and frame {launch_b} (Vid B).")
        
        sync_a = traj_a[launch_a:]
        sync_b = traj_b[launch_b:]
        
        min_len = min(len(sync_a), len(sync_b))
        return sync_a[:min_len], sync_b[:min_len]

    def triangulate(self, pts_a, pts_b, width, height):
        print("Calculating stereoscopic triangulation...")
        altitudes = []
        
        for (x_a, y_a), (x_b, y_b) in zip(pts_a, pts_b):
            # Map pixel points to relative angular views
            angle_x_a = (x_a - width/2) * (self.fov_x / width)
            angle_x_b = (x_b - width/2) * (self.fov_x / width)
            angle_y_a = (height/2 - y_a) * (self.fov_y / height)
            angle_y_b = (height/2 - y_b) * (self.fov_y / height)
            
            try:
                tan_a = math.tan(angle_x_a)
                tan_b = math.tan(angle_x_b)
                
                # Check for parallel division boundaries
                if abs(tan_a - tan_b) < 1e-5:
                    depth = np.nan
                else:
                    # Triangulation depth formula: Z = B / (tan(theta_a) - tan(theta_b))
                    depth = self.baseline / (tan_a - tan_b)
                
                # Derive altitude from both vantage points and average them
                alt_a = depth * math.tan(angle_y_a)
                alt_b = depth * math.tan(angle_y_b)
                altitude = (alt_a + alt_b) / 2.0
                
                # Weed out wild math anomalies/outliers
                if altitude < -10 or altitude > 300:
                    altitude = np.nan
                    
            except Exception:
                altitude = np.nan
                
            altitudes.append(altitude)
            
        alt_series = pd.Series(altitudes).interpolate(method='linear')
        return alt_series.to_numpy()

    def analyze_flight(self, altitudes, fps):
        print("Processing kinematic data...")
        time = np.arange(len(altitudes)) / fps
        
        # Smooth positioning data with a Savitzky-Golay filter
        window = min(31, len(altitudes) if len(altitudes) % 2 != 0 else len(altitudes) - 1)
        smooth_alt = savgol_filter(altitudes, window, 3) if window > 3 else altitudes
        
        # Zero baseline altitude reference
        smooth_alt = smooth_alt - smooth_alt[0]
        
        # First derivative: Velocity
        dt = 1.0 / fps
        velocity = np.gradient(smooth_alt, dt)
        velocity = savgol_filter(velocity, window, 3) if window > 3 else velocity
        
        # Second derivative: Acceleration
        acceleration = np.gradient(velocity, dt)
        
        metrics = {
            "max_altitude_m": np.max(smooth_alt),
            "time_to_apogee_s": time[np.argmax(smooth_alt)],
            "flight_duration_s": time[-1]
        }
        
        return time, smooth_alt, velocity, acceleration, metrics

    def export_and_visualize(self, time, alt, vel, acc, metrics, pts_a, pts_b):
        print("\n--- FLIGHT METRICS ---")
        print(f"Max Altitude:    {metrics['max_altitude_m']:.2f} m")
        print(f"Time to Apogee:  {metrics['time_to_apogee_s']:.2f} s")
        print(f"Flight Duration: {metrics['flight_duration_s']:.2f} s")
        print("----------------------\n")
        
        pd.DataFrame({'x_a': pts_a[:,0], 'y_a': pts_a[:,1], 
                      'x_b': pts_b[:,0], 'y_b': pts_b[:,1]}).to_csv("tracked_positions.csv", index=False)
                      
        pd.DataFrame({'time_s': time, 'altitude_m': alt, 
                      'velocity_ms': vel, 'acceleration_ms2': acc}).to_csv("flight_data.csv", index=False)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        ax1.plot(time, alt, 'b-', label='Altitude')
        ax1.set_ylabel('Altitude (m)')
        ax1.grid(True)
        ax1.set_title('Rocket Flight Kinematics')
        
        ax2.plot(time, vel, 'g-', label='Velocity')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True)
        
        ax3.plot(time, acc, 'r-', label='Acceleration')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig("flight_graphs.png", dpi=300)
        print("Saved: flight_data.csv, tracked_positions.csv, and flight_graphs.png")
        plt.show()


if __name__ == "__main__":
    # =========================================================================
    # --- CONFIGURATION ---
    # =========================================================================
    BASELINE_DISTANCE = 10.0   # Distance between your tripods in meters
    FOV_X = 65.0               # Camera Horizontal Field of View in degrees
    FOV_Y = 40.0               # Camera Vertical Field of View in degrees
    
    # Insert your custom HSV values here!
    # Works best with a vibrant/bright color that stands out from sky backgrounds.
    # Examples:
    # Fluorescent Orange: Lower [5, 150, 150], Upper [25, 255, 255]
    # Neon Green:         Lower [35, 100, 100], Upper [85, 255, 255]
    # Bright Pink/Red:    Lower [170, 100, 100], Upper [180, 255, 255]
    
    HSV_LOWER = [5, 150, 150]
    HSV_UPPER = [25, 255, 255]
    
    video_file_a = "video_a.mp4"
    video_file_b = "video_b.mp4"
    # =========================================================================

    if not os.path.exists(video_file_a) or not os.path.exists(video_file_b):
        print(f"Error: Ensure {video_file_a} and {video_file_b} are in this directory.")
        exit(1)

    tracker = RocketTracker(
        baseline_m=BASELINE_DISTANCE, 
        fov_degrees_x=FOV_X, 
        fov_degrees_y=FOV_Y,
        hsv_lower=HSV_LOWER,
        hsv_upper=HSV_UPPER
    )
    
    # Execute Pipeline
    pts_a, fps_a, w, h = tracker.extract_trajectory(video_file_a, "Camera A Tracking")
    pts_b, fps_b, _, _ = tracker.extract_trajectory(video_file_b, "Camera B Tracking")
    
    fps = (fps_a + fps_b) / 2.0
    sync_a, sync_b = tracker.synchronize_videos(pts_a, pts_b)
    altitudes = tracker.triangulate(sync_a, sync_b, w, h)
    time_arr, alt_arr, vel_arr, acc_arr, flight_metrics = tracker.analyze_flight(altitudes, fps)
    
    tracker.export_and_visualize(time_arr, alt_arr, vel_arr, acc_arr, flight_metrics, sync_a, sync_b)
