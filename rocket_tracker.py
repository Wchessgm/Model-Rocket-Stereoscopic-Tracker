import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import math
import os

class RocketTracker:
    def __init__(self, baseline_m, fov_degrees_x, fov_degrees_y):
        self.baseline = baseline_m
        self.fov_x = math.radians(fov_degrees_x)
        self.fov_y = math.radians(fov_degrees_y)
        
        # Fluorescent Orange HSV Range (Adjust based on lighting)
        self.hsv_lower = np.array([5, 150, 150])
        self.hsv_upper = np.array([25, 255, 255])
        
    def extract_trajectory(self, video_path, window_name="Tracking"):
        print(f"Processing {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        trajectory = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to HSV and threshold
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            
            # Reduce noise
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroid = (np.nan, np.nan)
            if contours:
                # Find the largest contour (assumed to be the rocket)
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 10: # Minimum area threshold
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroid = (cx, cy)
                        cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            
            trajectory.append(centroid)
            
            # Show debug view
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_idx += 1
            
        cap.release()
        cv2.destroyAllWindows()
        
        # Clean up data: Interpolate missing frames
        df = pd.DataFrame(trajectory, columns=['x', 'y'])
        df = df.interpolate(method='linear', limit_direction='both')
        
        return df.to_numpy(), fps, width, height

    def synchronize_videos(self, traj_a, traj_b):
        print("Synchronizing videos based on initial vertical movement...")
        # Detect first significant upward movement (negative Y direction in pixels)
        def find_launch_frame(traj):
            y_coords = traj[:, 1]
            for i in range(5, len(y_coords)):
                # If rocket moves up by more than 5 pixels compared to 5 frames ago
                if y_coords[i-5] - y_coords[i] > 5:
                    return i
            return 0
            
        launch_a = find_launch_frame(traj_a)
        launch_b = find_launch_frame(traj_b)
        
        print(f"Launch detected at frame {launch_a} for Video A and {launch_b} for Video B.")
        
        # Trim arrays to start at the exact moment of launch
        sync_a = traj_a[launch_a:]
        sync_b = traj_b[launch_b:]
        
        # Make them the same length
        min_len = min(len(sync_a), len(sync_b))
        return sync_a[:min_len], sync_b[:min_len]

    def triangulate(self, pts_a, pts_b, width, height):
        print("Triangulating altitude data...")
        altitudes = []
        
        for (x_a, y_a), (x_b, y_b) in zip(pts_a, pts_b):
            # Convert pixel coords to angles
            # Assume cameras are parallel, baseline separated along X axis
            angle_x_a = (x_a - width/2) * (self.fov_x / width)
            angle_x_b = (x_b - width/2) * (self.fov_x / width)
            angle_y_a = (height/2 - y_a) * (self.fov_y / height)
            angle_y_b = (height/2 - y_b) * (self.fov_y / height)
            
            # Calculate distance to rocket on the Z axis (depth)
            # Z = Baseline / (tan(theta_a) - tan(theta_b))
            try:
                tan_a = math.tan(angle_x_a)
                tan_b = math.tan(angle_x_b)
                if abs(tan_a - tan_b) < 1e-5:
                    depth = np.nan
                else:
                    depth = self.baseline / (tan_a - tan_b)
                
                # Calculate altitude (Y axis in real world)
                # Average the altitude calculations from both cameras
                alt_a = depth * math.tan(angle_y_a)
                alt_b = depth * math.tan(angle_y_b)
                altitude = (alt_a + alt_b) / 2.0
                
                # Discard wild outliers (negative altitude or beyond realistic B-class limits)
                if altitude < -10 or altitude > 300:
                    altitude = np.nan
                    
            except Exception:
                altitude = np.nan
                
            altitudes.append(altitude)
            
        # Interpolate any nan values caused by division by zero or outliers
        alt_series = pd.Series(altitudes).interpolate(method='linear')
        return alt_series.to_numpy()

    def analyze_flight(self, altitudes, fps):
        print("Calculating kinematics...")
        time = np.arange(len(altitudes)) / fps
        
        # Smooth altitude data using Savitzky-Golay filter to remove pixel jitter
        window = min(31, len(altitudes) if len(altitudes) % 2 != 0 else len(altitudes) - 1)
        if window > 3:
            smooth_alt = savgol_filter(altitudes, window, 3)
        else:
            smooth_alt = altitudes
            
        # Ensure altitude starts at 0
        smooth_alt = smooth_alt - smooth_alt[0]
        
        # Calculate velocity (dh/dt)
        dt = 1.0 / fps
        velocity = np.gradient(smooth_alt, dt)
        
        # Smooth velocity
        velocity = savgol_filter(velocity, window, 3) if window > 3 else velocity
        
        # Calculate acceleration (dv/dt)
        acceleration = np.gradient(velocity, dt)
        
        # Metrics
        max_alt = np.max(smooth_alt)
        apogee_idx = np.argmax(smooth_alt)
        time_to_apogee = time[apogee_idx]
        flight_duration = time[-1]
        
        metrics = {
            "max_altitude_m": max_alt,
            "time_to_apogee_s": time_to_apogee,
            "flight_duration_s": flight_duration
        }
        
        return time, smooth_alt, velocity, acceleration, metrics

    def export_and_visualize(self, time, alt, vel, acc, metrics, pts_a, pts_b):
        # 1. Print Metrics
        print("\n--- FLIGHT METRICS ---")
        print(f"Max Altitude:    {metrics['max_altitude_m']:.2f} m")
        print(f"Time to Apogee:  {metrics['time_to_apogee_s']:.2f} s")
        print(f"Flight Duration: {metrics['flight_duration_s']:.2f} s")
        print("----------------------\n")
        
        # 2. Export CSVs
        pd.DataFrame({'x_a': pts_a[:,0], 'y_a': pts_a[:,1], 
                      'x_b': pts_b[:,0], 'y_b': pts_b[:,1]}).to_csv("tracked_positions.csv", index=False)
                      
        pd.DataFrame({'time_s': time, 'altitude_m': alt, 
                      'velocity_ms': vel, 'acceleration_ms2': acc}).to_csv("flight_data.csv", index=False)
        
        # 3. Plot Graphs
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        ax1.plot(time, alt, 'b-', label='Altitude')
        ax1.set_ylabel('Altitude (m)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Rocket Flight Kinematics')
        
        ax2.plot(time, vel, 'g-', label='Velocity')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True)
        ax2.legend()
        
        ax3.plot(time, acc, 'r-', label='Acceleration')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig("flight_graphs.png", dpi=300)
        print("Saved flight_data.csv, tracked_positions.csv, and flight_graphs.png")
        plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Measure the exact distance between your two cameras in meters
    BASELINE_DISTANCE = 10.0 
    
    # Estimate of smartphone camera FOV in degrees (Common for standard phone lenses)
    FOV_X = 65.0 
    FOV_Y = 40.0
    
    video_file_a = "video_a.mp4"
    video_file_b = "video_b.mp4"
    # ---------------------

    if not os.path.exists(video_file_a) or not os.path.exists(video_file_b):
        print(f"Error: Ensure {video_file_a} and {video_file_b} exist in this directory.")
        exit(1)

    tracker = RocketTracker(baseline_m=BASELINE_DISTANCE, fov_degrees_x=FOV_X, fov_degrees_y=FOV_Y)
    
    # 1. Track
    pts_a, fps_a, w, h = tracker.extract_trajectory(video_file_a, "Camera A")
    pts_b, fps_b, _, _ = tracker.extract_trajectory(video_file_b, "Camera B")
    
    # Use average FPS if they differ slightly
    fps = (fps_a + fps_b) / 2.0
    
    # 2. Synchronize
    sync_a, sync_b = tracker.synchronize_videos(pts_a, pts_b)
    
    # 3. Triangulate
    altitudes = tracker.triangulate(sync_a, sync_b, w, h)
    
    # 4. Analyze
    time_arr, alt_arr, vel_arr, acc_arr, flight_metrics = tracker.analyze_flight(altitudes, fps)
    
    # 5. Export
    tracker.export_and_visualize(time_arr, alt_arr, vel_arr, acc_arr, flight_metrics, sync_a, sync_b)
