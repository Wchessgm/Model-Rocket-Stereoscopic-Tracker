# Model Rocket Stereoscopic Tracker

This Python application analyzes two 60 FPS smartphone videos of a model rocket launch, tracks the fluorescent orange nose cone using HSV color detection, synchronizes the footage, and estimates altitude, velocity, and acceleration using stereoscopic triangulation.

## Requirements

You will need Python 3.8+ installed. Install the required libraries using pip:

pip install opencv-python numpy matplotlib pandas scipy

## Setup Instructions

1. **Prepare your videos:** - Place cameras on tripods separated by a known baseline distance (e.g., 10 meters).
   - Aim both cameras at the launch pad. Keep them as parallel as possible.
   - Record at 60 FPS. 
   - Save the files as `video_a.mp4` and `video_b.mp4` in the same directory as the script.
2. **Measure your baseline:** Note the exact distance between the two camera tripods in meters.
3. **HSV Tuning:** The script defaults to tracking fluorescent orange. If your lighting is different, you may need to adjust the `HSV_LOWER` and `HSV_UPPER` values in the script. You can use an online HSV color picker to find the right range for your specific nose cone.

## How to Run

1. Open your terminal or command prompt.
2. Navigate to the folder containing the script and videos. ( cd function on cmd windows)
3. Run the script: `python rocket_tracker.py`
4. The script will process both videos (which may take a few minutes depending on your CPU), display tracking overlays (press 'q' to skip visual debugging), and generate your CSV and PNG outputs.

## Outputs

- `tracked_positions.csv`: Raw X/Y pixel coordinates for both cameras.
- `flight_data.csv`: Time, Altitude, Velocity, and Acceleration data.
- `flight_graphs.png`: Plots of the rocket's kinematics.
