# Model Rocket Stereoscopic Tracker

This Python application analyzes two 60 FPS smartphone videos of a model rocket launch, tracks a user-specified color (like a bright nose cone or fins) using customizable HSV thresholding, synchronizes the footage, and estimates altitude, velocity, and acceleration using stereoscopic triangulation.

## Requirements

You will need Python 3.8+ installed. Install the required libraries using pip:

pip install opencv-python numpy matplotlib pandas scipy

## Setup Instructions

1. **Prepare your videos:** - Place cameras on tripods separated by a known baseline distance (e.g., 10 meters).
   - Aim both cameras at the launch pad. Keep them as parallel as possible.
   - Record at 60 FPS and save as `video_a.mp4` and `video_b.mp4` in the script's directory.
2. **Choose Your Tracking Color:**
   - **Crucial Tip:** This tracker works best if the rocket features a **bright, high-contrast color** (like fluorescent orange, neon green, or bright pink) that stands out sharply against the blue sky, clouds, or ground vegetation.
   - Find the HSV (Hue, Saturation, Value) range for your rocket's color. You can use an online image color picker or an HSV color mapping tool.
3. **Configure the Script:**
   - Open `rocket_tracker.py` and scroll to the `--- CONFIGURATION ---` block at the bottom.
   - Input your camera baseline distance and your custom `HSV_LOWER` and `HSV_UPPER` boundary arrays.

## How to Run

1. Open your terminal or command prompt.
2. Navigate to the folder containing the script and videos.
3. Run the script: `python rocket_tracker.py`
4. Press 'q' on the video pop-ups if you want to skip visual debugging.

## Outputs

- `tracked_positions.csv`: Raw X/Y pixel coordinates for both cameras.
- `flight_data.csv`: Time, Altitude, Velocity, and Acceleration data.
- `flight_graphs.png`: High-resolution plots of the rocket's kinematics.
