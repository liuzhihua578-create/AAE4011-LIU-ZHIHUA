AAE4011 Assignment 1 — Q3: ROS-Based Vehicle Detection from Rosbag

> **Student Name:** [LIU Zhihua] | **Student ID:** [23101399D] | **Date:** [Mar/18/2026]

---

## 1. Overview

This project is a **ROS1 (Noetic)** rosbag-based vehicle detection pipeline: it replays a `.bag` containing `sensor_msgs/CompressedImage` using `rosbag play`, subscribes to an image topic, runs **Ultralytics YOLOv8** inference (default `yolov8n.pt`) for vehicle classes, and visualizes bounding boxes and live statistics in an OpenCV window.

---

## 2. Detection Method *(Q3.1 — 2 marks)*

This project uses **Ultralytics YOLOv8** as the object detector (see `UltralyticsYoloDetector` in `src/aae4011_vehicle_detection/detector.py`) for the following reasons:

- **Easy integration**: Ultralytics provides a simple Python API (`YOLO(model)`), which is straightforward to embed into a ROS node.
- **Real-time friendly**: lightweight models like `yolov8n.pt` typically achieve good inference speed on both CPU and GPU, fitting online visualization during rosbag playback.
- **Task-aligned outputs**: by default, the detector filters to common vehicle-related classes (e.g., `car/truck/bus/motorbike/motorcycle/train`) to reduce non-vehicle detections.

---

## 3. Repository Structure

This ROS package is named `aae4011_vehicle_detection`. The core repository layout is:

```text
aae4011_vehicle_detection/
├─ package.xml
├─ CMakeLists.txt
├─ setup.py
├─ launch/
│  └─ detect_from_bag.launch
├─ scripts/
│  ├─ vehicle_detector_node.py
│  ├─ bag_extract_and_report.py
│  ├─ bag_player_detector.py
│  ├─ extract_frames.py
│  └─ launch_with_bag_picker.py
└─ src/aae4011_vehicle_detection/
   ├─ __init__.py
   ├─ bag_index.py
   ├─ decode.py
   ├─ detector.py
   └─ render.py
```

---

## 4. Prerequisites

- OS: Ubuntu 20.04 (often run via WSL on Windows 10/11)
- ROS: ROS1 Noetic
- Python: Python 3 (default with ROS Noetic)
- ROS dependencies (declared in `package.xml` / `CMakeLists.txt`)
  - `rospy`
  - `sensor_msgs`
  - `cv_bridge`
  - `image_transport`
  - `rosbag`
- Python packages
  - `ultralytics`
  - `opencv-python`
  - `numpy`

Install example (Ubuntu/WSL):

```bash
sudo apt update
sudo apt install -y ros-noetic-cv-bridge ros-noetic-image-transport
pip3 install --user ultralytics opencv-python numpy
```

---

## 5. How to Run *(Q3.1 — 2 marks)*

### 5.1 Clone the repository

Place this package into your catkin workspace (example: `~/catkin_ws/src`). If you are using Windows + WSL, it is commonly convenient to symlink to the Windows filesystem.

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Option A (recommended on Windows + WSL): symlink to Windows filesystem
rm -rf aae4011_vehicle_detection
ln -s /mnt/c/Users/User/Desktop/vehicle-detection/aae4011_vehicle_detection aae4011_vehicle_detection
```

### 5.2 Install dependencies

Install ROS and Python dependencies as listed in **Prerequisites**.

### 5.3 Build the ROS package

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

### 5.4 Place the rosbag file

Put the provided `.bag` file somewhere convenient (under WSL, placing it under `/mnt/c/...` is often convenient if you want to reference Windows paths).

First, confirm the `.bag` contains an image topic of type `sensor_msgs/CompressedImage`:

```bash
rosbag info /path/to/file.bag
```

In the `topics:` list, find an entry like the following and note the topic name (example):

```text
topics: /hikcamera/image_2/compressed  1122 msgs : sensor_msgs/CompressedImage
```

### 5.5 Launch the pipeline

This project provides a ROS launch file `launch/detect_from_bag.launch`, which will:

- start `rosbag play --clock <bag_path>`
- start the detection node `scripts/vehicle_detector_node.py` to subscribe to compressed images, run inference, and visualize results

Run example (replace with your actual bag path and topic name):

```bash
roslaunch aae4011_vehicle_detection detect_from_bag.launch \
  bag_path:=/path/to/file.bag \
  image_topic:=/hikcamera/image_2/compressed \
  model:=yolov8n.pt \
  conf_threshold:=0.30
```

An OpenCV window named `Vehicle Detection (ROS)` should appear. Press **ESC** to exit.

---

## 6. Sample Results

<img width="546" height="160" alt="image" src="https://github.com/user-attachments/assets/03979a9f-088a-40f3-993f-4420ac20f6e1" />

<img width="1659" height="1318" alt="image" src="https://github.com/user-attachments/assets/9b7d14b8-cc49-436d-acaf-93810e1eb5d7" />



### 6.1 Image extraction summary (total frames, resolution, topic name)

Use `scripts/bag_extract_and_report.py` to auto-pick (or explicitly set) a `sensor_msgs/CompressedImage` topic, summarize frame count/resolution/duration/FPS, and optionally export frames as image files.

Report only (decode without writing images):

```bash
rosrun aae4011_vehicle_detection bag_extract_and_report.py \
  --bag /path/to/file.bag \
  --no_save
```

Export all frames (example output directory):

```bash
rosrun aae4011_vehicle_detection bag_extract_and_report.py \
  --bag /path/to/file.bag \
  --out_dir extracted_frames \
  --format png
```

Paste your key results here (example format):

- Topic name: `[fill in your topic name]`
- Total frames (message count): `[fill in]`
- Resolution: `[fill in, e.g., 1920 x 1080]`
- Duration: `[fill in]`
- Estimated FPS: `[fill in]`

### 6.2 Detection results (sample screenshot, detection statistics)

After running `roslaunch ... detect_from_bag.launch`, the OpenCV window overlays:

- Frame index, vehicles-in-frame count, total vehicles seen, and FPS
- Per-class cumulative counts (up to the top 6)

Add the following here:

- One screenshot of the visualization (with bounding boxes and HUD)
- A short statistics summary (e.g., total vehicles detected, dominant classes, etc.)

---

## 7. Video Demonstration *(Q3.2 — 5 marks)*

**Video Link:** [YouTube](https://youtu.be/LMgsO2oZ3mA)

The video (1–3 min) should include:

- (a) Launching `roslaunch aae4011_vehicle_detection detect_from_bag.launch ...` in the terminal
- (b) The OpenCV UI showing live detection results (at least several seconds)
- (c) A brief explanation: the image topic, model, threshold, and observed detection quality

---

## 8. Reflection & Critical Analysis *(Q3.3 — 8 marks, 300–500 words)*

### (a) What Did You Learn? *(2 marks)*

This project improved two concrete skill areas for me. First, I became comfortable with the ROS1 Noetic offline reproducibility workflow around rosbag: using `rosbag info` to quickly identify `sensor_msgs/CompressedImage` topics, replaying with `rosbag play --clock` so nodes run under simulated time, and implementing a subscriber node to process the replayed stream online. Second, I learned how to integrate a deep-learning detector as an “engineering component” inside ROS: wrapping Ultralytics YOLO inference behind a `Detector` interface, adding a vehicle-class whitelist and confidence thresholding so outputs better match the task, and exposing frame-level statistics in the visualization HUD (OpenCV) for easier verification and tuning.

### (b) How Did You Use AI Tools? *(2 marks)*

I used AI tools mainly in two ways during development. The first was “faster debugging and troubleshooting”: when the image window did not update, a topic name did not match, or dependencies were missing, AI helped provide a structured checklist (e.g., checking `rostopic hz`, confirming message types, confirming GUI environment settings). The second was “improving code structure and readability”: AI suggestions helped separate inference, decoding, and rendering into modules, and encouraged more robust parameter handling (e.g., exposing topic/model/threshold via ROS parameters). The limitations were also clear: AI cannot replace validation on the actual bag data—false positives/negatives under a specific viewpoint and lighting must be confirmed by real playback. Also, ROS/WSL GUI behavior differs across setups, so generic advice still required iteration based on the actual system configuration.

### (c) How to Improve Accuracy? *(2 marks)*

First, perform **fine-tuning or domain adaptation** for the course scenario: label a small set of frames extracted from the bag and do lightweight YOLOv8 fine-tuning so the model better matches the camera viewpoint, scale distribution, and motion blur, reducing missed detections and class confusion. Second, incorporate **better post-processing and temporal information**: use lightweight tracking (e.g., SORT/ByteTrack) to associate detections across frames, then stabilize outputs with trajectory smoothing or short-window voting to suppress single-frame false positives; additionally, tune NMS/thresholds by scenario (e.g., slightly lower thresholds for far small objects plus size/aspect-ratio filtering) to improve overall stability.

### (d) Real-World Challenges *(2 marks)*

There are two key challenges when deploying this pipeline on a real drone in real time. The first is **compute and latency constraints**: embedded platforms have limited GPU/CPU resources, and high input resolution/frame rate can increase inference latency, which impacts real-time control or avoidance decisions. This requires trade-offs among model size, input resolution, and inference frequency, and may require acceleration (e.g., TensorRT). The second is **environmental and sensor uncertainty**: real flights involve lighting changes, vibration, motion blur, occlusion, and weather effects, and camera exposure/compression settings can shift the image distribution. This can significantly reduce detection quality compared to offline bag playback, motivating stronger robustness strategies (augmentation, domain generalization, online adaptation) and better fault detection/fallback mechanisms (e.g., down-weight detection when confidence is unreliable or switch to other sensors).

---

## 9. References

- Ultralytics YOLO documentation and repository: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- ROS Noetic official documentation: [ROS Noetic](http://wiki.ros.org/noetic)

