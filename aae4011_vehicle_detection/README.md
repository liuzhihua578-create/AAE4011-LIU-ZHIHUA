# aae4011_vehicle_detection

ROS1 (Noetic) vehicle detection pipeline that can:

- Play a `.bag` with `rosbag play`
- Subscribe to a `sensor_msgs/CompressedImage` topic
- Run YOLO (Ultralytics) inference
- Visualize results in an OpenCV window
- Extract/export all frames from a bag and report image properties

This repository is commonly used from **Ubuntu 20.04 + ROS Noetic**, including **WSL on Windows 10/11**.

---

## Requirements

### System

- Ubuntu 20.04
- ROS Noetic
- Python 3

### ROS dependencies (catkin)

Declared in `package.xml` / `CMakeLists.txt`:

- `rospy`
- `sensor_msgs`
- `cv_bridge`
- `image_transport`
- `rosbag`

### Python packages

- `ultralytics`
- `opencv-python`
- `numpy`

Install examples:

```bash
sudo apt update
sudo apt install -y ros-noetic-cv-bridge ros-noetic-image-transport
pip3 install --user ultralytics opencv-python numpy
```

---

## Workspace setup (catkin)

Create a workspace if you don't have one:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
```

### Put the package in `~/catkin_ws/src`

You have two typical options:

- **Option A (recommended on Windows + WSL)**: symlink to Windows filesystem
- **Option B**: copy source into WSL

#### Option A: symlink (recommended)

```bash
cd ~/catkin_ws/src
rm -rf aae4011_vehicle_detection
ln -s /mnt/c/Users/User/Desktop/vehicle-detection/aae4011_vehicle_detection aae4011_vehicle_detection
```

#### Option B: copy

```bash
cd ~/catkin_ws/src
rm -rf aae4011_vehicle_detection
cp -r /mnt/c/Users/User/Desktop/vehicle-detection/aae4011_vehicle_detection aae4011_vehicle_detection
```

### Build

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

---

## Run: vehicle detection from a bag (ROS launch)

Launch file: `launch/detect_from_bag.launch`

Parameters:

- `bag_path` (required): path to `.bag`
- `image_topic` (optional): `sensor_msgs/CompressedImage` topic name
- `model` (optional): YOLO model path/name (e.g. `yolov8n.pt`)
- `conf_threshold` (optional): confidence threshold (e.g. `0.30`)

### 1) Find the correct image topic in the bag

```bash
rosbag info /path/to/file.bag
```

Look at the `topics:` section and pick the one with type:

- `sensor_msgs/CompressedImage`

Example:

```text
topics: /hikcamera/image_1/compressed  1122 msgs : sensor_msgs/CompressedImage
```

### 2) Run launch

```bash
roslaunch aae4011_vehicle_detection detect_from_bag.launch \
  bag_path:=/mnt/c/Users/User/Desktop/2026-02-23-15-58-29.bag \
  image_topic:=/hikcamera/image_1/compressed \
  model:=yolov8n.pt \
  conf_threshold:=0.30
```

An OpenCV window named `Vehicle Detection (ROS)` should appear.

Controls:

- Press **ESC** in the OpenCV window to exit

### Troubleshooting: “Waiting for images…”

If the window shows **Waiting for images…** and never updates:

1) Verify the topic exists and is being published during playback:

```bash
rostopic info /hikcamera/image_1/compressed
rostopic hz /hikcamera/image_1/compressed
```

2) Verify the bag really contains that topic:

```bash
rosbag info /path/to/file.bag
```

If `rostopic info` shows `Publishers: None`, then `rosbag play` is not publishing that topic (topic name mismatch).

---

## WSL notes (OpenCV window / GUI)

If you're using **Windows 10 + WSL** (without WSLg), OpenCV windows require an X server.

Symptoms when GUI is not configured:

- No window appears, even though the node runs

Fix (high level):

- Install and run an X server on Windows (e.g. VcXsrv/Xming)
- Set `DISPLAY` in WSL to point to it

Check:

```bash
echo $DISPLAY
```

If it is empty, OpenCV windows generally cannot open.

---

## Utility: extract frames and report bag image properties

### Script: `bag_extract_and_report.py` (recommended)

This script:

- Auto-picks the best `sensor_msgs/CompressedImage` topic (or you specify `--topic`)
- Decodes **all** frames
- Optionally exports frames to disk
- Reports image count, resolution, duration, and estimated FPS

#### Report only (decode all, no disk writes)

```bash
rosrun aae4011_vehicle_detection bag_extract_and_report.py \
  --bag /mnt/c/Users/User/Desktop/2026-02-23-15-58-29.bag \
  --no_save
```

#### Export all frames

```bash
rosrun aae4011_vehicle_detection bag_extract_and_report.py \
  --bag /mnt/c/Users/User/Desktop/2026-02-23-15-58-29.bag \
  --out_dir /mnt/c/Users/User/Desktop/bag_frames \
  --format png
```

#### Export every Nth frame

```bash
rosrun aae4011_vehicle_detection bag_extract_and_report.py \
  --bag /mnt/c/Users/User/Desktop/2026-02-23-15-58-29.bag \
  --out_dir /mnt/c/Users/User/Desktop/bag_frames \
  --step 5
```

---

## Other scripts

- `scripts/extract_frames.py`: basic frame extraction for a specified topic
- `scripts/bag_player_detector.py`: standalone bag player + detector UI (non-ROS runtime)
- `scripts/launch_with_bag_picker.py`: helper to select a bag file via a dialog (if tkinter is available)

---

## Project structure (high level)

- `launch/`: ROS launch files
- `scripts/`: ROS nodes / utilities (installed via `catkin_install_python`)
- `src/aae4011_vehicle_detection/`: importable Python package (decoder, bag indexing, rendering, detector wrapper)

## FUll RUN

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
rm -rf aae4011_vehicle_detection
ln -s /mnt/c/Users/User/Desktop/vehicle-detection/aae4011_vehicle_detection aae4011_vehicle_detection

cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

rosbag info /mnt/c/Users/User/Desktop/2026-02-23-15-58-29.bag

rosrun aae4011_vehicle_detection bag_extract_and_report.py \
  --bag /mnt/c/Users/User/Desktop/2026-02-23-15-58-29.bag \
  --out_dir /mnt/c/Users/User/Desktop/bag_frames \
  --format png

roslaunch aae4011_vehicle_detection detect_from_bag.launch \
  bag_path:=/mnt/c/Users/User/Desktop/2026-02-23-15-58-29.bag \
  image_topic:=/hikcamera/image_1/compressed \
  model:=yolov8n.pt \
  conf_threshold:=0.30 