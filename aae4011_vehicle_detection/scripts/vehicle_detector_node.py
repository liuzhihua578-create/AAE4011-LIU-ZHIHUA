#!/usr/bin/env python3
from __future__ import annotations

import os
import threading
import time
from collections import Counter

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from aae4011_vehicle_detection.detector import UltralyticsYoloDetector
from aae4011_vehicle_detection.render import draw_detections


class VehicleDetectorNode:
    def __init__(self):
        rospy.init_node("vehicle_detector_node")
        rospy.loginfo("vehicle_detector_node.py path: %s (pid=%s)", __file__, os.getpid())

        self.image_topic = rospy.get_param("~image_topic", "/hikcamera/image_2/compressed")
        self.conf_thres = float(rospy.get_param("~conf_threshold", 0.30))
        self.model_name = rospy.get_param("~model", "yolov8n.pt")
        self.device = rospy.get_param("~device", None)
        self.window_name = rospy.get_param("~window_name", "Vehicle Detection (ROS)")

        rospy.loginfo("Loading model: %s (conf=%.2f)", self.model_name, self.conf_thres)
        self.detector = UltralyticsYoloDetector(model=self.model_name, conf=self.conf_thres, device=self.device)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.per_class_counter = Counter()
        self.total_vehicle_count = 0

        self._t_last = time.time()
        self._fps = 0.0

        self._frame_lock = threading.Lock()
        self._last_bgr = None
        self._last_frame_count_shown = 0

        self.sub = rospy.Subscriber(
            self.image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("Subscribed to: %s", self.image_topic)
        self._try_show_startup_window()

    def _try_show_startup_window(self) -> None:
        """
        Create the OpenCV window immediately so users can tell whether GUI works,
        even if no frames are received (wrong topic / empty bag / playback ends early).
        """
        if os.name == "posix":
            rospy.loginfo(
                "GUI env: DISPLAY=%r WAYLAND_DISPLAY=%r XDG_RUNTIME_DIR=%r",
                os.environ.get("DISPLAY"),
                os.environ.get("WAYLAND_DISPLAY"),
                os.environ.get("XDG_RUNTIME_DIR"),
            )
        try:
            import numpy as np

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            placeholder = np.zeros((480, 848, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Waiting for images...",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                placeholder,
                f"Topic: {self.image_topic}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, placeholder)
            cv2.waitKey(1)

            # If OpenCV can't create a real window (common in headless/WSL without X),
            # this often returns < 0.
            try:
                vis = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                if vis < 0:
                    rospy.logwarn(
                        "OpenCV window was not created (WND_PROP_VISIBLE=%s). "
                        "If you're on WSL/SSH/headless, you likely need an X server and DISPLAY.",
                        vis,
                    )
            except Exception as e:
                rospy.logwarn("Could not query OpenCV window property: %r", e)
        except Exception as e:
            rospy.logwarn(
                "OpenCV GUI window could not be created (headless/WSL without X server?). "
                "Error: %r",
                e,
            )

    def image_callback(self, msg: CompressedImage) -> None:
        self.frame_count += 1

        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        self._fps = 1.0 / dt if dt > 1e-6 else self._fps

        bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self.detector.detect(bgr)

        vehicles_in_frame = 0
        for d in detections:
            vehicles_in_frame += 1
            self.total_vehicle_count += 1
            self.per_class_counter[d.cls_name] += 1

        draw_detections(bgr, detections, font_scale=0.9)

        hud = f"Frame: {self.frame_count}  Vehicles: {vehicles_in_frame}  Total: {self.total_vehicle_count}  FPS: {self._fps:.1f}"
        cv2.putText(bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 2, cv2.LINE_AA)

        y = bgr.shape[0] - 10
        for cls_name, cnt in self.per_class_counter.most_common(6):
            cv2.putText(bgr, f"{cls_name}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
            y -= 20

        # NOTE: Don't call cv2.imshow from rospy callback thread.
        # Some OpenCV backends (notably under WSL/X) won't reliably update from non-main threads.
        # We render in the main loop in spin().
        with self._frame_lock:
            self._last_bgr = bgr

    def spin(self) -> None:
        rospy.loginfo("Running. Press ESC in the OpenCV window to exit.")
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            frame = None
            with self._frame_lock:
                if self._last_bgr is not None and self._last_frame_count_shown != self.frame_count:
                    frame = self._last_bgr
                    self._last_frame_count_shown = self.frame_count

            if frame is not None:
                try:
                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        rospy.signal_shutdown("ESC pressed")
                        break
                except Exception as e:
                    rospy.logwarn_throttle(5.0, "OpenCV imshow/waitKey failed: %r", e)

            rate.sleep()
        cv2.destroyAllWindows()


def main() -> None:
    VehicleDetectorNode().spin()


if __name__ == "__main__":
    main()

