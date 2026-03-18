#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2

try:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
except Exception:
    tk = None
    askopenfilename = None

from aae4011_vehicle_detection.bag_index import StreamingBagFrameSource, scan_bag_basic_info
from aae4011_vehicle_detection.decode import decode_compressed_image
from aae4011_vehicle_detection.detector import UltralyticsYoloDetector
from aae4011_vehicle_detection.render import FrameStats, UiState, draw_detections, draw_player_overlay


def _choose_bag_file() -> Optional[str]:
    if tk is None or askopenfilename is None:
        print("tkinter is not available; pass bag path via --bag in your own wrapper.", file=sys.stderr)
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = askopenfilename(title="Select ROS bag file", filetypes=[("ROS bag files", "*.bag"), ("All files", "*.*")])
    root.destroy()
    return path or None


def _popup_confirm(info) -> bool:
    if tk is None:
        return True
    msg = (
        "Bag details\n\n"
        f"  Topic:        {info.topic}\n"
        f"  Frames:       {info.message_count}\n"
        f"  Resolution:   {info.width} x {info.height}\n"
        f"  Duration:     {info.duration_sec:.2f} s\n\n"
        "Start detection?"
    )
    ok = {"value": False}

    root = tk.Tk()
    root.title("Confirm")
    root.attributes("-topmost", True)
    root.geometry("520x240")

    frame = tk.Frame(root, padx=14, pady=14)
    frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(frame, text=msg, justify=tk.LEFT, font=("Segoe UI", 10)).pack(anchor=tk.W)

    btns = tk.Frame(frame)
    btns.pack(side=tk.BOTTOM, pady=(16, 0))

    def start():
        ok["value"] = True
        root.destroy()

    def cancel():
        root.destroy()

    tk.Button(btns, text="Start", width=12, command=start).pack(side=tk.LEFT, padx=6)
    tk.Button(btns, text="Cancel", width=12, command=cancel).pack(side=tk.LEFT, padx=6)
    root.mainloop()
    return bool(ok["value"])


@dataclass
class PlayerState:
    playing: bool = False
    speed: float = 1.0
    conf: float = 0.30
    font_scale: float = 0.9
    cache_size: int = 200


def main() -> None:
    bag_path = _choose_bag_file()
    if not bag_path:
        return
    if not os.path.isfile(bag_path):
        print(f"File not found: {bag_path}", file=sys.stderr)
        return

    info = scan_bag_basic_info(bag_path)
    if info is None:
        print("No sensor_msgs/CompressedImage topic found in bag.", file=sys.stderr)
        return

    print("\n========== Bag report ==========")
    print(f"Bag       : {info.bag_path}")
    print(f"Topic     : {info.topic}")
    print(f"Frames    : {info.message_count}")
    print(f"Resolution: {info.width} x {info.height}")
    print(f"Duration  : {info.duration_sec:.2f} s")
    print("================================\n")

    if not _popup_confirm(info):
        return

    st = PlayerState()
    src = StreamingBagFrameSource(info.bag_path, info.topic, cache_size=st.cache_size)

    detector = UltralyticsYoloDetector(model="yolov8n.pt", conf=st.conf)

    per_class = Counter()
    total_seen = 0

    win = "Vehicle Detection (Bag Player)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Confidence (%)", win, int(st.conf * 100), 99, lambda _: None)
    cv2.createTrackbar("Speed (0.25-4x)", win, 8, 40, lambda _: None)  # 8 -> ~1.0x
    cv2.createTrackbar("Font (0.5-2.0)", win, 10, 30, lambda _: None)
    cv2.createTrackbar("Cache (frames)", win, min(400, st.cache_size), 600, lambda _: None)

    frame_idx = 0
    first_stamp = info.first_stamp
    duration_sec = max(0.0, info.duration_sec)
    n_frames_hint = int(info.message_count)

    fps_render = 0.0
    t_last = time.time()

    def on_mouse(event, x, y, flags, param):
        nonlocal frame_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # progress bar sits near the bottom; map x to frame index
        w = info.width if info.width > 0 else 1920
        bar_margin = 12
        bar_w = max(1, w - 2 * bar_margin)
        rel = min(max(x - bar_margin, 0), bar_w)
        target = int((rel / bar_w) * max(1, n_frames_hint - 1))
        frame_idx = max(0, min(target, n_frames_hint - 1))

    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            st.conf = max(0.01, cv2.getTrackbarPos("Confidence (%)", win) / 100.0)
            speed_pos = cv2.getTrackbarPos("Speed (0.25-4x)", win)
            st.speed = 0.25 + (speed_pos / 40.0) * 3.75
            font_pos = cv2.getTrackbarPos("Font (0.5-2.0)", win)
            st.font_scale = 0.5 + (font_pos / 30.0) * 1.5
            st.cache_size = max(20, cv2.getTrackbarPos("Cache (frames)", win))
            src.cache_size = st.cache_size
            detector.set_confidence(st.conf)

            item = src.get(frame_idx)
            if item is None:
                break
            stamp, data = item
            decoded = decode_compressed_image(data)
            if decoded is None:
                frame_idx = min(frame_idx + 1, n_frames_hint - 1)
                continue

            bgr = decoded.bgr

            dets = detector.detect(bgr)
            vehicles_in_frame = 0
            for d in dets:
                vehicles_in_frame += 1
                total_seen += 1
                per_class[d.cls_name] += 1

            draw_detections(bgr, dets, font_scale=st.font_scale)

            now = time.time()
            dt = now - t_last
            t_last = now
            fps_render = 1.0 / dt if dt > 1e-6 else fps_render

            ui = UiState(playing=st.playing, speed=st.speed, conf=st.conf, font_scale=st.font_scale)
            stats = FrameStats(
                frame_idx=frame_idx,
                n_frames=n_frames_hint,
                t_sec=float(stamp - first_stamp),
                duration_sec=duration_sec,
                vehicles_in_frame=vehicles_in_frame,
                total_vehicles_seen=total_seen,
                per_class_total=dict(per_class),
                fps_render=fps_render,
            )
            draw_player_overlay(bgr, ui, stats)

            cv2.imshow(win, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                st.playing = not st.playing
            elif key in (81, ord("a")):  # left / A
                # seek ~5 sec backwards using timestamp list if available
                target_t = max(first_stamp, stamp - 5.0)
                src.ensure_scanned(max_messages=n_frames_hint)
                idx = frame_idx
                while idx > 0 and src.stamps[idx] > target_t:
                    idx -= 1
                frame_idx = idx
            elif key in (83, ord("d")):  # right / D
                target_t = min(first_stamp + duration_sec, stamp + 5.0)
                src.ensure_scanned(max_messages=n_frames_hint)
                idx = frame_idx
                while idx < n_frames_hint - 1 and src.stamps[idx] < target_t:
                    idx += 1
                frame_idx = idx

            if st.playing:
                frame_idx += 1
                if frame_idx >= n_frames_hint:
                    frame_idx = 0
                    per_class.clear()
                    total_seen = 0

                # crude pacing: estimate fps from message count / duration
                fps_bag = (n_frames_hint - 1) / duration_sec if duration_sec > 0 and n_frames_hint > 1 else 10.0
                desired = 1.0 / (fps_bag * st.speed)
                spent = time.time() - now
                if desired > spent:
                    time.sleep(desired - spent)
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

