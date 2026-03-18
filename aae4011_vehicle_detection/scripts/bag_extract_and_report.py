#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Optional

import cv2

from aae4011_vehicle_detection.bag_index import pick_best_compressed_image_topic, scan_bag_basic_info
from aae4011_vehicle_detection.decode import decode_compressed_image


def _fmt_secs(s: float) -> str:
    if s < 0:
        return f"{s:.3f}"
    m, sec = divmod(s, 60.0)
    h, m = divmod(m, 60.0)
    if h >= 1:
        return f"{int(h)}:{int(m):02d}:{sec:06.3f}"
    if m >= 1:
        return f"{int(m)}:{sec:06.3f}"
    return f"{sec:.3f}s"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Read a ROS1 .bag, find a sensor_msgs/CompressedImage topic, decode all frames, "
            "optionally export every frame as an image, and report image count & properties."
        )
    )
    p.add_argument("--bag", required=True, help="Path to .bag")
    p.add_argument("--topic", default="", help="CompressedImage topic name (auto-pick if empty)")
    p.add_argument("--out_dir", default="extracted_frames", help="Output directory for exported frames")
    p.add_argument("--format", default="png", choices=["png", "jpg", "jpeg", "webp"], help="Image file format")
    p.add_argument("--step", type=int, default=1, help="Save every Nth frame (default: 1 = save all)")
    p.add_argument("--no_save", action="store_true", help="Decode frames but do not write images to disk")
    p.add_argument("--max_frames", type=int, default=0, help="Limit processed frames (0 = no limit)")
    args = p.parse_args()

    bag_path = os.path.expanduser(args.bag)
    if not os.path.isfile(bag_path):
        print(f"[ERROR] Bag not found: {bag_path}", file=sys.stderr)
        return 2

    topic: Optional[str] = args.topic.strip() or None
    if not topic:
        pick = pick_best_compressed_image_topic(bag_path)
        if not pick:
            print("[ERROR] No sensor_msgs/CompressedImage topics found in bag.", file=sys.stderr)
            return 3
        topic = pick[0]

    info = scan_bag_basic_info(bag_path, topic=topic)
    if info is None:
        print(f"[ERROR] Could not read bag info for topic: {topic}", file=sys.stderr)
        return 4

    print("\n========== Bag image report ==========")
    print(f"Bag path        : {info.bag_path}")
    print(f"Topic           : {info.topic}")
    print(f"Message count   : {info.message_count}")
    if info.width and info.height:
        print(f"Resolution      : {info.width} x {info.height}")
    print(f"First stamp     : {info.first_stamp:.6f}")
    print(f"Last stamp      : {info.last_stamp:.6f}")
    print(f"Duration        : {_fmt_secs(info.duration_sec)}")
    fps_est = (info.message_count - 1) / info.duration_sec if info.duration_sec > 1e-6 and info.message_count > 1 else 0.0
    if fps_est > 0:
        print(f"FPS (estimated) : {fps_est:.3f}")
    print("======================================\n")

    step = max(1, int(args.step))
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else 0

    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)
        out_dir_abs = os.path.abspath(args.out_dir)
        print(f"Exporting frames to: {out_dir_abs}")
        print(f"Format            : {args.format}")
        print(f"Step              : {step}")
        if max_frames:
            print(f"Max frames        : {max_frames}")
        print()
    else:
        print("Decoding frames (no_save=true).")
        if max_frames:
            print(f"Max frames        : {max_frames}")
        print()

    # Stream messages without loading everything into memory.
    import rosbag

    processed = 0
    decoded_ok = 0
    saved = 0
    widths = set()
    heights = set()

    bag = rosbag.Bag(bag_path, "r")
    try:
        for _, msg, _ in bag.read_messages(topics=[info.topic]):
            processed += 1
            if max_frames and processed > max_frames:
                break

            data = bytes(getattr(msg, "data", b""))
            dec = decode_compressed_image(data)
            if dec is None:
                continue

            decoded_ok += 1
            widths.add(int(dec.width))
            heights.add(int(dec.height))

            if args.no_save:
                continue

            if (processed - 1) % step != 0:
                continue

            fn = os.path.join(args.out_dir, f"frame_{saved:06d}.{args.format}")
            cv2.imwrite(fn, dec.bgr)
            saved += 1
            if saved % 200 == 0:
                print(f"Saved {saved} frames...")
    finally:
        bag.close()

    print("\n========== Extraction summary ==========")
    print(f"Topic            : {info.topic}")
    print(f"Processed msgs   : {processed}")
    print(f"Decoded frames   : {decoded_ok}")
    if not args.no_save:
        print(f"Saved images     : {saved}")
        print(f"Out dir          : {os.path.abspath(args.out_dir)}")
    if widths and heights:
        if len(widths) == 1 and len(heights) == 1:
            print(f"Resolution       : {next(iter(widths))} x {next(iter(heights))}")
        else:
            print(f"Resolution range : widths={sorted(widths)} heights={sorted(heights)}")
    print("========================================\n")

    # Useful for scripts/CI: a single-line JSON-ish dump.
    print("report_dict =", asdict(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

