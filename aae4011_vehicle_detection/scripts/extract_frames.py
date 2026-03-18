#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import cv2
import rosbag
from cv_bridge import CvBridge


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frames from a rosbag CompressedImage topic.")
    p.add_argument("--bag", required=True, help="Path to .bag")
    p.add_argument("--topic", default="/hikcamera/image_2/compressed", help="CompressedImage topic name")
    p.add_argument("--out_dir", default="extracted_frames", help="Output directory")
    p.add_argument("--step", type=int, default=1, help="Save every Nth frame (default: 1)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bridge = CvBridge()

    print(f"Opening bag: {args.bag}")
    print(f"Topic      : {args.topic}")
    print(f"Out dir    : {os.path.abspath(args.out_dir)}")
    print(f"Step       : {args.step}")

    bag = rosbag.Bag(args.bag, "r")
    count = 0
    saved = 0
    width = None
    height = None
    first_stamp = None
    last_stamp = None

    try:
        for _, msg, _ in bag.read_messages(topics=[args.topic]):
            count += 1
            stamp = msg.header.stamp.to_sec()
            if first_stamp is None:
                first_stamp = stamp
            last_stamp = stamp

            if (count - 1) % max(1, args.step) != 0:
                continue

            bgr = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if width is None:
                height, width = bgr.shape[:2]

            fn = os.path.join(args.out_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(fn, bgr)
            saved += 1

            if saved % 50 == 0:
                print(f"Saved {saved} frames...")
    finally:
        bag.close()

    duration = (last_stamp - first_stamp) if (first_stamp is not None and last_stamp is not None) else 0.0
    print("\n====== Extraction Summary ======")
    print(f"Bag file    : {args.bag}")
    print(f"Topic       : {args.topic}")
    print(f"Total msgs  : {count}")
    print(f"Saved imgs  : {saved}")
    if width is not None:
        print(f"Resolution  : {width} x {height}")
    print(f"Duration(s) : {duration:.2f}")


if __name__ == "__main__":
    main()

