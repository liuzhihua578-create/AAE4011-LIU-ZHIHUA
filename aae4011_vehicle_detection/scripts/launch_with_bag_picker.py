#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import tkinter as tk
    from tkinter import simpledialog
    from tkinter.filedialog import askopenfilename
except Exception:
    tk = None
    askopenfilename = None
    simpledialog = None


@dataclass
class LaunchArgs:
    bag_path: str
    image_topic: str = "/hikcamera/image_2/compressed"
    conf_threshold: float = 0.30
    model: str = "yolov8n.pt"


def _roslaunch_escape(value: str) -> str:
    """
    roslaunch argument values are split on spaces; quoting does not behave like a shell.
    Escape spaces (and backslashes) so paths like 'Video 2/...' stay a single token.
    """
    if value is None:
        return ""
    return value.replace("\\", "\\\\").replace(" ", "\\ ")


def _pick_bag_file() -> Optional[str]:
    if tk is None or askopenfilename is None:
        print("tkinter is not available; cannot open file dialog.", file=sys.stderr)
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = askopenfilename(title="Select ROS bag file", filetypes=[("ROS bag files", "*.bag"), ("All files", "*.*")])
    root.destroy()
    return path or None


def _ask_params(defaults: LaunchArgs) -> LaunchArgs:
    """
    Ask optional parameters via small dialogs.
    If dialogs are unavailable, just return defaults.
    """
    if tk is None or simpledialog is None:
        return defaults

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    topic = simpledialog.askstring("Image topic", "CompressedImage topic:", initialvalue=defaults.image_topic, parent=root)
    if topic:
        defaults.image_topic = topic.strip()

    conf = simpledialog.askfloat("Confidence", "Confidence threshold (0.01~0.99):", initialvalue=defaults.conf_threshold, parent=root, minvalue=0.01, maxvalue=0.99)
    if conf is not None:
        defaults.conf_threshold = float(conf)

    model = simpledialog.askstring("Model", "Model (e.g. yolov8n.pt or /path/to/weights.pt):", initialvalue=defaults.model, parent=root)
    if model:
        defaults.model = model.strip()

    root.destroy()
    return defaults


def main() -> int:
    bag = _pick_bag_file()
    if not bag:
        return 1
    if not os.path.isfile(bag):
        print(f"File not found: {bag}", file=sys.stderr)
        return 2

    args = _ask_params(LaunchArgs(bag_path=bag))

    cmd = [
        "roslaunch",
        "aae4011_vehicle_detection",
        "detect_from_bag.launch",
        f"bag_path:={_roslaunch_escape(args.bag_path)}",
        f"image_topic:={args.image_topic}",
        f"conf_threshold:={args.conf_threshold:.2f}",
        f"model:={args.model}",
    ]

    print("Launching:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print("\nTip: press Ctrl+C here to stop roslaunch.\n")

    # Replace current process so Ctrl+C works naturally
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

