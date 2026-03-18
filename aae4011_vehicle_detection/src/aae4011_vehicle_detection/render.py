from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2

from .detector import Detection


@dataclass
class UiState:
    playing: bool
    speed: float
    conf: float
    font_scale: float


@dataclass
class FrameStats:
    frame_idx: int
    n_frames: int
    t_sec: float
    duration_sec: float
    vehicles_in_frame: int
    total_vehicles_seen: int
    per_class_total: Dict[str, int]
    fps_render: float


def _fmt_mmss(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def draw_detections(img_bgr, detections: Iterable[Detection], font_scale: float) -> None:
    fs = float(font_scale)
    thickness = max(1, int(2 * fs / 0.6))
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{det.cls_name} {det.conf:.2f}"
        cv2.putText(
            img_bgr,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * fs,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_player_overlay(img_bgr, ui: UiState, st: FrameStats) -> None:
    h, w = img_bgr.shape[:2]
    fs = float(ui.font_scale)
    thickness = max(1, int(2 * fs / 0.6))

    # Top HUD
    left = f"Frame {st.frame_idx + 1}/{st.n_frames}  Vehicles: {st.vehicles_in_frame}  Total: {st.total_vehicles_seen}"
    right = f"Conf: {ui.conf:.2f}  Speed: {ui.speed:.2f}x  FPS: {st.fps_render:.1f}"
    cv2.putText(img_bgr, left, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * fs, (80, 255, 80), thickness, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.75 * fs, thickness)
    cv2.putText(img_bgr, right, (max(10, w - rw - 10), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * fs, (255, 255, 255), thickness, cv2.LINE_AA)

    # Bottom translucent panel + progress bar
    panel_h = 64
    panel_y0 = h - panel_h
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, panel_y0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, img_bgr, 0.30, 0, img_bgr)

    bar_margin = 12
    bar_h = 14
    bar_y = h - 26
    bar_w = w - 2 * bar_margin
    cv2.rectangle(img_bgr, (bar_margin, bar_y), (bar_margin + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    if st.n_frames > 0:
        fill = int(bar_w * (st.frame_idx + 1) / st.n_frames)
        cv2.rectangle(img_bgr, (bar_margin, bar_y), (bar_margin + fill, bar_y + bar_h), (60, 200, 90), -1)

    time_text = f"{_fmt_mmss(st.t_sec)} / {_fmt_mmss(st.duration_sec)}"
    cv2.putText(img_bgr, time_text, (bar_margin, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * fs, (230, 230, 230), thickness, cv2.LINE_AA)

    # Per-class totals (right side, above panel)
    y = h - panel_h - 10
    for cls_name, cnt in sorted(st.per_class_total.items(), key=lambda x: (-x[1], x[0]))[:6]:
        cv2.putText(img_bgr, f"{cls_name}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60 * fs, (240, 240, 240), max(1, thickness - 1), cv2.LINE_AA)
        y -= int(18 * fs)

    if not ui.playing:
        text = "PAUSED  (Space to play)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.95 * fs, thickness + 1)
        x = (w - tw) // 2
        y = (h - th) // 2
        cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95 * fs, (30, 30, 30), thickness + 5, cv2.LINE_AA)
        cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95 * fs, (255, 255, 255), thickness + 1, cv2.LINE_AA)

