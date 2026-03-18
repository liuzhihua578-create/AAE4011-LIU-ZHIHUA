from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedFrame:
    bgr: np.ndarray
    width: int
    height: int


def decode_compressed_image(data: bytes) -> Optional[DecodedFrame]:
    """
    Decode sensor_msgs/CompressedImage.data bytes into OpenCV BGR image.
    Returns None if decode fails.
    """
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return DecodedFrame(bgr=img, width=w, height=h)

