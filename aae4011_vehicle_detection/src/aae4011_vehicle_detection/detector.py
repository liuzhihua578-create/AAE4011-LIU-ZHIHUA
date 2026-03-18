from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


VEHICLE_CLASSES_DEFAULT: Set[str] = {
    "car",
    "truck",
    "bus",
    "motorbike",
    "motorcycle",
    "train",
}


@dataclass(frozen=True)
class Detection:
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


class Detector:
    def detect(self, bgr_image) -> List[Detection]:
        raise NotImplementedError


class UltralyticsYoloDetector(Detector):
    """
    Thin wrapper around Ultralytics YOLO models.

    - Uses Ultralytics' built-in confidence filtering
    - Applies an additional class-name whitelist filter (vehicles by default)
    """

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.30,
        class_whitelist: Optional[Iterable[str]] = None,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO  # lazy import to keep ROS import time low

        self._model_name = model
        self._conf = float(conf)
        self._class_whitelist: Set[str] = (
            set(class_whitelist) if class_whitelist is not None else set(VEHICLE_CLASSES_DEFAULT)
        )
        self._yolo = YOLO(model)
        if device is not None:
            # Ultralytics accepts strings like "cpu", "0", "0,1"
            self._device = device
        else:
            self._device = None

    def set_confidence(self, conf: float) -> None:
        self._conf = float(conf)

    @property
    def model_name(self) -> str:
        return self._model_name

    def detect(self, bgr_image) -> List[Detection]:
        kwargs = {"verbose": False, "conf": self._conf}
        if self._device is not None:
            kwargs["device"] = self._device

        results = self._yolo(bgr_image, **kwargs)
        if not results:
            return []

        r0 = results[0]
        names: Dict[int, str] = r0.names
        dets: List[Detection] = []
        for box in r0.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name not in self._class_whitelist:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            dets.append(Detection(cls_name=cls_name, conf=conf, xyxy=(x1, y1, x2, y2)))
        return dets

