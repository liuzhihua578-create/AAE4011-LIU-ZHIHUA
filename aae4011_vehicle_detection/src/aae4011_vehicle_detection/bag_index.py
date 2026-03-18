from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BagInfo:
    bag_path: str
    topic: str
    message_count: int
    duration_sec: float
    first_stamp: float
    last_stamp: float
    width: int
    height: int


def pick_best_compressed_image_topic(bag_path: str) -> Optional[Tuple[str, int]]:
    """
    Choose the CompressedImage topic with the most messages.
    Returns (topic, message_count) or None.
    """
    import rosbag

    bag = rosbag.Bag(bag_path, "r")
    try:
        info = bag.get_type_and_topic_info()
        best_topic = None
        best_count = 0
        for t, tinfo in info.topics.items():
            if tinfo.msg_type == "sensor_msgs/CompressedImage" and tinfo.message_count > best_count:
                best_topic = t
                best_count = tinfo.message_count
        if best_topic is None or best_count <= 0:
            return None
        return best_topic, best_count
    finally:
        bag.close()


def scan_bag_basic_info(bag_path: str, topic: Optional[str] = None) -> Optional[BagInfo]:
    """
    Read rosbag metadata and compute:
    - message_count (for topic)
    - duration (using header stamps)
    - resolution (decode first frame only)
    """
    import rosbag
    import numpy as np
    import cv2

    chosen = topic
    if not chosen:
        pick = pick_best_compressed_image_topic(bag_path)
        if not pick:
            return None
        chosen = pick[0]

    bag = rosbag.Bag(bag_path, "r")
    try:
        info = bag.get_type_and_topic_info()
        if chosen not in info.topics:
            return None
        message_count = int(info.topics[chosen].message_count)
        if message_count <= 0:
            return None

        first_stamp = None
        last_stamp = None
        first_data = None

        for _, msg, _ in bag.read_messages(topics=[chosen]):
            stamp = msg.header.stamp.to_sec() if hasattr(msg, "header") else None
            if stamp is None:
                continue
            if first_stamp is None:
                first_stamp = float(stamp)
                first_data = bytes(msg.data)
            last_stamp = float(stamp)

        if first_stamp is None or last_stamp is None:
            return None
        duration_sec = max(0.0, last_stamp - first_stamp)

        w = 0
        h = 0
        if first_data:
            arr = np.frombuffer(first_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]

        return BagInfo(
            bag_path=bag_path,
            topic=chosen,
            message_count=message_count,
            duration_sec=duration_sec,
            first_stamp=first_stamp,
            last_stamp=last_stamp,
            width=int(w),
            height=int(h),
        )
    finally:
        bag.close()


class StreamingBagFrameSource:
    """
    Streaming frame source for sensor_msgs/CompressedImage from a rosbag.

    Design goal: avoid decoding or loading all frames into memory.

    Trade-off: random seeking far backwards may require reopening the bag
    and iterating to the target index (acceptable for coursework bags and UI use).
    """

    def __init__(self, bag_path: str, topic: str, cache_size: int = 200):
        import rosbag

        self.bag_path = bag_path
        self.topic = topic
        self.cache_size = max(20, int(cache_size))

        self._bag = rosbag.Bag(bag_path, "r")
        self._iter = self._bag.read_messages(topics=[topic])
        self._cursor = 0

        # cache: index -> (stamp_sec, data_bytes)
        self._cache: Dict[int, Tuple[float, bytes]] = {}

        # timestamps for UI mapping (we keep float list; small memory footprint)
        self.stamps: List[float] = []
        self._fully_scanned = False

    def close(self) -> None:
        try:
            self._bag.close()
        except Exception:
            pass

    def _reopen(self) -> None:
        import rosbag

        self.close()
        self._bag = rosbag.Bag(self.bag_path, "r")
        self._iter = self._bag.read_messages(topics=[self.topic])
        self._cursor = 0
        self._cache.clear()

    def ensure_scanned(self, max_messages: Optional[int] = None) -> None:
        """
        Scan timestamps (without decoding) so UI can show duration and allow better seeks.
        If max_messages is provided, scan at most that many messages.
        """
        if self._fully_scanned:
            return
        target = int(max_messages) if max_messages is not None else None

        # Keep behavior deterministic: rebuild stamps from scratch once requested.
        self.stamps.clear()
        self._reopen()
        n = 0
        for _, msg, _ in self._bag.read_messages(topics=[self.topic]):
            self.stamps.append(float(msg.header.stamp.to_sec()))
            n += 1
            if target is not None and n >= target:
                break
        else:
            self._fully_scanned = True

        # After scan, reopen again for playback reads (keeps logic simple).
        self._reopen()

    def get(self, index: int) -> Optional[Tuple[float, bytes]]:
        """
        Return (stamp_sec, data_bytes) for a given frame index.
        Uses a small LRU-ish cache and sequential reading.
        """
        if index < 0:
            return None

        if index in self._cache:
            return self._cache[index]

        if index < self._cursor:
            # backwards seek: reopen and read forward
            self._reopen()

        # read until reach target index
        while self._cursor <= index:
            try:
                _, msg, _ = next(self._iter)
            except StopIteration:
                return None
            stamp = float(msg.header.stamp.to_sec())
            data = bytes(msg.data)
            self._cache[self._cursor] = (stamp, data)
            if len(self.stamps) <= self._cursor:
                self.stamps.append(stamp)

            self._cursor += 1

            # keep cache bounded
            if len(self._cache) > self.cache_size:
                # drop farthest entries from current window (simple policy)
                min_keep = max(0, self._cursor - self.cache_size)
                drop_keys = [k for k in self._cache.keys() if k < min_keep]
                for k in drop_keys[: max(1, len(drop_keys) // 2)]:
                    self._cache.pop(k, None)

        return self._cache.get(index)

