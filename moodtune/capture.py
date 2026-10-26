"""Threaded OpenCV camera capture."""

from __future__ import annotations

import threading
from typing import Optional

import cv2

from .config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH


class CameraStream:
    """Keep the newest camera frame available without blocking callers."""

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._camera: Optional[cv2.VideoCapture] = None
        self._frame = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.error: Optional[str] = None

    def start(self) -> "CameraStream":
        if self._thread and self._thread.is_alive():
            return self
        self._camera = cv2.VideoCapture(self.index)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._camera.isOpened():
            self._camera.release()
            self._camera = None
            raise RuntimeError(f"Unable to open camera {self.index}")
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_frames, name="camera", daemon=True)
        self._thread.start()
        return self

    def _read_frames(self) -> None:
        assert self._camera is not None
        while not self._stop.is_set():
            success, frame = self._camera.read()
            if not success:
                self.error = "Camera stopped returning frames"
                break
            with self._lock:
                self._frame = frame

    def read(self):
        """Return the latest frame, or ``None`` before the first frame arrives."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.stop()


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """Open a camera directly for simple scripts and backwards compatibility."""
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return camera