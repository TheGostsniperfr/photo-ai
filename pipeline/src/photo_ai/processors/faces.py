"""InsightFace buffalo_l — face detection and embedding."""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis


@dataclass
class DetectedFace:
    bbox: list[float]       # [x1, y1, x2, y2]
    embedding: np.ndarray   # 512-dim
    confidence: float


class FaceProcessor:
    def __init__(self, similarity_threshold: float = 0.45):
        self._threshold = similarity_threshold
        self._app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, path: Path) -> list[DetectedFace]:
        img = cv2.imread(str(path))
        if img is None:
            return []
        faces = self._app.get(img)
        return [
            DetectedFace(
                bbox=face.bbox.tolist(),
                embedding=face.embedding,
                confidence=float(face.det_score),
            )
            for face in faces
            if face.det_score > 0.5
        ]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
