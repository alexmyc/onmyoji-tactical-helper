"""Thread-Safe AI Model Lifecycle & CUDA Memory Manager for Onmyoji Tactical Helper.

Isolates YOLOv8 and Gemma-4 GGUF initialization, thread-pool capping, `torch.no_grad()`
inference, and deterministic VRAM reclamation (`empty_cache` + `ipc_collect`).
"""

import gc
import logging
import os
import threading
import time
from typing import Any

import numpy as np
from ultralytics import YOLO

from onmyoji_helper.decision.llm_engine import TacticalEngine
from onmyoji_helper.infra.paths import resolve_asset_path

logger = logging.getLogger("OnmyojiHelper")


class ModelLifecycleManager:
    """Manages lazy loading, thread-safe inference, and VRAM cleanup for YOLO and LLM models."""

    def __init__(self, yolo_filename: str = "best.pt") -> None:
        """Initializes model holders and independent YOLO / LLM locks."""
        self.yolo_filename: str = yolo_filename
        self.yolo_model: YOLO | None = None
        self.tactical_engine: TacticalEngine | None = None
        self.yolo_imgsz: int = 640

        self.yolo_lock: threading.Lock = threading.Lock()
        self.llm_lock: threading.Lock = threading.Lock()

    def ensure_loaded(self, should_abort_fn: Any | None = None) -> bool:
        """Lazily initializes YOLOv8 and TacticalEngine unless aborted.

        Returns:
            True if YOLOv8 model is ready for battle tracking, False if missing or aborted.
        """
        if callable(should_abort_fn) and should_abort_fn():
            return False

        with self.yolo_lock:
            if callable(should_abort_fn) and should_abort_fn():
                return False
            if self.yolo_model is None:
                model_path = resolve_asset_path(self.yolo_filename)
                if not os.path.isfile(model_path):
                    logger.error(f"❌ [ModelLifecycle] 找不到 YOLO 模型檔案: {model_path}")
                    return False
                try:
                    import torch

                    torch.set_num_threads(min(2, os.cpu_count() or 2))
                    self.yolo_imgsz = 640 if torch.cuda.is_available() else 480
                except Exception:
                    self.yolo_imgsz = 480
                self.yolo_model = YOLO(model_path)

        with self.llm_lock:
            if self.tactical_engine is None:
                if not (callable(should_abort_fn) and should_abort_fn()):
                    self.tactical_engine = TacticalEngine()

        return True

    def predict_yolo(self, screen_color: np.ndarray) -> Any | None:
        """Runs YOLOv8 unit detection under `torch.no_grad()` and `yolo_lock`."""
        with self.yolo_lock:
            if self.yolo_model is None:
                return None
            try:
                import torch

                with torch.no_grad():
                    return self.yolo_model.predict(
                        screen_color,
                        imgsz=self.yolo_imgsz,
                        conf=0.30,
                        iou=0.45,
                        verbose=False,
                    )
            except Exception:
                return self.yolo_model.predict(
                    screen_color,
                    imgsz=self.yolo_imgsz,
                    conf=0.30,
                    iou=0.45,
                    verbose=False,
                )

    def run_llm_advice(self, payload: dict) -> dict | None:
        """Runs tactical reasoning under `llm_lock`."""
        with self.llm_lock:
            if self.tactical_engine is None:
                return None
            return self.tactical_engine.get_tactical_advice(payload, mode="LLM")

    @staticmethod
    def purge_cuda_cache() -> None:
        """Flushes unused PyTorch CUDA caching allocator blocks back to the OS driver."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    def unload_all(self) -> None:
        """Releases both YOLO and Gemma-4 models and reclaims CUDA VRAM."""
        with self.yolo_lock, self.llm_lock:
            if self.tactical_engine is not None:
                try:
                    self.tactical_engine.release()
                except Exception:
                    pass
                self.tactical_engine = None

            if self.yolo_model is not None:
                try:
                    if hasattr(self.yolo_model, "model"):
                        del self.yolo_model.model
                    del self.yolo_model
                except Exception:
                    pass
                self.yolo_model = None

            self.purge_cuda_cache()
            gc.collect()
            time.sleep(0.25)
