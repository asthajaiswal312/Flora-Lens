"""Flower-classification adapter for Keras/TensorFlow and PyTorch ResNet50 models.

Heavy ML frameworks are imported only when a corresponding model file is present,
so the interface can still run in clearly labelled preview mode without them.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image


class ModelNotReady(RuntimeError):
    """Raised when inference is requested without a loaded model."""


class FlowerPredictor:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_path: Path | None = None
        self.class_names: list[str] = []
        self.labels_loaded = False
        self.ready = False
        self.error: str | None = None
        self.device = "cpu"
        self.framework: str | None = None
        self.config: dict[str, Any] = {}
        self._model: Any = None
        self._torch: Any = None
        self._tf: Any = None
        self._np: Any = None
        self._transform: Any = None
        self._input_size = (224, 224)
        self._lock = threading.Lock()
        self.reload()

    @property
    def mode(self) -> str:
        if self.ready:
            return "model"
        if self.error:
            return "error"
        return "preview"

    @property
    def class_count(self) -> int:
        if self.class_names:
            return len(self.class_names)
        return int(os.getenv("BLOOMWATCH_NUM_CLASSES", "104"))

    def _find_model_file(self) -> Path | None:
        configured = os.getenv("BLOOMWATCH_MODEL")
        if configured:
            path = Path(configured).expanduser().resolve()
            return path if path.is_file() else None
        for name in (
            "botanical_model.h5",
            "botanical_model.keras",
            "resnet50_flowers.h5",
            "resnet50_flowers.keras",
            "resnet50_flowers.pth",
            "resnet50_flowers.pt",
            "model.h5",
            "model.keras",
            "model.pth",
            "model.pt",
        ):
            path = self.model_dir / name
            if path.is_file():
                return path
        return None

    @staticmethod
    def _normalise_labels(raw: Any) -> list[str]:
        if isinstance(raw, dict):
            for key in ("classes", "class_names", "labels", "class_to_idx", "class_mapping", "mapping"):
                if key in raw:
                    return FlowerPredictor._normalise_labels(raw[key])
            # Mapping of label -> integer output index.
            if raw and all(isinstance(value, int) for value in raw.values()):
                return [str(key) for key, _ in sorted(raw.items(), key=lambda item: item[1])]
            # Mapping of output index -> label.
            if raw and all(str(key).isdigit() for key in raw):
                return [str(raw[key]) for key in sorted(raw, key=lambda key: int(key))]
        if isinstance(raw, (list, tuple)):
            return [str(value) for value in raw]
        return []

    def _read_labels(self) -> list[str]:
        configured = os.getenv("BLOOMWATCH_CLASSES")
        candidates = [Path(configured).expanduser()] if configured else []
        candidates += [
            self.model_dir / "class_mapping.json",
            self.model_dir / "class_mapping",
            self.model_dir / "classes.json",
            self.model_dir / "classes.txt",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8-sig").strip()
            if path.suffix.lower() == ".json" or text.startswith(("{", "[")):
                labels = self._normalise_labels(json.loads(text))
            else:
                labels = [line.strip() for line in text.splitlines() if line.strip()]
            if labels:
                self.labels_loaded = True
                return labels
        return []

    def _read_config(self) -> dict[str, Any]:
        configured = os.getenv("BLOOMWATCH_MODEL_CONFIG")
        path = Path(configured).expanduser() if configured else self.model_dir / "model_config.json"
        config: dict[str, Any] = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(loaded, dict):
                raise ValueError("model_config.json must contain a JSON object.")
            config = loaded
        if os.getenv("BLOOMWATCH_PREPROCESS"):
            config["preprocessing"] = os.getenv("BLOOMWATCH_PREPROCESS")
        if os.getenv("BLOOMWATCH_OUTPUT"):
            config["output"] = os.getenv("BLOOMWATCH_OUTPUT")
        return config

    @staticmethod
    def _state_dict_from_checkpoint(checkpoint: Any) -> Any:
        if not isinstance(checkpoint, dict):
            return None
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            return checkpoint
        return None

    def reload(self) -> None:
        self.ready = False
        self.error = None
        self.framework = None
        self._model = None
        self._transform = None
        self.class_names = []
        self.labels_loaded = False
        self.config = {}
        self.model_path = self._find_model_file()
        if self.model_path is None:
            return

        try:
            self.class_names = self._read_labels()
            if not self.class_names:
                raise ValueError(
                    "No ordered class labels found. Add model/class_mapping.json or model/classes.json."
                )
            self.config = self._read_config()
            if self.model_path.suffix.lower() in (".h5", ".hdf5", ".keras"):
                self._load_keras()
            else:
                self._load_pytorch()
            self.ready = True
        except Exception as exc:  # Show setup errors in the UI without crashing the dashboard.
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready = False

    def _load_keras(self) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import numpy as np
        import tensorflow as tf

        self._np = np
        self._tf = tf
        self.framework = "keras"
        self.device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
        self._model = tf.keras.models.load_model(str(self.model_path), compile=False)

        input_shape = self._model.input_shape
        output_shape = self._model.output_shape
        if isinstance(input_shape, list) or isinstance(output_shape, list):
            raise ValueError("This adapter currently supports single-input, single-output Keras models.")
        if output_shape and output_shape[-1] is not None and int(output_shape[-1]) != len(self.class_names):
            raise ValueError(
                f"Model outputs {output_shape[-1]} classes but the mapping contains {len(self.class_names)} labels."
            )

        configured_size = self.config.get("input_size")
        if configured_size is not None:
            if not isinstance(configured_size, (list, tuple)) or len(configured_size) != 2:
                raise ValueError("input_size in model_config.json must be [height, width].")
            self._input_size = (int(configured_size[0]), int(configured_size[1]))
        elif input_shape and len(input_shape) == 4:
            height = int(input_shape[1]) if input_shape[1] is not None else 224
            width = int(input_shape[2]) if input_shape[2] is not None else 224
            self._input_size = (height, width)

    def _load_pytorch(self) -> None:
        import torch
        import torchvision
        from torchvision import transforms

        self._torch = torch
        self.framework = "pytorch"
        requested_device = os.getenv("BLOOMWATCH_DEVICE", "cpu")
        self.device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location="cpu")

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            state_dict = self._state_dict_from_checkpoint(checkpoint)
            if state_dict is None:
                raise ValueError("The checkpoint does not contain a supported model state_dict.")
            state_dict = {
                (key[7:] if key.startswith("module.") else key): value
                for key, value in state_dict.items()
            }
            model = torchvision.models.resnet50(weights=None)
            in_features = model.fc.in_features
            if any(key.startswith("fc.1.") for key in state_dict):
                model.fc = torch.nn.Sequential(
                    torch.nn.Dropout(0.0), torch.nn.Linear(in_features, len(self.class_names))
                )
            elif any(key.startswith("fc.0.") for key in state_dict):
                model.fc = torch.nn.Sequential(torch.nn.Linear(in_features, len(self.class_names)))
            else:
                model.fc = torch.nn.Linear(in_features, len(self.class_names))
            model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()
        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _predict_keras(self, image: Any, top_k: int) -> list[dict[str, Any]]:
        np = self._np
        height, width = self._input_size
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        preprocessing = str(self.config.get("preprocessing", "resnet50")).lower()
        if preprocessing == "resnet50":
            array = self._tf.keras.applications.resnet50.preprocess_input(array)
        elif preprocessing in ("rescale", "rescale_1_255"):
            array = array / 255.0
        elif preprocessing == "imagenet":
            array = array / 255.0
            array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
        elif preprocessing == "none":
            pass
        else:
            raise ValueError(f"Unsupported preprocessing mode: {preprocessing}")
        batch = np.expand_dims(array, axis=0)
        with self._lock:
            output = self._model.predict(batch, verbose=0)
        scores = np.asarray(output, dtype=np.float32)
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        if scores.ndim != 1 or scores.shape[0] != len(self.class_names):
            raise ValueError(f"Unexpected model output shape: {np.asarray(output).shape}")

        output_mode = str(self.config.get("output", "auto")).lower()
        looks_like_probabilities = bool(
            np.all(scores >= 0) and np.isclose(float(np.sum(scores)), 1.0, atol=1e-3)
        )
        if output_mode == "logits" or (output_mode == "auto" and not looks_like_probabilities):
            exp_scores = np.exp(scores - np.max(scores))
            scores = exp_scores / np.sum(exp_scores)
        elif output_mode not in ("auto", "probabilities", "logits"):
            raise ValueError(f"Unsupported output mode: {output_mode}")
        else:
            scores = np.clip(scores, 0, None)
            total = float(np.sum(scores))
            if total <= 0:
                raise ValueError("Model returned an invalid all-zero probability vector.")
            scores = scores / total

        k = min(top_k, scores.shape[0])
        indices = np.argsort(scores)[-k:][::-1]
        return [
            {
                "species": self.class_names[int(index)],
                "class_index": int(index),
                "confidence": round(float(scores[index]), 6),
            }
            for index in indices
        ]

    def _predict_pytorch(self, image: Any, top_k: int) -> list[dict[str, Any]]:
        torch = self._torch
        with self._lock, torch.inference_mode():
            tensor = self._transform(image).unsqueeze(0).to(self.device)
            logits = self._model(tensor)
            scores = torch.softmax(logits, dim=1)[0]
            k = min(top_k, scores.shape[0])
            values, indices = torch.topk(scores, k=k)
        return [
            {
                "species": self.class_names[int(index)],
                "class_index": int(index),
                "confidence": round(float(value), 6),
            }
            for value, index in zip(values.cpu(), indices.cpu())
        ]

    def predict(self, image: Any, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.ready or self._model is None:
            raise ModelNotReady(self.error or "Model checkpoint is not connected.")
        if self.framework == "keras":
            return self._predict_keras(image, top_k)
        return self._predict_pytorch(image, top_k)
