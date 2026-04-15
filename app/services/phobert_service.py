import json
import os
import torch
from pathlib import Path
from functools import lru_cache
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

class EmotionService:
    def __init__(self, model_path: str):
        self.device = torch.device("cpu")
        self.model_path = self._resolve_path(Path(model_path))
        
        # 1. Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # 2. Load & Fix Config (Tránh lỗi id2label bị thành list hoặc string key)
        self.config = self._load_config()
        self.config = self._normalize_config(self.config)

        # 3. Load Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            config=self.config,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def _resolve_path(self, path: Path) -> Path:
        """Tìm checkpoint mới nhất nếu đường dẫn chỉ định không có weights."""
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        if any((path / f).exists() for f in weight_files):
            return path
        
        checkpoints = []
        for checkpoint in path.parent.glob("checkpoint-*"):
            if any((checkpoint / f).exists() for f in weight_files):
                try:
                    step = int(checkpoint.name.split("-")[-1])
                except ValueError:
                    step = 0
                checkpoints.append((step, checkpoint))

        if not checkpoints:
            raise FileNotFoundError(f"Không tìm thấy model weights tại {path}")

        checkpoints.sort(key=lambda item: item[0], reverse=True)
        return checkpoints[0][1]

    def _normalize_config(self, config):
        """Đảm bảo id2label luôn là dict {int: str}."""
        id2label = getattr(config, "id2label", {0: "LABEL_0", 1: "LABEL_1"})
        
        if isinstance(id2label, list):
            id2label = {i: str(label) for i, label in enumerate(id2label)}
        
        # Ép kiểu key về int (phòng trường hợp json load ra key là string)
        normalized = {int(k): str(v) for k, v in id2label.items()}
        config.id2label = normalized
        config.label2id = {v: k for k, v in normalized.items()}
        return config

    def _load_config(self):
        """Tải config thô rồi chuẩn hóa id2label trước khi khởi tạo AutoConfig."""
        config_path = self.model_path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        raw_id2label = config_dict.get("id2label")
        if isinstance(raw_id2label, list):
            config_dict["id2label"] = {i: str(label) for i, label in enumerate(raw_id2label)}
        elif isinstance(raw_id2label, dict):
            config_dict["id2label"] = {int(k): str(v) for k, v in raw_id2label.items()}
        else:
            num_labels = int(config_dict.get("num_labels", 2))
            config_dict["id2label"] = {i: f"LABEL_{i}" for i in range(num_labels)}

        config_dict["label2id"] = {label: idx for idx, label in config_dict["id2label"].items()}
        config_dict["num_labels"] = len(config_dict["id2label"])

        model_type = config_dict.pop("model_type", "roberta")
        return AutoConfig.for_model(model_type, **config_dict)

    def predict(self, text: str) -> str:
        """Dự đoán cảm xúc từ văn bản."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            label_id = outputs.logits.argmax().item()
            return self.config.id2label.get(label_id, "Unknown")

# Khởi tạo instance duy nhất (Singleton) để dùng trong toàn bộ app
MODEL_PATH = os.getenv(
    "PHOBERT_MODEL_PATH",
    str(Path(__file__).resolve().parents[2] / "phobert-stage2" / "checkpoint-1638"),
)


@lru_cache(maxsize=1)
def get_emotion_service() -> EmotionService:
    return EmotionService(MODEL_PATH)