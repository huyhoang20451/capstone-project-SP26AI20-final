from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
from underthesea import word_tokenize


class MLEmotionService:
    def __init__(self, bundle_path: Optional[str] = None):
        # Resolve model path relative to project root by default.
        root_dir = Path(__file__).resolve().parents[2]
        self.bundle_path = Path(bundle_path) if bundle_path else root_dir / "emotion_hierarchical_model.joblib"

        loaded_bundle = joblib.load(str(self.bundle_path))

        self.model_l1 = loaded_bundle["model_l1"]
        self.expert_models = loaded_bundle["expert_models"]
        self.expert_encoders = loaded_bundle["expert_encoders"]
        self.label_encoder_l1 = loaded_bundle["le_l1"]
        self.teencode_dict: Dict[str, str] = loaded_bundle["teencode_dict"]
        self.embedder = loaded_bundle.get("embedder")

    def _get_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("keepitreal/vietnamese-sbert")
        return self.embedder

    def _preprocess_text(self, text: str) -> str:
        text_clean = text.lower().strip()
        words = [self.teencode_dict.get(w, w) for w in text_clean.split()]
        return word_tokenize(" ".join(words), format="text")

    def predict(self, text: str) -> Dict[str, str]:
        label_l1, label_detail = self.predict_labels(text)
        return {
            "emotion": label_l1,
            "detail_emotion": label_detail,
        }

    def predict_labels(self, text: str) -> Tuple[str, str]:
        embedder = self._get_embedder()
        text_segmented = self._preprocess_text(text)

        vector = embedder.encode([text_segmented])

        pred_l1 = self.model_l1.predict(vector)[0]
        label_l1 = self.label_encoder_l1.inverse_transform([pred_l1])[0]

        if label_l1 in self.expert_models:
            model_sub = self.expert_models[label_l1]
            encoder_sub = self.expert_encoders[label_l1]
            pred_sub = model_sub.predict(vector)[0]
            label_detail = encoder_sub.inverse_transform([pred_sub])[0]
            return label_l1, label_detail

        return label_l1, label_l1


ml_emotion_service = MLEmotionService()
