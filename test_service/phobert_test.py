import json
from pathlib import Path

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Đường dẫn tới thư mục bạn vừa tải về
MODEL_PATH = Path(r"C:\Users\User\Documents\web_emotion_chat\phobert-stage2\checkpoint-1638")


def has_model_weights(path: Path) -> bool:
    return any((path / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin"))


def resolve_model_path(path: Path) -> Path:
    if has_model_weights(path):
        return path

    parent = path.parent
    candidates = []
    for checkpoint_dir in parent.glob("checkpoint-*"):
        if checkpoint_dir.is_dir() and has_model_weights(checkpoint_dir):
            try:
                step = int(checkpoint_dir.name.split("-")[-1])
            except ValueError:
                step = -1
            candidates.append((step, checkpoint_dir))

    if not candidates:
        raise FileNotFoundError(
            f"Khong tim thay model.safetensors hoac pytorch_model.bin trong {path}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    print(f"Checkpoint '{path.name}' khong co weight, da chuyen sang '{selected.name}'.")
    return selected


MODEL_PATH = resolve_model_path(MODEL_PATH)

# Nạp Tokenizer (Bạn có thể dùng bản gốc nếu thư mục thiếu file vocab)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except:
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


def normalize_label_maps(config):
    id2label = getattr(config, "id2label", None)

    if isinstance(id2label, list):
        config.id2label = {idx: str(label) for idx, label in enumerate(id2label)}
    elif not isinstance(id2label, dict) or len(id2label) == 0:
        num_labels = getattr(config, "num_labels", 2)
        config.id2label = {idx: f"LABEL_{idx}" for idx in range(num_labels)}
    else:
        normalized = {}
        for key, value in id2label.items():
            try:
                normalized[int(key)] = str(value)
            except Exception:
                continue
        if len(normalized) == 0:
            num_labels = getattr(config, "num_labels", 2)
            normalized = {idx: f"LABEL_{idx}" for idx in range(num_labels)}
        config.id2label = normalized

    config.label2id = {label: idx for idx, label in config.id2label.items()}
    config.num_labels = len(config.id2label)
    return config


config_path = MODEL_PATH / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

raw_id2label = config_dict.get("id2label")
if isinstance(raw_id2label, list):
    config_dict["id2label"] = {idx: str(label) for idx, label in enumerate(raw_id2label)}
elif not isinstance(raw_id2label, dict):
    num_labels = int(config_dict.get("num_labels", 2))
    config_dict["id2label"] = {idx: f"LABEL_{idx}" for idx in range(num_labels)}
else:
    normalized = {}
    for key, value in raw_id2label.items():
        normalized[int(key)] = str(value)
    config_dict["id2label"] = normalized

config_dict["label2id"] = {label: idx for idx, label in config_dict["id2label"].items()}
config_dict["num_labels"] = len(config_dict["id2label"])

model_type = config_dict.pop("model_type", "roberta")
config = AutoConfig.for_model(model_type, **config_dict)
config = normalize_label_maps(config)

# Nạp Model (Tự động nhận diện safetensors và ép chạy trên CPU)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    config=config,
    low_cpu_mem_usage=True
)

device = torch.device("cpu")
model.to(device)
model.eval()

print("PhoBERT Stage 2 loaded successfully!")

def predict_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predicted_label = config.id2label.get(predicted_class_id, "Unknown")

    return predicted_label

if __name__ == "__main__":
    test_text = "Tôi cảm thấy rất vui hôm nay!"
    predicted_emotion = predict_emotion(test_text)
    print(f"Text: {test_text}")
    print(f"Predicted Emotion: {predicted_emotion}")