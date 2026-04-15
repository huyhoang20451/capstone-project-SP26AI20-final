import argparse
import json
import re
from pathlib import Path

import httpx
import numpy as np
from datasets import get_dataset_split_names, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm


def build_prompt(user_query: str) -> str:
    """Giữ format prompt gần với lúc training, yêu cầu trả JSON."""
    return f"""
<im_start>system
Bạn là AI nhận diện cảm xúc trong văn bản tiếng Việt.

Nhiệm vụ:
- Xác định cảm xúc của câu người dùng
- Sau đó phản hồi lại một câu phù hợp với cảm xúc đó

Trả về JSON:
{{
  "emotion": "...",
  "response": "..."
}}
<im_end>

<im_start>user
{user_query}
<im_end>

<im_start>assistant
"""


def extract_emotion(output_text: str) -> str | None:
    """Ưu tiên parse JSON; fallback regex nếu model trả text lẫn JSON."""
    text = (output_text or "").strip()
    if not text:
        return None

    # JSON thuần
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            emotion = data.get("emotion") or data.get("Emotion")
            if emotion:
                return str(emotion).strip()
    except Exception:
        pass

    # JSON trộn text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                emotion = data.get("emotion") or data.get("Emotion")
                if emotion:
                    return str(emotion).strip()
        except Exception:
            pass

    match = re.search(r'"emotion"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def call_ollama(
    client: httpx.Client,
    ollama_url: str,
    model_name: str,
    prompt: str,
    timeout: float,
) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    res = client.post(ollama_url, json=payload, timeout=timeout)
    res.raise_for_status()
    body = res.json()
    return str(body.get("response", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá Ollama trên 15% (mặc định) của split test từ Hugging Face dataset"
    )
    parser.add_argument("--dataset", required=True, help="Tên dataset trên Hugging Face, ví dụ: username/dataset")
    parser.add_argument("--config", default=None, help="Config name của dataset (nếu có)")
    parser.add_argument("--split", default="test", help="Split để đánh giá, mặc định: test")
    parser.add_argument("--text-column", default="user_query", help="Tên cột chứa input text")
    parser.add_argument("--label-column", default="Emotion", help="Tên cột nhãn thật")
    parser.add_argument("--sample-ratio", type=float, default=0.15, help="Tỉ lệ lấy mẫu từ split, mặc định 0.15")
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument("--model", default="tamly-model-withoutemotion", help="Tên model trong Ollama")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="API generate của Ollama")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout cho mỗi request")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn số mẫu sau khi sample (debug)")
    parser.add_argument("--output-dir", default="test_service/results", help="Thư mục lưu confusion matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0 < args.sample_ratio <= 1):
        raise ValueError("--sample-ratio phải nằm trong (0, 1].")

    print(f"Loading dataset: {args.dataset} | split={args.split}")
    split_names = get_dataset_split_names(args.dataset, args.config)

    if args.split in split_names:
        ds = load_dataset(args.dataset, args.config, split=args.split)
        if len(ds) == 0:
            raise RuntimeError("Split rỗng, không có dữ liệu để evaluate.")

        sample_size = max(1, int(len(ds) * args.sample_ratio))
        sampled = ds.shuffle(seed=args.seed).select(range(sample_size))
        print(f"Available splits: {split_names}")
        print(f"Total records in split '{args.split}': {len(ds)}")
        print(f"Sampled records: {len(sampled)} ({args.sample_ratio * 100:.1f}% of '{args.split}')")
    elif args.split == "test" and "train" in split_names:
        # Nhiều dataset HF chỉ có train; fallback lấy trực tiếp sample-ratio từ train để evaluate.
        train_ds = load_dataset(args.dataset, args.config, split="train")
        if len(train_ds) == 0:
            raise RuntimeError("Split train rỗng, không có dữ liệu để evaluate.")

        sample_size = max(1, int(len(train_ds) * args.sample_ratio))
        sampled = train_ds.shuffle(seed=args.seed).select(range(sample_size))
        print(f"Available splits: {split_names}")
        print(
            f"Split '{args.split}' không tồn tại. Fallback: dùng {args.sample_ratio * 100:.1f}% từ split 'train' làm tập evaluate."
        )
        print(f"Total records in split 'train': {len(train_ds)}")
        print(f"Sampled records: {len(sampled)}")
    else:
        raise ValueError(
            f"Unknown split '{args.split}'. Available splits: {split_names}. "
            "Hãy truyền --split phù hợp hoặc dùng dataset có split đó."
        )

    if args.max_samples is not None:
        sampled = sampled.select(range(min(len(sampled), args.max_samples)))

    y_true: list[str] = []
    y_pred: list[str] = []
    failed_count = 0

    with httpx.Client() as client:
        for sample in tqdm(sampled, desc="Evaluating with Ollama"):
            user_query = str(sample[args.text_column])
            true_label = str(sample[args.label_column]).strip()

            prompt = build_prompt(user_query)

            try:
                output_text = call_ollama(
                    client=client,
                    ollama_url=args.ollama_url,
                    model_name=args.model,
                    prompt=prompt,
                    timeout=args.timeout,
                )
                pred_label = extract_emotion(output_text)
            except Exception:
                failed_count += 1
                continue

            if not pred_label:
                failed_count += 1
                continue

            y_true.append(true_label)
            y_pred.append(pred_label)

    if not y_true:
        raise RuntimeError("Không thu được dự đoán hợp lệ nào từ Ollama.")

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print("\n===== Metrics =====")
    print(f"Valid predictions: {len(y_true)}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("===== Classification Matrix (Confusion Matrix) =====")
    print("Labels order:", labels)
    print(cm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "confusion_matrix.npy", cm)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    csv_lines = [",".join(["true\\pred"] + labels)]
    for i, row in enumerate(cm):
        csv_lines.append(",".join([labels[i]] + [str(int(v)) for v in row]))
    (output_dir / "confusion_matrix.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    print(f"\nSaved: {output_dir / 'confusion_matrix.npy'}")
    print(f"Saved: {output_dir / 'confusion_matrix.csv'}")
    print(f"Saved: {output_dir / 'labels.json'}")


if __name__ == "__main__":
    main()