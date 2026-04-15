import pickle
from pathlib import Path
from typing import Any
import warnings

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.exceptions import InconsistentVersionWarning


MODEL_PATH = Path("C:/Users/User/Documents/web_emotion_chat/highest_accuracy_model.pkl")

LABEL_MAP = {
    0: "Buồn bã",
    1: "Lo âu",
    2: "Lạc quan",
    3: "Cô đơn",
    4: "Other",
    5: "Vui vẻ",
    6: "Chán ghét",
    7: "Ngạc nhiên",
    8: "Sợ hãi",
    9: "Tức giận",
    10: "Highly Negative",
    11: "Trung lập",
    12: "Hối tiếc",
}

EMOTION_PROFILE = {
    "Buồn bã": "buồn bã mất mát đau lòng chán nản cô đơn suy sụp",
    "Lo âu": "lo âu bất an căng thẳng hồi hộp sợ hãi",
    "Lạc quan": "lạc quan tích cực hy vọng tin tưởng vui vẻ",
    "Cô đơn": "cô đơn trống vắng thiếu kết nối cần được chia sẻ",
    "Other": "cảm xúc khác không rõ ràng bình thường",
    "Vui vẻ": "vui vẻ hạnh phúc phấn khởi thoải mái",
    "Chán ghét": "chán ghét khó chịu bực bội mệt mỏi mất hứng",
    "Ngạc nhiên": "ngạc nhiên bất ngờ sửng sốt khó tin",
    "Sợ hãi": "sợ hãi run rẩy hoảng sợ lo sợ",
    "Tức giận": "tức giận bực tức nóng nảy phẫn nộ",
    "Highly Negative": "rất tiêu cực đau khổ tuyệt vọng nặng nề",
    "Trung lập": "trung lập bình thường ổn định không quá tích cực",
    "Hối tiếc": "hối tiếc ăn năn ân hận tiếc nuối",
}


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)

def calculate_cosine_similarity(text_a: str, text_b: str) -> float:
    """Tính cosine similarity giữa hai chuỗi văn bản."""
    vectorizer = HashingVectorizer(n_features=5000, alternate_sign=False, norm="l2")
    matrix = vectorizer.transform([text_a, text_b])
    vec_a = matrix[0]
    vec_b = matrix[1]

    dot_product = vec_a.multiply(vec_b).sum()
    norm1 = (vec_a.multiply(vec_a).sum()) ** 0.5
    norm2 = (vec_b.multiply(vec_b).sum()) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def label_to_text(label_id: int) -> str:
    label_name = LABEL_MAP.get(label_id, "Trung lập")
    return EMOTION_PROFILE.get(label_name, label_name)

def test_text_model(text: str) -> int:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    model = _load_pickle(MODEL_PATH)

    # Case 1: sklearn Pipeline -> predict trực tiếp từ text
    if hasattr(model, "named_steps") or hasattr(model, "steps"):
        prediction = model.predict([text])
        print("Predicted class:", prediction[0])
        return int(prediction[0])

    # Case 2: object có vectorizer riêng bên trong
    if hasattr(model, "vectorizer") and hasattr(model, "predict"):
        encoded_text = model.vectorizer.transform([text])
        prediction = model.predict(encoded_text)
        print("Predicted class:", prediction[0])
        return int(prediction[0])

    # Case 3: bare estimator (ví dụ SVC) -> không nhận text raw
    if hasattr(model, "predict"):
        expected_features = getattr(model, "n_features_in_", 5000)
        vectorizer = HashingVectorizer(
            n_features=expected_features,
            alternate_sign=False,
            norm="l2",
        )
        encoded_text = vectorizer.transform([text])
        prediction = model.predict(encoded_text)
        print("Model loaded:", type(model).__name__)
        print(
            "Warning: Dang dung HashingVectorizer fallback vi model khong co vectorizer di kem. "
            "Ket qua chi mang tinh tham khao."
        )
        print("Predicted class:", prediction[0])
        return int(prediction[0])

    raise ValueError(f"Unsupported model type: {type(model)}")


if __name__ == "__main__":
    sample_text = "Hôm nay tôi vui vẻ vì được đi chơi với bạn bè, nhưng cũng hơi lo lắng về công việc sắp tới."
    expected_class = 1
    ml_response = test_text_model(sample_text)
    cosine_score = calculate_cosine_similarity(
        label_to_text(ml_response),
        label_to_text(expected_class),
    )

    print(f"Predicted class: {ml_response}")
    print(f"Expected class: {expected_class}")
    print(f"Cosine similarity between labels: {cosine_score:.4f}")
    