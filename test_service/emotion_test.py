import joblib
import numpy as np
from underthesea import word_tokenize

# 1. Load bộ model bundle
loaded_bundle = joblib.load(r'C:\Users\User\Documents\web_emotion_chat\emotion_hierarchical_model.joblib')

# Trích xuất lại các thành phần
m_l1 = loaded_bundle['model_l1']
e_models = loaded_bundle['expert_models']
e_encoders = loaded_bundle['expert_encoders']
l_l1 = loaded_bundle['le_l1']
t_dict = loaded_bundle['teencode_dict']
embedder = loaded_bundle.get('embedder')

if embedder is None:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer('keepitreal/vietnamese-sbert')

# 2. Hàm Inference (Sử dụng các biến đã load)
def fast_inference(text, embedder):
    # Tiền xử lý nhanh
    text_clean = text.lower().strip()
    words = [t_dict.get(w, w) for w in text_clean.split()]
    text_segmented = word_tokenize(" ".join(words), format="text")
    
    # Vector hóa
    vector = embedder.encode([text_segmented])
    
    # Dự đoán Tầng 1
    pred_l1 = m_l1.predict(vector)[0]
    label_l1 = l_l1.inverse_transform([pred_l1])[0]
    
    # Kiểm tra chuyên gia
    if label_l1 in e_models:
        model_sub = e_models[label_l1]
        le_sub = e_encoders[label_l1]
        
        pred_sub = model_sub.predict(vector)[0]
        label_detail = le_sub.inverse_transform([pred_sub])[0]
        return label_l1, label_detail
    
    return label_l1, label_l1

# --- Chạy thử ---
# Lưu ý: embedder (SBERT) không lưu vào joblib vì nó rất nặng, 
# bạn nên khởi tạo lại: embedder = SentenceTransformer('keepitreal/vietnamese-sbert')

# Tính cosine similarity giữa hai nhãn cảm xúc
def calculate_cosine_similarity_between_two_labels(label_a, label_b, embedder):
    vec_a = embedder.encode([label_a])[0]
    vec_b = embedder.encode([label_b])[0]
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Ví dụ: So sánh "Bình thường" và "Tức giận"
similarity = calculate_cosine_similarity_between_two_labels("Bình thường", "Tức giận", embedder)
print(f"Cosine similarity giữa 'Bình thường' và 'Tức giận': {similarity:.4f}")

# Danh sách dữ liệu kiểm thử (Văn bản, L1 thực tế, Detail thực tế)
# 1. Định nghĩa bộ test data
test_data = [
    ("Tui bùn mún khóc lun á, hông ai hỉu tui hết...", "Negative_Sad", "Buồn bã"),
    ("Cái thói làm ăn vầy là tui chán ghét cực kì nha", "Negative_Sad", "Chán ghét"),
    ("Ngồi 1 mình giữa phố đông, thấy cô đơn vcl", "Negative_Sad", "Cô đơn"),
    ("Giá như hùi đó mình ko làm v, hối tiếc ghê", "Negative_Sad", "Hối tiếc"),
    ("Lo wá, ko bít mai đi thi có ổn áp ko nữa", "Negative_Sad", "Lo âu"),
    ("Trời ơi hnay tui zui quá xá là zui lunnn", "Positive", "Vui vẻ"),
    ("Mọi chuyện rùi sẽ ổn thui, cố lên tui ơi!", "Positive", "Lạc quan"),
    ("Wao, ko thể tin đc lun, bất ngờ thực sự!", "Surprise", "Ngạc nhiên"),
    ("Đm làm ăn như cc, bực cả mình, mún đấm ghê", "Anger", "Tức giận"),
    ("Sợ vcl, chỗ này nhìn âm u ghê rợn quá", "Fear", "Sợ hãi"),
    ("Cũng bth thui, ko có gì đặc biệt lắm", "Neutral_Other", "Trung lập"),
    ("Ủa alo, cái gì đang xảy ra v mọi người?", "Neutral_Other", "Other")
]

def run_test_and_accuracy(data, inference_func, embedder):
    correct_l1 = 0
    correct_detail = 0
    total = len(data)
    
    print(f"{'Văn bản đầu vào':<45} | {'Dự đoán (L1/Detail)':<30} | {'Check'}")
    print("-" * 90)
    
    for text, true_l1, true_detail in data:
        # Gọi hàm inference của bạn
        pred_l1, pred_detail = inference_func(text, embedder)
        
        is_l1_ok = (pred_l1 == true_l1)
        is_det_ok = (pred_detail == true_detail)
        
        if is_l1_ok: correct_l1 += 1
        if is_det_ok: correct_detail += 1
        
        icon = "✅" if is_det_ok else ("⚠️" if is_l1_ok else "❌")
        print(f"{text[:43]+'...':<45} | {f'{pred_l1}/{pred_detail}':<30} | {icon}")
    
    # Tính toán
    acc_l1 = (correct_l1 / total) * 100
    acc_det = (correct_detail / total) * 100
    
    print("-" * 90)
    print(f"📊 Accuracy Tầng 1 ({' -> '.join(set(g for _,g,_ in data))}): {acc_l1:.2f}%")
    print(f"📊 Accuracy Tầng 2 (Chi tiết cảm xúc): {acc_det:.2f}%")
    print(f"\nChú thích: ✅ Đúng hết | ⚠️ Đúng nhóm L1 nhưng sai Detail | ❌ Sai cả hai")

# 2. Thực thi
run_test_and_accuracy(test_data, fast_inference, embedder)
