from sentence_transformers import SentenceTransformer
sentences = ["Cô giáo đang ăn kem", "Chị gái đang thử món thịt dê"]

model = SentenceTransformer('keepitreal/vietnamese-sbert')
embeddings = model.encode(sentences)
print(embeddings)
