from app.models.db import engine
from sqlalchemy import text

with engine.connect() as conn:
    # Kiểm tra xem cột đã tồn tại chưa
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='messages' AND column_name='emotion_model_used'"))
    if not result.fetchone():
        # Thêm cột nếu chưa tồn tại
        conn.execute(text("ALTER TABLE messages ADD COLUMN emotion_model_used VARCHAR(100) DEFAULT 'Machine Learning'"))
        conn.commit()
        print('Column emotion_model_used added successfully!')
    else:
        print('Column emotion_model_used already exists!')
