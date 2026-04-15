import os

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import datetime

# Thay thế bằng thông tin DB của bạn qua biến môi trường.
# Cấu trúc: postgresql://username:password@host:port/database_name
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Hoang399100@localhost:5432/web_emotion_chat")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Quan hệ với bảng Messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" hoặc "assistant"
    content = Column(Text, nullable=False)
    emotion = Column(String(50))
    ml_detail_emotion = Column(String(50))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()