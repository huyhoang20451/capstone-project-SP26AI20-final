from sqlalchemy.orm import Session
from app.models.db import Conversation, Message

class HistoryService:
    @staticmethod
    def create_conversation(db: Session, first_msg: str):
        """Tạo một cuộc trò chuyện mới với tiêu đề từ tin nhắn đầu tiên."""
        title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
        new_conv = Conversation(title=title)
        db.add(new_conv)
        db.commit()
        db.refresh(new_conv)
        return new_conv

    @staticmethod
    def add_message(db: Session, conv_id: int, role: str, content: str, emotion: str = None, ml_detail_emotion: str = None):
        """Thêm một tin nhắn vào cuộc trò chuyện."""
        new_msg = Message(
            conversation_id=conv_id,
            role=role,
            content=content,
            emotion=emotion,
            ml_detail_emotion=ml_detail_emotion
        )
        db.add(new_msg)
        db.commit()
        db.refresh(new_msg)
        return new_msg

    @staticmethod
    def get_all_conversations(db: Session):
        """Lấy tất cả các cuộc trò chuyện."""
        return db.query(Conversation).order_by(Conversation.created_at.desc()).all()

    @staticmethod
    def get_messages_by_conv(db: Session, conv_id: int):
        """Lấy tất cả các tin nhắn của một cuộc trò chuyện."""
        return db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.timestamp.asc()).all()

    @staticmethod
    def get_all_history(db: Session):
        """Lấy danh sách tất cả các cuộc trò chuyện."""
        return db.query(Conversation).order_by(Conversation.created_at.desc()).all()

    @staticmethod
    def get_chat_detail(db: Session, chat_id: int):
        """Lấy chi tiết một cuộc trò chuyện theo ID."""
        return db.query(Conversation).filter(Conversation.id == chat_id).first()

    @staticmethod
    def get_conversation_by_id(db: Session, conversation_id: int):
        return db.query(Conversation).filter(Conversation.id == conversation_id).first()

    @staticmethod
    def save_chat(
        db: Session,
        user_msg: str,
        ai_res: str,
        emotion: str = None,
        ml_detail_emotion: str = None,
        conversation_id: int = None,
    ):
        """
        Lưu chat vào hội thoại đã chọn nếu có, nếu không thì tạo hội thoại mới.
        Trả về conversation và message phản hồi của assistant.
        """
        conv = None
        if conversation_id:
            conv = HistoryService.get_conversation_by_id(db, conversation_id)

        if conv is None:
            conv = HistoryService.create_conversation(db, user_msg)

        HistoryService.add_message(db, conv.id, "user", user_msg)

        ai_msg = HistoryService.add_message(
            db,
            conv.id,
            "assistant",
            ai_res,
            emotion=emotion,
            ml_detail_emotion=ml_detail_emotion,
        )

        return conv, ai_msg

history_service = HistoryService()