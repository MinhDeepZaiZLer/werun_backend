from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.database import Base
from datetime import datetime

class MessageDB(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(String, index=True)
    receiver_id = Column(String, index=True)
    encrypted_content = Column(String) # Chỉ lưu nội dung đã mã hóa
    timestamp = Column(DateTime, default=datetime.now)
    is_read = Column(Boolean, default=False)