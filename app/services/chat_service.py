# app/services/chat_service.py

from typing import List, Dict
from fastapi import WebSocket
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.models import MessageDB   # <-- MODEL DB chat
from app.services.security_service import encrypt_message, decrypt_message  # <-- THÊM DÒNG NÀY

class ChatService:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.message_history: List[dict] = [] 
        
        self.spam_corpus = [
            "mua ban nha dat", "co hoi dau tu", "trung thuong iphone",
            "click vao link", "vay von lai suat", "khuyen mai khung",
            "game bai doi thuong", "kiem tien online"
        ]
        self.vectorizer = TfidfVectorizer()
        self.spam_vectors = self.vectorizer.fit_transform(self.spam_corpus)

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        self.active_connections.pop(user_id, None)

    def is_spam(self, text: str) -> bool:
        if len(text) < 5:
            return False
        try:
            input_vec = self.vectorizer.transform([text.lower()])
            similarities = cosine_similarity(input_vec, self.spam_vectors)
            return similarities.max() > 0.4
        except:
            return False

    async def handle_message(self, user_id: str, raw_data: str, db):
        try:
            message_data = json.loads(raw_data)
            content = message_data.get("content", "")
            receiver_id = message_data.get("receiverId", "")

            if self.is_spam(content):
                await self.send_error(user_id, "Tin nhắn bị chặn (SPAM).")
                return

            # --- MÃ HÓA NỘI DUNG ---
            encrypted_content = encrypt_message(content)

            new_msg = MessageDB(
                sender_id=user_id,
                receiver_id=receiver_id,
                encrypted_content=encrypted_content,
                timestamp=datetime.now(),
                is_read=False
            )
            db.add(new_msg)
            db.commit()
            db.refresh(new_msg)

            final_msg = {
                "id": str(new_msg.id),
                "senderId": user_id,
                "receiverId": receiver_id,
                "content": content,
                "timestamp": new_msg.timestamp.isoformat(),
                "isRead": False
            }

            await self.send_to_user(receiver_id, final_msg)
            await self.send_to_user(user_id, final_msg) 

        except Exception as e:
            print("❌ Chat error:", e)

    def get_history(self, user1: str, user2: str, db):
        messages = db.query(MessageDB).filter(
            ((MessageDB.sender_id == user1) & (MessageDB.receiver_id == user2)) |
            ((MessageDB.sender_id == user2) & (MessageDB.receiver_id == user1))
        ).order_by(MessageDB.timestamp.asc()).all()
        
        return [
            {
                "id": str(m.id),
                "senderId": m.sender_id,
                "receiverId": m.receiver_id,
                "content": decrypt_message(m.encrypted_content),
                "timestamp": m.timestamp.isoformat(),
                "isRead": m.is_read
            }
            for m in messages
        ]

    async def send_to_user(self, user_id: str, message: dict):
        websocket = self.active_connections.get(user_id)
        if websocket:
            await websocket.send_json(message)

    async def send_error(self, user_id: str, error_msg: str):
        websocket = self.active_connections.get(user_id)
        if websocket:
            await websocket.send_json({"error": error_msg})\

chat_service = ChatService()