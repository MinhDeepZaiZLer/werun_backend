from typing import List, Dict
from fastapi import WebSocket
import json
from datetime import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChatService:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # Lưu trữ tạm thời (Sau này thay bằng DB thật)
        self.message_history: List[dict] = [] 
        
        # Init Spam Filter
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
        print(f"🔌 [Chat] User {user_id} connected.")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            print(f"❌ [Chat] User {user_id} disconnected.")

    def is_spam(self, text: str) -> bool:
        if not text or len(text) < 5: return False
        try:
            input_vec = self.vectorizer.transform([text.lower()])
            similarities = cosine_similarity(input_vec, self.spam_vectors)
            return similarities.max() > 0.4
        except: return False

    async def handle_message(self, user_id: str, raw_data: str):
        try:
            message_data = json.loads(raw_data)
            content = message_data.get("content", "")
            receiver_id = message_data.get("receiverId", "")

            # 1. Check Spam
            if self.is_spam(content):
                await self.send_error(user_id, "Tin nhắn bị chặn do nghi ngờ SPAM.")
                return

            # 2. Đóng gói
            final_msg = {
                "id": str(random.randint(10000, 99999)),
                "senderId": user_id,
                "receiverId": receiver_id,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "isRead": False
            }

            # 3. Lưu DB 
            self.message_history.append(final_msg)

            # 4. Gửi Realtime
            await self.send_to_user(receiver_id, final_msg)
            await self.send_to_user(user_id, final_msg) # Echo lại

        except Exception as e:
            print(f"⚠️ Lỗi xử lý tin nhắn: {e}")

    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(json.dumps(message))

    async def send_error(self, user_id: str, error_msg: str):
        await self.send_to_user(user_id, {"type": "error", "content": error_msg})

    def get_history(self, user1: str, user2: str):
        return [
            m for m in self.message_history
            if (m['senderId'] == user1 and m['receiverId'] == user2) or
               (m['senderId'] == user2 and m['receiverId'] == user1)
        ]

# Singleton
chat_service = ChatService()