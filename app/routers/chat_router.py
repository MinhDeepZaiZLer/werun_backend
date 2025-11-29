# app/routers/chat_router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from app.services.chat_service import chat_service
from app.database import get_db # Import dependency

router = APIRouter()

@router.websocket("/ws/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await chat_service.connect(websocket, user_id)
    
    # Lưu ý: WebSocket không dùng Depends(get_db) trực tiếp được như HTTP
    # Ta phải tự tạo session thủ công hoặc dùng context manager
    from app.database import SessionLocal
    db = SessionLocal()
    
    try:
        while True:
            data = await websocket.receive_text()
            await chat_service.handle_message(user_id, data, db) # Truyền db vào
    except WebSocketDisconnect:
        chat_service.disconnect(user_id)
    finally:
        db.close() # Nhớ đóng DB connection

@router.get("/api/v1/messages/{user_id}/{friend_id}")
def get_messages(user_id: str, friend_id: str, db: Session = Depends(get_db)):
    return chat_service.get_history(user_id, friend_id, db) # Truyền db vào