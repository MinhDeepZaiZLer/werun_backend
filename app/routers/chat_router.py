from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.chat_service import chat_service

router = APIRouter(tags=["Chat"])

@router.websocket("/ws/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await chat_service.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            await chat_service.handle_message(user_id, data)
    except WebSocketDisconnect:
        chat_service.disconnect(user_id)

@router.get("/api/v1/messages/{user_id}/{friend_id}")
def get_messages(user_id: str, friend_id: str):
    return chat_service.get_history(user_id, friend_id)