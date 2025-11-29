from fastapi import FastAPI
from app.routers import ai_router, chat_router

app = FastAPI(title="WeRun Backend")

app.include_router(ai_router.router)
app.include_router(chat_router.router)

@app.get("/")
def read_root():
    return {"message": "WeRun Backend (Modular) is Running!"}