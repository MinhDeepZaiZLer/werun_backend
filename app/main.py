# app/main.py

from dotenv import load_dotenv
load_dotenv()  # ← Load .env NGAY LẬP TỨC, trước mọi import khác!!!

from fastapi import FastAPI
from app.database import engine
from app import models
from app.routers import ai_router, chat_router
from app.models.models import Base  # ← trực tiếp
from app.database import engine

Base.metadata.create_all(bind=engine)


app = FastAPI(title="WeRun Backend")

# Router
app.include_router(ai_router.router)
app.include_router(chat_router.router)

@app.get("/")
def read_root():
    return {"message": "WeRun Backend (Modular) is Running!"}
