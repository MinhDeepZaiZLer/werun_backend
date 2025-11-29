# app/services/security_service.py

from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# Lấy key
SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    raise ValueError(
        "❌ SECRET_KEY không tồn tại! "
        "Dùng Fernet.generate_key() để tạo key và đặt vào file .env"
    )

try:
    cipher = Fernet(SECRET_KEY.encode())
except Exception as e:
    raise ValueError(f"❌ SECRET_KEY không hợp lệ: {e}")

def encrypt_message(text: str) -> str:
    """Mã hóa văn bản"""
    return cipher.encrypt(text.encode()).decode()

def decrypt_message(encrypted_text: str) -> str:
    """Giải mã văn bản"""
    try:
        return cipher.decrypt(encrypted_text.encode()).decode()
    except Exception:
        return "[Lỗi giải mã]"
