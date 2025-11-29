werun_backend/
│
├── gnn_model.pth            <-- File model AI (để ở root)
├── requirements.txt
│
└── app/                     <-- Thư mục code chính
    ├── __init__.py
    ├── main.py              <-- Chỉ dùng để khởi động app và nối các Router
    │
    ├── models/              <-- Chứa định nghĩa dữ liệu (Pydantic)
    │   ├── __init__.py
    │   └── schemas.py       <-- RouteRequest, RouteResponse...
    │
    ├── services/            <-- Chứa Logic xử lý (Core)
    │   ├── __init__.py
    │   ├── ai_service.py    <-- Logic GNN, OSMnx, tìm đường
    │   └── chat_service.py  <-- Logic Socket, Spam Filter, DB
    │
    └── routers/             <-- Chứa các API Endpoint
        ├── __init__.py
        ├── ai_router.py     <-- Endpoint /api/v1/generate_route
        └── chat_router.py   <-- Endpoint /ws/chat, /api/v1/messages