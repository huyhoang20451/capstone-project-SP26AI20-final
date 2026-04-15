from pathlib import Path
import sys

from app.models.db import init_db

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import endpoints

app = FastAPI(title="Vietnamese Emotion Chatbot")

@app.on_event("startup")
def on_startup():
    init_db() # Tự động tạo bảng trong Postgres nếu chưa có
    
# Mount thư mục static nếu có
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routes
app.include_router(endpoints.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)