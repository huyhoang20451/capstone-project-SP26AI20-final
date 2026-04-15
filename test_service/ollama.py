from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # Nếu bạn có dùng file css/js riêng
from pydantic import BaseModel
import httpx
import re

app = FastAPI()

# 1. Quan trọng: Khai báo thư mục templates
templates = Jinja2Templates(directory="app/templates")

class UserInput(BaseModel):
    message: str
    model: str | None = None

OLLAMA_URL = "http://localhost:11434/api/generate" # chỗ này dựa vào selection của người dùng, có thể để mặc định hoặc cập nhật sau 
DEFAULT_MODEL_NAME = "tamly-model-withoutemotion"

# Hàm bóc tách (giữ nguyên của bạn)
def parse_ai_response(text: str):
    emotion_match = re.search(r"Emotion:\s*(.*)", text, re.IGNORECASE)
    response_match = re.search(r"Response:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    emotion = emotion_match.group(1).strip() if emotion_match else "Bình thường"
    response = response_match.group(1).strip() if response_match else text.strip()
    return emotion, response

# 2. Route render trang HTML (Tránh lỗi 404 khi vào localhost:8000)
@app.get("/") # Thường load model ngay khi vào trang chủ
async def load_index(request: Request):
    OLLAMA_MODEL_URL = "http://localhost:11434/api/tags" # Lưu ý: endpoint đúng là /api/tags
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(OLLAMA_MODEL_URL, timeout=10.0)
            res.raise_for_status()
            # Lấy danh sách model từ Ollama
            models = res.json().get("models", [])
            
        # Truyền 'request' (viết thường) và danh sách 'models' vào HTML
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "models": models
        })
    except Exception as e:
        # Nếu lỗi (ví dụ Ollama chưa bật), vẫn cho vào trang nhưng models trống
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "models": [],
            "error": str(e)
        })

# 3. Route xử lý API (Phải khớp chính xác với lệnh fetch('/consult-api') trong HTML)
@app.post("/consult-api")
async def consult_api(data: UserInput):
    selected_model = data.model or DEFAULT_MODEL_NAME
    payload = {
        "model": selected_model,
        "prompt": data.message,
        "stream": False
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(OLLAMA_URL, json=payload, timeout=60.0)
            res.raise_for_status()
            raw_text = res.json().get("response", "")
            
            emotion, advice = parse_ai_response(raw_text)
            
            # Trả về đúng cấu trúc mà JavaScript đang đợi: result.data.advice
            return {
                "status": "success",
                "data": {
                    "emotion": emotion,
                    "advice": advice
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Lưu ý: Chạy localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)