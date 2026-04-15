import httpx
import re
import json
from typing import List, Dict, Tuple, Optional, AsyncIterator
import numpy as np
from app.config import ENABLED_EMOTION_MODELS, OLLAMA_BASE_URL, DEFAULT_LLM_MODEL

class LLMService:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.tags_url = f"{base_url}/api/tags"
        self.default_model = DEFAULT_LLM_MODEL

    async def get_available_models(self) -> List[Dict]:
        """Lấy danh sách các model hiện có trong Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(self.tags_url, timeout=5.0)
                res.raise_for_status()
                return res.json().get("models", [])
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return []

    async def get_available_emotion_models(self) -> List[str]:
        """Lấy danh sách các tên mô hình cảm xúc được cấu hình."""
        return ENABLED_EMOTION_MODELS
        
    
    def _strip_markdown_fence(self, text: str) -> str:
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return raw

    def _extract_json_payload(self, text: str) -> Optional[Dict]:
        candidate = self._strip_markdown_fence(text)

        # Trường hợp response là JSON thuần.
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Trường hợp response có thêm tiền tố/hậu tố, trích object JSON đầu tiên.
        start = candidate.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(candidate)):
            ch = candidate[i]

            if escaped:
                escaped = False
                continue

            if ch == "\\":
                escaped = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = candidate[start:i + 1]
                    try:
                        parsed = json.loads(snippet)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return None

        return None

    def _parse_ai_response(self, text: str) -> Tuple[str, str]:
        """Xử lý tách Emotion và Response từ chuỗi raw của AI."""
        raw = text.strip()

        payload = self._extract_json_payload(raw)
        if payload is not None:
            emotion = str(payload.get("Emotion") or payload.get("emotion") or "Bình thường").strip()
            advice = str(payload.get("Response") or payload.get("response") or "").strip()
            if advice:
                return emotion or "Bình thường", advice
            return emotion or "Bình thường", raw

        # 2) Fallback cho format text: Emotion: ... / Response: ...
        emotion_match = re.search(r"\"?Emotion\"?\s*:\s*\"?([^\"\n]+)\"?", raw, re.IGNORECASE)
        response_match = re.search(r"\"?Response\"?\s*:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)

        emotion = emotion_match.group(1).strip() if emotion_match else "Bình thường"
        advice = response_match.group(1).strip().strip('"') if response_match else raw

        return emotion, advice

    async def generate_response(self, message: str, model_name: Optional[str] = None) -> Dict:
        """Gửi prompt tới Ollama và trả về kết quả đã bóc tách."""
        selected_model = model_name or self.default_model
        payload = {
            "model": selected_model,
            "prompt": message,
            "stream": False
        }

        async with httpx.AsyncClient() as client:
            try:
                res = await client.post(self.generate_url, json=payload, timeout=60.0)
                res.raise_for_status()
                raw_text = res.json().get("response", "")
                
                emotion, advice = self._parse_ai_response(raw_text)
                
                return {
                    "status": "success",
                    "emotion": emotion,
                    "advice": advice
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"LLM Error: {str(e)}"
                }

    async def generate_response_stream(self, message: str, model_name: Optional[str] = None) -> AsyncIterator[Dict]:
        """Stream phản hồi từ Ollama theo từng chunk, sau đó trả về kết quả đã parse."""
        selected_model = model_name or self.default_model
        payload = {
            "model": selected_model,
            "prompt": message,
            "stream": True,
        }

        aggregated_text = ""

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.generate_url, json=payload, timeout=120.0) as res:
                    res.raise_for_status()

                    async for line in res.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk_payload = json.loads(line)
                        except Exception:
                            continue

                        chunk_text = chunk_payload.get("response", "")
                        if chunk_text:
                            aggregated_text += chunk_text
                            yield {
                                "type": "chunk",
                                "content": chunk_text,
                            }

                        if chunk_payload.get("done"):
                            break

            emotion, advice = self._parse_ai_response(aggregated_text)
            yield {
                "type": "final",
                "emotion": emotion,
                "advice": advice,
                "raw_text": aggregated_text,
                "show_emotion": True,
                "reliability_score": 1.0,
                "model_used": selected_model,
            }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"LLM Error: {str(e)}",
            }

    async def calculate_cosine_similarity_between_two_labels(self, label_a: str, label_b: str, embedder) -> float:
        """Tính cosine similarity giữa hai nhãn cảm xúc."""
        vec_a = embedder.encode([label_a])[0]
        vec_b = embedder.encode([label_b])[0]
        
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
# Khởi tạo instance
llm_service = LLMService()