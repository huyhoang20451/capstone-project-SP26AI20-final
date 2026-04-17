import asyncio
import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr

# Hugging Face Spaces thường không có Postgres sẵn.
os.environ.setdefault("DATABASE_URL", "sqlite:///./web_emotion_chat.db")

from app.config import EMOTION_MODELS
from app.models.db import SessionLocal, init_db
from app.services.history_service import history_service
from app.services.llm_service import llm_service
from app.services.ml_emotion_service import ml_emotion_service
from app.services.whisper_service import whisper_service


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return loop.run_until_complete(coro)

    # Fallback an toàn khi đang ở trong context có event loop chạy sẵn.
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def runner():
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as exc:
            error_box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_box:
        raise error_box["error"]

    return result_box.get("value")


def _resolve_emotion_model_key(selected_name: str | None) -> str:
    if not selected_name:
        return "ml"

    normalized = selected_name.strip().lower()
    if normalized in {"", "default"}:
        return "ml"

    for key, config in EMOTION_MODELS.items():
        if normalized in {key.lower(), str(config.get("name", "")).lower()}:
            return key

    if "phobert" in normalized:
        return "phobert"
    if "machine learning" in normalized or normalized == "ml":
        return "ml"

    return "ml"


def _safe_models() -> list[str]:
    try:
        models = _run_async(llm_service.get_available_models())
        names = [item.get("name") for item in models if item.get("name")]
        return names or [llm_service.default_model]
    except Exception:
        return [llm_service.default_model]


def _safe_emotion_models() -> list[str]:
    try:
        items = _run_async(llm_service.get_available_emotion_models())
        return ["default", *items]
    except Exception:
        return ["default", "Machine Learning", "PhoBERT Multitask"]


def _predict_emotion(text: str, selected_emotion_model: str) -> tuple[dict[str, Any], str | None]:
    selected_key = _resolve_emotion_model_key(selected_emotion_model)
    warning = None

    if selected_key == "phobert":
        try:
            from app.services.phobert_multitask_service import get_phobert_multitask_service

            phobert_service = get_phobert_multitask_service()
            return phobert_service.predict(text), warning
        except Exception as exc:
            warning = f"Không tải được PhoBERT ({exc}). Đang fallback về Machine Learning."

    return ml_emotion_service.predict(text), warning


def _build_emotion_footer(llm_emotion: str, ml_detail_emotion: str | None, similarity: float | None) -> str:
    lines = [f"LLM emotion: {llm_emotion}"]
    if ml_detail_emotion:
        lines.append(f"ML emotion: {ml_detail_emotion}")
    if similarity is not None:
        lines.append(f"Độ tương đồng cảm xúc: {similarity:.2%}")
    return "\n".join(["", "---", *lines])


def _chat(
    message: str,
    history: list[dict[str, str]],
    selected_model: str,
    selected_emotion_model: str,
    conversation_id: int | None,
):
    msg = (message or "").strip()
    if not msg:
        return history, conversation_id, "", "Vui lòng nhập nội dung trước khi gửi."

    db = SessionLocal()
    try:
        llm_response = _run_async(llm_service.generate_response(msg, selected_model))
        if llm_response.get("status") != "success":
            bot_text = f"Lỗi LLM: {llm_response.get('message', 'Unknown error')}"
            new_history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": bot_text},
            ]
            return new_history, conversation_id, "", ""

        ml_result, warning = _predict_emotion(msg, selected_emotion_model)
        llm_emotion = llm_response.get("emotion", "Bình thường")
        ml_emotion = ml_result.get("emotion", "Không xác định")
        ml_detail_emotion = ml_result.get("detail_emotion", ml_emotion)

        similarity = None
        try:
            embedder = ml_emotion_service._get_embedder()
            similarity = _run_async(
                llm_service.calculate_cosine_similarity_between_two_labels(
                    llm_emotion,
                    ml_detail_emotion,
                    embedder,
                )
            )
        except Exception:
            similarity = None

        advice = llm_response.get("advice", "")
        conv, _assistant_msg = history_service.save_chat(
            db=db,
            user_msg=msg,
            ai_res=advice,
            emotion=llm_emotion,
            ml_detail_emotion=ml_detail_emotion,
            conversation_id=conversation_id,
        )

        assistant_content = advice + _build_emotion_footer(llm_emotion, ml_detail_emotion, similarity)
        if warning:
            assistant_content = f"> {warning}\n\n" + assistant_content

        new_history = history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": assistant_content},
        ]
        return new_history, conv.id, "", ""
    except Exception as exc:
        fallback_text = f"Lỗi hệ thống: {exc}"
        new_history = history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": fallback_text},
        ]
        return new_history, conversation_id, "", ""
    finally:
        db.close()


def _transcribe(audio_path: str | None):
    if not audio_path:
        return "", "Chưa có audio để nhận dạng."

    result = whisper_service.transcribe_file(Path(audio_path))
    if result.get("status") != "success":
        return "", f"STT lỗi: {result.get('message', 'Unknown error')}"

    return result.get("text", "").strip(), "Nhận dạng giọng nói thành công."


def _clear_chat():
    return [], None, "", "Đã tạo cuộc hội thoại mới."


def build_app() -> gr.Blocks:
    init_db()

    model_choices = _safe_models()
    emotion_choices = _safe_emotion_models()

    with gr.Blocks(title="AI Mental Health Assistant") as demo:
        gr.Markdown("# AI Mental Health Assistant")
        gr.Markdown(
            "Ứng dụng tư vấn tâm lý tiếng Việt chạy bằng Gradio cho Hugging Face Spaces."
        )

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0] if model_choices else None,
                label="LLM Model",
                interactive=True,
            )
            emotion_dd = gr.Dropdown(
                choices=emotion_choices,
                value="default",
                label="Emotion Model",
                interactive=True,
            )

        chatbot = gr.Chatbot(
            value=[
                {
                    "role": "assistant",
                    "content": "Chào bạn, mình luôn ở đây để lắng nghe. Hôm nay bạn cảm thấy thế nào?",
                }
            ],
            label="Cuộc hội thoại",
            height=520,
        )

        status = gr.Markdown("")
        conversation_id_state = gr.State(value=None)

        with gr.Row():
            user_input = gr.Textbox(
                label="Tin nhắn",
                placeholder="Nhắn tin cho Trợ lý tâm lý...",
                lines=3,
                scale=5,
            )
            send_btn = gr.Button("Gửi", variant="primary", scale=1)

        with gr.Row():
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Speech-to-Text",
            )
            stt_btn = gr.Button("Nhận dạng giọng nói")
            new_chat_btn = gr.Button("Cuộc hội thoại mới")

        send_btn.click(
            _chat,
            inputs=[user_input, chatbot, model_dd, emotion_dd, conversation_id_state],
            outputs=[chatbot, conversation_id_state, user_input, status],
        )

        user_input.submit(
            _chat,
            inputs=[user_input, chatbot, model_dd, emotion_dd, conversation_id_state],
            outputs=[chatbot, conversation_id_state, user_input, status],
        )

        stt_btn.click(_transcribe, inputs=[audio_in], outputs=[user_input, status])

        new_chat_btn.click(
            _clear_chat,
            inputs=[],
            outputs=[chatbot, conversation_id_state, user_input, status],
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True, server_port=7860)