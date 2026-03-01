import os
import re
import json
import uuid
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Learning Dashboard AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Đọc cấu hình từ env (Groq) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"


def get_groq_client() -> OpenAI:
    """Groq client equivalent to `new Groq({ apiKey })` in JS."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not set in .env",
        )
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


class GenerateCardsRequest(BaseModel):
    content: str
    type: str = "flashcard"  # flashcard | qa


@app.post("/generate-cards", response_model=dict)
def generate_cards(req: GenerateCardsRequest):
    content = strip_html(req.content) or ""
    if not content:
        return {"cards": []}

    groq = get_groq_client()

    if req.type == "qa":
        system = (
            "You are a helpful teacher. Given a note text, generate short Q&A pairs for review. "
            "Return ONLY a valid JSON array of objects, each with keys: question, answer. "
            "Example: [{\"question\": \"...\", \"answer\": \"...\"}, ...]. No markdown, no explanation."
        )
        user = f"Note:\n{content[:8000]}\n\nGenerate 5-10 Q&A pairs in JSON array format."
    else:
        system = (
            "You are a helpful teacher. Given a note text, generate flash cards (term -> definition or concept -> explanation). "
            "Return ONLY a valid JSON array of objects, each with keys: question, answer. "
            "Example: [{\"question\": \"...\", \"answer\": \"...\"}, ...]. No markdown, no explanation."
        )
        user = f"Note:\n{content[:8000]}\n\nGenerate 5-10 flash cards in JSON array format."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        completion = groq.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.3,
            stream=False,
        )
        raw = completion.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        if not isinstance(data, list):
            data = [data]
        cards = []
        for item in data:
            q = (item.get("question") or item.get("q") or "").strip()
            a = (item.get("answer") or item.get("a") or "").strip()
            if q and a:
                cards.append({
                    "id": str(uuid.uuid4()),
                    "question": q,
                    "answer": a,
                    "type": req.type,
                })
        return {"cards": cards}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
def health():
    return {"ok": True}
