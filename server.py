"""Local FastAPI server for a configured SecanAi model artifact."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from settings import get_settings

LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()
MODEL: Any | None = None
TOKENIZER: Any | None = None
GENERATOR: Any | None = None
ACTIVE_MODEL: str | None = None
GENERATION_LOCK = asyncio.Semaphore(1)
INDEX_PATH = Path(__file__).resolve().with_name("index.html")


def resolve_model_source() -> str:
    """Prefer a trained local artifact; otherwise make the base-model fallback explicit."""
    assert SETTINGS.model_path is not None
    return str(SETTINGS.model_path) if SETTINGS.model_path.exists() else SETTINGS.base_model_id


def load_model() -> None:
    global MODEL, TOKENIZER, GENERATOR, ACTIVE_MODEL
    source = resolve_model_source()
    LOGGER.info("Loading model from %s", source)
    TOKENIZER = AutoTokenizer.from_pretrained(source)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    MODEL.eval()
    GENERATOR = pipeline("text-generation", model=MODEL, tokenizer=TOKENIZER)
    ACTIVE_MODEL = source


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield
    global MODEL, TOKENIZER, GENERATOR
    MODEL = TOKENIZER = GENERATOR = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="SecanAi Local Chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=SETTINGS.max_prompt_characters)
    max_new_tokens: int | None = Field(default=None, ge=1, le=SETTINGS.max_new_tokens)


@app.get("/health")
async def health() -> dict[str, str]:
    if GENERATOR is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return {"status": "ok"}


@app.get("/api/model")
async def model_metadata() -> dict[str, str | int]:
    return {
        "active_model": ACTIVE_MODEL or "not loaded",
        "configured_local_artifact": str(SETTINGS.model_path),
        "max_prompt_characters": SETTINGS.max_prompt_characters,
        "max_new_tokens": SETTINGS.max_new_tokens,
    }


def generate_response(prompt: str, max_new_tokens: int) -> str:
    assert TOKENIZER is not None and GENERATOR is not None
    messages = [{"role": "user", "content": prompt}]
    formatted = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = GENERATOR(
        formatted,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=TOKENIZER.pad_token_id,
        eos_token_id=TOKENIZER.eos_token_id,
    )
    return result[0]["generated_text"].strip()


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> dict[str, str]:
    if GENERATOR is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    try:
        async with GENERATION_LOCK:
            response = await asyncio.to_thread(
                generate_response, request.prompt.strip(), request.max_new_tokens or SETTINGS.max_new_tokens
            )
        return {"response": response or "I could not generate a response."}
    except Exception:
        LOGGER.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Generation failed. Check the local server log.") from None


@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse(INDEX_PATH)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=SETTINGS.host, port=SETTINGS.port, reload=False)
