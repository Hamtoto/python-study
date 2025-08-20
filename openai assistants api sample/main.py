from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import chat, threads, messages
import uvicorn

app = FastAPI(
    title="OpenAI Assistants API Backend",
    description="FastAPI backend for OpenAI Assistants API",
    version="1.0.0",
    debug=settings.debug
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(threads.router, prefix="/api/v1", tags=["Threads"])
app.include_router(messages.router, prefix="/api/v1", tags=["Messages"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "OpenAI Assistants API Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )