from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import assessment, health, llm_router, materials, support, tests, verification

app = FastAPI(
    title="АиСД Learning Platform API",
    description="API для платформы изучения алгоритмов и структур данных с агентными системами",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(health.router)
app.include_router(verification.router)
app.include_router(assessment.router)
app.include_router(materials.router)
app.include_router(tests.router)
app.include_router(llm_router.router)
app.include_router(support.router)


@app.get("/")
async def root():
    return {
        "message": "АиСД Learning Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }
