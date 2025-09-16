from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.routes import trading, backtesting, ai_agent, market_data
from app.database import init_db

app = FastAPI(
    title="ICT Trading AI Agent",
    description="Comprehensive ICT trading AI agent with 50+ concepts and backtesting",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(backtesting.router, prefix="/api/backtesting", tags=["Backtesting"])
app.include_router(ai_agent.router, prefix="/api/ai", tags=["AI Agent"])
app.include_router(market_data.router, prefix="/api/data", tags=["Market Data"])

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    await init_db()

@app.get("/")
async def root():
    return {
        "message": "ICT Trading AI Agent API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ICT Trading AI Agent"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )