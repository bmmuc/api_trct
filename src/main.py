"""
Time Series Anomaly Detection API
Main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Anomaly Detection API",
    description="API for training and inference of anomaly detection models on time series data",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Time Series Anomaly Detection API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }
