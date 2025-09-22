"""
FastAPI application for Student Performance Prediction
logging and enhanced features
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import os

from data import StudentData, PredictionResponse
from predict import StudentPerformancePredictor

# Setup logging
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict student grades based on study habits and performance metrics",
    version="1.0.0",
    contact={
        "name": "MLOps Lab Student",
        "email": "student@northeastern.edu",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = StudentPerformancePredictor()

# Request counter for monitoring
request_count = 0

@app.on_event("startup")
async def startup_event():
    """Log startup"""
    logger.info("Student Performance API started successfully")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Student Performance Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_performance(student: StudentData):
    """
    Predict student grade based on input features
    """
    global request_count
    request_count += 1
    
    try:
        # Log the request
        logger.info(f"Prediction request #{request_count}: {student.dict()}")
        
        # Make prediction
        predicted_grade, confidence, recommendation = predictor.predict(
            student.study_hours,
            student.attendance_rate,
            student.assignment_score,
            student.previous_grade
        )
        
        response = PredictionResponse(
            predicted_grade=predicted_grade,
            confidence=float(confidence),
            model_version=predictor.metadata.get('version', 'v1.0'),
            recommendation=recommendation
        )
        
        logger.info(f"Prediction successful: {predicted_grade} (confidence: {confidence:.2f})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "requests_processed": request_count,
        "model_loaded": predictor.model is not None
    }

@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    try:
        info = predictor.get_model_info()
        info['total_requests'] = request_count
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)