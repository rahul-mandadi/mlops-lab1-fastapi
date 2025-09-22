"""
Data models for the Student Performance API
student performance prediction
"""
from pydantic import BaseModel, Field
from typing import Optional

class StudentData(BaseModel):
    """Input data for student performance prediction"""
    study_hours: float = Field(..., ge=0, le=24, description="Daily study hours")
    attendance_rate: float = Field(..., ge=0, le=100, description="Attendance percentage")
    assignment_score: float = Field(..., ge=0, le=100, description="Average assignment score")
    previous_grade: float = Field(..., ge=0, le=100, description="Previous semester grade")
    
    class Config:
        schema_extra = {
            "example": {
                "study_hours": 5.5,
                "attendance_rate": 85.0,
                "assignment_score": 78.5,
                "previous_grade": 75.0
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_grade: str
    confidence: float
    model_version: str
    recommendation: Optional[str] = None