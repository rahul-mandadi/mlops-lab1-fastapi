# MLOps Lab 1: Student Performance Prediction API

## Overview

Production-ready FastAPI application for serving machine learning models with real-time predictions. This project demonstrates MLOps best practices including API design, model versioning, logging, and monitoring for a student performance prediction system.

## Features

- Real-time ML predictions using RandomForest classifier (82% accuracy)
- RESTful API with automatic documentation generation
- Input validation using Pydantic models
- Model versioning and metadata tracking
- Comprehensive logging for debugging and monitoring
- Health check and model information endpoints
- Personalized recommendations based on predictions

## Technology Stack

- **FastAPI**: Modern web framework for building APIs
- **Scikit-learn**: Machine learning library (RandomForest)
- **Pydantic**: Data validation using Python type annotations
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pandas/NumPy**: Data manipulation and processing

## Project Structure

```
mlops-lab1-fastapi/
├── src/
│   ├── main.py          # FastAPI application and endpoints
│   ├── data.py          # Pydantic models for request/response
│   ├── train.py         # Model training pipeline
│   └── predict.py       # Prediction logic and model loading
├── model/               # Saved model artifacts
│   ├── student_model.pkl
│   └── model_metadata.pkl
├── logs/                # Application logs
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/rahul-mandadi/mlops-lab1-fastapi.git
cd mlops-lab1-fastapi
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
cd src
python train.py
```

5. Start the API server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message and API information |
| POST | `/predict` | Make grade predictions |
| GET | `/health` | Health check and system status |
| GET | `/model/info` | Model metadata and information |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "study_hours": 6.0,
       "attendance_rate": 85.0,
       "assignment_score": 80.0,
       "previous_grade": 75.0
     }'
```

### Example Response

```json
{
  "predicted_grade": "B",
  "confidence": 0.87,
  "model_version": "v1.0",
  "recommendation": "Excellent work! Maintain consistency"
}
```

## Model Information

### Training Data
- Synthetic dataset of 1000 student records
- Features: study hours, attendance rate, assignment scores, previous grades
- Target: Letter grades (A, B, C, D, F)

### Performance Metrics
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 82%
- **Test Set Size**: 200 samples
- **Cross-validation**: 80/20 split

### Classification Report
```
Grade  Precision  Recall  F1-Score  Support
A      1.00       0.73    0.84      26
B      0.83       0.91    0.87      75
C      0.80       0.87    0.83      75
D      0.62       0.53    0.57      19
F      1.00       0.40    0.57      5
```

## Modifications from Base Lab

This implementation extends the original FastAPI lab with the following improvements:

| Feature | Original Lab | This Implementation |
|---------|--------------|---------------------|
| Use Case | Iris flower classification | Student performance prediction |
| Algorithm | Decision Tree | Random Forest Classifier |
| Accuracy | ~74% | 82% |
| Features | Basic API | Logging, recommendations, versioning |
| Documentation | Minimal | Comprehensive with examples |

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Logging
Application logs are stored in `logs/api.log` with the following format:
```
2025-01-14 10:30:45 - main - INFO - Prediction request #1: {...}
2025-01-14 10:30:45 - main - INFO - Prediction successful: B (confidence: 0.87)
```

## Future Enhancements

- Docker containerization for easier deployment
- CI/CD pipeline using GitHub Actions
- Authentication and rate limiting
- Database integration for prediction history
- A/B testing for model comparison
- Real-time monitoring dashboard
- Model retraining pipeline

## Course Information

- **Course**: DADS 7305 - Machine Learning Operations (MLOps)
- **Institution**: Northeastern University
- **Term**: Spring 2025
- **Professor**: Ramin Mohammadi

## Author

Rahul Reddy Mandadi  
Graduate Student, Data Science  
Northeastern University  

## License

This project is developed for educational purposes as part of the MLOps course at Northeastern University.