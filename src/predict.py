"""
Prediction module for Student Performance API
"""
import joblib
import numpy as np
import os

class StudentPerformancePredictor:
    def __init__(self):
        """Initialize the predictor with model and metadata"""
        model_path = os.path.join(os.path.dirname(__file__), "../model/student_model.pkl")
        metadata_path = os.path.join(os.path.dirname(__file__), "../model/model_metadata.pkl")
        
        self.model = joblib.load(model_path)
        
        # Load metadata if exists
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
        else:
            self.metadata = {'version': 'v1.0', 'model_type': 'RandomForestClassifier'}
    
    def predict(self, 
                study_hours: float,
                attendance_rate: float,
                assignment_score: float,
                previous_grade: float):
        """Make prediction for a student"""
        features = [[study_hours, attendance_rate, assignment_score, previous_grade]]
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get confidence (probability of predicted class)
        probabilities = self.model.predict_proba(features)[0]
        predicted_class_index = list(self.model.classes_).index(prediction)
        confidence = probabilities[predicted_class_index]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            prediction, study_hours, attendance_rate, assignment_score
        )
        
        return prediction, confidence, recommendation
    
    def _generate_recommendation(self, 
                                grade: str,
                                study_hours: float,
                                attendance: float,
                                assignments: float) -> str:
        """Generate personalized recommendations"""
        recommendations = []
        
        if grade in ['D', 'F']:
            if study_hours < 5:
                recommendations.append("Increase study hours to at least 5 per day")
            if attendance < 80:
                recommendations.append("Improve attendance to above 80%")
            if assignments < 70:
                recommendations.append("Focus on completing assignments with higher quality")
        elif grade == 'C':
            recommendations.append("Good progress! Consider joining study groups")
            if study_hours < 7:
                recommendations.append("Adding 1-2 more study hours could help reach grade B")
        elif grade == 'B':
            recommendations.append("Excellent work! Maintain consistency")
        else:  # Grade A
            recommendations.append("Outstanding performance! Keep it up!")
        
        return " | ".join(recommendations) if recommendations else "Keep up the good work!"
    
    def get_model_info(self):
        """Return model information"""
        return {
            "model_type": self.metadata.get('model_type', 'Unknown'),
            "version": self.metadata.get('version', 'Unknown'),
            "features": self.metadata.get('features', []),
            "accuracy": self.metadata.get('accuracy', 'Unknown')
        }