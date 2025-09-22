"""
Training script for Student Performance Predictor
RandomForestClassifier model to predict student grades based on study hours, attendance rate, assignment scores, and previous grades.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime

def generate_student_data(n_samples=1000):
    """Generate synthetic student performance data"""
    np.random.seed(42)
    
    # Generate features
    study_hours = np.random.uniform(0, 12, n_samples)
    attendance_rate = np.random.uniform(50, 100, n_samples)
    assignment_score = np.random.uniform(40, 100, n_samples)
    previous_grade = np.random.uniform(40, 100, n_samples)
    
    # Calculate grades based on weighted formula
    overall_score = (
        study_hours * 3 + 
        attendance_rate * 0.3 + 
        assignment_score * 0.3 + 
        previous_grade * 0.2
    )
    
    # Convert to letter grades
    grades = []
    for score in overall_score:
        if score >= 90:
            grades.append('A')
        elif score >= 75:
            grades.append('B')
        elif score >= 60:
            grades.append('C')
        elif score >= 50:
            grades.append('D')
        else:
            grades.append('F')
    
    # Create DataFrame
    df = pd.DataFrame({
        'study_hours': study_hours,
        'attendance_rate': attendance_rate,
        'assignment_score': assignment_score,
        'previous_grade': previous_grade,
        'grade': grades
    })
    
    return df

def train_model():
    """Train the student performance prediction model"""
    print("Generating student performance data...")
    data = generate_student_data(1000)
    
    # Prepare features and labels
    X = data.drop('grade', axis=1)
    y = data['grade']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train RandomForestClassifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_dir = "../model"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "student_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save model metadata
    metadata = {
        'version': 'v1.0',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'accuracy': accuracy,
        'model_type': 'RandomForestClassifier',
        'features': list(X.columns)
    }
    
    metadata_path = os.path.join(model_dir, "model_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print("Model metadata saved")
    
    return model, accuracy

if __name__ == "__main__":
    train_model()