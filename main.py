import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize API
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting microfinance customer churn",
    version="1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in production; change to your specific frontend URL later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust model loading path logic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "powei_churn_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at expected path: {model_path}")

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model file. Error: {str(e)}")

# Input validation schema
class Customer(BaseModel):
    Age: int
    Tenure_Months: int
    Balance: float
    NumOfProducts: int
    IsActiveMember: int
    EstimatedSalary: float
    Complains: int
    Satisfaction_Score: int

@app.get("/")
def home():
    return {
        "message": "Customer Churn API is running smoothly",
        "status": "healthy"
    }

@app.post("/predict")
def predict_churn(customer: Customer):
    try:
        # Convert Pydantic model directly into a DataFrame
        data = pd.DataFrame([customer.model_dump()])
        
        # Run inference
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        # Format human-readable output
        result = "Customer will churn" if prediction == 1 else "Customer will stay"
        
        return {
            "prediction": int(prediction),
            "result": result,
            "churn_probability": round(float(probability), 3)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during prediction: {str(e)}"
        )

# Local running configuration (ignored by Render production runners)
if __name__ == "__main__":
    import uvicorn
    # Pull port from environment variable for deployment flexibility
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
