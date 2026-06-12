from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Initialize API
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting microfinance customer churn",
    version="1.0"
)


# Load model
model = joblib.load("powei_churn_model.pkl")


# Input schema
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
        "message": "Customer Churn API is running"
    }



@app.post("/predict")
def predict_churn(customer: Customer):

    # convert input to dataframe
    
    data = pd.DataFrame([{
        "Age": customer.Age,
        "Tenure_Months": customer.Tenure_Months,
        "Balance": customer.Balance,
        "NumOfProducts": customer.NumOfProducts,
        "IsActiveMember": customer.IsActiveMember,
        "EstimatedSalary": customer.EstimatedSalary,
        "Complains": customer.Complains,
        "Satisfaction_Score": customer.Satisfaction_Score
    }])


    # prediction
    
    prediction = model.predict(data)[0]


    # probability
    
    probability = model.predict_proba(data)[0][1]


    if prediction == 1:
        result = "Customer will churn"
    else:
        result = "Customer will stay"



    return {
        "prediction": int(prediction),
        "result": result,
        "churn_probability": round(float(probability),3)
    }
