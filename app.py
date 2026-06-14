import streamlit as st
import requests

# 1. Page Title and Setup
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Predictor")
st.write("Enter customer metrics below to check their risk of leaving.")

# 2. Configured Endpoint and Your API Key
API_URL = "https://customer-churn-api-4x2o.onrender.com/predict"  # Replace with your deployed API endpoint
API_KEY = "rnd_NDfdnceco1EFZ8rg5j9i1EiRdYnW"  # Keep or paste your key here

# 3. Create Frontend Form Fields matching your exact Model requirements
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=5000.0, step=100.0)
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=10, value=1)
        
    with col2:
        is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=500.0)
        complains = st.selectbox("Has Complained before?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    
    # Form Submit Button
    submit_button = st.form_submit_button("Predict Churn")

# 4. Handle Form Submission and API Call
if submit_button:
    # Build payload with exact names and capitalization expected by your API
    payload = {
        "Age": int(age),
        "Tenure_Months": int(tenure_months),
        "Balance": float(balance),
        "NumOfProducts": int(num_of_products),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": float(estimated_salary),
        "Complains": int(complains),
        "Satisfaction_Score": int(satisfaction_score)
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    with st.spinner("Analyzing customer data on Render..."):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Check for standard response keys or fallback to raw dictionary output
                prob = result.get("churn_probability", result.get("probability", None))
                is_churn = result.get("is_churn", result.get("prediction", None))
                
                st.success("Analysis Complete!")
                
                if prob is not None:
                    if prob <= 1.0: 
                        prob *= 100
                    st.metric(label="Churn Probability", value=f"{prob:.1f}%")
                
                if is_churn is True or is_churn == 1:
                    st.error("⚠️ High Risk Warning: This customer is likely to churn.")
                elif is_churn == False or is_churn == 0:
                    st.success("✅ Low Risk: This customer is likely to stay.")
                else:
                    # Raw API backup display if keys differ
                    st.write("API Response Data:")
                    st.json(result)
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"Could not connect to API: {e}")
