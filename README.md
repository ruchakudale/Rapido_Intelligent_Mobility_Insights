# 🚖 Rapido Intelligent Mobility Insights  
### Ride Patterns • Cancellations • Fare Forecasting • Driver Analytics

An end-to-end Machine Learning project that transforms ride-hailing data into predictive intelligence using **4 ML models + Streamlit Dashboard**.

---

# 📌 Problem Statement

Ride-hailing platforms generate millions of bookings daily.  
Key operational challenges include:

- High ride cancellations  
- Inaccurate fare estimation  
- Inefficient driver allocation  
- Poor customer experience during peak demand  

This project builds a **unified ML-driven decision system** to predict ride outcomes, optimize pricing, and reduce operational risk.

---

# 🎯 Project Objectives

✔ Predict ride outcome before trip start  
✔ Estimate accurate fares dynamically  
✔ Identify high-risk customers & drivers  
✔ Enable data-driven operational decisions  

---

# 🧠 Machine Learning Models

| Model | Type | Purpose |
|---|---|---|
| Ride Outcome Model | Multi-Class Classification (XGBoost) | Predict Completed / Cancelled / Incomplete rides |
| Fare Prediction Model | Regression (RandomForest) | Predict ride fare |
| Customer Cancellation Risk | Binary Classification | Predict cancellation probability |
| Driver Delay Risk | Binary Classification | Predict delay/incomplete ride risk |

---

# 📊 Streamlit Dashboard

The project is deployed as an **interactive analytics & prediction dashboard**.

## 🔮 Prediction Modules

### 1️⃣ Ride Outcome Prediction
Predict if a ride will:
- Complete
- Cancel
- Become incomplete  

Includes:
- Real-time prediction  
- Probability output  
- Feature importance visualization  

---

### 2️⃣ Fare Prediction
Predict dynamic fare using:
- Distance
- Traffic & weather
- Surge multiplier
- Time features
- Vehicle type & city  

Outputs:
- Estimated fare ₹  
- Feature importance ranking  

---

### 3️⃣ Customer Cancellation Risk
Predict probability that a customer will cancel using:
- Historical cancellation rate
- Booking behaviour
- Ride history  

---

### 4️⃣ Driver Delay Risk
Predict whether driver may delay pickup using:
- Delay history
- Completion rate
- Ride acceptance behaviour  

---

# 📈 Analytics Dashboard

Visual insights generated from processed booking data:

- Cancellations by hour  
- Surge vs Traffic  
- Surge vs Weather  
- Peak surge hours  
- Top pickup locations  
- Top drop locations  

---

# 💼 Business Insights Generated

### Peak Cancellation Windows
Evening rush hours show the highest cancellation rates.

### High-Risk Ride Prediction
Long distance + high surge + peak hour = high cancellation risk.

### Driver Allocation Strategy
Assign high-reliability drivers during peak demand.

### Accurate Fare Estimation
Improves pricing accuracy and reduces revenue leakage.

### Operations Decision Support
Enables:
- Dynamic pricing
- Smart driver assignment
- Early cancellation prediction

---

# 🏗️ End-to-End ML Pipeline
Raw Data → Data Cleaning → Feature Engineering →
Train/Test Split → Model Training → Hyperparameter Tuning →
Model Evaluation → Model Saving → Streamlit Deployment


# ⚙️ Tech Stack

## Programming
- Python

## Data Science & ML
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Visualization & Deployment
- Matplotlib
- Streamlit

## Model Persistence
- Joblib

---

# 📁 Project Structure
rapido-ml-project/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── data_cleaning.ipynb
│ ├── feature_engineering.ipynb
│ ├── model_training.ipynb
│
├── models/
│ ├── ride_outcome_xgb.pkl
│ ├── final_fare_prediction_model.pkl
│ ├── final_customer_cancellation_model.pkl
│ ├── final_driver_delay_model.pkl
│
├── app/
│ └── streamlit_app.py
│
├── requirements.txt
└── README.md


---

# 🚀 How To Run Locally

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/rapido-ml-dashboard.git
cd rapido-ml-dashboard
```

## 2️⃣ Install Dependencies
pip install -r requirements.txt

## 3️⃣ Run Streamlit App
streamlit run app/streamlit_app.py

## 📊 Model Performance (Sample)
### Ride Outcome Model
- Accuracy: ~73–75%
- Balanced F1 across classes
### Fare Prediction Model
- Strong influence from surge & distance
### Customer Cancellation Model
- Reliable high-risk customer detection
### Driver Delay Model
- Identifies drivers likely to delay pickups

💡 Key ML Concepts Demonstrated
- Multi-Class Classification
- Binary Classification
- Regression Modeling
- Feature Engineering
- Pipeline + ColumnTransformer
- Hyperparameter Tuning (GridSearchCV)
- Handling Class Imbalance
- Model Interpretability
- Business-Driven ML

🎯 Business Impact

This system helps ride-hailing companies:
- Reduce cancellations
- Improve pricing strategy
- Optimize driver allocation
- Increase operational efficiency
- Improve customer experience

👩‍💻 Author

Rucha K 

Machine Learning & Data Science Enthusiast
