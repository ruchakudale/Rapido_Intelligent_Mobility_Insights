import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rapido Intelligent Mobility Dashboard", layout="wide")

# ================= LOAD MODELS =================
ride_model = joblib.load("../models/ride_outcome_xgb.pkl")
fare_model = joblib.load("../models/final_fare_prediction_model.pkl")
cancel_model = joblib.load("../models/final_customer_cancellation_model.pkl")
driver_model = joblib.load("../models/final_driver_delay_model.pkl")

# ================= LOAD DATA FOR DASHBOARD =================

df = pd.read_pickle("../data/processed/bookings_ready_data.pkl")

customers_df = pd.read_pickle('../data/processed/customer_ready_data.pkl')
drivers_df = pd.read_pickle('../data/processed/driver_ready_data.pkl')

st.title("🚖 Rapido Intelligent Mobility Insights Dashboard")

# ================= TABS =================
tabs = st.tabs([
    "Ride Outcome Prediction",
    "Fare Prediction",
    "Customer Cancellation Risk",
    "Driver Delay Risk",
    "Analytics Dashboard",
    "Business Insights"
])

# ============================================================
# TAB 1 — RIDE OUTCOME
# ============================================================
with tabs[0]:
    st.subheader("Ride Outcome Prediction")
    col1, col2, col3 = st.columns(3)
    st.write("Predict whether ride will COMPLETE or CANCEL")

    # ----- Feature Importance Extraction -----
    rf_clf = ride_model.named_steps["model"]
    preprocessor_clf = ride_model.named_steps["prep"]

    feature_names_clf = preprocessor_clf.get_feature_names_out()
    importances_clf = rf_clf.feature_importances_

    fi_clf_df = pd.DataFrame({
        "feature": feature_names_clf,
        "importance": importances_clf
    }).sort_values("importance", ascending=False)


    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        hour_of_day = st.slider("Hour of Day", 0, 23, 12,key="hour_of_day_ride")
        ride_distance_km = st.number_input("Ride Distance (km)", 0.0, 100.0, 5.0, key="ride_distance_km_ride")
        estimated_ride_time_min = st.number_input("Estimated Ride Time (min)", 1, 180, 20,key="estimated_ride_time_min")
        actual_ride_time_min = st.number_input("Actual Ride Time (min)", 1, 180, 20,key="actual_ride_time_min")
       

    # Column 2
    with col2:
        traffic_level = st.selectbox("Traffic Level", ["Low","Medium","High"],key="traffic_level_ride")
        weather_condition = st.selectbox("Weather", ["Clear","High Rain","Rain"],key="weather_condition_ride")
        base_fare = st.number_input("Base Fare", 0.0, 500.0, 50.0,key='base_fare_ride')
        surge_multiplier = st.slider("Surge Multiplier", 1.0, 5.0, 1.0,key="surge_multiplier_ride")

    # Column 3
    with col3:
        booking_value = st.number_input("Booking Value", 0.0, 1000.0, 100.0)
        incomplete_ride_reason = st.selectbox(
            "Incomplete Ride Reason",
            ["None", 'Driver Delay', 'App Issue', 'Customer No-show', 'Vehicle Issue']
        )
        customer_total_rides = st.number_input("Customer Total Rides", 0, 5000, 100)
    
    input_dict_clf = {
        "hour_of_day": hour_of_day,
        "ride_distance_km": ride_distance_km,
        "estimated_ride_time_min": estimated_ride_time_min,
        "actual_ride_time_min": actual_ride_time_min,
        "traffic_level": traffic_level,
        "weather_condition": weather_condition,
        "base_fare": base_fare,
        "surge_multiplier": surge_multiplier,
        "booking_value": booking_value,
        "incomplete_ride_reason": incomplete_ride_reason,
        "customer_total_rides": customer_total_rides
    }
    
    input_df_clf = pd.DataFrame([input_dict_clf])
    
    if st.button("Predict Ride Outcome"):
        pred = ride_model.predict(input_df_clf)[0]
        prob = (ride_model.predict_proba(input_df_clf)[0][1])*100

        st.subheader("Prediction Result")

        if pred == 1:
            st.success(f"✅ Ride likely to COMPLETE (Probability: {100-prob:.2f})")
        else:           
            st.error(f"❌ Ride likely to CANCEL (Probability: {prob:.2f})")

        st.markdown("---")
        st.subheader("📊 Feature Importance")
      
        top_features = fi_clf_df.head(7)

        fig, ax = plt.subplots(figsize=(8,3))
        ax.barh(top_features["feature"], top_features["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance")

        st.pyplot(fig)

# ============================================================
# TAB 2 — FARE PREDICTION
# ============================================================
with tabs[1]:
    
    st.subheader("🚕 Ride Fare Prediction Dashboard")

    # -------- Extract Feature Importance from Pipeline -------- #

    # Get trained RF model
    rf_estimator = fare_model.named_steps["model"]

    # Get preprocessor (ColumnTransformer)
    preprocessor = fare_model.named_steps["preprocessor"]

    # Get feature names after OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    # Get feature importances from RandomForest
    importances = rf_estimator.feature_importances_

    # Create dataframe
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    col1, col2, col3 = st.columns(3)

    # -------- Column 1 -------- #
    with col1:
        hour_of_day = st.slider("Hour of Day", 0, 23, 12,key='hour_of_day')
        ride_distance_km = st.number_input("Ride Distance (km)", 0.0, 100.0, 5.0)
        estimated_ride_time_min = st.number_input("Estimated Ride Time (min)", 1, 180, 20)
        base_fare = st.number_input("Base Fare", 0.0, 500.0, 50.0)

    # -------- Column 2 -------- #
    with col2:
        surge_multiplier = st.slider("Surge Multiplier", 1.0, 5.0, 1.0,key="surge_multiplier")
        is_weekend = st.selectbox("Is Weekend", [0, 1])
        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["Auto", "Bike", "Cab"],
            key="vehicle_type"
        )
        traffic_level = st.selectbox(
            "Traffic Level",
            ["Low", "Medium", "High"],
            key="traffic_level"
        )

    # -------- Column 3 -------- #
    with col3:
        weather_condition = st.selectbox(
            "Weather Condition",
            ["Clear", "Heavy Rain", "Rain"],
            key="weather_condition"
        )
        city = st.selectbox(
            "City",
            ["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad"],
            key="city"
        )
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            key="day_of_week"
        )

    # -------- Convert to dataframe -------- #
    input_dict = {
        "hour_of_day": hour_of_day,
        "vehicle_type": vehicle_type,
        "ride_distance_km": ride_distance_km,
        "traffic_level": traffic_level,
        "base_fare": base_fare,
        "surge_multiplier": surge_multiplier,
        "is_weekend": is_weekend,
        "estimated_ride_time_min": estimated_ride_time_min,
        "weather_condition": weather_condition,
        "city": city,
        "day_of_week": day_of_week
    }

    input_df = pd.DataFrame([input_dict])

    # -------- Prediction -------- #
    if st.button("Predict Fare"):
        prediction = fare_model.predict(input_df)[0]

        st.subheader("💰 Predicted Ride Fare")
        st.success(f"Estimated Fare: ₹ {prediction:,.2f}")
   
    st.markdown("---")
    st.subheader("📊 Feature Importance (Random Forest Regressor)")
    top_features = fi_df.head(5)

    fig, ax = plt.subplots(figsize=(9,3))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Top Important Features")

    st.pyplot(fig)
# ============================================================
# TAB 3 — CUSTOMER CANCELLATION
# ============================================================
with tabs[2]:
    st.subheader("Customer Cancellation Risk Prediction")
    st.write("Predict whether customer is likely to cancel ride")


    # ----- Feature importance extraction -----
    cust_rf = cancel_model.named_steps["model"]
    cust_preprocessor = cancel_model.named_steps["preprocessor"]

    cust_feature_names = cust_preprocessor.get_feature_names_out()
    cust_importances = cust_rf.feature_importances_

    cust_fi_df = pd.DataFrame({
        "feature": cust_feature_names,
        "importance": cust_importances
    }).sort_values("importance", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        total_bookings = st.number_input("Total Bookings", 0, 10000, 50, key="total_bookings_cancel")
        completed_rides = st.number_input("Completed Rides", 0, 10000, 40,key="completed_rides_cancel")

    with col2:
        cancelled_rides = st.number_input("Cancelled Rides", 0, 10000, 10,key="cancelled_rides_cancel")
        cancellation_rate = st.slider("Cancellation Rate", 0.0, 1.0, 0.2,key="cancellation_rate_cancel")

    cust_input = {
        "total_bookings": total_bookings,
        "completed_rides": completed_rides,
        "cancelled_rides": cancelled_rides,
        "cancellation_rate": cancellation_rate
    }

    cust_df = pd.DataFrame([cust_input])

    if st.button("Predict Customer Risk"):
        pred = cancel_model.predict(cust_df)[0]
        prob = cancel_model.predict_proba(cust_df)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"⚠️ High Cancellation Risk (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Cancellation Risk (Probability: {1-prob:.2f})")

        st.markdown("---")
        st.subheader("📊 Feature Importance")

        import matplotlib.pyplot as plt

        top_features = cust_fi_df.head(4)

        fig, ax = plt.subplots(figsize=(9,3))
        ax.barh(top_features["feature"], top_features["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title("Customer Cancellation Model Importance")

        st.pyplot(fig)

# ============================================================
# TAB 4 — DRIVER DELAY RISK
# ============================================================
with tabs[3]:
    st.subheader("🚗 Driver Delay Risk Prediction")
    st.write("Predict whether driver is likely to delay pickup")

    # ----- Feature importance extraction -----
    drv_rf = driver_model.named_steps["model"]
    drv_preprocessor = driver_model.named_steps["preprocessor"]

    drv_feature_names = drv_preprocessor.get_feature_names_out()
    drv_importances = drv_rf.feature_importances_

    drv_fi_df = pd.DataFrame({
        "feature": drv_feature_names,
        "importance": drv_importances
    }).sort_values("importance", ascending=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        total_assigned_rides = st.number_input("Total Assigned Rides", 0, 10000, 200)
        incomplete_rides = st.number_input("Incomplete Rides", 0, 10000, 20)
        delay_count = st.number_input("Delay Count", 0, 10000, 15)

    with col2:
        delay_rate = st.slider("Delay Rate", 0.0, 1.0, 0.1)
        avg_pickup_delay_min = st.number_input("Avg Pickup Delay (min)", 0.0, 60.0, 5.0)

    with col3:
        Completed_Rides = st.number_input("Completed Rides", 0, 10000, 180)
        Completion_Rate = st.slider("Completion Rate", 0.0, 1.0, 0.9)

    drv_input = {
        "total_assigned_rides": total_assigned_rides,
        "incomplete_rides": incomplete_rides,
        "delay_count": delay_count,
        "delay_rate": delay_rate,
        "avg_pickup_delay_min": avg_pickup_delay_min,
        "Completed_Rides": Completed_Rides,
        "Completion_Rate": Completion_Rate
    }

    drv_df = pd.DataFrame([drv_input])

    if st.button("Predict Driver Delay Risk"):
        pred = driver_model.predict(drv_df)[0]
        prob = driver_model.predict_proba(drv_df)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"⏰ High Delay Risk (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Delay Risk (Probability: {1-prob:.2f})")

        st.markdown("---")
        st.subheader("📊 Feature Importance")   
        top_features = drv_fi_df.head(7)

        fig, ax = plt.subplots(figsize=(9,3))
        ax.barh(top_features["feature"], top_features["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title("Driver Delay Model Importance")

        st.pyplot(fig)


# ============================================================
# TAB 5 — ANALYTICS DASHBOARD
# ============================================================
with tabs[4]:

    st.subheader("Cancellations by Hour")
    cancel_hour = df[df["booking_status"]=="Cancelled"].groupby("hour_of_day").size()
    st.line_chart(cancel_hour)

    st.subheader("Surge Behavior Pattern")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🚦 Surge vs Traffic Level")

        surge_traffic = df.groupby("traffic_level")["surge_multiplier"].mean()

        fig, ax = plt.subplots()
        ax.bar(surge_traffic.index, surge_traffic.values)
        ax.set_xlabel("Traffic Level")
        ax.set_ylabel("Average Surge")
        ax.set_title("Surge vs Traffic")
        st.pyplot(fig)
        st.write("Insight: How congestion affects pricing.")
        
    with col2:
        st.subheader("🌧️ Surge vs Weather")

        surge_weather = df.groupby("weather_condition")["surge_multiplier"].mean()

        fig, ax = plt.subplots()
        ax.bar(surge_weather.index, surge_weather.values)
        ax.set_xlabel("Weather")
        ax.set_ylabel("Average Surge")
        ax.set_title("Weather Impact on Surge")
        st.pyplot(fig)

        st.write("Insight: Rain & bad weather → higher surge.")
    with col3: 
        st.subheader("📈 Surge by Hour of Day")

        surge_hour = df.groupby("hour_of_day")["surge_multiplier"].mean()

        fig, ax = plt.subplots()
        ax.plot(surge_hour.index, surge_hour.values, marker='o')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Surge Multiplier")
        ax.set_title("Peak Surge Hours")

        st.pyplot(fig)
        st.write("Insight: Identifies peak demand hours.")


    st.subheader("📍 Pickup Location Demand")

    pickup_counts = df["pickup_location"].value_counts().sort_values(ascending= False).head(5)
    fig, ax = plt.subplots(figsize=(9,3))
    ax.barh(pickup_counts.index, pickup_counts.values)
    ax.set_ylabel("Locations")
    ax.set_xlabel("Number of Pickups")
    ax.set_title("Top Pickup Locations")

    st.pyplot(fig)

    st.subheader("🏁 Drop Location Demand")

    drop_counts = df["drop_location"].value_counts().sort_values(ascending= False).head(5)

    fig, ax = plt.subplots(figsize=(9,3))
    ax.barh(drop_counts.index, drop_counts.values)
    ax.set_ylabel("Locations")
    ax.set_xlabel("Number of Drops")
    ax.set_title("Top Drop Locations")

    st.pyplot(fig)
# ============================================================
# TAB 6 — BUSINESS INSIGHTS
# ============================================================
with tabs[5]:
    st.subheader("Business Insights")

    st.markdown("""
    ### Peak Cancellation Windows
    - Evening rush hours show highest cancellations.
    - Surge pricing strongly increases cancellation risk.

    ### High Risk Ride Prediction
    - Long distance + high surge + peak hour = risky ride.

    ### Driver Allocation Strategy
    - Avoid assigning low reliability drivers during peak hours.
    - Prefer high acceptance-rate drivers in surge areas.

    ### Accurate Fare Estimation
    - Fare prediction model reduces underpricing risk.

    ### Operations Decision Making
    - Predict cancellations early
    - Optimize pricing dynamically
    - Improve driver assignment
    """)
