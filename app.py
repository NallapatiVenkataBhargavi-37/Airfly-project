# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import tensorflow as tf

# ======================
# Load Data & Models
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_flights.csv")
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
        df["Month"] = df["FL_DATE"].dt.month
        df["Day"] = df["FL_DATE"].dt.day
        df["DayOfWeek"] = df["FL_DATE"].dt.dayofweek + 1
    return df

df = load_data()

# Load scaler and features
scaler = joblib.load("scaler.joblib")
meta = joblib.load("feature_meta.joblib")

# Load models safely
reg_model = tf.keras.models.load_model("delay_regressor.h5", compile=False)
clf_model = tf.keras.models.load_model("cancel_classifier.h5", compile=False)

# ======================
# Sidebar Navigation
# ======================
st.sidebar.title("✈️ Flight Delay Dashboard")
page = st.sidebar.radio("Go to", ["Visualizations", "Model Predictions"])

# ======================
# Visualizations (30+ Graphs)
# ======================
if page == "Visualizations":
    st.title("📊 Flight Data Insights (30+ Graphs)")

    # ---- Graphs 1-2: Avg Delay by Origin & Destination ----
    st.subheader("1-2. Average Departure Delay by Airport")
    col1, col2 = st.columns(2)
    orig_avg = df.groupby("ORIGIN")["DEP_DELAY"].mean().reset_index()
    dest_avg = df.groupby("DEST")["DEP_DELAY"].mean().reset_index()
    col1.plotly_chart(px.bar(orig_avg, x="ORIGIN", y="DEP_DELAY", color="DEP_DELAY", title="Avg Delay by Origin"))
    col2.plotly_chart(px.bar(dest_avg, x="DEST", y="DEP_DELAY", color="DEP_DELAY", title="Avg Delay by Destination"))

    # ---- Graph 3: Correlation Heatmap ----
    st.subheader("3. Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---- Graphs 4-5: Monthly & Day of Week Delay ----
    st.subheader("4-5. Delay Trends Over Time")
    col1, col2 = st.columns(2)
    monthly = df.groupby("Month")["DEP_DELAY"].mean().reset_index()
    dow = df.groupby("DayOfWeek")["DEP_DELAY"].mean().reset_index()
    col1.plotly_chart(px.line(monthly, x="Month", y="DEP_DELAY", markers=True, title="Avg Delay by Month"))
    col2.plotly_chart(px.bar(dow, x="DayOfWeek", y="DEP_DELAY", title="Avg Delay by Day of Week"))

    # ---- Graph 6: Hourly Delay ----
    if "CRS_DEP_TIME_HOUR" in df.columns:
        st.subheader("6. Hourly Departure Delay")
        hourly = df.groupby("CRS_DEP_TIME_HOUR")["DEP_DELAY"].mean().reset_index()
        st.plotly_chart(px.line(hourly, x="CRS_DEP_TIME_HOUR", y="DEP_DELAY", markers=True,
                                title="Avg Delay by Hour of Day"))

    # ---- Graphs 7-8: Airline Delay & Cancellation ----
    st.subheader("7-8. Airline Performance")
    col1, col2 = st.columns(2)
    airline_delay = df.groupby("AIRLINE")["DEP_DELAY"].mean().reset_index()
    cancel_airline = df.groupby("AIRLINE")["CANCELLED"].mean().reset_index()
    col1.plotly_chart(px.bar(airline_delay, x="AIRLINE", y="DEP_DELAY", color="DEP_DELAY", title="Avg Delay by Airline"))
    col2.plotly_chart(px.pie(cancel_airline, names="AIRLINE", values="CANCELLED", title="Cancellation Rate by Airline"))

    # ---- Graphs 9-10: Distance vs Delay & Top Routes ----
    st.subheader("9-10. Distance & Popular Routes")
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.scatter(df, x="DISTANCE", y="DEP_DELAY", color="AIRLINE", opacity=0.4, title="Distance vs Delay"))
    if "Route" in df.columns:
        top_routes = df["Route"].value_counts().head(20).reset_index()
        top_routes.columns = ["Route", "Count"]
        col2.plotly_chart(px.bar(top_routes, x="Route", y="Count", color="Count", title="Top 20 Routes"))

    # ---- Graphs 11-12: Weekend vs Weekday Delay & Boxplot ----
    st.subheader("11-12. Weekend vs Weekday & Airline Delay Distribution")
    df["IsWeekend"] = df["DayOfWeek"].isin([6,7]).astype(int)
    col1, col2 = st.columns(2)
    weekend_delay = df.groupby("IsWeekend")["DEP_DELAY"].mean().reset_index()
    col1.plotly_chart(px.bar(weekend_delay, x="IsWeekend", y="DEP_DELAY", color="IsWeekend", title="Weekend vs Weekday Delay"))
    col2.plotly_chart(px.box(df, x="AIRLINE", y="DEP_DELAY", title="Boxplot of Delay per Airline"))

    # ---- Graphs 13-14: Violin & Histogram ----
    st.subheader("13-14. Delay Distributions")
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.violin(df, x="AIRLINE", y="DEP_DELAY", color="AIRLINE", box=True, points="all", title="Violin Plot: Delay per Airline"))
    col2.plotly_chart(px.histogram(df, x="DEP_DELAY", nbins=50, title="Histogram of Delay"))

    # ---- Graph 15-16: KDE & Scatter Matrix ----
    st.subheader("15-16. Advanced Distributions")
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.density_heatmap(df, x="DISTANCE", y="DEP_DELAY", nbinsx=50, nbinsy=50, title="KDE: Distance vs Delay"))
    col2.plotly_chart(px.scatter_matrix(df, dimensions=["DISTANCE","DEP_DELAY","CRS_DEP_TIME_HOUR","CRS_ARR_TIME_HOUR"], color="AIRLINE", title="Scatter Matrix"))

    # ---- Graphs 17-18: Top Airports ----
    st.subheader("17-18. Top Departure & Arrival Airports")
    col1, col2 = st.columns(2)
    top_orig = df["ORIGIN"].value_counts().head(10).reset_index()
    top_orig.columns = ["Airport","Count"]
    top_dest = df["DEST"].value_counts().head(10).reset_index()
    top_dest.columns = ["Airport","Count"]
    col1.plotly_chart(px.bar(top_orig, x="Airport", y="Count", color="Count", title="Top 10 Departure Airports"))
    col2.plotly_chart(px.bar(top_dest, x="Airport", y="Count", color="Count", title="Top 10 Arrival Airports"))

    # ---- Graphs 19-20: Flight Number & Route Delay ----
    st.subheader("19-20. Flights & Route Delays")
    col1, col2 = st.columns(2)
    top_flights = df.groupby("FL_NUMBER")["DEP_DELAY"].mean().sort_values(ascending=False).head(10).reset_index()
    route_delay = df.groupby("Route")["DEP_DELAY"].mean().sort_values(ascending=False).head(10).reset_index()
    col1.plotly_chart(px.bar(top_flights, x="FL_NUMBER", y="DEP_DELAY", title="Top 10 Flights by Delay"))
    col2.plotly_chart(px.bar(route_delay, x="Route", y="DEP_DELAY", title="Top 10 Routes by Delay"))

    # ---- Graphs 21-22: Delay Trends by Airline ----
    st.subheader("21-22. Delay Trends")
    col1, col2 = st.columns(2)
    month_airline = df.groupby(["Month","AIRLINE"])["DEP_DELAY"].mean().reset_index()
    col1.plotly_chart(px.line(month_airline, x="Month", y="DEP_DELAY", color="AIRLINE", markers=True, title="Monthly Delay per Airline"))
    dow_airline = df.groupby(["DayOfWeek","AIRLINE"])["DEP_DELAY"].mean().reset_index()
    col2.plotly_chart(px.line(dow_airline, x="DayOfWeek", y="DEP_DELAY", color="AIRLINE", markers=True, title="Day-of-Week Delay per Airline"))

    # ---- Graphs 23-24: Delay by Arrival Hour & Distance ----
    st.subheader("23-24. Delay Patterns")
    col1, col2 = st.columns(2)
    arr_hour_delay = df.groupby("CRS_ARR_TIME_HOUR")["DEP_DELAY"].mean().reset_index()
    col1.plotly_chart(px.line(arr_hour_delay, x="CRS_ARR_TIME_HOUR", y="DEP_DELAY", markers=True, title="Delay vs Arrival Hour"))
    col2.plotly_chart(px.scatter(df, x="DISTANCE", y="DEP_DELAY", color="AIRLINE", title="Distance vs Delay per Airline"))

    # ---- Graphs 25-26: Cumulative Delay & Monthly Cancellation ----
    st.subheader("25-26. Cumulative Metrics")
    col1, col2 = st.columns(2)
    cum_delay = df.groupby("AIRLINE")["DEP_DELAY"].sum().reset_index()
    col1.plotly_chart(px.bar(cum_delay, x="AIRLINE", y="DEP_DELAY", color="DEP_DELAY", title="Cumulative Delay per Airline"))
    cancel_month = df.groupby("Month")["CANCELLED"].mean().reset_index()
    col2.plotly_chart(px.bar(cancel_month, x="Month", y="CANCELLED", color="CANCELLED", title="Monthly Cancellation Rate"))

    # ---- Graphs 27-28: Top Routes Pie Charts ----
    st.subheader("27-28. Top Routes by Flights & Cancelled")
    if "Route" in df.columns:
        route_count = df["Route"].value_counts().head(10).reset_index()
        route_count.columns = ["Route","Count"]
        route_cancel = df[df["CANCELLED"]==1]["Route"].value_counts().head(10).reset_index()
        route_cancel.columns = ["Route","Count"]
        col1, col2 = st.columns(2)
        col1.plotly_chart(px.pie(route_count, names="Route", values="Count", title="Top 10 Routes by Flights"))
        col2.plotly_chart(px.pie(route_cancel, names="Route", values="Count", title="Top 10 Cancelled Routes"))

    # ---- Graphs 29-30: Heatmaps & Airline Delay Sorted ----
    st.subheader("29-30. Heatmaps & Airline Comparison")
    col1, col2 = st.columns(2)
    heatmap_data = df.pivot_table(index="ORIGIN", columns="DEST", values="DEP_DELAY", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(heatmap_data, cmap="viridis", ax=ax)
    col1.pyplot(fig)
    airline_delay_sorted = airline_delay.sort_values("DEP_DELAY", ascending=False)
    col2.plotly_chart(px.bar(airline_delay_sorted, x="AIRLINE", y="DEP_DELAY", title="Airline Delay Comparison (Sorted)"))

    # ---- Interactive US Map ----
    st.subheader("🌎 US Airport Delays Map")
    if "ORIGIN" in df.columns and "DEP_DELAY" in df.columns:
        airport_delays = df.groupby("ORIGIN")["DEP_DELAY"].mean().reset_index()
        try:
            airports_coords = pd.read_csv("us_airports.csv")  # IATA,LAT,LON
            airport_delays = airport_delays.merge(airports_coords, left_on="ORIGIN", right_on="IATA")
            fig_map = px.scatter_mapbox(
                airport_delays, lat="LAT", lon="LON", size="DEP_DELAY", color="DEP_DELAY",
                hover_name="ORIGIN", color_continuous_scale=px.colors.cyclical.IceFire,
                size_max=20, zoom=3, mapbox_style="carto-positron",
                title="Average Departure Delay per Airport"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except FileNotFoundError:
            st.warning("⚠️ us_airports.csv not found. Map cannot be displayed.")

# ======================
# Model Predictions
# ======================
elif page == "Model Predictions":
    st.title("🤖 Flight Delay & Cancellation Prediction")

    with st.form("prediction_form"):
        airline = st.text_input("Airline Code (e.g., AA)")
        origin = st.text_input("Origin Airport (e.g., JFK)")
        dest = st.text_input("Destination Airport (e.g., LAX)")
        crs_dep_time = st.number_input("Scheduled Departure Hour (0-23)", 0, 23, 10)
        crs_arr_time = st.number_input("Scheduled Arrival Hour (0-23)", 0, 23, 12)
        flight_num = st.number_input("Flight Number", 1, 9999, 100)
        month = st.number_input("Month (1-12)", 1, 12, 1)
        dow = st.number_input("Day of Week (1=Mon, 7=Sun)", 1, 7, 1)
        distance = st.number_input("Flight Distance (miles)", 1, 5000, 500)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            "AIRLINE": [airline],
            "ORIGIN": [origin],
            "DEST": [dest],
            "CRS_DEP_TIME_HOUR": [crs_dep_time],
            "CRS_ARR_TIME_HOUR": [crs_arr_time],
            "FL_NUMBER": [flight_num],
            "Month": [month],
            "DayOfWeek": [dow],
            "DISTANCE": [distance],
            "IsWeekend": [1 if dow in [6, 7] else 0],
            "Route": [f"{origin}_{dest}"]
        }
        input_df = pd.DataFrame(input_dict)

        # Prepare features
        X_cat = pd.get_dummies(input_df[meta["cat_feats"]].astype(str), drop_first=True)
        X_num = input_df[meta["num_feats"]].astype(float)
        X_num[meta["num_feats"]] = scaler.transform(X_num[meta["num_feats"]])
        X = pd.concat([X_num, X_cat], axis=1).reindex(columns=meta["features"], fill_value=0)

        # Predictions
        delay_pred = reg_model.predict(X)[0][0]
        cancel_proba = clf_model.predict(X)[0][0]

        st.success(f"⏱ Predicted Delay: **{delay_pred:.2f} minutes**")
        st.warning(f"❌ Cancellation Probability: **{cancel_proba*100:.2f}%**")
