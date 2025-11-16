# U have to run Streamlit twice 1st for model training terminate and again run 2nd time for Inference and insert the value and predict the price

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# ----------------------------------------------------
# Build Pipeline Function
# ----------------------------------------------------
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    
    return full_pipeline

# ----------------------------------------------------
# Train Model if not exists
# ----------------------------------------------------
if not os.path.exists(MODEL_FILE):
    st.write("🔄 Training model for the first time... (this happens only once)")

    housing = pd.read_csv("housing.csv")

    # Create Stratified Test Set
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3., 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_idx].drop("income_cat", axis=1)

    housing = train_set.copy()

    # Features + Labels
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # Numerical + Categorical
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save files
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

# ----------------------------------------------------
# Load Model + Pipeline
# ----------------------------------------------------
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------
st.set_page_config(page_title="California House Price Prediction", layout="centered")

# Blue background
st.markdown("""
    <style>
    .main {
        background-color: #E3F2FD;
        padding: 25px;
        border-radius: 12px;
    }
    h1 { color:#1565C0; text-align:center; }
    .stButton>button {
        background-color:#1976D2;
        color:white;
        font-size:18px;
        border-radius:10px;
        padding:10px 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏡 California House Price Prediction</h1>", unsafe_allow_html=True)
st.write("Enter the house details below to predict the price:")

# ----------------------------------------------------
# INPUT FIELDS (same as features in CSV)
# ----------------------------------------------------
longitude = st.number_input("📍 Longitude", value=-122.23)
latitude = st.number_input("📍 Latitude", value=37.88)
housing_median_age = st.number_input("🏠 Median Age", value=30)
total_rooms = st.number_input("🚪 Total Rooms", value=2000)
total_bedrooms = st.number_input("🛏 Total Bedrooms", value=300)
population = st.number_input("👨‍👩‍👧 Population", value=800)
households = st.number_input("🏘 Households", value=300)
median_income = st.number_input("💰 Median Income", value=4.5)

ocean_proximity = st.selectbox(
    "🌊 Ocean Proximity",
    ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

# ----------------------------------------------------
# PREDICT BUTTON
# ----------------------------------------------------
if st.button("🔮 Predict Price"):
    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    transformed = pipeline.transform(input_data)
    prediction = model.predict(transformed)[0]

    st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")
