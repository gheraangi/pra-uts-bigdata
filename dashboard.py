import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px # type: ignore
import os

st.set_page_config(page_title="Smart Traffic", layout="wide")
st.title("🚦 Smart Traffic Dashboard")

# =========================
# CEK DATA
# =========================
if not os.path.exists("output"):
    st.error("⚠️ Data belum dibuat! Jalankan main_uts.py dulu")
    st.stop()

# =========================
# INIT SPARK (CUMA SEKALI)
# =========================
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("Dashboard") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

spark = init_spark()

# =========================
# LOAD DATA (CACHE BIAR CEPAT)
# =========================
@st.cache_data
def load_data():
    traffic = spark.read.parquet("output/traffic").toPandas()
    traffic_time = spark.read.parquet("output/traffic_time").toPandas()
    ml_data = spark.read.parquet("output/ml_data").toPandas()
    return traffic, traffic_time, ml_data

traffic, traffic_time, ml_data = load_data()

# =========================
# SIDEBAR
# =========================
location = st.sidebar.selectbox("Pilih Area", ["AreaA", "AreaB", "AreaC"])

# =========================
# KPI
# =========================
total = traffic["sum(vehicle_count)"].sum()
st.metric("Total Kendaraan", int(total))

# =========================
# GRAFIK
# =========================
fig = px.line(
    traffic_time,
    x="timestamp",
    y="sum(vehicle_count)",
    title="Tren Kendaraan"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# MODEL (CACHE BIAR GA RETRAIN)
# =========================
@st.cache_resource
def train_model(data):
    model = LinearRegression()
    X = data[["hour"]]
    y = data["vehicle_count"]
    model.fit(X, y)
    return model

model = train_model(ml_data)

# =========================
# PREDIKSI
# =========================
jam = st.slider("Pilih Jam Prediksi", 0, 23)
pred = model.predict([[jam]])

st.metric("Prediksi Kendaraan", int(pred[0]))