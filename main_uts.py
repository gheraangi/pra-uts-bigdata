from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour
import pandas as pd
import os
import shutil
import random

# 1. Init Spark
spark = SparkSession.builder.appName("Traffic Big Data").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Hapus folder output lama
if os.path.exists("output"):
    shutil.rmtree("output")

# 3. Buat data simulasi
sensor_data = []

locations = ["AreaA", "AreaB", "AreaC"]

for i in range(100):
    for loc in locations:
        sensor_data.append((
            f"2024-01-01 00:{i%60:02d}:00",
            loc,
            random.randint(10, 100)
        ))

# 4. Convert ke Spark DataFrame
df = spark.createDataFrame(sensor_data, ["timestamp", "location", "vehicle_count"])

# 5. Olah data
df = df.withColumn("hour", hour(col("timestamp")))

# Total kendaraan per lokasi
traffic = df.groupBy("location").sum("vehicle_count")

# Tren waktu
traffic_time = df.groupBy("timestamp").sum("vehicle_count")

# Data ML
ml_data = df.select("hour", "vehicle_count")

# 6. Simpan ke Parquet
traffic.write.parquet("output/traffic")
traffic_time.write.parquet("output/traffic_time")
ml_data.write.parquet("output/ml_data")

print("✅ SEMUA DATA BERHASIL DISIMPAN")

# 7. Stop Spark
spark.stop()