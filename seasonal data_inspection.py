import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Load and parse datetime
df = pd.read_csv("Seasonal Data Section Final Project/combined_hourly_energy.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

print(df.info())
print(df.head())

dupes = df[df["Datetime"].duplicated()]
print(f"Duplicated timestamps: {len(dupes)}")

start, end = df["Datetime"].min(), df["Datetime"].max()
print(f"Data range: {start} → {end}")
print(f"Number of records: {len(df)}")

# Filter to common window
mask = (df["Datetime"] >= "2005-05-01") & (df["Datetime"] <= "2018-08-03")
df = df.loc[mask].set_index("Datetime").sort_index()

# ✅ Remove duplicates in index before reindexing
dup_count = df.index.duplicated().sum()
print(f"Duplicated index timestamps before reindexing: {dup_count}")
if dup_count > 0:
    df = df[~df.index.duplicated(keep='first')]

# Reindex to a perfect hourly grid
idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df.reindex(idx)

# Check for missing hours
missing_hours = df.index[df.isna().all(axis=1)]
print(f"Missing hourly timestamps: {len(missing_hours)}")

# Summary of missing data per column
na_summary = df.isna().sum().sort_values(ascending=False)
na_pct = (na_summary / len(df) * 100).round(2)
missing_table = pd.DataFrame({"Missing_Count": na_summary, "Missing_%": na_pct})
print(missing_table)

# Create a copy for cleaning
df_clean = df.copy()

# Identify nearly complete columns (< 1% missing)
complete_cols = [c for c in df_clean.columns if df_clean[c].isna().mean() < 0.01]

# Interpolate and forward-fill small gaps
df_clean[complete_cols] = df_clean[complete_cols].interpolate(limit=6)
df_clean[complete_cols] = df_clean[complete_cols].ffill(limit=3)

# Drop highly incomplete columns
drop_cols = ["PJM_LOAD", "EKPC", "NI", "DEOK", "FE", "COMED"]
df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

# Reset index and save cleaned dataset
df_clean = df_clean.reset_index().rename(columns={"index": "Datetime"})
#df_clean.to_csv("hourly_energy_cleandata.csv", index=False)

print("✅ Cleaned dataset saved successfully.")
print("Remaining columns:", df_clean.columns.tolist())

######## DATA Section Deliverables ########

df = pd.read_csv("Seasonal Data Section Final Project/hourly_energy_cleandata.csv", parse_dates=["Datetime"]).set_index("Datetime")
df.index = pd.DatetimeIndex(df.index)

desc = df.describe().T[["mean","std","min","max"]].round(2)
print(desc)

print(df.isna().sum())
df.index = pd.to_datetime(df.index)

# 1️⃣ Long-term trend
plt.figure(figsize=(12,4))
plt.plot(df["PJM_HOURLY_EST"], color="teal")
plt.title("PJM Total Estimated Hourly Load (2005–2018)")
plt.ylabel("Energy Consumption (MW)")
plt.xlabel("Time")
plt.show()

# 2️⃣ One-year zoom to show seasonality
df["PJM_HOURLY_EST"]["2016"].plot(figsize=(12,4), color="darkcyan")
plt.title("PJM Total Estimated Hourly Load – Year 2016")
plt.xlabel("Time (Hourly across the year)")
plt.ylabel("Energy Consumption (MW)")
plt.show()

# 3️⃣ Weekly pattern (mean by day of week)
dow = df["PJM_HOURLY_EST"].groupby(df.index.dayofweek).mean()
dow = dow.rename(index={0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
dow = dow.rename_axis("Day")
dow.plot(kind="bar", color="cadetblue", figsize=(7,4))
plt.title("Average PJM Total Load by Day of Week")
plt.xlabel("Day of Week (Mon–Sun)")
plt.ylabel("Energy Consumption (MW)")
plt.show()

# 4️⃣ Daily pattern (mean by hour)
hod = df["PJM_HOURLY_EST"].groupby(df.index.hour).mean()
hod.plot(color="darkslategray", figsize=(10,4))
plt.title("Average PJM Total Load by Hour of Day")
plt.xlabel("Hour of the Day (0–23)")
plt.ylabel("Energy Consumption (MW)")
plt.show()

# 5️⃣ Outlier detection using Z-score
z = (df["PJM_HOURLY_EST"] - df["PJM_HOURLY_EST"].mean())/df["PJM_HOURLY_EST"].std()
outlier_pct = (np.abs(z) > 4).mean()*100
print(f"Outliers (>4σ): {outlier_pct:.3f}%")

# 6️⃣ Stationarity test and seasonal decomposition
target_col = "PJM_HOURLY_EST"
y = df[target_col].dropna()

# --- Augmented Dickey–Fuller Test (level) ---
adf_stat, adf_p, *_ = adfuller(y)
print("\nAugmented Dickey–Fuller Test (level):")
print(f"ADF Statistic: {adf_stat:.4f}")
print(f"p-value:       {adf_p:.4f}")
print("✅ Stationary (reject H0)."
      if adf_p < 0.05 else
      "❌ Non-stationary (fail to reject H0) — differencing will be needed for modeling.")

# --- Seasonal Decomposition: weekly (s=24*7) ---
decomp_w = seasonal_decompose(y, model="additive", period=24*7)
decomp_w.plot()
plt.suptitle(f"Seasonal Decomposition of {target_col} (Weekly Seasonality, s=168)", y=1.02)
plt.show()

# --- Daily seasonality (s=24) ---
decomp_d = seasonal_decompose(y, model="additive", period=24)
decomp_d.plot()
plt.suptitle(f"Seasonal Decomposition of {target_col} (Daily Seasonality, s=24)", y=1.02)
plt.show()
