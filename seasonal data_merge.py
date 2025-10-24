import os
from glob import glob
import pandas as pd

# Path where your CSV files are stored
path = r"C:\Users\andre\OneDrive\MSDS Masters\Fall 2025\MA641\Seasonal Data"
os.chdir(path)

# List all files
files = [
    "AEP_hourly.csv", "COMED_hourly.csv", "DAYTON_hourly.csv",
    "DEOK_hourly.csv", "DOM_hourly.csv", "DUQ_hourly.csv",
    "EKPC_hourly.csv", "FE_hourly.csv", "NI_hourly.csv",
    "PJME_hourly.csv", "PJMW_hourly.csv", "pjm_hourly_est.csv",
    "PJM_Load_hourly.csv"
]

# Empty list to hold dataframes
dfs = []

for f in files:
    # Read CSV file
    df = pd.read_csv(f, parse_dates=['Datetime'])
    
    # Derive variable name from file (remove "_hourly.csv" and make uppercase)
    var_name = f.replace('_hourly.csv', '').replace('.csv', '').upper()
    
    # Identify which column has numeric data (sometimes 'MW' or 'PJME_MW')
    mw_col = [col for col in df.columns if col.lower() != 'datetime'][0]
    
    # Keep only datetime + load column
    df = df[['Datetime', mw_col]]
    
    # Rename numeric column to utility name
    df = df.rename(columns={mw_col: var_name})
    
    # Add to list
    dfs.append(df)

# Merge all dataframes on 'Datetime'
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='Datetime', how='outer')

# Sort by datetime
merged_df = merged_df.sort_values('Datetime').reset_index(drop=True)

# Save combined dataset
merged_df.to_csv("combined_hourly_energy.csv", index=False)

print("✅ Combined dataset created successfully!")
print("Shape:", merged_df.shape)
print("Columns:", merged_df.columns.tolist())
