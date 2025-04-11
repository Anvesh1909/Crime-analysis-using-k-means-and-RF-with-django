import pandas as pd

import os

# Ensure the directory exists
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# File paths
file_paths = [
    "01_District_wise_crimes_committed_IPC_2001_2012.csv",
]

# Standardized column mapping
column_mapping = {
    "STATE/UT": "States/UTs",
    "DISTRICT": "District",
    "YEAR": "Year",
    "MURDER": "Murder",
    "RAPE": "RAPE",
    "THEFT": "THEFT",
    "States/UTs": "States/UTs",
    "District": "District",
    "Year": "Year",
    "Murder": "Murder",
    "Rape": "RAPE",
    "Theft": "THEFT"
}

# Columns to extract
columns_needed = ["States/UTs", "District", "Year", "Murder", "RAPE", "THEFT",'DOWRY DEATHS']

# Load, rename, and process datasets
dfs = []
for file_path in file_paths:
    df = pd.read_csv(file_path, encoding="utf-8")  # Handle special characters
    df.columns = df.columns.str.strip()  # Remove unwanted spaces

    # Rename columns for consistency
    df.rename(columns=column_mapping, inplace=True)
    
    # Select required columns
    selected_columns = [col for col in columns_needed if col in df.columns]
    df = df[selected_columns]
    
    # Convert numerical columns
    for col in ["Murder", "RAPE", "THEFT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    
    # Append cleaned dataframe
    dfs.append(df)

# Combine all datasets
final_df = pd.concat(dfs, ignore_index=True)

# Drop rows with missing essential values
final_df.dropna(subset=["States/UTs", "District", "Year"], inplace=True)

# Convert Year to integer
final_df["Year"] = final_df["Year"].astype(int)

# Display a sample
print(final_df.head())


output_file = os.path.join(output_dir, "cleaned_crime_data.csv")
final_df.to_csv(output_file, index=False)

print(f"File saved successfully at: {output_file}")