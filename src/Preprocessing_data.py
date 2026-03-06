import pandas as pd
import numpy as np
import os

# --- PATH CONFIGURATION ---
# Determine the base directory (Fan-STFT) relative to this script's location (src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'accelerometer.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'processed_data.csv')

# --- DATA LOADING ---
try:
    data = pd.read_csv(INPUT_PATH)
    print(f"Successfully loaded dataset from: {INPUT_PATH}")
except FileNotFoundError:
    print(f"Error: File not found at {INPUT_PATH}. Please verify the directory structure.")
    exit()

# Step 1: Feature Normalization (Min-Max Scaling)
def normalize_signal(signal):
    """Normalizes the input signal to a range between 0 and 1."""
    min_val = signal.min()
    max_val = signal.max()
    return (signal - min_val) / (max_val - min_val)

data['x'] = normalize_signal(data['x'])
data['y'] = normalize_signal(data['y'])
data['z'] = normalize_signal(data['z'])

# Step 2: Speed Range Categorization (Low, Medium, High)
def assign_speed_range(pctid):
    """Categorizes speed percentages into discrete operational ranges."""
    if 20 <= pctid <= 45:
        return 'Low'
    elif 50 <= pctid <= 75:
        return 'Medium'
    elif 80 <= pctid <= 100:
        return 'High'
    else:
        return None 

data['speed_range'] = data['pctid'].apply(assign_speed_range)

# Step 3: Class Label Merging (1 to 9)
def assign_merged_class(row):
    """Maps operational conditions and speed ranges to 9 distinct fault classes."""
    mapping = {
        (1, 'Low'): 1, (1, 'Medium'): 2, (1, 'High'): 3,
        (2, 'Low'): 4, (2, 'Medium'): 5, (2, 'High'): 6,
        (3, 'Low'): 7, (3, 'Medium'): 8, (3, 'High'): 9
    }
    return mapping.get((row['wconfid'], row['speed_range']))

data['label'] = data.apply(assign_merged_class, axis=1)
# Exclude samples outside defined operational ranges
data = data.dropna(subset=['label'])

# Step 4: Statistical Feature Extraction (Standard Deviation)
# Optimizing performance using grouped transform
print("Extracting statistical features...")
data['std_x'] = data.groupby(['pctid', 'wconfid'])['x'].transform('std')
data['std_y'] = data.groupby(['pctid', 'wconfid'])['y'].transform('std')
data['std_z'] = data.groupby(['pctid', 'wconfid'])['z'].transform('std')

# Construct the final feature-label DataFrame
output_df = data[['std_x', 'std_y', 'std_z', 'label']].copy()

# Step 5: Data Exporting
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_df.to_csv(OUTPUT_FILE, index=False)
print(f'Exported processed data: {OUTPUT_FILE} ({len(output_df)} samples)')

print("Preprocessing Pipeline Completed Successfully!")
