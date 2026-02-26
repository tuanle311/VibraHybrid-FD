import pandas as pd
import numpy as np
import os

# Read data from file
data = pd.read_csv("E:/Tuan/RMIT/Đề tài/Fault/Dataset/accelerometer/accelerometer.csv")  # Replace with your actual file path

# Step 1: Normalize x, y, z signals
def normalize_signal(signal):
    min_val = signal.min()
    max_val = signal.max()
    return (signal - min_val) / (max_val - min_val)

data['x'] = normalize_signal(data['x'])
data['y'] = normalize_signal(data['y'])
data['z'] = normalize_signal(data['z'])

# Step 2: Cluster speed (pctid) and create 3 speed ranges
def assign_speed_range(pctid):
    if 20 <= pctid <= 45:
        return 'Low'
    elif 50 <= pctid <= 75:
        return 'Medium'
    elif 80 <= pctid <= 100:
        return 'High'
    else:
        return None  # Values outside these ranges will not be labeled

data['speed_range'] = data['pctid'].apply(assign_speed_range)

# Create merged class labels (1 to 9)
def assign_merged_class(row):
    if row['speed_range'] is None:
        return None
    if row['wconfid'] == 1 and row['speed_range'] == 'Low':
        return 1
    elif row['wconfid'] == 1 and row['speed_range'] == 'Medium':
        return 2
    elif row['wconfid'] == 1 and row['speed_range'] == 'High':
        return 3
    elif row['wconfid'] == 2 and row['speed_range'] == 'Low':
        return 4
    elif row['wconfid'] == 2 and row['speed_range'] == 'Medium':
        return 5
    elif row['wconfid'] == 2 and row['speed_range'] == 'High':
        return 6
    elif row['wconfid'] == 3 and row['speed_range'] == 'Low':
        return 7
    elif row['wconfid'] == 3 and row['speed_range'] == 'Medium':
        return 8
    elif row['wconfid'] == 3 and row['speed_range'] == 'High':
        return 9
    return None

data['label'] = data.apply(assign_merged_class, axis=1)

# Remove samples without labels (those outside Low, Medium, High speed ranges)
data = data.dropna(subset=['label'])

# Step 3: Calculate standard deviation for each pctid and wconfid group
# Create a dictionary to store the standard deviation of each group
std_dict = {}

grouped = data.groupby(['pctid', 'wconfid'])
for (pctid, wconfid), group in grouped:
    std_x = group['x'].std()
    std_y = group['y'].std()
    std_z = group['z'].std()
    std_dict[(pctid, wconfid)] = (std_x, std_y, std_z)

# Step 4: Assign standard deviation to each sample and create a new DataFrame
std_x_list = []
std_y_list = []
std_z_list = []
label_list = []

# Iterate through each sample in the original data
for idx, row in data.iterrows():
    pctid = row['pctid']
    wconfid = row['wconfid']
    label = row['label']
    
    # Retrieve the standard deviation of the group the sample belongs to
    std_x, std_y, std_z = std_dict[(pctid, wconfid)]
    
    std_x_list.append(std_x)
    std_y_list.append(std_y)
    std_z_list.append(std_z)
    label_list.append(label)

# Create DataFrame for the processed data
output_df = pd.DataFrame({
    'std_x': std_x_list,
    'std_y': std_y_list,
    'std_z': std_z_list,
    'label': label_list
})

# Step 5: Export to a single CSV file
output_dir = 'output_csvs'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'processed_data.csv')
output_df.to_csv(output_file, index=False)
print(f'Created file: {output_file} with {len(output_df)} samples')

print("Completed!")