import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# --- UTILITY FUNCTIONS ---

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} minutes {remaining_seconds:.2f} seconds"

def create_output_folder(folder_name="Model_Evaluation_Baseline_Output"):
    """Create a directory to store the Confusion Matrices."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created output folder: '{folder_name}'")
    return folder_name

def get_model_metrics(y_test, y_pred, model_name, training_time):
    """Calculate and return performance metrics as float values (multiplied by 100)."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return pd.Series({
        'Model': model_name,
        'Accuracy (%)': accuracy * 100,
        'Precision (%)': precision * 100,
        'Recall (%)': recall * 100,
        'F1-Score (%)': f1 * 100,
        'Training Time (s)': training_time
    })

def print_detailed_evaluation(model_name, y_test, y_pred, output_folder):
    """Print detailed report, plot Confusion Matrix, and SAVE IMAGES."""
    
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    print(f"\n--- Model: {model_name} ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    
    # Automatically adjust target_names based on the actual number of classes to avoid mapping errors
    num_classes = len(np.unique(y_test))
    target_names = [f'Class {i+1}' for i in range(num_classes)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # SAVE IMAGE TO FOLDER
    plt.savefig(os.path.join(output_folder, f"{safe_name}_CM.png"))
    plt.close() 
    print(f"Confusion Matrix saved to {output_folder}/{safe_name}_CM.png")


# --- MAIN EXECUTION BLOCK ---

OUTPUT_FOLDER_PATH = create_output_folder()
total_start_time = time.time()
final_accuracies = {}
metrics_list = []

# 1. Load Data (Raw Baseline - 3 Features)
print("\n=== 1. Loading and Preparing Data (3 Raw Features) ===")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Dataset', 'processed_data.csv')

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}. Please check the path.")
    exit()

# Get 3 raw statistical features (std_x, std_y, std_z)
X_raw = data[['std_x', 'std_y', 'std_z']].values
y = data['label'].values

# Adjust labels to start from 0 (from 1-9 to 0-8)
y = y - 1
print(f"Total features used: {X_raw.shape[1]}")

# 2. Split Data (Train/Val/Test - 64/16/20)
print("=== 2. Splitting Data (Train/Val/Test) ===")
split_start_time = time.time()
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
split_time = time.time() - split_start_time
print(f"Train/Validation/Test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
print(f"Data Splitting Time: {format_time(split_time)}")

# 3. Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# --- 4. TRAIN AND EVALUATE 6 MODELS ---

print("\n=== 3. Training All 6 Raw Baseline Models ===")

# Model List and Parameters (WKNN and Bagged Tree removed)
models_to_run = [
    # 1. MLP
    ("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=40, random_state=42, verbose=False), True),
    # 2. Random Forest
    ("Random Forest", RandomForestClassifier(n_estimators=1, random_state=42, oob_score=True, n_jobs=-1), False),
    # 3. CatBoost
    ("CatBoost", CatBoostClassifier(iterations=10, depth=5, learning_rate=0.1, random_seed=42, verbose=0), False),
    # 4. LightGBM
    ("LightGBM", LGBMClassifier(n_estimators=1,max_depth=5, learning_rate=0.1, random_state=42, verbose=-1, early_stopping_rounds=10), False),
    # 5. Logistic Regression
    ("Logistic Regression", LogisticRegression(solver='saga', max_iter=500, random_state=42, n_jobs=-1), True),
    # 6. Quadratic SVM
    ("Quadratic Support Vector Machine", SVC(kernel='poly', degree=2, max_iter=500, random_state=42, verbose=False), True),
]

for name, model, needs_scaling in models_to_run:
    
    # Choose between scaled or raw data
    X_train_data = X_train_scaled if needs_scaling else X_train
    X_val_data = X_val_scaled if needs_scaling else X_val
    X_test_data = X_test_scaled if needs_scaling else X_test
    
    train_start_time = time.time()
    
    try:
        # Training
        if name == "LightGBM":
            # REMARK: early_stopping_rounds is now handled in the constructor
            model.fit(X_train_data, y_train, eval_set=[(X_val_data, y_val)])
        elif name == "CatBoost":
            # CatBoost also requires eval_set to trigger optimization mechanisms if applicable
            model.fit(X_train_data, y_train, eval_set=[(X_val_data, y_val)])
        else:
            model.fit(X_train_data, y_train)
            
        training_time = time.time() - train_start_time
        y_pred = model.predict(X_test_data)
        
        # 1. Collect Metrics
        metrics_series = get_model_metrics(y_test, y_pred, name, training_time)
        metrics_list.append(metrics_series)
        
        # 2. Print and Save report
        print_detailed_evaluation(name, y_test, y_pred, OUTPUT_FOLDER_PATH)
        
        # 3. Overfitting Check
        train_accuracy = model.score(X_train_data, y_train)
        if name == "Random Forest":
            val_accuracy = model.oob_score_
        elif needs_scaling and name not in ["LightGBM", "CatBoost"]:
            val_accuracy = model.score(X_val_data, y_val)
        else:
            val_accuracy = 'N/A'
            
        final_accuracies[name] = {
            'Training Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Training Time (s)': training_time
        }
        
    except Exception as e:
        training_time = time.time() - train_start_time 
        print(f"\nERROR: Model {name} failed to train/predict. Error: {e}")
        metrics_list.append(pd.Series({'Model': name, 'Accuracy (%)': np.nan}))
        final_accuracies[name] = {'Training Accuracy': np.nan, 'Validation Accuracy': np.nan, 'Test Accuracy': np.nan, 'Training Time (s)': 0}


# --- 5. SUMMARY TABLE AND OVERFITTING ANALYSIS ---

df_metrics_summary = pd.DataFrame(metrics_list).set_index('Model')

print("\n" + "="*50)
print("COMPREHENSIVE SUMMARY (6 Models)")
print("="*50)
print(df_metrics_summary.to_markdown())

print("\n=== Overfitting Analysis ===")
for model_name, accs in final_accuracies.items():
    if not np.isnan(accs['Test Accuracy']):
        train_acc = accs['Training Accuracy']
        test_acc = accs['Test Accuracy']
        print(f"Model: {model_name} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")

# Final time summary
total_time = time.time() - total_start_time
print(f"\nTotal Script Execution Time: {format_time(total_time)}")
