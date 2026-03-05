import numpy as np
import pandas as pd
import time
import os
from scipy.stats import skew, kurtosis
from scipy.signal import stft
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping  # Đã thêm early_stopping ở đây
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import seaborn as sns
from PyEMD import EMD

# --- GLOBAL VARIABLE DECLARATIONS ---
MAX_IMFS = 3

# --- UTILITY FUNCTIONS AND HHT FEATURE EXTRACTION ---

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} minutes {remaining_seconds:.2f} seconds"

def create_output_folder(folder_name="Model_Evaluation_STFT_HHT_Output"):
    """Create a folder to store Confusion Matrix images."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created output folder: '{folder_name}'")
    return folder_name

def calculate_imf_stats(imf_signals, max_imfs=3):
    """Compute 4 statistical features (Energy, Kurtosis, Skewness, StdDev)
    on the first IMFs."""
    stats_list = []
    
    # Only consider the first max_imfs IMFs
    num_imfs = min(imf_signals.shape[0], max_imfs)
    
    for i in range(num_imfs):
        imf = imf_signals[i]
        
        # 1. Energy
        energy = np.sum(imf**2)
        # 2. Kurtosis
        kurt = kurtosis(imf)
        # 3. Skewness
        skw = skew(imf)
        # 4. Standard Deviation
        std_dev = np.std(imf)
        
        stats_list.extend([energy, kurt, skw, std_dev])
        
    # Pad with zeros if the number of IMFs < max_imfs (to ensure fixed-length feature vectors)
    padding_needed = max_imfs * 4 - len(stats_list)
    stats_list.extend([0.0] * padding_needed)
    
    return stats_list

# HYBRID FEATURE EXTRACTION: STFT (20) + HHT/EMD (36) = 56 FEATURES
def extract_stft_hht_features(data, sampling_rate, nperseg=None, noverlap=None):
    features_list = []
    labels_list = data['label'].values
    grouped = data.groupby('label')
    emd = EMD()  # Initialize EMD sifter
    
    for wconfid, group in grouped:
        x_signal = group['std_x'].values
        y_signal = group['std_y'].values
        z_signal = group['std_z'].values
        signal_length = len(x_signal)
        
        # --- 1. STFT Feature Extraction (20 Features) ---
        if nperseg is None: 
            nperseg = max(16, signal_length // 2)
        effective_nperseg = min(nperseg, signal_length)
        if noverlap is None: 
            noverlap = effective_nperseg // 2
        effective_noverlap = min(noverlap, effective_nperseg - 1)
        if effective_nperseg < 2: 
            continue
        
        f, t, Zxx_x = stft(x_signal, fs=sampling_rate, nperseg=effective_nperseg, noverlap=effective_noverlap)
        _, _, Zxx_y = stft(y_signal, fs=sampling_rate, nperseg=effective_nperseg, noverlap=effective_noverlap)
        _, _, Zxx_z = stft(z_signal, fs=sampling_rate, nperseg=effective_nperseg, noverlap=effective_noverlap)
        
        mag_x, mag_y, mag_z = np.abs(Zxx_x), np.abs(Zxx_y), np.abs(Zxx_z)
        n_times = mag_x.shape[1]
        
        # Interpolation setup
        if n_times > 1:
            step = (signal_length - effective_nperseg) / (n_times - 1)
            time_indices = np.arange(effective_nperseg // 2, signal_length - effective_nperseg // 2 + 1, step)
        else:
            time_indices = np.array([(signal_length - effective_nperseg) / 2])
        if len(time_indices) > n_times: 
            time_indices = time_indices[:n_times]
        elif len(time_indices) < n_times: 
            time_indices = np.linspace(effective_nperseg // 2, signal_length - effective_nperseg // 2, n_times)
        sample_indices = np.arange(signal_length)
        
        stft_segment_features = []
        for time_idx in range(n_times):
            mag_x_t, mag_y_t, mag_z_t = mag_x[:, time_idx], mag_y[:, time_idx], mag_z[:, time_idx]
            all_mags_t = np.concatenate([mag_x_t, mag_y_t, mag_z_t])
            
            # Compute 20 features (simplified logic)
            if np.sum(all_mags_t) > 0:
                top_indices = np.argsort(all_mags_t)[-5:]
                dominant_frequency = np.sum(
                    f[top_indices % len(f)] * all_mags_t[top_indices]
                ) / np.sum(all_mags_t[top_indices])
                spectral_centroid = np.sum(np.tile(f, 3) * all_mags_t) / np.sum(all_mags_t)
                spectral_bandwidth = np.sqrt(
                    np.sum(((np.tile(f, 3) - spectral_centroid) ** 2) * all_mags_t) / np.sum(all_mags_t)
                )
                spectral_flatness = np.exp(np.mean(np.log(all_mags_t + 1e-6))) / np.mean(all_mags_t)
                spectral_entropy = -np.sum(
                    (all_mags_t / np.sum(all_mags_t)) * np.log((all_mags_t / np.sum(all_mags_t)) + 1e-6)
                )
            else:
                dominant_frequency = 0
                spectral_centroid = 0
                spectral_bandwidth = 0
                spectral_flatness = 0
                spectral_entropy = 0
            
            rms_energy = np.mean([np.sqrt(np.mean(mag**2)) for mag in [mag_x_t, mag_y_t, mag_z_t]])
            mean_magnitude_x, std_magnitude_x = np.mean(mag_x_t), np.std(mag_x_t)
            mean_magnitude_y, std_magnitude_y = np.mean(mag_y_t), np.std(mag_y_t)
            mean_magnitude_z, std_magnitude_z = np.mean(mag_z_t), np.std(mag_z_t)
            peak_freq_x, peak_freq_y, peak_freq_z = f[np.argmax(mag_x_t)], f[np.argmax(mag_y_t)], f[np.argmax(mag_z_t)]
            freq_skewness_x, freq_skewness_y, freq_skewness_z = skew(mag_x_t), skew(mag_y_t), skew(mag_z_t)
            freq_kurtosis = np.mean([kurtosis(mag_x_t), kurtosis(mag_y_t), kurtosis(mag_z_t)])
            total_power = np.sum(all_mags_t**2)
            
            stft_segment_features.append([
                dominant_frequency, spectral_centroid, spectral_bandwidth, spectral_flatness,
                rms_energy, mean_magnitude_x, std_magnitude_x, mean_magnitude_y, std_magnitude_y,
                mean_magnitude_z, std_magnitude_z, spectral_entropy, peak_freq_x, peak_freq_y, 
                peak_freq_z, freq_skewness_x, freq_skewness_y, freq_skewness_z, freq_kurtosis, total_power
            ])
        
        interpolators = [
            interp1d(time_indices, np.array(stft_segment_features)[:, i], kind='linear', fill_value='extrapolate') 
            for i in range(20)
        ]
        interpolated_stft_features = np.array([interp(sample_indices) for interp in interpolators]).T
        
        # --- 2. HHT Feature Extraction (36 Features) ---
        hht_features = []
        for signal in [x_signal, y_signal, z_signal]:
            # Decompose into IMFs
            try:
                imfs = emd.emd(signal)
            except:
                imfs = np.zeros((MAX_IMFS, signal_length))
            
            # Compute 4 statistics for the first 3 IMFs
            stats = calculate_imf_stats(imfs, max_imfs=MAX_IMFS)
            hht_features.extend(stats)
            
        # HHT features are single values per group, so we repeat them to match the signal length
        hht_features_matrix = np.tile(np.array(hht_features), (signal_length, 1))
        
        # --- 3. Combine and Store ---
        combined_features = np.hstack([interpolated_stft_features, hht_features_matrix])  # (signal_length, 56)
        features_list.append(combined_features)
    
    if not features_list: 
        raise ValueError("No features were extracted")
    
    X = np.vstack(features_list)
    return X, labels_list

def get_model_metrics(y_test, y_pred, model_name, training_time):
    """Compute and return performance metrics as floats (multiplied by 100)."""
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
    """Print detailed report and plot Confusion Matrix, saving the image."""
    
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    print(f"\n--- Model: {model_name} ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[f'Class {i+1}' for i in range(9)]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i+1}' for i in range(9)],
                yticklabels=[f'Class {i+1}' for i in range(9)])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # SAVE IMAGE TO FOLDER
    plt.savefig(os.path.join(output_folder, f"{safe_name}_CM.png"))
    plt.close() 
    print(f"Confusion Matrix saved to {output_folder}/{safe_name}_CM.png")

# --- MAIN EXECUTION BLOCK ---

OUTPUT_FOLDER_PATH = create_output_folder("Model_Evaluation_STFT_HHT_Output")  # Create folder
total_start_time = time.time()
final_accuracies = {}
metrics_list = []

# Load data
print("\n=== Loading Data ===")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Dataset', 'processed_data.csv')
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}. Please check the path.")
    exit()

# Hybrid feature extraction (STFT + HHT/EMD)
print("\n=== Extracting Hybrid STFT + HHT Features (56 Total) ===")
stft_hht_start_time = time.time()
X, y = extract_stft_hht_features(data, sampling_rate=1)
stft_hht_time = time.time() - stft_hht_start_time
print(f"Extraction Time: {format_time(stft_hht_time)}")
print(f"Total features extracted: {X.shape[1]}")  # Should be 56

# Adjust labels to start from 0 (from 1-9 to 0-8)
y = y - 1

# Handle NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Split data into train/test
print("\n=== Splitting Data ===")
split_start_time = time.time()
# Train/Test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train/Validation split (80/20 of training set)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
split_time = time.time() - split_start_time
print(f"Train/Validation/Test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
print(f"Data Splitting Time: {format_time(split_time)}")

# 3. Standardization (Required for MLP, QSVM, LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# --- TRAINING AND EVALUATING 6 MODELS ---

print("\n=== Training All 6 Hybrid STFT+HHT Models ===")

# List of models and parameters
models_to_run = [
    # 1. MLP (Requires scaled data)
    ("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=40, random_state=42, verbose=False), True),
    # 2. Random Forest (Does not require scaled data)
    ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1), False),
    # 3. CatBoost (Does not require scaled data)
    ("CatBoost", CatBoostClassifier(iterations=10, depth=5, learning_rate=0.1, random_seed=42, verbose=0), False),
    # 4. LightGBM (Sửa: bỏ tham số early_stopping_rounds ở khởi tạo)
    ("LightGBM", LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1), False),
    # 5. Logistic Regression (Sửa: bỏ multi_class='multinomial')
    ("Logistic Regression", LogisticRegression(solver='saga', max_iter=500, random_state=42, n_jobs=-1), True),
    # 6. Quadratic SVM (Requires scaled data)
    ("Quadratic Support vector Machine", SVC(kernel='poly', degree=2, max_iter=500, random_state=42, verbose=False), True),
]

for name, model, needs_scaling in models_to_run:
    
    # Choose scaled or raw data
    X_train_data = X_train_scaled if needs_scaling else X_train
    X_val_data = X_val_scaled if needs_scaling else X_val
    X_test_data = X_test_scaled if needs_scaling else X_test
    
    train_start_time = time.time()
    
    try:
        # Train model
        if name == "LightGBM":
            # Sửa: Sử dụng callbacks thay cho argument trực tiếp
            model.fit(
                X_train_data, y_train, 
                eval_set=[(X_val_data, y_val)], 
                callbacks=[early_stopping(stopping_rounds=10)]
            )
        elif name == "CatBoost":
            model.fit(X_train_data, y_train, eval_set=[(X_val_data, y_val)], verbose=0)
        else:
            model.fit(X_train_data, y_train)
            
        training_time = time.time() - train_start_time
        y_pred = model.predict(X_test_data)
        
        # 1. Collect detailed metrics
        metrics_series = get_model_metrics(y_test, y_pred, name, training_time)
        metrics_list.append(metrics_series)
        
        # 2. Print and save detailed report
        print_detailed_evaluation(name, y_test, y_pred, OUTPUT_FOLDER_PATH)
        
        # 3. Save results for overfitting check
        train_accuracy = model.score(X_train_data, y_train)
        if name == "Random Forest":
            val_accuracy = model.oob_score_ if hasattr(model, 'oob_score_') and model.oob_score_ is not None else 'N/A'
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
        print(f"\nERROR: Model {name} failed to train/predict. Time: {format_time(training_time)}. Error: {e}")
        # Add NaN values if training fails
        metrics_list.append(pd.Series({'Model': name, 'Accuracy (%)': np.nan, 'Precision (%)': np.nan, 'Recall (%)': np.nan, 'F1-Score (%)': np.nan, 'Training Time (s)': 0}))
        final_accuracies[name] = {'Training Accuracy': np.nan, 'Validation Accuracy': np.nan, 'Test Accuracy': np.nan, 'Training Time (s)': 0}


# --- SUMMARY AND OVERFITTING ANALYSIS ---

df_metrics_summary = pd.DataFrame(metrics_list).set_index('Model')

print("\n" + "="*50)
print("COMPREHENSIVE HYBRID STFT+HHT MODEL SUMMARY (56 Features)")
print("="*50)
# Summary table of Accuracy, Precision, Recall, F1-Score
print(df_metrics_summary.to_markdown())

print("\n=== Overfitting Analysis ===")
for model_name, accs in final_accuracies.items():
    if not np.isnan(accs['Test Accuracy']):
        train_acc = accs['Training Accuracy']
        test_acc = accs['Test Accuracy']
        val_acc = accs['Validation Accuracy']
        
        print(f"\nModel: {model_name}")
        print(f"Training Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        # Overfitting check logic
        if (isinstance(val_acc, float) and train_acc - val_acc > 0.1) or (train_acc - test_acc > 0.1):
            print(f"Warning: Potential overfitting detected! Gap: {train_acc - test_acc:.4f}.")
        else:
            print("No significant overfitting detected (or mild overfitting).")

# Time summary
total_time = time.time() - total_start_time
print("\n=== Summary of Computational Times ===")
print(f"STFT/HHT Extraction: {format_time(stft_hht_time)}")
print(f"Data Splitting: {format_time(split_time)}")
print(f"Total Script Execution Time: {format_time(total_time)}")

print("\nTraining and evaluation completed! All Confusion Matrices saved to the output folder.")
