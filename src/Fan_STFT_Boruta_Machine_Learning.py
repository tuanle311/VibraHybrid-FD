import numpy as np
import pandas as pd
import time
import os
from scipy.stats import skew, kurtosis
from scipy.signal import stft
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier # Giữ import này mặc dù WKNN bị loại, vì nó có thể hữu ích
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler 
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

# THƯ VIỆN SHAP
import shap 
# THƯ VIỆN HHT/EMD
from emd import sift 

# --- KHAI BÁO BIẾN TOÀN CỤC ---
OUTPUT_FOLDER = "Model_Evaluation_STFT_HHT_Boruta_Output"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Dataset', 'processed_data.csv')
SAMPLING_RATE = 1
MAX_IMFS = 3

# Tên 56 đặc trưng (20 STFT + 36 HHT)
imf_stats = ['IMF_Energy', 'IMF_Kurtosis', 'IMF_Skewness', 'IMF_StdDev']
axes = ['x', 'y', 'z']
new_hht_feature_names = []
for axis in axes:
    for i in range(1, MAX_IMFS + 1):
        for stat in imf_stats:
            new_hht_feature_names.append(f'{stat}_IMF{i}_{axis}')

FEATURE_NAMES_56 = [
    'dominant_frequency', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness',
    'rms_energy', 'mean_magnitude_x', 'std_magnitude_x', 'mean_magnitude_y', 'std_magnitude_y',
    'mean_magnitude_z', 'std_magnitude_z', 'spectral_entropy', 'peak_freq_x', 'peak_freq_y',
    'peak_freq_z', 'freq_skewness_x', 'freq_skewness_y', 'freq_skewness_z', 'freq_kurtosis',
    'total_power'
] + new_hht_feature_names

# --- 1. CÁC HÀM TIỆN ÍCH VÀ TRÍCH XUẤT ĐẶC TRƯNG ---

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} minutes {remaining_seconds:.2f} seconds"

def create_output_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created output folder: '{folder_name}'")
    return folder_name

def calculate_imf_stats(imf_signals, max_imfs=3):
    stats_list = []
    num_imfs = min(imf_signals.shape[0], max_imfs)
    
    for i in range(num_imfs):
        imf = imf_signals[i]
        energy = np.sum(imf**2)
        kurt = kurtosis(imf)
        skw = skew(imf)
        std_dev = np.std(imf)
        stats_list.extend([energy, kurt, skw, std_dev])
        
    padding_needed = max_imfs * 4 - len(stats_list)
    stats_list.extend([0.0] * padding_needed)
    return stats_list

def extract_stft_hht_features(data, sampling_rate, nperseg=None, noverlap=None):
    """Trích xuất 56 đặc trưng lai STFT + HHT/EMD."""
    features_list = []
    labels_list = data['label'].values
    grouped = data.groupby('label')
    emd_sifter = sift
    
    for wconfid, group in grouped:
        x_signal = group['std_x'].values
        y_signal = group['std_y'].values
        z_signal = group['std_z'].values
        signal_length = len(x_signal)
        
        # --- 1. STFT Feature Extraction ---
        if nperseg is None: nperseg = max(16, signal_length // 2)
        effective_nperseg = min(nperseg, signal_length)
        if noverlap is None: noverlap = effective_nperseg // 2
        effective_noverlap = min(noverlap, effective_nperseg - 1)
        if effective_nperseg < 2: continue
        
        f, t, Zxx_x = stft(x_signal, fs=sampling_rate, nperseg=effective_nperseg, noverlap=effective_noverlap)
        _, _, Zxx_y = stft(y_signal, fs=sampling_rate, nperseg=effective_nperseg, noverlap=effective_noverlap)
        _, _, Zxx_z = stft(z_signal, fs=sampling_rate, nperseg=effective_nperseg, noverlap=effective_noverlap)
        
        mag_x, mag_y, mag_z = np.abs(Zxx_x), np.abs(Zxx_y), np.abs(Zxx_z)
        n_times = mag_x.shape[1]
        
        if n_times > 1:
            step = (signal_length - effective_nperseg) / (n_times - 1)
            time_indices = np.arange(effective_nperseg // 2, signal_length - effective_nperseg // 2 + 1, step)
        else:
            time_indices = np.array([(signal_length - effective_nperseg) / 2])
        if len(time_indices) != n_times: time_indices = np.linspace(effective_nperseg // 2, signal_length - effective_nperseg // 2, n_times)
        sample_indices = np.arange(signal_length)
        
        stft_segment_features = []
        for time_idx in range(n_times):
            mag_x_t, mag_y_t, mag_z_t = mag_x[:, time_idx], mag_y[:, time_idx], mag_z[:, time_idx]
            all_mags_t = np.concatenate([mag_x_t, mag_y_t, mag_z_t])
            
            # Tính 20 đặc trưng STFT
            dominant_frequency = np.sum(f[np.argsort(all_mags_t)[-5:] % len(f)] * all_mags_t[np.argsort(all_mags_t)[-5:]]) / np.sum(all_mags_t[np.argsort(all_mags_t)[-5:]]) if np.sum(all_mags_t) > 0 else 0
            spectral_centroid = np.sum(np.tile(f, 3) * all_mags_t) / np.sum(all_mags_t) if np.sum(all_mags_t) > 0 else 0
            spectral_bandwidth = np.sqrt(np.sum(((np.tile(f, 3) - spectral_centroid)**2) * all_mags_t) / np.sum(all_mags_t)) if np.sum(all_mags_t) > 0 else 0
            spectral_flatness = np.exp(np.mean(np.log(all_mags_t + 1e-6))) / np.mean(all_mags_t) if np.mean(all_mags_t) > 0 else 0
            rms_energy = np.mean([np.sqrt(np.mean(mag**2)) for mag in [mag_x_t, mag_y_t, mag_z_t]])
            mean_magnitude_x, std_magnitude_x = np.mean(mag_x_t), np.std(mag_x_t)
            mean_magnitude_y, std_magnitude_y = np.mean(mag_y_t), np.std(mag_y_t)
            mean_magnitude_z, std_magnitude_z = np.mean(mag_z_t), np.std(mag_z_t)
            norm_spectrum = all_mags_t / np.sum(all_mags_t) if np.sum(all_mags_t) > 0 else np.zeros_like(all_mags_t)
            spectral_entropy = -np.sum(norm_spectrum * np.log(norm_spectrum + 1e-6))
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
            
        interpolated_stft_features = np.array([interp(sample_indices) for interp in [interp1d(time_indices, np.array(stft_segment_features)[:, i], kind='linear', fill_value='extrapolate') for i in range(20)]]).T
        
        # --- 2. HHT Feature Extraction ---
        hht_features = []
        for signal in [x_signal, y_signal, z_signal]:
            try:
                imfs = emd_sifter(signal) 
            except:
                imfs = np.zeros((MAX_IMFS, signal_length))
            
            stats = calculate_imf_stats(imfs, max_imfs=MAX_IMFS)
            hht_features.extend(stats)
            
        hht_features_matrix = np.tile(np.array(hht_features), (signal_length, 1))
        
        # --- 3. Kết hợp và Lưu trữ ---
        combined_features = np.hstack([interpolated_stft_features, hht_features_matrix])
        features_list.append(combined_features)
    
    if not features_list: raise ValueError("No features were extracted")
    
    X = np.vstack(features_list)
    return X, labels_list

def get_model_metrics_series(model_name, y_test, y_pred, training_time):
    """Tính toán và trả về các chỉ số hiệu suất dưới dạng số float (đã nhân 100)."""
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

def plot_detailed_metrics(model_name, y_test, y_pred, y_prob, output_folder, num_classes=9):
    """Vẽ và lưu Confusion Matrix, ROC Curve (Macro Avg), và Calibration Plot."""
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i+1}' for i in range(num_classes)],
                yticklabels=[f'Class {i+1}' for i in range(num_classes)],
                annot_kws={"size": 10})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{safe_name}_CM.png'))
    plt.close()
    
    # 2. ROC Curve (Macro Average)
    if y_prob is not None and y_prob.shape[1] == num_classes:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
        
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(all_fpr, mean_tpr,
                 label=f'Macro-average ROC (AUC = {roc_auc_macro:0.4f})',
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.title(f'ROC Curve (Macro-Avg) - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_folder, f'{safe_name}_ROC.png'))
        plt.close()

    # 3. Calibration Plot (All Classes)
    if y_prob is not None and y_prob.shape[1] == num_classes:
        plt.figure(figsize=(12, 8))
        for class_idx in range(num_classes):
            y_test_binary = (y_test == class_idx).astype(int)
            y_prob_class = y_prob[:, class_idx]
            if np.sum(y_test_binary) == 0: continue
            
            prob_true, prob_pred = calibration_curve(y_test_binary, y_prob_class, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=f'Class {class_idx + 1}')
            
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plots - {model_name} (All Classes)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f'{safe_name}_CALIBRATION.png'))
        plt.close()

def plot_shap_summary(model, X_train_bg, X_test_data, feature_names, model_name, output_folder, explainer_type):
    """Tính toán và vẽ biểu đồ SHAP Summary (cho cả Tree và Kernel Explainer)."""
    
    # 1. Lấy mẫu dữ liệu cho SHAP (giới hạn 1000 mẫu để chạy nhanh)
    if X_test_data.shape[0] > 1000:
         np.random.seed(42)
         X_sample = X_test_data[np.random.choice(X_test_data.shape[0], 1000, replace=False)]
    else:
         X_sample = X_test_data
         
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    
    print(f"Calculating SHAP values for {model_name} using {explainer_type}Explainer...")
    
    try:
        if explainer_type == "Tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_df)
            
        elif explainer_type == "Kernel":
            # CẢNH BÁO: Kernel Explainer rất chậm, chỉ lấy 100 mẫu background
            if X_train_bg.shape[0] > 100:
                np.random.seed(42)
                background = X_train_bg[np.random.choice(X_train_bg.shape[0], 100, replace=False)]
            else:
                background = X_train_bg
                
            print(f"!!! WARNING: {model_name} uses KernelExplainer on {background.shape[0]} background samples. This may be very slow.")

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample_df, nsamples=100)

        # 2. Vẽ biểu đồ Summary (Bar Plot cho Multi-class)
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values, 
                X_sample_df, 
                plot_type="bar", 
                show=False,
                class_names=[f'Class {i+1}' for i in range(len(shap_values))]
            )
        else:
            shap.summary_plot(
                shap_values, 
                X_sample_df, 
                plot_type="bar", 
                show=False
            )

        # 3. Lưu và Đóng Plot
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(output_folder, f'{safe_name}_SHAP_Summary.png'), bbox_inches='tight')
        plt.close()
        print(f"SHAP Summary Plot saved to {output_folder}/{safe_name}_SHAP_Summary.png")
        
    except Exception as e:
        print(f"Warning: SHAP calculation failed for {model_name}. Error: {e}")

def plot_radar_chart(results_df, output_folder):
    """Vẽ Radar Chart so sánh hiệu suất các mô hình (chỉ metrics) và lưu vào folder."""
    categories = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
    df_plot = results_df[categories].copy()
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    min_val = df_plot[categories].min().min() * 0.95
    max_val = df_plot[categories].max().max() * 1.05
    
    ax.set_yticks(np.linspace(min_val, max_val, 5))
    ax.set_yticklabels([f'{y:.1f}%' for y in np.linspace(min_val, max_val, 5)], color="grey", size=10)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    
    for model in df_plot.index.tolist():
        values = df_plot.loc[model].values.flatten().tolist()
        if np.any(np.isnan(values)): continue
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='-', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_title('Model Performance Comparison (Optimized Hybrid Features)', size=14, y=1.1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Radar_Chart_Comparison.png'))
    plt.close()
    print(f"Radar Chart saved to {output_folder}/Radar_Chart_Comparison.png")


# --- 3. KHỐI THỰC THI CHÍNH ---

output_folder_path = create_output_folder(OUTPUT_FOLDER)

total_start_time = time.time()

try:
    print("\n=== 1. Loading Data ===")
    load_start_time = time.time()
    data = pd.read_csv(DATA_PATH)
    load_time = time.time() - load_start_time
    
    # --- Trích xuất Đặc trưng (56 Features) ---
    print("\n=== 2. Extracting STFT + HHT Features (56 Total) ===")
    stft_hht_start_time = time.time()
    X, y = extract_stft_hht_features(data, sampling_rate=SAMPLING_RATE)
    stft_hht_time = time.time() - stft_hht_start_time
    print(f"Extraction complete. Time: {format_time(stft_hht_time)}")
    print(f"Total features extracted (STFT + HHT): {X.shape[1]}")
    
    y = y - 1
    
    # --- Standardization và Boruta ---
    print("\n=== 3. Standardization and Boruta Feature Selection ===")
    
    X_train_boruta, X_temp, y_train_boruta, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_boruta)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0) 
    
    # Boruta Selection
    boruta_start_time = time.time()
    rf_boruta = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    selector = BorutaPy(rf_boruta, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
    selector.fit(X_scaled, y_train_boruta)
    
    selected_indices = selector.support_
    selected_features = [FEATURE_NAMES_56[i] for i in range(len(FEATURE_NAMES_56)) if selected_indices[i]]
    boruta_time = time.time() - boruta_start_time
    
    print(f"Boruta Selection complete. Time: {format_time(boruta_time)}")
    print(f"Total features selected by Boruta: {len(selected_features)}")
    print(f"Selected Features: {selected_features}")

    # --- 4. Áp dụng Boruta và Final Split (X_train, X_test) ---
    print("\n=== 4. Final Data Split and Scaling (Using Selected Features) ===")
    
    X_full_selected = X[:, selected_indices]
    
    # Split lại (Train/Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_full_selected, y, test_size=0.2, random_state=42)
    # Chia Training thành Train/Validation (64/16)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    # Scaling lần cuối
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    # X_train_bg (Data Scaled for Kernel Explainer Background)
    # Sử dụng X_train_scaled vì các mô hình Kernel (MLP, LR, QSVM) đều cần data đã scale
    X_train_bg = X_train_scaled 
    
    # --- 5. HUẤN LUYỆN, ĐÁNH GIÁ VÀ SHAP ---
    
    results = []
    
    print("\n=== 5. Training, Evaluation, and SHAP Analysis ===")

    # Phân loại Explainer Type
    # LOẠI BỎ WKNN VÀ BAGGED TREE
    models_to_run = [
        # Kernel Explainer Models (Chậm - Cần scaled data)
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=50, random_state=42, verbose=False), True, "Kernel"),
        ("LR", LogisticRegression(solver='saga', multi_class='multinomial', max_iter=5000, random_state=42, n_jobs=-1), True, "Kernel"),
        ("QSVM", SVC(kernel='poly', degree=2, max_iter=5000, random_state=42, probability=True, verbose=False), True, "Kernel"),

        # Tree Explainer Models (Nhanh - Không cần scaled data)
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1), False, "Tree"),
        ("CatBoost", CatBoostClassifier(iterations=25, depth=7, learning_rate=0.1, random_seed=42, verbose=0), False, "Tree"),
        ("LightGBM", LGBMClassifier(n_estimators=1, max_depth=7, learning_rate=0.1, random_state=42, verbose=-1), False, "Tree"),
    ]
    
    for name, model, needs_scaling, explainer_type in models_to_run:
        train_start_time = time.time()
        
        # Chọn data phù hợp
        X_train_data = X_train_scaled if needs_scaling else X_train
        X_test_data = X_test_scaled if needs_scaling else X_test
        
        try:
            model.fit(X_train_data, y_train)
            training_time = time.time() - train_start_time
            y_pred = model.predict(X_test_data)
            y_prob = model.predict_proba(X_test_data) if hasattr(model, 'predict_proba') else None
            
            metrics_series = get_model_metrics_series(name, y_test, y_pred, training_time)
            results.append(metrics_series)
            
            # IN VÀ LƯU BIỂU ĐỒ CHI TIẾT (CM, ROC, CAL)
            print(f"\n--- DETAILED EVALUATION FOR: {name} ---")
            print(f"Training Time: {format_time(training_time)}")
            print(f"Accuracy: {metrics_series['Accuracy (%)']:.2f}% | F1-Score: {metrics_series['F1-Score (%)']:.2f}%")
            print(classification_report(y_test, y_pred, target_names=[f'Class {i+1}' for i in range(9)]))
            plot_detailed_metrics(name, y_test, y_pred, y_prob, output_folder_path)

            # --- SHAP ANALYSIS ---
            # X_test_data đã được scale nếu needs_scaling=True
            plot_shap_summary(model, X_train_bg, X_test_data, selected_features, name, output_folder_path, explainer_type)

        except Exception as e:
            training_time = time.time() - train_start_time
            print(f"\nERROR: Model {name} failed to train/predict. Time: {format_time(training_time)}. Metrics will be NaN. Error: {e}")
            results.append(pd.Series({'Model': name, 'Accuracy (%)': np.nan, 'Precision (%)': np.nan, 'Recall (%)': np.nan, 'F1-Score (%)': np.nan, 'Training Time (s)': training_time}))

    # 6. TỔNG HỢP VÀ VẼ BIỂU ĐỒ CUỐI CÙNG
    df_results = pd.DataFrame(results).set_index('Model')
    
    print("\n--- Summary of All Model Performance (STFT + Boruta + HHT) ---")
    print(df_results)
    
    # Vẽ Radar Chart so sánh metrics chính
    print("\n=== 7. Generating Radar Chart ===")
    plot_radar_chart(df_results, output_folder_path)
    
    df_results.to_csv(os.path.join(output_folder_path, 'All_Model_Metrics_Time.csv'))
    print(f"Metrics table saved to {output_folder_path}/All_Model_Metrics_Time.csv")

    total_time = time.time() - total_start_time
    print(f"\nTotal Script Execution Time: {format_time(total_time)}")

except FileNotFoundError:
    print(f"ERROR: File not found at {DATA_PATH}. Please check the DATA_PATH variable.")
except Exception as e:
    print(f"An unexpected error occurred during pipeline execution: {e}")
