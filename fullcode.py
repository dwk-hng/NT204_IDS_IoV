import matplotlib
matplotlib.use('Agg')  # Dùng backend không cần GUI, tránh lỗi tkinter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import subprocess
import importlib.util


def install_requirements():
    print("🔍 Đang kiểm tra môi trường và cài đặt các thư viện cần thiết...")

    tmp_pip_dir = os.path.join(os.getcwd(), ".tmp_pip")
    os.makedirs(tmp_pip_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Helper functions                                                    #
    # ------------------------------------------------------------------ #
    def pip_install(packages, extra_args=None):
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "--disable-pip-version-check",
        ]
        cmd += packages
        if extra_args:
            cmd += extra_args
        cmd += ["-q"]
        env = os.environ.copy()
        env["TMPDIR"] = tmp_pip_dir
        subprocess.check_call(cmd, env=env)

    def is_installed(module_name):
        return importlib.util.find_spec(module_name) is not None

    def install_if_missing(module_name, package_name=None, extra_args=None):
        pkg = package_name or module_name
        if not is_installed(module_name):
            print(f"  📦 Đang cài: {pkg}...")
            pip_install([pkg], extra_args=extra_args)
        else:
            print(f"  ✔️  Đã có: {pkg}")

    # ------------------------------------------------------------------ #
    #  1. Phát hiện GPU NVIDIA & CUDA version                             #
    # ------------------------------------------------------------------ #
    def detect_cuda_version():
        """
        Trả về chuỗi cuda version dạng 'cu121', 'cu118'...
        hoặc None nếu không có GPU / không có CUDA driver.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None  # Không có GPU

            # Lấy CUDA version từ nvidia-smi
            ver_result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            output = ver_result.stdout
            # nvidia-smi in ra "CUDA Version: 12.1" ở header
            import re
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                # Map sang PyTorch wheel tag (lấy version gần nhất PyTorch hỗ trợ)
                if major >= 12:
                    if minor >= 4:
                        return "cu124"
                    elif minor >= 1:
                        return "cu121"
                    else:
                        return "cu118"
                elif major == 11:
                    return "cu118"
                else:
                    return None  # CUDA quá cũ, fallback CPU
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None  # nvidia-smi không tồn tại = không có GPU

    cuda_version = detect_cuda_version()

    if cuda_version:
        print(f"\n✅ Phát hiện GPU NVIDIA với CUDA {cuda_version}!")
        print("   → Sẽ cài PyTorch + XGBoost bản GPU.\n")
    else:
        print("\n⚠️  Không phát hiện GPU NVIDIA (hoặc chưa cài CUDA driver).")
        print("   → Sẽ cài PyTorch bản CPU. XGBoost/CatBoost sẽ chạy CPU.\n")

    # ------------------------------------------------------------------ #
    #  2. Cài các thư viện thông thường (không phụ thuộc GPU)             #
    # ------------------------------------------------------------------ #
    install_if_missing("awscli")
    install_if_missing("catboost")
    install_if_missing("shap")
    install_if_missing("xgboost")
    install_if_missing("imblearn", "imbalanced-learn")
    install_if_missing("hyperopt")
    install_if_missing("pandas", "pandas==3.0.1")
    install_if_missing("joblib")
    install_if_missing("sklearn", "scikit-learn")
    install_if_missing("matplotlib")
    install_if_missing("seaborn")
    install_if_missing("lightgbm")

    # ------------------------------------------------------------------ #
    #  3. Cài PyTorch đúng version (GPU hoặc CPU)                        #
    # ------------------------------------------------------------------ #
    if not is_installed("torch"):
        if cuda_version:
            torch_index = f"https://download.pytorch.org/whl/{cuda_version}"
            print(f"  📦 Đang cài PyTorch bản GPU ({cuda_version})...")
            pip_install(["torch", "torchvision", "torchaudio"], extra_args=["--index-url", torch_index])
        else:
            print("  📦 Đang cài PyTorch bản CPU...")
            pip_install(["torch", "torchvision", "torchaudio"], extra_args=["--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        # Torch đã có — kiểm tra xem bản đang cài có hỗ trợ CUDA không
        try:
            import torch
            if cuda_version and not torch.cuda.is_available():
                print("  ⚠️  Torch đã cài nhưng KHÔNG hỗ trợ CUDA!")
                print("  🔄 Đang gỡ và cài lại PyTorch bản GPU...")
                pip_install(["--upgrade", "torch", "torchvision", "torchaudio"],
                            extra_args=["--index-url", f"https://download.pytorch.org/whl/{cuda_version}"])
            elif cuda_version and torch.cuda.is_available():
                print(f"  ✔️  PyTorch đã có + CUDA khả dụng ({torch.cuda.get_device_name(0)})")
            else:
                print("  ✔️  PyTorch đã có (CPU mode)")
        except Exception:
            print("  ✔️  PyTorch đã có")

    # ------------------------------------------------------------------ #
    #  4. Cấu hình OpenCL cho LightGBM GPU trên Linux (nếu có root)      #
    # ------------------------------------------------------------------ #
    if os.name == 'posix':
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.system("apt-get -qq install -y ocl-icd-libopencl1 clinfo > /dev/null")
            os.system("mkdir -p /etc/OpenCL/vendors && echo 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd")
        else:
            print("  ⚠️  Linux không có quyền root, bỏ qua cấu hình OpenCL.")

    # ------------------------------------------------------------------ #
    #  5. Báo cáo kết quả cuối                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "="*55)
    print("📋 KẾT QUẢ KIỂM TRA MÔI TRƯỜNG:")
    try:
        import torch
        print(f"   PyTorch version : {torch.__version__}")
        print(f"   CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU             : {torch.cuda.get_device_name(0)}")
            print(f"   VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception:
        print("   PyTorch         : Không load được")
    print("="*55 + "\n")

# Chạy cài đặt trước khi import thư viện
install_requirements()

import pandas as pd
import numpy as np
import gc
import torch
import joblib
import json
import time
from pathlib import Path

# Import thư viện Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import shap
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Any, cast


# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ TẠO THƯ MỤC TỰ ĐỘNG
# ==============================================================================
# Bỏ comment 2 dòng dưới nếu chạy trên môi trường Google Colab
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# Đường dẫn gốc
# base_dir = "/content/drive/MyDrive/[IDPS]-Dataste"
# Nếu chạy trên máy tính cá nhân, có thể đổi base_dir thành dạng thư mục hiện tại:
# base_dir = "[IDPS]-Dataste"
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "[IDPS]-Dataste")

csv_dir = os.path.join(base_dir, "IDS2018/CSV_Data")
cic_ddos_out_dir = os.path.join(base_dir, "cic_ddos_2018")
stage3_out_dir = os.path.join(base_dir, "output_stage3")
stage3_checkpoint_dir = os.path.join(base_dir, "checkpoint_rf_stage3")
stage4_out_dir = os.path.join(base_dir, "output_stage4_xai")

# Kiểm tra và tạo tất cả các thư mục cần thiết
directories_to_create = [
    base_dir, csv_dir, cic_ddos_out_dir, 
    stage3_out_dir, stage3_checkpoint_dir, stage4_out_dir
]

print("🔍 Đang kiểm tra hệ thống file và tạo thư mục...")
for directory in directories_to_create:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Đã tạo thư mục mới: {directory}")
    else:
        print(f"✔️ Thư mục đã tồn tại: {directory}")

# ==============================================================================
# PHẦN DOWNLOAD DATASET
# ==============================================================================
def download_dataset():
    print("\n--- BẮT ĐẦU TẢI DATASET TỪ AWS ---")
    
    # Lệnh sync đồng bộ toàn bộ thư mục từ S3 về thư mục đích
    aws_command = f'aws s3 sync --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" "{csv_dir}/"'
    
    print(f"🚀 Đang thực thi lệnh: {aws_command}")
    
    # Chạy lệnh thông qua hệ điều hành
    exit_code = os.system(aws_command)
    
    if exit_code == 0:
        print("✅ Hoàn tất tải Dataset bằng aws s3 sync!")
    else:
        print("❌ Có lỗi xảy ra trong quá trình tải Dataset. Vui lòng kiểm tra lại cấu hình AWS CLI hoặc đường dẫn.")

# ==============================================================================
# STAGE 0: PREPROCESSING
# ==============================================================================
def stage0_preprocessing():
    print("\n--- BẮT ĐẦU GIAI ĐOẠN 0: PREPROCESSING ---")
    csv_dir = os.path.join(base_dir, "IDS2018/CSV_Data")
    out_dir = os.path.join(base_dir, "cic_ddos_2018/")
    all_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]

    
    if len(all_files) == 0:
        print("❌ LỖI: Không tìm thấy file .csv nào trong thư mục CSV!")
        return
    else:
        print(f"✅ Tìm thấy {len(all_files)} file. Bắt đầu đọc dữ liệu bằng chiến thuật Chunking...")
        # Lấy danh sách file và lọc thủ công (tránh dùng glob)
        all_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]

        df_list = []

        # --- 2. ĐỌC FILE TIẾT KIỆM RAM (BẢO TỒN TỈ LỆ GỐC 10%) ---
        fraction = 0.02 # Lấy 10% toàn bộ dataset

        # Nhóm các nhãn siêu hiếm (dưới 2000 dòng theo bảng dataset gốc) để giữ 100%
        rare_classes = [
            'Web attack-SQL Injection',
            'SQL Injection',
            'Brute Force-XSS',
            'Brute Force -XSS', 
            'Brute Force-Web',
            'Brute Force -Web',
            'DDOS attack-LOIC-UDP',
            'Infiltration' # Infiltration tuy 161k nhưng tỷ lệ rớt dòng rất cao, nên ưu tiên giữ nhiều
        ]

        for f in all_files:
            print(f" 📂 Đang xử lý: {os.path.basename(f)}")
            chunks = pd.read_csv(f, low_memory=False, chunksize=250000)

            for chunk in chunks:
                chunk.columns = chunk.columns.str.strip()

                if 'Label' in chunk.columns:
                    # Tách riêng nhóm phổ biến và nhóm siêu hiếm
                    rare_data = chunk[chunk['Label'].isin(rare_classes)]
                    common_data = chunk[~chunk['Label'].isin(rare_classes)]

                    # Lấy 10% cho các class phổ biến (Giữ đúng tỉ lệ gốc giữa Benign, HOIC, Hulk...)
                    if len(common_data) > 0:
                        common_data = common_data.sample(frac=fraction, random_state=42)

                    # Gộp lại (Nhóm hiếm giữ 100% để sống sót, nhóm phổ biến giữ 10% để giữ tỉ lệ)
                    sampled_chunk = pd.concat([common_data, rare_data])
                    df_list.append(sampled_chunk)

            gc.collect()

        # --- 3. LÀM SẠCH DỮ LIỆU ---
        print("🧹 Đang làm sạch dữ liệu...")

        # Gộp dữ liệu đã lấy mẫu
        df_full = pd.concat(df_list, ignore_index=True)
        del df_list
        gc.collect()
        print(f"🎉 Đã gom xong! Kích thước dữ liệu tổng: {df_full.shape}")
        df_full.columns = df_full.columns.str.strip()

        # BÍ KÍP 1: Loại bỏ các cột "thừa" (chỉ có ở vài file) sinh ra hàng loạt NaN khi gộp.
        # Lệnh này sẽ xóa bất kỳ cột nào bị thiếu trên 50% dữ liệu.
        df_full.dropna(thresh=len(df_full) * 0.5, axis=1, inplace=True)

        # BÍ KÍP 2: Cứu các nhãn tấn công bị gán giá trị Infinity.
        # Thay vì biến Inf thành NaN (và bị xóa), ta ép nó về 0 để giữ lại luồng tấn công.
        df_full.replace([np.inf, -np.inf], 0, inplace=True)

        # Lúc này dropna() mới thực sự an toàn để xóa các dòng rác
        df_full.dropna(inplace=True)
        df_full = df_full[df_full['Label'] != 'Label'] # Bỏ dòng header lặp lại

        # --- 4. TÁCH BIẾN VÀ CHUẨN HÓA ---
        cols_to_drop = ['Timestamp', 'Label']
        X = df_full.drop(columns=[col for col in cols_to_drop if col in df_full.columns])
        y = df_full['Label']

        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # BÍ KÍP 3: Chặn đánh số Vô cực sinh ra sau khi ép kiểu String -> Numeric
        X.replace([np.inf, -np.inf], 0, inplace=True)

        print("🏷️ Đang mã hóa nhãn...")
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # In map nhãn để đối chiếu ở Stage sau
        label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print("Bảng nhãn:", label_mapping)

        print("⚖️ Đang chia tập Train/Test (80/20) và chuẩn hóa...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Xóa các biến cồng kềnh trước khi lưu
        del df_full, X, y, y_encoded
        gc.collect()

        # --- 5. LƯU THÀNH PYTORCH TENSORS ---
        print("💾 Đang lưu file .pt...")
        torch.save(torch.tensor(X_train_scaled, dtype=torch.float32), out_dir + "X_train.pt")
        torch.save(torch.tensor(y_train, dtype=torch.long), out_dir + "y_train.pt")
        torch.save(torch.tensor(X_test_scaled, dtype=torch.float32), out_dir + "X_test.pt")
        torch.save(torch.tensor(y_test, dtype=torch.long), out_dir + "y_test.pt")

        print("✅ HOÀN TẤT STAGE 0! Dữ liệu đã sẵn sàng cho stage1-cic18.ipynb.")

# ==============================================================================
# STAGE 1: TRAINING CÁC MÔ HÌNH
# ==============================================================================
def stage1_training():
    print("\n--- BẮT ĐẦU GIAI ĐOẠN 1: HUẤN LUYỆN MÔ HÌNH ---")
    print("🚀 Đang tải dữ liệu từ file .pt...")
    start_load = time.time()

    try:
        X_train = torch.load(os.path.join(cic_ddos_out_dir, "X_train.pt")).numpy()
        y_train = torch.load(os.path.join(cic_ddos_out_dir, "y_train.pt")).numpy()
        X_test = torch.load(os.path.join(cic_ddos_out_dir, "X_test.pt")).numpy()
        y_test = torch.load(os.path.join(cic_ddos_out_dir, "y_test.pt")).numpy()
    except Exception as e:
        print("❌ Không tìm thấy file .pt từ Stage 0. Lỗi:", e)
        return

    gc.collect()

    print(f"✅ Tải dữ liệu hoàn tất trong {time.time() - start_load:.2f}s!")
    print(f"Kích thước X_train: {X_train.shape}, Kích thước y_train: {y_train.shape}")
    print(f"Kích thước X_test: {X_test.shape}, Kích thước y_test: {y_test.shape}")
    
    # Đếm số lượng class để XGBoost biết (multiclass hay binary)
    num_classes = len(np.unique(y_train))
    objective = "multi:softmax" if num_classes > 2 else "binary:logistic"

    print(f"Bài toán có {num_classes} nhãn.")

    # Khởi tạo các model với cấu hình chạy trên GPU
    models = {
        #1. Random Forest (Dùng XGBoost Random Forest để tận dụng GPU thay vì sklearn CPU)
        "RandomForest_GPU": XGBRFClassifier(
            n_estimators=100,
            tree_method="hist",   # Thuật toán tối ưu cho GPU
            device="cuda",        # Bật GPU
            objective=objective,
            random_state=42,
            n_jobs=-1
        ),

        # # 2. XGBoost
        "XGBoost_GPU": XGBClassifier(
            n_estimators=100,
            tree_method="hist",
            device="cuda",
            objective=objective,
            random_state=42,
            n_jobs=-1
        ),

        # # 3. LightGBM (Chạy CPU để tránh bug float32 của OpenCL)
        "LightGBM_CPU": LGBMClassifier(
            n_estimators=100,
            device_type="cpu",    # Đổi thành cpu ở đây
            random_state=42,
            verbose=-1,
            n_jobs=-1             # Vẫn dùng tối đa số luồng CPU của Colab
        ),

        # 4. CatBoost (Chạy CPU)
        "CatBoost_GPU": CatBoostClassifier(
            iterations=100,
            task_type="GPU",
            # task_type="CPU",          
            devices='0',              # Chạy trên GPU đầu tiên
            border_count=32,          # 🛑 QUAN TRỌNG: Giảm số lượng bin (mặc định là 128 trên GPU). Đây là "chìa khóa" chống tràn VRAM.
            depth=6,                  # Giữ độ sâu của cây ở mức tiêu chuẩn (từ 4 đến 6) để tiết kiệm bộ nhớ
            gpu_ram_part=0.85,        # Chỉ cho phép CatBoost dùng tối đa 85% VRAM, để lại 15% cho hệ điều hành/Colab duy trì hoạt động
            random_seed=42,
            verbose=0                 # Tắt log rác
        )
    }

    # Tạo thư mục lưu kết quả
    output_dir = Path(cic_ddos_out_dir + "output_cic_ddos_2018/stage1_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"🚀 Đang huấn luyện: {name}...")

        # Huấn luyện mô hình
        model.fit(X_train, y_train)

        # Dự đoán
        y_pred = model.predict(X_test)

        # Tính toán 4 chỉ số đánh giá (dùng 'weighted' cho tập dữ liệu mất cân bằng)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Lấy thêm báo cáo chi tiết
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, zero_division=0)

        # In kết quả ra màn hình
        print(f"🎯 Accuracy  : {acc:.4f}")
        print(f"🎯 Precision : {prec:.4f}")
        print(f"🎯 Recall    : {rec:.4f}")
        print(f"🎯 F1-Score  : {f1:.4f}")

        # Lưu kết quả 4 chỉ số vào biến dict
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "classification_report": report_dict
        }

        # 💾 LƯU MODEL VÀO Ổ CỨNG
        model_path = output_dir / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"💾 Đã lưu model tại: {model_path}")

        # Lưu Report Text
        with open(output_dir / f"{name}_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

    print(f"\n{'='*50}")
    print("🎉 HOÀN TẤT! Đã train xong mô hình")

    print("ĐANG TẠO BÁO CÁO VÀ VẼ BIỂU ĐỒ CHO TỪNG MODEL...")

    # Danh sách tên các model đã train trong biến `models`
    model_names = list(models.keys())

    # 2. Tạo 4 biến path riêng biệt để tự do tuỳ chỉnh đường dẫn
    path_rf  = cic_ddos_out_dir + "output_cic_ddos_2018/stage1_output/RandomForest_GPU_model.joblib"
    path_xgb = cic_ddos_out_dir + "output_cic_ddos_2018/stage1_output/XGBoost_GPU_model.joblib"
    path_lgb = cic_ddos_out_dir + "output_cic_ddos_2018/stage1_output/LightGBM_CPU_model.joblib"
    path_cat = cic_ddos_out_dir + "output_cic_ddos_2018/stage1_output/CatBoost_GPU_model.joblib"

    # Gom lại thành dict để khớp với vòng lặp xử lý phía dưới
    model_paths = {
        "RandomForest_GPU": path_rf,
        "XGBoost_GPU": path_xgb,
        "LightGBM_CPU": path_lgb,
        "CatBoost_GPU": path_cat
    }

    # 3. Tạo 4 biến output_folder 
    output_folders = {
        name: f"{cic_ddos_out_dir}output_stage1_{name}/" for name in model_names
    }

    # Đảm bảo các thư mục output này tồn tại
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    # 4. Xác định số lượng class để xử lý ROC Curve (Nhị phân hay Đa lớp)
    classes = np.unique(y_test)
    n_classes = len(classes)
    is_multiclass = n_classes > 2

    if is_multiclass:
        y_test_binarized = label_binarize(y_test, classes=classes)

    # 5. Xử lý từng Model dựa trên đường dẫn đã khai báo
    for name, path in model_paths.items():
        print(f"\n[{name}] Đang tải mô hình...")

        # BƯỚC QUAN TRỌNG: Load model từ file .joblib
        try:
            model = joblib.load(path)
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file model tại {path}. Vui lòng kiểm tra lại đường dẫn! Bỏ qua model này.")
            continue

        print(f"[{name}] Đang dự đoán và đánh giá...")
        out_dir = output_folders[name]

        # Dự đoán (Predict & Predict_Proba)
        y_pred = model.predict(X_test)

        # Một số model không hỗ trợ predict_proba
        try:
            y_pred_proba = model.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None
            print(f"⚠️ {name} không hỗ trợ predict_proba, sẽ bỏ qua ROC Curve.")

        # --- LƯU METRICS (.TXT & .JSON) ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, zero_division=0)

        # Lưu .txt
        with open(os.path.join(out_dir, f"metrics_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"--- ĐÁNH GIÁ MÔ HÌNH: {name} ---\n")
            f.write(f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}\nF1-Score : {f1:.4f}\n\n")
            f.write("--- CLASSIFICATION REPORT ---\n")
            f.write(report_text)

        # Lưu .json
        json_data = {
            "model": name,
            "overall_metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1},
            "classification_report": report_dict
        }
        with open(os.path.join(out_dir, f"metrics_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        # --- VẼ BIỂU ĐỒ CONFUSION MATRIX ---
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=300)
        plt.close()

        # --- VẼ BIỂU ĐỒ ROC CURVE ---
        if y_pred_proba is not None:
            plt.figure(figsize=(10, 8))

            if not is_multiclass:
                # ROC cho nhị phân
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            else:
                # ROC cho đa lớp
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Tính Macro-average ROC
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                mean_tpr /= n_classes
                roc_auc_macro = auc(all_fpr, mean_tpr)

                plt.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
                        label=f'Macro-average ROC (area = {roc_auc_macro:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"roc_curve_{name}.png"), dpi=300)
            plt.close()

        print(f"✅ Đã hoàn thành xử lý cho {name}!")



# ==============================================================================
# STAGE 2: FEATURE SELECTION & SMOTE
# ==============================================================================
def stage2_feature_selection():
    print("\n--- BẮT ĐẦU GIAI ĐOẠN 2: FEATURE SELECTION VÀ SMOTE-ENN ---")
    print("📥 ĐANG TẢI DỮ LIỆU TỪ STAGE 1...")
    
    try:
        X_train_tensor = torch.load(os.path.join(cic_ddos_out_dir, "X_train.pt"))
        y_train_tensor = torch.load(os.path.join(cic_ddos_out_dir, "y_train.pt"))
    except Exception as e:
        print("❌ Không thể load dữ liệu:", e)
        return

    feature_columns = [f"feature_{i}" for i in range(X_train_tensor.shape[1])]
    X_train = pd.DataFrame(X_train_tensor.cpu().numpy(), columns=feature_columns)
    y_train = pd.Series(y_train_tensor.cpu().numpy().squeeze())

    print("🔍 ĐANG DÒ TÌM THRESHOLD CHO 20 FEATURES...")
    rf_finder = RandomForestRegressor(n_estimators=30, max_depth=10, max_samples=0.1, random_state=42, n_jobs=-1)
    rf_finder.fit(X_train, y_train)

    importances = rf_finder.feature_importances_
    sorted_importances = np.sort(importances)[::-1]

    target_features = 20
    magic_threshold = sorted_importances[target_features - 1]

    print("\n" + "="*60)
    print(f"🎯 CON SỐ THRESHOLD ĐỂ LẤY ĐÚNG {target_features} FEATURES LÀ: {magic_threshold}")
    print("="*60 + "\n")

    print("--- BẮT ĐẦU GIAI ĐOẠN 2 (PYTORCH TENSORS) ---")

    def load_stage1_tensors() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load dữ liệu từ các file .pt của Stage 1 và chuyển sang Pandas để xử lý."""
        try:
            X_train_tensor = torch.load(cic_ddos_out_dir + "/X_train.pt")
            X_test_tensor = torch.load(cic_ddos_out_dir + "/X_test.pt")
            y_train_tensor = torch.load(cic_ddos_out_dir + "/y_train.pt")
            y_test_tensor = torch.load(cic_ddos_out_dir + "/y_test.pt")

            X_train_np = X_train_tensor.cpu().numpy()
            X_test_np = X_test_tensor.cpu().numpy()
            y_train_np = y_train_tensor.cpu().numpy().squeeze()
            y_test_np = y_test_tensor.cpu().numpy().squeeze()

            feature_columns = [f"feature_{i}" for i in range(X_train_np.shape[1])]

            X_train_local = pd.DataFrame(X_train_np, columns=feature_columns)
            X_test_local = pd.DataFrame(X_test_np, columns=feature_columns)
            y_train_local = pd.Series(y_train_np)
            y_test_local = pd.Series(y_test_np)

            print("Đã load thành công các file .pt từ Stage 1.")
            return X_train_local, X_test_local, y_train_local, y_test_local

        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file .pt. Vui lòng kiểm tra lại đường dẫn! Chi tiết: {e}")
            raise

    # 1. Load dữ liệu
    X_train, X_test, y_train, y_test = load_stage1_tensors()

    # --- KHỞI TẠO ĐƯỜNG DẪN CHECKPOINT ---
    checkpoint_path = os.path.join(cic_ddos_out_dir, "stage2_rf_checkpoint.pt")

    # 2. Feature Selection (Có cơ chế Checkpoint)
    if os.path.exists(checkpoint_path):
        print("\n✅ Tìm thấy Checkpoint Feature Selection! Đang tải dữ liệu...")
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        X_train_selected = checkpoint_data['X_train_selected']
        X_test_selected = checkpoint_data['X_test_selected']
        important_features = checkpoint_data['important_features']
        print(f"Đã tải {len(important_features)} features từ checkpoint.")
    else:
        print("\nĐang chạy Feature Selection...")
        rf_regressor = RandomForestRegressor(
            n_estimators=30,
            max_depth=10,
            max_samples=0.1,
            random_state=42,
            n_jobs=-1
        )

        # Huấn luyện mô hình để lấy độ quan trọng (Đã bổ sung dòng fit bị thiếu)
        rf_regressor.fit(X_train, y_train)

        # Lọc các features
        # threshold = 0.015225
        threshold = magic_threshold
        important_features = X_train.columns[rf_regressor.feature_importances_ >= threshold]
        print(f"Số lượng features giữ lại: {len(important_features)} / {len(X_train.columns)}")

        X_train_selected = X_train[important_features]
        X_test_selected = X_test[important_features]

        # --- LƯU CHECKPOINT ---
        print("Đang lưu Checkpoint cho Feature Selection...")
        torch.save({
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'important_features': important_features
        }, checkpoint_path)
        print("✅ Đã lưu Checkpoint thành công!")
    

    # 3. Xử lý mất cân bằng dữ liệu với SMOTE-ENN
    print("\n--- CHUẨN BỊ SMOTE-ENN ---")

    # Đếm số lượng mẫu của class thiểu số (hiếm) nhất
    min_samples = y_train.value_counts().min()
    print(f"Nhãn tấn công hiếm nhất hiện đang có: {min_samples} mẫu.")

    # Tự động điều chỉnh số hàng xóm (k_neighbors) sao cho không vượt quá số mẫu hiện có
    # Trừ 1 vì không tính chính nó, tối đa vẫn giữ là 5 như mặc định của SMOTE
    k_neighbors_dynamic = min(5, min_samples - 1)

    if k_neighbors_dynamic < 1:
        print("⚠️ CẢNH BÁO CRITICAL: Có class chỉ có 1 mẫu, SMOTE không thể nội suy!")
        print("Bạn cần quay lại Stage 0 và tăng frac=0.05 (5%) hoặc không dùng sample() cho các class hiếm.")
    else:
        print(f"Đang chạy SMOTE-ENN với k_neighbors = {k_neighbors_dynamic} (Quá trình này có thể tốn nhiều thời gian)...")

        # Khởi tạo SMOTE với thông số tùy chỉnh
        custom_smote = SMOTE(random_state=42, k_neighbors=k_neighbors_dynamic)

        # Truyền SMOTE tùy chỉnh vào SMOTEENN
        smote_enn = SMOTEENN(smote=custom_smote, random_state=42, n_jobs=-1)

        X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_selected, y_train)

        print(f"Kích thước tập train ban đầu: {X_train_selected.shape}")
        print(f"Kích thước tập train sau cân bằng (SMOTE-ENN): {X_train_resampled.shape}")

    # 4. Chuyển đổi dữ liệu trở lại thành PyTorch Tensors và lưu file Stage 2
    print("\nĐang lưu kết quả Stage 2 thành các file .pt...")
    save_dir = cic_ddos_out_dir

    X_train_stage2_tensor = torch.tensor(X_train_resampled.values, dtype=torch.float32)
    y_train_stage2_tensor = torch.tensor(y_train_resampled.values, dtype=torch.long)

    X_test_stage2_tensor = torch.tensor(X_test_selected.values, dtype=torch.float32)
    y_test_stage2_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Lưu thành file .pt
    torch.save(X_train_stage2_tensor, os.path.join(save_dir, "stage2_X_train.pt"))
    torch.save(y_train_stage2_tensor, os.path.join(save_dir, "stage2_y_train.pt"))
    torch.save(X_test_stage2_tensor, os.path.join(save_dir, "stage2_X_test.pt"))
    torch.save(y_test_stage2_tensor, os.path.join(save_dir, "stage2_y_test.pt"))
    torch.save(list(important_features), os.path.join(save_dir, "stage2_important_features.pt"))

    print("🎉 Hoàn thành Giai đoạn 2! Các file stage2_*.pt đã được lưu an toàn.")
    print("✅ Đã lưu Checkpoint thành công cho tập dữ liệu cân bằng ở Stage 2!")

# ==============================================================================
# STAGE 3: HYPEROPT
# ==============================================================================
def stage3_hyperopt():
    print("\n--- BẮT ĐẦU GIAI ĐOẠN 3: HYPEROPT CHO RANDOM FOREST ---")
    base_dir_path = Path(base_dir)
    
    stage2_dir = Path(cic_ddos_out_dir)
    out_dir = Path(stage3_out_dir)
    hyperopt_checkpoint_path = Path(stage3_checkpoint_dir) / 'stage3_hyperopt_checkpoint.joblib'
    hyperopt_best_params_path = Path(stage3_checkpoint_dir) / 'stage3_hyperopt_best_params.joblib'

    print("Đang tải dữ liệu từ Stage 2...")
    try:
        X_train_tensor = torch.load(stage2_dir / "stage2_X_train.pt")
        y_train_tensor = torch.load(stage2_dir / "stage2_y_train.pt")
        X_test_tensor = torch.load(stage2_dir / "stage2_X_test.pt")
        y_test_tensor = torch.load(stage2_dir / "stage2_y_test.pt")
        important_features = torch.load(stage2_dir / "stage2_important_features.pt")
        
        X_train_resampled = pd.DataFrame(X_train_tensor.numpy(), columns=important_features)
        y_train_resampled = pd.Series(y_train_tensor.numpy())
        X_test_selected = pd.DataFrame(X_test_tensor.numpy(), columns=important_features)
        y_test = pd.Series(y_test_tensor.numpy())
    except:
        print("⚠️ Không tìm thấy file tensor stage 2.")
        return

    # Định nghĩa Search Space cho Hyperopt
    space = {
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        'max_features': hp.quniform('max_features', 1, 29, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
        'n_estimators': hp.quniform('n_estimators', 10, 200, 1)
    }

    print("--- BẮT ĐẦU GIAI ĐOẠN 3 (HYPEROPT) ---")
    HYPEROPT_MAX_EVALS = 25
    HYPEROPT_EVAL_CHUNK_SIZE = 1
    # HYPEROPT_DATA_CHUNK_SIZE = 1000000
    HYPEROPT_DATA_CHUNK_SIZE = 500000
    HYPEROPT_CHECKPOINT_PATH = base_dir_path / 'checkpoint_rf_stage3' / 'stage3_hyperopt_checkpoint.joblib'
    HYPEROPT_BEST_PARAMS_PATH = base_dir_path / 'checkpoint_rf_stage3'/ 'stage3_hyperopt_best_params.joblib'


    def _slice_chunk(data, start, end):
        if hasattr(data, 'iloc'):
            return data.iloc[start:end]
        return data[start:end]


    def _build_chunk_ranges(total_size, chunk_size):
        if total_size <= 0:
            return []
        return [(i, min(i + chunk_size, total_size)) for i in range(0, total_size, chunk_size)]


    def _make_objective(X_chunk, y_chunk):
        def objective(params):
            model = RandomForestClassifier(
                criterion=params['criterion'],
                max_depth=int(params['max_depth']),
                max_features=int(params['max_features']),
                min_samples_leaf=int(params['min_samples_leaf']),
                min_samples_split=int(params['min_samples_split']),
                n_estimators=int(params['n_estimators']),
                random_state=106
            )

            # Hyperopt tối thiểu hóa loss nên dùng âm của f1_macro.
            score = cross_val_score(model, X_chunk, y_chunk, scoring='f1_macro', cv=3).mean()
            return {'loss': -score, 'status': STATUS_OK}

        return objective


    hyperopt_total_samples = len(X_train_resampled)
    hyperopt_data_ranges = _build_chunk_ranges(hyperopt_total_samples, HYPEROPT_DATA_CHUNK_SIZE)

    if not hyperopt_data_ranges:
        raise ValueError('X_train_resampled rỗng, không thể chạy Hyperopt.')

    checkpoint_dir = base_dir_path / 'checkpoint_rf_stage3'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Đã chuẩn bị thư mục checkpoint tại: {checkpoint_dir}")

    trials = Trials()
    best = None
    completed_evals = 0
    next_data_chunk_idx = 0

    if os.path.exists(HYPEROPT_CHECKPOINT_PATH):
        hyperopt_checkpoint = joblib.load(HYPEROPT_CHECKPOINT_PATH)
        saved_trials = hyperopt_checkpoint.get('trials')
        if isinstance(saved_trials, Trials):
            trials = saved_trials
        best = hyperopt_checkpoint.get('best')
        completed_evals = int(hyperopt_checkpoint.get('completed_evals', len(trials.trials)))
        next_data_chunk_idx = int(hyperopt_checkpoint.get('next_data_chunk_idx', 0))
        print(f"[HYPEROPT] Resume từ {completed_evals}/{HYPEROPT_MAX_EVALS} evaluations.")

    while completed_evals < HYPEROPT_MAX_EVALS:
        range_start, range_end = hyperopt_data_ranges[next_data_chunk_idx]
        X_hyperopt_chunk = _slice_chunk(X_train_resampled, range_start, range_end)
        y_hyperopt_chunk = _slice_chunk(y_train_resampled, range_start, range_end)

        target_evals = min(completed_evals + HYPEROPT_EVAL_CHUNK_SIZE, HYPEROPT_MAX_EVALS)
        best = fmin(
            fn=_make_objective(X_hyperopt_chunk, y_hyperopt_chunk),
            space=space,
            algo=tpe.suggest,
            max_evals=target_evals,
            trials=trials
        )

        completed_evals = len(trials.trials)
        next_data_chunk_idx = (next_data_chunk_idx + 1) % len(hyperopt_data_ranges)

        hyperopt_checkpoint = {
            'trials': trials,
            'best': best,
            'completed_evals': completed_evals,
            'next_data_chunk_idx': next_data_chunk_idx,
            'max_evals': HYPEROPT_MAX_EVALS,
            'eval_chunk_size': HYPEROPT_EVAL_CHUNK_SIZE,
            'data_chunk_size': HYPEROPT_DATA_CHUNK_SIZE
        }
        joblib.dump(hyperopt_checkpoint, HYPEROPT_CHECKPOINT_PATH)
        print(
            f"[HYPEROPT CHECKPOINT] Đã lưu tại eval {completed_evals}/{HYPEROPT_MAX_EVALS} "
            f"(data chunk {range_start}:{range_end})."
        )

    joblib.dump(best, HYPEROPT_BEST_PARAMS_PATH)
    print(f"Đã lưu best params từ Hyperopt tại: {HYPEROPT_BEST_PARAMS_PATH}")

    print(f"Bộ tham số tối ưu tìm được: {best}")

    if best is None:
        raise RuntimeError('Không tìm được bộ tham số Hyperopt hợp lệ.')
    
    # Đường dẫn file lưu tham số tốt nhất
    rf_params_file = base_dir_path / "best_params_rf_stage3.json"

    # Cờ (flag) để theo dõi xem có cần ghi file mới không
    need_to_save = True

    if rf_params_file.exists():
        try:
            print(f" Đã tìm thấy file tham số tại: {rf_params_file}")
            with open(rf_params_file, "r") as f:
                best_rf = json.load(f)
            print(" Đã tải xong bộ tham số tối ưu:", best_rf)
            need_to_save = False # Đọc thành công thì không cần ghi lại
        except json.JSONDecodeError:
            print(" File JSON cũ bị lỗi dở dang. Đang chuẩn bị ghi đè file mới...")
            need_to_save = True

    # Nếu file chưa có hoặc file cũ bị lỗi thì tiến hành ép kiểu và lưu lại
    if need_to_save:
        # Ép kiểu dữ liệu từ Numpy về Python chuẩn
        best_python_types = {}
        for key, value in best.items():
            if isinstance(value, np.integer):
                best_python_types[key] = int(value)
            elif isinstance(value, np.floating):
                best_python_types[key] = float(value)
            else:
                best_python_types[key] = value

        # LƯU LẠI THÀNH FILE JSON ĐỂ LẦN SAU DÙNG
        with open(rf_params_file, "w") as f:
            json.dump(best_python_types, f, indent=4)
        print(f" Đã lưu thành công bộ tham số tối ưu MỚI vào: {rf_params_file}")

    best_rf = joblib.load(base_dir_path / "checkpoint_rf_stage3" / "stage3_hyperopt_best_params.joblib")

    
    print("--- 1. HUẤN LUYỆN VÀ ĐÁNH GIÁ RANDOM FOREST ---")

    # Tạo thư mục lưu kết quả Stage 3
    out_dir = base_dir_path / "output_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Giả định đã có best_rf từ Hyperopt. Nếu chưa, đây là ví dụ để code không báo lỗi:
    # best_rf = {'criterion': 0, 'max_depth': 30, 'max_features': 5, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}

    best_criterion = 'gini' if best_rf['criterion'] == 0 else 'entropy'
    final_rf = RandomForestClassifier(
        criterion=best_criterion,
        max_depth=int(best_rf['max_depth']),
        max_features=int(best_rf['max_features']),
        min_samples_leaf=int(best_rf['min_samples_leaf']),
        min_samples_split=int(best_rf['min_samples_split']),
        n_estimators=int(best_rf['n_estimators']),
        random_state=106,
        n_jobs=-1 # RF không chạy được GPU, dùng CPU đa luồng
    )

    print("Đang huấn luyện Random Forest...")
    final_rf.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = final_rf.predict(X_test_selected)

    # Tạo report
    report_text_rf = classification_report(y_test, y_pred_rf, digits=4)
    report_dict_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    print("\nRandom Forest Performance:")
    print(report_text_rf)

    # Lưu file báo cáo
    (out_dir / "rf_report.txt").write_text(report_text_rf, encoding="utf-8")
    with open(out_dir / "rf_report.json", "w") as f:
        json.dump(report_dict_rf, f, indent=2)
    pd.DataFrame(report_dict_rf).transpose().to_csv(out_dir / "rf_report.csv")

    # Lưu mô hình
    joblib.dump(final_rf, out_dir / "stage3_rf_model.joblib")
    print("Đã lưu mô hình và báo cáo Random Forest thành công!")


    print("--- BẮT ĐẦU GIAI ĐOẠN 3: HYPEROPT CHO XGBOOST (GPU) ---")

    # 1. Cấu hình đường dẫn
    out_dir = base_dir_path / "output_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    xgb_params_file = out_dir / "best_params_xgb.json"

    # 2. Load dữ liệu từ Stage 2
    # stage2_data = joblib.load(base_dir / "stage2_processed.pt")
    # X_train_resampled = stage2_data["X_train_resampled"]
    # y_train_resampled = stage2_data["y_train_resampled"]

    stage2_dir = base_dir_path / "cic_ddos_2018"

    X_train_tensor = torch.load(stage2_dir / "stage2_X_train.pt")
    y_train_tensor = torch.load(stage2_dir / "stage2_y_train.pt")
    important_features = torch.load(stage2_dir / "stage2_important_features.pt")

    X_train_resampled = pd.DataFrame(X_train_tensor.numpy(), columns=important_features)
    y_train_resampled = pd.Series(y_train_tensor.numpy())

    # --- CHIẾN THUẬT TIẾT KIỆM THỜI GIAN ---
    # Lấy mẫu 1,000,000 dòng ngẫu nhiên để tìm tham số (vẫn đủ lớn để đại diện)
    # Nếu muốn chạy trên toàn bộ 25 triệu dòng, hãy comment 3 dòng dưới đây.
    # sample_size = 1000000
    # idx = np.random.choice(len(X_train_resampled), sample_size, replace=False)
    # X_train_tuning = X_train_resampled.iloc[idx]
    # y_train_tuning = y_train_resampled.iloc[idx]
    # print(f"Đang sử dụng {sample_size} mẫu để tối ưu hóa tham số...")

    X_train_tuning = X_train_resampled
    y_train_tuning = y_train_resampled

    # 3. Kiểm tra nếu đã có file tham số thì load lên, không thì mới chạy fmin
    if xgb_params_file.exists():
        print(f"✅ Đã tìm thấy tham số XGBoost tại: {xgb_params_file}")
        with open(xgb_params_file, "r") as f:
            best_xgb = json.load(f)
        print("Thông số đã load:", best_xgb)
    else:
        print("🚀 Bắt đầu chạy Hyperopt trên GPU...")

        # Định nghĩa không gian tìm kiếm dành riêng cho XGBoost
        space_xgb = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'gamma': hp.uniform('gamma', 0, 5),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
        }

        def objective_xgb(params):
            model = XGBClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                min_child_weight=int(params['min_child_weight']),
                random_state=106,
                eval_metric='mlogloss',
                # device='cpu',
                # # TỐI ƯU GPU
                tree_method='hist',
                device='cuda',
                n_jobs=-1
            )

            # Cross-validation với 3-fold
            score = cross_val_score(model, X_train_tuning, y_train_tuning, scoring='f1_macro', cv=3).mean()
            return {'loss': -score, 'status': STATUS_OK}

        trials_xgb = Trials()
        best_xgb = fmin(
            fn=objective_xgb,
            space=space_xgb,
            algo=tpe.suggest,
            max_evals=20, # có thể tăng lên 50 nếu có thời gian
            trials=trials_xgb
        )

    print("--- 2. HUẤN LUYỆN VÀ ĐÁNH GIÁ XGBOOST (GPU) ---")
    # Giả định best_xgb từ Hyperopt:
    # best_xgb = {'n_estimators': 150, 'max_depth': 15, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1}
    # nhớ chạy lại cell load best_rf

    final_xgb = XGBClassifier(
        n_estimators=int(best_xgb['n_estimators']),
        max_depth=int(best_xgb['max_depth']),
        learning_rate=best_xgb['learning_rate'],
        subsample=best_xgb['subsample'],
        colsample_bytree=best_xgb['colsample_bytree'],
        min_child_weight=int(best_xgb['min_child_weight']),
        random_state=106,
        # device='cpu',
        tree_method='hist', # Ép dùng GPU
        device='cuda',      # Ép dùng GPU
        eval_metric='mlogloss'
    )

    print("Đang huấn luyện XGBoost trên GPU...")
    final_xgb.fit(X_train_resampled, y_train_resampled)
    y_pred_xgb = final_xgb.predict(X_test_selected)

    # Tạo report
    report_text_xgb = classification_report(y_test, y_pred_xgb, digits=4)
    report_dict_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
    print("\nXGBoost Performance:")
    print(report_text_xgb)

    # Lưu file báo cáo
    (out_dir / "xgb_report.txt").write_text(report_text_xgb, encoding="utf-8")
    with open(out_dir / "xgb_report.json", "w") as f:
        json.dump(report_dict_xgb, f, indent=2)
    pd.DataFrame(report_dict_xgb).transpose().to_csv(out_dir / "xgb_report.csv")

    # Lưu mô hình
    joblib.dump(final_xgb, out_dir / "stage3_xgb_model.joblib")
    print("Đã lưu mô hình và báo cáo XGBoost thành công!")

    print("--- 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ CATBOOST (GPU) ---")

    out_dir = base_dir_path / "output_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_params_file = out_dir / "best_params_cat.json"

    if cat_params_file.exists():
        print(f"✅ Đã tìm thấy tham số CatBoost tại: {cat_params_file}")
        with open(cat_params_file, "r") as f:
            best_cat = json.load(f)
    else:
        print("🚀 Bắt đầu chạy Hyperopt cho CatBoost trên GPU...")
        space_cat = {
            'iterations': hp.quniform('iterations', 50, 500, 10),
            'depth': hp.quniform('depth', 4, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10)
        }

        def objective_cat(params):
            model = CatBoostClassifier(
                iterations=int(params['iterations']),
                depth=int(params['depth']),
                learning_rate=params['learning_rate'],
                l2_leaf_reg=params['l2_leaf_reg'],
                random_seed=106,
                # # task_type='GPU',
                # task_type='CPU',   # SỬA Ở ĐÂY: Dùng CPU lúc dò tham số để tránh lỗi State == nullptr
                # # thread_count=-1,   # Tận dụng tối đa CPU đa luồng của Colab
                task_type='GPU',
                devices='0',          # GPU đầu tiên
                border_count=32,      # QUAN TRỌNG: giảm xuống để tránh tràn VRAM
                gpu_ram_part=0.85,    # Chỉ dùng 85% VRAM
                verbose=False
            )
            score = cross_val_score(model, X_train_tuning, y_train_tuning, scoring='f1_macro', cv=3).mean()
            return {'loss': -score, 'status': STATUS_OK}

        trials_cat = Trials()
        best_cat_raw = fmin(fn=objective_cat, space=space_cat, algo=tpe.suggest, max_evals=20, trials=trials_cat)

        # Xử lý ép kiểu JSON giống đợt trước để không bị lỗi TypeError
        best_cat = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in best_cat_raw.items()}
        with open(cat_params_file, "w") as f:
            json.dump(best_cat, f, indent=4)

    final_cat = CatBoostClassifier(
        iterations=int(best_cat['iterations']),
        depth=int(best_cat['depth']),
        learning_rate=best_cat['learning_rate'],
        l2_leaf_reg=best_cat['l2_leaf_reg'],
        random_seed=106,
        # # task_type='GPU', # Ép dùng GPU rất mượt
        # task_type='CPU',   # SỬA Ở ĐÂY: Dùng CPU lúc dò tham số để tránh lỗi State == nullptr
        # # thread_count=-1,   # Tận dụng tối đa CPU đa luồng của Colab
        task_type='GPU',
        devices='0',
        border_count=32,
        gpu_ram_part=0.85,
        verbose=False
    )

    print("Đang huấn luyện CatBoost trên GPU...")
    final_cat.fit(X_train_resampled, y_train_resampled)
    y_pred_cat = final_cat.predict(X_test_selected)

    # Tạo report
    report_text_cat = classification_report(y_test, y_pred_cat, digits=4)
    report_dict_cat = classification_report(y_test, y_pred_cat, output_dict=True)
    print("\nCatBoost Performance:")
    print(report_text_cat)

    # Lưu file báo cáo
    (out_dir / "cat_report.txt").write_text(report_text_cat, encoding="utf-8")
    with open(out_dir / "cat_report.json", "w") as f:
        json.dump(report_dict_cat, f, indent=2)
    pd.DataFrame(report_dict_cat).transpose().to_csv(out_dir / "cat_report.csv")

    # Lưu mô hình
    joblib.dump(final_cat, out_dir / "stage3_cat_model.joblib")
    print("Đã lưu mô hình và báo cáo CatBoost thành công!")

    print("--- 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ LIGHTGBM (GPU) ---")

    out_dir = base_dir_path / "output_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    lgb_params_file = out_dir / "best_params_lgb.json"

    if lgb_params_file.exists():
        print(f" Đã tìm thấy tham số LightGBM tại: {lgb_params_file}")
        with open(lgb_params_file, "r") as f:
            best_lgb = json.load(f)
    else:
        print(" Bắt đầu chạy Hyperopt cho LightGBM trên GPU...")
        space_lgb = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }

        def objective_lgb(params):
            model = LGBMClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                num_leaves=int(params['num_leaves']),
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=106,
                # device='gpu', # Ép dùng GPU khi dò tham số
                device='cpu',
                verbose=-1,
                n_jobs=-1
            )
            score = cross_val_score(model, X_train_tuning, y_train_tuning, scoring='f1_macro', cv=3).mean()
            return {'loss': -score, 'status': STATUS_OK}

        trials_lgb = Trials()
        best_lgb_raw = fmin(fn=objective_lgb, space=space_lgb, algo=tpe.suggest, max_evals=20, trials=trials_lgb)

        # Xử lý ép kiểu JSON để tránh TypeError
        best_lgb = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in best_lgb_raw.items()}
        with open(lgb_params_file, "w") as f:
            json.dump(best_lgb, f, indent=4)

    final_lgb = LGBMClassifier(
        n_estimators=int(best_lgb['n_estimators']),
        max_depth=int(best_lgb['max_depth']),
        num_leaves=int(best_lgb['num_leaves']),
        learning_rate=best_lgb['learning_rate'],
        subsample=best_lgb['subsample'],
        colsample_bytree=best_lgb['colsample_bytree'],
        random_state=106,
        # device='gpu', # Ép dùng GPU
        device='cpu',
        verbose=-1
    )

    print("Đang huấn luyện LightGBM trên GPU...")
    final_lgb.fit(X_train_resampled, y_train_resampled)
    y_pred_lgb = final_lgb.predict(X_test_selected)

    # Tạo report
    report_text_lgb = classification_report(y_test, y_pred_lgb, digits=4)
    report_dict_lgb = classification_report(y_test, y_pred_lgb, output_dict=True)
    print("\nLightGBM Performance:")
    print(report_text_lgb)

    # Lưu file báo cáo
    (out_dir / "lgb_report.txt").write_text(report_text_lgb, encoding="utf-8")
    with open(out_dir / "lgb_report.json", "w") as f:
        json.dump(report_dict_lgb, f, indent=2)
    pd.DataFrame(report_dict_lgb).transpose().to_csv(out_dir / "lgb_report.csv")

    # Lưu mô hình
    joblib.dump(final_lgb, out_dir / "stage3_lgb_model.joblib")
    print("Đã lưu mô hình và báo cáo LightGBM thành công!")


        
# ==============================================================================
# STAGE 4: EXPLAINABLE AI (XAI)
# ==============================================================================
def stage4_xai():
    print("\n--- BẮT ĐẦU GIAI ĐOẠN 4: XAI VỚI SHAP ---")
    base_dir_path = Path(base_dir)
    if importlib.util.find_spec("IPython") is not None:
        try:
            shap.initjs()
        except Exception as e:
            print(
                f"⚠️ Không thể khởi tạo SHAP JS trong môi trường hiện tại, tiếp tục chạy bình thường. Chi tiết: {e}"
            )
    else:
        print(
            "ℹ️ Không có IPython, bỏ qua shap.initjs() vì bạn đang chạy script terminal."
        )
    out_dir = Path(stage4_out_dir)
    print("Đã cấu hình xong thư mục lưu kết quả XAI tại:", out_dir)

    print("Đang tải dữ liệu Test và Mô hình...")

    # Load dữ liệu test từ file .pt đã lưu ở Giai đoạn 2
    try:
        X_test_tensor = torch.load(base_dir_path / "cic_ddos_2018" / "stage2_X_test.pt")
        y_test_tensor = torch.load(base_dir_path / "cic_ddos_2018" / "stage2_y_test.pt")
        important_features = torch.load(
            base_dir_path / "cic_ddos_2018" / "stage2_important_features.pt"
        )

        # Chuyển về Pandas DataFrame để SHAP có thể hiển thị tên các tính năng (features)
        X_test_selected = pd.DataFrame(
            X_test_tensor.numpy(), columns=important_features
        )
        y_test = y_test_tensor.numpy()
        print("✅ Tải dữ liệu test thành công!")
    except Exception as e:
        print("❌ Lỗi khi tải dữ liệu:", e)
        return

    import os

    print("Đang khôi phục tên Feature gốc từ tất cả các file CSV...")

    # 1. Quét tất cả các file CSV
    # csv_dir = base_dir_path / "IDS2018/CSV_Data" # Cho linux
    csv_dir = base_dir_path / "IDS2018" / "CSV_Data" # Cho Windows
    # all_files = [
    #     os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")
    # ]

    import glob
    # all_files = glob.glob(str(csv_dir / "**/*.csv"), recursive=True)
    all_files = glob.glob(glob.escape(str(csv_dir)) + "/*.csv")
    print(f"Tìm thấy {len(all_files)} file CSV tại: {csv_dir}")

    def _sanitize_for_filename(text):
        return "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text)
        ).strip("_")

    def _build_class_name_lookup(csv_files):
        label_values = set()
        for csv_file in csv_files:
            try:
                for chunk in pd.read_csv(
                    csv_file, usecols=["Label"], chunksize=250000, low_memory=False
                ):
                    labels = chunk["Label"].astype(str).str.strip()
                    labels = labels[labels != "Label"]
                    label_values.update(labels.unique().tolist())
            except Exception:
                continue

        sorted_labels = sorted(label_values)
        return {idx: label_name for idx, label_name in enumerate(sorted_labels)}

    class_id_to_name = _build_class_name_lookup(all_files)
    print(f"Đã khôi phục mapping {len(class_id_to_name)} class từ dữ liệu gốc.")

    # 2. Đọc nhanh header (nrows=0) của TẤT CẢ các file rồi gộp lại để lấy danh sách cột tổng hợp
    df_list_empty = []
    for f in all_files:
        df_empty = pd.read_csv(f, nrows=0, low_memory=False)
        df_list_empty.append(df_empty)

    df_full_empty = pd.concat(df_list_empty, ignore_index=True)
    df_full_empty.columns = df_full_empty.columns.str.strip()

    # 3. Loại bỏ 'Timestamp' và 'Label' y hệt như Stage 0
    cols_to_drop = ["Timestamp", "Label"]
    original_cols = [c for c in df_full_empty.columns if c not in cols_to_drop]

    # 4. Ánh xạ ngược lại tên thật
    real_feature_names = []
    for f in important_features:
        idx = int(f.split("_")[1])

        # Kẹp thêm điều kiện an toàn để code không bao giờ crash nữa
        if idx < len(original_cols):
            real_feature_names.append(original_cols[idx])
        else:
            real_feature_names.append(f"Extra_Feature_{idx}")

    # 5. Gán lại tên thật cho DataFrame SHAP
    X_test_selected.columns = real_feature_names

    print(
        f"✅ Đã khôi phục thành công tên thật cho {len(real_feature_names)} features!"
    )
    print("Ví dụ vài features:", real_feature_names[:5])

    print("--- KHỞI CHẠY HỆ THỐNG PHÂN TÍCH XAI TỰ ĐỘNG ---")

    # 1. Danh sách các model đã train ở Stage 3
    model_files = [
        "stage3_rf_model.joblib",
        "stage3_xgb_model.joblib",
        "stage3_lgb_model.joblib",
        "stage3_cat_model.joblib",
    ]

    # 2. Lấy mẫu 20000 dòng theo stratified để đảm bảo mọi class đều có đại diện
    from sklearn.model_selection import train_test_split
    sample_size = min(20000, len(X_test_selected))
    _, X_test_sample, _, sample_idx = train_test_split(
        X_test_selected,
        np.arange(len(y_test)),
        test_size=sample_size,
        stratify=y_test,
        random_state=42
    )
    y_test_sample = y_test[sample_idx]

    # Lấy danh sách tất cả các Class hiện có trong dữ liệu Test
    unique_classes = np.unique(y_test)
    print(f"Tổng số class cần phân tích: {len(unique_classes)}")

    # 3. VÒNG LẶP CHÍNH: Duyệt qua từng mô hình
    for model_file in model_files:
        model_path = base_dir_path / "output_stage3" / model_file
        # Lấy tên ngắn gọn của model (rf, xgb, lgb, cat)
        model_name = model_file.split("_model")[0].replace("stage3_", "")

        # if model_name in ("lgb", "xgb"):  # Tạm thời bỏ qua
        #     print(f"\n⏭️ Bỏ qua {model_name.upper()} tạm thời.")
        #     continue
        
        if not model_path.exists():
            print(
                f"\n⚠️ Bỏ qua {model_name.upper()} vì không tìm thấy file {model_file}"
            )
            continue

        print(f"\n{'='*60}")
        print(f"🚀 ĐANG PHÂN TÍCH XAI CHO MÔ HÌNH: {model_name.upper()}")

        # Tạo thư mục riêng cho model này để lưu hình cho gọn
        model_out_dir = out_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        best_model = joblib.load(model_path)

        # Đưa về CPU nếu có hỗ trợ để SHAP không bị lỗi CUDA
        if hasattr(best_model, "set_params"):
            try:
                best_model.set_params(device="cpu")
            except:
                pass  # Bỏ qua nếu model (như Random Forest) không có tham số device

        # Khởi tạo Explainer
        print(f"⏳ Đang tính toán SHAP values cho {model_name.upper()}...")

        if model_name == "rf":
            from joblib import Parallel, delayed

            total_cores = os.cpu_count() or 1
            if total_cores <= 2:
                n_jobs = 1
            elif total_cores <= 8:
                n_jobs = total_cores - 1
            else:
                # n_jobs = max(total_cores - 2, 8)
                n_jobs = 2

            # Linux server: ưu tiên threads để giảm overhead copy/pickle model lớn.
            rf_backend_env = os.environ.get("IDPS_SHAP_RF_BACKEND", "").strip().lower()
            if rf_backend_env in {"threads", "processes"}:
                rf_backend = rf_backend_env
            else:
                rf_backend = "threads" if os.name == "posix" else "processes"

            shap_sample_size = min(10000, len(X_test_sample))
            rng = np.random.default_rng(42)
            rf_idx = rng.choice(len(X_test_sample), shap_sample_size, replace=False)
            X_shap_rf = X_test_sample.iloc[rf_idx].reset_index(drop=True)

            # ── CHECKPOINT: Nếu đã tính rồi thì load lại, không tính lại ──
            shap_cache_path = os.path.join(os.getcwd(), "_shap_rf_cache.joblib")

            if os.path.exists(shap_cache_path):
                print(f"✅ Tìm thấy SHAP cache RF, load lại thay vì tính lại...")
                shap_values_obj = joblib.load(shap_cache_path)
            else:
                print(
                    f"⚡ RF+SHAP: {total_cores} cores → dùng {n_jobs} workers "
                    f"({rf_backend}), {shap_sample_size} mẫu..."
                )

                tmp_model_path = os.path.join(os.getcwd(), "_tmp_rf_shap.joblib")
                joblib.dump(best_model, tmp_model_path)
                batches = np.array_split(X_shap_rf, n_jobs)

                def _shap_batch_proc(batch_values, model_path):
                    import shap, joblib

                    m = joblib.load(model_path)
                    exp = shap.TreeExplainer(m)
                    return exp(batch_values)

                results = Parallel(n_jobs=n_jobs, prefer=rf_backend, verbose=10)(
                    # delayed(_shap_batch_proc)(b.values, tmp_model_path) for b in batches
                    delayed(_shap_batch_proc)(b, tmp_model_path) for b in batches
                )

                os.remove(tmp_model_path)

                combined_values = np.concatenate([r.values for r in results], axis=0)
                combined_data = np.concatenate([r.data for r in results], axis=0)
                combined_base = np.concatenate(
                    [np.atleast_1d(r.base_values) for r in results], axis=0
                )

                shap_values_obj = shap.Explanation(
                    values=combined_values,
                    base_values=combined_base,
                    data=combined_data,
                    feature_names=list(X_shap_rf.columns),
                )

                # Lưu cache lại để lần sau không tính lại
                joblib.dump(shap_values_obj, shap_cache_path)
                print(
                    f"✅ Gộp xong {n_jobs} batch SHAP! Đã lưu cache tại {shap_cache_path}"
                )

        else:
            shap_cache_path = os.path.join(
                os.getcwd(), f"_shap_{model_name}_cache.joblib"
            )

            # Tạo index trước để dùng cho cả trường hợp có cache lẫn không
            if model_name == "cat":
                from sklearn.model_selection import train_test_split
                shap_sample_size = min(20000, len(X_test_selected))
                # shap_sample_size = len(X_test_selected)
                _, X_shap_input, _, cat_idx = train_test_split(
                    X_test_selected,
                    np.arange(len(y_test)),
                    test_size=shap_sample_size,
                    stratify=y_test,
                    random_state=42
                )
                X_shap_input = X_shap_input.reset_index(drop=True)
                print(f"⚠️  CAT+SHAP: dùng {shap_sample_size} mẫu...")
            else:
                cat_idx = None
                X_shap_input = X_test_sample 

        if model_name == "cat":
            from catboost import Pool
            print(">>> Dùng CatBoost native SHAP...")
            pool = Pool(X_shap_input.values, feature_names=list(important_features))
            native_shap = np.asarray(
                best_model.get_feature_importance(type="ShapValues", data=pool)
            )

            if native_shap.ndim == 2:
                shap_matrix = native_shap[:, :-1]
                base_values = native_shap[:, -1]
            elif native_shap.ndim == 3:
                shap_matrix = np.transpose(native_shap[:, :, :-1], (0, 2, 1))
                base_values = native_shap[:, :, -1]
            else:
                raise RuntimeError(f"Unsupported native SHAP shape: {native_shap.shape}")

            shap_values_obj = shap.Explanation(
                values=shap_matrix,
                base_values=base_values,
                data=X_shap_input.values,
                feature_names=list(X_shap_input.columns),
            )
            joblib.dump(shap_values_obj, shap_cache_path)
            print(f"✅ Tính xong! Đã lưu cache tại {shap_cache_path}")

        else:
            if os.path.exists(shap_cache_path):
                print(f"✅ Tìm thấy SHAP cache {model_name.upper()}, load lại thay vì tính lại...")
                shap_values_obj = joblib.load(shap_cache_path)
            else:
                print(f"⏳ Tính toán SHAP cho {model_name.upper()}...")
                explainer = shap.TreeExplainer(best_model)
                shap_values_obj = explainer(X_shap_input.values)
                shap_values_obj.feature_names = list(X_shap_input.columns)
                joblib.dump(shap_values_obj, shap_cache_path)
                print(f"✅ Tính xong! Đã lưu cache tại {shap_cache_path}") 

        # Kiểm tra số chiều (Multiclass thường trả về 3 chiều)
        is_multiclass = len(shap_values_obj.shape) == 3

        if model_name == "rf":
            X_plot = X_shap_rf
            y_plot = y_test_sample[rf_idx]
        elif model_name == "cat":
            X_plot = X_shap_input
            y_plot = y_test[cat_idx]
        else:
            X_plot = X_test_sample
            y_plot = y_test_sample

        # --- A. BAR PLOT (Tổng hợp mọi class) ---
        print(f"📊 Đang vẽ Bar Plot tổng hợp...")
        plt.figure(figsize=(12, 8))
        plt.title(f"SHAP Feature Importance ({model_name.upper()})", fontsize=14)

        if is_multiclass:
            mean_shap = shap.Explanation(
                values=np.abs(shap_values_obj.values).mean(axis=(0, 2)),
                feature_names=shap_values_obj.feature_names,
            )
            shap.plots.bar(mean_shap, max_display=15, show=False)
        else:
            shap.plots.bar(shap_values_obj, max_display=15, show=False)

        plt.savefig(
            model_out_dir / f"shap_bar_plot_{model_name}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()  # Giải phóng RAM

        # --- B. VÒNG LẶP CHO TỪNG CLASS (SUMMARY PLOT & WATERFALL) ---
        for cls in unique_classes:
            cls_idx = int(cls)
            cls_name = class_id_to_name.get(cls_idx, f"Class_{cls_idx}")
            cls_name_slug = _sanitize_for_filename(cls_name)
            print(f"   -> Đang xử lý cho Class {cls_idx}: {cls_name}...")

            # 1. SUMMARY DOT PLOT (Toàn cục cho 1 class)
            plt.figure(figsize=(12, 8))
            plt.title(
                f"SHAP Summary Plot - Tầm quan trọng ({cls_name}) - {model_name.upper()}",
                fontsize=14,
            )
            if is_multiclass:
                # shap.summary_plot(shap_values_obj[:, :, cls_idx], X_test_sample, show=False)
                shap.summary_plot(shap_values_obj[:, :, cls_idx], X_plot, show=False)
            else:
                shap.summary_plot(shap_values_obj, X_plot, show=False)
            plt.savefig(
                model_out_dir
                / f"shap_summary_class{cls_idx}_{cls_name_slug}_{model_name}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # 2. WATERFALL PLOT (Giải thích cục bộ cho 1 gói tin đại diện của class này)
            # Tìm tất cả các mẫu thuộc class này trong tập sample 2000 dòng
            indices_of_class = np.where(y_plot == cls_idx)[0]

            if len(indices_of_class) > 0:
                # instance_idx = indices_of_class[0] # Lấy mẫu ĐẦU TIÊN đại diện
                # Lấy mẫu GẦN TRUNG TÂM NHẤT của class này
                class_samples = X_plot.iloc[indices_of_class]
                class_center = class_samples.mean(axis=0)  # Vector trung bình của class
                distances = np.linalg.norm(
                    class_samples.values - class_center.values, axis=1
                )  # Khoảng cách Euclidean
                instance_idx = indices_of_class[
                    np.argmin(distances)
                ]  # Mẫu gần tâm nhất
                predicted_class = int(
                    best_model.predict(X_plot.iloc[[instance_idx]].values).flatten()[0]
                )
                predicted_class_name = class_id_to_name.get(
                    predicted_class, f"Class_{predicted_class}"
                )

                plt.figure(figsize=(10, 6))
                if is_multiclass:
                    shap.plots.waterfall(
                        shap_values_obj[instance_idx, :, predicted_class], show=False
                    )
                else:
                    shap.plots.waterfall(shap_values_obj[instance_idx], show=False)

                plt.title(
                    f"[{model_name.upper()}] Giải thích gói tin {cls_name} "
                    f"(Dự đoán: {predicted_class_name})",
                    fontsize=14,
                )
                plt.savefig(
                    model_out_dir
                    / f"shap_waterfall_class{cls_idx}_{cls_name_slug}_{model_name}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()
            else:
                print(
                    f"      (Không có gói tin nào thuộc class {cls_name} lọt vào tập sample {len(y_plot)} dòng để vẽ Waterfall)"
                )

        print(f"✅ Hoàn tất phân tích {model_name.upper()}!")

    print("\n🎉🎉 ĐÃ HOÀN TẤT CHẠY XAI CHO TẤT CẢ MODEL VÀ TẤT CẢ CLASS! 🎉🎉")

    print("✅ Đã tạo và lưu các biểu đồ giải thích cục bộ/toàn cục thành công!")



# ==============================================================================
# ĐIỀU CHỈNH LUỒNG CHẠY Ở ĐÂY
# ==============================================================================
if __name__ == "__main__":
    print("=========================================================")
    print(" QUY TRÌNH IDS2018 END-TO-END PIPELINE ")
    print("=========================================================")
    
    # Bật/tắt các hàm bên dưới tuỳ thuộc vào stage muốn chạy:
    
    download_dataset()
    stage0_preprocessing()
    stage1_training()
    stage2_feature_selection()
    stage3_hyperopt()
    stage4_xai()
    
    print("\n✅ CHƯƠNG TRÌNH KẾT THÚC.")
