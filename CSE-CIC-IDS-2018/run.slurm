#!/bin/bash
#SBATCH --job-name=idps_train
#SBATCH --output=/datastore/inseclab/idps/logs/mps_%j.out
#SBATCH --error=/datastore/inseclab/idps/logs/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:2
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

# Quan trọng - Số vRAM cần dùng (MiB) để tìm GPU phù hợp trên bất kỳ card nào.
# Ví dụ: 30000 ~ 30GB VRAM trống. (Giảm từ 40000 để tránh timeout gpu_check.sh)
REQUIRED_VRAM=40000

# Preflight VRAM guard trước khi chạy python (tránh crash do VRAM tụt ngay trước lúc start).
VRAM_HEADROOM_MIB=1024
VRAM_CHECK_INTERVAL=15
VRAM_STABLE_ROUNDS=2
MAX_PRECHECK_WAIT_SECONDS=0  # 0 = doi vo han den khi du VRAM

# Đường dẫn tuyệt đối để tránh Slurm spool path (read-only)
PROJECT_DIR="/datastore/inseclab/idps"

# Cấu trúc thư mục runtime
LOG_DIR="$PROJECT_DIR/logs"
CACHE_DIR="$PROJECT_DIR/cache"
TMP_CACHE_DIR="$CACHE_DIR/tmp"
PIP_CACHE_DIR="$CACHE_DIR/pip"
TORCH_CACHE_DIR="$CACHE_DIR/torch"
MPL_CACHE_DIR="$CACHE_DIR/matplotlib"
JOBLIB_CACHE_DIR="$CACHE_DIR/joblib"
XDG_CACHE_DIR="$CACHE_DIR/xdg"
MPS_BASE_DIR="$CACHE_DIR/mps"
mkdir -p "$LOG_DIR" "$TMP_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_CACHE_DIR" "$MPL_CACHE_DIR" "$JOBLIB_CACHE_DIR" "$XDG_CACHE_DIR"
mkdir -p "$MPS_BASE_DIR"

echo "=== JOB START $(date) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PROJECT_DIR=$PROJECT_DIR"

# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f
module load shared python312

# Kích hoạt venv cục bộ trong project
if [ ! -f "$PROJECT_DIR/venv/bin/activate" ]; then
    echo "❌ Không tìm thấy venv tại $PROJECT_DIR/venv"
    echo "👉 Hãy chạy trước: bash setup_venv.sh"
    exit 1
fi
source "$PROJECT_DIR/venv/bin/activate"

# Cache/log để tránh đầy /tmp hoặc /home
if [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "ℹ️ SLURM_TMPDIR detected, using node-local scratch for temp/cache"
    export TMPDIR="$SLURM_TMPDIR"
    export PIP_CACHE_DIR="$PIP_CACHE_DIR"
    export TORCH_HOME="$SLURM_TMPDIR/torch"
    export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
    export JOBLIB_TEMP_FOLDER="$SLURM_TMPDIR/joblib"
    export XDG_CACHE_HOME="$SLURM_TMPDIR/xdg"
    mkdir -p "$TMPDIR" "$TORCH_HOME" "$MPLCONFIGDIR" "$JOBLIB_TEMP_FOLDER" "$XDG_CACHE_HOME"
else
    echo "ℹ️ No SLURM_TMPDIR, falling back to project cache"
    export TMPDIR="$TMP_CACHE_DIR"
    export PIP_CACHE_DIR="$PIP_CACHE_DIR"
    export TORCH_HOME="$TORCH_CACHE_DIR"
    export MPLCONFIGDIR="$MPL_CACHE_DIR"
    export JOBLIB_TEMP_FOLDER="$JOBLIB_CACHE_DIR"
    export XDG_CACHE_HOME="$XDG_CACHE_DIR"
fi

# Chặn oversubscription thread (nguyên nhân crash OpenBLAS ở SMOTE-ENN).
export IDPS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export IDPS_SMOTEENN_JOBS=1
export OMP_NUM_THREADS="$IDPS_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$IDPS_NUM_THREADS"
export MKL_NUM_THREADS="$IDPS_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$IDPS_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$IDPS_NUM_THREADS"
export BLIS_NUM_THREADS="$IDPS_NUM_THREADS"

# Dùng venv đã cài sẵn, không pip install lúc runtime
export IDPS_SKIP_RUNTIME_PIP=1

# Xóa biến môi trường Slurm để tự chọn GPU
unset CUDA_VISIBLE_DEVICES

# Debug: Log GPU status trước khi gọi gpu_check.sh
echo "🔍 GPU Status tại thời điểm gọi gpu_check.sh:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "⚠️ nvidia-smi không khả dụng"

# --- GỌI HELPER --- (Quan trọng, cần gọi hàm này (có sẵn) để tìm GPU có vRAM trống >= REQUIRED_VRAM, nếu không tìn thấy GPU đủ vRAM thì hàm CHECK_OUT sẽ đưa job vào lại hàng đợi để chờ tìm slot khác; sau 5 lần requeue mà vẫn chưa tìm được slot thì sẽ trả về mã lỗi để kết thúc job)
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0 # Thoát để Slurm đưa vào queue lại
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1 # Lỗi thật sự (sau 5 lần requeue), dừng Job
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"

# =========================================================
# KHỞI TẠO PRIVATE MPS SERVER
# =========================================================
# Khởi tạo Private MPS với Pipe riêng cho từng JOB ID (để dễ quit)
export CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/nvidia-mps-job$SLURM_JOB_ID/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPS_BASE_DIR/nvidia-mps-job$SLURM_JOB_ID/log"

# Xóa sạch dấu vết cũ nếu có và tạo mới (quan trọng!)
rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

export CUDA_VISIBLE_DEVICES=$BEST_GPU # Quan trọng

# =========================================================
# KIEM TRA VRAM ON DINH TRUOC KHI CHAY
# =========================================================
EFFECTIVE_REQUIRED_VRAM=$((REQUIRED_VRAM + VRAM_HEADROOM_MIB))
STABLE_COUNT=0
ELAPSED_SECONDS=0

echo "Preflight VRAM: can >= ${EFFECTIVE_REQUIRED_VRAM} MiB (REQUIRED=${REQUIRED_VRAM} + HEADROOM=${VRAM_HEADROOM_MIB})"
while true; do
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$BEST_GPU" | tr -d ' ')

    if ! [[ "$FREE_VRAM" =~ ^[0-9]+$ ]]; then
        echo "⚠️ Khong doc duoc VRAM tren GPU $BEST_GPU (gia tri: '$FREE_VRAM'). Thu lai sau ${VRAM_CHECK_INTERVAL}s..."
        STABLE_COUNT=0
    elif [ "$FREE_VRAM" -ge "$EFFECTIVE_REQUIRED_VRAM" ]; then
        STABLE_COUNT=$((STABLE_COUNT + 1))
        echo "✅ GPU $BEST_GPU du VRAM: ${FREE_VRAM} MiB (lan on dinh ${STABLE_COUNT}/${VRAM_STABLE_ROUNDS})"
        if [ "$STABLE_COUNT" -ge "$VRAM_STABLE_ROUNDS" ]; then
            break
        fi
    else
        echo "⏳ GPU $BEST_GPU chua du VRAM: ${FREE_VRAM} MiB < ${EFFECTIVE_REQUIRED_VRAM} MiB. Cho tiep..."
        STABLE_COUNT=0
    fi

    sleep "$VRAM_CHECK_INTERVAL"
    ELAPSED_SECONDS=$((ELAPSED_SECONDS + VRAM_CHECK_INTERVAL))

    if [ "$MAX_PRECHECK_WAIT_SECONDS" -gt 0 ] && [ "$ELAPSED_SECONDS" -ge "$MAX_PRECHECK_WAIT_SECONDS" ]; then
        echo "❌ Da cho ${ELAPSED_SECONDS}s nhung van chua du VRAM on dinh. Ket thuc de tranh crash giua chung."
        exit 1
    fi
done

# =========================================================
# CHẠY CODE
# =========================================================
cd "$PROJECT_DIR"

# fullcode.py chạy toàn bộ pipeline, có thể mất nhiều giờ.
python -u fullcode.py 2>&1 | tee -a "$LOG_DIR/train_${SLURM_JOB_ID}.log"

echo "=== JOB END $(date) ==="
