#!/bin/bash
# setup_and_run.sh — full pipeline setup and execution
# usage: bash setup_and_run.sh
# run from the CivicLensNepal project root

set -e  # stop on any error

echo "=================================================="
echo " CivicLensNepal Pipeline — Full Setup and Run"
echo "=================================================="

# ── 1. python version check ───────────────────────────────────────────────────

echo ""
echo "[1/7] Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_VERSION" -gt 12 ]; then
    echo "WARNING: Python 3.13+ detected. PyTorch CUDA wheels may not be available."
    echo "         If torch install fails, install Python 3.12 and rerun."
fi

# ── 2. virtual environment ────────────────────────────────────────────────────

echo ""
echo "[2/7] Setting up virtual environment..."
if [ ! -d "cln_env" ]; then
    python3 -m venv cln_env
    echo "  venv created"
else
    echo "  venv already exists, skipping"
fi

source cln_env/bin/activate
echo "  venv activated"

# ── 3. detect CUDA version and install torch ──────────────────────────────────

echo ""
echo "[3/7] Installing PyTorch with CUDA support..."

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    echo "  CUDA version detected: $CUDA_VERSION"
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif [ "$CUDA_MAJOR" -ge 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    else
        echo "  CUDA version too old — installing CPU-only torch"
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi

    echo "  installing torch from: $TORCH_INDEX"
    pip install torch torchvision --index-url $TORCH_INDEX --quiet

    # verify CUDA is actually usable
    GPU_OK=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [ "$GPU_OK" = "True" ]; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        echo "  GPU ready: $GPU_NAME"
    else
        echo "  WARNING: torch installed but CUDA not available — will run on CPU"
    fi
else
    echo "  nvidia-smi not found — installing CPU-only torch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# ── 4. install remaining dependencies ─────────────────────────────────────────

echo ""
echo "[4/7] Installing dependencies..."
pip install \
    pymupdf \
    sentence-transformers \
    chromadb \
    tqdm \
    PyYAML \
    easyocr \
    python-dotenv \
    numpy \
    --quiet

echo "  all dependencies installed"

# ── 5. clone nep-ttf2utf ──────────────────────────────────────────────────────

echo ""
echo "[5/7] Setting up Preeti converter..."
if [ ! -d "nep-ttf2utf" ]; then
    git clone https://github.com/sapradhan/nep-ttf2utf.git --quiet
    echo "  nep-ttf2utf cloned"
else
    echo "  nep-ttf2utf already exists, skipping"
fi

# ── 6. check data is present ──────────────────────────────────────────────────

echo ""
echo "[6/7] Checking data..."
if [ ! -d "data/raw" ]; then
    echo "ERROR: data/raw not found."
    echo "       Copy your PDFs to data/raw/ before running this script."
    exit 1
fi

PDF_COUNT=$(find data/raw -name "*.pdf" | wc -l)
echo "  found $PDF_COUNT PDFs in data/raw"

# ── 7. run the pipeline ───────────────────────────────────────────────────────

echo ""
echo "[7/7] Running pipeline..."
echo ""

# determine batch size based on available VRAM
if command -v nvidia-smi &> /dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$VRAM_MB" -ge 8000 ]; then
        BATCH_SIZE=512
    elif [ "$VRAM_MB" -ge 6000 ]; then
        BATCH_SIZE=256
    else
        BATCH_SIZE=128
    fi
    echo "  VRAM: ${VRAM_MB}MB — using batch size $BATCH_SIZE"
else
    BATCH_SIZE=64
    echo "  no GPU — using batch size $BATCH_SIZE"
fi

echo ""
echo "--- pass 1: extraction (no OCR) ---"
python3 pipeline.py --extract-only --skip-ocr

echo ""
echo "--- pass 2: extraction (OCR for remaining scanned PDFs) ---"
python3 pipeline.py --extract-only

echo ""
echo "--- pass 3: chunking ---"
python3 pipeline.py --chunk-only

echo ""
echo "--- pass 4: embedding ---"
python3 pipeline.py --embed-only --batch-size $BATCH_SIZE

echo ""
echo "=================================================="
echo " Pipeline complete!"
echo " Copy data/chromadb/ back to your main machine."
echo "=================================================="