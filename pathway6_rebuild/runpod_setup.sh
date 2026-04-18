#!/usr/bin/env bash
# Pathway 6 Rebuild — RunPod H100 environment setup
# Run once after pod starts: bash runpod_setup.sh
set -euo pipefail

echo "=== Pathway 6 RunPod Setup ==="

# 1. Verify GPU
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Use an H100 pod."
    exit 1
fi

# 2. Clone repo (skip if already present)
REPO_DIR="${HOME}/topo-confidence"
if [ -d "$REPO_DIR/.git" ]; then
    echo ""
    echo "--- Repo exists, pulling latest ---"
    cd "$REPO_DIR" && git pull
else
    echo ""
    echo "--- Cloning repo ---"
    git clone https://github.com/musicofhel/topo-confidence.git "$REPO_DIR"
fi

# 3. Install Python dependencies
echo ""
echo "--- Installing dependencies ---"
pip install -q -r "$REPO_DIR/pathway6_rebuild/requirements_runpod.txt"

# 4. Verify critical imports
echo ""
echo "--- Verifying Python environment ---"
python -c "
import torch, transformers, ripser, persim, sympy, sklearn, datasets, scipy
print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
print(f'transformers {transformers.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('All imports OK')
"

# 5. Pre-download models (saves time during runs)
echo ""
echo "--- Pre-caching models ---"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading Qwen2.5-1.5B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
    torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True)
print('  1.5B cached.')

print('Downloading Qwen2.5-7B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',
    torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
print('  7B cached.')
"

# 6. Pre-download GSM8K dataset
echo ""
echo "--- Pre-caching GSM8K dataset ---"
python -c "
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main', split='test')
print(f'GSM8K test: {len(ds)} problems cached.')
"

# 7. Verify data files exist
echo ""
echo "--- Verifying data files ---"
cd "$REPO_DIR"
for f in \
    pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy \
    pathway6_rebuild/phase0_relabel/temperature_generations_v2.json \
    pathway6_rebuild/phase0_relabel/holdout_temperature_generations_v2.json \
    data/experiment1_v2/trajectories.npz \
    pathway1/phase1/features_train400.npy \
    pathway1/phase1/features_holdout100.npy; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Setup complete. Run: bash pathway6_rebuild/runpod_run_all.sh ==="
