#!/bin/bash
# v8 pod setup — run once after rsync. Idempotent.
set -e
cd /workspace
pip install -q -U "transformers>=4.49" datasets accelerate hf_transfer 2>&1 | tail -2
mkdir -p /workspace/pod_payload /workspace/pod_results
python3 -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP-OK
