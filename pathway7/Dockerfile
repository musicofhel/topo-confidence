# Pathway 7: Non-Euclidean PH Pipeline — Reproducible RunPod Environment
# Build:  docker build -t topo-confidence-p7 .
# Run:    docker run --gpus all -v /workspace:/workspace -it topo-confidence-p7
#
# On RunPod: use as custom Docker image, or build on pod and tag.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="musicofhel"
LABEL description="topo-confidence pathway7: non-Euclidean PH + zigzag + new benchmarks"

# ── System dependencies for Dionysus2 C++ source build ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-all-dev \
        cmake \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (pinned) ────────────────────────────────────────────
COPY requirements_pathway7.txt /tmp/requirements_pathway7.txt
RUN pip install --no-cache-dir -r /tmp/requirements_pathway7.txt

# ── Dionysus2 source build (no PyPI wheels, needs Boost + C++14) ────────────
# Expect ~2-4 min build time
RUN pip install --no-cache-dir dionysus==2.0.10

# ── HumanEval harness (official OpenAI) ─────────────────────────────────────
RUN git clone https://github.com/openai/human-eval /opt/human-eval \
    && cd /opt/human-eval \
    && pip install --no-cache-dir -e .

# ── Pre-download BBH CoT prompts (Suzgun et al.) ───────────────────────────
RUN mkdir -p /opt/bbh-prompts && cd /opt/bbh-prompts \
    && for subset in \
        tracking_shuffled_objects_seven_objects \
        logical_deduction_seven_objects \
        web_of_lies; do \
        curl -sL "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/${subset}.txt" \
            -o "${subset}.txt"; \
    done

# ── Optional: pre-cache models + datasets (uncomment to bake into image) ───
# This adds ~6GB to image size but saves runtime download
# RUN python -c "\
# from transformers import AutoTokenizer, AutoModelForCausalLM; \
# import torch; \
# AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True); \
# AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', \
#     torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True); \
# print('1.5B cached'); \
# "
# RUN python -c "\
# from datasets import load_dataset; \
# load_dataset('openai/openai_humaneval', split='test'); \
# load_dataset('openai/gsm8k', 'main', split='test'); \
# for s in ['tracking_shuffled_objects_seven_objects', 'logical_deduction_seven_objects', 'web_of_lies']: \
#     load_dataset('lukaemon/bbh', s, split='test'); \
# print('Datasets cached'); \
# "

# ── Verify critical imports ─────────────────────────────────────────────────
RUN python -c "\
import torch, transformers, ripser, persim, gudhi, sklearn, scipy, tadasets; \
import dionysus; \
print(f'torch {torch.__version__}, transformers {transformers.__version__}'); \
print(f'ripser OK, gudhi {gudhi.__version__}, dionysus OK'); \
print('All imports verified'); \
"

WORKDIR /workspace
CMD ["/bin/bash"]
