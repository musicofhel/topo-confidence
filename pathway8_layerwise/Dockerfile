# Pathway 8: Layer-wise PH Pipeline — Extends Pathway 7 Docker environment
# Build:  docker build -t topo-confidence-p8 -f pathway8_layerwise/Dockerfile .
# Run:    docker run --gpus all -v /workspace:/workspace -it topo-confidence-p8

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="musicofhel"
LABEL description="topo-confidence pathway8: layer-wise PH + CoE + D2HScore experiments"

# -- System dependencies for Dionysus2 C++ source build -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-all-dev \
        cmake \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# -- Pathway 7 base dependencies (pinned) ------------------------------------
COPY pathway7/requirements_pathway7.txt /tmp/requirements_pathway7.txt
RUN pip install --no-cache-dir -r /tmp/requirements_pathway7.txt

# -- Pathway 8 additional dependencies ---------------------------------------
COPY pathway8_layerwise/requirements_pathway8.txt /tmp/requirements_pathway8.txt
RUN pip install --no-cache-dir -r /tmp/requirements_pathway8.txt

# -- Dionysus2 source build (no PyPI wheels, needs Boost + C++14) -------------
RUN pip install --no-cache-dir dionysus==2.0.10

# -- HumanEval via EvalPlus (sandboxed, safe) --------------------------------
RUN pip install --no-cache-dir "evalplus>=0.3.1"

# -- Pre-download BBH CoT prompts (Suzgun et al.) ----------------------------
RUN mkdir -p /opt/bbh-prompts && cd /opt/bbh-prompts \
    && for subset in \
        tracking_shuffled_objects_seven_objects \
        logical_deduction_seven_objects \
        web_of_lies; do \
        curl -sL "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/${subset}.txt" \
            -o "${subset}.txt"; \
    done

# -- Verify critical imports -------------------------------------------------
RUN python -c "\
import torch, transformers, ripser, sklearn, scipy, numpy; \
from skdim.id import TwoNN; \
print(f'torch {torch.__version__}, transformers {transformers.__version__}'); \
print(f'numpy {numpy.__version__} (must be <2.0)'); \
print(f'ripser OK, sklearn OK, skdim OK'); \
print('All imports verified'); \
"

WORKDIR /workspace
CMD ["/bin/bash"]
