"""Pathway 11 H100 pod run — paths, stage names, model constants.

Thin layer over pathway8_layerwise/config.py. Adds pathway11-specific
directories and the 7B model name. All stages import from here so the
orchestrator can refer to paths symbolically.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Expose pathway8 config
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pathway8_layerwise.config import (  # noqa: E402, F401
    BBH_DATA_DIR,
    HIDDEN_DIM,
    MATH500_DATA_DIR,
    MODEL_NAME,  # 1.5B, used for stages 2, 3, 4a
    N_PROBLEMS,
    N_TOTAL_LAYERS,
    N_TRANSFORMER_LAYERS,
    is_done,
    load_all_layer_states,
    load_manifest,
    mark_done,
)

# ---- Pathway 11 directories -----------------------------------------------
P11_DIR = Path(__file__).resolve().parent
DATA_DIR = P11_DIR / "data"
RESULTS_DIR = P11_DIR / "results"
LOGS_DIR = P11_DIR / "logs"

MATH500_7B_DATA_DIR = DATA_DIR / "math500_7b"
K8_DATA_DIR = DATA_DIR / "k8_selfconsistency"

# ---- Model constants ------------------------------------------------------
MODEL_NAME_7B = "Qwen/Qwen2.5-7B-Instruct"  # matches pathway6_rebuild/phase6_5
HIDDEN_DIM_7B = 3584

# ---- Stage defaults -------------------------------------------------------
MAX_NEW_TOKENS = 1024
K_SAMPLES = 8
SAMPLING_TEMPERATURE = 0.7
STEERING_LAYER = 19  # L19 is pathway 10 v2's steering layer

# ---- BBH subset names (match pathway8_layerwise/extract_bbh.py:BBH_SUBSETS)
BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]
