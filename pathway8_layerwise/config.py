"""Constants, paths, and shared data-loading utilities for Pathway 8.

All experiments import from here to ensure consistent splits and labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
P8_DIR = Path(__file__).resolve().parent

DATA_DIR = P8_DIR / "data"
RESULTS_DIR = P8_DIR / "results"
LOGS_DIR = P8_DIR / "logs"

# Existing data paths
TRAJ_PATH = REPO_ROOT / "data" / "experiment1_v2" / "trajectories.npz"
LABELS_V2_PATH = REPO_ROOT / "pathway6_rebuild" / "phase0_relabel" / "baseline_correct_v2.npy"
OLD_LABELS_PATH = REPO_ROOT / "pathway2" / "track_a" / "phase0" / "baseline_correct.npy"
FEATURES_TRAIN_PATH = REPO_ROOT / "pathway1" / "phase1" / "features_train400.npy"
FEATURES_HOLDOUT_PATH = REPO_ROOT / "pathway1" / "phase1" / "features_holdout100.npy"
TIER_PATH = REPO_ROOT / "pathway1" / "phase2" / "tier_assignments.json"
FEATURE_NAMES_PATH = REPO_ROOT / "pathway6_rebuild" / "phase2_completion" / "feature_names_v2.json"
BASELINE_PROBS_PATH = REPO_ROOT / "pathway7" / "results_phase71" / "probs_baseline_holdout.npy"

# Pathway 8 data subdirectories
MATH500_DATA_DIR = DATA_DIR / "math500"
HUMANEVAL_DATA_DIR = DATA_DIR / "humaneval"
BBH_DATA_DIR = DATA_DIR / "bbh"

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_TRANSFORMER_LAYERS = 28  # layers 1-28 (layer 0 = embedding)
HIDDEN_DIM = 1536
N_TOTAL_LAYERS = 29  # including embedding layer 0

# ---------------------------------------------------------------------------
# Feature extraction constants
# ---------------------------------------------------------------------------

N_PCA_PER_LAYER = 20  # PCA components per layer (smaller than global 45)
N_PH_FEATURES = 6  # PH features per layer
TOTAL_LAYERWISE_FEATURES = N_TRANSFORMER_LAYERS * N_PH_FEATURES  # 168
SUBSAMPLE = 100  # max points for ripser
MAX_DIM = 1  # H0 + H1
SEED = 42

# The 6 PH features computed per layer
PH_FEATURE_NAMES = [
    "H0_total_persistence",
    "H0_n_features",
    "H0_entropy",
    "H1_persistence_entropy",
    "H1_n_features",
    "H1_max_lifetime",
]

# ---------------------------------------------------------------------------
# Classifier constants (match pathway6_rebuild)
# ---------------------------------------------------------------------------

LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42)
SSS_RANDOM_STATE = 9999
SSS_TEST_SIZE = 100
N_PROBLEMS = 500

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_labels_and_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load corrected labels and reconstruct train/holdout split.

    CRITICAL: Features in features_train400.npy were saved using SSS split
    with OLD labels (57 correct). We must use OLD labels to reconstruct the
    split ordering, then apply NEW labels (104 correct) for actual training.

    Returns (new_labels, train_idx, holdout_idx) where indices are sorted.
    """
    new_labels = np.load(LABELS_V2_PATH)
    old_labels = np.load(OLD_LABELS_PATH)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=SSS_TEST_SIZE, random_state=SSS_RANDOM_STATE
    )
    train_idx_sss, hold_idx_sss = next(
        sss.split(np.zeros(len(old_labels)), old_labels.astype(int))
    )

    train_idx = np.sort(train_idx_sss)
    holdout_idx = np.sort(hold_idx_sss)

    return new_labels, train_idx, holdout_idx


def load_existing_abc_features() -> tuple[np.ndarray, np.ndarray]:
    """Load existing 44 ABC-tier features with SSS ordering fix.

    Returns (X_train, X_holdout) in sorted global index order.
    """
    tier_data = json.loads(TIER_PATH.read_text())
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
    abc_cols = [
        i for i, name in enumerate(feature_names)
        if tier_data.get(name) in ("A", "B", "C")
    ]

    old_labels = np.load(OLD_LABELS_PATH)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=SSS_TEST_SIZE, random_state=SSS_RANDOM_STATE
    )
    train_sss, hold_sss = next(
        sss.split(np.zeros(len(old_labels)), old_labels.astype(int))
    )
    train_sort = np.argsort(train_sss)
    hold_sort = np.argsort(hold_sss)

    X_train = np.load(FEATURES_TRAIN_PATH)[train_sort][:, abc_cols]
    X_holdout = np.load(FEATURES_HOLDOUT_PATH)[hold_sort][:, abc_cols]

    return X_train, X_holdout


def load_all_layer_states(
    data_dir: Path,
    n_problems: int | None = None,
) -> list[np.ndarray | None]:
    """Load per-problem all-layer states from npz checkpoint files.

    Returns list of (29, n_tokens, 1536) arrays (or None for missing problems).
    """
    if n_problems is None:
        n_problems = len(list(data_dir.glob("problem_*.npz")))
        if n_problems == 0:
            raise FileNotFoundError(f"No problem_*.npz files in {data_dir}")

    states_list: list[np.ndarray | None] = []
    for i in range(n_problems):
        path = data_dir / f"problem_{i:03d}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            states_list.append(data["states"])
        else:
            states_list.append(None)

    loaded = sum(1 for s in states_list if s is not None)
    print(f"  Loaded {loaded}/{n_problems} all-layer state files from {data_dir}")
    return states_list


def load_manifest(data_dir: Path) -> dict:
    """Load manifest.json from a data directory."""
    path = data_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.json in {data_dir}")
    return json.loads(path.read_text())


def make_layerwise_feature_names(
    layers: range | None = None,
) -> list[str]:
    """Generate feature names like L01_H0_total_persistence, etc."""
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)
    names = []
    for layer in layers:
        for feat in PH_FEATURE_NAMES:
            names.append(f"L{layer:02d}_{feat}")
    return names


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def mark_done(stage: str, results_dir: Path | None = None) -> None:
    """Write a .done marker file for a completed stage."""
    d = results_dir or RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stage}.done").touch()


def is_done(stage: str, results_dir: Path | None = None) -> bool:
    """Check if a stage has a .done marker."""
    d = results_dir or RESULTS_DIR
    return (d / f"{stage}.done").exists()
