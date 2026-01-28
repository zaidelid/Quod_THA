"""
Configuration file for project paths and constants.

This module centralizes all file paths and constants used across the project,
ensuring consistency between notebooks, scripts, and production code.
"""

from pathlib import Path

# ============================================================================
# Project Structure
# ============================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# Data Directories
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# ============================================================================
# Output Directories
# ============================================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================================
# Model Constants
# ============================================================================

#
# NOTE: Keep "capacity" knobs that are chosen via validation OUT of config
# (e.g., the final `n_estimators`). Defaults here are intended to be stable,
# reusable hyperparameters that define model behavior.
#

# --- Tweedie GLM defaults ---
DEFAULT_TWEEDIE_POWER = 1.5
DEFAULT_TWEEDIE_ALPHA = 0.0
DEFAULT_TWEEDIE_MAX_ITER = 5000

# --- XGBoost (Poisson) defaults ---
# Validation split for early stopping is chosen relative to the inference cutoff T:
#   V = T - DEFAULT_XGB_VAL_OFFSET_MONTHS
DEFAULT_XGB_VAL_OFFSET_MONTHS = 2

# Tree / regularization knobs (NOT tuned by early stopping in this project)
DEFAULT_XGB_MAX_DEPTH = 5
DEFAULT_XGB_MIN_CHILD_WEIGHT = 1.0
DEFAULT_XGB_SUBSAMPLE = 1.0
DEFAULT_XGB_COLSAMPLE_BYTREE = 1.0
DEFAULT_XGB_GAMMA = 0.0
DEFAULT_XGB_REG_ALPHA = 0.0
DEFAULT_XGB_REG_LAMBDA = 1.0

# Training procedure knobs
DEFAULT_XGB_EARLY_STOPPING_ROUNDS = 50
DEFAULT_XGB_RANDOM_STATE = 42

# ============================================================================
# Utility Functions
# ============================================================================

def ensure_directories():
    """
    Create all necessary directories if they don't exist.
    
    This is useful to call at the start of scripts/notebooks to ensure
    all output directories are available.
    """
    directories = [
        DATA_DIR,
        DATA_RAW,
        DATA_PROCESSED,
        OUTPUTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories
