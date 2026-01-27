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

# (Add model-related constants here as needed)
# Example:
# DEFAULT_TWEEDIE_POWER = 1.5
# DEFAULT_TRAIN_TEST_SPLIT_DATE = "2019-01-31"

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
