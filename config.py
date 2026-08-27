from pathlib import Path
import torch


# Project directory
BASE_DIR = Path(__file__).resolve().parent


# Model path
MODEL_PATH = BASE_DIR / "models" / "best.pt"


# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Default confidence
DEFAULT_CONFIDENCE = 0.5


# Default camera
DEFAULT_CAMERA_INDEX = 0