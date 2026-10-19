"""Project paths shared by all modules and scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "course_files_export" / "Zadatak"
CAMERA_CAL_DIR = DATA_DIR / "camera_cal"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
TEST_VIDEOS_DIR = DATA_DIR / "test_videos"

OUTPUT_IMAGES_DIR = ROOT / "output_images"
OUTPUT_VIDEOS_DIR = ROOT / "output_videos"
CALIBRATION_FILE = ROOT / "calibration.npz"
