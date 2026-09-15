"""发布版的集中路径与模型常量。"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data/simulated"
LEVEL4_EXPERIMENTAL_DIR = ROOT_DIR / "data/evaluation"
WEIGHTS_DIR = ROOT_DIR / "artifacts/weights"
OUTPUT_REPORTS_DIR = ROOT_DIR / "artifacts/reports"

MODULE1_DARE_WEIGHTS = WEIGHTS_DIR / "dare_base.pth"
MODULE2_DARE_FUSION_WEIGHTS = WEIGHTS_DIR / "projection_base.pth"
MODULE2_GRAPH_WEIGHTS = WEIGHTS_DIR / "graph_base.pth"

WAVE_LEN = 1800
NUM_FGS = 15
NUM_QUERIES = 128
CHUNK_SIZE = 50_000

