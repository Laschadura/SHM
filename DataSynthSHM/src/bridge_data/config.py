from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent.parent   # <repo root>
DATA_DIR   = BASE_DIR / "data" / "Data"
LABELS_DIR = BASE_DIR / "data" / "Labels"

IMAGE_SHAPE     = (256, 768)
SKIP_TESTS      = [23, 24]
EXPECTED_LENGTH = 12_000                # samples per CSV (≈60 s @ 200 Hz)

# ID → label-file mapping
PERSPECTIVE_MAP = {
    "A": "Arch_Intrados",
    "B": "North_Spandrel_Wall",
    "C": "South_Spandrel_Wall",
}
