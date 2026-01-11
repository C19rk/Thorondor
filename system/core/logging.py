import logging, csv
from core.config import LOG_FILE, CSV_FILE

# === LOGGING ===
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# === CSV ===
with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    f.seek(0, 2)
    if f.tell() == 0:
        writer.writerow(["timestamp", "camera", "object", "confidence", "x1", "y1", "x2", "y2"])