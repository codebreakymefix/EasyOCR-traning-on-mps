from pathlib import Path
from PIL import Image
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────
images_dir = Path("/Users/marcodelarca/Desktop/Work/fineasyocr/sroieplay/train")
# valid extensions (add more if you need)
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
# ────────────────────────────────────────────────────────────────

widths, heights = [], []
for img_path in images_dir.iterdir():
    if img_path.suffix.lower() in exts:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

# guard against empty folder
if not widths:
    raise RuntimeError(f"No images found in {images_dir}")

# compute stats
avg_w = np.mean(widths)
avg_h = np.mean(heights)
avg_px = np.mean([w*h for w, h in zip(widths, heights)])

print(f"Found {len(widths)} images")
print(f"Average width:  {avg_w:.1f}px")
print(f"Average height: {avg_h:.1f}px")
print(f"Average pixels: {avg_px:,.0f}px²")  # total pixel count
