import pandas as pd
from pathlib import Path
import os

root          = Path("/Users/marcodelarca/Desktop/Work/fineasyocr/sroieplay")
jsonl_path    = root / "test" / "metadata.jsonl"
images_dir    = root / "test"
output_csv    = root / "test" / "output.csv"
name_template = "{:05d}.jpg"
# ────────────────────────────────────────────────────────────────────────
print(os.path.exists(jsonl_path))
# 1) Read + sort
df = (
    #Json files
    pd.read_json(jsonl_path, lines=True)
      .sort_values("file_name")
      .rename(columns={"file_name":"old_name", "text":"words"})
      .reset_index(drop=True)
)

# 2) Build new filenames in a vectorized way
df["filename"] = df.index.map(lambda i: name_template.format(i+1))

# 3) Rename files on disk
for old, new in zip(df["old_name"], df["filename"]):
    src = images_dir/old
    dst = images_dir/new
    src.rename(dst)

# 4) Write out just the two columns you need
df[["filename","words"]].to_csv(output_csv, index=False)
