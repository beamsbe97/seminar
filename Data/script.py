
# ---- EDIT PATHS ----

from pathlib import Path

# ---- EDIT THESE PATHS ----

jpeg_dir = Path("pascal-5i/VOC2012/JPEGImages")
txt_path = Path("splits/pascal/trn/fold2_copy.txt")
mask_dir = Path("pascal-5i/VOC2012/SegmentationClass")

# ---- Collect existing files once (FAST) ----
existing_jpegs = {p.stem for p in jpeg_dir.glob("*.jpg")}
existing_masks = {p.stem for p in mask_dir.glob("*.png")}

# ---- Read txt entries ----
entries = [
    line.strip()
    for line in txt_path.read_text().splitlines()
    if line.strip()
]

valid_entries = []

for entry in entries:
    base_name = entry.split("__")[0]

    if base_name in existing_jpegs and base_name in existing_masks:
        valid_entries.append(entry)  # keep original format

# ---- Overwrite txt file ----
txt_path.write_text("\n".join(valid_entries) + "\n")

print(f"Original entries: {len(entries)}")
print(f"Valid entries: {len(valid_entries)}")
print(f"Removed: {len(entries) - len(valid_entries)}")