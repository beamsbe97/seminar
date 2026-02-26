from pathlib import Path
import re

# Paths
txt_path = Path("splits/pascal/trn/fold2_copy.txt")   # your txt file
root_dir = Path("pascal-5i/VOC2012/SegmentationClass")       # CHANGE THIS

# 1️⃣ Recursively collect all existing .png filenames
existing_pngs = {p.name for p in root_dir.rglob("*.png")}

# 2️⃣ Read txt file
lines = txt_path.read_text().splitlines()

valid_files = []

for line in lines:
    line = line.strip()
    if not line:
        continue

    # Replace "__anything" with ".png"
    filename = re.sub(r"__.*$", ".png", line)

    # Check if filename exists anywhere in folder tree
    if filename in existing_pngs:
        valid_files.append(filename)

# 3️⃣ Overwrite txt with cleaned entries
txt_path.write_text("\n".join(valid_files) + "\n")

print(f"Original entries: {len(lines)}")
print(f"Valid files found: {len(valid_files)}")
print(f"Removed: {len(lines) - len(valid_files)}")