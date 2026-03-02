
from pathlib import Path

jpeg_dir = Path("pascal-5i/VOC2012/JPEGImages")
#txt_path = Path("splits/pascal/trn/fold2_copy.txt")
mask_dir = Path("pascal-5i/VOC2012/SegmentationClass")
data_dir = Path("splits/pascal/trn")                  # folder containing fold*.txt

# ---- Collect existing files once (FAST) ----
existing_jpegs = {p.stem for p in jpeg_dir.rglob("*.jpg")}
existing_masks = {p.stem for p in mask_dir.rglob("*.png")}

# ---- Process each fold file ----
fold_files = list(data_dir.glob("fold*.txt"))

for txt_path in fold_files:
    entries = [
        line.strip()
        for line in txt_path.read_text().splitlines()
        if line.strip()
    ]

    valid_entries = []

    for entry in entries:
        base_name = entry.split("__")[0]

        if base_name in existing_jpegs and base_name in existing_masks:
            valid_entries.append(entry)

    txt_path.write_text("\n".join(valid_entries) + "\n")

    print(f"{txt_path.name}: "
          f"Original={len(entries)}, "
          f"Valid={len(valid_entries)}, "
          f"Removed={len(entries) - len(valid_entries)}")