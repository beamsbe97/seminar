import json
import os
import glob

# 1. Configuration - UPDATE THESE PATHS
json_input_path = 'pascal-5i/VOC2012/features_vit-laion2b_pixel-level_trn/folder0_top_50-similarity_copy.json'
json_output_path = 'pascal-5i/VOC2012/features_vit-laion2b_pixel-level_trn/folder0_top_50-similarity_copy.json'
json_dir = 'pascal-5i/VOC2012/features_vit-laion2b_pixel-level_trn'

# Path to where your .png masks are stored (e.g., VOC2012/SegmentationClass)
mask_dir = 'pascal-5i/VOC2012/SegmentationClass/' 

def mask_exists(name):
    """Checks if the .png mask file exists on disk."""
    return os.path.exists(os.path.join(mask_dir, name + '.png'))
# 2. Find all JSON files in the directory
json_files = glob.glob(os.path.join(json_dir, "*.json"))

if not json_files:
    print(f"No JSON files found in {json_dir}. Please check your path.")
    exit()

print(f"Found {len(json_files)} JSON files to process.")

for json_path in json_files:
    # Skip files that are already cleaned
    print(f"\nProcessing: {os.path.basename(json_path)}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    cleaned_data = {}
    removed_keys = 0
    removed_values_count = 0

    # 3. Apply Filtering Rules
    for key, value_list in data.items():
        # Rule 1: Remove the entire entry if the KEY (Query) mask is missing
        if not mask_exists(key):
            removed_keys += 1
            continue
        
        # Rule 2: Remove names from the list if the VALUE (Support) mask is missing
        original_len = len(value_list)
        filtered_values = [v for v in value_list if mask_exists(v)]
        
        removed_values_count += (original_len - len(filtered_values))
        
        # Save to new dict
        cleaned_data[key] = filtered_values

    # 4. Save the cleaned file
    #output_path = json_path.replace('.json', f'{output_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(cleaned_data, f)
    remaining_entries = len(cleaned_data)

    print(f"Done. Saved to {os.path.basename(json_path)}")
    print(f" - Query masks missing (Entries removed): {removed_keys}")
    print(f" - Support masks missing (Names removed): {removed_values_count}")
    print(f"  > ENTRIES REMAINING:  {remaining_entries}")

print("\nAll files processed successfully.")