import json
import os

# 1. Configuration - UPDATE THESE PATHS
json_input_path = 'pascal-5i/VOC2012/features_vit-laion2b_pixel-level_trn/folder0_top_50-similarity_copy.json'
json_output_path = 'pascal-5i/VOC2012/features_vit-laion2b_pixel-level_trn/folder0_top_50-similarity_copy.json'


# Path to where your .png masks are stored (e.g., VOC2012/SegmentationClass)
mask_dir = 'pascal-5i/VOC2012/SegmentationClass/' 

def mask_exists(name):
    """Checks if the .png mask file exists on disk."""
    return os.path.exists(os.path.join(mask_dir, name + '.png'))

# 2. Load the original JSON
print(f"Loading {json_input_path}...")
with open(json_input_path, 'r') as f:
    data = json.load(f)

cleaned_data = {}
removed_keys = 0
removed_values_count = 0

print("Filtering based on .png mask existence...")

# 3. Apply Filtering Rules
for key, value_list in data.items():
    # Rule 1: If the KEY (Query) mask doesn't exist, remove the entire entry
    if not mask_exists(key):
        removed_keys += 1
        continue
    
    # Rule 2: If a VALUE (Support) mask doesn't exist, remove it from the list
    original_len = len(value_list)
    filtered_values = [v for v in value_list if mask_exists(v)]
    
    removed_values_count += (original_len - len(filtered_values))
    
    # Only keep the entry if the query mask exists
    cleaned_data[key] = filtered_values

# 4. Save results
print(f"\n--- Cleaning Complete ---")
print(f"Entries removed because Query mask was missing: {removed_keys}")
print(f"Names removed from lists because Support mask was missing: {removed_values_count}")
print(f"Final count of valid Query entries: {len(cleaned_data)}")

with open(json_output_path, 'w') as f:
    json.dump(cleaned_data, f)

print(f"Cleaned file saved to: {json_output_path}")