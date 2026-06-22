import os
import shutil
import random

# Configuration
SOURCE_DIR = './LABELS'
DEST_DIR = './BALANCED_LABELS'
TARGET_IMAGES_PER_PERSON = 50

print(f"--- Starting Data Balancing ---")
print(f"Target: {TARGET_IMAGES_PER_PERSON} images per person.\n")

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# Iterate through every person in your original dataset
for person_name in os.listdir(SOURCE_DIR):
    source_person_dir = os.path.join(SOURCE_DIR, person_name)
    
    if os.path.isdir(source_person_dir):
        # Create matching folder in the new balanced directory
        dest_person_dir = os.path.join(DEST_DIR, person_name)
        if not os.path.exists(dest_person_dir):
            os.makedirs(dest_person_dir)

        # Get all image files for this person
        all_files = [f for f in os.listdir(source_person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Determine how many files to copy
        if len(all_files) > TARGET_IMAGES_PER_PERSON:
            # If they have too much data, pick exactly 50 randomly
            selected_files = random.sample(all_files, TARGET_IMAGES_PER_PERSON)
            print(f"[{person_name}] Scaled down from {len(all_files)} -> {TARGET_IMAGES_PER_PERSON} images.")
        else:
            # If they have 50 or fewer, take all of them
            selected_files = all_files
            print(f"[{person_name}] Kept all {len(all_files)} images (Below target threshold).")

        # Copy the selected files over
        for file_name in selected_files:
            src_file = os.path.join(source_person_dir, file_name)
            dst_file = os.path.join(dest_person_dir, file_name)
            shutil.copyfile(src_file, dst_file) # <--- The fix

print(f"\nSuccess! Your balanced dataset is now in '{DEST_DIR}'.")
print("Your original data in './LABELS' was left untouched.")