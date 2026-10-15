import os
import pandas as pd

# Define your directory
original_dir = './LABELS'
output_csv = 'dataset_comparison.csv'

print("Scanning datasets...\n")

# Helper function to count only image files in a given folder
def count_images(folder_path):
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# 1. Get a master list of all student names from the folder
all_people = []
if os.path.exists(original_dir):
    all_people = [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))]

# 2. Count the images for each person
data = []
for person in all_people:
    orig_count = count_images(os.path.join(original_dir, person))
    data.append({
        'Name': person,
        'Original Count (LABELS)': orig_count
    })

# 3. Convert to a DataFrame and sort it
df = pd.DataFrame(data)

if not df.empty:
    # Sort by the people who had the most data
    df = df.sort_values(by='Original Count (LABELS)', ascending=False)

    # 4. Save to CSV and print a quick summary
    df.to_csv(output_csv, index=False)
    total_orig = df['Original Count (LABELS)'].sum()

    print("--- Data Summary ---")
    print(f"Total People : {len(df)}")
    print(f"Total Images : {total_orig}")
    print("-" * 20)
    print(f"Success! Report saved to '{output_csv}'")
else:
    print("No data found in the LABELS directory.")