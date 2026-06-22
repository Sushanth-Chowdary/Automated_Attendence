import os
import pandas as pd

# Define your directories
original_dir = './LABELS'
balanced_dir = './BALANCED_LABELS'
output_csv = 'dataset_comparison.csv'

print("Scanning datasets...\n")

# Helper function to count only image files in a given folder
def count_images(folder_path):
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# 1. Get a master list of all student names from both folders
all_people = set()
if os.path.exists(original_dir):
    all_people.update([d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))])
if os.path.exists(balanced_dir):
    all_people.update([d for d in os.listdir(balanced_dir) if os.path.isdir(os.path.join(balanced_dir, d))])

# 2. Count the images for each person in both directories
data = []
for person in all_people:
    orig_count = count_images(os.path.join(original_dir, person))
    bal_count = count_images(os.path.join(balanced_dir, person))
    
    data.append({
        'Name': person,
        'Original Count (LABELS)': orig_count,
        'Balanced Count (BALANCED_LABELS)': bal_count,
        'Images Removed': orig_count - bal_count
    })

# 3. Convert to a DataFrame and sort it
df = pd.DataFrame(data)

# Sort by the people who had the most original data
df = df.sort_values(by='Original Count (LABELS)', ascending=False)

# 4. Save to CSV and print a quick summary
df.to_csv(output_csv, index=False)

total_orig = df['Original Count (LABELS)'].sum()
total_bal = df['Balanced Count (BALANCED_LABELS)'].sum()

print("--- Data Summary ---")
print(f"Total People           : {len(df)}")
print(f"Total Original Images  : {total_orig}")
print(f"Total Balanced Images  : {total_bal}")
print(f"Total Images Removed   : {total_orig - total_bal}")
print("-" * 20)
print(f"Success! Full report saved to '{output_csv}'")