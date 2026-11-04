import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import random

# ==========================================
# 0. Configuration & Setup
# ==========================================
dataset_path = './LABELS' 
output_dir = './Below_80_Analysis'
excel_path = './Results/Final_Report_With_Efficiency.xlsx'
mapping_path = './Results/name_mapping.csv'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
# ==========================================
# 1. Load Name Mapping 
# ==========================================
print(f"--- Phase 1: Loading mapping from '{mapping_path}' ---")
df_map = pd.read_csv(mapping_path)
# Create a dictionary mapping the Roll Number (e.g., EE25B010) to Short Name
roll_to_short = dict(zip(df_map['Roll_Name'], df_map['Short_Name']))

# ==========================================
# 2. Identify Students Below 80% Efficiency
# ==========================================
print(f"--- Phase 2: Identifying students below 80% from '{excel_path}' ---")
df_excel = pd.read_excel(excel_path, sheet_name='Attendance Efficiency', header=None)

# Find row containing student names
start_row = None
for i in range(len(df_excel)):
    if str(df_excel.iloc[i, 0]).startswith('Student Name'):
        start_row = i + 2
        break

if start_row is None:
    print("Could not find 'Student Name' row in excel.")
    exit()

true_cols = [j for j, val in enumerate(df_excel.iloc[2]) if str(val) == 'True']
est_cols = [j for j, val in enumerate(df_excel.iloc[2]) if str(val) == 'Estimated']

# List of tuples: (full_name, short_name, efficiency)
below_80_students = []

for i in range(start_row, len(df_excel)):
    name_str = str(df_excel.iloc[i, 0]).strip()
    if pd.isna(name_str) or name_str == 'nan' or 'Total Overall Accuracy' in name_str:
        continue
        
    true_present = 0
    est_correct = 0
    
    for t_col, e_col in zip(true_cols, est_cols):
        true_val = df_excel.iloc[i, t_col]
        est_val = df_excel.iloc[i, e_col]
        
        if true_val == 'Present':
            true_present += 1
            if est_val == 'Present':
                est_correct += 1
                
    if true_present > 0:
        eff = est_correct / true_present
        if eff < 0.8:
            # Extract roll number (e.g., "EE25B001" from "EE25B001 - ACHANTA ADITYA RAM")
            roll_num = name_str.split(' - ')[0].strip()
            short_name = roll_to_short.get(roll_num)
            
            if short_name:
                below_80_students.append((name_str, short_name.strip(), eff))
            else:
                print(f"  [Warning] Mapping not found for roll number: {roll_num}")

print(f"Found {len(below_80_students)} students with < 80% accuracy:")
for full_name, short_name, eff in below_80_students:
    print(f"  - {short_name} (Efficiency: {eff * 100:.2f}%)")

# ==========================================
# 3. Load the Data for Target Students
# ==========================================
X_real = []
labels_all = []
centroids = []
labels_unique = []

print(f"\n--- Phase 3: Loading Embeddings from '{dataset_path}' ---")

for full_name, short_name, eff in below_80_students:
    person_dir = os.path.join(dataset_path, short_name)
    if os.path.isdir(person_dir):
        # Depending on how your embeddings are named, adjust the filename here if needed
        csv_path = os.path.join(person_dir, f"{short_name}_sota_v2_embeddings.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            embeddings = df.values
            
            X_real.extend(embeddings.tolist())
            labels_all.extend([short_name] * len(embeddings))
            
            centroid = np.mean(embeddings, axis=0)
            centroids.append(centroid)
            labels_unique.append(short_name)
        else:
            print(f"  [Warning] Embeddings CSV not found for {short_name} at {csv_path}")
    else:
        print(f"  [Warning] Directory not found for {short_name} in LABELS")

X_real = np.array(X_real)
labels_all = np.array(labels_all)
X_centroids = np.array(centroids)
labels_unique = np.array(labels_unique)

print(f"Loaded {len(X_real)} total faces across {len(labels_unique)} target people.\n")

if len(X_real) < 5:
    print("Not enough data to visualize. Exiting.")
    exit()

# ==========================================
# 4. Cosine Similarity Analysis & Heatmap
# ==========================================
print("--- Phase 4: Calculating Face Similarity ---")
similarity_matrix = cosine_similarity(X_centroids)

pairs = []
for i in range(len(labels_unique)):
    for j in range(i + 1, len(labels_unique)):
        score = similarity_matrix[i, j]
        pairs.append({
            'Person 1': labels_unique[i], 
            'Person 2': labels_unique[j], 
            'Similarity (Raw)': score,
            'Similarity (%)': f"{score * 100:.2f}%"
        })

pairs.sort(key=lambda x: x['Similarity (Raw)'], reverse=True)

print("\nTop Similar Pairs Among <80% Group:")
for i in range(min(10, len(pairs))):
    print(f"  {i+1}. {pairs[i]['Person 1']} & {pairs[i]['Person 2']} -> {pairs[i]['Similarity (%)']}")

csv_path = os.path.join(output_dir, 'below_80_similarity_report.csv')
pd.DataFrame(pairs).to_csv(csv_path, index=False)
print(f"Saved Similarity CSV to: {csv_path}")

plt.figure(figsize=(10, 8))
sns.heatmap(
    similarity_matrix, 
    xticklabels=labels_unique, 
    yticklabels=labels_unique, 
    cmap="YlOrRd", 
    square=True,
    vmin=0.0, vmax=1.0,
    annot=True, fmt=".2f", annot_kws={"size": 10}
)
plt.title("Face Similarity Matrix (<80% Accuracy Group)", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

heatmap_path = os.path.join(output_dir, 'below_80_similarity_heatmap.png')
plt.savefig(heatmap_path, dpi=300)
plt.close() 
print(f"Saved Heatmap to: {heatmap_path}\n")

# ==========================================
# 5. t-SNE Clustering Visualization
# ==========================================
print("--- Phase 5: Generating t-SNE Cluster Map ---")

pca_components = min(50, len(X_real))
if pca_components > 0:
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_real)
    
    perplexity_value = min(30, len(X_real) - 1) 
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    X_2d = tsne.fit_transform(X_pca)
    
    plot_df = pd.DataFrame({'X': X_2d[:, 0], 'Y': X_2d[:, 1], 'Person': labels_all})
    
    plt.figure(figsize=(12, 8)) 
    sns.set_theme(style="whitegrid")
    
    # Grab distinct colors for the students
    colors_tab20 = plt.cm.tab20.colors
    full_palette = list(colors_tab20)[:len(labels_unique)]
    random.seed(42) 
    random.shuffle(full_palette) 
    
    scatter = sns.scatterplot(
        x='X', y='Y',
        hue='Person',
        palette=full_palette,
        data=plot_df,
        legend="full",
        alpha=0.8,
        s=80
    )
    
    plt.title("Face Embeddings t-SNE Map (<80% Accuracy Group)", fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.tight_layout()
    
    cluster_path = os.path.join(output_dir, 'below_80_cluster_map.png')
    plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE Cluster Map to: {cluster_path}")

print("\n--- Analysis Complete! ---")