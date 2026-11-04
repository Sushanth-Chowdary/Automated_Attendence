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
output_dir = './Clusters and Similarity'
excel_path = './Results/Final_Report_With_Efficiency.xlsx'
mapping_path = './Results/name_mapping.csv'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 1. Load Name Mapping 
# ==========================================
print(f"--- Phase 1: Loading mapping from '{mapping_path}' ---")
df_map = pd.read_csv(mapping_path)
roll_to_short = dict(zip(df_map['Roll_Name'], df_map['Short_Name']))

# ==========================================
# 2. Identify Students Below 80% Efficiency
# ==========================================
print(f"--- Phase 2: Identifying students below 80% from '{excel_path}' ---")
df_excel = pd.read_excel(excel_path, sheet_name='Attendance Efficiency', header=None)

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

below_80_students_short_names = set()

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
            roll_num = name_str.split(' - ')[0].strip()
            short_name = roll_to_short.get(roll_num)
            if short_name:
                below_80_students_short_names.add(short_name.strip())

print(f"Found {len(below_80_students_short_names)} students with < 80% accuracy:")
for name in below_80_students_short_names:
    print(f"  - {name}")

# ==========================================
# 3. Load the Data for ALL Students
# ==========================================
X_real = []
labels_all = []
centroids = []
labels_unique = []

print(f"\n--- Phase 3: Loading ALL Embeddings from '{dataset_path}' ---")

for person_name in sorted(os.listdir(dataset_path)): 
    person_dir = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_dir):
        csv_path = os.path.join(person_dir, f"{person_name}_sota_v2_embeddings.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            embeddings = df.values
            
            X_real.extend(embeddings.tolist())
            labels_all.extend([person_name] * len(embeddings))
            
            centroid = np.mean(embeddings, axis=0)
            centroids.append(centroid)
            labels_unique.append(person_name)

X_real = np.array(X_real)
labels_all = np.array(labels_all)
X_centroids = np.array(centroids)
labels_unique = np.array(labels_unique)

print(f"Loaded {len(X_real)} total faces across {len(labels_unique)} people.\n")

if len(X_real) < 5:
    print("Not enough data to visualize. Exiting.")
    exit()

# ==========================================
# 4. t-SNE Clustering Visualization (ALL students, <80% Highlighted)
# ==========================================
print("--- Phase 4: Generating t-SNE Cluster Map ---")

pca_components = min(50, len(X_real))
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_real)

perplexity_value = min(30, len(X_real) - 1) 
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
X_2d = tsne.fit_transform(X_pca)

plot_df = pd.DataFrame({'X': X_2d[:, 0], 'Y': X_2d[:, 1], 'Person': labels_all})

# Create custom palette
# Set light gray for >= 80%, distinct colors for < 80%
colors_tab20 = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
random.seed(42)
random.shuffle(colors_tab20)

color_idx = 0
custom_palette = {}
for person in labels_unique:
    if person in below_80_students_short_names:
        custom_palette[person] = colors_tab20[color_idx % len(colors_tab20)]
        color_idx += 1
    else:
        custom_palette[person] = '#d3d3d3' # light gray

# Reorder plot_df to plot gray points first, then highlighted points on top
plot_df['is_highlighted'] = plot_df['Person'].isin(below_80_students_short_names)
plot_df = plot_df.sort_values('is_highlighted')

plt.figure(figsize=(16, 12)) 
sns.set_theme(style="whitegrid")

scatter = sns.scatterplot(
    x='X', y='Y',
    hue='Person',
    palette=custom_palette,
    data=plot_df,
    legend=False, # We'll create a custom legend to only show highlighted students
    alpha=0.7,
    s=60
)

# Custom legend for only the highlighted students
handles = []
labels = []
for person in below_80_students_short_names:
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[person], markersize=10))
    labels.append(person)

handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d3d3d3', markersize=10))
labels.append('Others (>= 80%)')

plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10, title="Accuracy < 80%")

plt.title("Face Embeddings t-SNE Map (Low Accuracy Group Highlighted)", fontsize=18, fontweight='bold')
plt.xlabel("Dimension 1", fontsize=12)
plt.ylabel("Dimension 2", fontsize=12)
plt.tight_layout()

cluster_path = os.path.join(output_dir, 'cluster_map_highlighted.png')
plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved t-SNE Cluster Map Image to: {cluster_path}\n")

print("--- Analysis Complete! ---")