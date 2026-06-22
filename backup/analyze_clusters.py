import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. Configuration & Setup
# ==========================================
# Make sure to point this to your BALANCED folder!
dataset_path = './LABELS' 
output_dir = './Clusters and Similarity'

# Create the output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 1. Load the Data (Once for both tasks)
# ==========================================
X_real = []
labels_all = []
centroids = []
labels_unique = []

print(f"--- Phase 1: Loading Data from '{dataset_path}' ---")

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_dir):
        csv_path = os.path.join(person_dir, f"{person_name}_embeddings.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            embeddings = df.values
            
            # Data for t-SNE (Every single face)
            X_real.extend(embeddings.tolist())
            labels_all.extend([person_name] * len(embeddings))
            
            # Data for Similarity (The "Average Face" per person)
            centroid = np.mean(embeddings, axis=0)
            centroids.append(centroid)
            labels_unique.append(person_name)

X_real = np.array(X_real)
labels_all = np.array(labels_all)
X_centroids = np.array(centroids)
labels_unique = np.array(labels_unique)

print(f"Loaded {len(X_real)} total faces across {len(labels_unique)} people.\n")

if len(X_real) < 5:
    print("Not enough data to visualize. Please extract embeddings first.")
    exit()

# ==========================================
# 2. Cosine Similarity Analysis, Heatmap & CSV
# ==========================================
print("--- Phase 2: Calculating Face Similarity ---")
similarity_matrix = cosine_similarity(X_centroids)

# A. Build the complete list of all pairs
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

# Sort by highest similarity first
pairs.sort(key=lambda x: x['Similarity (Raw)'], reverse=True)

# B. Print Top 10 to Terminal
print("\nTop 10 Most Similar Pairs (Highest Risk of Misclassification):")
for i in range(min(10, len(pairs))):
    p1 = pairs[i]['Person 1']
    p2 = pairs[i]['Person 2']
    score_pct = pairs[i]['Similarity (%)']
    print(f"  {i+1}. {p1} & {p2} -> {score_pct} similar")

# C. Save the full list to CSV
csv_path = os.path.join(output_dir, 'similarity_report.csv')
sim_df = pd.DataFrame(pairs)
sim_df.to_csv(csv_path, index=False)
print(f"\nSaved Full Similarity CSV Report to: {csv_path}")

# D. Draw and Save Heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(
    similarity_matrix, 
    xticklabels=labels_unique, 
    yticklabels=labels_unique, 
    cmap="YlOrRd", 
    square=True,
    vmin=0.0, vmax=1.0 
)
plt.title("Face Similarity Matrix (Red = High Similarity / Danger)", fontsize=20, fontweight='bold')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

heatmap_path = os.path.join(output_dir, 'similarity_heatmap.png')
plt.savefig(heatmap_path, dpi=300)
print(f"Saved Heatmap Image to: {heatmap_path}\n")
plt.close() 

# ==========================================
# 3. t-SNE Clustering Visualization
# ==========================================
print("--- Phase 3: Generating t-SNE Cluster Map ---")
print("Compressing 512 dimensions down to 2 dimensions (this might take a few seconds)...")

pca_components = min(50, len(X_real))
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_real)

perplexity_value = min(30, len(X_real) - 1) 
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
X_2d = tsne.fit_transform(X_pca)

plot_df = pd.DataFrame({'X': X_2d[:, 0], 'Y': X_2d[:, 1], 'Person': labels_all})

plt.figure(figsize=(16, 12)) 
sns.set_theme(style="whitegrid")

scatter = sns.scatterplot(
    x='X', y='Y',
    hue='Person',
    palette=sns.color_palette("husl", len(labels_unique)),
    data=plot_df,
    legend="full",
    alpha=0.7,   
    s=60         
)

plt.title("Face Embeddings Visualization (t-SNE Map)", fontsize=18, fontweight='bold')
plt.xlabel("Dimension 1", fontsize=12)
plt.ylabel("Dimension 2", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8)
plt.tight_layout()

cluster_path = os.path.join(output_dir, 'cluster_map.png')
plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
print(f"Saved Cluster Map Image to: {cluster_path}\n")
plt.close()

print("--- Analysis Complete! ---")