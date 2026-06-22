import os
import shutil
import cv2
import numpy as np
import argparse
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def get_args():
    parser = argparse.ArgumentParser(description="Deterministic Face Clustering")
    # Path Arguments (No longer hard-coded inside the functions)
    parser.add_argument("--embeddings", type=str, required=True, help="Path to .npy embeddings")
    parser.add_argument("--source", type=str, required=True, help="Path to raw face images")
    parser.add_argument("--output", type=str, default="./clusters", help="Path to save clusters")
    
    # Hyperparameters
    parser.add_argument("--threshold", type=float, default=0.45, help="Distance threshold for clustering")
    parser.add_argument("--min_size", type=int, default=10, help="Min faces to keep a cluster")
    
    parser.add_argument("--video", type=str, default=None, help="Prefix of the specific video to cluster, e.g., 'simplescreenrecorder-2026-03-02_12.14.08_20260302'")
    
    return parser.parse_args()

def cluster_video(video_prefix, args):
    print(f"\n--- Processing: {video_prefix} ---")

    embed_file = os.path.join(args.embeddings, f"{video_prefix}.npy")
    names_file = os.path.join(args.embeddings, f"{video_prefix}_faces.npy")

    if not os.path.exists(embed_file) or not os.path.exists(names_file):
        print(f"❌ Missing files for {video_prefix}")
        return

    embeddings = np.load(embed_file)
    filenames = np.load(names_file)

    if len(embeddings) < 2:
        return

    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Scalable Deterministic Clustering
    # If the dataset is too large, use Micro-clustering first to reduce memory footprint (~133GB memory error fix)
    if len(embeddings) > 20000:
        print(f"Dataset is large ({len(embeddings)} faces). Using scalable hierarchical clustering...")
        
        # 1. Cluster into a smaller number of micro-clusters (K-Means) to prevent hanging
        n_micro_clusters = min(2000, max(500, len(embeddings) // 50))
        mbk = MiniBatchKMeans(
            n_clusters=n_micro_clusters, 
            batch_size=10000, 
            n_init=1,
            max_iter=50,
            compute_labels=True,
            random_state=42
        )
        micro_labels = mbk.fit_predict(embeddings)
        micro_centers = mbk.cluster_centers_
        
        # Normalize micro_centers after KMeans for cosine metric
        micro_centers = micro_centers / (np.linalg.norm(micro_centers, axis=1, keepdims=True) + 1e-8)
        
        # 2. Hierarchical clustering on the micro-centers
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=args.threshold
        )
        macro_labels = model.fit_predict(micro_centers)
        
        # 3. Map back to original points
        labels = macro_labels[micro_labels]
    else:
        # Standard clustering for smaller datasets
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=args.threshold
        )
        labels = model.fit_predict(embeddings)

    unique_labels = np.unique(labels)

    # Setup Output Dir
    video_output_dir = Path(args.output) / video_prefix
    if video_output_dir.exists():
        shutil.rmtree(video_output_dir)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    bg_folder = video_output_dir / "background_and_noise"
    bg_folder.mkdir(parents=True, exist_ok=True)

    def save_cluster(label):
        indices = np.where(labels == label)[0]
        
        # Determine target path
        is_noise = len(indices) < args.min_size
        target_folder = bg_folder if is_noise else (video_output_dir / f"person_{label:03d}")

        if not target_folder.exists():
            target_folder.mkdir(parents=True, exist_ok=True)

        cluster_embeddings = []
        best_score = -1
        best_image_path = None
        valid_files_count = 0

        for idx in indices:
            src_path = Path(args.source) / filenames[idx]

            if not src_path.exists():
                continue

            dst_path = target_folder / filenames[idx]
            shutil.copy2(src_path, dst_path)
            valid_files_count += 1

            # Quality Check
            img = cv2.imread(str(src_path))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                score = (img.shape[0] * img.shape[1]) * cv2.Laplacian(gray, cv2.CV_64F).var()
                if score > best_score:
                    best_score = score
                    best_image_path = dst_path
            
            cluster_embeddings.append(embeddings[idx])

        # Metadata for valid clusters
        if valid_files_count > 0 and not is_noise:
            if best_image_path:
                shutil.copy(best_image_path, target_folder / "0000_BEST_FACE.jpg")
            
            avg_emb = np.mean(cluster_embeddings, axis=0)
            avg_emb /= (np.linalg.norm(avg_emb) + 1e-8)
            np.save(target_folder / "cluster_embedding.npy", avg_emb)

    # Use ThreadPoolExecutor to save images using all CPU cores in parallel
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 2) * 4)) as executor:
        futures = {executor.submit(save_cluster, label): label for label in unique_labels}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Saving Clusters", leave=False):
            future.result()

def find_matching_video_prefix(video_prefix, args):
    path_embeds = Path(args.embeddings)
    exact_embed = path_embeds / f"{video_prefix}.npy"
    exact_faces = path_embeds / f"{video_prefix}_faces.npy"

    if exact_embed.exists() and exact_faces.exists():
        return video_prefix

    available_prefixes = [f.stem for f in path_embeds.glob("*.npy") if "_faces" not in f.name]
    matched = [p for p in available_prefixes if video_prefix in p]

    if len(matched) == 1:
        print(f"[INFO] Resolved video prefix '{video_prefix}' to '{matched[0]}'")
        return matched[0]

    if len(matched) > 1:
        exact_starts = [p for p in matched if p.startswith(video_prefix)]
        if len(exact_starts) == 1:
            print(f"[INFO] Resolved video prefix '{video_prefix}' to '{exact_starts[0]}'")
            return exact_starts[0]

        print(f"❌ Multiple embeddings match '{video_prefix}':")
        for p in matched:
            print(f"  - {p}")
        print("Please pass a more specific prefix from the available embedding filenames.")
        return None

    print(f"❌ No embeddings found matching '{video_prefix}'.")
    if available_prefixes:
        print("Available embedding prefixes include:")
        for p in sorted(available_prefixes)[:20]:
            print(f"  - {p}")
        if len(available_prefixes) > 20:
            print(f"  ...and {len(available_prefixes) - 20} more")
    return None


def main():
    args = get_args()
    
    path_embeds = Path(args.embeddings)
    if not path_embeds.exists():
        print("❌ Embeddings directory not found.")
        return

    if args.video:
        resolved_prefix = find_matching_video_prefix(args.video, args)
        if resolved_prefix is None:
            return
        all_npy_files = [resolved_prefix]
    else:
        all_npy_files = [f.stem for f in path_embeds.glob("*.npy") if "_faces" not in f.name]

    print(f"Found {len(all_npy_files)} videos. Clustering with threshold {args.threshold}...")

    for prefix in all_npy_files:
        cluster_video(prefix, args)

    print(f"\n🎉 Finished! Results in: {args.output}")

if __name__ == "__main__":
    main()