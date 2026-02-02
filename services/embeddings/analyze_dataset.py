import csv
import numpy as np
from pathlib import Path
from collections import Counter

TRAINING_DATA_DIR = Path("./training_data")


def parse_embedding_string(embedding_str: str) -> np.ndarray:
    embedding_str = embedding_str.strip('[]')
    return np.array([float(x) for x in embedding_str.split(',')])


def analyze_user_data(csv_path: Path):
    user_id = csv_path.stem.replace("user_", "")
    
    labels = []
    embeddings = []
    titles = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(float(row["liked"]))
            embeddings.append(parse_embedding_string(row["embedding"]))
            titles.append(row["title"][:60] if row["title"] else "")
    
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    
    print(f"\n{'='*70}")
    print(f"User: {user_id}")
    print(f"{'='*70}")
    
    print(f"\nDataset Size: {len(labels)} samples")
    
    label_counts = Counter(labels)
    print(f"\nLabel Distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(labels)) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"  {label:+4.1f}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    unique_labels = len(label_counts)
    print(f"\nUnique label values: {unique_labels}")
    
    if len(labels) > 1:
        print(f"\nLabel Statistics:")
        print(f"  Mean:   {np.mean(labels):+.3f}")
        print(f"  Median: {np.median(labels):+.3f}")
        print(f"  Std:    {np.std(labels):.3f}")
        print(f"  Min:    {np.min(labels):+.3f}")
        print(f"  Max:    {np.max(labels):+.3f}")
    
    if len(embeddings) > 1:
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        pairwise_similarities = np.dot(embeddings, embeddings.T)
        
        np.fill_diagonal(pairwise_similarities, -np.inf)
        max_similarity = np.max(pairwise_similarities)
        min_similarity = np.min(pairwise_similarities[pairwise_similarities > -np.inf])
        mean_similarity = np.mean(pairwise_similarities[pairwise_similarities > -np.inf])
        
        print(f"\nEmbedding Statistics:")
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  Norm mean: {np.mean(embedding_norms):.3f} (std: {np.std(embedding_norms):.3f})")
        print(f"\nPairwise Cosine Similarity:")
        print(f"  Mean: {mean_similarity:.3f}")
        print(f"  Min:  {min_similarity:.3f}")
        print(f"  Max:  {max_similarity:.3f}")
    
    class_balance = max(label_counts.values()) / len(labels)
    print(f"\nClass Balance:")
    print(f"  Majority class ratio: {class_balance:.1%}")
    if class_balance > 0.7:
        print(f"  ⚠️  Warning: Dataset is imbalanced (>{class_balance:.0%} in one class)")
    
    if len(labels) < 20:
        print(f"  ⚠️  Warning: Very small dataset (<20 samples)")
    elif len(labels) < 50:
        print(f"  ⚠️  Warning: Small dataset (<50 samples)")
    
    if unique_labels == 3 and set(label_counts.keys()) == {-1.0, 0.0, 1.0}:
        print(f"\n  ℹ️  Using discrete labels {-1, 0, 1}. Consider continuous ratings for better training.")
    
    print(f"\nSample Articles by Label:")
    for label in sorted(label_counts.keys(), reverse=True):
        matching_indices = np.where(labels == label)[0][:3]
        print(f"\n  Label {label:+.1f}:")
        for idx in matching_indices:
            print(f"    - {titles[idx]}")
    
    return {
        "user_id": user_id,
        "num_samples": len(labels),
        "label_distribution": dict(label_counts),
        "mean_label": float(np.mean(labels)),
        "std_label": float(np.std(labels)),
    }


def analyze_all_users():
    csv_files = list(TRAINING_DATA_DIR.glob("user_*.csv"))
    
    if not csv_files:
        print(f"No training data found in {TRAINING_DATA_DIR}")
        print("Run export_training_data.py first")
        return
    
    print(f"Found {len(csv_files)} user(s) with training data")
    
    all_stats = []
    for csv_path in csv_files:
        stats = analyze_user_data(csv_path)
        all_stats.append(stats)
    
    if len(all_stats) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL USERS")
        print(f"{'='*70}")
        
        total_samples = sum(s["num_samples"] for s in all_stats)
        print(f"\nTotal samples across all users: {total_samples}")
        print(f"Average samples per user: {total_samples / len(all_stats):.1f}")


if __name__ == "__main__":
    analyze_all_users()
