import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json
import argparse

TRAINING_DATA_DIR = Path("./training_data")
MODEL_OUTPUT_DIR = Path("./preference_models")
EMBEDDING_DIM = 384


class PreferenceDataset(Dataset):
    def __init__(self, embeddings, labels, weights=None, middle_embeddings=None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
        self.middle_embeddings = torch.FloatTensor(middle_embeddings) if middle_embeddings is not None else None
        if weights is not None:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.ones_like(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.middle_embeddings is not None:
            # Concatenate regular and middle embeddings
            combined = torch.cat([self.embeddings[idx], self.middle_embeddings[idx]])
            return combined, self.labels[idx], self.weights[idx]
        return self.embeddings[idx], self.labels[idx], self.weights[idx]


class UserPreferenceHead(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dims=[256, 128], dropout=0.4, use_layernorm=True):
        super().__init__()
        layers = []
        prev_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def parse_embedding_string(embedding_str: str) -> np.ndarray:
    embedding_str = embedding_str.strip('[]')
    return np.array([float(x) for x in embedding_str.split(',')])


def load_user_data(csv_path: Path, augment_cols: list[str] = None):
    embeddings = []
    aug_embeddings = []
    labels = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            embedding = parse_embedding_string(row["embedding"])
            embeddings.append(embedding)
            
            if augment_cols:
                current_aug = []
                for col in augment_cols:
                    if col in row:
                        current_aug.extend(parse_embedding_string(row[col]))
                    else:
                        current_aug.extend(np.zeros_like(embedding))
                aug_embeddings.append(current_aug)
            elif "l6_embedding" in row and row["l6_embedding"]:
                aug_embeddings.append(parse_embedding_string(row["l6_embedding"]))
            elif "middle_embedding" in row:
                aug_embeddings.append(parse_embedding_string(row["middle_embedding"]))
                
            labels.append(float(row["liked"]))
    
    if aug_embeddings:
        return np.array(embeddings), np.array(labels), np.array(aug_embeddings)
    return np.array(embeddings), np.array(labels), None


def train_model_core(
    embeddings: np.ndarray,
    labels: np.ndarray,
    middle_embeddings: np.ndarray = None,
    # Model Config
    hidden_dims: list = [256, 128],
    dropout: float = 0.4,
    use_layernorm: bool = True,
    # Training Config
    use_input_norm: bool = True,
    use_scheduler: bool = True,
    use_weights: bool = True,
    seed: int = 42,
    verbose: bool = True
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Normalize embeddings
    if use_input_norm:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        if middle_embeddings is not None:
            middle_embeddings = middle_embeddings / np.linalg.norm(middle_embeddings, axis=1, keepdims=True)
    
    num_samples = len(labels)
    
    if verbose:
        print(f"Total samples: {num_samples}")
        print(f"Label distribution: {np.bincount(labels.astype(int) + 1, minlength=3)} ([-1, 0, 1])")
    
    if num_samples < 10:
        print("WARNING: Very few samples. Model may not train well.")
    
    if num_samples < 20:
        if verbose:
            print("Using all data for training (no validation split due to small dataset)")
        train_embeddings, train_labels = embeddings, labels
        val_embeddings, val_labels = embeddings[:5], labels[:5]
        
        if middle_embeddings is not None:
            train_middle, val_middle = middle_embeddings, middle_embeddings[:5]
        else:
            train_middle, val_middle = None, None
    else:
        # We need to split middle_embeddings too if they exist
        if middle_embeddings is not None:
            (train_embeddings, val_embeddings, 
             train_labels, val_labels,
             train_middle, val_middle) = train_test_split(
                embeddings, labels, middle_embeddings,
                test_size=0.2, random_state=seed, stratify=labels
            )
        else:
            train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
                embeddings, labels, test_size=0.2, random_state=seed, stratify=labels
            )
            train_middle, val_middle = None, None

    # Calculate weights to handle class imbalance
    if use_weights:
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        # Total samples / (num_classes * class_count)
        class_weights = {lbl: len(train_labels) / (len(unique_labels) * count) for lbl, count in zip(unique_labels, counts)}
        train_sample_weights = np.array([class_weights[lbl] for lbl in train_labels])
        if verbose:
            print("Class weights:", {k: round(v, 4) for k, v in class_weights.items()})
    else:
        if verbose:
            print("Class weights: Disabled (Uniform)")
        train_sample_weights = None
    
    train_dataset = PreferenceDataset(train_embeddings, train_labels, train_sample_weights, train_middle)
    val_dataset = PreferenceDataset(val_embeddings, val_labels, None, val_middle) # No weights needed for val
    
    train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    
    # Adjust input dimension if using middle embeddings
    input_dim = EMBEDDING_DIM
    if middle_embeddings is not None:
        # middle_embeddings might be multiple layers concatenated
        # embeddings.shape[1] is always EMBEDDING_DIM
        # middle_embeddings.shape[1] is (num_layers * EMBEDDING_DIM)
        input_dim = EMBEDDING_DIM + middle_embeddings.shape[1]
        
    model = UserPreferenceHead(embedding_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, use_layernorm=use_layernorm)
    criterion = nn.MSELoss(reduction='none') # We'll apply weights manually
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_embeddings, batch_labels, batch_weights in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_embeddings)
            raw_loss = criterion(predictions, batch_labels)
            weighted_loss = (raw_loss * batch_weights).mean()
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_embeddings, batch_labels, _ in val_loader:
                predictions = model(batch_embeddings)
                # For validation, we use standard MSE to track actual error
                loss = criterion(predictions, batch_labels).mean()
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if scheduler:
            scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            lr_str = f", LR = {optimizer.param_groups[0]['lr']:.6f}" if scheduler else ""
            print(f"Epoch {epoch+1:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}{lr_str}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        if train_middle is not None:
            train_combined = torch.cat([torch.FloatTensor(train_embeddings), torch.FloatTensor(train_middle)], dim=1)
            val_combined = torch.cat([torch.FloatTensor(val_embeddings), torch.FloatTensor(val_middle)], dim=1)
            train_preds = model(train_combined).numpy()
            val_preds = model(val_combined).numpy()
        else:
            train_preds = model(torch.FloatTensor(train_embeddings)).numpy()
            val_preds = model(torch.FloatTensor(val_embeddings)).numpy()
    
    train_mae = np.mean(np.abs(train_preds - train_labels))
    val_mae = np.mean(np.abs(val_preds - val_labels))

    # Calculate per-class MAE
    per_class_mae = {}
    if verbose:
        print(f"\nFinal Results:")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")
        
        print("\n  Per-label Val MAE:")
    
    unique_val_labels = np.unique(val_labels)
    for lbl in sorted(unique_val_labels):
        mask = val_labels == lbl
        mae_lbl = np.mean(np.abs(val_preds[mask] - val_labels[mask]))
        per_class_mae[str(lbl)] = float(mae_lbl)
        if verbose:
            print(f"    Label {lbl:+.1f}: {mae_lbl:.4f} (n={np.sum(mask)})")

    return {
        "model_state": best_model_state if best_model_state else model.state_dict(),
        "best_val_loss": best_val_loss,
        "val_mae": val_mae,
        "per_class_mae": per_class_mae
    }


def train_user_model(
    user_id: str, 
    csv_path: Path, 
    use_weights: bool = True,
    # Model Config
    hidden_dims: list = [256, 128],
    dropout: float = 0.4,
    use_layernorm: bool = True,
    # Training Config
    use_input_norm: bool = True,
    use_scheduler: bool = True,
    augment_cols: list[str] = ["l6_embedding"],
    seed: int = 42,
    verbose: bool = True
):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training model for user {user_id} (Seed: {seed})")
        print(f"Config: Dims={hidden_dims}, LN={use_layernorm}, Drop={dropout}, InNorm={use_input_norm}, Sched={use_scheduler}")
        print(f"{'='*60}")
    
    load_result = load_user_data(csv_path, augment_cols)
    if len(load_result) == 3:
        embeddings, labels, middle_embeddings = load_result
    else:
        embeddings, labels = load_result
        middle_embeddings = None
    
    result = train_model_core(
        embeddings, labels, middle_embeddings,
        hidden_dims=hidden_dims, dropout=dropout, use_layernorm=use_layernorm,
        use_input_norm=use_input_norm, use_scheduler=use_scheduler, use_weights=use_weights,
        seed=seed, verbose=verbose
    )
    
    if verbose:
        user_model_dir = MODEL_OUTPUT_DIR / user_id
        user_model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result["model_state"], user_model_dir / "model.pt")
        print(f"\nModel saved to {user_model_dir}/model.pt")

    return result


def run_benchmark():
    csv_files = list(TRAINING_DATA_DIR.glob("user_*.csv"))
    base_csv_files = [f for f in csv_files if "augmented" not in f.stem]
    
    if not base_csv_files:
        print("No data found.")
        return
    
    csv_path = base_csv_files[0]
    user_id = csv_path.stem.replace("user_", "")
    
    hidden_dims = [[256, 128]]
    dropouts = [0.2]
    base_cfg = {
        "use_layernorm": True, "use_input_norm": True, "use_scheduler": True, "augment_cols": ["l6_embedding"],
    }
    
    config_templates = [
        {
            "name": f"{hidden} {dropout}",
            "path": csv_path,
            "hidden_dims": hidden,
            "dropout": dropout,
            **base_cfg
        }
        for hidden in hidden_dims for dropout in dropouts
    ]
        
    print(f"Benchmarking on user {user_id} with 5 seeds per config...")
    
    seeds = [42, 123, 999, 2024, 7]
    aggregated_results = []
    
    for tmpl in config_templates:
        metrics = {"val_loss": [], "val_mae": [], "neg_mae": [], "zero_mae": [], "pos_mae": []}
        
        print(f"\nEvaluating {tmpl['name']}...")
        for seed in seeds:
            # Note: We need to pass seed to train_user_model if we want strict control,
            # but currently train_test_split uses random_state=42 hardcoded.
            # To vary it, we must modify train_user_model to accept a seed.
            # For now, I will modify train_user_model to accept a seed.
            res = train_user_model(
                user_id, tmpl["path"], use_weights=True, verbose=False,
                hidden_dims=tmpl["hidden_dims"],
                dropout=tmpl["dropout"],
                use_layernorm=tmpl["use_layernorm"],
                use_input_norm=tmpl["use_input_norm"],
                use_scheduler=tmpl["use_scheduler"],
                augment_cols=tmpl.get("augment_cols"),
                seed=seed 
            )
            
            metrics["val_loss"].append(res["best_val_loss"])
            metrics["val_mae"].append(res["val_mae"])
            metrics["neg_mae"].append(res["per_class_mae"].get("-1.0", 0.0))
            metrics["zero_mae"].append(res["per_class_mae"].get("0.0", 0.0))
            metrics["pos_mae"].append(res["per_class_mae"].get("1.0", 0.0))
            
        # Calculate averages
        avg_res = {k: np.mean(v) for k, v in metrics.items()}
        std_res = {k: np.std(v) for k, v in metrics.items()}
        aggregated_results.append({"name": tmpl["name"], "avg": avg_res, "std": std_res})

    print("\n" + "="*120)
    print(f"{'Model Configuration':<25} | {'Val Loss':<18} | {'Val MAE':<18} | {'-1.0 MAE':<18} | {'0.0 MAE':<18} | {'+1.0 MAE':<18}")
    print("-" * 120)
    
    for r in aggregated_results:
        avg = r["avg"]
        print(f"{r['name']:<25} | {avg['val_loss']:.4f}             | {avg['val_mae']:.4f}             | {avg['neg_mae']:.4f}             | {avg['zero_mae']:.4f}             | {avg['pos_mae']:.4f}")
    print("="*120)


def train_all_users(use_weights: bool = True):
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
    
    csv_files = list(TRAINING_DATA_DIR.glob("user_*.csv"))
    
    if not csv_files:
        print(f"No training data found in {TRAINING_DATA_DIR}")
        print("Run export_training_data.py first")
        return
    
    print(f"Found {len(csv_files)} user(s) with training data")
    
    for csv_path in csv_files:
        user_id = csv_path.stem.replace("user_", "")
        # Default to "Big + Advanced" for production training
        train_user_model(user_id, csv_path, use_weights=use_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train user preference models.")
    parser.add_argument("--no-weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--benchmark", action="store_true", help="Run model comparison benchmark")
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark()
    else:
        train_all_users(use_weights=not args.no_weights)
