import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .config import TRAINING_DATA_DIR, MODEL_OUTPUT_DIR, EMBEDDING_DIM

TRAINING_DATA_DIR = Path(TRAINING_DATA_DIR)
MODEL_OUTPUT_DIR = Path(MODEL_OUTPUT_DIR)


class PreferenceDataset(Dataset):
    def __init__(self, embeddings, labels, weights=None, middle_embeddings=None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
        self.middle_embeddings = (
            torch.FloatTensor(middle_embeddings)
            if middle_embeddings is not None
            else None
        )
        if weights is not None:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.ones_like(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.middle_embeddings is not None:
            combined = torch.cat([self.embeddings[idx], self.middle_embeddings[idx]])
            return combined, self.labels[idx], self.weights[idx]
        return self.embeddings[idx], self.labels[idx], self.weights[idx]


class UserPreferenceHead(nn.Module):
    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=[256, 128],
        dropout=0.4,
        use_layernorm=True,
    ):
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
    embedding_str = embedding_str.strip("[]")
    return np.array([float(x) for x in embedding_str.split(",")])


def load_user_data(csv_path: Path, augment_cols: list[str] = None):
    embeddings = []
    aug_embeddings = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as f:
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
    initial_state: dict = None,
    sample_weights: np.ndarray = None,
    hidden_dims: list = [256, 128],
    dropout: float = 0.4,
    use_layernorm: bool = True,
    seed: int = 42,
    verbose: bool = True,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    if middle_embeddings is not None:
        middle_embeddings = middle_embeddings / (
            np.linalg.norm(middle_embeddings, axis=1, keepdims=True) + 1e-9
        )

    num_samples = len(labels)

    is_fine_tuning = initial_state is not None

    if is_fine_tuning:
        lr = 0.0002
        num_epochs = 5
        batch_size = 16
        patience = 2
        use_scheduler = False

        dropout = min(dropout + 0.1, 0.5)

    else:
        lr = 0.001
        num_epochs = 50
        batch_size = 32
        patience = 10
        use_scheduler = True

    if num_samples < 5:
        return None

    if is_fine_tuning:
        train_embeddings, train_labels = embeddings, labels
        train_middle = middle_embeddings
        train_weights = (
            sample_weights if sample_weights is not None else np.ones_like(labels)
        )

        val_embeddings, val_labels = embeddings, labels
        val_middle = middle_embeddings
        val_loader = DataLoader(
            PreferenceDataset(val_embeddings, val_labels, None, val_middle),
            batch_size=len(labels),
        )
    else:
        split_data = train_test_split(
            embeddings,
            labels,
            middle_embeddings
            if middle_embeddings is not None
            else np.zeros((len(labels), 1)),
            sample_weights if sample_weights is not None else np.ones_like(labels),
            test_size=0.2,
            random_state=seed,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )
        (
            train_embeddings,
            val_embeddings,
            train_labels,
            val_labels,
            train_middle,
            val_middle,
            train_weights,
            _,
        ) = split_data

        if middle_embeddings is None:
            train_middle, val_middle = None, None

        val_dataset = PreferenceDataset(val_embeddings, val_labels, None, val_middle)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    train_dataset = PreferenceDataset(
        train_embeddings, train_labels, train_weights, train_middle
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = EMBEDDING_DIM
    if middle_embeddings is not None:
        input_dim += middle_embeddings.shape[1]

    model = UserPreferenceHead(
        embedding_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_layernorm=use_layernorm,
    )

    if initial_state:
        model.load_state_dict(initial_state)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    initial_val_loss = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_e, batch_l, batch_w in train_loader:
            optimizer.zero_grad()
            preds = model(batch_e)
            loss = (criterion(preds, batch_l) * batch_w).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_e, batch_l, _ in val_loader:
                preds = model(batch_e)
                val_loss += criterion(preds, batch_l).mean().item()

        if epoch == 0:
            initial_val_loss = val_loss
        if is_fine_tuning and val_loss > (initial_val_loss * 2.0) and val_loss > 0.2:
            if verbose:
                print(f"Loss exploded ({val_loss}), aborting fine-tuning")
            return None

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return {
        "model_state": best_model_state if best_model_state else model.state_dict(),
        "best_val_loss": best_val_loss,
        "is_fine_tuning": is_fine_tuning,
    }


def train_user_model(
    user_id: str,
    csv_path: Path,
    use_weights: bool = True,
    hidden_dims: list = [256, 128],
    dropout: float = 0.4,
    use_layernorm: bool = True,
    use_input_norm: bool = True,
    use_scheduler: bool = True,
    augment_cols: list[str] = ["l6_embedding"],
    seed: int = 42,
    verbose: bool = True,
):
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training model for user {user_id} (Seed: {seed})")
        print(
            f"Config: Dims={hidden_dims}, LN={use_layernorm}, Drop={dropout}, InNorm={use_input_norm}, Sched={use_scheduler}"
        )
        print(f"{'=' * 60}")

    load_result = load_user_data(csv_path, augment_cols)
    if len(load_result) == 3:
        embeddings, labels, middle_embeddings = load_result
    else:
        embeddings, labels = load_result
        middle_embeddings = None

    result = train_model_core(
        embeddings,
        labels,
        middle_embeddings,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_layernorm=use_layernorm,
        use_input_norm=use_input_norm,
        use_scheduler=use_scheduler,
        use_weights=use_weights,
        seed=seed,
        verbose=verbose,
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
        "use_layernorm": True,
        "use_input_norm": True,
        "use_scheduler": True,
        "augment_cols": ["l6_embedding"],
    }

    config_templates = [
        {
            "name": f"{hidden} {dropout}",
            "path": csv_path,
            "hidden_dims": hidden,
            "dropout": dropout,
            **base_cfg,
        }
        for hidden in hidden_dims
        for dropout in dropouts
    ]

    print(f"Benchmarking on user {user_id} with 5 seeds per config...")

    seeds = [42, 123, 999, 2024, 7]
    aggregated_results = []

    for tmpl in config_templates:
        metrics = {
            "val_loss": [],
            "val_mae": [],
            "neg_mae": [],
            "zero_mae": [],
            "pos_mae": [],
        }

        print(f"\nEvaluating {tmpl['name']}...")
        for seed in seeds:
            res = train_user_model(
                user_id,
                tmpl["path"],
                use_weights=True,
                verbose=False,
                hidden_dims=tmpl["hidden_dims"],
                dropout=tmpl["dropout"],
                use_layernorm=tmpl["use_layernorm"],
                use_input_norm=tmpl["use_input_norm"],
                use_scheduler=tmpl["use_scheduler"],
                augment_cols=tmpl.get("augment_cols"),
                seed=seed,
            )

            metrics["val_loss"].append(res["best_val_loss"])
            metrics["val_mae"].append(res["val_mae"])
            metrics["neg_mae"].append(res["per_class_mae"].get("-1.0", 0.0))
            metrics["zero_mae"].append(res["per_class_mae"].get("0.0", 0.0))
            metrics["pos_mae"].append(res["per_class_mae"].get("1.0", 0.0))

        avg_res = {k: np.mean(v) for k, v in metrics.items()}
        std_res = {k: np.std(v) for k, v in metrics.items()}
        aggregated_results.append(
            {"name": tmpl["name"], "avg": avg_res, "std": std_res}
        )

    print("\n" + "=" * 120)
    print(
        f"{'Model Configuration':<25} | {'Val Loss':<18} | {'Val MAE':<18} | {'-1.0 MAE':<18} | {'0.0 MAE':<18} | {'+1.0 MAE':<18}"
    )
    print("-" * 120)

    for r in aggregated_results:
        avg = r["avg"]
        print(
            f"{r['name']:<25} | {avg['val_loss']:.4f}             | {avg['val_mae']:.4f}             | {avg['neg_mae']:.4f}             | {avg['zero_mae']:.4f}             | {avg['pos_mae']:.4f}"
        )
    print("=" * 120)


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
        train_user_model(user_id, csv_path, use_weights=use_weights)
