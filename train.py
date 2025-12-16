# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models import get_model


def count_trainable_params(model: nn.Module) -> int:
    """Return number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc


def run_search(
    candidates,
    train_ds,
    val_ds,
    num_classes,
    device: str = "cpu",
    epochs_per_trial: int = 2,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    """
    Lightweight architecture search:

    - For each candidate model name:
        * Build the model
        * Count parameters
        * Train for a few epochs on train_ds
        * Evaluate on val_ds
    - Pick the best model by validation accuracy.

    Returns:
      best_name: str
      best_score: float (best validation accuracy)
      best_state_dict_on_cpu: dict
      search_logs: list of dicts with keys:
          - "name"
          - "val_acc"
          - "params"
    """
    best_name, best_score, best_state = None, -1.0, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    search_logs = []

    for name in candidates:
        print(f"--- Trial: {name} ---")
        # Pretrained=True for big backbones; for tiny/small CNN it just builds from scratch
        model = get_model(name, num_classes, pretrained=True).to(device)
        param_count = count_trainable_params(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for e in range(epochs_per_trial):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            print(
                f"{name} epoch {e+1}/{epochs_per_trial}: "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} val_acc={val_acc:.3f}"
            )

        # final eval for model selection
        _, final_val_acc = eval_epoch(model, val_loader, criterion, device)

        search_logs.append({
            "name": name,
            "val_acc": float(final_val_acc),
            "params": int(param_count),
        })

        if final_val_acc > best_score:
            best_score = final_val_acc
            best_name = name
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_name, best_score, best_state, search_logs


def fine_tune_model(
    model,
    train_ds,
    val_ds,
    device: str = "cpu",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    """
    Continue training (fine-tuning) a given model for more epochs.
    Returns (best_model, best_val_acc).
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val_acc = 0.0

    for e in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(
            f"[Fine-tune] epoch {e+1}/{epochs}: "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_acc
