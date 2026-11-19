import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_dataloaders
from models import Conv1DAE, TemporalTransformer
from utils.metrics import calc_metrics

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='conv1d_ae',
                        choices=['conv1d_ae', 'temporal_transformer'])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=96)
    parser.add_argument('--root', type=str, default='ServerMachineDataset')
    return parser.parse_args()

def main():
    set_seed(42)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("saved_weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_loader, test_loader, scaler = build_dataloaders(
        root_dir=args.root, window_size=args.window_size, batch_size=args.batch_size
    )

    if args.model == "conv1d_ae":
        model = Conv1DAE(in_channels=38, window_len=args.window_size).to(device)
    elif args.model == "temporal_transformer":
        model = TemporalTransformer(in_channels=38, window_len=args.window_size).to(device)
    else:
        raise NotImplementedError

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_path = os.path.join("logs", f"log_{args.model}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "roc_auc", "pr_auc", "pa_f1"])

    best_f1 = 0.0
    for epoch in range(1, args.epoch + 1):
        # Train
        model.train()
        total_train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x.shape[0]
        train_loss = total_train_loss / len(train_loader.dataset)

        # Test & Evaluate
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                recon = model(x)
                score = torch.mean((recon - x) ** 2, dim=(1, 2))
                all_scores.extend(score.cpu().numpy())
                all_labels.extend(y.numpy())

        roc_auc, pr_auc, pa_f1 = calc_metrics(np.array(all_labels), np.array(all_scores))

        print(f"Epoch {epoch:2d} | Loss:{train_loss:.4f} | ROC-AUC:{roc_auc:.4f} | PR-AUC:{pr_auc:.4f} | PA-F1:{pa_f1:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, roc_auc, pr_auc, pa_f1])

        if pa_f1 > best_f1:
            best_f1 = pa_f1
            torch.save(model.state_dict(), os.path.join("saved_weights", f"best_{args.model}.pth"))

    print(f"\nFinished! Best PA-F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
