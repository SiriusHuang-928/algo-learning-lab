import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_dataloaders
from models.cnn_baseline import SimpleCNN
from models.resnet import ResNet18
from utils.metrics import calculate_classification_metrics, save_training_log
from models.efficientnet_b0 import EfficientNetB0
from models.vit_baseline import ViT_Baseline


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Wafer Defect Classification Training")
    parser.add_argument("--model", type=str, default="cnn_baseline",
                        choices=["cnn_baseline", "resnet18", "efficientnet_b0", "vit"], help="model select")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sample_num', type=int, default=50000)
    parser.add_argument('--save_dir', type=str, default='./saved_weights')
    parser.add_argument('--figure_dir', type=str, default='./output_figures')
    parser.add_argument('--weight_filename', type=str, default=None)
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        total += x_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    metrics = calculate_classification_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    start_time = time.time()
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        total += x_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    infer_time = time.time() - start_time
    avg_loss = total_loss / total
    metrics = calculate_classification_metrics(all_labels, all_preds)
    metrics["infer_time_total"] = round(infer_time, 3)
    metrics["infer_time_per_sample"] = round(infer_time / total, 6)
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动填充文件名，解决None报错
    if args.weight_filename is None:
        args.weight_filename = f"{args.model}_best.pth"

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    # 每个模型独立日志，不会互相覆盖
    log_path = os.path.join(args.figure_dir, f"{args.model}_training_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    # Build data
    train_loader, val_loader, label_names = build_dataloaders(
        batch_size=args.batch_size, sample_num=args.sample_num
    )
    num_classes = len(label_names)

    # Build model
    if args.model == "cnn_baseline":
        model = SimpleCNN(num_classes=num_classes).to(device)
    elif args.model == "resnet18":
        model = ResNet18(num_classes=num_classes).to(device)
    elif args.model == "efficientnet_b0":
        model = EfficientNetB0(num_classes=num_classes).to(device)
    elif args.model == "vit":
        model = ViT_Baseline(patch_size=4, in_channels=1, num_classes=9, embed_dim=128, depth=4, num_heads=4).to(device)
    else:
        raise ValueError(f"unsupported model {args.model}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params} ({total_params/1e6:.4f} M)")
    print(f"Classes: {label_names}")
    print(f"Start training on {device}...\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, val_labels, val_preds = evaluate(model, val_loader, criterion, device)

        # Save log
        save_training_log(
            epoch + 1,
            train_loss, train_metrics,
            val_loss, val_metrics,
            save_path=log_path
        )

        # Print log
        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_metrics['accuracy']:.2%}  F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_metrics['accuracy']:.2%}  F1: {val_metrics['f1']:.4f}"
        )
        print(
            f"  Val infer total: {val_metrics['infer_time_total']:.3f}s | "
            f"Per sample: {val_metrics['infer_time_per_sample']*1000:.4f} ms"
        )

        # Save best model
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.weight_filename))
            print(f"  -> New best model saved, accuracy: {best_acc:.2%}")
        # 每轮保存last权重保底
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_last.pth"))

    total_time = time.time() - start_time
    print(f"\nTraining finished. Best validation accuracy: {best_acc:.2%}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Average time per epoch: {total_time / args.epochs:.2f}s")


if __name__ == "__main__":
    main()
