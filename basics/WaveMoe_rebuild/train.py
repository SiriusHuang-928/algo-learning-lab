import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from dataset import build_dataset
from models import CustomTransformerClassifier
from utils.metrics import calculate_classification_metrics
from utils.visualize import plot_learning_curves, plot_confusion_matrix, plot_attention_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="WaveMoE Training Pipeline")

    # Dataset config
    parser.add_argument('--total_samples', type=int, default=10000)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seq_len', type=int, default=20)

    # Model config
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--norm_mode', type=str, default='pre', choices=['pre', 'post'])
    parser.add_argument('--ffn_mode', type=str, default='moe', choices=['relu', 'swiglu', 'moe'])
    parser.add_argument('--window_size', type=int, default=None)

    # MoE config
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--expert_dim', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)

    # Training config
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    # Path config
    parser.add_argument('--save_dir', type=str, default='./saved_weights')
    parser.add_argument('--figure_dir', type=str, default='./output_figures')
    parser.add_argument('--weight_filename', type=str, default='wavemoe_best.pth')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, aux_loss_weight):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, _, total_aux_loss = model(x_batch)

        main_loss = criterion(logits, y_batch)
        loss = main_loss + aux_loss_weight * total_aux_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    metrics = calculate_classification_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, aux_loss_weight):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits, _, total_aux_loss = model(x_batch)
        main_loss = criterion(logits, y_batch)
        loss = main_loss + aux_loss_weight * total_aux_loss

        total_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        total += x_batch.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    metrics = calculate_classification_metrics(all_labels, all_preds)
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    # Build dataset
    print("Building dataset...")
    signals, labels = build_dataset(args.total_samples)
    split_idx = int(args.total_samples * args.train_ratio)

    train_set = TensorDataset(signals[:split_idx], labels[:split_idx])
    val_set = TensorDataset(signals[split_idx:], labels[split_idx:])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )

    # Initialize model
    model = CustomTransformerClassifier(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=args.num_classes,
        dropout=args.dropout,
        norm_mode=args.norm_mode,
        ffn_mode=args.ffn_mode,
        window_size=args.window_size,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        top_k=args.top_k
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nStart training on {device}...")
    print(f"FFN mode: {args.ffn_mode}")
    if args.ffn_mode == "moe":
        print(f"Experts: {args.num_experts}, Top-k: {args.top_k}, Aux loss weight: {args.aux_loss_weight}")

    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, args.aux_loss_weight)
        val_loss, val_metrics, val_labels, val_preds = evaluate(model, val_loader, criterion, device, args.aux_loss_weight)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_metrics['accuracy']:.2%} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_metrics['accuracy']:.2%}"
        )

        # Print expert usage for MoE mode
        if args.ffn_mode == "moe":
            expert_usage = model.encoder_layers[0].ffn.last_expert_usage
            print(f"  Expert usage: {expert_usage.cpu().numpy().round(3)}")

        # Save best checkpoint
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.weight_filename))
            print(f"  -> New best model saved, accuracy: {best_acc:.2%}")

    print(f"\nTraining finished. Best validation accuracy: {best_acc:.2%}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, args.figure_dir)
    plot_confusion_matrix(val_labels, val_preds, ["No Peak", "Single Peak", "Double Peak"], args.figure_dir)

    # Load best model for attention visualization
    model.load_state_dict(torch.load(os.path.join(args.save_dir, args.weight_filename), map_location=device))
    plot_attention_visualization(model, args.figure_dir, args.seq_len)

    print(f"All results saved to {args.figure_dir}/")


if __name__ == "__main__":
    main()
