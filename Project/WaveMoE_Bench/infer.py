import os
import argparse
import torch
import numpy as np
import time

from dataset import generate_waveform
from models import CustomTransformerClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="WaveMoE Batch Inference Demo")
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--weight_path', type=str, default='./saved_weights/wavemoe_best.pth')
    parser.add_argument('--seq_len', type=int, default=20)

    # Model config (must match training checkpoint)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--norm_mode', type=str, default='pre')
    parser.add_argument('--ffn_mode', type=str, default='moe')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--expert_dim', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LABEL_NAMES = ["No Peak", "Single Peak", "Double Peak"]

    # Load model
    model = CustomTransformerClassifier(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=args.num_classes,
        norm_mode=args.norm_mode,
        ffn_mode=args.ffn_mode,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        top_k=args.top_k
    ).to(device)

    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.eval()

    print("=" * 55)
    print(f"Offline Batch Inference Demo | Device: {device}")
    print(f"Model mode: {args.ffn_mode}")
    print("=" * 55)

    correct = 0
    with torch.no_grad():
        for idx in range(args.sample_num):
            sig, true_label = generate_waveform(args.seq_len)
            x_tensor = torch.from_numpy(sig).float().unsqueeze(0).to(device)

            logits, _, _ = model(x_tensor)
            pred_label = torch.argmax(logits, dim=1).item()

            if pred_label == true_label:
                correct += 1

            print(
                f"Sample {idx+1:2d} | "
                f"True: {LABEL_NAMES[true_label]:12s} | "
                f"Pred: {LABEL_NAMES[pred_label]}"
            )

    acc = correct / args.sample_num
    print("-" * 55)
    print(f"Total Accuracy: {correct}/{args.sample_num} = {acc:.1%}")
    print("=" * 55)


if __name__ == "__main__":
    main()
