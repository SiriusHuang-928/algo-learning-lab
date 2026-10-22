import argparse
import os
import torch
from models import CustomTransformerClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")

    # Model checkpoint & output path
    parser.add_argument('--weight_path', type=str, default='./saved_weights/moe_4expert.pth')
    parser.add_argument('--output_path', type=str, default='./saved_weights/wavemoe.onnx')

    # Model architecture (must match training config)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--norm_mode', type=str, default='pre', choices=['pre', 'post'])
    parser.add_argument('--ffn_mode', type=str, default='moe', choices=['relu', 'swiglu', 'moe'])
    parser.add_argument('--window_size', type=int, default=None)

    # MoE specific params
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--expert_dim', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")  # ONNX导出统一用CPU，兼容性更好

    # 1. Build model architecture
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

    # 2. Load trained weights
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.weight_path}")
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.eval()

    # 3. Create dummy input (fixed sequence length, dynamic batch)
    dummy_input = torch.randn(1, args.seq_len, device=device)

    # 4. Export ONNX
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['waveform_input'],
        output_names=['logits', 'attention_weights', 'aux_loss'],
        dynamic_axes={
            'waveform_input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'attention_weights': {0: 'batch_size'},
        }
    )

    # 5. Print summary
    file_size = os.path.getsize(args.output_path) / 1024
    print(f"✅ Model exported successfully!")
    print(f"   Output file: {args.output_path}")
    print(f"   File size: {file_size:.2f} KB")
    print(f"   FFN mode: {args.ffn_mode}")
    print(f"   Input shape: [batch_size, {args.seq_len}]")
    print(f"   Output: logits, attention_weights, aux_loss")


if __name__ == "__main__":
    main()
