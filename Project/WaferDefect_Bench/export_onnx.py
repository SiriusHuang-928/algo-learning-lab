import torch
import argparse
from models import SimpleCNN, ResNet18, EfficientNetB0, ViT_Baseline

device = torch.device("cpu")

def get_model(model_name):
    if model_name == "cnn_baseline":
        return SimpleCNN(num_classes=9)
    elif model_name == "resnet18":
        return ResNet18(num_classes=9)
    elif model_name == "efficientnet_b0":
        return EfficientNetB0(num_classes=9)
    elif model_name == "vit":
        return ViT_Baseline(img_size=24, patch_size=4, num_classes=9)
    else:
        raise ValueError("model not support")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="model.onnx")
    args = parser.parse_args()

    model = get_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy_input = torch.randn(1, 1, 24, 24, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        args.save_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"ONNX saved to {args.save_path}")
