import torch
import cv2
import argparse
from model import cnn_baseline, resnet18, efficientnet_b0, vit
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'Donut']

def get_model(model_name):
    if model_name == "cnn_baseline":
        return cnn_baseline.CNNBaseline(num_classes=9)
    elif model_name == "resnet18":
        return resnet18.ResNet18(num_classes=9)
    elif model_name == "efficientnet_b0":
        return efficientnet_b0.EfficientNetB0(num_classes=9)
    elif model_name == "vit":
        return vit.ViT_Baseline(img_size=24, patch_size=4, num_classes=9)
    else:
        raise ValueError("model not support")

def infer_single(img_path, model, transform):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (24,24))
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
    return class_names[pred]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()

    model = get_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    pred_cls = infer_single(args.img_path, model, None)
    print(f"Predict Result: {pred_cls}")
