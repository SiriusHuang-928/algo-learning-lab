import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from dataset import generate_waveform, SEQ_LEN
from model import CustomTransformerClassifier

# ==================== Configuration ====================
WEIGHT_PATH = "./saved_weights/custom_transformer_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["No Peak", "Single Peak", "Double Peak"]


def batch_infer(sample_num: int = 10) -> None:
    """
    Run offline batch inference, print per-sample results and overall accuracy.
    Cross-device compatible with both CPU and GPU.
    """
    # Load model with cross-device compatibility
    model = CustomTransformerClassifier().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()

    print("=" * 55)
    print(f"Offline Batch Inference Demo | Device: {DEVICE}")
    print("=" * 55)

    correct = 0

    with torch.no_grad():
        for idx in range(sample_num):
            # Generate waveform and convert to tensor
            sig, true_label = generate_waveform(SEQ_LEN)
            x_tensor = torch.from_numpy(sig).float().unsqueeze(0).to(DEVICE)

            # Forward pass (ignore attention weights)
            logits, _ = model(x_tensor)
            pred_label = torch.argmax(logits, dim=1).item()

            if pred_label == true_label:
                correct += 1

            # Print readable result
            print(
                f"Sample {idx+1:2d} | "
                f"True: {LABEL_NAMES[true_label]:12s} | "
                f"Pred: {LABEL_NAMES[pred_label]}"
            )

    # Print final summary
    acc = correct / sample_num
    print("-" * 55)
    print(f"Total Accuracy: {correct}/{sample_num} = {acc:.1%}")
    print("=" * 55)


if __name__ == "__main__":
    batch_infer(sample_num=10)
