import torch
from dataset import generate_waveform, SEQ_LEN
from model import TransformerTimeSeries

WEIGHT_PATH = "./saved_weights/transformer_best.pth"

def batch_infer(sample_num=10):
    model = TransformerTimeSeries()
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.eval()
    print("===== Offline Batch Inference Demo =====")
    with torch.no_grad():
        for _ in range(sample_num):
            sig, true_label = generate_waveform(SEQ_LEN)
            x_tensor = torch.from_numpy(sig).unsqueeze(0)
            logits = model(x_tensor)
            pred_label = torch.argmax(logits, dim=1).item()
            print(f"True:{true_label} | Pred:{pred_label}")

if __name__ == "__main__":
    batch_infer(10)
