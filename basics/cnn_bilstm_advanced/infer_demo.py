import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from dataset import generate_waveform, SEQ_LEN, NUM_CLASSES
from model import CNNBiLSTM

WEIGHT_PATH = "./saved_weights/cnn_bilstm_best.pth"

def batch_infer_test(n=10):
    model = CNNBiLSTM()
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.eval()
    correct = 0
    print("===== Offline Batch Inference Demo =====")
    with torch.no_grad():
        for _ in range(n):
            sig, true_lab = generate_waveform(SEQ_LEN)
            tensor = torch.from_numpy(sig).unsqueeze(0).unsqueeze(1)
            logit = model(tensor)
            pred = torch.argmax(logit, dim=1).item()
            if pred == true_lab:
                correct +=1
            print(f"True:{true_lab} | Pred:{pred}")
    print(f"\nInfer Acc: {correct/n:.2f}")

if __name__ == "__main__":
    batch_infer_test(10)
