import argparse
import time
import numpy as np
import onnxruntime as ort
from dataset import generate_waveform


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Runtime inference demo & benchmark")
    parser.add_argument('--onnx_path', type=str, default='./saved_weights/wavemoe.onnx')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--test_samples', type=int, default=200)
    parser.add_argument('--warmup_samples', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    LABEL_NAMES = ["No Peak", "Single Peak", "Double Peak"]

    # 1. Initialize ONNX Runtime session
    session = ort.InferenceSession(
        args.onnx_path,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    print("=" * 65)
    print("ONNX Runtime Inference Benchmark")
    print(f"Model path: {args.onnx_path}")
    print(f"Test samples: {args.test_samples} | Warmup: {args.warmup_samples}")
    print("=" * 65)

    # 2. Warmup (eliminate initialization overhead)
    print("\n[Warmup] Running warmup samples...")
    for _ in range(args.warmup_samples):
        dummy = np.random.randn(1, args.seq_len).astype(np.float32)
        session.run(output_names, {input_name: dummy})

    # 3. Formal benchmark
    print("\n[Benchmark] Running inference test...")
    correct = 0
    all_times = []

    for i in range(args.test_samples):
        # Generate test sample
        signal, true_label = generate_waveform(args.seq_len)
        input_data = signal.reshape(1, -1).astype(np.float32)

        # Inference & timing
        t_start = time.perf_counter()
        outputs = session.run(output_names, {input_name: input_data})
        t_end = time.perf_counter()

        # Get classification result (first output = logits)
        logits = outputs[0]
        pred_label = np.argmax(logits, axis=1)[0]
        if pred_label == true_label:
            correct += 1

        all_times.append(t_end - t_start)

    # 4. Calculate metrics
    total_time = sum(all_times)
    avg_time_ms = np.mean(all_times) * 1000
    p95_time_ms = np.percentile(all_times, 95) * 1000
    accuracy = correct / args.test_samples
    throughput = args.test_samples / total_time

    # 5. Print results
    print("\n" + "=" * 65)
    print(f"Accuracy:          {accuracy:.2%}")
    print(f"Total time:        {total_time:.3f} s")
    print(f"Avg per sample:    {avg_time_ms:.3f} ms")
    print(f"P95 latency:       {p95_time_ms:.3f} ms")
    print(f"Throughput:        {throughput:.2f} samples/s")
    print("=" * 65)


if __name__ == "__main__":
    main()
