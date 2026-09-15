import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. Create 1D array =====================
# Simulate microelectrode raw signal
sig = np.array([0.2, 0.5, 0.9, 0.7, 0.3, 0.1, 0.4, 0.8])
print("Original 1D signal sig =", sig)
print("Array length:", len(sig))
print("-" * 60)

# ===================== 2. Array slicing =====================
sig_start3 = sig[3:]       # from index 3 to end
sig_mid = sig[2:6]         # index 2 ~ 5 (left-closed right-open)
sig_step2 = sig[::2]       # step = 2
sig_rev = sig[::-1]        # reverse array
print("slice sig[3:]  :", sig_start3)
print("slice sig[2:6] :", sig_mid)
print("slice sig[::2] :", sig_step2)
print("reverse sig[::-1]:", sig_rev)
print("-" * 60)

# ===================== 3. Concatenate arrays =====================
sig1 = np.array([0.1, 0.2, 0.3])
sig2 = np.array([0.7, 0.8, 0.9])
sig_concat = np.concatenate([sig1, sig2])
print("sig1 =", sig1)
print("sig2 =", sig2)
print("Concatenated array:", sig_concat)
print("-" * 60)

# ===================== 4. Z-score normalization =====================
# formula: x_norm = (x - mean) / std
mean_val = np.mean(sig)
std_val = np.std(sig)
sig_norm = (sig - mean_val) / std_val
print(f"Signal mean = {mean_val:.4f}")
print(f"Signal standard deviation = {std_val:.4f}")
print("Normalized signal:", np.round(sig_norm, 4))
print("-" * 60)

# ===================== 5. Add Gaussian noise =====================
np.random.seed(42)  # fixed random seed for reproducibility
noise = np.random.normal(loc=0, scale=0.08, size=len(sig))
sig_noisy = sig + noise

print("Gaussian noise =", np.round(noise, 3))
print("Noisy sensor signal =", np.round(sig_noisy, 3))

# Plot
plt.figure(figsize=(10,4))
plt.plot(sig, label="Raw Signal", linewidth=2)
plt.plot(sig_noisy, label="Signal with Gaussian Noise", linestyle="--")
plt.legend()
plt.title("1D Sensor Signal + Gaussian Noise Demo")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()
