import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    baseline = pd.read_csv("results/csv/baseline_results.csv")
    secure = pd.read_csv("results/csv/secure_results.csv")

    combined = pd.concat([baseline, secure], ignore_index=True)

    print("\n=== Combined Results ===")
    print(combined)

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(combined["mode"], combined["trimmed_avg_latency_sec"])
    plt.title("Baseline vs Secure Inference Latency (Trimmed Avg)")
    plt.xlabel("Mode")
    plt.ylabel("Latency (sec)")
    plt.tight_layout()
    plt.savefig("results/plots/baseline_vs_secure_latency.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(combined["mode"], combined["memory_mb"])
    plt.title("Baseline vs Secure Memory Usage")
    plt.xlabel("Mode")
    plt.ylabel("Memory (MB)")
    plt.tight_layout()
    plt.savefig("results/plots/baseline_vs_secure_memory.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(combined["mode"], combined["accuracy"])
    plt.title("Baseline vs Secure Accuracy")
    plt.xlabel("Mode")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("results/plots/baseline_vs_secure_accuracy.png")
    plt.close()

    baseline_latency = combined.loc[combined["mode"] == "baseline", "trimmed_avg_latency_sec"].values[0]
    secure_latency = combined.loc[combined["mode"] == "secure", "trimmed_avg_latency_sec"].values[0]

    overhead = ((secure_latency - baseline_latency) / baseline_latency) * 100

    print(f"\nTrimmed latency overhead of secure inference: {overhead:.2f}%")

    baseline_p50 = combined.loc[combined["mode"] == "baseline", "p50_latency_sec"].values[0]
    secure_p50 = combined.loc[combined["mode"] == "secure", "p50_latency_sec"].values[0]
    p50_overhead = ((secure_p50 - baseline_p50) / baseline_p50) * 100

    print(f"P50 latency overhead of secure inference: {p50_overhead:.2f}%")

    baseline_acc = combined.loc[combined["mode"] == "baseline", "accuracy"].values[0]
    secure_acc = combined.loc[combined["mode"] == "secure", "accuracy"].values[0]

    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Secure accuracy: {secure_acc:.4f}")


if __name__ == "__main__":
    main()