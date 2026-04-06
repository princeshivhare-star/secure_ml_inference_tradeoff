import os
import pandas as pd
import psutil


def get_cpu_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return mem_mb


def save_results_to_csv(results, filepath):
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def summarize_latencies(latencies):
    if not latencies:
        return {}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(sorted_vals, p):
        idx = int(p * (len(sorted_vals) - 1))
        return sorted_vals[idx]

    trimmed = latencies_sorted
    if n >= 10:
        low = int(0.05 * n)
        high = int(0.95 * n)
        trimmed = latencies_sorted[low:high] if high > low else latencies_sorted

    return {
        "count": n,
        "avg_latency_sec": sum(latencies_sorted) / n,
        "trimmed_avg_latency_sec": sum(trimmed) / len(trimmed),
        "min_latency_sec": latencies_sorted[0],
        "max_latency_sec": latencies_sorted[-1],
        "p50_latency_sec": percentile(latencies_sorted, 0.50),
        "p95_latency_sec": percentile(latencies_sorted, 0.95),
    }