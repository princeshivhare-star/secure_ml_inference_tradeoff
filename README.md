# Secure vs Non-Secure ML Inference: Performance-Security Trade-off

This project studies the trade-off between inference performance and security in machine learning systems. It compares normal (baseline) inference with a security-hardened inference pipeline and evaluates their latency, memory usage, accuracy, and timing behavior.

## Motivation
Modern ML inference systems often prioritize speed, but execution patterns can leak information through timing behavior. This project explores whether lightweight defenses such as fixed batch padding and dummy compute can reduce leakage at the cost of extra overhead.

## Objectives
- Measure baseline inference latency, memory usage, and accuracy
- Observe timing variation across different input classes
- Implement secure inference defenses
- Compare secure vs non-secure inference overhead
- Visualize performance-security trade-offs

## Tech Stack
- Python
- PyTorch
- torchvision
- pandas
- matplotlib
- NumPy
- psutil

## Project Structure
```bash
secure_ml_inference_tradeoff/
│
├── models/
│   └── mnist_cnn.pth
├── src/
│   ├── model.py
│   ├── inference.py
│   ├── profiler.py
│   ├── defenses.py
│   └── leakage_analyzer.py
├── experiments/
│   ├── train_model.py
│   ├── run_baseline.py
│   ├── run_secure.py
│   ├── classwise_timing.py
│   └── compare_results.py
├── results/
│   ├── csv/
│   └── plots/
├── requirements.txt
└── README.md