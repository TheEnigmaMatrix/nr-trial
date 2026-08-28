# RL-SDN Controller: Three-Layer AI-Driven SDN Architecture

A production-grade, three-layer Reinforcement Learning driven Software-Defined Networking (SDN) Controller implementing control plane & data plane separation.

## 🌟 Key Features
- **Three-Layer Decoupled System**:
  - **Layer 1: RL Control Plane**: PyTorch Deep Q-Network (DQN / Dueling DQN) agent executing policy evaluation every 10–100ms.
  - **Layer 2: SDN Abstraction Layer**: Uniform forwarding table management and telemetry stats API supporting simulation mode, OpenFlow (Ryu), and P4Runtime gRPC.
  - **Layer 3: Data Plane Forwarding**: High-performance Asyncio-based packet simulator with per-link FIFO queues, queue depth metrics, packet dropping, and Poisson/bursty traffic generators.
- **Baselines & Metrics Comparison**: Benchmarks RL Agent dynamic routing against static OSPF (Dijkstra Shortest Path) and Round-Robin (ECMP / Load Balancing).
- **ONNX Export**: Exports trained PyTorch policy models to ONNX (`model.onnx`) for production C++/edge runtime deployment.
- **Rich Terminal UI**: Live CLI visualizer displaying real-time metrics, per-link queue depths, drop rates, and throughput.

---

## 🛠️ Tech Stack & Dependencies
- **Python**: 3.9+
- **RL & ML**: PyTorch, Gymnasium, ONNX
- **Networking**: NetworkX, PyYAML, Asyncio
- **Terminal UI**: Rich, Matplotlib
- **Testing**: pytest

---

## 🚀 Quick Start Instructions

### 1. Setup Environment
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run RL Training
Train the RL Agent on the multi-path topology defined in `configs/topology.yaml`:
```bash
.venv/bin/python -m rl_sdn_controller.cli.main train --episodes 20 --export-onnx model.onnx
```

### 3. Run Benchmark Comparison (RL vs OSPF vs Round-Robin)
Run parallel evaluation comparing RL against static baselines:
```bash
.venv/bin/python -m rl_sdn_controller.cli.main benchmark --episodes 15
```

### 4. Export Model to ONNX
Export trained policy for C++ inference engines:
```bash
.venv/bin/python -m rl_sdn_controller.cli.main export --output model.onnx
```

### 5. Run Unit & Integration Tests
```bash
.venv/bin/pytest tests/ -v
```
# nr-trial
