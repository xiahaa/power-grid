# Physics-Informed Graph Mamba for Power Grid State & Parameter Estimation

A novel deep learning framework for **real-time joint state and parameter estimation** in unobservable distribution networks using Graph Mamba with physics constraints.

## 🌟 Key Features

- **Spatial-Temporal Decoupled Architecture**: Combines Graph Attention Networks (GAT) for spatial encoding with Mamba blocks for temporal dynamics
- **Dual-Head Output**: Simultaneous estimation of states (voltage, angle) and parameters (line resistance, reactance)
- **Physics-Informed**: Differentiable optimization layer enforcing power flow equations as hard constraints
- **Robustness**: Handles sparse measurements, topology changes, and missing data
- **Scalability**: Efficient on large systems (IEEE 118-bus+) thanks to Mamba's linear complexity

## 🏗️ Architecture

```
Input: Sparse Measurements Z_t (P, Q, V)
   ↓
Spatial Encoder (GAT/GraphSage) → Extract spatial features H_t^spatial
   ↓
Temporal Core (Mamba Block) → Capture long-term dependencies
   ↓
Dual Heads:
   ├─→ State Head: V_t, θ_t (voltage magnitude, angle)
   └─→ Parameter Head: R_ij, X_ij (line impedance)
   ↓
Physics Projector: Enforce KCL/KVL via differentiable optimization
```

## 📦 Installation

```bash
# Create conda environment
conda create -n graph-mamba python=3.10
conda activate graph-mamba

# Install dependencies
pip install -r requirements.txt

# Install mamba-ssm (requires CUDA)
pip install mamba-ssm
```

## 🚀 Quick Start

### 1. Generate Training Data

```bash
# Generate IEEE 33-bus data with parameter drift
python scripts/generate_data.py --system ieee33 --hours 24 --parameter_drift

# Generate IEEE 118-bus data
python scripts/generate_data.py --system ieee118 --hours 48
```

### 2. Train Model

```bash
# Train on IEEE 33-bus
python scripts/train.py --config configs/ieee33_config.yaml

# Train on IEEE 118-bus with physics constraints
python scripts/train.py --config configs/ieee118_config.yaml --physics_weight 0.1
```

### 3. Evaluate & Test

```bash
# Standard evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Robustness test (missing measurements, topology changes)
python scripts/test_robustness.py --checkpoint checkpoints/best_model.pt
```

## 📊 Datasets

- **IEEE 33-bus**: Distribution system with DERs (PV, storage)
- **IEEE 118-bus**: Transmission system for scalability testing
- **Dynamic scenarios**:
  - Parameter drift (line aging, temperature effects)
  - Sparse PMU placement (20-40% coverage)
  - Topology changes (line outages)

## 🧪 Experiments

Comprehensive benchmarks against:
- Classical EKF (Extended Kalman Filter)
- WLS (Weighted Least Squares)
- LSTM-based methods
- GNN-only baselines

Metrics:
- **State estimation**: RMSE of voltage magnitude/angle
- **Parameter estimation**: MAE of line impedance
- **Robustness**: Performance under missing data/topology changes
- **Speed**: Inference time on large grids

## 📁 Project Structure

```
.
├── data/                   # Generated datasets
├── configs/                # Configuration files
├── src/
│   ├── models/            # Graph Mamba architecture
│   ├── data/              # Data generation & loading
│   ├── physics/           # Power flow constraints
│   ├── utils/             # Utilities
│   └── train/             # Training logic
├── scripts/               # Executable scripts
├── notebooks/             # Jupyter notebooks for analysis
└── tests/                 # Unit tests
```

## 📖 Citation

If you use this code, please cite:

```bibtex
@article{yourlastname2026graphmamba,
  title={Real-Time Joint State and Parameter Estimation in Unobservable Distribution Networks: A Physics-Informed Graph Mamba Approach},
  author={Your Name},
  journal={IEEE Transactions on Power Systems},
  year={2026}
}
```

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- Pandapower
- mamba-ssm (requires CUDA 11.8+)
- NumPy, SciPy, Pandas

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions, contact: [your-email@example.com]
