# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd differentiable-eskf-on-voltage-manifolds-for-power-grid-estimation

# Create conda environment
conda create -n graph-mamba python=3.10
conda activate graph-mamba

# Install dependencies
pip install -r requirements.txt

# Install mamba-ssm (requires CUDA 11.8+)
pip install mamba-ssm
```

### 2. Generate Data

```bash
# Generate IEEE 33-bus dataset (small, fast)
python scripts/generate_data.py --system ieee33 --num_scenarios 100 --parameter_drift

# For full dataset (1000 scenarios, ~30 min)
python scripts/generate_data.py --system ieee33 --num_scenarios 1000 --parameter_drift
```

**Output:** `data/raw/ieee33_dataset.pkl`

### 3. Train Model

```bash
# Train on IEEE 33-bus (GPU recommended)
python scripts/train.py --config configs/ieee33_config.yaml
```

**Expected time:**
- 100 scenarios: ~10 min (GPU) / ~1 hour (CPU)
- 1000 scenarios: ~1 hour (GPU) / ~10 hours (CPU)

**Outputs:**
- Checkpoints: `checkpoints/ieee33/best_model.pt`
- Logs: `logs/ieee33/` (TensorBoard)

### 4. Evaluate

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint checkpoints/ieee33/best_model.pt \
    --config configs/ieee33_config.yaml \
    --plot

# Run robustness tests
python scripts/test_robustness.py \
    --checkpoint checkpoints/ieee33/best_model.pt \
    --config configs/ieee33_config.yaml \
    --all --plot
```

**Outputs:**
- Evaluation plots: `checkpoints/ieee33/evaluation_plots/`
- Robustness plots: `checkpoints/ieee33/robustness_plots/`

### 5. Monitor Training (Optional)

```bash
# In separate terminal
tensorboard --logdir logs/ieee33
```

Open browser: `http://localhost:6006`

---

## 📊 Expected Results

### State Estimation (IEEE 33-bus)
- **Voltage Magnitude RMSE**: < 0.005 p.u.
- **Voltage Angle MAE**: < 0.01 rad

### Parameter Estimation
- **Line Resistance MAE**: < 0.05 Ω/km
- **Line Reactance MAE**: < 0.03 Ω/km

### Robustness
- **60% missing data**: < 0.01 p.u. error
- **3 line outages**: < 0.015 p.u. error
- **20% bad data**: < 0.02 p.u. error

---

## 🔧 Troubleshooting

### Issue: `mamba-ssm` installation fails

**Solution:**
```bash
# Ensure CUDA 11.8+ is installed
nvidia-smi

# Install with specific CUDA version
pip install mamba-ssm --extra-index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory during training

**Solution:** Reduce batch size in config
```yaml
training:
  batch_size: 16  # Change from 32 to 16
```

### Issue: Power flow fails during data generation

**Solution:** This is normal for ~1-2% of time steps. The generator automatically reuses previous valid state.

---

## 📈 Next Steps

1. **Scale to larger system:**
   ```bash
   python scripts/generate_data.py --system ieee118 --num_scenarios 2000
   python scripts/train.py --config configs/ieee118_config.yaml
   ```

2. **Hyperparameter tuning:** Modify `configs/*.yaml`

3. **Custom network:** Extend `PowerGridDataGenerator` for your grid

4. **Ablation studies:** Compare GAT vs GraphSage, Mamba vs LSTM

---

## 📚 Documentation

- **Architecture Details**: See `docs/architecture.md`
- **Physics Constraints**: See `docs/physics.md`
- **API Reference**: See docstrings in source code

---

## 💡 Tips

- **Use GPU**: Training is 10x faster on GPU
- **Start small**: Test with 100 scenarios before full dataset
- **Monitor physics loss**: Should decrease to < 0.01
- **Check voltage range**: Should stay within [0.95, 1.05] p.u.

---

## 🐛 Report Issues

Found a bug? Open an issue on GitHub with:
- Error message
- System info (`python --version`, `nvidia-smi`)
- Config file
- Steps to reproduce
