#!/bin/bash
# Complete workflow example

echo "=========================================="
echo "Physics-Informed Graph Mamba - Full Demo"
echo "=========================================="
echo ""

# 1. Generate data
echo "Step 1/5: Generating IEEE 33-bus dataset..."
python scripts/generate_data.py \
    --system ieee33 \
    --num_scenarios 100 \
    --time_steps 288 \
    --parameter_drift \
    --pmu_coverage 0.3 \
    --pv_penetration 0.4

echo ""
echo "✓ Data generation complete!"
echo ""

# 2. Train model
echo "Step 2/5: Training Graph Mamba model..."
python scripts/train.py \
    --config configs/ieee33_config.yaml

echo ""
echo "✓ Training complete!"
echo ""

# 3. Evaluate
echo "Step 3/5: Evaluating on test set..."
python scripts/evaluate.py \
    --checkpoint checkpoints/ieee33/best_model.pt \
    --config configs/ieee33_config.yaml \
    --plot

echo ""
echo "✓ Evaluation complete!"
echo ""

# 4. Robustness tests
echo "Step 4/5: Running robustness tests..."
python scripts/test_robustness.py \
    --checkpoint checkpoints/ieee33/best_model.pt \
    --config configs/ieee33_config.yaml \
    --all \
    --plot

echo ""
echo "✓ Robustness tests complete!"
echo ""

# 5. Summary
echo "Step 5/5: Results summary"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - Dataset: data/raw/ieee33_dataset.pkl"
echo "  - Model: checkpoints/ieee33/best_model.pt"
echo "  - Plots: checkpoints/ieee33/evaluation_plots/"
echo "  - Logs: logs/ieee33/"
echo ""
echo "To view training curves:"
echo "  tensorboard --logdir logs/ieee33"
echo ""
echo "=========================================="
echo "✓ All tasks completed successfully!"
echo "=========================================="
