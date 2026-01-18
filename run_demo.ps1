# Complete workflow example (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Physics-Informed Graph Mamba - Full Demo" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Generate data
Write-Host "Step 1/5: Generating IEEE 33-bus dataset..." -ForegroundColor Yellow
python scripts/generate_data.py `
    --system ieee33 `
    --num_scenarios 100 `
    --time_steps 288 `
    --parameter_drift `
    --pmu_coverage 0.3 `
    --pv_penetration 0.4

Write-Host ""
Write-Host "✓ Data generation complete!" -ForegroundColor Green
Write-Host ""

# 2. Train model
Write-Host "Step 2/5: Training Graph Mamba model..." -ForegroundColor Yellow
python scripts/train.py --config configs/ieee33_config.yaml

Write-Host ""
Write-Host "✓ Training complete!" -ForegroundColor Green
Write-Host ""

# 3. Evaluate
Write-Host "Step 3/5: Evaluating on test set..." -ForegroundColor Yellow
python scripts/evaluate.py `
    --checkpoint checkpoints/ieee33/best_model.pt `
    --config configs/ieee33_config.yaml `
    --plot

Write-Host ""
Write-Host "✓ Evaluation complete!" -ForegroundColor Green
Write-Host ""

# 4. Robustness tests
Write-Host "Step 4/5: Running robustness tests..." -ForegroundColor Yellow
python scripts/test_robustness.py `
    --checkpoint checkpoints/ieee33/best_model.pt `
    --config configs/ieee33_config.yaml `
    --all `
    --plot

Write-Host ""
Write-Host "✓ Robustness tests complete!" -ForegroundColor Green
Write-Host ""

# 5. Summary
Write-Host "Step 5/5: Results summary" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Generated files:"
Write-Host "  - Dataset: data/raw/ieee33_dataset.pkl"
Write-Host "  - Model: checkpoints/ieee33/best_model.pt"
Write-Host "  - Plots: checkpoints/ieee33/evaluation_plots/"
Write-Host "  - Logs: logs/ieee33/"
Write-Host ""
Write-Host "To view training curves:"
Write-Host "  tensorboard --logdir logs/ieee33"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ All tasks completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
