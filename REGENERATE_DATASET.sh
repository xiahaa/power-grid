#!/bin/bash
# Script to regenerate the dataset with fixes
# Run this to create a new, high-quality dataset

echo "=========================================="
echo "Regenerating IEEE 33 Dataset with Fixes"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating graphmamba environment..."
conda activate graphmamba

# Backup old dataset (if exists)
if [ -f "data/raw/ieee33_dataset.pkl" ]; then
    echo "Backing up old dataset..."
    mv data/raw/ieee33_dataset.pkl data/raw/ieee33_dataset_OLD_BROKEN.pkl
    echo "  Old dataset saved as: ieee33_dataset_OLD_BROKEN.pkl"
fi

# Generate new dataset
echo ""
echo "Generating new dataset (100 scenarios, 288 timesteps)..."
echo "This will take a few minutes..."
echo ""

python scripts/generate_data.py \
    --system ieee33 \
    --num_scenarios 100 \
    --time_steps 288 \
    --pmu_coverage 0.3 \
    --pv_penetration 0.4 \
    --parameter_drift \
    --seed 42 \
    --output data/raw

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Dataset regenerated"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Verify data quality:"
    echo "   python scripts/test_fixed_generator.py"
    echo ""
    echo "2. Train model:"
    echo "   python scripts/train.py --config configs/ieee33_config.yaml"
    echo ""
else
    echo ""
    echo "❌ ERROR: Dataset generation failed"
    echo "Check the error messages above"
    echo ""
fi
