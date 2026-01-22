# Training Issues Identified and Fixed

## Summary
The training loss was not decreasing due to **multiple critical issues**. All have been fixed except for the underlying data quality problem which requires regenerating the dataset.

## Issues Found

### 1. ✅ FIXED: Dimension Mismatch in Ground Truth
**Problem:** True states had shape `[batch_size, 1, num_nodes]` but model outputs `[batch_size, num_nodes]`
**Impact:** Loss computation was comparing wrong tensor shapes
**Fix:** Added `.squeeze(1)` in `collate_fn` to remove the time dimension

**File:** `src/data/dataloader.py` line 168-171

### 2. ✅ FIXED: Physics Loss Dominating Training
**Problem:** Physics loss was ~5000, completely overwhelming state loss (~0.02) and parameter loss (~1.0)
**Impact:** Model only learned to reduce physics violations, not actual prediction accuracy
**Fix:** Reduced `physics_weight` from `0.1` to `0.0001`

**File:** `configs/ieee33_config.yaml` line 66

### 3. ✅ FIXED: Gradient Explosion in Parameter Head
**Problem:** Gradient ratios up to 19,639x in parameter head, especially final layer
**Impact:** Unstable training, parameters not learning properly
**Fix:**
- Added small initialization (`xavier_uniform` with `gain=0.01`) to final layers
- Changed from `softplus` to `sigmoid` with scaling for parameter outputs
- Constrained parameters to `[0.01, 3.0]` range instead of unbounded

**Files:**
- `src/models/graph_mamba.py` lines 217-222 (ParameterHead init)
- `src/models/graph_mamba.py` lines 274-280 (ParameterHead forward)
- `src/models/graph_mamba.py` lines 170-175 (StateHead init)

### 4. ⚠️ **CRITICAL - NOT FIXED: Data Quality Issues**
**Problem:** Power flow simulations failed for most timesteps, resulting in flat voltage profiles (all buses = 1.0 p.u.)
**Statistics:**
- Out of 19,390 training sequences, ~19,325 (99.7%) have completely flat voltage profiles
- Even "good" samples have very low variation (std < 0.001)
- Timesteps > 10 almost all have v_mag = 1.0 exactly

**Impact:** Model cannot learn meaningful patterns from corrupted/flat data
**Temporary Workaround:** Limited training to first 20 timesteps only, but quality still poor
**Real Fix Needed:** Regenerate dataset with proper power flow settings

**Evidence:**
```python
# Timesteps with zero variation:
t=189-287: all buses exactly 1.0
# Only t=0-4 have reasonable variation (std > 0.001)
```

## Changes Made

### Modified Files:
1. **src/data/dataloader.py**
   - Added dimension squeezing for true_states
   - Added filtering for flat voltage profiles
   - Limited to early timesteps only

2. **src/models/graph_mamba.py**
   - Better initialization for StateHead and ParameterHead final layers
   - Changed parameter output activation from softplus to sigmoid+scaling

3. **configs/ieee33_config.yaml**
   - Reduced physics_weight from 0.1 to 0.0001

## Training Improvements

### Before Fixes:
- Total loss: ~500
- Physics loss: ~5000 (dominating)
- State loss: ~0.02
- Gradient ratios: up to 19,639x
- Target std: 0.0 (incorrect data)

### After Fixes:
- Total loss: ~1.3
- Physics loss: ~0.04 (after weight reduction)
- State loss: ~0.00002
- Gradient ratios: max 389x (much better, one outlier)
- Target std: still ~0.0 (data quality issue remains)

## Recommendations

### URGENT: Regenerate Dataset
The current dataset is unusable for training. The data generation script needs fixes:

1. **Check power flow convergence:** Many timesteps are failing
2. **Increase load diversity:** Current loads may be too uniform
3. **Verify PV profiles:** PV generation might be zeroing out net load
4. **Use realistic load curves:** Current profiles lead to flat voltages
5. **Add more variation:** Voltage should range 0.95-1.05 p.u., not 0.9999-1.0000

### Short-term Workaround
If regeneration is not immediately possible:
1. Use only timesteps 0-5 from each scenario (best variation)
2. Consider synthetic data augmentation
3. Train on measurement reconstruction first, then add parameter estimation

### Model Architecture
Current architecture is sound after fixes. Main recommendations:
1. Consider layer normalization in temporal encoder
2. May need to adjust learning rate once data is fixed
3. Physics weight of 0.0001 is reasonable given current physics loss scale

## Next Steps
1. **Priority 1:** Fix data generation script and regenerate dataset
2. **Priority 2:** Validate new dataset has proper variation (std > 0.001 for v_mag)
3. **Priority 3:** Resume training with fixes already implemented
4. **Priority 4:** Monitor gradient norms and adjust if needed

## Files to Review
- `src/data/data_generator.py` - Check power flow settings and load profiles
- `configs/ieee33_config.yaml` - Verify data generation parameters
- Training logs - Should show decreasing loss once data is fixed
