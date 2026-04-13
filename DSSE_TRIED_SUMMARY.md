# IEEE123 DSSE: What Has Been Tried

## Goal

Improve DSSE performance on the IEEE123 dataset using the GraphMamba / multi-rate Mamba pipeline, with attention to:

- sparse measurements
- temporal variation
- topology changes
- voltage magnitude accuracy (`v_mag_rmse`)

---

## 1. Initial direction: joint estimation path

### Tried
- Audited the repository and repaired the active `v2` training/evaluation path.
- Re-ran the older joint state + line-parameter estimation route.

### Outcome
- The codebase had drifted away from pure DSSE.
- Joint estimation was not the right direction for the current goal.
- The user then explicitly pivoted to **DSSE-only**.

---

## 2. DSSE-only pivot

### Tried
Implemented DSSE-only support across the pipeline:
- disabled line-parameter estimation
- set `parameter_weight: 0.0`
- updated training, loss, config, evaluation, and tests
- added PMU angle `v_ang` as an input feature

### Outcome
- DSSE-only training/evaluation worked.
- A full DSSE-only GraphMamba retrain was completed.
- Fair comparison against saved baselines showed GraphMamba underperforming older WLS/GNN/LSTM/Transformer baselines.

Artifacts:
- `results/ieee123_dsse_final_test_eval.json`
- `results/baseline_test_comparison.json`

---

## 3. Early ablations

### Tried
1. **PMU angle input ablation**
2. **No-physics ablation**
3. Smoke-data runs to iterate faster
4. Fix for a staged-training early-stopping bug

### Outcome
- Some runs looked promising at first.
- The promising no-physics result did **not** reproduce cleanly.
- These ablations did not produce a reliable improvement plan.

Artifacts:
- `results/ieee123_dsse_nophysics_test_eval.json`
- `results/ieee123_smoke_dsse_eval.json`
- `results/ieee123_smoke_nophysics_eval.json`

---

## 4. Diagnosis-first phase

### Tried
Added diagnosis tooling and inspected the pipeline directly.

Key checks:
- topology feature flow
- physics/runtime config wiring
- feeder/substation hierarchical physics activation
- slack-bus handling
- checkpoint compatibility
- per-feeder and per-bus error concentration

### Main findings
1. **Topology features were effectively dead** under the active trainer path.
2. Several **physics config fields were not actually used at runtime**.
3. **Hierarchical feeder/substation physics** was present in code but not contributing correctly.
4. **Slack bus** handling was missing.
5. Physics loss was numerically too weak.
6. Remaining voltage error looked feeder-local.

Artifact:
- `results/ieee123_dsse_pipeline_diagnosis.json`

---

## 5. Structural repairs

### Tried
Implemented/fixed:
- topology feature computation from measurement deltas
- full physics config/runtime alignment
- hierarchical feeder/substation physics activation using measurement-derived `p_bus/q_bus`
- `slack_bus` support
- node-count normalization of bus physics loss
- stronger `physics_weight`
- legacy checkpoint compatibility for 3-input to 4-input migration

### Outcome
- Structural warnings were removed.
- Physics became active and configurable.
- Small `avg_loss` improvements appeared, but these changes alone did not solve `v_mag_rmse`.

Artifacts:
- `results/ieee123_dsse_physics_ft_pilot_base_test.json`
- `results/ieee123_dsse_physics_ft_pilot_epoch4_test.json`

---

## 6. Feeder-local mitigation attempts

### Tried
1. **Feeder-weighted state loss**
2. **Feeder-conditioned `v_mag` residual refinement**
3. **Measurement-aware feeder refinement** using PMU/SCADA/combined coverage features
4. Restricting refinement to problematic feeders `1–4`

### Outcome before masking fix
These experiments typically improved:
- `avg_loss`
- `v_ang_rmse`
- `v_mag_mae`

But they did **not** reliably improve the main target:
- `v_mag_rmse`

Artifacts:
- `results/ieee123_dsse_feeder_weighted_pilot_base_test.json`
- `results/ieee123_dsse_feeder_weighted_pilot_epoch4_test.json`
- `results/ieee123_dsse_feeder_refine_pilot_base_test.json`
- `results/ieee123_dsse_feeder_refine_pilot_epoch4_test.json`
- `results/ieee123_dsse_feeder_refine_pilot_v2_base_test.json`
- `results/ieee123_dsse_feeder_refine_pilot_v2_epoch8_test.json`

---

## 7. Critical discovery: invalid target nodes were corrupting training/evaluation

### Tried
Inspected bus-level error structure and raw dataset content.

### Finding
A small set of node indices in the IEEE123 tensor were invalid/disconnected/padded in the active dataset representation:
- `22`
- `43`
- `64`
- `85`
- `124`

Observed behavior:
- zero graph degree
- NaN targets in raw data
- later converted to zero by the dataloader
- produced repeated ~`0.92` voltage errors
- heavily distorted `v_mag_rmse`

This changed the interpretation of nearly all earlier DSSE results.

---

## 8. Valid-target masking fix

### Tried
Implemented end-to-end valid-target masking:
- preserve target-validity masks in the dataloader
- exclude invalid target nodes in `state_loss()`
- mask validation aggregation
- mask evaluation aggregation
- make feeder/per-bus/physics metrics NaN-safe
- add focused tests

### Files changed
- `src/data/multi_rate_dataloader.py`
- `src/train/loss_v2.py`
- `src/train/trainer_v2.py`
- `scripts/evaluate_v2.py`
- `src/evaluation/metrics.py`
- `tests/test_dsse_phase2.py`

### Outcome
- Focused regression tests passed.
- Recomputed metrics on valid nodes only were dramatically lower and more realistic.

Masked evaluation artifacts:
- `results/ieee123_dsse_feeder_refine_pilot_v2_base_test_masked.json`
- `results/ieee123_dsse_feeder_refine_pilot_v2_epoch8_test_masked.json`
- `results/ieee123_dsse_feeder_refine_pilot_v2_best_test_masked.json`

---

## 9. Retraining after masking fix

### Tried
Re-ran the targeted feeder-refinement pilot with the corrected valid-node masking in the DSSE loss.

### Outcome
This produced the current best GraphMamba result on the masked test split:

- `v_mag_rmse`: **0.009819**
- `v_mag_mae`: **0.007928**
- `v_ang_rmse`: **0.003111**
- `voltage_violation_rate`: **0.0**

Best artifact:
- `results/ieee123_dsse_feeder_refine_pilot_v2_best_test_masked.json`

Checkpoint:
- `checkpoints/ieee123_dsse_feeder_refine_pilot_v2/best_model.pt`

---

## 10. Fair rerun of baselines under the same masking

### Tried
Re-ran the saved baselines on the same capped masked test split.

### Masked test results

| Model | `v_mag_rmse` | `v_mag_mae` | `v_ang_rmse` |
|---|---:|---:|---:|
| WLS | 0.005782 | 0.003796 | 0.002011 |
| LSTM | 0.005730 | 0.003753 | 0.001872 |
| Transformer | 0.005807 | 0.003784 | 0.001969 |
| GraphMamba | 0.009819 | 0.007928 | 0.003111 |
| GNN | 0.016217 | 0.009094 | 0.002868 |

Artifacts:
- `results/baseline_test_comparison_masked.json`
- `results/wls_baseline_test_eval_masked.json`
- `results/gnn_baseline_test_eval_masked.json`
- `results/lstm_baseline_test_eval_masked.json`
- `results/transformer_baseline_test_eval_masked.json`

### Current conclusion
After correcting invalid-node handling:
- GraphMamba is **much better than previously thought**.
- But it still **does not beat WLS / LSTM / Transformer** on the current masked IEEE123 test split.
- It does beat the saved GNN baseline.

---

## 11. What has been tried, in short

### Tried and kept
- DSSE-only mode
- PMU angle input
- topology feature repair
- physics/runtime repair
- slack-bus support
- hierarchical physics activation
- legacy checkpoint compatibility
- feeder-conditioned voltage refinement
- measurement-aware feeder refinement
- valid-target masking fix

### Tried but not sufficient alone
- no-physics ablation
- generic stronger physics weighting
- feeder-weighted loss only
- feeder-ID-only refinement

### Main lesson
The biggest recent improvement came not from a new architecture trick, but from fixing a **data/target validity bug** that had been corrupting both optimization and evaluation.

---

## 12. Current status

### Best current GraphMamba result
- file: `results/ieee123_dsse_feeder_refine_pilot_v2_best_test_masked.json`
- checkpoint: `checkpoints/ieee123_dsse_feeder_refine_pilot_v2/best_model.pt`

### Current standing
- better than GNN
- worse than WLS, LSTM, Transformer on masked valid-node evaluation

### Most likely next direction
Investigate why simpler temporal baselines still beat GraphMamba, likely focusing on:
- whether graph/spatial blocks are overcomplicating a mostly temporal/smooth target
- whether the current staged fine-tune is too conservative
- whether the GraphMamba objective should more directly emphasize valid-node `v_mag` accuracy
