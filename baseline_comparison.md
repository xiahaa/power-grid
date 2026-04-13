# IEEE123 DSSE Comparison Summary

## Final GraphMamba / Multi-Rate Mamba results

Best checkpoint:
- `checkpoints/ieee123_dsse/best_model.pt`
- Training summary from the latest run: best validation loss `0.068328` at epoch `8`

Evaluation outputs:
- Test: `results/ieee123_dsse_final_test_eval.json`
- Validation: `results/ieee123_dsse_final_val_eval.json`
- Earlier DSSE result snapshot: `results/ieee123_dsse_eval.json`

### Final best checkpoint metrics

| Split | v_mag_rmse | v_mag_mae | v_ang_rmse | v_ang_mae | avg_loss |
|---|---:|---:|---:|---:|---:|
| Test | 0.189001 | 0.072502 | 0.005630 | 0.003131 | 0.036567 |
| Val  | 0.188839 | 0.071589 | 0.005614 | 0.003077 | 0.036505 |

## Comparison with earlier DSSE result file

Earlier DSSE snapshot (`results/ieee123_dsse_eval.json`) vs final best checkpoint on test:

| Metric | Earlier DSSE | Final best DSSE | Change |
|---|---:|---:|---:|
| v_mag_rmse | 0.183542 | 0.189001 | worse |
| v_mag_mae | 0.091089 | 0.072502 | better |
| v_ang_rmse | 0.004927 | 0.005630 | worse |
| v_ang_mae | 0.002954 | 0.003131 | slightly worse |
| avg_loss | 0.048172 | 0.036567 | better |
| voltage_violation_rate | 1.0000 | 0.9600 | better |
| voltage_max_deviation | 0.104222 | 0.090435 | better |
| v_mag_p95_error | 0.120930 | 0.054401 | much better |
| v_ang_p95_error | 0.010073 | 0.005544 | much better |

Interpretation:
- The latest checkpoint improved typical-case performance (`MAE`, `p95`, physics violations).
- `RMSE` did not improve, which indicates a smaller number of large-error windows still dominate the squared-error metric.

## Comparison with learned baselines

### Same-test-split comparison

Saved baseline checkpoints were evaluated on the same IEEE123 **test** split as the final DSSE checkpoint.

Generated files:
- `results/wls_baseline_test_eval.json`
- `results/gnn_baseline_test_eval.json`
- `results/lstm_baseline_test_eval.json`
- `results/transformer_baseline_test_eval.json`
- `results/baseline_test_comparison.json`

| Model | v_mag_rmse | v_mag_mae | v_ang_rmse | v_ang_mae |
|---|---:|---:|---:|---:|
| DSSE GraphMamba | 0.189001 | 0.072502 | 0.005630 | 0.003131 |
| WLS baseline | 0.170065 | 0.037062 | 0.001810 | 0.001163 |
| GNN baseline | 0.170710 | 0.041749 | 0.003128 | 0.001893 |
| LSTM baseline | 0.170067 | 0.037068 | 0.001714 | 0.001159 |
| Transformer baseline | 0.170069 | 0.037058 | 0.001805 | 0.001182 |

Same-test relative gap:
- vs WLS: `v_mag_mae` is `1.96x` higher, `v_ang_mae` is `2.69x` higher
- vs GNN: `v_mag_mae` is `1.74x` higher, `v_ang_mae` is `1.65x` higher
- vs LSTM: `v_mag_mae` is `1.96x` higher, `v_ang_mae` is `2.70x` higher
- vs Transformer: `v_mag_mae` is `1.96x` higher, `v_ang_mae` is `2.65x` higher

All saved baselines remain stronger than the current DSSE GraphMamba checkpoint on the matched test split.

### Note on baseline validation summary

The older validation-only summary is still available in `checkpoints/baselines/all_results.json`, but the table above is the fairer comparison because it uses the exact same test split as the DSSE checkpoint.

## Error pattern observations

Per-feeder test RMSE from the final checkpoint:

| Feeder | v_mag_rmse | v_ang_rmse |
|---|---:|---:|
| 0 | 0.043133 | 0.002842 |
| 1 | 0.211456 | 0.005976 |
| 2 | 0.210206 | 0.005767 |
| 3 | 0.210016 | 0.005651 |
| 4 | 0.209711 | 0.005649 |
| 5 | 0.039128 | 0.002342 |

This strongly suggests that the dominant voltage-magnitude error comes from feeders `1-4`, while feeders `0` and `5` are much easier.

Topology-change coverage in the test split is not dominant:
- sequence windows in test split: `868`
- windows containing topology changes: `150`
- rate: `17.3%`

This means topology changes are a plausible source of outliers, but they are not frequent enough by themselves to explain the full accuracy gap without an additional sensitivity problem.

## Current working hypothesis

The final DSSE GraphMamba model appears to reduce error for the bulk of windows, but still produces a relatively small set of large feeder-localized mistakes. That combination naturally improves `MAE` and `p95` while leaving `RMSE` flat or worse.

Most plausible causes:
1. The multi-rate fusion path is helping typical cases but is not robust on a subset of harder feeder conditions.
2. A minority of windows, possibly including topology-change windows, still create large voltage-magnitude misses on feeders `1-4`.
3. Early convergence at epoch `8` suggests the current objective/architecture reaches a plateau quickly and does not close the gap to the simpler learned baselines.

## Recommended next experiments

1. **Target the RMSE tail directly**
	- add per-window error logging and inspect the worst `1%` test windows
	- compare whether those windows align with topology changes, missing PMU coverage, or specific feeder regimes

2. **Ablate fusion and physics terms**
	- train without physics penalty
	- train with simpler SCADA-only / PMU-only / no-cross-attention variants
	- check whether the current fusion block helps mean error but hurts robustness

3. **Rebalance the objective toward hard cases**
	- test Huber loss or a mixed `MAE + RMSE` objective for state estimation
	- consider feeder-balanced weighting if feeders `1-4` dominate the tail

4. **Investigate why baselines share the same worst buses**
	- the saved baselines all show very large magnitude errors on buses `22`, `43`, `64`, `85`, and `124`
	- determine whether these are consistently weakly observed buses, data artifacts, or normalization edge cases

## First improvement-pass smoke ablation

After wiring PMU angle into the model input and adding a no-physics ablation path, a smoke checkpoint was evaluated using:

- config: `configs/ieee123_smoke_nophysics.yaml`
- checkpoint: `checkpoints/ieee123_smoke_nophysics/best_model.pt`
- output: `results/ieee123_smoke_nophysics_eval.json`

Smoke-run fix:
- the earlier apparent "stall after epoch 1" was not a staged-trainer deadlock
- the root cause was that the smoke config still expanded into `9730` train sequences and `1217` train batches per epoch
- smoke configs now cap split sizes, so the smoke ablation runs as an actual short sanity check

Completed smoke checkpoint note:
- the rerun now completes all `3` smoke epochs
- best smoke ablation checkpoint is from epoch `3`
- best validation loss: `0.058487`

### Smoke comparison vs earlier smoke DSSE

| Metric | Earlier smoke DSSE | PMU-angle + no-physics smoke | Change |
|---|---:|---:|---:|
| v_mag_rmse | 0.117732 | 0.089285 | better |
| v_mag_mae | 0.093148 | 0.039010 | much better |
| v_ang_rmse | 0.004786 | 0.004883 | slightly worse |
| v_ang_mae | 0.003725 | 0.003921 | slightly worse |
| voltage_violation_rate | 1.000000 | 1.000000 | unchanged |
| voltage_max_deviation | 0.160533 | 0.124436 | better |
| v_mag_p95_error | 0.122819 | 0.052327 | much better |

Interpretation:
- The first ablation signal is positive for the main target metric, `v_mag_rmse`.
- Most of the gain appears on voltage magnitude rather than angle.
- The completed smoke rerun still supports the same direction, although the gain is smaller than the earlier epoch-1 snapshot suggested.
- The next meaningful run is the same ablation on the full IEEE123 DSSE config, then a like-for-like comparison against the current full DSSE checkpoint.

## Full PMU-angle + no-physics ablation

The same ablation was then rerun on the full IEEE123 DSSE config after fixing staged early stopping so that patience resets per stage:

- config: `configs/ieee123_dsse_nophysics.yaml`
- checkpoint: `checkpoints/ieee123_dsse_nophysics/best_model.pt`
- output: `results/ieee123_dsse_nophysics_test_eval.json`

Run summary:
- training completed in `5.74` hours
- best validation loss: `0.067521`
- best checkpoint epoch: `6`
- total executed epochs: `93`

### Comparison vs current final DSSE test result

| Metric | Final DSSE | No-physics ablation | Change |
|---|---:|---:|---:|
| v_mag_rmse | 0.189001 | 0.189069 | slightly worse |
| v_mag_mae | 0.072502 | 0.070313 | better |
| v_mag_p95_error | 0.054401 | 0.053599 | slightly better |
| v_ang_rmse | 0.005630 | 0.005231 | better |
| v_ang_mae | 0.003131 | 0.003063 | slightly better |
| v_ang_p95_error | 0.005544 | 0.009993 | worse |
| voltage_violation_rate | 0.960000 | 0.999811 | worse |
| voltage_max_deviation | 0.090435 | 0.099977 | worse |
| avg_loss | 0.036567 | 0.035764 | slightly better |

Interpretation:
- After fixing the trainer bug and rerunning, the earlier strong no-physics gain did **not** reproduce.
- The rerun is roughly neutral on average magnitude error, modestly better on angle RMSE/MAE, and slightly better on `avg_loss`.
- But voltage-violation metrics are worse, and `v_ang_p95_error` is clearly worse.
- The earlier optimistic full-ablation result should therefore be treated as invalidated by the trainer bug.
- Current conclusion: PMU angle input is still harmless-to-helpful, but removing physics entirely is **not** a robust improvement on the full run.
