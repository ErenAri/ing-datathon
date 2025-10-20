# Next Steps (ING Datathon)

This memo captures current status, recommended next steps, and ready-to-run commands to resume quickly.

## Current Snapshot
- Blend space comparison (OOF composite):
  - Probability blend: 1.149420 (better)
  - Rank blend: 1.147111
- Calibration: no OOF gain observed (method=none, gamma=1.00) for current setup.
- Domain AUC:
  - Baseline (final features in a naive run): ~1.0000 (undesirable)
  - Iterative adversarial filtering result (from outputs/reports/iter_adv_report.json):
    - domain_auc_final = 0.74079 (≤ target 0.75)
    - oof_composite_final = 1.14820 (small change vs baseline)
  - Dropped features include time/leak-like drivers (e.g., tenure, month_sin/cos, various days_since_*), which substantially reduced domain separability.

Artifacts to consult:
- outputs/reports/validation_summary.json (final blend, per-month metrics, weights)
- outputs/reports/blend_per_month.json, blend_global.json (weights)
- outputs/reports/iter_adv_report.json (iterative adversarial filtering history/results)
- outputs/reports/adversarial_drop.csv (drift-ranked features and dropped flags)

## Immediate Action Plan
1) Adopt adversarial filtering + iterative filtering (safe domain AUC)
   - Goal: keep domain AUC ≤ 0.75 with minimal composite impact.
   - Command:
     - PowerShell (Windows):
       - `python .\main.py --models lgb xgb --last-n 6 --blend-space proba --adv-filter --adv-drop-k 80 --iter-adv --iter-adv-target 0.75 --adv-diag-failfast`

2) Expand models for uplift (CatBoost + Two-Stage)
   - After Step 1 confirms domain AUC ~0.7–0.75, include additional heads:
     - `python .\main.py --models lgb xgb cat two --last-n 6 --blend-space proba --adv-filter --adv-drop-k 80 --iter-adv --iter-adv-target 0.75 --cat-seeds 9`
   - Expect improvements in Recall@10/Lift@10 from CatBoost and Two-Stage; re-check OOF composite and per-month stability.

3) Calibration (only if beneficial)
   - Retry polynomially when OOF improves; keep guard to skip if no OOF gain:
     - `--calib auto --gamma-grid 0.80,0.85,0.90,0.95,1.00,1.05,1.10`

4) Validate & Persist
   - Check FINAL SUMMARY (console) and these artifacts after each run:
     - outputs/reports/validation_summary.json (month metrics, gamma, blend space)
     - outputs/reports/blend_per_month.json and blend_global.json
     - outputs/reports/iter_adv_report.json (if iter-adv used)

5) Optional Fast Diagnostics
   - Identify top drift features without full training:
     - `python adversarial_filter.py --k 80 --pickles-dir .`
   - Confirms alignment with features dropped by iter-adv.

## Stretch Goals (if time allows)
- Optuna tuning with composite objective (last-6 months, oof_composite_monthwise) for LGB/XGB/Cat.
- Segment-aware calibration beyond tenure (e.g., product activity bins) if artifacts show segment-specific miscalibration.
- Finalize a Kaggle-ready notebook: metric definition, leakage proof, domain AUC report, per-month table, final inference.

## Notes & Tips
- Keep `--blend-space proba` as default (currently superior). Revisit rank blend only if adding many heads changes ordering behavior.
- Leave calibration enabled only if OOF improves; otherwise skip to avoid overfitting tails.
- Maintain time-decay and gap defaults (gap=1, lambda≈0.2); adjust if recent months dominate leaderboard distribution.

