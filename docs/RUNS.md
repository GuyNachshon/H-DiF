# H-DiF Training Runs — Consolidated Record

Source of truth: wandb project `guy-na8/h-dif` (pulled live via API, 2026-07-05).
All numbers below come directly from run config/summary/history — nothing invented.
Companion to [ROADMAP.md](ROADMAP.md) (narrative verdicts) and [research/PLAN.md](research/PLAN.md) (gates).

## Headline table

| Run | wandb id | x0 mode / α | edge_weight | depths | params | steps (done/target) | status | best val/edge_ssim | final val/edge_ssim | best val/ssim | final val/ssim | final val/mse | final train loss | steps/sec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run1 (TIR x0) | `r483vui0` | TIR (uniform t) | n/a | [2,2,4] | 17.84M | 40000/40000 | finished | 0.9249 @10k | 0.8953 | 0.3193 | 0.3193 | 0.3087 | 0.000574 | 7.44 |
| run2 (noise x0) | `ulnf168l` | noise | n/a | [2,2,4] | 17.84M | 36450/40000 | **crashed** | 0.5276 @25k | 0.5047 | 0.0934 @20k | 0.0538 | 0.3845 | 0.02610 | 12.62 |
| run3 retry-1 | `0jmimymd` | blend 0.7 | n/a | [2,2,4] | 17.84M | 3150/40000 | **crashed** | n/a (no val logged) | n/a | n/a | n/a | n/a | 0.03296 | 21.83 |
| run3 (blend 0.7) | `r3pswolz` | blend 0.7 | n/a | [2,2,4] | 17.84M | 39950/40000 | finished | 0.7333 @30k | 0.6635 | 0.1559 @10k | 0.0684 | 0.3304 | 0.01998 | 21.78 |
| sweep α=0.45 | `rccskdc4` | blend 0.45 | n/a | [2,2,4] | 17.84M | 20000/20000 | finished | 0.0816 @10k | 0.0511 | 0.0077 @5k | -0.0089 | 0.5709 | 0.01548 | 22.15 |
| sweep α=0.55 | `gw4jag7e` | blend 0.55 | n/a | [2,2,4] | 17.84M | 20000/20000 | finished | 0.4249 @10k | 0.3343 | 0.0474 @10k | 0.0263 | 0.4176 | 0.02178 | 22.09 |
| capacity probe | `c9trjf9q` | noise | n/a | [2,2,8] | 29.38M | 20000/20000 | finished | 0.3211 @10k | 0.3208 | 0.0620 @20k | 0.0620 | 0.4472 | 0.02522 | 18.91 |
| run4 (200k edge-loss) | `kaai54ab` | blend 0.7 | 0.5 | [2,2,8] | 29.38M | 171400/200000 | **running** | 0.4681 @170k | 0.4681 | 0.0498 @5k | 0.0199 | 0.4432 | 0.02762 | 18.04 |

Notes on the table:
- "best" is the max value seen across all logged validation points (every 5000 steps); "final" is the last logged point. For a metric that trends up-then-down (most edge_ssim curves), best and final can differ a lot — see per-run detail.
- `0jmimymd` is a first attempt at run3 that crashed at step 3150 before any validation pass (`val_every: 5000`); `r3pswolz` is the restart that completed. Both share identical config.
- run4 is still in progress (step 171400/200000 as of the pull); all run4 numbers are a snapshot, not final.

## Config deltas (vs. base `rectified_flow.yaml`)

Base recipe (run1): `x0_mode` unset (TIR broadcast), `t_sampling: uniform`, depths `[2,2,4]`, no edge loss, lr 5e-4, 40k steps, midpoint/2-step sampling.

| Run | x0_mode | x0_alpha | edge_weight | depths | steps | lr | t_sampling | sampler |
|---|---|---|---|---|---|---|---|---|
| run1 | (TIR, implicit) | — | — | [2,2,4] | 40000 | 5e-4 | uniform | midpoint/2 |
| run2 | noise | — | — | [2,2,4] | 40000 | 5e-4 | logit_normal | midpoint/2 |
| run3 (both attempts) | blend | 0.7 | — | [2,2,4] | 40000 | 5e-4 | logit_normal | midpoint/2 |
| sweep α=0.45 | blend | 0.45 | — | [2,2,4] | 20000 | 5e-4 | logit_normal | midpoint/2 |
| sweep α=0.55 | blend | 0.55 | — | [2,2,4] | 20000 | 5e-4 | logit_normal | midpoint/2 |
| capacity probe | noise | (0.45 logged, unused for noise mode) | — | [2,2,8] | 20000 | 5e-4 | logit_normal | midpoint/2 |
| run4 | blend | 0.7 | 0.5 | [2,2,8] | 200000 | 4e-4 | logit_normal | euler/4 |

Params: `[2,2,4]` depths → **17,843,386** (17.84M); `[2,2,8]` depths → **29,377,754** (29.38M) — computed by instantiating `build_hdit()` from each config, not estimated.

## Per-run detail

### run1 — TIR x0 (`r483vui0`), finished, 40000/40000 steps, 6080s runtime
Val edge_ssim trajectory: 0.7465 (5k) → 0.9249 (10k) → 0.9107 (15k) → 0.9147 (20k) → 0.8399 (25k) → 0.9213 (30k) → 0.8953 (35k, final logged point). Peaks near the Phase-1 gate (≥0.90) early and oscillates around it — structure learns fast under TIR-as-x0, but per ROADMAP.md the color channel collapsed (not captured by edge_ssim, which only scores Canny-edge structure).

### run2 — noise x0 (`ulnf168l`), **crashed** at step 36450/40000, 2045s runtime
Edge_ssim never exceeds 0.53 (peak 0.5276 @25k); val/ssim stays under 0.1 throughout. Confirms ROADMAP's "structure 0.49-0.59" characterization, run trending flat/slightly down by its last point (0.5047 @35k) before the crash.

### run3 — blend α=0.7 (`r3pswolz`, restart of crashed `0jmimymd`), finished, 39950/40000 steps, 2097s runtime
Edge_ssim climbs steadily: 0.4794 (5k) → 0.6069 (10k) → 0.6875 (15k) → 0.7315 (20k) → 0.7137 (25k) → 0.7333 (30k, peak) → 0.6635 (35k, final). Best result of the three x0-mode variants at matched 40k budget; still below the 0.90 gate. `0jmimymd` died at step 3150 (228s runtime) before its first validation pass — identical config to `r3pswolz`, contributes no metrics.

### sweep α=0.45 (`rccskdc4`), finished, 20000/20000 steps, 1084s runtime
Edge_ssim collapses to near-zero and trends down: 0.0789 (5k) → 0.0816 (10k, peak) → 0.0744 (15k) → 0.0511 (20k, final). val/ssim goes negative by the end (-0.0089). Confirms ROADMAP's "α=0.45 → garbage" / training-cliff verdict.

### sweep α=0.55 (`gw4jag7e`), finished, 20000/20000 steps, 1044s runtime
Edge_ssim peaks at 0.4249 (10k) then declines: 0.4129 (15k) → 0.3343 (20k, final). Below run3's α=0.7 result at the same 20k-step mark (0.7315), supporting "α≤0.55 hits a training cliff" — the degradation is milder than α=0.45 but the trend is still downward, not toward the gate.

### capacity probe (`c9trjf9q`), noise x0, depths [2,2,8] (29.38M params), finished, 20000/20000 steps, 1210s runtime
Edge_ssim: 0.2596 (5k) → 0.3211 (10k, peak) → 0.2913 (15k) → 0.3208 (20k, final) — roughly flat/oscillating in the 0.26-0.32 band. Tests whether extra depth alone rescues noise-x0; at matched 20k steps it's higher than sweep α=0.45/0.55 but still far below blend-0.7's run3 trajectory.

### run4 — 200k edge-loss run (`kaai54ab`), blend α=0.7, edge_weight 0.5, depths [2,2,8], **running**, 171400/200000 steps so far, 10263s (~2.85h) elapsed
Edge_ssim trend across the full logged history (34 points, every 5000 steps): starts at 0.34 (5k), dips to a 0.19-0.30 band through ~90k steps, then climbs from ~100k onward: 0.378 (100k) → 0.415 (125k) → 0.465 (130k) → 0.446 (140k) → 0.459 (145k) → 0.438 (155k) → 0.462 (165k) → 0.468 (170k, latest). This is the "0.44 plateau" the team lead flagged — it is climbing slowly in the most recent 70k steps but has not broken decisively past ~0.47, well short of both the 0.90 gate and run3's unweighted-loss 0.73 peak at just 30k steps. train loss 0.0276, lr has decayed to 2.93e-5 (near its 1e-5 floor), steps/sec 18.04 (down from the 21-22 range of the 17.8M-param 40k/20k runs, consistent with the larger 29.38M model).

## GPU memory headroom

No direct GPU-memory metric is logged to wandb for any run (confirmed: no `gpu`, `memory`, or `vram` keys in any run's summary/history). The only concrete measurement available is from the team's own preflight note: **2.96 GiB / 24 GiB (~12.3%) used at batch_size=16, 256², bf16, on the 17.84M-param model** (runs 1-3, both sweeps).

Estimate for run4's 29.38M-param model (batch_size=16 unchanged, same resolution): parameter and optimizer-state memory scales roughly linearly with param count (AdamW keeps 2 extra fp32 moments per param), while activation memory (dominated by batch size × resolution × depth) grows with the added depth at the innermost level ([2,2,4]→[2,2,8], i.e. only the global-attention bottleneck level got deeper, not the higher-resolution outer levels). A conservative linear scaling on param count alone (29.38/17.84 ≈ 1.65×) puts run4 around **~4.9 GiB (~20% of 24 GiB)** — likely an overestimate of the true figure since activation memory (not param memory) usually dominates at this batch size and the outer (high-token-count) levels didn't grow. No direct measurement exists to confirm; this is an estimate, not logged evidence.

## Open bottlenecks

- bs16 at ~12% VRAM utilization (2.96 GiB / 24 GiB on the 17.8M model) — large unused headroom, batch size or model size could grow substantially before hitting the 4090's ceiling.
- 17.8M-30M param models vs. the 557M-param HDiT reference architecture — current runs are 1-2 orders of magnitude smaller than the paper's validated scale.
- No perceptual loss (LPIPS is a listed dependency and a Phase-4 gate target, but not yet wired into the training loss).
- Data is KAIST + LLVIP only (~40k pairs across 2 narrow domains) — no diversity beyond thermal-pedestrian surveillance scenes.
- Edge-loss proxy misalignment: adding the differentiable Sobel edge-consistency loss (run4, edge_weight=0.5) produced a lower edge_ssim plateau (~0.44-0.47) than the unweighted blend-0.7 recipe (run3, 0.73 peak at only 30k steps) — the proxy loss designed to help the edge_ssim gate is currently correlating with a worse outcome on that same metric, though run4 is also running 5x longer at 1.65x model size with a different LR schedule, so the comparison isn't a clean single-variable A/B.
