# H-DiF Execution Roadmap

Operational companion to [research/PLAN.md](research/PLAN.md). That doc defines *what* and *why*; this one tracks *how*, *with what*, and *where we are*. Update the status lines as phases close.

---

## Phase 0 — Scaffold ✅ (2026-07-04)

Repo built and tested (10/10 CPU tests). Load-bearing decisions, each independently reached by two designers (Opus + Codex):

| Decision | Rationale |
|---|---|
| Vendor k-diffusion's `image_transformer_v2.py` → `src/models/hdit.py` (git rev `4601bf0`; PyPI never shipped HDiT) | Attention-residual cache needs pre-softmax logits; fused SDPA never materializes them. Two surgical edits: `cache=` threading + explicit-softmax branch gated on `use_cache`. |
| Attention cache **only at the global-attention bottleneck** (~256 tokens @ 256²) | Outer width-128 level = 4096 tokens ≈ 1GB/head if materialized; it's shifted-window anyway. Enforced by `build.py` assert (`hw ≤ 1024`). |
| Own rectified-flow loss + Euler/midpoint samplers (~30 lines) | k-diffusion's wrappers are EDM/sigma-shaped. CaReFlow/PMRF deliberately NOT built — they're escape hatches, `trajectory_straightness()` measures whether we need them. |
| `in_channels=5` = RGB `x_t` ⊕ [TIR, Canny] conditioning | Keeps the API stable regardless of the open Phase-2 x0 question. |
| No NATTEN / flash-attn anywhere | Wheel-availability risk; shifted-window outer config is pure PyTorch. |

## Phase 1 — Foundational Bridge 🔄 (in progress)

**Goal:** validate the thermal→RGB rectified-flow bridge learns structure. Gate: **edge-SSIM ≥ 0.90** (SSIM on Canny outlines of input TIR vs. generated RGB — PLAN §1.4; raw-image SSIM is *not* the gate since colors are underdetermined).

**Current run (signal run):**
- 40k steps, batch 16, 256², bf16, RTX 4090 (RunPod EU-RO-1, $0.69/hr), est. 4-5h ≈ $3-5
- Data: KAIST (HF mirror, stride-5 ≈ 9.5k train pairs + 2.2k official-test val) + LLVIP (12k train / 3.5k val stills). ~22k train / ~5.7k val total.
- Val every 5k steps: raw SSIM, edge-SSIM, MSE, sample grids → wandb project `h-dif`; checkpoints → HF `GuyNachshon/h-dif` (private)
- Hardened: resume (weights+opt+step+wandb id), grad clip 1.0, non-finite-loss skip guard, headless opencv

**Run 1-3 + sweep verdicts (2026-07-05):** run 1 (TIR x0): edge-SSIM 0.92, color collapsed. Run 2 (noise x0): color fixed, structure 0.49-0.59 (NFE-48 ceiling 0.71 = representational limit). Run 3 (blend α=0.7): 0.73/color alive. α-sweep: α≤0.55 hits a training cliff (α=0.45 → garbage) — α is NOT a viable dial to 0.90. Inference-time edge guidance: inert (falsified). 200k recipe: α=0.7 + differentiable Sobel edge loss (w=0.5, (1-t)-weighted) + [2,2,8] + stride-2 KAIST + hflip + euler-4.

**Then:** if edge-SSIM trends toward the gate → full 200k run on the same volume (data persists). If fine structures blur → increase outer `depths` / skip-connection weight (PLAN §1.4 risk flag).

**Deferred into this phase's full run:** EMA weights (AveragedModel, ~5 lines), LR schedule.

**Pretrained-init verdict (2026-07-05, dual-verified):** PLAN §1.2's HDiT-1B/DiT-XL warm-start is not possible — the HDiT authors never released weights (GitHub/Zenodo/HF all checked; our config is byte-identical to their unweighted oxford-flowers recipe), and DiT-XL/EDM2 are incompatible architecture families (latent-space/U-Net, mismatched channels & conditioning). All runs are scratch-init. Long-term option if ever needed: self-pretrain this exact config on a generic RGB corpus first (separate budget decision).

**Gate redefinition (2026-07-05, lit-validated):** Canny-SSIM is retired as the structural gate — it rewards luminance-copying (run 1 scored 0.92 by not colorizing) and punishes color-induced edges (run 4's real improvement scored 0.44). New composite gate: **edge-recall ≥ target AND Hasler-Süsstrunk colorfulness ≥ 0.6× GT mean**, with LPIPS/FID reported. NFE relaxed to 8 during quality iteration; compress back to ≤4 via reflow before deployment (PLAN's NFE≤4 is a deployment constraint, not a development one).

**Quality recipe v2 (post run-4 post-mortem + lit dive):** LPIPS 0.1 on x1_hat gated to t>0.5 (PixelGen recipe, FID 23.7→10.0 in pixel-space RF); two-sided Sobel loss deleted (3 compounding bugs — per-image amax the killer); SGA-style asymmetric edge term available but off; bs 64 + lr 8e-4. Scale question (30M vs ~110M) being settled empirically by twin 40k probes. Escalations if LPIPS plateaus: P-DINO patch loss (PixelGen: →7.46), then corrected SGA. Adversarial rejected (PixelGen + Codex concur: unstable in pixel space).

**Queued research (user-raised):** JEPA-family semantic conditioning — pretrain I-JEPA/V-JEPA on unlabeled KAIST thermal video, feed frozen features as conditioning alongside TIR+Canny; targets the one-to-many color ambiguity via semantics. Slots after the objective/scale probes. V-JEPA-style temporal feature prediction noted for Phase 3.

**Data (deferred until probes report):** FLIR-aligned (5.1k) + M3FD (4.2k) + MFNet (1.6k) ≈ +11k aligned automotive pairs, breaks the 2-domain narrowness; RoadScene (221) as held-out generalization probe. Current failures are objective pathologies, not generalization gaps — data adds come after.

## Phase 2 — Cross-Spectral Flow Refinement ⬜

**Goal:** sharp, realistic global texture in ≤ 4 NFE. Gates: **FID ≤ 18.0** (static scenes), NFE ≤ 4 with Euler/midpoint.

Work items, in order:
1. Add FID eval script (holdout KAIST/LLVIP partitions).
2. Measure `trajectory_straightness` across checkpoints — decide from data:
   - Straight enough → ship, skip CaReFlow/PMRF entirely.
   - Curved (needs ≥10 steps for sharpness) → PMRF alignment step; hallucination across spectra → CaReFlow cyclic constraint (mirror backward velocity).
3. **Open research question (PLAN §2.5):** one-to-many color mapping (identical thermal signature → many valid colors). Does the flow average to gray? Diagnose from val samples; candidate fixes (in escalation order): stochastic x0 perturbation, CFG-style conditioning dropout, conditional discriminator at outer layers (PLAN Phase-4 fallback).
4. **x0 formulation decision** (currently: TIR broadcast to 3ch): compare vs. noise-seeded x0 on the same 40k budget. One A/B, pick, move on.
5. **Optimizer A/B before the 200k run:** AdamW (control, current recipe) vs. **Muon** — well-validated sample-efficiency gains on transformer pretraining; a 40k A/B costs ~$4. **Gefen** (github.com/ndvbd/Gefen, 8-bit quantized AdamW states) is shelved until we're memory-bound: at ~50M params optimizer states are ~0.4GB — nothing to save on a 24GB card — and it's early-stage with custom CUDA kernels, unvalidated on diffusion/flow models. Revisit if we adopt the HDiT-1B init (PLAN §1.2), where ~8GB of AdamW states makes its 8× reduction decisive.

## Phase 3 — Temporal Stabilization ⬜

**Goal:** flicker-free video. Gates: **FVD ≤ 350**, static-region temporal variance ≤ 1.5% over 500 frames.

The machinery already exists (`AttnResidualCache`, `throttle_gamma`, RAFT wrapper, `clip_len` in dataset) — this phase *activates and tunes* it:
1. Flip `temporal.enabled: true`; train with `clip_len > 1` (KAIST provides real video clips; LLVIP stills contribute only to the spatial loss).
2. Add the multi-frame temporal velocity loss (penalize intensity shifts unless RAFT flow explains them) — the one Phase-3 piece not yet coded.
3. Tune γ from 0.35 with the throttle curve; ghosting on moving objects ⇒ γ too high (PLAN §3.3).
4. FVD eval over 60s continuous blocks.
5. **Open question (PLAN §3.4):** parallax from loosely-calibrated rigs — KAIST's rig misalignment is the natural testbed.

## Phase 4 — Evaluation Matrix & Deployment ⬜

Run the full PLAN §4 matrix; each row has a prescribed fallback:

| Metric | Target | Fallback if missed |
|---|---|---|
| LPIPS | ≤ 0.12 | conditional multi-scale discriminator at outer layers |
| Color drift (static, 500 frames) | ≤ 1.5% | 3D spatiotemporal conv wrapper on cross-attention |
| Throughput (1× RTX 4090) | ≥ 20 FPS | Rectified-CFG++ / patch-pruning static blocks |
| FP16 TensorRT vs FP32 | ≤ 4.5% loss | QAT on outer transformer projections |

Deployment: ONNX export → TensorRT FP16 engine → `inference.py` RTSP streaming path (already stubbed).

---

## Operations

- **Infra:** RunPod pod `h-dif-phase1` (4090, EU-RO-1) + 200GB network volume `h-dif-data` (datasets persist across pods). Driver 570 ⇒ cu128 torch, resolved natively by uv.lock's Linux fork. **40GB container disk is too small** for venv (7G) + full staged dataset (~28G) — provision **100GB+** for the 200k run; until then only the train split stages to NVMe (launch_train.sh handles it).
- **Data sources:** KAIST official links are dead; use HF mirror `koifisharriet/KAIST-Multispectral-Pedestrian-Benchmark` (stride-filter with `allow_patterns` — HF fetches ~7 files/s, so file count dominates). LLVIP via gdown `1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv`.
- **Tracking:** wandb project `h-dif`; artifacts/checkpoints → HF `GuyNachshon/h-dif`; code → GitHub `GuyNachshon/H-DiF`.
- **Budget:** signal runs ~$5; full 200k run ~$16-20 on 4090. Balance-check `runpodctl user` before long runs. Stop pods after runs — the volume keeps the data.
- **Workflow:** Fable orchestrates; reasoning → deep-reasoner (Opus), mechanical → fast-worker, independent second designs/audits → Codex; high-stakes calls get Opus + Codex in parallel, synthesized blind.
