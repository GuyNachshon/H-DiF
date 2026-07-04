# H-DiF: Hourglass Diffusion Flow Network (Thermal-to-RGB)

## Project Overview
Real-time, pixel-space Cross-Spectral Video-to-Video translation pipeline.
- **Stack:** PyTorch 2.x, CUDA 12+, k-diffusion, RAFT (Optical Flow).
- **Core Engine:** Symmetrical Hierarchical Hourglass Transformer (HDiT) + Rectified Flow Matching ODE.

## System Commands
- **Environment:** `mamba env create -f environment.yml && mamba activate hdif`
- **Train Bridge:** `python src/train.py --config config/rectified_flow.yaml --batch_size 16`
- **Inference Server:** `python src/inference.py --checkpoint latest.pt --source rtsp://camera_stream`
- **Test Suite:** `pytest tests/unit/ -v` (Run `pytest tests/temporal/` for flow checks)

## Architecture Conventions & Guardrails
- **No Latent Compression:** All processing occurs in native pixel space via hierarchical patch tokens ($4 \times 4$). Never inject VAE/VQ-GAN dependencies into the main tensor loop.
- **Conditioning Format:** Inputs to the outer transformer layers MUST be a 2-channel concatenated tensor: `[Normalized_TIR, High_Pass_Canny_Edge]`.
- **Temporal Stability:** Every active attention layer must fetch and cache its preceding state. 
  - *Equation:* `Attention_t = Softmax((QK^T)/sqrt(d_k) + gamma * Cached_Attention_t-1)`
  - *Constraint:* Ensure `Cached_Attention` is detached from the gradient graph to prevent backward memory explosion.

## Fable & Agentic Workflow Rules
- **Code Generation:** Do not add abstractions, refactors, or hypothetical helper classes beyond the immediate task. Trust framework guarantees; validate data *only* at the system boundary (e.g., initial raw video stream ingestion).
- **Tool Execution:** Run parallel speculative file reads or searches when tracking down matrix dimensions across the `k-diffusion` integration. Do not narrate options.
- **Thinking Blocks:** Never instruct or prompt the model to copy or explain its internal reasoning steps in user-facing comments or responses. Rely exclusively on implicit adaptive thinking.
- **Error Handling:** If texture ghosting occurs across sequential clips, dynamically throttle `gamma` relative to the magnitude of the local RAFT optical flow vector.

## Critical Verification Thresholds
- **SSIM Baseline:** Struct outlines must maintain an `SSIM >= 0.90` relative to source TIR frames.
- **Inference Target:** Ensure the ODE velocity loop converges in `NFE <= 4` using a midpoint solver.

## Orchestration workflow  
You (Fable) are the orchestrator. Plan, decompose, synthesize.  
- Reasoning-heavy phases → deep-reasoner  
- Mechanical work → fast-worker  
- Codex (/codex:rescue --background) is a cracked engineer on par with deep-reasoner, from a different perspective. Treat as a peer, not a reviewer.  
- High-stakes decisions: task Opus + Codex on the same problem in parallel, synthesize the best of both, without showing either the other's answer. Keep your own context lean.   

## Training
Use runpod and wandb for training. you have the runpod skills and wandb skills, and their keys in .env
for version control and artifacts use huggingface. its key is in .env as well.
You can provision yourself a GPU instance in runpod, try to balance power and time.