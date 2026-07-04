# Research Project: Hourglass Diffusion Flow Network (H-DiF) for Thermal-to-RGB Video Translation

**Objective:** To build a high-fidelity, structurally precise Video-to-Video (V2V) architecture that colorizes Thermal Infrared (TIR) video sequences into photorealistic Visible Spectrum (RGB) counterparts by processing the frames hierarchically in native pixel space, enforcing temporal stability via explicit Attention Residual matrices, and leveraging Rectified Flow Matching for ultra-fast ODE sampling.

---

## Architecture Blueprint

```
             [Input Thermal Frame X_TIR at time t]
                             │
                             ▼
                 [Patchify & Convolution]
                             │
            ┌────────────────┴────────────────┐
            ▼ (High-Res Skip)                 ▼ (Downsample)
     [Outer Transformer]               [Inner Bottleneck]
    (Local Micro-Textures)            (Global──────────────┬────────────────┘
                             ▼
                 [Attention Residual Injection] ◄── [Stored Attention Matrix from t-1]
                             │
                             ▼
                [Rectified Velocity Prediction]
                             │
                             ▼
             [1-4 Step Euler/Midpoint ODE Solver]
                             │
                             ▼
                 [Colorized RGB Frame at time t]

```

---

## Phase 1: Foundational Framework & Hierarchical Tokenization

### 1.1 Objective

To instantiate a Hierarchical Hourglass Diffusion Transformer (HDiT) backbone that operates natively in pixel space, bypassing VAE compression bottlenecks while achieving sub-quadratic scaling.

### 1.2 Infrastructure & Pre-trained Initializations

* **Core packages.
* **Pre-trained Baselines:** Initialize the spatial structural weights using a pre-trained **HDiT-1B** or **DiT-XL** backbone checkpoint (originally trained on ImageNet or FFHQ-1024). This provides the model with rich, low-level structural priors.

### 1.3 Action Steps

1. **Hierarchical Setup:** Define the spatial patching configuration. Set the patch size to $4 \times 4$. The architecture will feature an outer block width of $128$ (for handling raw edges and localized textures) and a middle bottleneck block width of $512$ (for broad scene and semantic evaluation).
2. **Spectral Modification:** Reconfigure the model's input layer from standard 3-channel RGB to accept a 1-channel normalized TIR input frame, concatenated with a single channel representing the high-pass TIR Canny gradient map to lock down localized micro-edges.

### 1.4 Validations, Flags, & Thresholds

* **Validation Method:** Evaluate structural edge retention on validation splits.
* **Success Threshold:** Structural Similarity Index (**SSIM**) between the structural outlines of the input thermal frame and the generated RGB frame must be $\ge 0.90$.
* **Risk Flags:** If fine lines (like window panelling, text markings, or distant foliage) appear blurry or merged in the output space, increase the depth of the outer high-resolution layers (`depths` hyperparameter) and apply a higher weight coefficient to the skip-connections.

### 1.5 Phase 1 Critical Research Questions

> * Does pixel-space patching introduce blocking artifacts at the boundaries of high-temperature blooms? Should a sub-pixel distance transform layer be appended to the tokenizer to smooth thermal gradient borders?
> 
> 

---

## Phase 2: Cross-Spectral Rectified Flow Matching

### 2.1 Objective

To model the cross-spectral translation as a straight velocity vector traveling directly from the source thermal distribution ($\pi_0$) to the target visible spectrum distribution ($\pi_1$) over a normalized time index $t \in [0, 1]$.

### 2.2 Datasets

* **Primary Source Materials:** **KAIST Multispectral Dataset** (aligned day/night driving video segments), **FLIR Thermal Starter Dataset** (synchronized thermal-visible pairs), and **LLVIP Dataset** (aligned low-light pedestrian sequences).

### 2.3 Action Steps

1. **Flow Target Formulation:** Configure a deterministic Ordinary Differential Equation (ODE) trajectory loop. Train the network to predict the constant velocity field $v_\theta(X_t, t)$ that maps:

$$X_t = (1-t)X_{\text{TIR}} + tX_{\text{RGB}}$$


2. **Cyclic Regularization:** Implement a **Cyclic Adaptive Rectified Flow (CaReFlow)** pipeline constraint. Program a mirror backward velocity layer ensuring that the synthetic output can be mapped directly back to the original thermal source, preventing cross-spectral hallucination.

### 2.4 Validations, Flags, & Thresholds

* **Validation Method:** Compute trajectory straightness by analyzing the variance of the predicted velocity vector across varying intervals of the generation interval.
* **Success Threshold:** The model must resolve realistic global textures in $\le 4$ function evaluations (NFEs) using a basic Euler ODE solver during inference. Fréchet Inception Distance (**FID**) on static scenes must score $\le 18.0$.
* **Risk Flags:** If the model requires $\ge 10$ steps to produce sharp textures, the probability trajectory is curvature-warped. Trigger an explicit **Posterior-Mean Rectified Flow (PMRF)** alignment step to straighten the transport path.

### 2.5 Phase 2 Critical Research Questions

> * How does the "one-to-many" mapping problem behave when multiple distinct colors (e.g., a fleet of identical hot car engines painted red, blue, or yellow) map to an identical thermal signature? Will the flow matching loop average these out o a neutral gray or achieve multi-modal distribution coverage?
> 
> 

---

## Phase 3: Attention Residual Temporal Stabilization

### 3.1 Objective

To eliminate frame-to-frame video flickering by passing attention memory forward through time, constraining temporal drift across continuous frame evaluation cycles.

### 3.2 Action Steps

1. **Attention Matrix Caching:** Modify the core attention block code to save the key-query attention weight maps from frame $t-1$ into an active GPU memory cache.
2. **Residual Injection Layer:** Intercept the self-attention mechanism of the current frame $t$ and apply the decayed residual attention matrix from the past frame:

$$\text{Attention}_t = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \gamma \cdot \text{Attention}_{t-1}\right)$$


* Set the initial decay coefficient $\gamma = 0.35$.


3. **Motion Vector Optimization:** Inject a multi-frame temporal velocity loss metric, penalizing rapid shifts in local pixel intensities unless accompanied by strong optical flow shifts computed by a parallel **RAFT** pipeline processing the raw thermal feed.

### 3.3 Validations, Flags, & Thresholds

* **Validation Method:** Measure inter-frame consistency across a continuous 60-second video block using the **Fréchet Video Distance (FVD)** and local pixel variance calculations.
* **Success Threshold:** **FVD** $\le 350$. Local temporal variance across non-moving objects must not exceed $1.5\%$.
* **Risk Flags:** If moving objects leave a visual trailing ghosting artifact behind them, the temporal decay coefficient $\gamma$ is too high. Dynamically throttle $\gamma$ downward using an adaptive function tied to the absolute magnitude of the local optical flow velocity.

### 3.4 Phase 3 Critical Research Questions

> * Can attention residuals fully compensate for camera parallax errors present in loosely calibrated dual-camera thermal/RGB rigs, or will the spatial displacement cause localized artifact tearing?
> 
> 

---

## Phase 4: Rigorous Multi-Metric Evaluation Matrix

To determineif the H-DiF framework establishes a new state-of-the-art capability profile, the system will be evaluated continuously against the following rigorous baseline target metrics:

| Metric Category | Target Baseline | Testing/Validation Protocol | Strategic Action on Failure |
| --- | --- | --- | --- |
| **Perceptual Fidelity** | $\text{LPIPS} \le 0.12$ | Evaluate generation accuracy on holdout test partitions of the LLVIP and KAIST benchmarks. | Introduce a conditional multi-scale discriminator loss layer at the outer hourglass layers. |
| **Temporal Stability** | $\le 1.5\%$ color drift | Run pixel-level color variance checks on static areas across a continuous 500-frame video segment. | Integrate a 3D spatiotemporal convolution wrapper around the primary cross-attention layers. |
| **Inference Efficiency** | $\ge 20\text{ FPS}$ | Benchmark multi-step ODE sampling velocity on an isolated workstation containing 1x RTX 4090 GPU. | Implement **Rectified-CFG++** to reduce guidance overhead or apply patch-pruning to static background blocks. |
| **Edge Quantization Degradation** | $\le 4.5\%$ accuracy loss | Compare the performance profiles of an FP16 TensorRT compiled ONNX graph versus the unquantized FP32 engine. | Apply Quantization-Aware Training (QAT) directly onto the outer Transformer projections. |
