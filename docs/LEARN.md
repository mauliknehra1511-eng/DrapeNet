# Learn DrapeNet step by step

This guide assumes you know first-year maths (calculus, vectors) and basic Python.
Each step has: **concept → what to read/watch → what to do in this repo → check yourself**.
Estimated hours are for a 2nd-year student working seriously.

| Step | Topic | Hours |
|---|---|---|
| 0 | Python, NumPy, PyTorch tensors & autograd | 8–12 |
| 1 | Fabric mechanics: drape, bending, shear (textile side) | 5–6 |
| 2 | Cloth simulation: particles, energies, time stepping | 10–12 |
| 3 | Building the virtual Cusick drape tester (`sim.py`) | 8–10 |
| 4 | Synthetic dataset generation (`generate.py`) | 3–4 |
| 5 | CNN regression (`model.py`, `train.py`) | 8–10 |
| 6 | Differentiable physics / inverse problems (`fit_physics.py`) | 5–6 |
| 7 | Real-fabric validation (`predict_real.py`) | 6–8 |
| 8 | Write-up, figures, GitHub, CV bullets | 4–5 |
| **Total** | | **≈ 57–73 h** (≈ 5–6 weeks at 12 h/week) |

---

## Step 0 — Python + PyTorch basics (8–12 h)

**Concept.** A PyTorch *tensor* is a NumPy array that (a) can live on a GPU and (b) remembers the
operations applied to it so it can compute derivatives automatically (*autograd*).

**Resources.**
- PyTorch "Learn the Basics" tutorial (pytorch.org/tutorials) — tensors, autograd, nn.Module, training loop.
- 3Blue1Brown "Neural networks" series (chapters 1–4) for intuition about gradient descent.

**Do.**
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
E = x**3 + 2*x            # an "energy"
E.backward()
print(x.grad)             # dE/dx = 3x^2 + 2 = 14
```
The simulator uses exactly this trick: we write the fabric's potential energy `E(x)` and let
autograd give us the force `F = -dE/dx` on every particle.

**Check yourself.** Can you write a loop that minimises `(x-3)^2` with gradient descent?

---

## Step 1 — Fabric mechanics (5–6 h)

**Drape** is how a fabric hangs in folds under its own weight. Industry measures it with the
**Cusick drape tester** (BS 5058 / ISO 9073-9): a 30 cm circle of fabric is clamped between two
18 cm discs; the unsupported ring falls into folds ("nodes"); a light from below casts a shadow.

**Drape coefficient**
`DC = (A_shadow − A_disc) / (A_specimen − A_disc)`
- DC ≈ 0 → very limp (silk, chiffon); DC ≈ 1 → stiff (heavy denim, canvas).

**What controls drape?** Mainly three properties (Kawabata KES-F system):
- **Bending rigidity B** (N·m, per unit width) — resistance to curving. Warp (B1) and weft (B2) can differ.
- **Shear rigidity G** — resistance to the yarn grid turning from squares into diamonds. Needed for
  a flat circle to form *double-curved* folds.
- **Areal weight w** (N/m² = GSM × 9.81 / 1000) — the load.

**Peirce bending length** `C = (B / w)^(1/3)` (the cantilever test). It is the natural length scale;
the dataset figure shows DC is mostly a function of `B/w`. This is why one number (DC) cannot tell
you B1, B2 and G separately — the *shape* of the folds (node count, symmetry, depth) carries the rest.
**That is the research question of this project.**

**Read.** Cusick (1968) "The measurement of fabric drape", J. Textile Inst. 59; Hu (ed.)
*Structure and Mechanics of Woven Fabrics* ch. on bending/shear; any KES-F overview.

**Check yourself.** Why does a fabric with B1 ≫ B2 drape into an oval shadow?

---

## Step 2 — Cloth simulation theory (10–12 h)

**Model.** Represent the fabric as a grid of particles (spacing h). Each particle has mass `m = w h² / g`.

**Energies** (see `DrapeSim.energy`):
- Bending: `E_b = ½ B κ² · area`, curvature `κ ≈ |x_{i−1} − 2x_i + x_{i+1}| / h²` (2nd finite difference).
- Shear: `E_s = ½ G γ² · area`, γ ≈ cos(angle between the two quad edges) (0 when square).
- Gravity: `E_g = Σ m g z`.
- Stretch: woven fabric hardly stretches under its own weight, so instead of a stiff spring (which
  would need tiny time steps) we use **position-based dynamics (PBD)**: after every step, move
  particles so every edge length returns to h (`project_stretch`).

**Time stepping** (damped explicit integration):
```
v ← (1 − damping)·v + dt · F/m
x ← project_constraints(x + dt·v)
```
Explicit methods blow up if `dt > ~2/ω_max`, where ω_max = sqrt(k_max/m). Bending stiffness
on a grid scales like `B/h⁴`, so stiff fabrics need smaller dt. The code computes a **per-sample**
stable dt — a bug hit during development: using one global dt for the whole batch made the soft samples never settle.

**Resources.**
- Matthias Müller, "Ten Minute Physics" videos 14–15 (cloth, PBD) — free on YouTube; best intro.
- Baraff & Witkin 1998, "Large steps in cloth simulation" (classic paper; read sections 1–4).
- Bridson, Marino, Fedkiw 2003, "Simulation of clothing with folds and wrinkles".

**Check yourself.** What happens to the required dt if you double the grid resolution?

---

## Step 3 — The virtual Cusick tester, `drapenet/sim.py` (8–10 h)

Read the file top to bottom. Key ideas:
1. `inside` = particles within the 15 cm radius; `pinned` = within the 9 cm disc (reset every step).
2. `collide` stops cloth that has fallen below the disc from passing *through* the pedestal.
3. Tiny random z-noise breaks the perfect symmetry so folds can form (real fabric is never perfect).
4. Everything is batched: `B1, B2, G, w` are vectors, so 32 different fabrics simulate in parallel.
5. `drape_coefficient` measures the max radius in 180 angular bins → shadow area → DC.
6. `node_count` counts maxima of the smoothed outline r(θ) = number of folds.

**Do.** Run
```bash
python -c "import torch; from drapenet.sim import *; s=DrapeSim(); \
x=s.run(torch.tensor([1e-5]),torch.tensor([1e-5]),torch.tensor([.5]),torch.tensor([1.5]),steps=2500); \
print(drape_coefficient(x,s)[0], node_count(drape_coefficient(x,s)[1]))"
```
Then vary B by ×10 and watch DC change. `python -m drapenet.visualize` draws `figures/example_drapes.png`.

**Validation done.** (a) DC rises monotonically with B/w; (b) 2000 vs 4000 steps changes DC by
< 0.01 on average (converged); (c) node counts 5–12 match typical Cusick results for apparel fabric.

---

## Step 4 — Synthetic data, `drapenet/generate.py` (3–4 h)

We sample thousands of random-but-realistic fabrics (log-uniform B in 1e-6…1e-4 N·m ≈ KES-B
0.01–1 gf·cm²/cm, anisotropy up to ×3 each way, G 0.05–5, 60–300 GSM), simulate each, and save
the top-view **silhouette** (64×64 image = what a camera/Cusick shadow sees), DC and node count.

This is **simulation-based inference**: when the inverse problem (image → properties) is hard,
generate lots of forward examples (properties → image) and learn the inverse with ML.

```bash
python -m drapenet.generate --n 3072 --batch 32 --steps 2500   # ~45 min on a laptop CPU
```

---

## Step 5 — CNN regression, `model.py` + `train.py` (8–10 h)

**Concept.** A convolutional neural network learns image features (fold edges, lobes, symmetry).
Our CNN: 4 conv blocks (conv-BN-GELU ×2 + maxpool) → global average pooling → concat weight w →
MLP → 3 outputs (log10 B1, log10 B2, log10 G). We predict **logs** because properties span
orders of magnitude, and **standardise** the targets so each output has equal weight in the loss.

**Augmentation.** Flipping the image left-right or up-down keeps the warp and weft axes the same,
so the labels stay valid. Rotating by 90° would swap warp↔weft, so we don't do it.

**Baseline.** A good project always compares against the obvious method: a polynomial in
(DC, w). If the CNN can't beat it, the image adds nothing. See `checkpoints/test_metrics.json`.

**Resources.** Stanford CS231n notes (modules 1–2), "Dive into Deep Learning" ch. 7–8.

```bash
python -m drapenet.train --epochs 60     # ~5–10 min on Apple-silicon (MPS) / any GPU
```

---

## Step 6 — Differentiable physics, `fit_physics.py` (5–6 h)

Because the whole simulator is PyTorch, we can compute ∂(simulated outline)/∂(B1,B2,G) and run
gradient descent **on the physical parameters themselves** to match a target drape:

```
loss(p) = ‖ outline(sim(10^p)) − outline_target ‖²     p = (log B1, log B2, log G)
```
`create_graph=True` makes autograd differentiate *through* the force computation at every time
step (backprop-through-time). The polar outline uses a soft-max over particle radii, so it's smooth.

**CNN vs physics fitting.** The CNN is instant but only as good as its training distribution;
physics fitting is slow (~1 min per fabric) but works for any fabric the simulator can represent.
**Hybrid:** start the fit from the CNN prediction — this is the "amortised + refinement" pattern
used in modern scientific ML.

**Read.** Hu et al. 2020 "DiffTaichi"; Liang, Lin, Koltun 2019 "Differentiable Cloth Simulation
for Inverse Problems" (NeurIPS); Santesteban et al. 2022 "SNUG"; Grigorev et al. 2023 "HOOD".

```bash
python -m drapenet.fit_physics --iters 40 --steps 600
```

---

## Step 7 — Real fabrics (6–8 h) — do this, it makes the project credible

1. Collect 8–10 fabrics from the department/market (silk, poplin, denim, knit jersey, canvas…).
2. Make a "kitchen Cusick tester": a 18 cm plate on a jar, fabric circles cut to 30 cm with
   the warp direction marked, phone on a stack of books directly above, dark background.
3. Weigh each circle (kitchen scale, 0.1 g) → GSM.
4. Run `python -m drapenet.predict_real photo.jpg --gsm 150 --px_per_m <pixels per metre>`.
5. Measure ground truth bending length with the **cantilever test** (ASTM D1388: slide a 2.5 cm strip
   off a table edge until it touches a 41.5° slope; C = overhang/2). Compare with predicted C.
6. If the department has KES-F or a real Cusick tester, use it — ask a professor/lab staff.
   Report correlation (Spearman ρ) between predicted and measured C. Even ρ≈0.8 on 10 fabrics is a
   strong result for an undergrad project.

An Arduino extension: an HC-SR04 / VL53L0X distance sensor on a servo sweeping the drape edge
gives r(θ) without a camera — feed it to `fit_physics.outline`-style matching.

---

## Step 8 — Write-up & CV (4–5 h)

See the README "CV bullets" section. Put the figures in the README, state the
baseline comparison honestly, and list limitations (grid resolution, no friction, simplified
shear model, sim-to-real gap). Professors value honest limitations more than inflated claims.
