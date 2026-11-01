"""Make README figures: example drapes (3D + silhouette), CNN predicted-vs-true, DC baseline."""
import glob, json, os, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drapenet.sim import DrapeSim, silhouette, drape_coefficient

os.makedirs("figures", exist_ok=True)
NAMES = ["log10 B1 (warp bending)", "log10 B2 (weft bending)", "log10 G (shear)"]


def example_drapes():
    torch.set_num_threads(os.cpu_count())
    sim = DrapeSim(41)
    cases = [("stiff (denim-like)", 3e-5, 3e-5, 1.0, 2.5), ("soft (silk-like)", 1.5e-6, 1.5e-6, 0.1, 0.8),
             ("anisotropic B1=10xB2", 3e-5, 3e-6, 0.5, 1.5), ("high shear", 5e-6, 5e-6, 4.0, 1.5)]
    B1, B2, G, w = (torch.tensor([c[i] for c in cases]) for i in range(1, 5))
    x = sim.run(B1, B2, G, w, steps=2500)
    dc, _ = drape_coefficient(x, sim); sil = silhouette(x, sim.inside)
    fig = plt.figure(figsize=(14, 7))
    for k, c in enumerate(cases):
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        p = x[k].numpy()
        ax.plot_surface(p[..., 0], p[..., 1], p[..., 2], cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(f"{c[0]}\nDC = {dc[k]:.2f}", fontsize=9); ax.set_axis_off(); ax.view_init(25, 35)
        ax.set_box_aspect((1, 1, 0.6))
        ax2 = fig.add_subplot(2, 4, k + 5)
        ax2.imshow(sil[k].numpy().T, origin="lower", cmap="gray_r"); ax2.set_axis_off()
        ax2.set_title("top-view silhouette (CNN input)", fontsize=8)
    plt.tight_layout(); plt.savefig("figures/example_drapes.png", dpi=130); plt.close()


def pred_vs_true():
    if not os.path.exists("checkpoints/test_pred.npy"):
        return
    p, y = np.load("checkpoints/test_pred.npy"); m = json.load(open("checkpoints/test_metrics.json"))
    keys = ["B1_warp_bending", "B2_weft_bending", "G_shear"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, ax in enumerate(axs):
        ax.scatter(y[:, i], p[:, i], s=6, alpha=.5)
        lo, hi = y[:, i].min(), y[:, i].max(); ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("true " + NAMES[i]); ax.set_ylabel("predicted")
        ax.set_title(f"CNN R²={m[keys[i]]['R2']:.3f}  (DC-baseline R²={m[keys[i]]['baseline_DC_poly_R2']:.3f})", fontsize=9)
    plt.tight_layout(); plt.savefig("figures/pred_vs_true.png", dpi=130); plt.close()


def dc_hist():
    sh = [torch.load(f) for f in sorted(glob.glob("data/shard_*.pt"))]
    if not sh:
        return
    dc = torch.cat([s["dc"] for s in sh]); B = torch.cat([s["B1"] * s["B2"] for s in sh]).sqrt()
    w = torch.cat([s["w"] for s in sh])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(dc.numpy(), 40); axs[0].set_xlabel("drape coefficient"); axs[0].set_title(f"{len(dc)} simulated drapes")
    sc = axs[1].scatter((B / w).log10().numpy(), dc.numpy(), c=w.numpy(), s=4, cmap="plasma")
    axs[1].set_xlabel("log10( sqrt(B1·B2) / w )  [m³]"); axs[1].set_ylabel("DC")
    axs[1].set_title("DC is governed by bending length³ = B/w"); plt.colorbar(sc, ax=axs[1], label="w (N/m²)")
    plt.tight_layout(); plt.savefig("figures/dataset.png", dpi=130); plt.close()


if __name__ == "__main__":
    example_drapes(); dc_hist(); pred_vs_true(); print("figures written")
