"""Generate a synthetic Cusick-drape dataset.

Sampled parameters (log-uniform, realistic apparel-fabric ranges):
  B1 (warp bending rigidity)   1e-6 .. 1e-4  N*m      (KES-B ~ 0.01-1 gf*cm^2/cm)
  B2 (weft) = B1 * anisotropy, anisotropy in 10^[-0.5, 0.5]
  G  (effective shear param)   0.05 .. 5     N/m
  w  (areal weight)            0.6 .. 3.0    N/m^2   (~60 - 300 g/m^2)
Saves shards  data/shard_XXX.pt  with silhouettes (64x64), params, drape coefficient, node count.
"""
import argparse, os, time, torch
from drapenet.sim import DrapeSim, silhouette, drape_coefficient, node_count


def sample(n, g):
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)
    B1 = 10 ** u(-6, -4)
    B2 = B1 * 10 ** u(-0.5, 0.5)
    G = 10 ** u(-1.3, 0.7)
    w = u(0.6, 3.0)
    return B1, B2, G, w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(a.out, exist_ok=True)
    sim = DrapeSim(41)
    for k in range(a.n // a.batch):
        f = f"{a.out}/shard_{k:03d}.pt"
        if os.path.exists(f):
            continue
        g = torch.Generator().manual_seed(a.seed * 100000 + k)
        B1, B2, G, w = sample(a.batch, g)
        t = time.time()
        x = sim.run(B1, B2, G, w, steps=a.steps, seed=a.seed * 100000 + k)
        dc, rmax = drape_coefficient(x, sim)
        torch.save(dict(sil=silhouette(x, sim.inside).to(torch.uint8), B1=B1, B2=B2, G=G, w=w,
                        dc=dc, nodes=node_count(rmax), height=x[..., 2].half()), f)
        print(f"{f}  {time.time()-t:.1f}s  DC {dc.min():.2f}-{dc.max():.2f}", flush=True)
