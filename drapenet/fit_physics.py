"""Method 2: differentiable-physics inverse fitting.

Given a target drape (silhouette or polar outline r(theta)), optimise log B1, log B2, log G
by back-propagating through the simulator. Here we fit the polar outline r(theta), which is
smooth and differentiable. Optionally initialise from the CNN prediction (hybrid method).
"""
import argparse, math, time, torch
from drapenet.sim import DrapeSim, drape_coefficient, R_DISC


def outline(x, sim, nb=90, tau=200.0):
    """Soft (differentiable) polar outline: soft-max of particle radius per angle bin."""
    pts = x[..., :2][:, sim.inside]
    r = pts.norm(dim=-1); th = torch.atan2(pts[..., 1], pts[..., 0])
    centers = torch.linspace(-math.pi, math.pi, nb + 1)[:-1] + math.pi / nb
    d = torch.remainder(th[..., None] - centers + math.pi, 2 * math.pi) - math.pi  # (b,P,nb)
    wgt = torch.exp(-(d / (2 * math.pi / nb)) ** 2)
    return (torch.logsumexp(tau * r[..., None] + torch.log(wgt + 1e-12), 1) - torch.log(wgt.sum(1))) / tau


def fit(target_outline, w, init, sim, iters=40, steps=500, lr=0.08, seed=0, log=print):
    p = init.clone().requires_grad_(True)  # (3,) log10 B1, B2, G
    opt = torch.optim.Adam([p], lr=lr)
    hist = []
    for it in range(iters):
        B1, B2, G = (10 ** p).unbind()
        x = sim.run(B1[None], B2[None], G[None], w[None], steps=steps, seed=seed, grad=True, iters=6)
        loss = ((outline(x, sim)[0] - target_outline) ** 2).mean() * 1e4
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append((loss.item(), p.detach().clone()))
        log(f"iter {it:2d} loss {loss.item():.4f} logB1 {p[0]:.3f} logB2 {p[1]:.3f} logG {p[2]:.3f}")
    return p.detach(), hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30); ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--cnn_init", action="store_true", help="hybrid: start from the CNN prediction")
    ap.add_argument("--lr", type=float, default=0.08)
    a = ap.parse_args()
    torch.set_num_threads(8)
    sim = DrapeSim(41)
    true = torch.tensor([-4.7, -5.0, -0.4]); w = torch.tensor(1.5)
    with torch.no_grad():
        B1, B2, G = (10 ** true).unbind()
        xt = sim.run(B1[None], B2[None], G[None], w[None], steps=a.steps, seed=0)
        tgt = outline(xt, sim)[0]
    init = torch.tensor([-5.3, -5.3, -0.4])
    if a.cnn_init:
        from drapenet.model import DrapeCNN
        from drapenet.sim import silhouette
        ck = torch.load("checkpoints/drapecnn.pt"); m = DrapeCNN(); m.load_state_dict(ck["model"]); m.eval()
        with torch.no_grad():
            init = m(silhouette(xt, sim.inside)[:, None].float(), (w[None] - ck["wmu"]) / ck["wsd"])[0] * ck["sd"] + ck["mu"]
        print("CNN initial guess", init.tolist())
    t = time.time()
    p, _ = fit(tgt, w, init, sim, iters=a.iters, steps=a.steps, lr=a.lr)
    print("true", true.tolist(), "fitted", p.tolist(), f"{time.time()-t:.0f}s")
