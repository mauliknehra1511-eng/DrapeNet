"""Train DrapeCNN on the synthetic dataset.

Targets: log10 B1, log10 B2, log10 G (standardised). Input: silhouette + areal weight.
Augmentation: a drape test has no preferred orientation for the *camera*, but warp/weft matter,
so we only use symmetry-preserving flips (x->-x, y->-y), which keep warp/weft axes unchanged.
"""
import argparse, glob, json, os, torch, numpy as np
from drapenet.model import DrapeCNN


def load(d):
    sh = [torch.load(f) for f in sorted(glob.glob(f"{d}/shard_*.pt"))]
    cat = lambda k: torch.cat([s[k] for s in sh])
    X = cat("sil").float()[:, None]
    Y = torch.stack([cat("B1").log10(), cat("B2").log10(), cat("G").log10()], 1)
    return X, Y, cat("w"), cat("dc")


def r2(p, y):
    return 1 - ((p - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    X, Y, W, DC = load(a.data)
    n = len(X); perm = torch.randperm(n); nt = int(0.8 * n); nv = int(0.1 * n)
    tr, va, te = perm[:nt], perm[nt:nt + nv], perm[nt + nv:]
    mu, sd = Y[tr].mean(0), Y[tr].std(0); wmu, wsd = W[tr].mean(), W[tr].std()
    Yn, Wn = (Y - mu) / sd, (W - wmu) / wsd
    m = DrapeCNN().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * ((nt + 63) // 64))
    best, hist = 1e9, []
    for ep in range(a.epochs):
        m.train(); idx = tr[torch.randperm(nt)]
        for i in range(0, nt, 64):
            b = idx[i:i + 64]; x = X[b]
            if torch.rand(1) < .5: x = x.flip(2)
            if torch.rand(1) < .5: x = x.flip(3)
            loss = torch.nn.functional.smooth_l1_loss(m(x.to(dev), Wn[b].to(dev)), Yn[b].to(dev), beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            pv = m(X[va].to(dev), Wn[va].to(dev)).cpu()
        vl = ((pv - Yn[va]) ** 2).mean().item(); hist.append(vl)
        if vl < best:
            best = vl; os.makedirs(a.out, exist_ok=True)
            torch.save(dict(model=m.state_dict(), mu=mu, sd=sd, wmu=wmu, wsd=wsd), f"{a.out}/drapecnn.pt")
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"ep {ep:3d} train {loss.item():.3f} val_mse {vl:.3f}  R2 {r2(pv, Yn[va]).numpy().round(3)}", flush=True)
    ck = torch.load(f"{a.out}/drapecnn.pt"); m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        p = m(X[te].to(dev), Wn[te].to(dev)).cpu() * sd + mu
    y = Y[te]
    res = {name: dict(R2=float(r2(p[:, i:i+1], y[:, i:i+1])), MAE_log10=float((p[:, i] - y[:, i]).abs().mean()),
                      median_factor_error=float(10 ** (p[:, i] - y[:, i]).abs().median()))
           for i, name in enumerate(["B1_warp_bending", "B2_weft_bending", "G_shear"])}
    # baseline: predict log B from drape coefficient + weight only (classic single-number approach)
    A = torch.stack([DC, W, DC * W, DC ** 2, torch.ones_like(DC)], 1)
    coef = torch.linalg.lstsq(A[tr], Y[tr]).solution
    pb = A[te] @ coef
    for i, name in enumerate(["B1_warp_bending", "B2_weft_bending", "G_shear"]):
        res[name]["baseline_DC_poly_R2"] = float(r2(pb[:, i:i+1], y[:, i:i+1]))
    res["n_train"], res["n_test"] = int(nt), int(len(te))
    json.dump(res, open(f"{a.out}/test_metrics.json", "w"), indent=2)
    np.save(f"{a.out}/test_pred.npy", np.stack([p.numpy(), y.numpy()]))
    print(json.dumps(res, indent=2))
