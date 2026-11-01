"""Batched cloth simulator for a virtual Cusick drape test (PyTorch).

Physical setup (BS 5058 / ISO 9073-9 Cusick drape tester):
  - circular specimen, diameter 30 cm
  - clamped between two support discs, diameter 18 cm
  - the unsupported annulus hangs under gravity and forms folds ("nodes")

Model: a square grid of particles (spacing h) masked to a circle.
  - stretch: near-inextensible distance constraints (position-based dynamics, PBD)
  - shear:   energy 0.5 * G * gamma^2 * area, gamma = in-plane angle change of each quad
  - bending: energy 0.5 * B * kappa^2 * area, kappa ~ |x[i-1] - 2x[i] + x[i+1]| / h^2,
             with separate rigidities for warp (B1, along x) and weft (B2, along y)
  - gravity: areal weight w (N/m^2)
Every sample in the batch can have different (B1, B2, G, w). All ops are PyTorch, so the
simulator is differentiable end-to-end (used in fit_physics.py).
"""
import math
import torch

R_SPEC, R_DISC = 0.15, 0.09  # m


class DrapeSim:
    def __init__(self, n=41, device="cpu", dtype=torch.float32):
        self.n, self.device, self.dtype = n, device, dtype
        self.h = 2 * R_SPEC / (n - 1)
        lin = torch.linspace(-R_SPEC, R_SPEC, n, dtype=dtype, device=device)
        X, Y = torch.meshgrid(lin, lin, indexing="ij")  # X: warp direction, Y: weft direction
        r = torch.sqrt(X**2 + Y**2)
        self.inside = r <= R_SPEC + 1e-6           # (n, n) particles that exist
        self.pinned = r <= R_DISC                   # clamped between the discs
        self.rest = torch.stack([X, Y, torch.zeros_like(X)], -1)  # (n, n, 3)
        ins = self.inside
        # masks for edges / hinges / quads (all endpoints must exist)
        self.ex = (ins[1:, :] & ins[:-1, :]).to(dtype)              # edges along x
        self.ey = (ins[:, 1:] & ins[:, :-1]).to(dtype)              # edges along y
        self.bx = (ins[2:, :] & ins[1:-1, :] & ins[:-2, :]).to(dtype)  # bending hinges along x
        self.by = (ins[:, 2:] & ins[:, 1:-1] & ins[:, :-2]).to(dtype)
        self.q = (ins[1:, 1:] & ins[:-1, 1:] & ins[1:, :-1] & ins[:-1, :-1]).to(dtype)  # quads
        self.free = (ins & ~self.pinned).to(dtype)[..., None]
        self.npart = int(ins.sum())

    # ---------- energies (per sample) ----------
    def energy(self, x, B1, B2, G, w):
        """x: (b, n, n, 3); params: (b,). Returns (b,) potential energy in J (without stretch)."""
        h = self.h
        d2x = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        d2y = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        Eb = 0.5 * B1 * ((d2x**2).sum(-1) * self.bx).sum((1, 2)) / h**2 \
           + 0.5 * B2 * ((d2y**2).sum(-1) * self.by).sum((1, 2)) / h**2
        e1 = x[:, 1:, :-1] - x[:, :-1, :-1]
        e2 = x[:, :-1, 1:] - x[:, :-1, :-1]
        cos = (e1 * e2).sum(-1) / (e1.norm(dim=-1) * e2.norm(dim=-1) + 1e-12)
        Es = 0.5 * G * ((cos**2) * self.q).sum((1, 2)) * h**2  # gamma ~ cos(angle) for small shear
        mask = self.inside.to(self.dtype)
        Eg = w * h**2 * (x[..., 2] * mask).sum((1, 2))
        return Eb + Es + Eg

    # ---------- inextensibility (PBD Jacobi projection) ----------
    def project_stretch(self, x, iters):
        h, f = self.h, self.free
        for _ in range(iters):
            corr = torch.zeros_like(x)
            cnt = torch.zeros_like(x[..., :1])
            for dim, m in ((1, self.ex), (2, self.ey)):
                a = x.narrow(dim, 0, x.shape[dim] - 1)
                b = x.narrow(dim, 1, x.shape[dim] - 1)
                d = b - a
                L = d.norm(dim=-1, keepdim=True) + 1e-12
                c = (L - h) / L * d * 0.5 * m[None, ..., None]
                corr.narrow(dim, 0, x.shape[dim] - 1).add_(c)
                corr.narrow(dim, 1, x.shape[dim] - 1).sub_(c)
                cnt.narrow(dim, 0, x.shape[dim] - 1).add_(m[None, ..., None])
                cnt.narrow(dim, 1, x.shape[dim] - 1).add_(m[None, ..., None])
            x = x + 1.5 * corr / cnt.clamp(min=1) * f  # over-relaxed Jacobi
        return x

    def collide(self, x):
        """Fabric below the disc plane cannot go inside the pedestal (radius R_DISC*0.98)."""
        rxy = x[..., :2].norm(dim=-1, keepdim=True)
        rmin = R_DISC * 0.98
        below = (x[..., 2:3] < -0.002)
        push = below & (rxy < rmin)
        scale = torch.where(push, rmin / (rxy + 1e-9), torch.ones_like(rxy))
        return torch.cat([x[..., :2] * scale, x[..., 2:3]], -1)

    def run(self, B1, B2, G, w, steps=600, dt=None, damping=0.03, iters=8, noise=1e-3,
            seed=None, grad=False):
        """Quasi-static drape via damped dynamics. Returns final positions (b, n, n, 3)."""
        bsz = B1.shape[0]
        g = 9.81
        mass = (w * self.h**2 / g)[:, None, None, None]  # kg per particle
        if dt is None:  # per-sample stable explicit step (stiff samples get smaller steps)
            kmax = 16 * torch.maximum(B1, B2) / self.h**4 * self.h**2 + 4 * G
            dt = (0.9 * torch.sqrt(mass.flatten() / kmax)).clamp(max=4e-3)
        dt = torch.as_tensor(dt, dtype=self.dtype).expand(bsz)[:, None, None, None]
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        x = self.rest.expand(bsz, -1, -1, -1).clone()
        x[..., 2] += noise * torch.randn(x.shape[:-1], generator=gen).to(x) * self.free[..., 0]
        v = torch.zeros_like(x)
        P = lambda t: t[:, None, None]
        for _ in range(steps):
            with torch.enable_grad():
                xr = (x if x.requires_grad else x.requires_grad_(True)) if grad else x.detach().requires_grad_(True)
                E = self.energy(xr, B1, B2, G, w).sum()
                F = -torch.autograd.grad(E, xr, create_graph=grad)[0]
            v = (1 - damping) * v + dt * F / mass * self.free
            xn = self.project_stretch(x + dt * v, iters)
            xn = self.collide(xn)
            xn = torch.where(self.pinned[None, ..., None], self.rest.expand_as(xn), xn)
            v = (xn - x) / dt
            x = xn if grad else xn.detach()
        return x


# ---------- measurements ----------
def silhouette(x, inside, res=64):
    """Top-view shadow image (b, res, res) in [-R_SPEC, R_SPEC]^2, like the Cusick paper ring."""
    b = x.shape[0]
    pts = x[..., :2][:, inside]  # (b, P, 2)
    img = torch.zeros(b, res * res)
    # splat each particle as a small disc by rasterising quads would be ideal; a dense
    # splat at 2x resolution followed by closing is fast and good enough.
    idx = ((pts / R_SPEC + 1) * 0.5 * (res - 1)).round().long().clamp(0, res - 1)
    flat = idx[..., 0] * res + idx[..., 1]
    img.scatter_(1, flat, 1.0)
    img = img.view(b, 1, res, res)
    k = 3
    img = torch.nn.functional.max_pool2d(img, k, 1, k // 2)               # dilate
    img = -torch.nn.functional.max_pool2d(-img, k, 1, k // 2)             # erode (closing)
    img = fill_holes(img)
    return img[:, 0]


def fill_holes(img):
    """Fill interior holes of a binary image by flood-filling the background from the border."""
    b, _, H, W = img.shape
    bg = torch.zeros_like(img)
    bg[..., 0, :] = 1 - img[..., 0, :]; bg[..., -1, :] = 1 - img[..., -1, :]
    bg[..., :, 0] = 1 - img[..., :, 0]; bg[..., :, -1] = 1 - img[..., :, -1]
    for _ in range(H):
        nb = torch.nn.functional.max_pool2d(bg, 3, 1, 1) * (1 - img)
        if torch.equal(nb, bg):
            break
        bg = nb
    return 1 - bg


def drape_coefficient(x, sim):
    """DC = (A_shadow - A_disc) / (A_specimen - A_disc), from the projected outline.
    Uses the polar outline of the boundary particles (robust, resolution independent)."""
    ins, n = sim.inside, sim.n
    pts = x[..., :2][:, ins]
    r = pts.norm(dim=-1)
    th = torch.atan2(pts[..., 1], pts[..., 0])
    nb = 180
    bins = ((th + math.pi) / (2 * math.pi) * nb).long().clamp(0, nb - 1)
    rmax = torch.zeros(x.shape[0], nb).scatter_reduce(1, bins, r, "amax", include_self=True)
    rmax = torch.maximum(rmax, torch.full_like(rmax, R_DISC))
    area = 0.5 * (rmax**2).sum(1) * (2 * math.pi / nb)
    A_d, A_s = math.pi * R_DISC**2, math.pi * R_SPEC**2
    return ((area - A_d) / (A_s - A_d)).clamp(0, 1), rmax


def node_count(rmax):
    """Number of folds = number of local maxima of the smoothed polar outline r(theta)."""
    k = torch.tensor([1, 4, 6, 4, 1], dtype=rmax.dtype) / 16
    r = rmax
    for _ in range(3):
        r = torch.nn.functional.conv1d(torch.cat([r[:, -2:], r, r[:, :2]], 1)[:, None], k[None, None])[:, 0]
    prom = (r - r.mean(1, keepdim=True))
    is_max = (r > r.roll(1, 1)) & (r >= r.roll(-1, 1)) & (prom > 0.002)
    return is_max.sum(1)
