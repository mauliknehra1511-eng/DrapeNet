"""Predict fabric bending/shear rigidity from a REAL drape photo.

Protocol (no drape tester needed):
 1. Cut a 30 cm diameter circle of fabric; mark the warp direction with a pen arrow.
 2. Place it centred on an 18 cm diameter disc (plate/tin lid) on a ~25 cm tall stand.
 3. Phone directly above (use a level), dark background, warp arrow pointing to the RIGHT of the image.
 4. Weigh the circle -> GSM = grams / (pi*0.15^2).
 5. Also photograph a 30 cm reference (ruler/the flat specimen) to know pixels per metre, OR crop the
    photo so that the square image spans exactly 30 cm (the flat specimen diameter).

Usage:
  python -m drapenet.predict_real photo.jpg --gsm 150 [--px_per_m 2000] [--invert]
"""
import argparse, math, numpy as np, torch
from PIL import Image
from drapenet.model import DrapeCNN
from drapenet.sim import R_SPEC, fill_holes

G0 = 9.81


def photo_to_silhouette(path, px_per_m=None, invert=False, res=64):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    thr = 0.5 * (np.percentile(g, 5) + np.percentile(g, 95))  # simple global threshold
    mask = g > thr
    if invert:
        mask = ~mask
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    half = R_SPEC * px_per_m if px_per_m else max(g.shape) / 2
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    m = Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize((res, res), Image.BILINEAR)
    t = torch.tensor(np.asarray(m) > 127, dtype=torch.float32)[None, None]
    # image rows = y (down), cols = x; simulator silhouette is indexed [x, y] with y up
    t = fill_holes(t)[0, 0].flip(0).T.contiguous()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo"); ap.add_argument("--gsm", type=float, required=True)
    ap.add_argument("--px_per_m", type=float); ap.add_argument("--invert", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/drapecnn.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt); m = DrapeCNN(); m.load_state_dict(ck["model"]); m.eval()
    sil = photo_to_silhouette(a.photo, a.px_per_m, a.invert)
    w = a.gsm * 1e-3 * G0  # N/m^2
    with torch.no_grad():
        p = m(sil[None, None], ((torch.tensor([w]) - ck["wmu"]) / ck["wsd"]))[0] * ck["sd"] + ck["mu"]
    B1, B2, G = (10 ** p).tolist()
    for name, B in [("warp", B1), ("weft", B2)]:
        C = (B / w) ** (1 / 3)  # Peirce bending length
        print(f"{name}: bending rigidity B = {B:.2e} N·m  ({B * 1e7 / G0 / 100:.3f} gf·cm²/cm)  bending length C = {C*100:.2f} cm")
    print(f"effective shear parameter G = {G:.2f} N/m")
    Image.fromarray((sil.T.flip(0).numpy() * 255).astype(np.uint8)).resize((256, 256)).save("silhouette_used.png")
    print("saved silhouette_used.png (check it looks right!)")


if __name__ == "__main__":
    main()
