"""Picture of the von Mises-Fisher window logic on the bottom views.

For each checkpoint (env MODELS="label|ckpt|n;label|ckpt|n;..."), with the image tokens drawn on the hemisphere in an
azimuthal projection (distance from the centre = polar angle from the optical axis, direction = azimuth psi):
  1. the test image (pyramid of n sides seen from below);
  2. the turn-following part of the token features: for the scalar channel with the largest frequency-n component that
     turns with the object (over 300 test images), that channel's deviation from its mean, for this image; an n-lobed
     pattern on the outer rings is what the model has to read;
  3. the pose token's attention weights over the image tokens in one block, for all 8 heads at once (each head's weights
     scaled to its own maximum; colour = head), at the true pose: each head is a narrow von Mises-Fisher window around
     one direction ("key direction" the query picks out), its width set by s_h;
  4. one head's window for pose hypotheses turned about the optical axis by 0, 1/4, 1/2 and 3/4 of the n-gon's period
     360/n degrees: the window is fixed in the body frame, so it moves along the ring with the hypothesis and samples
     the n-lobed pattern at the hypothesised turn.
env: COSDIR_SNAPSHOT (patched snapshot, cap on s_h raised), BDATA (renders: test/images/pyr<n>_<i>.png,
test/rotations.npz), EXTRA_ARGS ('--dircloud_nside 8 --dircloud_grid 14'), BLOCK (default 1), OUT."""
import os, sys, math, json
os.environ.setdefault("COSDIR_DATA", "/nonexistent")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
import lib
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

D = os.environ["BDATA"]; BLOCK = int(os.environ.get("BLOCK", 1)); NIMG = 300
HEADC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

def rz(phi):
    c, s = math.cos(phi), math.sin(phi)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

def proj(pts):
    pol = np.degrees(np.arccos(np.clip(pts[:, 2], -1, 1))); psi = np.arctan2(pts[:, 1], pts[:, 0])
    return pol * np.cos(psi), pol * np.sin(psi)

def frame(ax, n):
    for r in (30, 60, 90): ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#cccccc", lw=0.6))
    ax.set_xlim(-95, 95); ax.set_ylim(-95, 95); ax.set_aspect("equal"); ax.axis("off")

def run(label, ckpt, n, axes):
    lib.CKPT = ckpt; torch.set_num_threads(4)
    m = lib.load_model(fused=False)
    rots = np.load(D + "/test/rotations.npz")[f"pyr{n}"][:NIMG]
    theta = np.arctan2(rots[:, 0, 1, 0], rots[:, 0, 0, 0])
    with torch.no_grad():
        cond = torch.cat([m.condition(torch.from_numpy(np.stack([np.asarray(Image.open(D + f"/test/images/pyr{n}_{j:05d}.png").convert("RGB"))
                          for j in range(i, min(i + 25, NIMG))])).permute(0, 3, 1, 2).float() / 255) for i in range(0, NIMG, 25)])
    P = cond.shape[1] - 1; pts = cond[0, :P, 1:4].numpy(); sig = cond[:, :P, 8:72].double().numpy()      # (N, P, 64)
    x, y = proj(pts)
    coh = (sig * np.exp(1j * n * theta)[:, None, None]).mean(0)                     # (P, 64) turn-following part
    ch = int(np.abs(coh).sum(0).argmax()); img = 0
    dev = sig[img, :, ch] - sig[:, :, ch].mean(0)
    axes[0].imshow(Image.open(D + f"/test/images/pyr{n}_{img:05d}.png")); axes[0].axis("off")
    axes[0].set_title(f"{label}\nturn {np.degrees(theta[img]) % (360 / n):.1f}° (mod {360 / n:.1f}°)", fontsize=9, loc="left")
    v = np.abs(dev).max()
    axes[1].scatter(x, y, c=dev, cmap="RdBu_r", vmin=-v, vmax=v, s=9); frame(axes[1], n)
    axes[1].set_title(f"turn-following token feature\n(channel {ch}, frequency {n})", fontsize=9, loc="left")
    R0 = torch.from_numpy(rots[img, 0].astype(np.float32))
    from probe import pose_tokens, stream
    from attn import parts
    def weights(R):
        vf = m.vector_field; out = []
        hooks = [a.register_forward_pre_hook(lambda mod, args, out=out: out.append(parts(mod, args[0], args[1], args[3], args[4])))
                 for a in vf.angular_attention]
        with torch.no_grad(): stream(vf, pose_tokens(R[None]), torch.full((1,), 0.9), cond[img:img + 1].float())
        for hk in hooks: hk.remove()
        return np.stack([(o_[0] + o_[1] + o_[2]).softmax(-1)[0, :, 0, 1:P + 1].numpy() for o_ in out])   # (blocks, H, P)
    W = weights(R0)
    # the block and head that put the most weight on the ring where the n-lobed pattern lives (20 to 65 deg from the
    # axis); most other heads look at the rim, the axis or the pose token itself
    pol = np.degrees(np.arccos(np.clip(pts[:, 2], -1, 1))); ring = (pol >= 20) & (pol <= 65)
    score = W[:, :, ring].sum(-1); blk, h = np.unravel_index(score.argmax(), score.shape)
    blk = BLOCK if os.environ.get("BLOCK") else int(blk); h = int(score[blk].argmax()); w = W[blk]
    s_h = m.vector_field.angular_attention[blk].log_scale.exp().clamp(max=1000).tolist()
    rgb = np.ones((P, 3)); alpha = np.zeros(P)
    for hh in range(w.shape[0]):
        wn = w[hh] / w[hh].max(); c = np.array(to_rgb(HEADC[hh]))
        take = wn > alpha; rgb[take] = c; alpha[take] = wn[take]
    axes[2].scatter(x, y, c="#eeeeee", s=6)
    axes[2].scatter(x, y, c=rgb, alpha=np.clip(alpha, 0, 1), s=14 * (0.3 + alpha)); frame(axes[2], n)
    axes[2].set_title(f"pose-token windows, block {blk}, 8 heads\ns_h {min(s_h):.0f} to {max(s_h):.0f}", fontsize=9, loc="left")
    axes[3].scatter(x, y, c=dev, cmap="RdBu_r", vmin=-v, vmax=v, s=9, alpha=0.35)
    for k, frac in enumerate((0, 0.25, 0.5, 0.75)):
        Rk = rz(2 * math.pi / n * frac) @ R0
        wk = weights(Rk)[blk, h]; wk = wk / wk.max()
        top = wk > 0.25
        axes[3].scatter(x[top], y[top], s=40 * wk[top], facecolors="none", edgecolors=HEADC[k * 2], lw=1.2,
                        label=f"hypothesis turned {frac * 360 / n:.0f}°")
    frame(axes[3], n); axes[3].legend(fontsize=6.5, loc="lower left", frameon=False, bbox_to_anchor=(-0.05, -0.02), ncol=2)
    axes[3].set_title(f"head {h} (s_h {s_h[h]:.0f}, {score[blk, h]:.0%} of its weight on the ring)\nas the pose hypothesis turns by a quarter period", fontsize=9, loc="left")
    return dict(label=label, n=n, channel=ch, s_h=s_h, block=blk, head=h, score=float(score[blk, h]))

if __name__ == "__main__":
    models = [m.split("|") for m in os.environ["MODELS"].split(";")]
    fig, ax = plt.subplots(len(models), 4, figsize=(15, 4.4 * len(models)), squeeze=False)
    info = [run(lab, ck, int(n), ax[i]) for i, (lab, ck, n) in enumerate(models)]
    fig.tight_layout(); out = os.environ.get("OUT", "windows.png"); fig.savefig(out, dpi=150, facecolor="white"); print(out, json.dumps(info))
