"""Data for the 3D picture of the von Mises-Fisher window logic (viz3d_plot.py). For one checkpoint (env CKPT, NGON,
COSDIR_SNAPSHOT, BDATA, EXTRA_ARGS) and test image 0 it saves to OUT (.npz):
  pts (P, 3)      the image tokens' directions on the hemisphere (z = optical axis);
  dev (P,)        the turn-following token feature for this image: the scalar channel with the largest frequency-n
                  component that turns with the object (over 300 test images), minus its mean;
  W (B, H, P)     the pose token's attention weights over the image tokens, every block B and head H, at the true pose;
  ring_block, ring_head  the head with the most weight on the ring where the n-lobed pattern lives (20-65 deg);
  slide (4, P)    that head's weights for pose hypotheses turned about the optical axis by 0, 1/4, 1/2, 3/4 of 360/n;
  s_h (H,)        that block's inverse temperatures; turn, n."""
import os, sys, math
os.environ.setdefault("COSDIR_DATA", "/nonexistent")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
import lib
from probe import pose_tokens, stream
from attn import parts

D, n, NIMG = os.environ["BDATA"], int(os.environ["NGON"]), 300
lib.CKPT = os.environ["CKPT"]; torch.set_num_threads(4)
m = lib.load_model(fused=False)
rots = np.load(D + "/test/rotations.npz")[f"pyr{n}"][:NIMG]
theta = np.arctan2(rots[:, 0, 1, 0], rots[:, 0, 0, 0])
with torch.no_grad():
    cond = torch.cat([m.condition(torch.from_numpy(np.stack([np.asarray(Image.open(D + f"/test/images/pyr{n}_{j:05d}.png").convert("RGB"))
                      for j in range(i, min(i + 25, NIMG))])).permute(0, 3, 1, 2).float() / 255) for i in range(0, NIMG, 25)])
P = cond.shape[1] - 1; pts = cond[0, :P, 1:4].numpy(); sig = cond[:, :P, 8:72].double().numpy()
coh = (sig * np.exp(1j * n * theta)[:, None, None]).mean(0)
ch = int(np.abs(coh).sum(0).argmax()); img = 0
dev = sig[img, :, ch] - sig[:, :, ch].mean(0)

def rz(phi):
    c, s = math.cos(phi), math.sin(phi)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

def weights(R):
    out = []
    hooks = [a.register_forward_pre_hook(lambda mod, args, out=out: out.append(parts(mod, args[0], args[1], args[3], args[4])))
             for a in m.vector_field.angular_attention]
    with torch.no_grad():
        stream(m.vector_field, pose_tokens(R[None]), torch.full((1,), 0.9), cond[img:img + 1].float())
    for h in hooks: h.remove()
    return np.stack([(o[0] + o[1] + o[2]).softmax(-1)[0, :, 0, 1:P + 1].numpy() for o in out])

R0 = torch.from_numpy(rots[img, 0].astype(np.float32))
W = weights(R0)
pol = np.degrees(np.arccos(np.clip(pts[:, 2], -1, 1))); ring = (pol >= 20) & (pol <= 65)
score = W[:, :, ring].sum(-1); b, h = np.unravel_index(score.argmax(), score.shape)
slide = np.stack([weights(rz(2 * math.pi / n * f) @ R0)[b, h] for f in (0, 0.25, 0.5, 0.75)])
s_h = m.vector_field.angular_attention[int(b)].log_scale.exp().clamp(max=1000).detach().numpy()
np.savez(os.environ["OUT"], pts=pts, dev=dev, W=W, ring_block=int(b), ring_head=int(h), slide=slide, s_h=s_h,
         turn=float(theta[img]), n=n, ring_share=float(score[b, h]))
print("saved", os.environ["OUT"], "block", int(b), "head", int(h), "ring share %.2f" % score[b, h])
