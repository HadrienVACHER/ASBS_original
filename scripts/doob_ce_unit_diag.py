""" diagnostic of the failed unit test of rules_doob_ce.md (post hoc, descriptive): same seeds and settings, adds the exact controller and a breakdown of the error. python scripts/doob_ce_unit_diag.py (CPU, < 1 min) """
import sys, math, json, torch, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import doob_ce as D, smc_noise as S
from adjoint_samplers.components.sde import VESDE, ControlledSDE
from adjoint_samplers.utils import train_utils
class Zero(torch.nn.Module):
    def forward(self, t, x): return torch.zeros_like(x)
mt = torch.tensor([1.0, 0.0]); s = 0.7
class E2:
    def eval(self, x): return 0.5 * ((x - mt) ** 2).sum(-1) / s ** 2 + 2 * math.log(s) + math.log(2 * math.pi)
torch.manual_seed(0)
c = S.Cell.__new__(S.Cell); c.dev = "cpu"; c.D = 2; c.ts = train_utils.get_timesteps(t0=0.0, t1=1.0, steps=200, rescale_t=None); c.NT = len(c.ts) - 1; c.energy = E2(); ref = VESDE(0.001, 2.0); c.sde = ControlledSDE(ref, Zero()); c.sde.train(False)
X, E, lw, _, _ = D.simulate(c, 100000, "cpu", chunk=20000); net = D.new_net(2); D.train(c, net, X, E, lw, 5000, 1024, log_every=10**9)
dts = c.ts[1:] - c.ts[:-1]; g = torch.stack([ref.diff(t) for t in c.ts[:-1]]); R = torch.flip(torch.cumsum(torch.flip(g ** 2 * dts, [0]), 0), [0]); V = R[0]; lam = 1 / R + 1 / s ** 2 - 1 / V
def vstar(x, k): mu = (x / R[k] + mt / s ** 2) / lam[k]; return g[k] * (mu - x) / R[k]
class Exact(torch.nn.Module):
    def forward(self, t, x): k = int(torch.argmin((c.ts[:-1] - t).abs())); return vstar(x, k)
Xv, _, lwv, Xe, _ = D.simulate(c, 4000, "cpu", chunk=4000)
with torch.no_grad():
    vs = torch.stack([vstar(Xv[:, k], k) for k in range(c.NT)], 1); vh = net(c.ts[:-1].repeat(4000), Xv.reshape(-1, 2)).reshape(vs.shape)
d2 = ((vh - vs) ** 2).sum(-1) * dts; n2 = (vs ** 2).sum(-1) * dts; w = torch.softmax(lwv, 0).float(); out = {}
out["rel_err_Q"] = float((d2.sum() / n2.sum()).sqrt()); out["rel_err_P"] = float(((w[:, None] * d2).sum() / (w[:, None] * n2).sum()).sqrt())
for a, b in ((0, 50), (50, 100), (100, 150), (150, 199)):
    out[f"steps_{a}_{b}"] = dict(err_Q=float((d2[:, a:b].sum() / n2[:, a:b].sum()).sqrt()), share_of_err2_Q=float(d2[:, a:b].sum() / d2.sum()))
# by distance of the current state to the P* marginal (Mahalanobis, from the weighted pool)
wm = torch.softmax(lw, 0).float(); m_ = torch.einsum("i,ikd->kd", wm, X); sd = torch.einsum("i,ikd->kd", wm, (X - m_) ** 2).sqrt().clamp(min=1e-6); z = (((Xv - m_) / sd) ** 2).sum(-1).sqrt(); z[:, 0] = 0
for lo, hi in ((0, 2), (2, 3), (3, 1e9)):
    m = (z >= lo) & (z < hi); out[f"mahalanobis_{lo}_{hi}"] = dict(frac_points=float(m.float().mean()), share_of_err2_Q=float(d2[m].sum() / d2.sum()), err_Q=float((d2[m].sum() / n2[m].sum()).sqrt()))
for name, ctrl in (("exact_vstar", Exact()), ("learned", net)):
    c.sde = ControlledSDE(ref, D.Corrected(Zero(), [ctrl], ref)); c.sde.train(False); torch.manual_seed(5); l2, X2 = c.forward(c.noise(20000))
    out[name] = dict(ess=S.ess_rel(l2), mean=X2.mean(0).tolist(), std=X2.std(0).tolist(), logz=float(torch.logsumexp(l2, 0) - math.log(len(l2))))
json.dump(out, open("figures/doob_ce/unit_diag.json", "w"), indent=1); print(json.dumps(out, indent=1))
