""" rules/rules_doob_ce.md: correction of the dynamics of a frozen AS controller by path-weighted regression of the Brownian increments (PICE / path cross-entropy).
    Repository scheme: X_{k+1} = X_k + g_k^2 u dt_k + g_k sqrt(dt_k) eps_k, so u_new = u_theta + v_phi / g_k.
    python scripts/doob_ce.py unit                                   Gaussian target, zero control, closed-form v* (CPU, < 1 min)
    python scripts/doob_ce.py smoke [--device cuda]                  environment / plumbing / timing
    python scripts/doob_ce.py run CELL [--M 10000] [--it 1]          one iteration (iteration j loads the corrections of 1..j-1 at the same M)
    python scripts/doob_ce.py harm CELL                              descriptive arm (h): regression of log h
    python scripts/doob_ce.py verdict
    CELL in {A, B, C} """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import dart_lib as L
import smc_noise as S
from adjoint_samplers.components.model import FourierMLP
from adjoint_samplers.components.sde import VESDE, ControlledSDE
from adjoint_samplers.utils import train_utils
OUT = ROOT / "figures" / "doob_ce"
EPOCH = dict(A=1500, B=1500, C=250, d8=0); STEPS, BATCH, LR, NEVAL, NVAL, EVAL_SEED = 30000, 4096, 1e-3, 10000, 2000, 12345


def new_net(D, out=None, device="cpu"):
    z = torch.nn.init.zeros_
    return FourierMLP(dim=D, num_layers=4, channels=128, activation=torch.nn.GELU(), last_bias_init=z, last_weight_init=z, dim_out=out).to(device)


class Corrected(torch.nn.Module):
    """ u_theta + sum_j v_j / g(t) """
    def __init__(self, base, vs, ref):
        super().__init__(); self.base, self.vs, self.ref = base, torch.nn.ModuleList(vs), ref

    def forward(self, t, x):
        u = self.base(t, x)
        for v in self.vs: u = u + v(t, x) / self.ref.diff(t)
        return u


class GradCtrl(torch.nn.Module):
    """ arm (h): u = grad_x f """
    def __init__(self, f):
        super().__init__(); self.f = f

    def forward(self, t, x):
        with torch.enable_grad():
            y = x.detach().requires_grad_(True); g, = torch.autograd.grad(self.f(t, y).sum(), y)
        return g.detach()


@torch.no_grad()
def simulate(cell, M, store, chunk=10000, keep_cum=False):
    """ states X (M, NT, D), innovations E (M, NT, D, half precision), path log weights, endpoints; same formula as Cell.forward """
    NT, D = cell.NT, cell.D
    X = torch.empty(M, NT, D, device=store); E = torch.empty(M, NT, D, device=store, dtype=torch.float16); lw = torch.empty(M, dtype=torch.float64); Xend = torch.empty(M, D)
    C = torch.empty(M, NT, device=store) if keep_cum else None
    for a in range(0, M, chunk):
        N = min(chunk, M - a); x = torch.zeros(N, D, device=cell.dev); lr = torch.zeros(N, device=cell.dev); V = 0.0
        Xc = torch.empty(N, NT, D, device=cell.dev); Ec = torch.empty_like(Xc); Cc = torch.empty(N, NT, device=cell.dev)
        for i in range(NT):
            t = cell.ts[i]; dt = cell.ts[i + 1] - t; v = cell.sde.diff(t) ** 2 * dt; m = cell.sde.drift(t, x) * dt; e = torch.randn(N, D, device=cell.dev)
            Xc[:, i] = x; Ec[:, i] = e; Cc[:, i] = lr; lr += 0.5 * (m ** 2).sum(-1) / v + (e * m).sum(-1) / v.sqrt(); x = cell.sde.propagate(x, m + v.sqrt() * e); V += float(v)
        lref = -0.5 * (x ** 2).sum(-1) / V - 0.5 * D * math.log(2 * math.pi * V)
        lw[a:a + N] = (-lr - cell.energy.eval(x) - lref).double().cpu(); Xend[a:a + N] = x.cpu(); X[a:a + N] = Xc.to(store); E[a:a + N] = Ec.half().to(store)
        if keep_cum: C[a:a + N] = Cc.to(store)
    return X, E, lw, Xend, C


def train(cell, net, X, E, lw, steps, batch, C=None, log_every=1000):
    """ weighted regression; trajectories drawn with probability omega, time steps uniform. C given: arm (h), target y = lw + cum """
    store = X.device; cdf = torch.cumsum(torch.softmax(lw, 0), 0).to(store); NT = cell.NT; ts = cell.ts.to(cell.dev); dts = (ts[1:] - ts[:-1])
    opt = torch.optim.Adam(net.parameters(), lr=LR, fused=(cell.dev == "cuda")); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lwd = lw.float().to(store) if C is not None else None; curve = []; acc = torch.zeros((), device=cell.dev); t0 = time.time()
    for s in range(steps):
        i = torch.searchsorted(cdf, torch.rand(batch, device=store, dtype=torch.float64)).clamp(max=len(lw) - 1); k = torch.randint(NT, (batch,), device=store)
        x = X[i, k].to(cell.dev); kd = k.to(cell.dev); t = ts[kd]; dt = dts[kd]
        if C is None:
            e = E[i, k].to(cell.dev).float(); loss = ((e - dt.sqrt()[:, None] * net(t, x)) ** 2).sum(-1).mean()
        else:
            y = (lwd[i] + C[i, k]).to(cell.dev); loss = (dt * (net(t, x)[:, 0] - y) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); acc += loss.detach()
        if (s + 1) % log_every == 0:
            curve.append(float(acc) / log_every); acc.zero_(); print(f"  step {s + 1}/{steps} loss {curve[-1]:.5f} t {time.time() - t0:.0f}s", flush=True)
    return curve


@torch.no_grad()
def reg_loss(cell, net, X, E, lw, chunk=100):
    """ exact weighted loss over a pool, minus the loss of v = 0 (negative = better than no correction) """
    w = torch.softmax(lw, 0); ts = cell.ts.to(cell.dev); dts = ts[1:] - ts[:-1]; tot = 0.0; NT = cell.NT
    for a in range(0, len(lw), chunk):
        x = X[a:a + chunk].to(cell.dev); e = E[a:a + chunk].to(cell.dev).float(); n = len(x); t = ts[:-1].repeat(n); v = net(t, x.reshape(n * NT, -1)).reshape(n, NT, -1)
        d = ((e - dts.sqrt()[None, :, None] * v) ** 2).sum((-1, -2)) - (e ** 2).sum((-1, -2)); tot += float((w[a:a + n] * d.double().cpu()).sum())
    return tot


def voronoi_tv(cell, X):
    b = torch.cdist(X.double(), cell.mu.double()).argmin(1); p = S.props(b, cell.K); return L.tv(p, cell.pt), p.tolist()


@torch.no_grad()
def evaluate(cell, n=NEVAL, chunk=2500):
    """ n raw samples under the current controller (common evaluation noise) """
    torch.manual_seed(EVAL_SEED); lws, xs = [], []; t0 = time.time()
    for a in range(0, n, chunk):
        lw, X = cell.forward(cell.noise(min(chunk, n - a))); lws.append(lw.cpu()); xs.append(X)
    lw = torch.cat(lws); X = torch.cat(xs); ts = time.time() - t0; b = S.basins(cell, X); p = S.props(b, cell.K); vtv, vp = voronoi_tv(cell, X); w = torch.softmax(lw, 0); pis = S.props(b.cpu(), cell.K, w)
    return dict(n=n, tv=L.tv(p, cell.pt), p=p.tolist(), tv_voronoi=vtv, p_voronoi=vp, frac_assigned=float((b >= 0).double().mean()), ess=S.ess_rel(lw), khat=L.khat(lw.numpy()), logz=float(torch.logsumexp(lw, 0) - math.log(n)),
                tv_is=L.tv(pis, cell.pt), time_sim_s=ts, **S.metrics(cell, X, b))


def weight_stats(cell, lw, Xend):
    b = torch.cat([S.basins(cell, Xend[a:a + 20000].to(cell.dev)).cpu() for a in range(0, len(Xend), 20000)]); w = torch.softmax(lw, 0); m = b >= 0
    return dict(ess=S.ess_rel(lw), ess_abs=S.ess_rel(lw) * len(lw), khat=L.khat(lw.numpy()), max_w=float(w.max()), logz=float(torch.logsumexp(lw, 0) - math.log(len(lw))), counts=torch.bincount(b[m], minlength=cell.K).tolist(),
                weighted_mass=S.props(b, cell.K, w).tolist(), weighted_unassigned=float(w[~m].sum()))


def load_cell(name, device, M, it):
    c = S.Cell(name, EPOCH[name], device); base = c.sde.u; vs = []
    for j in range(1, it):
        v = new_net(c.D, device=device); v.load_state_dict(torch.load(OUT / f"v_{name}_M{M}_it{j}.pt", map_location=device, weights_only=False)); v.eval(); vs.append(v)
    c.sde.u = Corrected(base, vs, c.sde.ref_sde).to(device); c.sde.train(False); return c, base, vs


def dump(name, out):
    OUT.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT / name, "w"), indent=1, default=float)


def unit(a):
    class Zero(torch.nn.Module):
        def forward(self, t, x): return torch.zeros_like(x)

    mt = torch.tensor([1.0, 0.0]); s = 0.7

    class E2:
        def eval(self, x): return 0.5 * ((x - mt) ** 2).sum(-1) / s ** 2 + 2 * math.log(s) + math.log(2 * math.pi)

    torch.manual_seed(0); t0 = time.time()
    c = S.Cell.__new__(S.Cell); c.dev = "cpu"; c.D = 2; c.ts = train_utils.get_timesteps(t0=0.0, t1=1.0, steps=200, rescale_t=None); c.NT = len(c.ts) - 1; c.energy = E2(); ref = VESDE(0.001, 2.0); c.sde = ControlledSDE(ref, Zero()); c.sde.train(False)
    X, E, lw, Xend, _ = simulate(c, a.M or 100000, "cpu", chunk=20000); net = new_net(2); curve = train(c, net, X, E, lw, a.steps or 5000, 1024)
    dts = c.ts[1:] - c.ts[:-1]; g = torch.stack([ref.diff(t) for t in c.ts[:-1]]); vk = g ** 2 * dts; R = torch.flip(torch.cumsum(torch.flip(vk, [0]), 0), [0]); V = R[0]
    Xv, _, lwv, _, _ = simulate(c, 4000, "cpu", chunk=4000)
    with torch.no_grad():
        lam = 1 / R + 1 / s ** 2 - 1 / V; mu = (Xv / R[None, :, None] + mt[None, None, :] / s ** 2) / lam[None, :, None]; vs_ = g[None, :, None] * (mu - Xv) / R[None, :, None]
        vh = net(c.ts[:-1].repeat(4000), Xv.reshape(-1, 2)).reshape(vs_.shape); d2 = (((vh - vs_) ** 2).sum(-1) * dts).sum(-1).double(); n2 = ((vs_ ** 2).sum(-1) * dts).sum(-1).double(); w = torch.softmax(lwv, 0)
    eq = float((d2.sum() / n2.sum()).sqrt()); ep = float(((w * d2).sum() / (w * n2).sum()).sqrt())
    c.sde = ControlledSDE(ref, Corrected(Zero(), [net], ref)); c.sde.train(False); lw2, X2 = c.forward(c.noise(20000))
    res = dict(rel_err_Q=eq, rel_err_P=ep, ok=bool(eq <= 0.25), mean=X2.mean(0).tolist(), std=X2.std(0).tolist(), target_mean=mt.tolist(), target_std=s, ess_before=S.ess_rel(lw), ess_after=S.ess_rel(lw2), logz_after=float(torch.logsumexp(lw2, 0) - math.log(len(lw2))),
               curve=curve, M=len(lw), steps=a.steps or 5000, time_s=time.time() - t0)
    dump("unit.json", res); print({k: v for k, v in res.items() if k != "curve"}); sys.exit(0 if res["ok"] else "unit test failed")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("cell", nargs="?"); ap.add_argument("--M", type=int, default=None); ap.add_argument("--it", type=int, default=1); ap.add_argument("--steps", type=int, default=None); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "unit": unit(a)
    elif a.cmd == "smoke":
        t0 = time.time()
        if a.device == "cuda": assert torch.cuda.is_available(), "no CUDA device"
        cells = {n: S.Cell(n, EPOCH[n], a.device) for n in ("A", "B", "C", "d8")}; c = cells["d8"]; torch.manual_seed(1)
        X, E, lw, Xend, C = simulate(c, 256, a.device, keep_cum=True); net = new_net(c.D, device=a.device); train(c, net, X, E, lw, 20, 256, log_every=10); reg_loss(c, net, X, E, lw); weight_stats(c, lw, Xend)
        f = new_net(c.D, 1, a.device); train(c, f, X, E, lw, 10, 256, C=C, log_every=10); c.sde.u = GradCtrl(f); c.forward(c.noise(64))
        # same noise: simulate() and Cell.forward give the same log weights
        c = cells["B"]; torch.manual_seed(7); _, _, l1, x1, _ = simulate(c, 128, a.device); torch.manual_seed(7); xi = torch.stack([torch.randn(128, c.D, device=a.device) for _ in range(c.NT)]); l2, x2 = c.forward(xi)
        diff = float((x1 - x2.cpu()).abs().max()); assert diff < 1e-3, diff
        torch.manual_seed(2); t = time.time(); X, E, lw, Xend, _ = simulate(c, 10000, a.device); tsim = time.time() - t; net = new_net(c.D, device=a.device); train(c, net, X, E, lw, 200, BATCH, log_every=200); t = time.time(); train(c, net, X, E, lw, 1000, BATCH, log_every=1000); tstep = (time.time() - t) / 1000
        del X, E; X, E, lw, _, _ = simulate(c, 2000, "cpu"); t = time.time(); train(c, net, X, E, lw, 500, BATCH, log_every=500); tstep_cpu = (time.time() - t) / 500
        base = c.sde.u; c.sde.u = Corrected(base, [net], c.sde.ref_sde); t = time.time(); c.forward(c.noise(2500)); teval = (time.time() - t) * 4
        out = dict(ok=True, same_noise_max_diff=diff, sim10k_s=tsim, train_step_s=tstep, train_step_cpu_store_s=tstep_cpu, eval10k_corrected_s=teval, gpu=(torch.cuda.get_device_name(0) if a.device == "cuda" else None), torch=torch.__version__, time_s=time.time() - t0)
        dump("smoke.json", out); print("SMOKE OK", out)
    elif a.cmd == "run":
        t0 = time.time(); M = a.M or 10000; name = a.cell; tag = f"{name}_M{M}_it{a.it}"; c, base, vs = load_cell(name, a.device, M, a.it)
        if a.it == 1 and not (OUT / f"as_{name}.json").exists():
            r = evaluate(c); rng = np.random.default_rng(5); pt = c.pt; r["floor_tv"] = float(np.median([L.tv(rng.multinomial(NEVAL, pt / pt.sum()) / NEVAL, pt) for _ in range(200)])); r["p_true"] = pt.tolist(); dump(f"as_{name}.json", r); print("AS", {k: r[k] for k in ("tv", "tv_voronoi", "q1", "ess", "logz")}, flush=True)
        torch.manual_seed(1000 * ord(name[0]) + 10 * a.it + M % 997); store = a.device if M <= 20000 else "cpu"; t = time.time(); X, E, lw, Xend, _ = simulate(c, M, store); tsim = time.time() - t
        ws = weight_stats(c, lw, Xend); print("weights", {k: ws[k] for k in ("ess", "khat", "max_w", "logz")}, flush=True)
        torch.manual_seed(a.it + M % 991); net = new_net(c.D, device=a.device); t = time.time(); curve = train(c, net, X, E, lw, a.steps or STEPS, BATCH); ttr = time.time() - t
        net.eval(); ltr = reg_loss(c, net, X[:20000], E[:20000], lw[:20000]); del X, E
        torch.manual_seed(99 + a.it); Xv, Ev, lwv, _, _ = simulate(c, NVAL, a.device); lval = reg_loss(c, net, Xv, Ev, lwv); lval_ess = S.ess_rel(lwv); del Xv, Ev
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, OUT / f"v_{tag}.pt"); c.sde.u = Corrected(base, vs + [net], c.sde.ref_sde); c.sde.train(False)
        r = evaluate(c)
        out = dict(cell=name, epoch=EPOCH[name], M=M, it=a.it, steps=a.steps or STEPS, batch=BATCH, lr=LR, p_true=c.pt.tolist(), train_weights=ws, curve=curve, loss_minus_zero_train=ltr, loss_minus_zero_val=lval, val_ess=lval_ess, new=r, time_sim_s=tsim, time_train_s=ttr, time_s=time.time() - t0)
        dump(f"run_{tag}.json", out); print(tag, "new", {k: r[k] for k in ("tv", "tv_voronoi", "q1", "ess", "logz")}, "loss-zero train %.4f val %.4f" % (ltr, lval), flush=True)
    elif a.cmd == "harm":
        t0 = time.time(); M = a.M or 10000; name = a.cell; c, base, _ = load_cell(name, a.device, M, 1); torch.manual_seed(1000 * ord(name[0]) + 10 + M % 997)
        X, E, lw, Xend, C = simulate(c, M, a.device, keep_cum=True); ws = weight_stats(c, lw, Xend); f = new_net(c.D, 1, a.device); torch.manual_seed(1 + M % 991); curve = train(c, f, X, E, lw, a.steps or STEPS, BATCH, C=C); del X, E, C
        f.eval(); torch.save({k: v.cpu() for k, v in f.state_dict().items()}, OUT / f"logh_{name}_M{M}.pt"); c.sde.u = GradCtrl(f); c.sde.train(False); r = evaluate(c)
        dump(f"harm_{name}_M{M}.json", dict(cell=name, M=M, train_weights=ws, curve=curve, new=r, time_s=time.time() - t0)); print("h", name, {k: r[k] for k in ("tv", "tv_voronoi", "q1", "ess", "logz")}, flush=True)
    elif a.cmd == "verdict":
        V = {}
        for n in "ABC":
            fa, fr = OUT / f"as_{n}.json", OUT / f"run_{n}_M10000_it1.json"
            if not (fa.exists() and fr.exists()): V[n] = "non exécuté"; continue
            a_, j = json.load(open(fa)), json.load(open(fr)); r = j["new"]
            V[n] = dict(tv_as=a_["tv"], tv_new=r["tv"], q1_as=a_["q1"], q1_new=r["q1"], ess_as=a_["ess"], ess_new=r["ess"], ess_train=j["train_weights"]["ess"], H=bool(r["tv"] <= 0.5 * a_["tv"] and r["q1"] <= 1.1 * a_["q1"]), mechanism=bool(j["train_weights"]["ess"] < 0.01))
        for h, n in (("H1", "A"), ("H2", "B"), ("H3", "C")): V[h] = V[n]["H"] if isinstance(V[n], dict) else None
        V["H4"] = bool(V["A"]["ess_new"] >= V["A"]["ess_as"]) if isinstance(V["A"], dict) else None
        dump("verdict.json", V); print(json.dumps(V, indent=1))


if __name__ == "__main__":
    main()
