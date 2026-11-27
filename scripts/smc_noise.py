""" rules/rules_smc_noise.md: SMC in the space of the Brownian innovations of a frozen AS controller, with tempering of the path log weight.
    python scripts/smc_noise.py unit                                  zero control on a 2-d Gaussian target (Z = 1)
    python scripts/smc_noise.py quenchtest                            torch quench against basin_lib
    python scripts/smc_noise.py trial CELL EPOCH [--M 256] [--device cpu]     one adaptive pass, timing per level
    python scripts/smc_noise.py run CELL EPOCH REP [--M 2048] [--device cuda]  adaptive pass, frozen-calendar pass, equal-cost path IS
    python scripts/smc_noise.py smoke [--device cuda]                            environment / GPU / paths check, < 5 min
    python scripts/smc_noise.py darting CELL EPOCH REP                local CPU: bank + darting on the initial particles
    python scripts/smc_noise.py verdict
    CELL in {A, B, C, d8} """
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import gmm_dk, dart_lib as L
TAU, KP, ACC, GAIN, MAXLEV, EPS = 0.5, 5, 0.25, 2.0, 500, 1e-6
OUT = ROOT / "figures" / "smc_noise"; RES = ROOT / "results" / "smc_noise"


class Cell:
    def __init__(self, name, epoch, device):
        import train, hydra  # noqa: registers resolvers
        from omegaconf import OmegaConf
        from adjoint_samplers.components.sde import ControlledSDE
        from adjoint_samplers.utils import train_utils
        F = ROOT / "figures"
        if name in "AB":
            cfg = OmegaConf.load(F / "gmmdk_step4/configs/gmmdk4_d128_sx0.5.config.yaml"); sx = "0.5" if name == "A" else "0.75"
            if name == "B": cfg.sigma_max = float(json.load(open(F / "gmmdk_step4/q_gmmdk4_d128_sx0.75.json"))["sigma_max"])
            ck = F / f"gmmdk_step4/controllers/gmmdk4_d128_sx{sx}_ep{epoch}.pt"
        elif name == "C":
            cfg = OmegaConf.load(F / "smc_noise_inputs/config.yaml"); ck = F / f"smc_noise_inputs/controller_ep{epoch}.pt"
        else:
            cfg = OmegaConf.load(F / "smc_noise_inputs/d8/config.yaml"); ck = F / "smc_noise_inputs/d8/controller_final.pt"
        OmegaConf.resolve(cfg); self.dev = device; self.name = name; self.D = int(cfg.dim)
        self.ts = train_utils.get_timesteps(**cfg.timesteps).to(device); self.energy = hydra.utils.instantiate(cfg.energy, device=device)
        ref = hydra.utils.instantiate(cfg.ref_sde).to(device); ctrl = hydra.utils.instantiate(cfg.controller).to(device)
        ctrl.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["controller"]); self.sde = ControlledSDE(ref, ctrl).to(device); self.sde.train(False)
        self.NT = len(self.ts) - 1; self.nu1 = float(sum(self.sde.diff(self.ts[i]) ** 2 * (self.ts[i + 1] - self.ts[i]) for i in range(self.NT)))
        dist = self.energy.dist; self.mos = name == "C"; self.K = int(dist.means.shape[0])
        self.mu = dist.means.to(device).float(); self.logw = dist.weights.to(device).float().log()
        if self.mos:
            self.pt = np.array(json.load(open(F / f"mog_mos_truth_mosw_d{self.D}.json"))["p"])
        else:
            self.pt = gmm_dk.true_proportions(self.D, self.K, float(cfg.delta)).numpy()
        self.pt = self.pt / self.pt.sum()
        Y, conv = quench(self, self.mu.clone(), tol=1e-4, maxit=5000); assert bool(conv.all()), "minima not converged"; self.M = Y

    @torch.no_grad()
    def forward(self, xi, chunk=20000):
        """ xi (NT, N, D) -> (log weights, endpoints): same formula as girsanov.log_path_weights with the innovations given """
        lws, xs = [], []
        for a in range(0, xi.shape[1], chunk):
            e = xi[:, a:a + chunk]; N = e.shape[1]; x = torch.zeros(N, self.D, device=self.dev); lr = torch.zeros(N, device=self.dev); V = 0.0
            for i in range(self.NT):
                t = self.ts[i]; dt = self.ts[i + 1] - t; v = self.sde.diff(t) ** 2 * dt; m = self.sde.drift(t, x) * dt; ei = e[i]
                lr += 0.5 * (m ** 2).sum(-1) / v + (ei * m).sum(-1) / v.sqrt(); x = self.sde.propagate(x, m + v.sqrt() * ei); V += float(v)
            lref = -0.5 * (x ** 2).sum(-1) / V - 0.5 * self.D * math.log(2 * math.pi * V)
            lws.append(-lr - self.energy.eval(x) - lref); xs.append(x)
        return torch.cat(lws).double(), torch.cat(xs)

    def noise(self, N):
        return torch.randn(self.NT, N, self.D, device=self.dev)


@torch.no_grad()
def _lp_grad(cell, Y, chunk=4000):
    lp, g, ks = [], [], []
    for a in range(0, len(Y), chunk):
        y = Y[a:a + chunk]; u = y[:, None, :] - cell.mu[None]
        if cell.mos:
            lc = cell.logw + (-(1.5) * torch.log1p(u * u / 2.0)).sum(-1) + cell.D * (math.lgamma(1.5) - math.lgamma(1.0) - 0.5 * math.log(2 * math.pi))
        else:
            lc = cell.logw - 0.5 * (u * u).sum(-1) - 0.5 * cell.D * math.log(2 * math.pi)
        l = torch.logsumexp(lc, 1); r = torch.exp(lc - l[:, None])
        gi = (r[:, :, None] * (-3.0 * u / (2.0 + u * u) if cell.mos else -u)).sum(1)
        lp.append(l); g.append(gi); ks.append(lc.argmax(1))
    return torch.cat(lp), torch.cat(g), torch.cat(ks)


@torch.no_grad()
def quench(cell, X, maxit=3000, tol=3e-3):
    """ batched version of basin_lib.quench in float64 """
    mu, logw = cell.mu.double(), cell.logw.double(); keep_mu, keep_w = cell.mu, cell.logw; cell.mu, cell.logw = mu, logw
    try:
        Y = X.double().clone(); n = len(Y); s = torch.full((n,), 0.7, device=Y.device, dtype=Y.dtype); conv = torch.zeros(n, dtype=torch.bool, device=Y.device)
        active = torch.arange(n, device=Y.device); lp, g, ks = _lp_grad(cell, Y)
        for _ in range(maxit):
            done = g.norm(dim=1) < (tol if tol >= 1e-3 else 1e-6); conv[active[done]] = True; keep = ~done
            if not keep.any(): break
            active, lp, g, ks, s = active[keep], lp[keep], g[keep], ks[keep], s[keep]; Ya = Y[active]
            P = 1.0 if not cell.mos else (2.0 + (Ya - cell.mu[ks]) ** 2) / 3.0
            Yn = Ya + s[:, None] * P * g; lpn, gn, ksn = _lp_grad(cell, Yn); ok = lpn > lp
            Y[active[ok]] = Yn[ok]; lp = torch.where(ok, lpn, lp); g = torch.where(ok[:, None], gn, g); ks = torch.where(ok, ksn, ks)
            s = torch.where(ok, torch.clamp(1.5 * s, max=0.7), 0.5 * s); stuck = s < 1e-12
            if stuck.any():
                k2 = ~stuck; active, lp, g, ks, s = active[k2], lp[k2], g[k2], ks[k2], s[k2]
                if len(active) == 0: break
        return Y, conv
    finally:
        cell.mu, cell.logw = keep_mu, keep_w


@torch.no_grad()
def basins(cell, X):
    Y, conv = quench(cell, X); d = torch.cdist(Y, cell.M.double()); dm, b = d.min(1)
    return torch.where(conv & (dm < 0.5), b, torch.full_like(b, -1))


def props(b, K, w=None):
    m = b >= 0
    if m.sum() == 0: return np.full(K, np.nan)
    if w is None: return torch.bincount(b[m], minlength=K).double().cpu().numpy() / int(m.sum())
    ww = w[m].double(); return (torch.bincount(b[m], weights=ww, minlength=K) / ww.sum()).cpu().numpy()


def ess_rel(lw):
    return float(torch.exp(2 * torch.logsumexp(lw, 0) - torch.logsumexp(2 * lw, 0)) / len(lw))


def delta_beta(lw, rest):
    if ess_rel(rest * lw) >= TAU: return rest
    lo, hi = 0.0, rest
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if ess_rel(mid * lw) >= TAU else (lo, mid)
    return max(lo, 1e-12)


def systematic(lw):
    n = len(lw); w = torch.softmax(lw, 0); u = (torch.rand(1, device=lw.device).double() + torch.arange(n, device=lw.device)) / n
    return torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=n - 1)


def smc(cell, M, calendar=None, save=True, log=False, kp=KP):
    """ calendar None = adaptive pass; otherwise the frozen list [(beta, rho), ...] of an adaptive pass """
    t0 = time.time(); xi = cell.noise(M); lw, X = cell.forward(xi); nsim = M; beta, logz, rho = 0.0, 0.0, 0.9
    root = torch.arange(M, device=cell.dev); levels = []
    b0 = basins(cell, X); saved = dict(lw=[lw.float().cpu()], basin=[b0.cpu().to(torch.int8)], parent=[], X0=X.float().cpu())
    levels.append(dict(beta=0.0, p=props(b0, cell.K).tolist(), logz=0.0, counts=torch.bincount(b0[b0 >= 0], minlength=cell.K).cpu().tolist(), unassigned=int((b0 < 0).sum())))
    for lev in range(MAXLEV):
        if beta >= 1.0: break
        if calendar is None:
            db = delta_beta(lw, 1.0 - beta); new_beta = min(1.0, beta + db)
        else:
            new_beta, rho = calendar[lev]; db = new_beta - beta
        inc = db * lw; step_logz = float(torch.logsumexp(inc, 0) - math.log(M)); logz += step_logz; ess = ess_rel(inc)
        idx = systematic(inc); xi = xi[:, idx]; lw, X, root = lw[idx], X[idx], root[idx]; beta = new_beta; acc = 0.0
        for _ in range(kp):
            prop = rho * xi + math.sqrt(1 - rho ** 2) * torch.randn_like(xi); lwp, Xp = cell.forward(prop); nsim += M
            ok = torch.log(torch.rand(M, device=cell.dev).double()) < beta * (lwp - lw)
            xi = torch.where(ok[None, :, None], prop, xi); lw = torch.where(ok, lwp, lw); X = torch.where(ok[:, None], Xp, X); acc += float(ok.double().mean()) / kp
        b = basins(cell, X)
        if save: saved["lw"].append(lw.float().cpu()); saved["basin"].append(b.cpu().to(torch.int8)); saved["parent"].append(idx.cpu().to(torch.int32))
        levels.append(dict(beta=beta, dbeta=db, rho=rho, accept=acc, ess_inc=ess, logz=logz, ancestors=int(len(torch.unique(root))), p=props(b, cell.K).tolist(), t=time.time() - t0))
        if log: print(f"  level {lev + 1}: beta {beta:.4f} (d {db:.4g}) rho {rho:.5f} acc {acc:.3f} ess {ess:.3f} logZ {logz:+.3f} ancestors {levels[-1]['ancestors']} t {time.time() - t0:.1f}s", flush=True)
        if calendar is None:
            one_m = math.exp(math.log(max(1 - rho, EPS)) + GAIN * (acc - ACC)); rho = 1 - min(1.0, max(EPS, one_m))
    done = beta >= 1.0
    return dict(logz=logz, beta_end=beta, J=len(levels) - 1, done=done, nsim=nsim, levels=levels, time_s=time.time() - t0), X, saved


def calendar_of(res):
    return [(lv["beta"], lv["rho"]) for lv in res["levels"][1:]]


def metrics(cell, X, b):
    Mmin = cell.M.double(); d = torch.cdist(X.double(), Mmin).min(1).values; rng = np.random.default_rng(2026)
    ex = exact_draws(cell, 10000, rng); dex = torch.cdist(torch.tensor(ex, device=cell.dev), Mmin).min(1).values
    pos = []
    for k in range(cell.K):
        xk = X[b == k].double()
        if len(xk) < 2: pos.append(None); continue
        c = xk.median(0).values if cell.mos else xk.mean(0); pos.append(float((c - Mmin[k]).norm()))
    return dict(q1=float(d.median() / dex.median()), position_error=pos)


def exact_draws(cell, n, rng):
    w = np.exp(cell.logw.cpu().numpy().astype(np.float64)); w /= w.sum(); comp = rng.choice(cell.K, size=n, p=w); mu = cell.mu.cpu().numpy().astype(np.float64)
    if cell.mos:
        return mu[comp] + rng.standard_normal((n, cell.D)) / np.sqrt(rng.gamma(1.0, 1.0, (n, cell.D)))
    return mu[comp] + rng.standard_normal((n, cell.D))


def path_is(cell, n, chunk=4096):
    lws, bs = [], []
    for a in range(0, n, chunk):
        lw, X = cell.forward(cell.noise(min(chunk, n - a))); lws.append(lw.cpu()); bs.append(basins(cell, X).cpu())
    lw = torch.cat(lws); b = torch.cat(bs); w = torch.softmax(lw, 0)
    p = props(b, cell.K, w); return dict(n=n, logz=float(torch.logsumexp(lw, 0) - math.log(n)), ess=ess_rel(lw), p=p.tolist(), tv=L.tv(p, cell.pt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("cell", nargs="?"); ap.add_argument("epoch", type=int, nargs="?"); ap.add_argument("rep", type=int, nargs="?"); ap.add_argument("--M", type=int, default=2048); ap.add_argument("--rho", type=float, default=0.9); ap.add_argument("--chains", type=int, default=512); ap.add_argument("--steps", type=int, default=500); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "unit":
        from adjoint_samplers.components.sde import VESDE, ControlledSDE
        from adjoint_samplers.utils import train_utils

        class Zero(torch.nn.Module):
            def forward(self, t, x): return torch.zeros_like(x)

        class E2:
            def eval(self, x):
                mu = torch.tensor([1.0, 0.0]); s = 0.7; return 0.5 * ((x - mu) ** 2).sum(-1) / s ** 2 + 2 * math.log(s) + math.log(2 * math.pi)

        c = Cell.__new__(Cell); c.dev = "cpu"; c.D = 2; c.ts = train_utils.get_timesteps(t0=0.0, t1=1.0, steps=200, rescale_t=None); c.NT = len(c.ts) - 1; c.energy = E2()
        c.sde = ControlledSDE(VESDE(0.001, 2.0), Zero()); c.sde.train(False); c.K = 1; c.mos = False; c.mu = torch.tensor([[1.0, 0.0]]); c.logw = torch.zeros(1); c.M = c.mu.clone(); c.pt = np.ones(1)
        # the unit test only needs Z: basins use a single mode of unit variance (not used)
        globals()["basins"] = lambda cell, X: torch.zeros(len(X), dtype=torch.long)
        torch.manual_seed(0); za, zf, J = [], [], []
        for r in range(20):
            ra, _, _ = smc(c, 512, save=False); za.append(math.exp(ra["logz"])); J.append(ra["J"])
            rf, _, _ = smc(c, 512, calendar=calendar_of(ra), save=False); zf.append(math.exp(rf["logz"]))
        res = {}
        for n, z in (("adaptive", za), ("frozen", zf)):
            m, se = float(np.mean(z)), float(np.std(z, ddof=1) / math.sqrt(len(z))); res[n] = dict(mean_z=m, se=se, ok=bool(abs(m - 1) <= 3 * se)); print(n, f"mean Z {m:.4f} +- {se:.4f} ok {res[n]['ok']}")
        res["levels"] = J; res["ok"] = res["adaptive"]["ok"] and res["frozen"]["ok"]; json.dump(res, open(OUT / "unit.json", "w"), indent=1)
        sys.exit(0 if res["ok"] else "unit test failed")
    elif a.cmd == "quenchtest":
        import basin_lib as B
        res = {}
        for nm, ep in (("B", 1500), ("C", 250)):
            c = Cell(nm, ep, "cpu"); torch.manual_seed(3); lw, X = c.forward(c.noise(2000)); x = X.double().numpy()
            mu = c.mu.double().numpy(); wt = np.exp(c.logw.double().numpy()); pi = L.Mix("S" if c.mos else "G", mu, wt / wt.sum(), 1.0)
            M = B.polish_minima(pi); b0, _, _ = B.basins(pi, x, M); b1 = basins(c, X).numpy(); res[nm] = dict(agreement=float((b0 == b1).mean()), assigned_ref=float((b0 >= 0).mean())); print(nm, res[nm], flush=True)
        res["ok"] = all(v["agreement"] >= 0.999 for v in res.values()); json.dump(res, open(OUT / "quenchtest.json", "w"), indent=1); sys.exit(0 if res["ok"] else "quench test failed")
    elif a.cmd == "trial":
        torch.manual_seed(11); c = Cell(a.cell, a.epoch, a.device); r, X, _ = smc(c, a.M, save=False, log=True)
        out = dict(cell=a.cell, M=a.M, J=r["J"], done=r["done"], logz=r["logz"], time_s=r["time_s"], time_per_level_s=r["time_s"] / max(1, r["J"]), nsim=r["nsim"], D=c.D)
        print(out); json.dump(dict(out, levels=r["levels"]), open(OUT / f"trial_{a.cell}.json", "w"), indent=1)
    elif a.cmd == "run":
        torch.manual_seed(1000 * a.rep + a.epoch + ord(a.cell[0])); c = Cell(a.cell, a.epoch, a.device); RES.mkdir(parents=True, exist_ok=True); name = f"{a.cell}_ep{a.epoch}_r{a.rep}"
        r1, X1, s1 = smc(c, a.M); print(name, f"pass1 J {r1['J']} logZ {r1['logz']:+.4f} t {r1['time_s']:.0f}s", flush=True)
        r2, X2, s2 = smc(c, a.M, calendar=calendar_of(r1)); print(name, f"pass2 logZ {r2['logz']:+.4f} t {r2['time_s']:.0f}s", flush=True)
        out = dict(cell=a.cell, epoch=a.epoch, rep=a.rep, M=a.M, p_true=c.pt.tolist(), n_energy_pass1=r1["nsim"], n_energy_pass2=r2["nsim"])
        for n, r, X, s in (("pass1", r1, X1, s1), ("pass2", r2, X2, s2)):
            b = basins(c, X); p = props(b, c.K); out[n] = dict(r, tv=L.tv(p, c.pt), p=p.tolist(), frac_assigned=float((b >= 0).double().mean()), **metrics(c, X, b))
            torch.save(dict(X=X.float().cpu(), basin_final=b.cpu(), **s), RES / f"{name}_{n}.pt")
        t = time.time(); out["is_equal_cost"] = path_is(c, a.M * (1 + r1["J"] * KP)); out["is_equal_cost"]["time_s"] = time.time() - t
        w0 = torch.softmax(s2["lw"][0].double(), 0); b0 = s2["basin"][0].long(); p0 = props(b0, c.K, w0)
        out["is_basic"] = dict(n=a.M, logz=float(torch.logsumexp(s2["lw"][0].double(), 0) - math.log(a.M)), tv=L.tv(p0, c.pt))
        json.dump(out, open(OUT / f"run_{name}.json", "w"), indent=1, default=float); print(name, "IS equal cost logZ %+.4f tv %.4f | pass2 tv %.4f q1 %.3f" % (out["is_equal_cost"]["logz"], out["is_equal_cost"]["tv"], out["pass2"]["tv"], out["pass2"]["q1"]), flush=True)
    elif a.cmd == "smoke":                                                                 # environment / GPU / paths check (PBS: qsub -v SMOKE=1); writes figures/smc_noise/smoke.json only
        t0 = time.time()
        if a.device == "cuda": assert torch.cuda.is_available(), "no CUDA device"
        cells = {(nm, e): Cell(nm, e, a.device) for nm, ep in (("A", (1000, 1250, 1500)), ("B", (1000, 1250, 1500)), ("C", (250, 500, 750))) for e in ep}
        torch.manual_seed(1); c = cells[("B", 1000)]; RES.mkdir(parents=True, exist_ok=True)
        r, X, s = smc(c, 64, calendar=[(1.0, 0.9)], kp=1, log=True); b = basins(c, X); p = props(b, c.K); is_ = path_is(c, 128); cm = cells[("C", 250)]; rc, Xc, _ = smc(cm, 64, calendar=[(1.0, 0.9)], kp=1, save=False)
        torch.save(dict(X=X.cpu(), **s), RES / "smoke.pt")
        out = dict(ok=True, device=a.device, gpu=(torch.cuda.get_device_name(0) if a.device == "cuda" else None), torch=torch.__version__, cells_loaded=len(cells), B_levels=r["J"], B_tv=L.tv(p, c.pt), B_metrics=metrics(c, X, b), is_n=is_["n"], C_logz=rc["logz"], time_s=time.time() - t0)
        json.dump(out, open(OUT / "smoke.json", "w"), indent=1, default=float); print("SMOKE OK", out)
    elif a.cmd == "darting":
        import basin_lib as B
        d = torch.load(RES / f"{a.cell}_ep{a.epoch}_r{a.rep}_pass2.pt", map_location="cpu", weights_only=False); x = d["X0"].double().numpy(); c = Cell(a.cell, a.epoch, "cpu")
        mu = c.mu.double().numpy(); wt = np.exp(c.logw.double().numpy()); wt /= wt.sum(); pi = L.Mix("S" if c.mos else "G", mu, wt, 1.0); M = B.polish_minima(pi); S = 2 if a.cell == "B" else (50 if a.cell == "C" else 2)
        rng = np.random.default_rng([a.rep, a.epoch]); X0 = rng.standard_normal((1000, c.D)) * math.sqrt(c.nu1); Y, conv, it = B.quench(pi, X0); bank = []
        for y in Y[conv]:
            if not bank or np.linalg.norm(np.array(bank) - y, axis=1).min() > B.MERGE: bank.append(y)
        r = L.darting(pi, x, mu, c.pt, seed=7 + a.rep, SW=(S,), minima=np.array(bank), keep_final=True); z = r["by_S"][S]; Xf = z.pop("X_final"); b, _, _ = B.basins(pi, Xf, M); p = B.basin_props(b, c.K)
        dex = B.dist_to_nearest_minimum(B.sample_exact(pi, 10000, np.random.default_rng(2026)), M); out = dict(S=S, tv=L.tv(p, c.pt), q1=float(np.median(B.dist_to_nearest_minimum(Xf, M)) / np.median(dex)), cost_dart=z["cost"], cost_bank=1000 * (it + 1), n_bank=len(bank))
        json.dump(out, open(OUT / f"dart_{a.cell}_ep{a.epoch}_r{a.rep}.json", "w"), indent=1, default=float); print(out)
    elif a.cmd == "pcn":                                                                      # rules_pcn_path.md (frozen by the amendment of 2026-11-27): cell B, rho in {0, 0.9, 0.99, 0.999}, 512 chains, 500 steps, burn-in 20 %
        rho = a.rho; nch, steps = a.chains, a.steps; torch.manual_seed(a.epoch); c = Cell(a.cell, a.epoch, a.device); f = OUT / f"pcn_ep{a.epoch}_rho{rho}.json"
        t0 = time.time(); xi = c.noise(nch); lw, X = c.forward(xi); acc, ps, q1s = 0.0, [], []
        for st in range(steps):
            prop = rho * xi + math.sqrt(1 - rho ** 2) * torch.randn_like(xi); lwp, Xp = c.forward(prop); ok = torch.log(torch.rand(nch, device=c.dev).double()) < lwp - lw
            xi = torch.where(ok[None, :, None], prop, xi); lw = torch.where(ok, lwp, lw); X = torch.where(ok[:, None], Xp, X); acc += float(ok.double().mean()) / steps
            if st >= steps // 5: ps.append(basins(c, X).cpu())
        b = torch.cat(ps); p = props(b, c.K); out = dict(epoch=a.epoch, rho=rho, chains=nch, steps=steps, accept=acc, tv=L.tv(p, c.pt), p=p.tolist(), time_s=time.time() - t0, **{"final_" + k: v for k, v in metrics(c, X, basins(c, X)).items()})
        isf = OUT / f"pcn_is_ep{a.epoch}.json"
        if not isf.exists(): torch.manual_seed(5000 + a.epoch); json.dump(path_is(c, nch * (steps + 1)), open(isf, "w"), indent=1)
        json.dump(out, open(f, "w"), indent=1, default=float); print(out)
    elif a.cmd == "pcnverdict":
        eps = (1000, 1250, 1500); V = {}
        isv = float(np.mean([json.load(open(OUT / f"pcn_is_ep{e}.json"))["tv"] for e in eps]))
        for rho in (0.0, 0.9, 0.99, 0.999):
            r = [json.load(open(OUT / f"pcn_ep{e}_rho{rho}.json")) for e in eps]; tv = float(np.mean([x["tv"] for x in r])); ac = float(np.mean([x["accept"] for x in r]))
            V[str(rho)] = dict(tv=tv, tv_per_epoch=[x["tv"] for x in r], accept=ac, J1_part=bool(rho > 0 and tv <= 0.5 * isv), J2=bool(0.1 <= ac <= 0.6))
        V["tv_is_same_budget"] = isv; V["J1"] = any(V[str(r)]["J1_part"] for r in (0.9, 0.99, 0.999)); V["J1_and_J2"] = any(V[str(r)]["J1_part"] and V[str(r)]["J2"] for r in (0.9, 0.99, 0.999))
        json.dump(V, open(OUT / "pcn_verdict.json", "w"), indent=1); print(json.dumps(V, indent=1))
    elif a.cmd == "verdict":
        import glob
        R = {}
        for f in glob.glob(str(OUT / "run_*.json")):
            j = json.load(open(f)); R.setdefault(j["cell"], []).append(j)
        V = {}

        def snap(rs, floor, pt):
            """ per snapshot (amendment of 2026-11-27 for cell C): S1, S2 on the 3 repetitions; basin counts at beta = 0 of pass 2 and the true mass of the basins empty at the start (Th. C) """
            md = lambda g: float(np.median([g(j) for j in rs]))
            e = md(lambda j: abs(j["pass2"]["logz"])); ei = md(lambda j: abs(j["is_equal_cost"]["logz"])); tv = md(lambda j: j["pass2"]["tv"]); tvi = md(lambda j: j["is_equal_cost"]["tv"])
            q1 = md(lambda j: j["pass2"]["q1"])
            counts = [j["pass2"]["levels"][0]["counts"] for j in rs]; empty = [[k for k, n in enumerate(c_) if n == 0] for c_ in counts]
            return dict(n_runs=len(rs), q1=q1, logz_err_smc=e, logz_err_is=ei, tv_smc=tv, tv_is=tvi, S1=bool(e <= 0.1 and e <= 0.5 * ei), S2=bool(tv <= 0.5 * tvi and tv <= 2 * floor), counts_beta0=counts, empty_basins_beta0=empty, true_mass_empty=[float(sum(pt[k] for k in em)) for em in empty])
        for cell, rs in sorted(R.items()):
            if cell == "d8": continue
            rng = np.random.default_rng(5); pt = np.array(rs[0]["p_true"]); M = rs[0]["M"]
            floor = float(np.median([L.tv(rng.multinomial(M, pt / pt.sum()) / M, pt) for _ in range(200)]))
            med = lambda g: float(np.median([g(j) for j in rs])); q = lambda g: [float(np.quantile([g(j) for j in rs], z)) for z in (0.1, 0.5, 0.9)]
            e2 = lambda j: abs(j["pass2"]["logz"]); eis = lambda j: abs(j["is_equal_cost"]["logz"])
            s4 = []
            for j in rs:
                q0 = np.array(j["pass2"]["levels"][0]["p"]); thr = 0.25 * np.minimum(q0, pt); s4.append(all(bool(np.all(np.array(lv["p"]) >= thr - 1e-12)) for lv in j["pass2"]["levels"]))
            V[cell] = dict(per_snapshot={str(e): snap([j for j in rs if j["epoch"] == e], floor, pt) for e in sorted({j["epoch"] for j in rs})}, n_runs=len(rs), floor=floor, all_done=all(j["pass2"]["done"] for j in rs), logz_err_smc=q(e2), logz_err_is=q(eis), logz_err_pass1=q(lambda j: abs(j["pass1"]["logz"])), tv_smc=q(lambda j: j["pass2"]["tv"]), tv_is=q(lambda j: j["is_equal_cost"]["tv"]), tv_pass1=q(lambda j: j["pass1"]["tv"]),
                           q1=q(lambda j: j["pass2"]["q1"]), J=[j["pass2"]["J"] for j in rs], n_s4_ok=int(sum(s4)), S1=bool(med(e2) <= 0.1 and med(e2) <= 0.5 * med(eis)), S2=bool(med(lambda j: j["pass2"]["tv"]) <= 0.5 * med(lambda j: j["is_equal_cost"]["tv"]) and med(lambda j: j["pass2"]["tv"]) <= 2 * floor), S3_all_runs=bool(med(lambda j: j["pass2"]["q1"]) <= {"B": 1.02, "C": 1.2}.get(cell, np.inf)),
                           S3=bool((float(np.median([j["pass2"]["q1"] for j in rs if cell != "C" or j["epoch"] == 250])) if any(cell != "C" or j["epoch"] == 250 for j in rs) else np.inf) <= {"B": 1.02, "C": 1.2}.get(cell, np.inf)),
                           safe_A=bool(med(e2) <= 1.5 * med(eis) and med(lambda j: j["pass2"]["tv"]) <= 1.5 * med(lambda j: j["is_equal_cost"]["tv"])), complete=len(rs) == 9)
        if all(k in V for k in "ABC"):
            c250 = V["C"]["per_snapshot"].get("250"); V["S1"] = bool(V["B"]["S1"] and c250 and c250["S1"]); V["S2"] = bool(V["B"]["S2"] and c250 and c250["S2"]); V["S3"] = bool(V["B"]["S3"] and c250 and c250["q1"] <= 1.2); V["C_degraded_snapshots_descriptive"] = {k: v for k, v in V["C"]["per_snapshot"].items() if k != "250"}; V["S4"] = all(V[k]["n_s4_ok"] == V[k]["n_runs"] for k in "ABC") and V["A"]["safe_A"]
        json.dump(V, open(OUT / "verdict.json", "w"), indent=1, default=float); print(json.dumps(V, indent=1, default=float))
