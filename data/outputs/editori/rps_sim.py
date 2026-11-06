"""Test: does the occupancy map of RPS trajectories concentrate on Nash or on TASP?
Games and Nash/TASP from Cason, Friedman & Hopkins (2014). Simulated play:
8 agents, logit choice revisions, gradual movement toward the chosen vertex
(as in their Continuous-Slow treatment: ~10 s to move vertex to vertex)."""
import numpy as np, json, sys
from scipy.stats import gaussian_kde

rng = np.random.default_rng(20261010)
G = {'Ua': np.array([[60, 0, 66], [72, 60, 30], [0, 72, 60]], float),
     'S':  np.array([[36, 24, 66], [96, 36, 30], [24, 96, 36]], float)}
NASH = np.array([.25, .25, .50])
TASP_PAPER = np.array([.242, .31, .449])

def simulate(A, lam, periods, N=8, T=180, dt=0.2, rev_rate=0.5, move_time=10.0, mixed=False):
    trajs = []
    for _ in range(periods):
        m = rng.dirichlet(np.ones(3), size=N)
        tgt = m.copy()
        tr = np.empty((int(T / dt), 3))
        for k in range(tr.shape[0]):
            x = m.mean(0); u = A @ x
            rev = rng.random(N) < rev_rate * dt
            if rev.any():
                pr = np.exp(lam * (u - u.max())); pr /= pr.sum()
                if mixed: tgt[rev] = pr
                else: tgt[rev] = np.eye(3)[rng.choice(3, size=rev.sum(), p=pr)]
            m = m + (tgt - m) * min(1, dt / (move_time / 2))
            tr[k] = m.mean(0)
        trajs.append(tr)
    return trajs

def features(tr, A):
    u = tr @ A.T                     # payoff of each pure strategy
    avg = (tr * u).sum(1, keepdims=True)
    vel = np.vstack([np.zeros((1, 3)), np.diff(tr, axis=0)])
    return np.hstack([tr, u, avg, u - avg, vel])

def analyse(name, A, lam, periods, mixed=False):
    trajs = simulate(A, lam, periods, mixed=mixed)
    burn = 50  # drop first 10 s of each period (random start)
    X = np.vstack([features(t, A)[burn:] for t in trajs])
    S = np.vstack([t[burn:] for t in trajs])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    w, V = np.linalg.eigh(np.cov(Z.T)); V = V[:, ::-1][:, :2]; w = w[::-1]
    P = Z @ V
    def proj(s):
        f = features(np.tile(s, (2, 1)), A)[1:].copy(); f[:, -3:] = 0
        return ((f - mu) / sd) @ V
    sub = P[rng.choice(len(P), min(20000, len(P)), replace=False)]
    kde = gaussian_kde(sub.T)
    dens_pts = kde(sub.T)
    def pct(pt):  # share of observed time spent in less dense places than pt
        return float((dens_pts < kde(pt.T)[0]).mean())
    grid_best = sub[np.argmax(dens_pts)]
    # mode in simplex space: state with highest density
    mode_state = S[np.argmin(((P - grid_best) ** 2).sum(1))]
    mean_state = S.mean(0)
    r = dict(game=name, lam=lam, mixed=mixed, periods=periods,
             evr_2d=float(w[:2].sum() / w.sum()),
             mean_state=mean_state.round(3).tolist(),
             dist_mean_to_nash=float(np.linalg.norm(mean_state - NASH)),
             dist_mean_to_tasp=float(np.linalg.norm(mean_state - TASP_PAPER)),
             mode_state=mode_state.round(3).tolist(),
             dist_mode_to_nash=float(np.linalg.norm(mode_state - NASH)),
             dist_mode_to_tasp=float(np.linalg.norm(mode_state - TASP_PAPER)),
             density_pct_at_nash=pct(proj(NASH)),
             density_pct_at_tasp=pct(proj(TASP_PAPER)),
             share_time_near_nash=float((np.linalg.norm(S - NASH, axis=1) < .05).mean()),
             mean_dist_from_nash=float(np.linalg.norm(S - NASH, axis=1).mean()))
    tag=f'{name}_{lam}_{"mix" if mixed else "pure"}'; np.save(f'P_{tag}.npy', sub); np.save(f'S_{tag}.npy', S[::5]); np.save(f'L_{tag}.npy', np.array([proj(NASH)[0], proj(TASP_PAPER)[0]]))
    return r

if __name__ == '__main__':
    res = []
    for name in ['S', 'Ua']:
        for lam in [0.2, 0.5]:
          for mixed in [False, True]:
            r = analyse(name, G[name], lam, periods=int(sys.argv[1]), mixed=mixed)
            print(json.dumps(r)); res.append(r)
    json.dump(res, open('results.json', 'w'), indent=1)
