"""PCA occupancy map on simulated Selten's Horse play.
Tree (paper notation, p_i = prob of R): P1 L->G3(A), R->P2; P2 R->(1,1,1), L->G3(B);
G3 at A: L (0,0,0), R (3,2,2); at B: L (0,0,1), R (4,4,0).
Design as Neugebauer-Sadrieh-Selten: groups of 9 (3 per role), strangers, each period
100 automated plays, subject chooses frequency of R. 250 periods."""
import numpy as np, json
rng = np.random.default_rng(7)
PA = {('A', 0): np.array([0, 0, 0.]), ('A', 1): np.array([3, 2, 2.]),
      ('B', 0): np.array([0, 0, 1.]), ('B', 1): np.array([4, 4, 0.])}
END = np.array([1, 1, 1.])

def play(p1, p2, p3, n=100):
    """realised average payoffs and counterfactual payoffs of L/R for each role."""
    a1 = rng.random(n) < p1; a2 = rng.random(n) < p2; a3 = rng.random(n) < p3
    pay = np.zeros((n, 3))
    for k in range(n):
        if not a1[k]: pay[k] = PA[('A', int(a3[k]))]
        elif a2[k]: pay[k] = END
        else: pay[k] = PA[('B', int(a3[k]))]
    f2, f3 = a2.mean(), a3.mean(); f1 = a1.mean()
    acts = np.stack([a1, a2, a3], 1)
    reached = np.stack([np.ones(n, bool), a1, ~(a1 & a2)], 1)
    earn = np.zeros((3, 2))  # summed payoff obtained with L / R, only when the role actually moved
    for r in range(3):
        for a in (0, 1):
            sel = reached[:, r] & (acts[:, r] == a)
            earn[r, a] = pay[sel, r].sum() / n
    # ex-post payoffs of pure L / pure R for each role against others' realised frequencies
    cf = np.zeros((3, 2))
    cf[0, 0] = 3 * f3 * 1                    # P1 L
    cf[0, 1] = f2 * 1 + (1 - f2) * 4 * f3     # P1 R
    cf[1, 0] = 4 * f3                         # P2 L (only payoff-relevant if reached)
    cf[1, 1] = 1
    rA, rB = (1 - f1), f1 * (1 - f2)          # how often G3 is reached at A / B
    cf[2, 0] = rB * 1
    cf[2, 1] = rA * 2
    return pay.mean(0), cf, np.array([f1, f2, f3]), earn

def simulate(rule, periods=250, groups=60, step=0.03, noise=0.02):
    rows = []
    for g in range(groups):
        p = np.full((3, 3), 0.5)              # role x subject
        prop = np.ones((3, 3, 2))             # RL propensities
        for t in range(periods):
            # strangers: random matching of one subject per role (3 matches)
            perm = [rng.permutation(3) for _ in range(3)]
            for m in range(3):
                i = [perm[r][m] for r in range(3)]
                pay, cf, f, earn = play(p[0, i[0]], p[1, i[1]], p[2, i[2]])
                for r in range(3):
                    s = i[r]
                    if rule == 'impulse':
                        d = np.sign(cf[r, 1] - cf[r, 0]) * step
                        p[r, s] = np.clip(p[r, s] + d + rng.normal(0, noise), 0, 1)
                    else:  # Roth-Erev reinforcement with realised payoffs
                        prop[r, s] = 0.95 * prop[r, s] + earn[r]   # forgetting + reinforcement
                        p[r, s] = np.clip(prop[r, s, 1] / prop[r, s].sum() + rng.normal(0, noise), 0, 1)
                rows.append([g, t, p[0, i[0]], p[1, i[1]], p[2, i[2]], *pay])
    return np.array(rows)

def analyse(rule):
    R = simulate(rule)
    late = R[R[:, 1] >= 150]
    X = late[:, 2:8]                          # p1,p2,p3 + payoffs of the 3 roles
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    w, V = np.linalg.eigh(np.cov(Z.T)); w, V = w[::-1], V[:, ::-1]
    P = Z @ V[:, :2]
    p = late[:, 2:5]
    near_tpe = (p[:, 0] > .8) & (p[:, 1] > .8) & (p[:, 2] < .35)
    near_imp = (p[:, 0] < .2) & (p[:, 1] > .3) & (p[:, 2] > .8)
    eff = (p[:, 2] > .8) & ~near_imp
    tot = late[:, 5:8].sum(1)
    out = dict(rule=rule, evr_2d=float(w[:2].sum() / w.sum()),
               mean_p=p.mean(0).round(2).tolist(),
               share_near_TPE=float(near_tpe.mean()), share_near_imperfect=float(near_imp.mean()),
               share_other_P3_R=float(eff.mean()),
               mean_total_payoff=float(tot.mean()),
               loadings_PC1=dict(zip(['p1', 'p2', 'p3', 'u1', 'u2', 'u3'], V[:, 0].round(2).tolist())))
    np.save(f'P_{rule}.npy', P); np.save(f'p_{rule}.npy', p); np.save(f'tot_{rule}.npy', tot)
    np.save(f'traj_{rule}.npy', R[:, :5])
    return out

if __name__ == '__main__':
    res = [analyse(r) for r in ['impulse', 'reinforcement']]
    for r in res: print(json.dumps(r))
    json.dump(res, open('results.json', 'w'), indent=1)
