"""Sasso-carta-forbice su dati umani: Wang, Xu & Zhou (2014), dataset WXZ2014 di stratEst.
Segue PREREG_RPS_UMANI.md (scritta prima della prima versione di questo script).
La sezione 'post_hoc' del JSON contiene le correzioni aggiunte DOPO la verifica indipendente
(vedi REGISTRO_RPS_UMANI.md): non sono pre-registrate.
Uso:  python rps_umani.py WXZ2014.rda [lisciatura=10]"""
import sys, json
import numpy as np, pyreadr
from scipy.stats import gaussian_kde

SEED = 20261011
NPERM = 2000
MOVES = ['r', 'p', 's']
BEATS = {0: 2, 1: 0, 2: 1}          # r batte s, p batte r, s batte p


def _matchings(L):
    if not L:
        yield []
        return
    a = L[0]
    for b in L[1:]:
        rest = [x for x in L if x not in (a, b)]
        for m in _matchings(rest):
            yield [(a, b)] + m


def load(path):
    d = pyreadr.read_r(path)['WXZ2014']
    d['id'] = d['id'].astype(int)
    piv = lambda col: d.pivot(index='period', columns='id', values=col)
    C = piv('choice'); ids = list(C.columns)
    C, O, R = C.values.astype(str), piv('other.choice').values.astype(str), piv('result').values.astype(str)
    n = C.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = np.mean((C[:, j] == O[:, i]) & (C[:, i] == O[:, j]))
    groups = sorted({tuple(sorted(set(np.argsort(-M[i])[:5]) | {i})) for i in range(n)})
    assert len(groups) == 12 and all(len(g) == 6 for g in groups)
    for g in groups:                      # controllo forte: in ogni round il gruppo si appaia perfettamente
        for t in range(C.shape[0]):
            assert any(all(C[t, a] == O[t, b] and C[t, b] == O[t, a] for a, b in m) for m in _matchings(list(g)))
    Q = np.stack([np.stack([(C[:, list(g)] == m).mean(1) for m in MOVES], 1) for g in groups])
    return Q, [[ids[i] for i in g] for g in groups], C, R


def smooth(Q, w):
    k = np.ones(w) / w
    return np.stack([np.stack([np.convolve(Q[g, :, j], k, mode='valid') for j in range(3)], 1)
                     for g in range(Q.shape[0])])          # causale: il valore al round t usa t-w+1..t


def features(S, velocity=True):
    adv = np.stack([S[..., BEATS[k]] - S[..., [i for i in range(3) if BEATS[i] == k][0]]
                    for k in range(3)], -1)
    parts = [S, adv]
    if velocity:
        parts.append(np.concatenate([np.zeros_like(S[:, :1]), np.diff(S, axis=1)], 1))
    return np.concatenate(parts, -1)


def bary(S):                               # r, p, s ai vertici in senso antiorario: L > 0 = ciclo R -> P -> S
    V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    return S @ V


def ang_mom(Y, c):
    a = Y[:, :-1] - c
    dy = np.diff(Y, axis=1)
    return (a[..., 0] * dy[..., 1] - a[..., 1] * dy[..., 0]).mean(1)


def fit_map(F):
    X = F.reshape(-1, F.shape[-1])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    w, V = np.linalg.eigh(np.cov(Z.T))
    order = np.argsort(w)[::-1]
    return mu, sd, w[order], V[:, order]


def project(F, mu, sd, V):
    return ((F - mu) / sd) @ V[:, :2]


def wxz_freq(Q):
    """Frequenza dei cicli come in WXZ (2014): angolo attorno al centro (2,2,2) fra stati discreti
    consecutivi; transizioni che toccano il centro o collineari contano zero; cicli per round."""
    N = np.rint(Q * 6).astype(int)
    out = []
    for g in range(N.shape[0]):
        th = 0.0
        for t in range(N.shape[1] - 1):
            a = N[g, t] - 2; b = N[g, t + 1] - 2
            if not a.any() or not b.any():
                continue
            sgn = int(np.cross(a, b).sum())
            if sgn == 0:
                continue
            th += np.sign(sgn) * np.arccos(np.clip(a @ b / np.sqrt(a @ a) / np.sqrt(b @ b), -1, 1))
        out.append(th / (2 * np.pi) / (N.shape[1] - 1))
    return np.array(out)


def cond_resp(C, R):
    """Risposte individuali al round successivo, dopo vittoria e dopo sconfitta:
    ripetere, spostarsi in senso antiorario (R->P, P->S, S->R) o orario."""
    idx = np.vectorize({'r': 0, 'p': 1, 's': 2}.get)(C)
    shift = (idx[1:] - idx[:-1]) % 3        # 0 ripete, 1 antiorario, 2 orario
    out = {}
    for res in ['win', 'lose', 'tie']:
        m = R[:-1] == res
        out[res] = {k: float((shift[m] == v).mean()) for k, v in [('stay', 0), ('ccw', 1), ('cw', 2)]}
    return out


def swap_ps(C, R):
    """Scambio delle etichette p <-> s. Uno scambio di due mosse inverte chi batte chi,
    quindi vittorie e sconfitte si scambiano."""
    C2 = np.where(C == 'p', 's', np.where(C == 's', 'p', C))
    R2 = np.where(R == 'win', 'lose', np.where(R == 'lose', 'win', R))
    return C2, R2


def run(path, w):
    rng = np.random.default_rng(SEED)
    Q, groups, C, R = load(path)
    S = smooth(Q, w)
    F = features(S)
    mu, sd, eig, V = fit_map(F)
    Y = project(F, mu, sd, V)
    nash_f = features(np.full((1, 2, 3), 1 / 3))[0, 1]
    c = project(nash_f, mu, sd, V)
    cs = bary(np.full(3, 1 / 3))
    half = S.shape[1] // 2
    # post hoc: mappa senza velocità (solo posizione)
    Fp_ = features(S, velocity=False); mu2, sd2, eig2, V2 = fit_map(Fp_)
    Y2 = project(Fp_, mu2, sd2, V2); c2 = project(features(np.full((1, 2, 3), 1 / 3), False)[0, 1], mu2, sd2, V2)

    obs = dict(simplex=ang_mom(bary(S), cs), map=ang_mom(Y, c), map_h2=ang_mom(Y[:, half:], c),
               simplex_h2=ang_mom(bary(S)[:, half:], cs), map_pos=ang_mom(Y2, c2))
    null = {k: [] for k in obs}
    null_g = []                             # per gruppo, simplesso
    for _ in range(NPERM):
        Qp = np.stack([Q[g, rng.permutation(Q.shape[1])] for g in range(Q.shape[0])])
        Sp = smooth(Qp, w); Yp = project(features(Sp), mu, sd, V)
        Yp2 = project(features(Sp, False), mu2, sd2, V2)
        sg = ang_mom(bary(Sp), cs); null_g.append(sg)
        null['simplex'].append(sg.mean())
        null['map'].append(ang_mom(Yp, c).mean())
        null['map_h2'].append(ang_mom(Yp[:, half:], c).mean())
        null['simplex_h2'].append(ang_mom(bary(Sp)[:, half:], cs).mean())
        null['map_pos'].append(ang_mom(Yp2, c2).mean())
    res = dict(smoothing=w, n_groups=len(groups), periods_used=int(S.shape[1]), nperm=NPERM,
               groups_ids=groups, evr_2d=float(eig[:2].sum() / eig.sum()),
               eig=eig.round(4).tolist(), loadings_pc1=V[:, 0].round(3).tolist(),
               loadings_pc2=V[:, 1].round(3).tolist(),
               feature_names=['sh_r', 'sh_p', 'sh_s', 'adv_r', 'adv_p', 'adv_s', 'vel_r', 'vel_p', 'vel_s'])

    def summ(v, nl, centred):
        nl = np.array(nl); m = v.mean()
        ref = nl.mean() if centred else 0.0
        nb = np.abs(nl - ref)
        return dict(mean=float(m), per_group=v.round(6).tolist(), n_pos=int((v > 0).sum()),
                    n_neg=int((v < 0).sum()), n_same_sign_as_mean=int((np.sign(v) == np.sign(m)).sum()),
                    null_mean=float(nl.mean()),
                    null_p95_abs=float(np.percentile(nb, 95)),
                    p=float((1 + (nb >= abs(m - ref)).sum()) / (NPERM + 1)))
    for k in ['simplex', 'map', 'map_h2']:          # come pre-registrato (null centrato in zero)
        res[k] = summ(obs[k], null[k], centred=False)
    P = Y.reshape(-1, 2)
    kde = gaussian_kde(P.T); dens = kde(P.T)
    res['density_pct_at_nash'] = float((dens < kde(c.reshape(2, 1))[0]).mean())
    res['nash_proj'] = c.round(4).tolist()
    res['share_time_near_nash_005'] = float((np.linalg.norm(S - 1 / 3, axis=-1) < .05).mean())
    res['mean_dist_from_nash'] = float(np.linalg.norm(S - 1 / 3, axis=-1).mean())
    res['mean_shares'] = S.reshape(-1, 3).mean(0).round(4).tolist()
    lam_null = []
    for _ in range(200):
        Qc = np.stack([np.stack([np.roll(Q[g, :, j], rng.integers(Q.shape[1])) for j in range(3)], 1)
                       for g in range(Q.shape[0])])
        lam_null.append(fit_map(features(smooth(Qc, w)))[2][:2])
    lam_null = np.array(lam_null)
    res['A3'] = dict(eig12=eig[:2].round(4).tolist(), null_p95=np.percentile(lam_null, 95, 0).round(4).tolist(),
                     note='6 coordinate su 9 sono lineari nelle 2 quote libere: confronto non informativo')
    angs = []
    for _ in range(200):
        idx = rng.permutation(12); a, b = idx[:6], idx[6:]
        Va = fit_map(F[a])[3][:, :2]; Vb = fit_map(F[b])[3][:, :2]
        s = np.linalg.svd(Va.T @ Vb, compute_uv=False)
        angs.append(np.degrees(np.arccos(np.clip(s.min(), -1, 1))))
    res['A5'] = dict(median_deg=float(np.median(angs)), p95_deg=float(np.percentile(angs, 95)))

    # ---- post hoc (dopo la verifica indipendente; non pre-registrato) ----
    null_g = np.array(null_g)
    pg = [(1 + (np.abs(null_g[:, g]) >= abs(obs['simplex'][g])).sum()) / (NPERM + 1) for g in range(12)]
    f_as_is = wxz_freq(Q)
    Qs = Q[..., [0, 2, 1]]                  # scambio delle etichette p <-> s
    f_swap = wxz_freq(Qs)
    res['post_hoc'] = dict(
        note='aggiunto dopo la verifica indipendente dell\'11/10/2026; non pre-registrato',
        map_centred=summ(obs['map'], null['map'], centred=True),
        map_h2_centred=summ(obs['map_h2'], null['map_h2'], centred=True),
        map_position_only=summ(obs['map_pos'], null['map_pos'], centred=False),
        simplex_h2=summ(obs['simplex_h2'], null['simplex_h2'], centred=False),
        simplex_per_group_p=[float(x) for x in pg],
        simplex_groups_p05=int(sum(x < .05 for x in pg)),
        wxz_freq_as_is=dict(per_group=f_as_is.round(4).tolist(), mean=float(f_as_is.mean()),
                            se=float(f_as_is.std(ddof=1) / np.sqrt(12)), n_neg=int((f_as_is < 0).sum())),
        wxz_freq_swap_ps=dict(mean=float(f_swap.mean()), se=float(f_swap.std(ddof=1) / np.sqrt(12)),
                              n_pos=int((f_swap > 0).sum())),
        cond_resp_as_is=cond_resp(C, R),
        cond_resp_swap_ps=cond_resp(*swap_ps(C, R)),
        wxz_published=dict(freq_range=[0.018, 0.031], direction='R->P->S (antiorario, f > 0)',
                           source='Wang, Xu & Zhou (2014), arXiv:1404.5199'))
    tag = '' if w == 10 else f'_w{w}'
    np.save(f'Y_umani{tag}.npy', Y); np.save(f'nash_umani{tag}.npy', c)
    np.save(f'Ypos_umani{tag}.npy', Y2); np.save(f'nashpos_umani{tag}.npy', c2)   # mappa senza velocità (figura)
    json.dump(res, open(f'risultati_rps_umani{tag}.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    r = run(sys.argv[1], w)
    print(json.dumps({k: r[k] for k in ['smoothing', 'evr_2d', 'density_pct_at_nash']}, indent=1))
    for k in ['simplex', 'map', 'map_h2']:
        print(k, {kk: r[k][kk] for kk in ['mean', 'n_pos', 'n_neg', 'null_mean', 'null_p95_abs', 'p']})
    ph = r['post_hoc']
    for k in ['map_centred', 'map_h2_centred', 'map_position_only', 'simplex_h2']:
        print('post_hoc', k, {kk: ph[k][kk] for kk in ['mean', 'n_pos', 'n_neg', 'null_mean', 'null_p95_abs', 'p']})
    for k in ['simplex_groups_p05', 'wxz_freq_as_is', 'wxz_freq_swap_ps', 'cond_resp_as_is', 'cond_resp_swap_ps']:
        print('post_hoc', k, ph[k])
