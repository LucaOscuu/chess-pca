"""Parsing HandHQ (.phhs) -> pannello giocatore x blocco di 10 mani."""
import re, glob, ast
import numpy as np, pandas as pd
from treys import Card, Evaluator
EV = Evaluator()

def parse_file(path, stake):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    for block in re.split(r"\n(?=\[\d+\]\n)", txt):
        d = {}
        for line in block.split("\n"):
            m = re.match(r"(\w+) = (.*)$", line)
            if m: d[m.group(1)] = m.group(2)
        try:
            players = ast.literal_eval(d["players"]); acts = ast.literal_eval(d["actions"])
            blinds = [float(x) for x in ast.literal_eval(d["blinds_or_straddles"])]
            stacks = [float(x) for x in ast.literal_eval(d["starting_stacks"].replace("inf", "0"))]
        except Exception:
            continue
        n = len(players); bb = max(blinds) if max(blinds) > 0 else 1
        street = 0; commit = [blinds[i] for i in range(n)]; inv_prev = [0.0] * n
        folded = [False] * n; folded_pre = [False] * n
        st = {k: [0] * n for k in ["vpip", "pfr", "pre_r", "pre_a", "post_r", "post_a", "bets", "calls"]}
        sawflop = [False] * n; showdown = [False] * n; shown = [None] * n; board = []
        last_street_commit = None
        for a in acts:
            p = a.split()
            if p[0] == "d":
                if p[1] == "db":
                    cards = p[2]; board += [cards[j:j+2] for j in range(0, len(cards), 2)]
                    inv_prev = [inv_prev[i] + commit[i] for i in range(n)]; last_street_commit = commit
                    commit = [0.0] * n; street += 1
                    if street == 1:
                        for i in range(n): sawflop[i] = not folded[i]
                continue
            if not p[0].startswith("p"): continue
            i = int(p[0][1:]) - 1
            if i >= n: continue
            op = p[1]
            if op == "f":
                folded[i] = True
                if street == 0: folded_pre[i] = True
                (st["pre_a"] if street == 0 else st["post_a"])[i] += 1
            elif op == "cc":
                mx = max(commit); add = min(mx - commit[i], stacks[i] - inv_prev[i] - commit[i])
                if street == 0 and mx > commit[i]: st["vpip"][i] = 1
                commit[i] += max(add, 0); st["calls"][i] += 1 if mx > 0 else 0
                (st["pre_a"] if street == 0 else st["post_a"])[i] += 1
            elif op == "cbr":
                amt = float(p[2]); commit[i] = min(amt, stacks[i] - inv_prev[i])
                if street == 0: st["vpip"][i] = 1; st["pfr"][i] = 1; st["pre_r"][i] += 1
                else: st["post_r"][i] += 1
                st["bets"][i] += 1
                (st["pre_a"] if street == 0 else st["post_a"])[i] += 1
            elif op == "sm":
                showdown[i] = True
                shown[i] = p[2] if len(p) > 2 and "?" not in p[2] else None
        # restituzione della puntata non chiamata sull'ultima strada
        c = sorted(commit, reverse=True)
        if len(c) > 1 and c[0] > c[1]:
            j = commit.index(c[0]); commit[j] = c[1]
        inv = [inv_prev[i] + commit[i] for i in range(n)]
        alive = [i for i in range(n) if not folded[i]]
        win = [0.0] * n
        if len(alive) == 1:
            win[alive[0]] = sum(inv)
        else:
            known = [i for i in alive if shown[i]]
            if not known or len(board) < 5:
                continue  # showdown non ricostruibile: mano esclusa
            try:
                bd = [Card.new(c) for c in board[:5]]
                score = {i: EV.evaluate([Card.new(shown[i][:2]), Card.new(shown[i][2:4])], bd) for i in known}
            except Exception:
                continue
            levels = sorted(set(inv[i] for i in alive))
            prev = 0.0
            for lv in levels:
                pot = sum(min(inv[j], lv) - min(inv[j], prev) for j in range(n))
                elig = [i for i in known if inv[i] >= lv]
                if not elig:
                    elig = [i for i in alive if inv[i] >= lv]
                    for i in elig: win[i] += pot / len(elig)
                else:
                    best = min(score[i] for i in elig); ws = [i for i in elig if score[i] == best]
                    for i in ws: win[i] += pot / len(ws)
                prev = lv
        for i in range(n):
            yield dict(stake=stake, player=players[i], hand=int(d.get("hand", 0)), time=d.get("time", ""), day=int(d.get("day", 0)),
                       vpip=st["vpip"][i], pfr=st["pfr"][i], fold_pre=int(folded_pre[i]), saw_flop=int(sawflop[i]),
                       showdown=int(showdown[i]), pre_r=st["pre_r"][i], pre_a=st["pre_a"][i], post_r=st["post_r"][i],
                       post_a=st["post_a"][i], bets=st["bets"][i], calls=st["calls"][i], net_bb=(win[i] - inv[i]) / bb,
                       n_players=n)

rows = []
for stake in ("low", "high"):
    for f in sorted(glob.glob(f"work/poker/raw/{stake}_*.phhs")):
        rows.extend(parse_file(f, stake))
H = pd.DataFrame(rows)
print(H.shape, H.groupby("stake").size().to_dict())
print(H.groupby("stake")[["vpip", "pfr", "fold_pre", "saw_flop", "showdown", "net_bb"]].mean().round(3))
print("somma net_bb per posta (dovrebbe essere ~ -rake):", H.groupby("stake").net_bb.sum().round(0).to_dict())
H = H.sort_values(["stake", "player", "day", "time", "hand"])
H["k"] = H.groupby(["stake", "player"]).cumcount()
cnt = H.groupby(["stake", "player"]).size()
keep = cnt[cnt >= 200].index
H = H.set_index(["stake", "player"]).loc[keep].reset_index()
tot = H.groupby(["stake", "player"]).net_bb.sum().rename("net_tot")
H = H[H.k < 300]; H["block"] = H.k // 10
B = H.groupby(["stake", "player", "block"]).agg(vpip=("vpip", "mean"), pfr=("pfr", "mean"), fold_pre=("fold_pre", "mean"),
    saw_flop=("saw_flop", "mean"), showdown=("showdown", "mean"), pre_r=("pre_r", "sum"), pre_a=("pre_a", "sum"),
    post_r=("post_r", "sum"), post_a=("post_a", "sum"), bets=("bets", "sum"), calls=("calls", "sum"), net_block=("net_bb", "sum")).reset_index()
B["pre_aggr"] = B.pre_r / B.pre_a.clip(lower=1); B["post_aggr"] = B.post_r / B.post_a.clip(lower=1); B["af"] = B.bets / (B.calls + 1)
B = B.merge(tot.reset_index(), on=["stake", "player"])
B["result_num"] = np.sign(B.net_tot).astype(int)
B["traj"] = B.stake + ":" + B.player
B.to_parquet("work/poker/panel_poker.parquet", index=False)
print(B.groupby("stake").agg(giocatori=("player", "nunique"), blocchi=("block", "size")))
print(B.groupby("stake").apply(lambda x: (x.groupby("player").result_num.first() > 0).mean()).round(2))
