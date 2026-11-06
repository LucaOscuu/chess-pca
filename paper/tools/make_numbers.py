"""Genera paper/numbers.tex e paper/numbers_index.md leggendo i file dei risultati.

REGOLA: nessun numero del paper si scrive a mano. Nel testo si usa \\val{chiave}.
Ogni chiave ha una fonte (file + campo) e una descrizione. Se un file manca o un campo
non esiste, lo script si ferma con errore: meglio un errore che un numero sbagliato.

Uso (dalla cartella principale della repo):  python paper/tools/make_numbers.py
"""
from __future__ import annotations
import csv, json, math, re, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo chess-pca
OUT = ROOT / "data" / "outputs"
DATA = ROOT / "data"
PAPER = ROOT / "paper"

ENTRIES: list[tuple[str, str, str, str, str]] = []   # key, value, source, description, status


# ---------------------------------------------------------------- helpers
def J(rel, *path):
    obj = json.loads((OUT / rel).read_text(encoding="utf-8").replace("NaN", "null"))
    for p in path:
        obj = obj[p]
    return obj

def CSV(rel, key_col, key, col):
    with open(OUT / rel, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row[key_col] == key:
                return float(row[col])
    raise KeyError(f"{rel}: {key_col}={key} non trovato")

def MD(rel, pattern):
    """Estrae un numero da un registro .md (per i pochi valori che esistono solo lì)."""
    txt = (OUT / rel).read_text(encoding="utf-8")
    m = re.search(pattern, txt)
    if not m:
        raise KeyError(f"{rel}: pattern {pattern!r} non trovato")
    return float(m.group(1).replace(".", "").replace(",", "."))

def f(x, d=1):
    """Arrotondamento half-up sul valore decimale (0.945 -> 0.95), come nei registri."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("valore mancante")
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))

def pp(x, d=1):            # quota 0-1 -> percentuale
    return f(100 * x, d)

def thousands(n):
    return f"{int(round(n)):,}"

def pval(p):
    """p da permutazioni: riporta il valore, mai 0."""
    if p < 0.0005:
        return f"{p:.4f}"
    return f"{p:.3f}"

def add(key, value, source, desc, status="ok"):
    if not re.fullmatch(r"[a-z0-9.\-]+", key):
        raise ValueError(f"chiave non valida: {key}")
    if any(k == key for k, *_ in ENTRIES):
        raise ValueError(f"chiave duplicata: {key}")
    ENTRIES.append((key, str(value), source, desc, status))


# ================================================================ SCACCHI: corpora
def corpora():
    try:
        import pandas as pd
    except ImportError:
        sys.exit("serve pandas + pyarrow per leggere i parquet dei corpora")
    caches = {
        "full": "cache/white_data_max10000_min10.parquet",
        "d6": "cache_d6/white_data_max2000_min10.parquet",
        "d3": "cache_d3/white_data_max2000_min10.parquet",
        "d1": "cache_d1/white_data_max2000_min10.parquet",
        "random": "cache_random/white_data_max2000_min10.parquet",
        "3check": "cache_3check/white_data_max2000_min10.parquet",
        "antichess": "cache_antichess/white_data_max2000_min10.parquet",
    }
    for name, rel in caches.items():
        d = pd.read_parquet(DATA / rel)
        g = d.groupby("game_id")["result"].first()
        add(f"chess.{name}.games", thousands(g.size), f"data/{rel}", f"partite nel corpus {name}")
        add(f"chess.{name}.draws", pp((g == "D").mean()), f"data/{rel}", f"% patte nel corpus {name}")
        add(f"chess.{name}.decisive", pp((g != "D").mean()), f"data/{rel}", f"% partite decisive nel corpus {name}")
        if name == "full":
            feats = [c for c in d.columns if c not in ("game_id", "ply", "move_num", "result", "opening_fen")]
            add("chess.phi.d", len(feats), f"data/{rel}", "numero di coordinate del feature map")
            b = pd.read_parquet(DATA / rel.replace("white", "black"))
            add("chess.full.states", thousands(len(d) + len(b)), f"data/{rel} + black", "stati (righe) del corpus full")
    m = pd.read_csv(DATA / "umani/partite_umane_meta.csv")
    for cls, key in (("standard", "classical"), ("fast", "fast")):
        s = m[m.cls == cls]
        add(f"human.{key}.games", thousands(len(s)), "data/umani/partite_umane_meta.csv", f"partite umane {key}")
        add(f"human.{key}.elo", thousands((s.welo.mean() + s.belo.mean()) / 2), "data/umani/partite_umane_meta.csv", f"Elo medio {key}")
    add("human.minelo", thousands(min(m.welo.min(), m.belo.min())), "data/umani/partite_umane_meta.csv", "Elo minimo (filtro >= 2400)")


# ================================================================ SCACCHI: struttura e obiettivo
def chess_structure():
    s = "analisi_settembre/26_09/A3_null_dipendente_T2_26_09.json"
    for c in ("full", "depth6", "random", "3check"):
        add(f"a3.kstar.{c}", J(s, c, "k_star"), s, f"assi sopra il null che conserva la dipendenza, {c}")

    s = "a2_partial_antichess.csv"
    rows = list(csv.DictReader(open(OUT / s, encoding="utf-8")))
    raw = sum(r["inv_grezza"] == "True" for r in rows)
    par = sum(r["inv_parziale"] == "True" for r in rows)
    add("a2.anti.rawinv", raw, s, "inversioni grezze di segno antichess vs depth6 (|r|>0.05)")
    add("a2.anti.partinv", par, s, "inversioni parziali (controllando il materiale), |r|>0.05")
    s = "analisi_settembre/27_09/T3_27_09.json"
    add("a2.anti.rawinv.ci", J(s, "antichess", "inversioni_grezze_robuste"), s,
        "inversioni grezze con IC bootstrap per partita che escludono zero (criterio IC, include |r|<0.05)")
    add("a2.anti.partinv.ci", J(s, "antichess", "inversioni_parziali_robuste"), s, "inversioni parziali con IC che escludono zero")
    add("a2.matcorr.ref", f(J(s, "antichess", "corr_material_esito", "ref"), 3), s, "corr(material, esito) depth6")
    add("a2.matcorr.anti", f(J(s, "antichess", "corr_material_esito", "tgt"), 3), s, "corr(material, esito) antichess")
    add("a2.matcorr.3check", f(J(s, "3check", "corr_material_esito", "tgt"), 3), s, "corr(material, esito) Three-Check")
    add("a2.3check.rawinv", J(s, "3check", "inversioni_grezze_robuste"), s, "inversioni grezze Three-Check")
    add("a2.3check.partinv.ci", J(s, "3check", "inversioni_parziali_robuste"), s, "inversioni parziali con IC Three-Check (mobility)")

    s = "analisi_settembre/27_09/T5_27_09.json"
    add("rot.anti.angle", f(J(s, "antichess", "angolo")), s, "angolo piano antichess vs depth6 (finestra T5, mosse per giocatore, appaiato)")
    add("rot.anti.p95", f(J(s, "antichess", "p95_appaiato")), s, "null p95 appaiato antichess")
    add("rot.anti.pairs", thousands(J(s, "antichess", "coppie")), s, "coppie appaiate antichess")
    add("rot.anti.p", pval(J(s, "antichess", "p_appaiato")), s, "p antichess")
    add("rot.3check12.angle", f(J(s, "3check", "angolo")), s, "angolo Three-Check vs depth6 (finestra T5)")
    add("rot.3check12.p95", f(J(s, "3check", "p95_appaiato")), s, "null p95 Three-Check (finestra T5)")

    s = "analisi_settembre/27_09/T4_27_09.json"
    for c, k in (("random", "random"), ("3check", "3check"), ("depth1", "d1")):
        add(f"rot.{k}.angle", f(J(s, c, "angolo")), s, f"angolo {c} vs depth6, finestra corta (mosse per giocatore), appaiato per apertura")
        add(f"rot.{k}.p95", f(J(s, c, "p95_appaiato")), s, f"null p95 appaiato {c}")
        add(f"rot.{k}.p", pval(J(s, c, "p_appaiato")), s, f"p appaiato {c}")
        add(f"rot.{k}.pairs", thousands(J(s, c, "coppie")), s, f"coppie {c}")


# ================================================================ SCACCHI: occupazione
def chess_occupancy():
    s = "occupancy_grouped/occupancy_grouped_full.json"
    for side in ("W", "B"):
        sl = side.lower()
        add(f"occ.full.mm.{sl}", f(J(s, "material_matched", side, "signed_gap_test_pp")), s, f"gap occupazione full, materiale appaiato, {side} (pp)")
        add(f"occ.full.mm.{sl}.p95", f(J(s, "material_matched", side, "placebo_p95_pp")), s, f"placebo p95 (pp) {side}")
        add(f"occ.full.mm.{sl}.p", pval(J(s, "material_matched", side, "placebo_p")), s, f"p placebo per partita {side}")
        add(f"occ.full.mm.{sl}.rb", f(abs(J(s, "material_matched", side, "rb_test")), 2), s, f"rank-biserial |rb| {side}")
        add(f"occ.full.all.{sl}", f(J(s, "all_plies", side, "signed_gap_test_pp")), s, f"gap non appaiato {side} (pp)")
    add("occ.perms", thousands(J(s, "perms")), s, "permutazioni del placebo")
    s = "occupancy_grouped/occupancy_grouped_full_w20.json"
    for side in ("W", "B"):
        add(f"occ.full20.mm.{side.lower()}", f(J(s, "material_matched", side, "signed_gap_test_pp")), s, f"gap full finestre corte (mosse per giocatore) {side}")
    s = "occupancy_grouped/occupancy_grouped_3check_whole.json"   # traiettorie intere (come full)
    for side in ("W", "B"):
        add(f"occ.3check.mm.{side.lower()}", f(J(s, "material_matched", side, "signed_gap_test_pp")), s, f"gap Three-Check, traiettorie intere, {side}")

    s = "analisi_settembre/26_09/occupazione_100_divisioni_T3a_26_09.json"
    for side in ("W", "B"):
        sl = side.lower()
        add(f"occ.splits.{sl}.med", f(J(s, side, "mediana")), s, f"mediana gap su 100 divisioni {side}")
        add(f"occ.splits.{sl}.lo", f(J(s, side, "p5")), s, f"5° percentile {side}")
        add(f"occ.splits.{sl}.hi", f(J(s, side, "p95")), s, f"95° percentile {side}")
        add(f"occ.splits.{sl}.share", pp(J(s, side, "quota_positivo_e_sopra_1_6"), 0), s, f"% divisioni sopra 1.6 pp {side}")
    s = "analisi_settembre/26_09/occupazione_stouffer_T3b_26_09.json"
    for c, k in (("depth1", "d1"), ("depth3", "d3"), ("depth6", "d6")):
        add(f"occ.gen.{k}", f(J(s, c, "gap_medio_pp")), s, f"gap medio W/B {c}")
        add(f"occ.gen.{k}.p", pval(J(s, c, "p")), s, f"p {c}")
    add("occ.gen.z", f(J(s, "stouffer", "z"), 2), s, "z di Stouffer corpora generati")
    add("occ.gen.p", pval(J(s, "stouffer", "p")), s, "p di Stouffer")


# ================================================================ SCACCHI: umani, flusso
def chess_humans():
    s = "umani/rotazione.csv"
    pairs = {"classica vs full": "cl.full", "veloci vs full": "fa.full", "classica vs veloci": "cl.fa",
             "classica vs depth6": "cl.d6", "veloci vs depth6": "fa.d6", "full vs depth6": "full.d6"}
    with open(OUT / s, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["finestra"] == "primaria_post_apertura"]
    for r in rows:
        k = pairs[r["contrasto"]]
        add(f"hum.rot.{k}", f(float(r["angolo"])), s, f"angolo {r['contrasto']} (righe 10-29)")
        add(f"hum.rot.{k}.p95", f(float(r["null_p95"])), s, f"null p95 {r['contrasto']}")
        add(f"hum.rot.{k}.p", pval(float(r["p"])), s, f"p {r['contrasto']}")

    s = "umani/occupazione.json"
    for corp, k in (("classica_intere", "cl"), ("veloci_intere", "fa")):
        for side in ("W", "B"):
            add(f"hum.occ.{k}.{side.lower()}", f(J(s, corp, "material_matched", side, "signed_gap_test_pp")), s, f"gap occupazione {corp} {side}")
            add(f"hum.occ.{k}.{side.lower()}.p", pval(J(s, corp, "material_matched", side, "placebo_p")), s, f"p {corp} {side}")
    for corp, k in (("classica_post_apertura", "clpost"),):
        for side in ("W", "B"):
            add(f"hum.occ.{k}.{side.lower()}", f(J(s, corp, "material_matched", side, "signed_gap_test_pp")), s, f"gap {corp} {side}")

    s = "umani/esiti_obiettivo.json"
    add("hum.matcorr.cl", f(J(s, "obiettivo", "classica", "r_material_esito"), 3), s, "corr(material, esito) classica")
    add("hum.matcorr.fa", f(J(s, "obiettivo", "veloci", "r_material_esito"), 3), s, "corr(material, esito) veloci")
    add("hum.matcorr.full", f(J(s, "obiettivo", "full", "r_material_esito"), 3), s, "corr(material, esito) full")
    add("hum.inv.cl", len(J(s, "obiettivo", "classica", "inv_parziali")), s, "inversioni parziali classica")
    add("hum.inv.fa", len(J(s, "obiettivo", "veloci", "inv_parziali")), s, "inversioni parziali veloci")
    for c, k in (("classica", "cl"), ("veloci", "fa"), ("full", "full"), ("depth6", "d6")):
        add(f"hum.draws.{k}", pp(J(s, "esiti", c, "D")), s, f"% patte {c}")

    s = "umani/flusso.json"
    for pair, k in (("classica vs veloci", "cl.fa"), ("classica vs full", "cl.full"), ("veloci vs full", "fa.full"),
                    ("full vs depth6", "full.d6"), ("classica vs depth6", "cl.d6")):
        add(f"flow.{k}.ratio", f(J(s, "M2_campo", pair, "rapporto_oss_null")), s, f"campo di flusso {pair}: osservato / mediana null")
        add(f"flow.{k}.p", pval(J(s, "M2_campo", pair, "p")), s, f"p {pair}")
    s2 = "umani/flusso_robustezza.json"
    add("flow.cl.fa.dec.ratio", f(J(s2, "solo_decisive", "classica vs veloci", "rapporto_oss_null")), s2, "flusso classica vs veloci, solo decisive")
    add("flow.cl.fa.dec.p", pval(J(s2, "solo_decisive", "classica vs veloci", "p")), s2, "p solo decisive")
    # M3: risposta all'impulso — PROVVISORIA finché non c'è il controllo per la regressione verso la media
    for c, k in (("classica", "cl"), ("veloci", "fa"), ("full", "full"), ("depth6", "d6")):
        add(f"prov.imp.{k}", f(J(s, "M3_impulso", c, "proj_dopo_impulso"), 2), s, f"proiezione del passo dopo impulso, {c}", "provvisorio")
        add(f"prov.imp.{k}.none", f(J(s, "M3_impulso", c, "proj_senza_impulso"), 2), s, f"proiezione senza impulso, {c}", "provvisorio")


# ================================================================ SCACCHI: robustezza (appendice)
def chess_robustness():
    s = "analisi_settembre/27_09/T6_27_09.json"
    for c in ("mobility", "pawns_passed_diff", "captures_avail", "cnt_pawn"):
        add(f"lev.full.{c.replace('_', '-')}", f(J(s, "leva", c)), s, f"leva {c} sul corpus full (gradi)")
    add("lev.full.p95", f(J(s, "null_p95")), s, "null p95 leva full")
    add("lev.full.above", J(s, "sopra_null"), s, "coordinate sopra il null (full)")
    add("lev.full.abovemax", len(J(s, "sopra_33_4")), s, "coordinate sopra il contrasto massimo 33.4")
    s = "feature_leverage_depth6.csv"
    for c in ("pawns_passed_diff", "cnt_pawn", "mobility", "captures_avail"):
        add(f"lev.d6.{c.replace('_', '-')}", f(CSV(s, "feature", c, "leva_deg")), s, f"leva {c} su depth6")
    s = "analisi_settembre/26_09/permanova_B2_26_09.json"
    for part, k in (("sistema", "sys"), ("strumento", "instr"), ("interazione", "inter"), ("residuo_campionamento", "samp")):
        add(f"perm.a.{k}", pp(J(s, "A_randomizzazione", part), 0), s, f"PERMANOVA disegno A, quota {part} (%)")
        add(f"perm.b.{k}", pp(J(s, "B_trasformazioni_monotone", part), 0), s, f"PERMANOVA disegno B, quota {part} (%)")
    s = "analisi_settembre/27_09/T2a_27_09.json"
    add("fa.angle", f(J(s, "angolo_FA")), s, "contrasto random vs depth6 con analisi fattoriale")
    add("fa.pcaangle", f(J(s, "angolo_PCA_stesso_campione")), s, "stesso contrasto con PCA")
    s = "analisi_settembre/27_09/T2c_27_09.json"
    add("fa.occ.w", f(J(s, "W", "mediana")), s, "occupazione con analisi fattoriale W")
    add("fa.occ.b", f(J(s, "B", "mediana")), s, "occupazione con analisi fattoriale B")
    s = "analisi_settembre/26_09/occupazione_strumenti_B1_26_09.json"
    for inst, k in (("I1_senza_leva_alta", "i1"), ("I2_solo_materiale", "i2"), ("I3_solo_posizionali", "i3")):
        add(f"inst.{k}.w", f(J(s, inst, "W", "mediana")), s, f"occupazione {inst} W")
        add(f"inst.{k}.b", f(J(s, inst, "B", "mediana")), s, f"occupazione {inst} B")
        add(f"inst.{k}.d", J(s, inst, "d"), s, f"coordinate {inst}")
    s = "analisi_settembre/26_09/persistenza_T1_26_09.json"
    for c, k in (("full", "full"), ("depth6", "d6")):
        for side in ("W", "B"):
            add(f"pers.{k}.{side.lower()}", pp(J(s, c, side, "principale", "eccesso_su_blocchi")), s, f"eccesso persistenza {c} {side} (%)")
            add(f"pers.{k}.{side.lower()}.trunc", pp(J(s, c, side, "escluse_troncate", "eccesso_su_blocchi")), s, f"eccesso senza visite terminali {c} {side} (%)")


# ================================================================ MERCATI
def markets():
    r = "mercati/REGISTRO_MERCATI.md"
    add("mkt.markets", thousands(MD(r, r"Sono ([\d.]+) mercati")), r, "mercati nella replica Huber et al.")
    add("mkt.traders", thousands(MD(r, r"e ([\d.]+) trader")), r, "trader")
    add("mkt.traj", thousands(MD(r, r"([\d.]+) traiettorie trader")), r, "traiettorie trader x ripetizione")
    s = "mercati/risultati_mercati.json"
    for st, k in (("completo", "full"), ("solo_comportamento", "beh")):
        add(f"mkt.{k}.split", f(J(s, st, "T0_split_half_mediana_p95")[0]), s, f"split-half mediano {st}")
        add(f"mkt.{k}.split.p95", f(J(s, st, "T0_split_half_mediana_p95")[1]), s, f"split-half p95 {st}")
        add(f"mkt.{k}.evr", pp(J(s, st, "T0_varianza_spiegata_PC1_PC2"), 0), s, f"varianza spiegata PC1+PC2 {st} (%)")
        add(f"mkt.{k}.rot.rep", f(J(s, st, "T1_rep1_vs_rep2", "angolo")), s, f"rotazione rip.1 vs 2 {st}")
        add(f"mkt.{k}.rot.rep.p95", f(J(s, st, "T1_rep1_vs_rep2", "null_p95")), s, f"null p95 rip. {st}")
        add(f"mkt.{k}.rot.skill", f(J(s, st, "T1_abilita_alta_vs_bassa", "angolo")), s, f"rotazione abilita {st}")
        add(f"mkt.{k}.rot.skill.p95", f(J(s, st, "T1_abilita_alta_vs_bassa", "null_p95")), s, f"null p95 abilita {st}")
        add(f"mkt.{k}.occ.mm", f(J(s, st, "T2_occupazione_ricchezza_appaiata", "gap_test_pp")), s, f"occupazione a ricchezza appaiata {st} (pp)")
        add(f"mkt.{k}.occ.mm.p95", f(J(s, st, "T2_occupazione_ricchezza_appaiata", "placebo_p95")), s, f"placebo p95 {st}")
        add(f"mkt.{k}.occ.mm.p", pval(J(s, st, "T2_occupazione_ricchezza_appaiata", "p")), s, f"p {st}")
        add(f"mkt.{k}.occ.all", f(J(s, st, "T2_occupazione_tutte", "gap_test_pp")), s, f"occupazione non appaiata {st}")
        add(f"mkt.{k}.flow.skill", f(J(s, st, "T3_flusso_abilita", "rapporto")), s, f"flusso abilita {st} (x rumore)")
        add(f"mkt.{k}.flow.rep", f(J(s, st, "T3_flusso_rep1_vs_rep2", "rapporto")), s, f"flusso ripetizione {st}")
        add(f"mkt.{k}.flow.p", pval(J(s, st, "T3_flusso_abilita", "p")), s, f"p flusso abilita {st}")
    s = "mercati/risultati_bacino_ldt.json"
    for st, k in (("completo", "full"), ("solo_comportamento", "beh")):
        add(f"mkt.{k}.basin.rho", f(J(s, st, "B_bacino_abilita", "rho_mediana"), 2), s, f"rho abilita-bacino {st}")
        add(f"mkt.{k}.basin.rholo", f(J(s, st, "B_bacino_abilita", "rho_min_max")[0], 2), s, f"rho min {st}")
        add(f"mkt.{k}.basin.rhohi", f(J(s, st, "B_bacino_abilita", "rho_min_max")[1], 2), s, f"rho max {st}")
        add(f"mkt.{k}.basin.beta0", f(J(s, st, "B_bacino_abilita", "beta_abilita_solo"), 2), s, f"beta abilita senza bacino {st}")
        add(f"mkt.{k}.basin.beta1", f(J(s, st, "B_bacino_abilita", "beta_abilita_con_bacino"), 2), s, f"beta abilita con bacino {st}")
    s = "mercati/risultati_baseline.json"
    for st, k in (("completo", "full"), ("solo_comportamento", "beh")):
        for m in ("basin", "logit", "kmeans"):
            add(f"mkt.{k}.auc.{m}", f(J(s, st, m)[0], 2), s, f"AUC mediana {m} {st}")
    s = "mercati/risultati_ldt2.json"
    add("mkt.ldt.diff", f(J(s, "D1_tutti", "diff"), 2), s, "learning direction D1: differenza (segno opposto alla predizione)")
    add("mkt.ldt.p", pval(J(s, "D1_tutti", "p")), s, "p D1")
    add("mkt.ldt.low.diff", f(J(s, "D2_bassa_abilita", "diff"), 2), s, "D1 nei trader a bassa abilita")
    add("mkt.ldt.low.p", pval(J(s, "D2_bassa_abilita", "p")), s, "p bassa abilita")
    add("mkt.ldt.high.p", pval(J(s, "D2_alta_abilita", "p")), s, "p alta abilita")
    s = "mercati/risultati_ldt_controllo.json"
    add("mkt.ldtc.proj", f(J(s, "completo|proj~err", "beta_entro_cella"), 2), s, "passo verso la direzione vincente dopo errore, entro cella")
    add("mkt.ldtc.proj.lo", f(J(s, "completo|proj~err", "ic95")[0], 2), s, "IC basso")
    add("mkt.ldtc.proj.hi", f(J(s, "completo|proj~err", "ic95")[1], 2), s, "IC alto")
    markets_design()


# ================================================================ MERCATI: chiavi aggiunte per la Sez. 5 (10/10/2026, sessione 3)
def markets_design():
    """Disegno e risultati dei mercati citati nella Sez. 5, letti dai registri e dai JSON."""
    r = "mercati/REGISTRO_MERCATI.md"
    add("mkt.reps", int(MD(r, r"gioca (\d+) ripetizioni")), r, "ripetizioni per trader")
    add("mkt.periods", int(MD(r, r"ripetizioni da (\d+) periodi")), r, "periodi per ripetizione")
    add("mkt.fv.start", int(MD(r, r"scende da (\d+) a \d+")), r, "valore fondamentale iniziale")
    add("mkt.fv.end", int(MD(r, r"scende da \d+ a (\d+)")), r, "valore fondamentale finale")
    add("mkt.obs", thousands(MD(r, r"traiettorie trader × ripetizione, ([\d.]+) osservazioni")), r, "osservazioni trader x periodo")
    add("mkt.perms.short", thousands(MD(r, r"= [\d,]+ \(([\d.]+) permutazioni\) e")), r, "permutazioni di T1 e T3 (rotazioni e flusso)")
    add("mkt.perms.long", thousands(MD(r, r"e 1/[\d.]+ = [\d,]+ \(([\d.]+) permutazioni\)")), r, "permutazioni del placebo di occupazione (T2)")
    add("mkt.flow.grid", int(MD(r, r"griglia (\d)×")), r, "lato della griglia del flusso (celle per asse)")
    add("mkt.basin.splits", int(MD(r, r"Si ripete su (\d+) divisioni dei mercati")), r, "divisioni dei mercati per il test del bacino")
    p = "mercati/PREDIZIONI_MERCATI.md"
    add("mkt.wealth.lo", f(MD(p, r"wealth_rel in \[([\d,]+);"), 1), p, "finestra di ricchezza appaiata: estremo basso")
    add("mkt.wealth.hi", f(MD(p, r"wealth_rel in \[[\d,]+; ([\d,]+)\]"), 1), p, "finestra di ricchezza appaiata: estremo alto")
    s = "mercati/risultati_mercati.json"
    for k, st in (("full", "completo"), ("beh", "solo_comportamento")):
        add(f"mkt.{k}.nfeat", len(J(s, st, "loadings_PC1")), s, f"numero di feature {st}")
        add(f"mkt.{k}.lam1", f(J(s, st, "T0_autovalori")[0], 2), s, f"primo autovalore {st}")
        add(f"mkt.{k}.lam2", f(J(s, st, "T0_autovalori")[1], 2), s, f"secondo autovalore {st}")
        add(f"mkt.{k}.lam.p95", f(J(s, st, "T0_null_p95")[0], 2), s, f"null p95 del primo autovalore (colonne mescolate) {st}")
        add(f"mkt.{k}.rot.rep.p", pval(J(s, st, "T1_rep1_vs_rep2", "p")), s, f"p rotazione rip.1 vs 2 {st}")
        add(f"mkt.{k}.rot.skill.p", pval(J(s, st, "T1_abilita_alta_vs_bassa", "p")), s, f"p rotazione abilita {st}")
        add(f"mkt.{k}.occ.all.p95", f(J(s, st, "T2_occupazione_tutte", "placebo_p95")), s, f"placebo p95 occupazione non appaiata {st}")
        add(f"mkt.{k}.flow.rep.p", pval(J(s, st, "T3_flusso_rep1_vs_rep2", "p")), s, f"p flusso ripetizione {st}")
    s = "mercati/risultati_bacino_ldt.json"
    for k, st in (("full", "completo"), ("beh", "solo_comportamento")):
        add(f"mkt.{k}.basin.p", pval(J(s, st, "B_bacino_abilita", "p_mediana")), s, f"p mediano abilita-bacino {st}")
        add(f"mkt.{k}.basin.share", pp(J(s, st, "B_bacino_abilita", "quota_split_p05"), 0), s, f"% divisioni con p <= 0,05 {st}")
        add(f"mkt.{k}.basin.modal", max(J(s, st, "B_bacino_abilita", "bacini_scelti").values()), s, f"divisioni in cui viene scelta la regione piu frequente {st}")
    add("mkt.ldt.raw.bid", f(J(s, "completo", "C_learning_direction", "C1_delta_bid_errore_meno_senza")[0], 2), s, "C1 grezzo: variazione del prezzo dei bid dopo aver pagato troppo")
    add("mkt.ldt.raw.proj", f(J(s, "completo", "C_learning_direction", "C2_proiezione_errore_meno_senza")[0], 2), s, "C2 grezzo: passo verso la direzione vincente dopo errore (regressione verso la media)")
    s = "mercati/risultati_ldt_controllo.json"
    add("mkt.ldtc.bid", f(J(s, "completo|dbid~over", "beta_entro_cella"), 2), s, "C1 entro cella: variazione bid dopo aver pagato troppo, completo")
    add("mkt.ldtc.bid.lo", f(J(s, "completo|dbid~over", "ic95")[0], 2), s, "IC basso")
    add("mkt.ldtc.bid.hi", f(J(s, "completo|dbid~over", "ic95")[1], 2), s, "IC alto")
    add("mkt.ldtc.beh.bid", f(J(s, "solo_comportamento|dbid~over", "beta_entro_cella"), 2), s, "C1 entro cella, run solo_comportamento (stessa variabile grezza dei bid, cambia solo la divisione casuale dei mercati)")
    s = "mercati/risultati_ldt2.json"
    add("mkt.ldt.hl.diff", f(J(s, "D2_alta_meno_bassa", "diff"), 2), s, "D2: differenza alta meno bassa abilita")
    add("mkt.ldt.hl.p", pval(J(s, "D2_alta_meno_bassa", "p")), s, "p differenza alta meno bassa abilita")
    # aggiunte dopo la verifica indipendente della Sez. 5 (10/10/2026)
    s = "mercati/risultati_bacino_ldt.json"
    add("mkt.ldt.c3.diff", f(J(s, "completo", "C_learning_direction", "C3_alta_meno_bassa"), 2), s, "C3: differenza alta meno bassa abilita (grezza)")
    add("mkt.ldt.c3.p", pval(J(s, "completo", "C_learning_direction", "C3_p")), s, "p C3")
    s = "mercati/risultati_ldt_controllo.json"
    add("mkt.ldtc.beh.proj.lo", f(J(s, "solo_comportamento|proj~err", "ic95")[0], 2), s, "C2 entro cella, solo comportamento: IC basso")
    add("mkt.ldtc.beh.proj.hi", f(J(s, "solo_comportamento|proj~err", "ic95")[1], 2), s, "C2 entro cella, solo comportamento: IC alto")
    # parametri che esistono solo negli script di analisi
    s = "src/chess_pca/mercati/bacino_ldt.py"
    txt = (ROOT / s).read_text(encoding="utf-8")
    m = re.search(r"for _ in range\(1000 if s == 0 else (\d+)\)", txt)
    if not m:
        raise KeyError(f"{s}: permutazioni del test B non trovate")
    add("mkt.basin.perms", int(m.group(1)), s, "permutazioni del test B nelle divisioni dopo la prima (1.000 solo nella prima)")
    p = "mercati/PREDIZIONI_BACINO_LDT.md"
    add("mkt.basin.perms.reg", thousands(MD(p, r"stesso mercato \(([\d.]+) permutazioni\)")), p, "permutazioni del test B registrate")
    s = "src/chess_pca/mercati/analisi_mercati.py"
    txt = (ROOT / s).read_text(encoding="utf-8")
    m = re.search(r'R = R\[T\.groupby\("traj"\)\.size\(\) >= (\d+)\]', txt)
    if not m:
        raise KeyError(f"{s}: soglia minima di periodi non trovata")
    add("mkt.occ.minper", int(m.group(1)), s, "periodi minimi nella finestra di ricchezza per tenere una traiettoria (T2)")
    s = "src/chess_pca/umani/flusso.py"
    txt = (ROOT / s).read_text(encoding="utf-8")
    m = re.search(r"np\.linspace\(0, 1, (\d+)\)\[1:-1\]\); qy", txt)
    if not m:
        raise KeyError(f"{s}: griglia del flusso non trovata")
    add("chess.flow.grid", int(m.group(1)) - 1, s, "celle per asse della griglia del flusso negli scacchi (quantili)")
    m = re.search(r"ok = w >= (\d+)", txt)
    if not m:
        raise KeyError(f"{s}: soglia per cella non trovata")
    add("flow.mincell", int(m.group(1)), s, "osservazioni minime per cella in entrambi i gruppi (uguale in analisi_mercati.py)")


# ================================================================ GIOCHI DEGLI EDITOR (simulazioni)
def editors():
    s = "editori/risultati_rps.json"
    res = J(s)
    def pick(game, lam, mixed):
        for r in res:
            if r["game"] == game and r["lam"] == lam and r["mixed"] == mixed:
                return r
        raise KeyError((game, lam, mixed))
    r = pick("S", 0.2, True)
    add("rps.s.near", pp(r["share_time_near_nash"], 0), s, "% tempo a <0.05 da Nash, gioco stabile (lambda 0.2, mix)", "simulato")
    add("rps.s.mode", f(math.dist(r["mode_state"], [.25, .25, .5]), 2), s, "distanza moda-Nash, gioco stabile", "simulato")
    for lam, k in ((0.2, "l2"), (0.5, "l5")):
        r = pick("Ua", lam, True)
        add(f"rps.u.{k}.near", pp(r["share_time_near_nash"], 0), s, f"% tempo vicino a Nash, instabile lambda {lam}", "simulato")
        add(f"rps.u.{k}.dnash", f(r["dist_mean_to_nash"], 2), s, f"distanza media-Nash, instabile lambda {lam}", "simulato")
        add(f"rps.u.{k}.dtasp", f(r["dist_mean_to_tasp"], 2), s, f"distanza media-TASP, instabile lambda {lam}", "simulato")
        add(f"rps.u.{k}.dist", f(r["mean_dist_from_nash"], 2), s, f"distanza media da Nash degli stati, instabile lambda {lam}", "simulato")
    add("rps.periods", res[0]["periods"], s, "periodi simulati per gioco", "simulato")
    # aggiunte per la scaletta della Sez. 3 (10/10/2026, sessione 3)
    add("rps.s.pure.near", pp(pick("S", 0.2, False)["share_time_near_nash"], 0), s, "% tempo a <0.05 da Nash, gioco stabile, lambda 0.2, mosse pure", "simulato")
    add("rps.s.l5.near.max", pp(max(pick("S", 0.5, m)["share_time_near_nash"] for m in (False, True)), 0), s, "% massima di tempo vicino a Nash, gioco stabile, lambda 0.5 (cicla)", "simulato")
    add("rps.s.dens.nash", pp(pick("S", 0.2, True)["density_pct_at_nash"], 0), s, "% del tempo osservato in punti della mappa meno densi della proiezione di Nash, stabile lambda 0.2 mix", "simulato")
    add("rps.u.dens.max", pp(max(max(r["density_pct_at_nash"], r["density_pct_at_tasp"]) for r in res if r["game"] == "Ua"), 1), s, "massimo (fra configurazioni instabili, Nash e TASP) della % di tempo in punti meno densi", "simulato")
    add("rps.evr.min", pp(min(r["evr_2d"] for r in res), 0), s, "varianza spiegata PC1+PC2 minima fra le configurazioni RPS (%)", "simulato")
    add("rps.evr.max", pp(max(r["evr_2d"] for r in res), 0), s, "varianza spiegata PC1+PC2 massima fra le configurazioni RPS (%)", "simulato")
    add("rps.near.thr", f(float(re.search(r"axis=1\) < (\.\d+)\)", (OUT / "editori/rps_sim.py").read_text(encoding="utf-8")).group(1)), 2), "editori/rps_sim.py", "soglia di distanza da Nash per 'vicino a Nash' (simplesso)", "simulato")
    add("rps.agents", int(re.search(r"N=(\d+)", (OUT / "editori/rps_sim.py").read_text(encoding="utf-8")).group(1)), "editori/rps_sim.py", "agenti per gruppo simulato (come CFH)", "simulato")
    s = "editori/cfh2014_tabella2.json"
    t = J(s)
    unstable = {k: v for k, v in t["trattamenti"].items() if not k.startswith("S")}
    closer = sum(math.dist(v, t["tasp_Ub" if k.startswith("Ub") else "tasp_Ua"]) < math.dist(v, t["nash"]) for k, v in unstable.items())
    add("cfh.unstable.n", len(unstable), s, "trattamenti instabili in CFH Tab. 2")
    add("cfh.unstable.closertasp", closer, s, "trattamenti instabili con media piu vicina al TASP che a Nash")
    s = "editori/risultati_cavallo.json"
    h = {r["rule"]: r for r in J(s)}
    add("horse.imp.tpe", pp(h["impulse"]["share_near_TPE"], 0), s, "% periodi vicino all'equilibrio perfetto, impulse response", "simulato")
    add("horse.rl.tpe", pp(h["reinforcement"]["share_near_TPE"], 0), s, "% periodi vicino all'equilibrio perfetto, reinforcement", "simulato")
    add("horse.imp.pay", f(h["impulse"]["mean_total_payoff"]), s, "payoff totale medio impulse", "simulato")
    add("horse.rl.pay", f(h["reinforcement"]["mean_total_payoff"]), s, "payoff totale medio reinforcement", "simulato")
    add("horse.eff.max", pp(max(h["impulse"]["share_other_P3_R"], h["reinforcement"]["share_other_P3_R"]), 1), s,
        "% massima di periodi nella zona efficiente (G3 gioca R), fra le due regole", "simulato")
    add("horse.imperf.max", pp(max(h["impulse"]["share_near_imperfect"], h["reinforcement"]["share_near_imperfect"]), 0), s,
        "% massima di periodi vicino all'equilibrio imperfetto, fra le due regole", "simulato")
    add("horse.imp.evr", pp(h["impulse"]["evr_2d"], 0), s, "varianza spiegata PC1+PC2, impulse (%)", "simulato")
    add("horse.rl.evr", pp(h["reinforcement"]["evr_2d"], 0), s, "varianza spiegata PC1+PC2, reinforcement (%)", "simulato")
    add("cfh.stable.n", sum(k.startswith("S") for k in t["trattamenti"]), "editori/cfh2014_tabella2.json", "trattamenti stabili in CFH Tab. 2")
    s = "editori/risultati_cavallo_2.json"          # secondo lancio, PREREG_CAVALLO_2.md
    h2 = J(s)
    add("horse2.evr", pp(h2["evr_comune_PC1_PC2"], 0), s, "varianza spiegata PC1+PC2 della mappa comune (%)", "simulato")
    add("horse2.strangers.max", pp(max(r["share_zona_strangers"] for r in h2["regole"].values()), 1), s, "% massima di periodi nella zona degli strangers di NSS, fra le due regole", "simulato")
    add("horse2.dens.max", pp(max(max(r["dens_pct_strangers"], r["dens_pct_imperfetto"]) for r in h2["regole"].values()), 1), s, "massimo della % di tempo in punti meno densi dei profili umani proiettati", "simulato")
    add("horse2.imperf.max", pp(max(r["share_zona_imperfetto"] for r in h2["regole"].values()), 0), s, "% massima di periodi vicino all'equilibrio imperfetto (secondo lancio)", "simulato")
    add("horse2.thr", f(float(re.search(r"\|p1 − 0,5\| ≤ (\d+,\d+)", (OUT / "editori/PREREG_CAVALLO_2.md").read_text(encoding="utf-8")).group(1).replace(",", ".")), 2), "editori/PREREG_CAVALLO_2.md", "semi-ampiezza della zona strangers su p1 e p3")
    s = "editori/nss2025_fatti.json"                # numeri del paper, trascritti
    n = J(s)
    add("nss.str.p1", pp(n["strangers_lungo_periodo"]["p1_R"], 0), s, "NSS: % di R del giocatore 1, strangers lungo periodo")
    add("nss.str.p3", pp(n["strangers_lungo_periodo"]["p3_R"], 0), s, "NSS: % di R del giocatore 3, strangers lungo periodo")
    add("nss.str.p2lo", pp(n["strangers_lungo_periodo"]["p2_R_min"], 0), s, "NSS: % di R del giocatore 2, minimo")
    add("nss.str.p2hi", pp(n["strangers_lungo_periodo"]["p2_R_max"], 0), s, "NSS: % di R del giocatore 2, massimo")
    add("nss.part.z2", pp(n["frequenza_esito_imperfetto_z2"]["partners"], 0), s, "NSS: % esito imperfetto, partners")
    add("nss.str.z2", pp(n["frequenza_esito_imperfetto_z2"]["strangers_lungo"], 0), s, "NSS: % esito imperfetto, strangers lungo periodo")
    add("nss.imp.periods", n["impulse_response"]["periodi_convergenza"], s, "NSS: periodi alla convergenza della simulazione impulse response")


# ================================================================ RPS SU DATI UMANI (Wang, Xu & Zhou 2014; aggiunto 18/10/2026)
def rps_humans():
    """Sostituto pubblico dei dati di CFH: sottoinsieme WXZ2014 (stratEst). Registro: editori/rps_umani/REGISTRO_RPS_UMANI.md.
    Il verso del ciclo NON si usa (etichette probabilmente scambiate nel file distribuito)."""
    s = "editori/rps_umani/risultati_rps_umani.json"
    r = "editori/rps_umani/REGISTRO_RPS_UMANI.md"
    add("rpsh.subjects", int(MD(r, r"- (\d+) studenti")), r, "RPS umani: soggetti nel sottoinsieme pubblico")
    add("rpsh.rounds", int(MD(r, r"studenti, (\d+) round")), r, "RPS umani: round per gruppo")
    add("rpsh.groups", J(s, "n_groups"), s, "RPS umani: gruppi (ricostruiti)")
    add("rpsh.gsize", len(J(s, "groups_ids")[0]), s, "RPS umani: giocatori per gruppo")
    add("rpsh.samesign", J(s, "simplex", "n_same_sign_as_mean"), s, "RPS umani: gruppi con rotazione dello stesso segno nel simplesso (lisciatura 10)")
    add("rpsh.samesign.w5", J("editori/rps_umani/risultati_rps_umani_w5.json", "simplex", "n_same_sign_as_mean"), "editori/rps_umani/risultati_rps_umani_w5.json", "RPS umani: idem, lisciatura 5")
    add("rpsh.indiv", J(s, "post_hoc", "simplex_groups_p05"), s, "RPS umani: gruppi con rotazione significativa da soli (5% non corretto; post hoc)")
    add("rpsh.h2.samesign", J(s, "post_hoc", "simplex_h2", "n_same_sign_as_mean"), s, "RPS umani: gruppi con lo stesso segno nella seconda meta' (post hoc)")
    add("rpsh.p", pval(J(s, "simplex", "p")), s, "RPS umani: p della rotazione nel simplesso (permutazione dell'ordine dei round)")
    add("rpsh.perms", thousands(J(s, "nperm")), s, "RPS umani: permutazioni dell'ordine dei round")
    add("rpsh.posmap.samesign", J(s, "post_hoc", "map_position_only", "n_same_sign_as_mean"), s, "RPS umani: gruppi con rotazione dello stesso segno sulla mappa senza velocita' (post hoc)")


# ================================================================ SCACCHI: costanti di disegno (finestre, soglie, ripetizioni)
def chess_design():
    """Parametri di disegno citati nel testo della Sez. 4, letti dai file dei risultati o dalle pre-registrazioni."""
    s = "occupancy_grouped/occupancy_grouped_full_w20.json"
    add("win.short", J(s, "window"), s, "finestra corta: righe per prospettiva = mosse per giocatore (occupazione full_w20, contrasti T4, leva T6)")
    s = "analisi_settembre/27_09/PREREG_27_09.md"
    add("win.anti", int(MD(s, r"Finestre di (\d+) ply \(a 20")), s, "finestra del contrasto antichess/Three-Check (T5): righe per prospettiva = mosse per giocatore")
    s = "occupancy_grouped/REGISTRO_OCCUPAZIONE.md"
    add("occ.mm.cp", int(MD(s, r"\|material\| ≤ (\d+) cp")), s, "soglia dell'appaiamento sul materiale (centipawn)")
    s = "analisi_settembre/26_09/PREREG_26_09.md"
    add("occ.splits.n", int(MD(s, r"ripetuta su (\d+) divisioni")), s, "numero di divisioni training/test (T3a)")
    s = "analisi_settembre/27_09/REGISTRO_27_09.md"
    add("a2.thr", f(MD(s, r"inversioni grezze \(\\\|r\\\| > ([\d,]+)\)"), 2), s, "soglia |r| per contare un'inversione di segno (A2)")
    add("rot.perms", thousands(MD(s, r"20 ply, ([\d.]+) permutazioni")), s, "permutazioni dei contrasti appaiati (T4)")
    # aggiunte dopo la verifica indipendente del 10/10/2026
    s = "a2_partial_antichess.csv"
    rows = list(csv.DictReader(open(OUT / s, encoding="utf-8")))
    add("a2.anti.ncomp", sum(r["grezza_tgt"] != "" for r in rows), s, "coordinate con correlazione calcolabile in antichess (checks_avail senza varianza)")
    s = "a2_partial_3check.csv"
    add("a2.3check.checks.partref", f(CSV(s, "", "checks_avail", "parziale_ref"), 3), s, "corr. parziale checks_avail-esito, depth6 (predizione Three-Check)")
    add("a2.3check.checks.parttgt", f(CSV(s, "", "checks_avail", "parziale_tgt"), 3), s, "corr. parziale checks_avail-esito, Three-Check")
    s = "analisi_settembre/27_09/T5_27_09.json"
    add("rot.3check12.p", pval(J(s, "3check", "p_appaiato")), s, "p Three-Check (finestra T5)")
    s = "occupancy_grouped/occupancy_grouped_3check_whole.json"
    for side in ("W", "B"):
        add(f"occ.3check.mm.{side.lower()}.p", pval(J(s, "material_matched", side, "placebo_p")), s, f"p placebo Three-Check {side}")
        add(f"occ.3check.mm.{side.lower()}.p95", f(J(s, "material_matched", side, "placebo_p95_pp")), s, f"placebo p95 Three-Check {side}")
    s = "occupancy_grouped/occupancy_grouped_depth6.json"
    if J(s, "window") != 20:
        raise ValueError("occupancy_grouped_depth6.json: finestra attesa 20")
    for side in ("W", "B"):
        add(f"occ.d6w20.mm.{side.lower()}", f(J(s, "material_matched", side, "signed_gap_test_pp")), s, f"gap depth6 finestre corte {side}")
        add(f"occ.d6w20.mm.{side.lower()}.p", pval(J(s, "material_matched", side, "placebo_p")), s, f"p depth6 finestre corte {side}")
    s = "umani/flusso_robustezza.json"
    add("flow.cl.fa.dec1040.ratio", f(J(s, "decisive_e_10_40", "classica vs veloci", "rapporto_oss_null")), s, "flusso classica vs veloci, decisive e righe 10-40")
    add("flow.cl.fa.dec1040.p", pval(J(s, "decisive_e_10_40", "classica vs veloci", "p")), s, "p decisive e righe 10-40")


# ================================================================ main
def main():
    for fn in (corpora, chess_structure, chess_occupancy, chess_humans, chess_robustness, chess_design, markets, editors, rps_humans):
        fn()
    lines = [
        "% FILE GENERATO DA paper/tools/make_numbers.py -- NON MODIFICARE A MANO",
        "% Ogni numero del paper e' qui. Nel testo: \\val{chiave}. Fonte di ogni chiave in numbers_index.md",
        "\\makeatletter",
        "\\newcommand{\\val}[1]{\\ifcsname val@#1\\endcsname\\csname val@#1\\endcsname\\else\\errmessage{Chiave numerica non definita: #1}\\fi}",
    ]
    for key, value, source, desc, status in ENTRIES:
        tag = "" if status == "ok" else f" [{status.upper()}]"
        lines.append(f"\\expandafter\\def\\csname val@{key}\\endcsname{{{value}}}  % {desc}{tag} | {source}")
    lines.append("\\makeatother")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = ["# Indice dei numeri (generato da make_numbers.py — non modificare)", "",
          "Stato: `ok` = verificato dal file; `simulato` = dati simulati (Sez. 3); `provvisorio` = da NON usare finché non c'è il controllo.", "",
          "| chiave | valore | stato | descrizione | fonte |", "|---|---|---|---|---|"]
    for key, value, source, desc, status in ENTRIES:
        md.append(f"| `{key}` | {value} | {status} | {desc} | `{source}` |")
    (PAPER / "numbers_index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(ENTRIES)} valori scritti in paper/numbers.tex e paper/numbers_index.md")


if __name__ == "__main__":
    main()
