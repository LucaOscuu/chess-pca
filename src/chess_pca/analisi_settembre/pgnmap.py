import re, pandas as pd
def parse(path):
    txt=open(path,encoding='utf-8',errors='ignore').read()
    games=re.split(r'\n(?=\[Event )', txt)
    out=[]
    for g in games:
        fen=re.search(r'\[FEN "([^"]+)"\]', g)
        body=re.sub(r'\[[^\]]*\]','',g); body=re.sub(r'\{[^}]*\}','',body)
        toks=[t for t in body.split() if not re.match(r'^\d+\.+$',t) and t not in ('1-0','0-1','1/2-1/2','*') and not re.match(r'^\d+\.\.\.$',t)]
        toks=[re.sub(r'^\d+\.+','',t) for t in toks]; toks=[t for t in toks if t]
        stm = fen.group(1).split()[1] if fen else 'w'
        n=len(toks); w=(n+1)//2 if stm=='w' else n//2; b=n-w
        out.append((' '.join(fen.group(1).split()[:4]) if fen else None, w, b))
    return pd.DataFrame(out, columns=['start','nw','nb'])
def mapping(pgn, cache_white, min_moves=10):
    p=parse(pgn); kept=p[(p.nw>=min_moves)&(p.nb>=min_moves)].reset_index(drop=True)
    c=pd.read_parquet(cache_white).groupby('game_id').size()
    return kept, c
