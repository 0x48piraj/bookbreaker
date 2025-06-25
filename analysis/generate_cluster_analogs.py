#!/usr/bin/env python3
"""
generate_cluster_analogs.py

Generate new gambit candidates whose feature vectors lie
close to known gambit clusters.

Overview:
  - Loads clustered gambit data (with PCA+KMeans results).
  - Extracts a subset of non-gambit openings from the Lichess TSV.
  - From each seed line, generates sacrificial deviations at a fixed ply.
  - Evaluates candidates using Stockfish and extracts key features.
  - Compares each to gambit cluster centroids in normalized feature space.
  - Keeps those within average cluster radii — considered 'gambit analogs'.

Requirements:
  - Extracted gambit features `warehouse/features/gambit_features.csv` (raw features)
  - Clustered gambit features at `warehouse/clusters/gambit_features_clustered.csv` (with 'cluster' column)
  - Lichess openings TSV file at `warehouse/lichess-chess-openings/all.tsv`
  - Stockfish binary lives at `stockfish/stockfish-windows-x86-64-avx2.exe`
  - Python packages: chess, pandas, numpy, scikit-learn, tqdm

Outputs:
  - `warehouse/candidates/cluster_analog_candidates.csv` (new viable gambit candidates)

Usage:
  $ python generate_cluster_analogs.py
"""

import os
import pandas as pd
import numpy as np
import chess
import chess.engine

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ───── CONFIG ─────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLUSTER_CSV    = os.path.join(BASE_DIR, "warehouse", "clusters", "gambit_features_clustered.csv")
OPENINGS_TSV   = os.path.join(BASE_DIR, "warehouse", "lichess-chess-openings", "all.tsv")
STOCKFISH      = os.path.join(BASE_DIR, "stockfish", "stockfish-windows-x86-64-avx2.exe")

DEPTHS         = [8, 12, 16, 20]
PLY_DEPTH      = 12    # your desired horizon
RADIUS_FACTOR  = 2.0
TOP_K          = 20

# ───── HELPERS ─────
PIECE_VAL = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}

def mat_imb(board): 
    return sum(PIECE_VAL.get(p.piece_type,0)*(1 if p.color else -1)
               for p in board.piece_map().values())

def compute_material_imbalance(board: chess.Board) -> int:
    """
    Returns (white_material - black_material) in pawn units.
    """
    imbalance = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VAL.get(piece.piece_type, 0)
        imbalance += val if piece.color == chess.WHITE else -val
    return imbalance

def scan_for_sacrifices(uci_list, min_ply=4, max_ply=12):
    """
    Scan plies [min_ply…max_ply] in the UCI move list.
    Whenever the move at that ply drops material, record the prefix [0:ply].
    Returns a list of lists-of-chess.Move.
    """
    candidates = []
    max_ply = min(max_ply, len(uci_list))
    for ply in range(min_ply, max_ply + 1):
        board = chess.Board()
        # push first ply-1 moves
        for uci in uci_list[:ply-1]:
            board.push_uci(uci)
        before = compute_material_imbalance(board)
        # attempt the ply-th move
        try:
            mv = chess.Move.from_uci(uci_list[ply-1])
            board.push(mv)
        except Exception:
            continue
        after = compute_material_imbalance(board)
        if after < before:
            # rebuild the exact Move objects for the prefix
            tmp = chess.Board()
            seq = []
            for u in uci_list[:ply]:
                m = chess.Move.from_uci(u)
                tmp.push(m)
                seq.append(m)
            candidates.append(seq)
    return candidates

def feat(seq, eng):
    """Compute feature dict for a sequence of chess.Move objects."""
    b = chess.Board()
    for m in seq:
        b.push(m)
    f = {
        "mat_imbalance": mat_imb(b),
        "branching":     b.legal_moves.count()
    }

    # static evals
    for d in DEPTHS:
        info = eng.analyse(b, chess.engine.Limit(depth=d))
        # use keyword mate_score
        raw = info.get("score")
        if raw is not None:
            sc = raw.white().score(mate_score=10000)
            f[f"eval_d{d}"] = sc / 100.0
        else:
            f[f"eval_d{d}"] = None

    # swing: top-2 multipv lines
    try:
        infos = eng.analyse(
            b,
            chess.engine.Limit(depth=DEPTHS[-1]),
            multipv=2
        )
    except:
        infos = []

    scores = []
    for inf in infos:
        raw = inf.get("score")
        if raw is not None:
            scores.append(raw.white().score(mate_score=10000) / 100.0)
    f["swing"] = abs(scores[0] - scores[1]) if len(scores) >= 2 else 0.0

    f["motifs"] = 0
    return f

# ───── 1. Load clusters & compute centroids/radii ─────
dfc = pd.read_csv(CLUSTER_CSV)
num_cols = [c for c in dfc.columns if any(c.startswith(pref) for pref in
           ("mat_imbalance","branching","eval_d","swing","motifs"))]
scaler = StandardScaler().fit(dfc[num_cols])
Xs     = scaler.transform(dfc[num_cols])

centroids, radii = {}, {}
print("[+] Cluster thresholds (radius × factor):")
for c in sorted(dfc.cluster.unique()):
    Xi = Xs[dfc.cluster==c]
    mu = Xi.mean(axis=0)
    rad = np.linalg.norm(Xi-mu,axis=1).mean() * RADIUS_FACTOR
    centroids[c], radii[c] = mu, rad
    print(f"  C{c}: {rad:.3f}")

# ───── 2. Load seeds from UCI ─────
df_open = pd.read_csv(OPENINGS_TSV, sep="\t", dtype=str)
# use the 'uci' column
df_open["uci_list"] = df_open["uci"].str.split()
# pick non-gambit, with enough UCI moves
mask = ~df_open.name.str.lower().str.contains("gambit", na=False)
df_seed = df_open[mask].copy()
df_seed = df_seed[df_seed["uci_list"].map(len) >= PLY_DEPTH]
df_seed = df_seed.drop_duplicates("eco").head(50)

print(f"[*] Using {len(df_seed)} seeds (ply ≥ {PLY_DEPTH}).")

# ───── 3. Generate & filter ─────
eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
all_cands, accepted = [], []

for _, row in tqdm(df_seed.iterrows(), total=len(df_seed)):
    uci_seq = row["uci_list"]
    cands = scan_for_sacrifices(uci_seq, min_ply=4, max_ply=PLY_DEPTH)
    print(f" [*] Seed {row.eco}: {len(cands)} sacrificial deviations up to ply {PLY_DEPTH}")
    for seq in cands:
        f = feat(seq, eng)
        x = np.array([f[col] for col in num_cols])
        xs= scaler.transform([x])[0]
        dmap = {c: np.linalg.norm(xs-centroids[c]) for c in centroids}
        best, dist = min(dmap.items(), key=lambda kv: kv[1])
        all_cands.append((seq, best, dist, f))
        if dist <= radii[best]:
            accepted.append((seq,best,dist,f))

eng.quit()

# ───── 4. Fallback to Top-K if needed ─────
if not accepted and all_cands:
    print(f"[*] No hits; selecting Top-{TOP_K} by distance anyway.")
    all_cands.sort(key=lambda x: x[2])
    accepted = all_cands[:TOP_K]

print(f"[+] Generated {len(all_cands)} total, accepted {len(accepted)}.")

# ───── 5. Save ─────
rows = []
for seq, cl, dist, f in accepted:
    rows.append({
        "cluster": cl,
        "distance": dist,
        "moves_uci": " ".join(m.uci() for m in seq),
        **f
    })

out = pd.DataFrame(rows)
out_path = os.path.join(BASE_DIR, "warehouse", "candidates", "cluster_analog_candidates.csv")
out.to_csv(out_path, index=False)
print(f"[+] Saved {len(rows)} candidates to {out_path}")
