#!/usr/bin/env python3
"""
extract_and_cluster_gambits.py

Extract chess gambit features from Lichess openings TSV data and
perform PCA + KMeans clustering to group gambits by their engine-driven
and structural characteristics.

Steps:
  1. Load and filter gambit lines from `warehouse/lichess-chess-openings/all.tsv`.
  2. Extract features using Stockfish engine analysis:
     - material imbalance
     - branching factor (legal moves count)
     - evaluation scores at multiple depths
     - evaluation swing between best and second-best lines
  3. Perform PCA dimensionality reduction and KMeans clustering.
  4. Save raw features and clustered results to CSV files.

Requirements:
  - Lichess openings TSV file at `warehouse/lichess-chess-openings/all.tsv`
  - Stockfish binary lives at `stockfish/stockfish-windows-x86-64-avx2.exe`
  - Python packages: chess, pandas, scikit-learn, tqdm

Outputs:
  - `warehouse/features/gambit_features.csv` (raw extracted gambit features)
  - `warehouse/clusters/gambit_features_clustered.csv` (clustered gambit features with 'cluster' column)

Usage:
  $ python extract_and_cluster_gambits.py
"""

import os
import re
import pandas as pd
import chess
import chess.engine

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition    import PCA
from sklearn.cluster         import KMeans

# ─────── CONFIG ───────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TSV_PATH        = os.path.join(BASE_DIR,
                               "warehouse",
                               "lichess-chess-openings",
                               "all.tsv")
STOCKFISH_PATH  = os.path.join(BASE_DIR, "stockfish", "stockfish-windows-x86-64-avx2.exe")
DEPTHS          = [8, 12, 16, 20]
CLUSTER_COUNT   = 5

OUT_RAW_CSV     = os.path.join(BASE_DIR, "warehouse", "features", "gambit_features.csv")
OUT_CLUSTER_CSV = os.path.join(BASE_DIR, "warehouse", "clusters", "gambit_features_clustered.csv")

# ─────── HELPERS ───────
PIECE_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
}

def compute_material_imbalance(board: chess.Board) -> int:
    """
    Returns (white_material - black_material) in pawn units.
    """
    imbalance = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        imbalance += val if piece.color == chess.WHITE else -val
    return imbalance

def parse_san_moves(pgn_str: str):
    """
    Strip move numbers from a PGN-like string and return SAN tokens.
    """
    tokens = pgn_str.strip().split()
    return [tok for tok in tokens if not re.match(r'^\d+\.$', tok)]

# ─────── 1. LOAD & FILTER ───────
print("[*] Loading TSV and filtering gambits...")
df0 = pd.read_csv(TSV_PATH, sep="\t", dtype=str)

mask_gambit   = df0["name"].str.lower().str.contains("gambit", na=False)
mask_declined = df0["name"].str.lower().str.contains("declined", na=False)
gambits_df    = df0[mask_gambit & ~mask_declined].copy()
print(f"[+] {len(gambits_df)} gambit lines after filtering.")

os.makedirs(os.path.dirname(OUT_RAW_CSV), exist_ok=True)

# ─────── 2. FEATURE EXTRACTION ───────
print("[*] Extracting engine-driven features...")
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

rows = []
for _, row in tqdm(gambits_df.iterrows(), total=len(gambits_df)):
    name     = row["name"]
    pgn      = row["pgn"]
    san_moves = parse_san_moves(pgn)

    board = chess.Board()
    valid = True
    for san in san_moves:
        try:
            board.push_san(san)
        except Exception:
            valid = False
            break
    if not valid:
        continue

    mat_imbalance = compute_material_imbalance(board)
    branching     = board.legal_moves.count()

    # Eval trajectory
    evals = []
    for d in DEPTHS:
        info  = engine.analyse(board, chess.engine.Limit(depth=d))
        score = info["score"].white().score(mate_score=10000)
        evals.append(score / 100.0)

    # Swing via analyse + multipv=2
    try:
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=DEPTHS[-1]),
            multipv=2
        )
        scores = [inf["score"].white().score(mate_score=10000)/100.0
                  for inf in infos if "score" in inf]
        swing  = abs(scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    except Exception:
        swing = 0.0

    motif_count = 0

    rows.append({
        "eco":            row["eco"],
        "name":           name,
        "moves":          " ".join(san_moves),
        "mat_imbalance":  mat_imbalance,
        "branching":      branching,
        "swing":          swing,
        **{f"eval_d{d}": e for d, e in zip(DEPTHS, evals)},
        "motifs":         motif_count,
    })

engine.quit()

df = pd.DataFrame(rows)
df.to_csv(OUT_RAW_CSV, index=False)
print(f"[+] Saved raw features to {OUT_RAW_CSV} (rows: {len(df)})")

# ─────── 3. CLUSTERING ───────
print("[*] Running PCA + K-means clustering...")
num_cols = [c for c in df.columns
            if c.startswith(("mat_imbalance","branching","eval_d","swing","motifs"))]
X  = StandardScaler().fit_transform(df[num_cols])

pca = PCA(n_components=min(5, X.shape[1]))
Xp  = pca.fit_transform(X)

km     = KMeans(n_clusters=CLUSTER_COUNT, random_state=42)
labels = km.fit_predict(Xp)

df["cluster"] = labels

os.makedirs(os.path.dirname(OUT_CLUSTER_CSV), exist_ok=True)
df.to_csv(OUT_CLUSTER_CSV, index=False)
print(f"[+] Clustering complete; output written to {OUT_CLUSTER_CSV}")

# ─────── 4. QUICK INSPECTION ───────
print("\nSample per cluster:")
for c in range(CLUSTER_COUNT):
    sub = df[df.cluster == c]
    sample = sub["name"].sample(min(5, len(sub))).tolist()
    print(f" • Cluster {c} ({len(sub)} items): {sample}")
