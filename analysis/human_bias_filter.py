import os
import pandas as pd
import chess
import chess.engine

# CONFIG
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANDIDATES_CSV = os.path.join(BASE_DIR, "warehouse", "candidates", "cluster_analog_candidates.csv")
STOCKFISH_PATH = os.path.join(BASE_DIR, "stockfish", "stockfish-windows-x86-64-avx2.exe")

OUT_RANKED_CSV = os.path.join(BASE_DIR, "warehouse", "ranked", "ranked_candidates.csv")
DEPTH_SHALLOW = 8
DEPTH_DEEP = 24

# Scoring weights
ALPHA = 1.0  # 1 - dist
BETA  = 1.5  # shallow eval
GAMMA = 1.0  # deep punishment
DELTA = 0.5  # branching

os.makedirs(os.path.dirname(OUT_RANKED_CSV), exist_ok=True)

df = pd.read_csv(CANDIDATES_CSV)

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

rows = []
for _, row in df.iterrows():
    uci_seq = row["moves_uci"].split()
    board = chess.Board()
    for u in uci_seq:
        try:
            board.push_uci(u)
        except Exception:
            continue

    # Dual eval
    shallow = engine.analyse(board, chess.engine.Limit(depth=DEPTH_SHALLOW))["score"].white().score(mate_score=10000) / 100.0
    deep    = engine.analyse(board, chess.engine.Limit(depth=DEPTH_DEEP))["score"].white().score(mate_score=10000) / 100.0

    gap     = shallow - deep
    novelty = 1.0  # placeholder

    score = (
        ALPHA * (1 - row["distance"]) +
        BETA  * shallow -
        GAMMA * abs(deep) +
        DELTA * row["branching"]
    )

    rows.append({
        **row.to_dict(),
        "eval_shallow": shallow,
        "eval_deep": deep,
        "trap_gap": gap,
        "novelty_score": novelty,
        "score": score
    })

engine.quit()

df_ranked = pd.DataFrame(rows).sort_values("score", ascending=False)
df_ranked.to_csv(OUT_RANKED_CSV, index=False)

print(f"[+] Saved ranked candidates to {OUT_RANKED_CSV}")
