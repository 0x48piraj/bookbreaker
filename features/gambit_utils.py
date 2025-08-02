import chess

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

def sac_at_ply(uci_list, max_ply=20):
    board = chess.Board()
    for ply, uci in enumerate(uci_list[:max_ply], start=1):
        before = compute_material_imbalance(board)
        try:
            board.push_uci(uci)
        except Exception:
            return None
        after = compute_material_imbalance(board)
        if after < before:
            return ply
    return None


# Example: King's Gambit
uci = "e2e4 e7e5 f2f4 e5f4".split()
print("First sac at ply:", sac_at_ply(uci))