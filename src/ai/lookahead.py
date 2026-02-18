"""3-piece lookahead search for the Tetris CNN evaluator.

Uses beam-pruned search to look 1-3 pieces ahead, collecting all leaf board
states for a single batch CNN evaluation. Much faster than naive recursion.

Algorithm at depth D with beam width K:
- At each node: enumerate all valid placements (~30 per piece)
- Keep top-K by CNN score for the next level (beam pruning)
- Recurse to depth D using next preview piece
- At leaves: CNN-score all boards in one batch call
- Return the root placement that maximizes expected leaf score

Complexity with K=10:
  depth 1: 30 × 30 = 900 leaf boards (1 batch)
  depth 2: 10 × 10 × 30 = 3,000 leaf boards (~3 batches)
  depth 3: 10 × 10 × 10 × 30 = 30,000 leaf boards (~30 batches, ~1s)
"""

from __future__ import annotations

import numpy as np

from ..game.pieces import PieceType, ROTATION_COUNT, get_cells
from ..game.tetris_sim import BOARD_ROWS, BOARD_COLS
from .board_evaluator import _simulate_drop


def _enumerate_binary(board: np.ndarray, piece: PieceType
                      ) -> list[tuple[int, int, np.ndarray]]:
    """Enumerate all valid (rot, col, result_board) for a piece on binary board."""
    results = []
    num_rots = ROTATION_COUNT[piece]
    for rot in range(num_rots):
        cells = get_cells(piece, rot)
        min_col = min(c for _, c in cells)
        max_col = max(c for _, c in cells)
        for col in range(-min_col, BOARD_COLS - max_col):
            result = _simulate_drop(board, cells, col)
            if result is not None:
                result_board, game_over = result
                if not game_over:
                    results.append((rot, col, result_board))
    return results


def find_best_placement_lookahead(
    evaluator,
    board: np.ndarray,
    current_piece: PieceType,
    preview: list[PieceType],
    depth: int = 2,
    beam_width: int = 10,
) -> tuple[int, int, float]:
    """Find best placement using N-piece lookahead.

    Args:
        evaluator: BoardEvaluator (fischly CNN).
        board: 20x10 bool binary board (current settled state).
        current_piece: The piece to place now.
        preview: List of upcoming pieces (1-3 elements).
        depth: How many preview pieces to look ahead (0 = CNN only, like baseline).
        beam_width: At each non-leaf level, prune to top-K candidates.

    Returns:
        (rotation, column, score) — score is the best leaf CNN score.
    """
    if depth <= 0 or not preview:
        # Degenerate case: just use the CNN directly
        return evaluator.find_best_placement(board, current_piece)

    # Enumerate root placements
    root_placements = _enumerate_binary(board, current_piece)
    if not root_placements:
        return 0, BOARD_COLS // 2, float("-inf")

    # Collect (root_idx, leaf_board) pairs — one leaf per root for depth=1,
    # many per root for depth=2+. We recursively build the tree.
    # Strategy: for each root candidate, compute the best score achievable
    # by playing optimally for `depth` more pieces. Then pick the root with
    # the highest such score.

    root_boards = [(rot, col, rb) for rot, col, rb in root_placements]
    best_scores = _search_level(
        evaluator, root_boards, preview, depth, beam_width
    )

    best_root_idx = int(np.argmax(best_scores))
    rot, col, _ = root_boards[best_root_idx]
    return rot, col, float(best_scores[best_root_idx])


def _search_level(
    evaluator,
    candidates: list[tuple[int, int, np.ndarray]],
    preview: list[PieceType],
    depth: int,
    beam_width: int,
) -> np.ndarray:
    """For each candidate board, compute the best score achievable with lookahead.

    Returns an array of scores, one per candidate.
    """
    if depth <= 0 or not preview:
        # Score all candidate boards with CNN
        boards = [rb for _, _, rb in candidates]
        return evaluator.evaluate_batch(boards)

    next_piece = preview[0]
    remaining_preview = preview[1:]

    # For each candidate, enumerate next-piece placements and recurse
    # We collect all (parent_idx, child_board) pairs for batched scoring
    parent_best_scores = np.full(len(candidates), float("-inf"))

    # Batch approach: enumerate children for all parents, prune, recurse
    # Group children by parent, flatten, recurse on all, then reduce by max
    all_children: list[tuple[int, int, int, np.ndarray]] = []  # (parent_idx, rot, col, board)

    for parent_idx, (_, _, parent_board) in enumerate(candidates):
        children = _enumerate_binary(parent_board, next_piece)
        if not children:
            # Game over state — this parent gets -inf
            continue

        # Beam pruning: if depth > 1, score and keep top-K children
        if depth > 1 and len(children) > beam_width:
            child_boards = [rb for _, _, rb in children]
            child_scores = evaluator.evaluate_batch(child_boards)
            top_k_idx = np.argpartition(child_scores, -beam_width)[-beam_width:]
            children = [children[i] for i in top_k_idx]

        for rot, col, child_board in children:
            all_children.append((parent_idx, rot, col, child_board))

    if not all_children:
        return parent_best_scores

    # Recurse on all children at once
    child_candidates = [(r, c, b) for _, r, c, b in all_children]
    child_scores = _search_level(
        evaluator, child_candidates, remaining_preview, depth - 1, beam_width
    )

    # Reduce: each parent gets the max score over its children
    for i, (parent_idx, _, _, _) in enumerate(all_children):
        if child_scores[i] > parent_best_scores[parent_idx]:
            parent_best_scores[parent_idx] = child_scores[i]

    return parent_best_scores
