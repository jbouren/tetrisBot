"""Board evaluator using pretrained CNN from fischly/tetris-ai.

The CNN scores how "good" a board state is (Q-value). To pick a move:
enumerate all valid placements, simulate each, score the resulting boards,
and pick the placement with the highest score.

The CNN architecture (must match the pretrained weights):
    Conv2d(1, 32, 5) -> ReLU
    Conv2d(32, 64, 3) -> ReLU
    Conv2d(64, 64, 3) -> ReLU
    MaxPool2d(2)
    Flatten(384)
    Linear(384, 256) -> ReLU
    Linear(256, 128) -> ReLU
    Linear(128, 1)

Input: (batch, 1, 20, 10) float tensor (binary board)
Output: (batch, 1) Q-value score
"""

import numpy as np
import torch
import torch.nn as nn

from ..game.pieces import PieceType, ROTATION_COUNT, get_cells
from ..game.tetris_sim import BOARD_COLS, BOARD_ROWS


def build_cnn():
    """Build the CNN matching fischly/tetris-ai architecture."""
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(1 * 6 * 64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


class BoardEvaluator:
    """Evaluates Tetris boards using a pretrained CNN."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = build_cnn()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, board: np.ndarray) -> float:
        """Score a single 20x10 board. Higher = better."""
        tensor = torch.from_numpy(board.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        return self.model(tensor).item()

    @torch.no_grad()
    def evaluate_batch(self, boards: list[np.ndarray]) -> np.ndarray:
        """Score multiple boards at once. Returns array of scores."""
        batch = np.stack([b.astype(np.float32) for b in boards])
        tensor = torch.from_numpy(batch).unsqueeze(1).to(self.device)  # (N, 1, 20, 10)
        scores = self.model(tensor).squeeze(1).cpu().numpy()
        return scores

    def find_best_placement(self, board: np.ndarray, piece: PieceType
                            ) -> tuple[int, int, float]:
        """Find the best (rotation, column) for a piece on the given board.

        Enumerates all valid placements, simulates each, scores the resulting
        boards, and returns the placement with the highest score.

        Returns (rotation, column, score).
        """
        candidates = []
        result_boards = []

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
                        candidates.append((rot, col))
                        result_boards.append(result_board)

        if not candidates:
            # All placements lead to game over — just pick first valid column
            return 0, BOARD_COLS // 2, float("-inf")

        scores = self.evaluate_batch(result_boards)
        best_idx = int(np.argmax(scores))
        rot, col = candidates[best_idx]
        return rot, col, float(scores[best_idx])

    def find_best_placement_with_tiebreaker(
        self, board: np.ndarray, color_board: np.ndarray, piece: PieceType,
        threshold: float = 0.02
    ) -> tuple[int, int, float]:
        """Find best placement, using color adjacency to break CNN score ties.

        When multiple placements score within `threshold` fraction of the best
        CNN score, pick the one that creates the most same-type adjacent pairs.

        Returns (rotation, column, cnn_score).
        """
        from .monoblock import simulate_drop_color, color_adjacency_bonus

        candidates = []
        result_boards = []

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
                        candidates.append((rot, col))
                        result_boards.append(result_board)

        if not candidates:
            return 0, BOARD_COLS // 2, float("-inf")

        scores = self.evaluate_batch(result_boards)
        max_score = float(np.max(scores))

        # Find tied candidates: all with score >= max_score * (1 - threshold)
        # Handle negative scores: threshold is a fraction of the absolute max value
        if max_score >= 0:
            cutoff = max_score * (1.0 - threshold)
        else:
            cutoff = max_score * (1.0 + threshold)

        tied_mask = scores >= cutoff
        tied_indices = np.where(tied_mask)[0]

        if len(tied_indices) == 1:
            # No tiebreaker needed
            best_idx = int(tied_indices[0])
            return candidates[best_idx][0], candidates[best_idx][1], float(scores[best_idx])

        # Compute color adjacency bonus for tied candidates only
        best_bonus = -1
        best_idx = int(tied_indices[0])
        for idx in tied_indices:
            rot, col = candidates[idx]
            color_result = simulate_drop_color(color_board, piece, rot, col)
            if color_result is None:
                continue
            color_after = color_result[0]
            bonus = color_adjacency_bonus(color_board, color_after)
            if bonus > best_bonus:
                best_bonus = bonus
                best_idx = int(idx)

        rot, col = candidates[best_idx]
        return rot, col, float(scores[best_idx])


def _simulate_drop(board: np.ndarray, cells: list[tuple[int, int]], column: int
                   ) -> tuple[np.ndarray, bool] | None:
    """Drop piece cells at column, return (resulting_board, game_over) or None."""
    # Validate column bounds
    for _, dc in cells:
        c = column + dc
        if c < 0 or c >= BOARD_COLS:
            return None

    # Find landing row
    landing_row = None
    for start_row in range(BOARD_ROWS + 1):
        collision = False
        for dr, dc in cells:
            r = start_row + dr
            c = column + dc
            if r >= BOARD_ROWS:
                collision = True
                break
            if r >= 0 and board[r, c]:
                collision = True
                break
        if collision:
            landing_row = start_row - 1 if start_row > 0 else None
            break
    else:
        max_dr = max(dr for dr, _ in cells)
        landing_row = BOARD_ROWS - 1 - max_dr

    if landing_row is None:
        return None

    # Check placement validity
    for dr, dc in cells:
        if landing_row + dr < 0:
            return None

    # Place piece
    new_board = board.copy()
    for dr, dc in cells:
        new_board[landing_row + dr, column + dc] = True

    # Clear lines
    full_rows = np.where(new_board.all(axis=1))[0]
    if len(full_rows) > 0:
        keep = ~new_board.all(axis=1)
        remaining = new_board[keep]
        new_board = np.vstack([
            np.zeros((len(full_rows), BOARD_COLS), dtype=bool),
            remaining,
        ])

    game_over = bool(new_board[:2].any())
    return new_board, game_over


def _find_landing_row(board: np.ndarray, cells: list[tuple[int, int]], column: int) -> int | None:
    """Find the row where a piece would land, without placing it."""
    # Validate column bounds first
    for _, dc in cells:
        c = column + dc
        if not (0 <= c < BOARD_COLS):
            return None

    # Find landing row by checking for collisions from top to bottom
    landing_row = None
    for start_row in range(BOARD_ROWS + 1):
        collision = False
        for dr, dc in cells:
            r = start_row + dr
            c = column + dc
            if r >= BOARD_ROWS or (r >= 0 and board[r, c]):
                collision = True
                break
        if collision:
            landing_row = start_row - 1
            break
    else:
        # No collision, piece lands at the bottom
        max_dr = max(dr for dr, _ in cells) if cells else 0
        landing_row = BOARD_ROWS - 1 - max_dr

    # Check if placement is valid (not off the top of the board)
    for dr, _ in cells:
        if landing_row + dr < 0:
            return None

    return landing_row
