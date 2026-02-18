"""Color-aware board evaluator using Double DQN CNN.

Unlike the fischly CNN which sees binary (occupied/empty) boards, this CNN
takes 8-channel input:
  Channel 0: binary occupancy
  Channels 1-7: one-hot per PieceType (I=1, J=2, L=3, O=4, S=5, T=6, Z=7)

This allows the network to learn TNT's monoblock/multiblock mechanic —
clustering same-type pieces in 4x4 regions for bonus points.

Same "enumerate all placements, score each, pick best" strategy as BoardEvaluator.
"""

import numpy as np
import torch
import torch.nn as nn

from ..game.pieces import PieceType, ROTATION_COUNT, get_cells
from ..game.tetris_sim import BOARD_COLS, BOARD_ROWS
from .monoblock import EMPTY, simulate_drop_color

NUM_PIECE_TYPES = 7  # I, J, L, O, S, T, Z
NUM_CHANNELS = 1 + NUM_PIECE_TYPES  # occupancy + 7 piece-type channels


def build_color_cnn():
    """Build color-aware CNN (8-channel input, same conv structure as fischly)."""
    return nn.Sequential(
        nn.Conv2d(in_channels=NUM_CHANNELS, out_channels=32, kernel_size=5),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 6 * 1, 256),  # 20->16->14->12, pool->6; 10->6->4->2, pool->1
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def board_to_tensor(color_board: np.ndarray) -> torch.Tensor:
    """Convert int8 color board to 8-channel float tensor.

    Args:
        color_board: (20, 10) int8 array where EMPTY=-1 and values 0-6 are piece types.

    Returns:
        (8, 20, 10) float32 tensor.
    """
    # Vectorized: compare board against all 7 piece types at once
    # piece_ids shape: (7, 1, 1) for broadcasting against (20, 10)
    piece_ids = np.arange(NUM_PIECE_TYPES, dtype=np.int8).reshape(NUM_PIECE_TYPES, 1, 1)
    channels = np.zeros((NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    channels[0] = (color_board != EMPTY)
    channels[1:] = (color_board == piece_ids)
    return torch.from_numpy(channels)


# Pre-allocated piece ID array for vectorized batch conversion
_PIECE_IDS = np.arange(NUM_PIECE_TYPES, dtype=np.int8).reshape(1, NUM_PIECE_TYPES, 1, 1)


def batch_boards_to_tensor(color_boards: list[np.ndarray]) -> torch.Tensor:
    """Convert list of color boards to batched tensor. Fully vectorized.

    Returns: (N, 8, 20, 10) float32 tensor.
    """
    # Stack all boards: (N, 20, 10)
    stacked = np.stack(color_boards)  # (N, 20, 10)
    n = stacked.shape[0]
    result = np.empty((n, NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    # Channel 0: occupancy — (N, 20, 10)
    result[:, 0] = (stacked != EMPTY)
    # Channels 1-7: one-hot — broadcast (N, 1, 20, 10) == (1, 7, 1, 1) -> (N, 7, 20, 10)
    result[:, 1:] = (stacked[:, np.newaxis, :, :] == _PIECE_IDS)
    return torch.from_numpy(result)


def init_from_fischly(color_cnn: nn.Sequential, fischly_weights_path: str) -> None:
    """Initialize color CNN from fischly's pretrained binary CNN weights.

    Copies all layers directly except the first Conv2d:
    - fischly conv1: (32, 1, 5, 5) — 1 input channel (occupancy)
    - color conv1:   (32, 8, 5, 5) — 8 input channels

    For conv1: fischly's weights go into channel 0 (occupancy),
    channels 1-7 (piece types) are zero-initialized. This means the
    model starts playing exactly like fischly (~280 lines) and learns
    color awareness incrementally during fine-tuning.
    """
    fischly_state = torch.load(fischly_weights_path, map_location="cpu")
    color_state = color_cnn.state_dict()

    for key in fischly_state:
        if key == "0.weight":
            # Conv1 weight: fischly (32, 1, 5, 5) -> color (32, 8, 5, 5)
            color_state[key].zero_()
            color_state[key][:, 0:1, :, :] = fischly_state[key]
        elif key in color_state:
            # All other layers: same shape, copy directly
            color_state[key].copy_(fischly_state[key])

    color_cnn.load_state_dict(color_state)


class ColorBoardEvaluator:
    """Evaluates Tetris boards using a color-aware CNN."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = build_color_cnn()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, color_board: np.ndarray) -> float:
        """Score a single 20x10 color board. Higher = better."""
        tensor = board_to_tensor(color_board).unsqueeze(0).to(self.device)
        return self.model(tensor).item()

    @torch.no_grad()
    def evaluate_batch(self, color_boards: list[np.ndarray]) -> np.ndarray:
        """Score multiple color boards at once. Returns array of scores."""
        tensor = batch_boards_to_tensor(color_boards).to(self.device)
        scores = self.model(tensor).squeeze(1).cpu().numpy()
        return scores

    def find_best_placement(self, color_board: np.ndarray, piece_type: PieceType
                            ) -> tuple[int, int, float]:
        """Find the best (rotation, column) for a piece on the given color board.

        Enumerates all valid placements, simulates each on the color board,
        scores the resulting boards with the CNN, and returns the best.

        Returns (rotation, column, score).
        """
        candidates = []
        result_boards = []

        num_rots = ROTATION_COUNT[piece_type]
        for rot in range(num_rots):
            cells = get_cells(piece_type, rot)
            min_col = min(c for _, c in cells)
            max_col = max(c for _, c in cells)

            for col in range(-min_col, BOARD_COLS - max_col):
                result = simulate_drop_color(color_board, piece_type, rot, col)
                if result is None:
                    continue
                new_board, _lines = result
                # Check game over: top 2 rows occupied
                if np.any(new_board[:2] != EMPTY):
                    continue
                candidates.append((rot, col))
                result_boards.append(new_board)

        if not candidates:
            return 0, BOARD_COLS // 2, float("-inf")

        scores = self.evaluate_batch(result_boards)
        best_idx = int(np.argmax(scores))
        rot, col = candidates[best_idx]
        return rot, col, float(scores[best_idx])
