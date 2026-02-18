"""Gymnasium environment wrapping TetrisSim for RL training."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .pieces import PieceType, ROTATION_COUNT, get_cells
from .tetris_sim import BOARD_COLS, BOARD_ROWS, TetrisSim

NUM_PIECE_TYPES = len(PieceType)  # 7
MAX_STEPS = 2000

# Reward shaping
LINE_REWARDS = {0: 0, 1: 1, 2: 3, 3: 5, 4: 8}
GAME_OVER_REWARD = -2
HEIGHT_PENALTY = -0.01  # per row of max board height, per step


class TetrisEnv(gym.Env):
    """Gymnasium wrapper around TetrisSim.

    Observation: flat float32 array of 228 elements:
        - board: 200 (20x10 binary)
        - current piece: 7 (one-hot)
        - next 3 pieces: 21 (3x7 one-hot)

    Action: Discrete(40) = rotation * 10 + column
        - rotation = action // 10 (0-3), clamped to valid range
        - column = action % 10 (0-9), clamped to valid range
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, seed=None, render_mode=None):
        super().__init__()
        self.sim = TetrisSim(seed=seed)
        self.render_mode = render_mode
        self._steps = 0

        obs_size = BOARD_ROWS * BOARD_COLS + NUM_PIECE_TYPES * 4  # 200 + 28 = 228
        self.observation_space = spaces.Box(0, 1, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(40)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim._rng.seed(seed)
        self.sim.reset()
        self._steps = 0
        return self._encode_obs(), {}

    def step(self, action):
        rotation = int(action) // 10
        column = int(action) % 10

        # Clamp column to valid range for this piece+rotation
        piece = self.sim.current_piece
        rotation = rotation % ROTATION_COUNT[piece]
        cells = get_cells(piece, rotation)
        min_col_offset = min(c for _, c in cells)
        max_col_offset = max(c for _, c in cells)
        min_valid = -min_col_offset
        max_valid = BOARD_COLS - 1 - max_col_offset
        column = max(min_valid, min(column, max_valid))

        _, sim_reward, done, info = self.sim.step(rotation, column)
        self._steps += 1

        lines = info.get("lines", 0)
        if done:
            reward = float(GAME_OVER_REWARD)
        else:
            reward = float(LINE_REWARDS.get(lines, lines * 2))
            # Height penalty: discourage tall boards
            max_height = 0
            for row in range(BOARD_ROWS):
                if self.sim.board[row].any():
                    max_height = BOARD_ROWS - row
                    break
            reward += HEIGHT_PENALTY * max_height

        truncated = self._steps >= MAX_STEPS and not done
        if truncated:
            done = False  # terminated=False, truncated=True

        return self._encode_obs(), reward, done, truncated, info

    def _encode_obs(self):
        board_flat = self.sim.board.flatten().astype(np.float32)

        current_oh = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
        current_oh[int(self.sim.current_piece)] = 1.0

        next_oh = np.zeros(3 * NUM_PIECE_TYPES, dtype=np.float32)
        for i, p in enumerate(self.sim.next_pieces[:3]):
            next_oh[i * NUM_PIECE_TYPES + int(p)] = 1.0

        return np.concatenate([board_flat, current_oh, next_oh])

    def render(self):
        if self.render_mode == "ansi":
            return self.sim.render()
        return None
