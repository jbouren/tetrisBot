"""Gymnasium environment with 2D spatial observations for CNN training."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .pieces import PieceType, ROTATION_COUNT, get_cells
from .tetris_sim import BOARD_COLS, BOARD_ROWS, TetrisSim

NUM_PIECE_TYPES = len(PieceType)  # 7
MAX_STEPS = 2000

# Observation channels: board(1) + current_piece(7) + next_pieces(3*7) = 29
NUM_CHANNELS = 1 + NUM_PIECE_TYPES * 4  # 29

# Reward shaping (same as MLP version)
LINE_REWARDS = {0: 0, 1: 1, 2: 3, 3: 5, 4: 8}
GAME_OVER_REWARD = -2
HEIGHT_PENALTY = -0.01


class TetrisEnvCNN(gym.Env):
    """Gymnasium wrapper with 2D spatial observations for CNN policies.

    Observation: float32 array of shape (20, 10, 29):
        - Channel 0: board occupancy (binary)
        - Channels 1-7: current piece one-hot (broadcast across board)
        - Channels 8-14: next piece 1 one-hot (broadcast)
        - Channels 15-21: next piece 2 one-hot (broadcast)
        - Channels 22-28: next piece 3 one-hot (broadcast)

    Action: Discrete(40) = rotation * 10 + column (same as MLP version)
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, seed=None, render_mode=None):
        super().__init__()
        self.sim = TetrisSim(seed=seed)
        self.render_mode = render_mode
        self._steps = 0

        self.observation_space = spaces.Box(
            0, 1, shape=(BOARD_ROWS, BOARD_COLS, NUM_CHANNELS), dtype=np.float32
        )
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
            max_height = 0
            for row in range(BOARD_ROWS):
                if self.sim.board[row].any():
                    max_height = BOARD_ROWS - row
                    break
            reward += HEIGHT_PENALTY * max_height

        truncated = self._steps >= MAX_STEPS and not done
        if truncated:
            done = False

        return self._encode_obs(), reward, done, truncated, info

    def _encode_obs(self):
        obs = np.zeros((BOARD_ROWS, BOARD_COLS, NUM_CHANNELS), dtype=np.float32)

        # Channel 0: board occupancy
        obs[:, :, 0] = self.sim.board.astype(np.float32)

        # Channels 1-7: current piece one-hot (broadcast to all cells)
        obs[:, :, 1 + int(self.sim.current_piece)] = 1.0

        # Channels 8-28: next 3 pieces one-hot (broadcast)
        for i, p in enumerate(self.sim.next_pieces[:3]):
            obs[:, :, 1 + (i + 1) * NUM_PIECE_TYPES + int(p)] = 1.0

        return obs

    def render(self):
        if self.render_mode == "ansi":
            return self.sim.render()
        return None
