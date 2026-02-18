#!/usr/bin/env python3
"""Train a color-aware board evaluator using Double DQN.

Closely follows fischly/tetris-ai's training approach:
- Episode-based training (not step-based)
- Pro replays from expert play (20% of batches)
- Small replay buffer (10k) for fresh, high-quality data
- High learning rate (3e-3) with step decay
- Gamma ramps from 0.8 → 0.94
- Self-play game every 5 episodes

Step 1: Generate pro replays first:
    .venv/bin/python3 tools/generate_pro_replays.py models/good-cnn-2.pt --games 500

Step 2: Train:
    .venv/bin/python3 tools/train_dqn.py --device cuda --pro-replays pro-replays/expert.npz
    .venv/bin/python3 tools/train_dqn.py --device cuda  # without pro replays
"""

import argparse
import copy
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.color_board_evaluator import (
    build_color_cnn,
    batch_boards_to_tensor,
    init_from_fischly,
)
from src.ai.fast_sim import (
    EMPTY, enumerate_placements_py, warmup as warmup_numba,
)
from src.game.pieces import PieceType
from src.game.tetris_sim import TetrisSim, BOARD_COLS, BOARD_ROWS

# Reward scale matching fischly
REWARD_SCALE = {0: 0, 1: 50, 2: 150, 3: 300, 4: 800}
GAME_OVER_REWARD = -5000
MONOSQUARE_SCALED = 2000   # gold square (mono) — huge bonus to teach clustering
MULTISQUARE_SCALED = 1000  # silver square (multi)
ADJACENCY_REWARD_PER_PAIR = 25  # reward for each new same-type adjacent cell pair


def color_adjacency_bonus(color_board_before, color_board_after):
    """Count new same-type adjacent pairs created by the placement.

    Checks horizontal and vertical neighbors. Returns the increase in
    same-type adjacency count from before to after placement.
    """
    def _count_adj(board):
        # Horizontal: board[:, :-1] == board[:, 1:] where both are occupied
        occupied = board >= 0  # EMPTY = -1
        h_match = (board[:, :-1] == board[:, 1:]) & occupied[:, :-1] & occupied[:, 1:]
        # Vertical: board[:-1, :] == board[1:, :] where both are occupied
        v_match = (board[:-1, :] == board[1:, :]) & occupied[:-1, :] & occupied[1:, :]
        return int(h_match.sum()) + int(v_match.sum())

    return _count_adj(color_board_after) - _count_adj(color_board_before)


def rescale_reward(lines_cleared, new_mono, new_multi, done,
                   color_before=None, color_after=None):
    if done:
        return float(GAME_OVER_REWARD)
    r = float(REWARD_SCALE.get(lines_cleared, lines_cleared * 100))
    r += new_mono * MONOSQUARE_SCALED + new_multi * MULTISQUARE_SCALED
    if color_before is not None and color_after is not None:
        r += color_adjacency_bonus(color_before, color_after) * ADJACENCY_REWARD_PER_PAIR
    return r


class ProReplayBuffer:
    """Loads pre-recorded expert transitions from .npz file."""

    def __init__(self, path: str, device: torch.device):
        data = np.load(path)
        self.states = data["states"]       # (N, 20, 10) int8
        self.next_states = data["next_states"]
        self.rewards = data["rewards"]     # (N,) float32
        self.dones = data["dones"]         # (N,) bool
        self.n = len(self.states)
        self.device = device
        print(f"Loaded {self.n} pro replay transitions from {path}")

    def sample_batch(self, batch_size):
        """Return (state_tensor, next_state_tensor, rewards, dones) on device."""
        idx = np.random.randint(0, self.n, size=batch_size)
        states = [self.states[i] for i in idx]
        next_states = [self.next_states[i] for i in idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx].astype(np.float32)

        state_tensor = batch_boards_to_tensor(states).to(self.device)
        next_tensor = batch_boards_to_tensor(next_states).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        return state_tensor, next_tensor, rewards_t, dones_t


def play_one_game(sim, online_net, device, epsilon):
    """Play one game using epsilon-greedy, return list of transitions and stats."""
    sim.reset()
    color_board = sim.get_color_board()
    transitions = []

    while not sim.game_over:
        piece = sim.current_piece
        result_boards, rot_col, lines_arr, count = enumerate_placements_py(color_board, piece)

        if count == 0:
            # No valid non-game-over placements — game is effectively over
            transitions.append((color_board.copy(), GAME_OVER_REWARD, color_board.copy(), True))
            break

        # Epsilon-greedy: pick action
        if random.random() < epsilon:
            idx = random.randrange(count)
        else:
            boards_list = [result_boards[i] for i in range(count)]
            with torch.no_grad():
                tensor = batch_boards_to_tensor(boards_list).to(device)
                scores = online_net(tensor).squeeze(1)
                idx = int(scores.argmax().item())

        rot = int(rot_col[idx, 0])
        col = int(rot_col[idx, 1])

        _, sim_reward, done, info = sim.step(rot, col)

        line_count = info.get("lines", 0) if not done else 0
        nm = info.get("new_mono_squares", 0) if not done else 0
        nmulti = info.get("new_multi_squares", 0) if not done else 0

        next_color_board = sim.get_color_board()
        reward = rescale_reward(line_count, nm, nmulti, done,
                                color_before=color_board, color_after=next_color_board)
        transitions.append((color_board.copy(), reward, next_color_board.copy(), done))
        color_board = next_color_board

        if done:
            break

    return transitions, sim.lines_cleared, sim.pieces_placed, sim.mono_squares, sim.multi_squares


def train(args):
    device = torch.device(args.device)
    print(f"Training on {device}")
    print(f"Episodes: {args.episodes}, LR: {args.lr}, Batch: {args.batch_size}, Buffer: {args.buffer_size}")
    print(f"Gamma: {args.gamma_start}→{args.gamma_end}, Epsilon: {args.epsilon_start}→{args.epsilon_end}")

    # Warmup Numba
    print("Warming up Numba JIT...")
    warmup_numba()

    # Load pro replays
    pro_replays = None
    if args.pro_replays:
        pro_replays = ProReplayBuffer(args.pro_replays, device)

    # Networks — transfer learn from fischly or Xavier init
    online_net = build_color_cnn()
    if args.pretrained:
        print(f"Transfer learning from {args.pretrained}")
        init_from_fischly(online_net, args.pretrained)
    else:
        for m in online_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    online_net = online_net.to(device)

    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()

    optimizer = optim.AdamW(online_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
    loss_fn = nn.MSELoss()

    # Replay buffer — small, like fischly
    replay_buffer = deque(maxlen=args.buffer_size)

    # Schedules
    eps_step = (args.epsilon_end - args.epsilon_start) / args.epsilon_duration
    gamma_duration = int(args.episodes * args.gamma_ramp_fraction)

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    recent_lines = deque(maxlen=100)
    recent_pieces = deque(maxlen=100)
    recent_mono = deque(maxlen=100)
    recent_multi = deque(maxlen=100)
    recent_losses = deque(maxlen=100)

    epsilon = args.epsilon_start
    total_train_steps = 0
    start_time = time.time()

    # Fill replay buffer with initial games
    print("Filling replay buffer with initial games...")
    while len(replay_buffer) < args.min_buffer:
        transitions, _, _, _, _ = play_one_game(
            TetrisSim(), online_net, device, epsilon=1.0
        )
        replay_buffer.extend(
            (s, r, ns, d) for s, r, ns, d in transitions
        )
    print(f"Buffer filled: {len(replay_buffer)} transitions")

    for episode in range(1, args.episodes + 1):
        # Gamma schedule (power curve like fischly's DecayingDiscount)
        if episode < gamma_duration:
            gamma = args.gamma_start + (args.gamma_end - args.gamma_start) * (episode / gamma_duration) ** 0.6
        else:
            gamma = args.gamma_end

        # Play a self-play game every SIMULATE_EVERY episodes
        if episode % args.simulate_every == 1:
            sim = TetrisSim()
            transitions, lines, pieces, mono, multi = play_one_game(
                sim, online_net, device, epsilon
            )
            replay_buffer.extend(
                (s, r, ns, d) for s, r, ns, d in transitions
            )
            # Trim buffer
            while len(replay_buffer) > args.buffer_size:
                replay_buffer.popleft()

            recent_lines.append(lines)
            recent_pieces.append(pieces)
            recent_mono.append(mono)
            recent_multi.append(multi)

        # Decide: pro replay or self-play batch
        use_pro = (pro_replays is not None and random.random() < args.pro_replay_chance)

        if use_pro:
            state_tensor, next_tensor, rewards_t, dones_t = pro_replays.sample_batch(args.batch_size)
        else:
            batch = random.sample(list(replay_buffer), min(args.batch_size, len(replay_buffer)))
            states, rewards, next_states, dones_list = zip(*batch)
            state_tensor = batch_boards_to_tensor(list(states)).to(device)
            next_tensor = batch_boards_to_tensor(list(next_states)).to(device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones_t = torch.tensor([float(d) for d in dones_list], dtype=torch.float32, device=device)

        # Q-learning update
        online_net.train()
        q_values = online_net(state_tensor).squeeze(1)

        with torch.no_grad():
            next_q = target_net(next_tensor).squeeze(1)
            td_target = rewards_t + gamma * next_q * (1 - dones_t)

        loss = loss_fn(q_values, td_target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
        optimizer.step()

        recent_losses.append(loss.item())
        total_train_steps += 1

        # Update schedules
        if epsilon > args.epsilon_end:
            epsilon = max(args.epsilon_end, epsilon + eps_step)
        scheduler.step()

        # Update target network
        if episode % args.target_sync == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if episode % args.log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            avg_lines = np.mean(recent_lines) if recent_lines else 0
            avg_pieces = np.mean(recent_pieces) if recent_pieces else 0
            avg_mono = np.mean(recent_mono) if recent_mono else 0
            avg_multi = np.mean(recent_multi) if recent_multi else 0
            lr = optimizer.param_groups[0]["lr"]
            # TNT score estimate: 1/line + ~3 strips/square (gold=10, silver=5)
            tnt_score = avg_lines + avg_mono * 30 + avg_multi * 15
            print(f"Ep {episode:>6d}/{args.episodes} | "
                  f"eps={epsilon:.3f} g={gamma:.3f} lr={lr:.1e} | "
                  f"loss={avg_loss:.1f} | "
                  f"lines={avg_lines:.1f} pieces={avg_pieces:.0f} | "
                  f"mono={avg_mono:.2f} multi={avg_multi:.2f} | "
                  f"TNT≈{tnt_score:.1f} | "
                  f"buf={len(replay_buffer)} | {elapsed:.0f}s")

        # Save checkpoint
        if episode % args.save_every == 0:
            path = save_dir / f"color_cnn_ep{episode}.pt"
            torch.save(online_net.state_dict(), path)
            print(f"  Saved: {path}")

    # Final save
    final_path = save_dir / "color_cnn_final.pt"
    torch.save(online_net.state_dict(), final_path)
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s. Model: {final_path}")
    if recent_lines:
        avg_l = np.mean(recent_lines)
        avg_m = np.mean(recent_mono)
        avg_mu = np.mean(recent_multi)
        tnt = avg_l + avg_m * 30 + avg_mu * 15
        print(f"Last 100 games: avg lines={avg_l:.1f}, "
              f"pieces={np.mean(recent_pieces):.0f}, "
              f"mono={avg_m:.2f}, multi={avg_mu:.2f}, "
              f"TNT score≈{tnt:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Train color-aware DQN board evaluator")
    parser.add_argument("--episodes", type=int, default=6000,
                        help="Total training episodes (gradient updates)")
    parser.add_argument("--batch-size", type=int, default=164,
                        help="Training batch size")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Replay buffer capacity")
    parser.add_argument("--min-buffer", type=int, default=1000,
                        help="Min transitions before training starts")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Initial learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.9,
                        help="LR decay factor per step")
    parser.add_argument("--lr-step", type=int, default=300,
                        help="Episodes between LR decay steps")
    parser.add_argument("--gamma-start", type=float, default=0.80,
                        help="Discount factor start")
    parser.add_argument("--gamma-end", type=float, default=0.94,
                        help="Discount factor end")
    parser.add_argument("--gamma-ramp-fraction", type=float, default=0.67,
                        help="Fraction of episodes for gamma ramp")
    parser.add_argument("--epsilon-start", type=float, default=0.5,
                        help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.08,
                        help="Final epsilon")
    parser.add_argument("--epsilon-duration", type=int, default=3000,
                        help="Episodes for epsilon decay")
    parser.add_argument("--target-sync", type=int, default=100,
                        help="Episodes between target network sync")
    parser.add_argument("--simulate-every", type=int, default=5,
                        help="Play a self-play game every N episodes")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to fischly binary CNN weights for transfer learning")
    parser.add_argument("--pro-replays", type=str, default=None,
                        help="Path to pro replay .npz file")
    parser.add_argument("--pro-replay-chance", type=float, default=0.2,
                        help="Probability of using pro replay batch")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--save-every", type=int, default=10000,
                        help="Episodes between checkpoint saves")
    parser.add_argument("--save-path", type=str, default="models/color-dqn",
                        help="Directory for checkpoints")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Episodes between log output")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
