#!/usr/bin/env python3
"""Watch a trained RL model play The New Tetris on the emulator.

Uses tap-based movement with CV cross-validation:
  - Rotate to target rotation, tap LEFT/RIGHT to target column, hard drop
  - CV resyncs internal board every N frames to prevent drift
  - Internal sim game over doesn't stop play (emulator is ground truth)

Usage:
  .venv/bin/python3 tools/play_rl.py models/ppo_tetris_10000k.zip
  .venv/bin/python3 tools/play_rl.py models/ppo_tetris_10000k.zip --verbose
  .venv/bin/python3 tools/play_rl.py models/ppo_tetris_10000k.zip --resync-interval 300
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.pieces import PieceType, ROTATION_COUNT, get_cells
from src.game.tetris_env import BOARD_COLS, NUM_PIECE_TYPES
from src.game.vision import read_board, read_preview

BOARD_ROWS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("play_rl")
logging.getLogger("src.emulator.core").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Timing constants (calibrated from calibrate_inputs.py)
# ---------------------------------------------------------------------------

ROTATE_HOLD = 3       # Frames to hold A/B for rotation
ROTATE_RELEASE = 3    # Frames to release between rotations
TAP_HOLD = 4          # Frames to hold LEFT/RIGHT per tap
TAP_RELEASE = 3       # Frames to release between taps
HARD_DROP_HOLD = 2    # Frames to hold U_DPAD for hard drop
LOCK_WAIT_FRAMES = 20 # Wait after hard drop for piece to lock + next to spawn

# ---------------------------------------------------------------------------
# Spawn columns (calibrated: I-piece confirmed at col 4 from CV)
# ---------------------------------------------------------------------------

SPAWN_COL = {
    PieceType.I: 4,
    PieceType.O: 4,
    PieceType.T: 4,
    PieceType.S: 4,
    PieceType.Z: 4,
    PieceType.J: 4,
    PieceType.L: 4,
}


def piece_center_col(piece, rotation):
    cells = get_cells(piece, rotation)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    return (min_c + max_c) / 2.0


def estimated_col_after_rotation(piece, rotation):
    spawn = SPAWN_COL[piece]
    center0 = piece_center_col(piece, 0)
    centerN = piece_center_col(piece, rotation)
    return round(spawn + center0 - centerN)


# ---------------------------------------------------------------------------
# Internal board simulation (mirrors TetrisSim logic)
# ---------------------------------------------------------------------------

def find_landing_row(board, cells, column):
    for start_row in range(BOARD_ROWS + 1):
        for dr, dc in cells:
            r = start_row + dr
            c = column + dc
            if r >= BOARD_ROWS:
                return start_row - 1 if start_row > 0 else None
            if r >= 0 and board[r, c]:
                return start_row - 1 if start_row > 0 else None
    max_dr = max(dr for dr, _ in cells)
    return BOARD_ROWS - 1 - max_dr


def simulate_place(board, piece, rotation, column):
    cells = get_cells(piece, rotation)
    for _, dc in cells:
        c = column + dc
        if c < 0 or c >= BOARD_COLS:
            return board, 0, True
    landing_row = find_landing_row(board, cells, column)
    if landing_row is None:
        return board, 0, True
    for dr, dc in cells:
        if landing_row + dr < 0:
            return board, 0, True
    new_board = board.copy()
    for dr, dc in cells:
        new_board[landing_row + dr, column + dc] = True
    full_rows = np.where(new_board.all(axis=1))[0]
    n_cleared = len(full_rows)
    if n_cleared > 0:
        keep = ~new_board.all(axis=1)
        remaining = new_board[keep]
        new_board = np.vstack([
            np.zeros((n_cleared, BOARD_COLS), dtype=bool),
            remaining,
        ])
    game_over = bool(new_board[:2].any())
    return new_board, n_cleared, game_over


# ---------------------------------------------------------------------------
# Observation encoding (matches TetrisEnv._encode_obs)
# ---------------------------------------------------------------------------

def encode_obs(board, current_piece, next_pieces):
    board_flat = board.flatten().astype(np.float32)
    current_oh = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    current_oh[int(current_piece)] = 1.0
    next_oh = np.zeros(3 * NUM_PIECE_TYPES, dtype=np.float32)
    for i, p in enumerate(next_pieces[:3]):
        if p is not None:
            next_oh[i * NUM_PIECE_TYPES + int(p)] = 1.0
    return np.concatenate([board_flat, current_oh, next_oh])


def clamp_action(action, piece):
    rotation = int(action) // 10
    column = int(action) % 10
    rotation = rotation % ROTATION_COUNT[piece]
    cells = get_cells(piece, rotation)
    min_col_offset = min(c for _, c in cells)
    max_col_offset = max(c for _, c in cells)
    min_valid = -min_col_offset
    max_valid = BOARD_COLS - 1 - max_col_offset
    column = max(min_valid, min(column, max_valid))
    return rotation, column


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def board_to_ascii(board):
    lines = ["+" + "-" * BOARD_COLS + "+"]
    for row in range(BOARD_ROWS):
        lines.append("|" + "".join("#" if board[row, c] else "." for c in range(BOARD_COLS)) + "|")
    lines.append("+" + "-" * BOARD_COLS + "+")
    return "\n".join(lines)


def find_piece_positions(board, settled):
    diff = board & ~settled
    return list(zip(*np.where(diff)))


def describe_positions(positions):
    if not positions:
        return "none"
    rows = sorted(set(r for r, _ in positions))
    cols = sorted(set(c for _, c in positions))
    return f"rows={rows} cols={cols} ({len(positions)} cells)"


# ---------------------------------------------------------------------------
# Emulator input
# ---------------------------------------------------------------------------

def hold_button(core, input_server, button, frames, frame_delay):
    state = ControllerState()
    setattr(state, button, 1)
    for _ in range(frames):
        input_server.set_state(state)
        core.advance_frame()
        time.sleep(frame_delay)
    input_server.clear()


def wait_frames(core, n, frame_delay):
    for _ in range(n):
        core.advance_frame()
        time.sleep(frame_delay)


def cv_snapshot(core, settled, label):
    """Take a CV screenshot, log positions, return (frame, board). Advances 1 frame."""
    frame = core.read_screen()
    board = read_board(frame)
    positions = find_piece_positions(board, settled)
    if positions:
        falling = [(r, c) for r, c in positions if r < 15]
        ghost = [(r, c) for r, c in positions if r >= 15]
        logger.info("[%s] falling=%s  ghost=%s",
                    label, describe_positions(falling), describe_positions(ghost))
    else:
        logger.info("[%s] no new cells vs settled board", label)
    logger.info("[%s] board:\n%s", label, board_to_ascii(board))
    return frame, board


# ---------------------------------------------------------------------------
# Placement execution
# ---------------------------------------------------------------------------

def execute_placement(core, input_server, piece, rotation, column, frame_delay):
    """Execute placement: rotate, move, hard drop, wait. Returns frames used."""
    frames = 0

    # Rotate
    rot_diff = rotation % ROTATION_COUNT[piece]
    if rot_diff > 0:
        if rot_diff <= 2:
            for _ in range(rot_diff):
                hold_button(core, input_server, "A_BUTTON", ROTATE_HOLD, frame_delay)
                wait_frames(core, ROTATE_RELEASE, frame_delay)
                frames += ROTATE_HOLD + ROTATE_RELEASE
        else:
            hold_button(core, input_server, "B_BUTTON", ROTATE_HOLD, frame_delay)
            wait_frames(core, ROTATE_RELEASE, frame_delay)
            frames += ROTATE_HOLD + ROTATE_RELEASE

    # Move
    current_col = estimated_col_after_rotation(piece, rotation)
    delta = column - current_col
    if delta != 0:
        direction = "L_DPAD" if delta < 0 else "R_DPAD"
        for _ in range(abs(delta)):
            hold_button(core, input_server, direction, TAP_HOLD, frame_delay)
            wait_frames(core, TAP_RELEASE, frame_delay)
            frames += TAP_HOLD + TAP_RELEASE

    # Hard drop + lock wait
    hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, frame_delay)
    frames += HARD_DROP_HOLD
    wait_frames(core, LOCK_WAIT_FRAMES, frame_delay)
    frames += LOCK_WAIT_FRAMES

    return frames


def execute_placement_verbose(core, input_server, piece, rotation, column,
                               frame_delay, settled_board):
    """Execute placement with CV screenshot after each phase (for debugging)."""
    frames = 0

    _, _ = cv_snapshot(core, settled_board, "PHASE 0: spawn")
    frames += 1

    rot_diff = rotation % ROTATION_COUNT[piece]
    if rot_diff > 0:
        if rot_diff <= 2:
            for _ in range(rot_diff):
                hold_button(core, input_server, "A_BUTTON", ROTATE_HOLD, frame_delay)
                wait_frames(core, ROTATE_RELEASE, frame_delay)
                frames += ROTATE_HOLD + ROTATE_RELEASE
        else:
            hold_button(core, input_server, "B_BUTTON", ROTATE_HOLD, frame_delay)
            wait_frames(core, ROTATE_RELEASE, frame_delay)
            frames += ROTATE_HOLD + ROTATE_RELEASE
        _, _ = cv_snapshot(core, settled_board, f"PHASE 1: after {rot_diff} rotation(s)")
        frames += 1

    current_col = estimated_col_after_rotation(piece, rotation)
    delta = column - current_col
    logger.info("Movement: spawn=%d est_after_rot=%d target=%d delta=%d",
                SPAWN_COL[piece], current_col, column, delta)

    if delta != 0:
        direction = "L_DPAD" if delta < 0 else "R_DPAD"
        for _ in range(abs(delta)):
            hold_button(core, input_server, direction, TAP_HOLD, frame_delay)
            wait_frames(core, TAP_RELEASE, frame_delay)
            frames += TAP_HOLD + TAP_RELEASE
        _, _ = cv_snapshot(core, settled_board, f"PHASE 2: after {abs(delta)} {direction} taps")
        frames += 1

    hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, frame_delay)
    frames += HARD_DROP_HOLD
    wait_frames(core, LOCK_WAIT_FRAMES, frame_delay)
    frames += LOCK_WAIT_FRAMES

    _, _ = cv_snapshot(core, settled_board, "PHASE 3: after hard drop + lock")
    frames += 1

    return frames


# ---------------------------------------------------------------------------
# CV resync
# ---------------------------------------------------------------------------

def resync_from_cv(core, internal_board):
    """Read CV board and replace internal board, stripping top 4 rows (falling piece).

    Returns (new_internal_board, cv_board, frame).
    """
    frame = core.read_screen()
    cv_board = read_board(frame)

    # Strip top 4 rows where the next falling piece is visible
    synced = cv_board.copy()
    synced[:4] = False

    # Count drift before resync
    diff = (internal_board[4:] != cv_board[4:]).sum()
    logger.info("CV RESYNC: %d differing cells (rows 4-19) before resync", diff)
    if diff > 0:
        logger.info("  Internal (rows 14-19):\n%s", board_to_ascii(internal_board))
        logger.info("  CV (rows 14-19):\n%s", board_to_ascii(cv_board))

    return synced, cv_board, frame


# ---------------------------------------------------------------------------
# Input sanity test
# ---------------------------------------------------------------------------

def test_single_rotation(core, input_server, frame_delay, settled_board):
    logger.info("=" * 60)
    logger.info("INPUT SANITY TEST: single A_BUTTON press")
    logger.info("=" * 60)

    _, board_before = cv_snapshot(core, settled_board, "BEFORE rotation")
    ghost_before = find_piece_positions(board_before, settled_board)

    hold_button(core, input_server, "A_BUTTON", ROTATE_HOLD, frame_delay)
    wait_frames(core, ROTATE_RELEASE, frame_delay)

    _, board_after = cv_snapshot(core, settled_board, "AFTER rotation")
    ghost_after = find_piece_positions(board_after, settled_board)

    before_cols = sorted(set(c for _, c in ghost_before)) if ghost_before else []
    after_cols = sorted(set(c for _, c in ghost_after)) if ghost_after else []
    before_rows = sorted(set(r for r, _ in ghost_before)) if ghost_before else []
    after_rows = sorted(set(r for r, _ in ghost_after)) if ghost_after else []

    changed = (before_cols != after_cols) or (before_rows != after_rows)
    if changed:
        logger.info("ROTATION REGISTERED!")
    else:
        logger.warning("ROTATION NOT DETECTED.")
    logger.info("=" * 60)
    return changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Watch trained RL agent play on emulator")
    parser.add_argument("model", help="Path to trained model (.zip)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    parser.add_argument("--max-pieces", type=int, default=0,
                        help="Stop after N pieces (0 = unlimited)")
    parser.add_argument("--resync-interval", type=int, default=0,
                        help="(deprecated, resync now happens every piece)")
    parser.add_argument("--verbose", action="store_true",
                        help="Full CV diagnostics at each phase (slow, for debugging)")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip the input sanity test at startup")
    parser.add_argument("--core-lib", default="lib/libmupen64plus.so.2.0.0")
    parser.add_argument("--plugin-dir", default="lib")
    parser.add_argument("--data-dir", default="lib/data")
    parser.add_argument("--rom", default="/mnt/c/code/n64/roms/New Tetris, The (USA).z64")
    parser.add_argument("--gfx", default="lib/mupen64plus-video-GLideN64.so")
    parser.add_argument("--audio", default="lib/mupen64plus-audio-sdl.so")
    parser.add_argument("--input", default="lib/mupen64plus-input-bot.so")
    parser.add_argument("--rsp", default="lib/mupen64plus-rsp-hle.so")
    args = parser.parse_args()

    frame_delay = (1 / 60) / args.speed

    logger.info("Loading model from %s", args.model)
    model = PPO.load(args.model, device="cpu")

    input_server = InputServer(port=8082)
    input_server.start()

    logger.info("Starting emulator...")
    core = Mupen64PlusCore(args.core_lib, args.plugin_dir, args.data_dir)
    core.startup()
    core.load_rom(args.rom)
    core.attach_plugins(gfx=args.gfx, audio=args.audio, input=args.input, rsp=args.rsp)
    core.execute()

    logger.info("Waiting for emulator to boot...")
    time.sleep(5)

    # Load save state (matching calibrate_inputs.py: resume before load)
    logger.info("Loading save state (slot 1)...")
    core.resume()
    time.sleep(0.5)
    core.load_state(slot=1)
    time.sleep(2)
    core.pause()
    time.sleep(0.3)
    core.advance_frame()
    time.sleep(0.1)

    settled_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    # Input sanity test
    if not args.skip_sanity:
        if not test_single_rotation(core, input_server, frame_delay, settled_board):
            logger.error("Input sanity test FAILED. Aborting.")
            input_server.clear()
            core.resume()
            input_server.stop()
            return
        # Reload state fresh
        input_server.clear()
        core.resume()
        time.sleep(0.5)
        core.load_state(slot=1)
        time.sleep(2)
        core.pause()
        time.sleep(0.3)
        core.advance_frame()
        time.sleep(0.1)

    # Bootstrap: hard-drop first piece to sync preview queue
    logger.info("Bootstrap: hard-dropping first piece...")
    frame = core.read_screen()
    preview_before = read_preview(frame)
    logger.info("Preview before bootstrap: %s",
                [p.name if p else "?" for p in preview_before])

    hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, frame_delay)
    wait_frames(core, LOCK_WAIT_FRAMES, frame_delay)

    frame = core.read_screen()
    preview_after = read_preview(frame)
    board_after = read_board(frame)
    logger.info("Preview after bootstrap: %s",
                [p.name if p else "?" for p in preview_after])

    settled_board = board_after.copy()
    internal_board = settled_board.copy()

    current_piece = preview_before[0]
    if current_piece is None:
        logger.warning("Could not read first preview, defaulting to T")
        current_piece = PieceType.T
    preview = preview_after

    pieces_placed = 0
    lines_cleared_total = 0
    total_frames = 0
    resync_count = 0
    last_preview = preview

    limit_str = f"max {args.max_pieces}" if args.max_pieces > 0 else "unlimited"
    logger.info("=== Starting RL playback (current: %s, %s pieces, CV resync every piece) ===",
                current_piece.name, limit_str)

    try:
        while True:
            if args.max_pieces > 0 and pieces_placed >= args.max_pieces:
                logger.info("Reached max pieces limit (%d). Stopping.", args.max_pieces)
                break

            # Build observation from internal board
            obs = encode_obs(internal_board, current_piece, preview)
            action, _ = model.predict(obs, deterministic=True)
            rotation, column = clamp_action(action, current_piece)

            logger.info("Piece #%d: %s rot=%d col=%d (action=%d)",
                        pieces_placed + 1, current_piece.name, rotation, column, int(action))

            # Simulate placement on internal board
            new_board, lines, game_over = simulate_place(
                internal_board, current_piece, rotation, column
            )
            if game_over:
                # Internal sim thinks game over — resync from CV instead of stopping
                logger.warning("Internal sim game over — resyncing from CV...")
                synced, cv_board, frame = resync_from_cv(core, internal_board)
                internal_board = synced
                total_frames += 1  # read_screen advances 1 frame
                resync_count += 1

                # Check if the emulator board is actually topped out.
                # Use the synced board (top 4 rows stripped) so falling/ghost
                # piece doesn't trigger false positives. Real game over = rows
                # 4-7 are heavily occupied (>75% full = >30 of 40 cells).
                top_fill = int(synced[4:8].sum())
                logger.info("Top fill check: %d/40 cells in rows 4-7", top_fill)
                if top_fill > 30:
                    logger.info("Board topped out! Pieces: %d, Lines: %d",
                                pieces_placed, lines_cleared_total)
                    break
                else:
                    logger.info("Emulator still has room. Continuing with CV board.")
                    continue
            else:
                internal_board = new_board
                lines_cleared_total += lines
                if lines > 0:
                    logger.info("  +%d lines (total: %d)", lines, lines_cleared_total)

            # Execute on emulator
            if args.verbose:
                placement_frames = execute_placement_verbose(
                    core, input_server, current_piece,
                    rotation, column, frame_delay, settled_board
                )
            else:
                placement_frames = execute_placement(
                    core, input_server, current_piece,
                    rotation, column, frame_delay
                )

            total_frames += placement_frames


            pieces_placed += 1

            # Read CV for preview (always needed)
            frame = core.read_screen()
            cv_board = read_board(frame)
            total_frames += 1

            settled_board = cv_board.copy()
            new_preview = read_preview(frame)

            # Always resync internal board from CV (strip top 4 rows = falling piece)
            synced_board = cv_board.copy()
            synced_board[:4] = False
            drift = int((internal_board[4:] != cv_board[4:]).sum())
            if drift > 0:
                resync_count += 1
                if drift > 8:
                    logger.warning("CV resync: %d cells differ (rows 4-19)", drift)
            internal_board = synced_board

            # Update current piece from preview queue
            if last_preview[0] is not None:
                current_piece = last_preview[0]
            elif new_preview[0] is not None:
                current_piece = new_preview[0]
                logger.warning("Used new_preview[0] as current piece fallback")
            else:
                logger.warning("Preview unreadable, keeping: %s", current_piece.name)

            last_preview = new_preview
            preview = new_preview

            # Periodic status
            if pieces_placed % 10 == 0:
                logger.info("--- Status: %d pieces, %d lines, %d total frames, %d resyncs ---",
                            pieces_placed, lines_cleared_total, total_frames, resync_count)
                logger.info("Board:\n%s", board_to_ascii(internal_board))

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        input_server.clear()
        logger.info("=== FINAL: %d pieces, %d lines, %d frames, %d resyncs ===",
                    pieces_placed, lines_cleared_total, total_frames, resync_count)
        logger.info("Final internal board:\n%s", board_to_ascii(internal_board))
        logger.info("Resuming emulator...")
        core.resume()
        input_server.stop()


if __name__ == "__main__":
    main()
