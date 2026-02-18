#!/usr/bin/env python3
"""Watch the CNN board evaluator play The New Tetris on the emulator.

Uses the pretrained fischly/tetris-ai CNN to score all possible placements
and pick the best one. Same emulator control as play_rl.py.

Usage:
  .venv/bin/python3 tools/play_eval.py models/good-cnn-1.pt
  .venv/bin/python3 tools/play_eval.py models/good-cnn-1.pt --verbose
"""

import argparse
import logging
import os
import sys
import time
import shutil
from pathlib import Path


import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai.board_evaluator import BoardEvaluator, _find_landing_row
from src.ai.lookahead import find_best_placement_lookahead
from src.ai.color_board_evaluator import ColorBoardEvaluator
from src.ai.monoblock import (
    EMPTY, UNKNOWN, make_color_board, simulate_drop_color,
    find_best_placement_combined, monoblock_score,
)
from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.pieces import PieceType, ROTATION_COUNT, get_cells, get_width
from src.game.tetris_sim import BOARD_COLS
from src.game.vision import (
    read_board, read_board_brightness, read_board_clean, read_preview, strip_ghost,
    find_ghost_piece, read_reserve_piece,
    sample_preview_slot, classify_color, PREVIEW_SLOTS,
    PREVIEW_PIECE_X0, PREVIEW_PIECE_X1,
)

BOARD_ROWS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("play_eval")
logging.getLogger("src.emulator.core").setLevel(logging.WARNING)

# Timing constants (calibrated for CV-first deterministic model)
ROTATE_HOLD = 4          # frames to hold A_BUTTON for rotation
ROTATE_RELEASE = 3       # frames between rotations
TAP_HOLD = 4             # frames to hold D-pad for movement
TAP_RELEASE = 3          # frames between taps
HARD_DROP_HOLD = 2       # frames for hard drop
SPAWN_WAIT = 10           # before first input: piece controllability
CV_SETTLE_FRAMES = 12     # after hard drop: wait for lock, read CV (ghost-free)
POST_CV_FRAMES = 90     # after CV read: wait for next piece spawn + preview update

SPAWN_COL = {p: 4 for p in PieceType}
SPAWN_COL[PieceType.I] = 3  # I piece confirmed via diagnostic: spawns at col 3


def clear_screenshot_dir():
    """Delete all screenshots from the mupen64plus screenshot directory."""
    # Mupen64PlusCore sets this internally, but we need it for cleanup
    screenshot_dir = Path.home() / ".local/share/mupen64plus/screenshot"
    if screenshot_dir.exists():
        logger.info("Clearing screenshot directory: %s", screenshot_dir)
        for f in screenshot_dir.glob("*.png"):
            try:
                f.unlink()
            except OSError as e:
                logger.warning("Failed to delete screenshot %s: %s", f, e)
    else:
        logger.info("Screenshot directory does not exist, skipping cleanup.")


def clear_full_rows(board):
    """Programmatically clear completed rows from CV board.

    Returns (cleared_board, lines_cleared).
    """
    full_rows = board.all(axis=1)
    lines_cleared = int(full_rows.sum())
    if lines_cleared > 0:
        remaining = board[~full_rows]
        board = np.vstack([
            np.zeros((lines_cleared, BOARD_COLS), dtype=bool),
            remaining,
        ])
    return board, lines_cleared



def pname(p):
    """Safely get piece name — handles PieceType.I (value 0) correctly."""
    return p.name if p is not None else "?"


def pnames(pieces):
    """Format a list of pieces for logging."""
    return [pname(p) for p in pieces]


def piece_center_col(piece, rotation):
    """Get the center column of a piece in a given rotation."""
    cells = get_cells(piece, rotation)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    return (min_c + max_c) / 2.0


def estimated_col_after_rotation(piece, rotation, spawn_col):
    """Estimate piece bounding-box left column after rotation (center-preserving).

    TNT rotates pieces around their center, so the bounding box shifts.
    This model was calibrated against the real emulator in play_rl.py.
    """
    center0 = piece_center_col(piece, 0)
    centerN = piece_center_col(piece, rotation)
    return round(spawn_col + center0 - centerN)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def board_to_ascii(board):
    lines = ["+" + "-" * BOARD_COLS + "+"]
    for row in range(BOARD_ROWS):
        lines.append("|" + "".join("#" if board[row, c] else "." for c in range(BOARD_COLS)) + "|")
    lines.append("+" + "-" * BOARD_COLS + "+")
    return "\n".join(lines)


def board_with_placement(board, piece, rotation, column):
    """Show board with planned placement overlaid as 'X' characters."""
    from src.ai.board_evaluator import _simulate_drop

    cells = get_cells(piece, rotation)
    result = _simulate_drop(board, cells, column)
    if result is None:
        return board_to_ascii(board) + "\n  (placement simulation failed!)"

    placed_board = result[0]
    new_cells = placed_board & ~board

    col_labels = "".join(str(c) for c in range(BOARD_COLS))
    lines = [" " + col_labels, "+" + "-" * BOARD_COLS + "+"]
    for row in range(BOARD_ROWS):
        row_str = "|"
        for c in range(BOARD_COLS):
            if new_cells[row, c]:
                row_str += "X"
            elif board[row, c]:
                row_str += "#"
            else:
                row_str += "."
        row_str += f"| {row:2d}"
        lines.append(row_str)
    lines.append("+" + "-" * BOARD_COLS + "+")
    return "\n".join(lines)


def board_comparison(board_before, board_after, label_before="BEFORE", label_after="AFTER"):
    """Side-by-side ASCII boards with diff highlighted."""
    diff_new = board_after & ~board_before
    diff_gone = board_before & ~board_after

    col_labels = "".join(str(c) for c in range(BOARD_COLS))
    header = f"  {label_before:<{BOARD_COLS + 2}}   {label_after:<{BOARD_COLS + 2}}"
    border = "+" + "-" * BOARD_COLS + "+"
    lines = [header, f" {col_labels}            {col_labels}", f"{border}        {border}"]
    for row in range(BOARD_ROWS):
        left = "|"
        right = "|"
        for c in range(BOARD_COLS):
            left += "#" if board_before[row, c] else "."
            if diff_new[row, c]:
                right += "X"  # new cell
            elif diff_gone[row, c]:
                right += "~"  # disappeared cell (line clear)
            elif board_after[row, c]:
                right += "#"
            else:
                right += "."
        left += "|"
        right += f"| {row:2d}"
        lines.append(f"{left}        {right}")
    lines.append(f"{border}        {border}")
    lines.append("  X=new cell  ~=cleared  #=existing")
    return "\n".join(lines)


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



def find_piece_column(board, settled_board):
    """Find the column of the currently falling piece's bounding box (diff-based)."""
    diff = board & ~settled_board
    if not diff.any():
        return None

    cols = np.where(diff.any(axis=0))[0]
    if len(cols) == 0:
        return None

    return cols.min()


def find_piece_column_brightness(frame, settled_board, threshold=45, max_row=8):
    """Find falling piece column using brightness (falling pieces are brighter).

    TNT falling pieces have brightness ~100-140, settled pieces dim to ~50-75.
    Only searches top rows (0 to max_row) since the piece just spawned.
    This avoids false positives from TNT 4x4 square highlights on settled pieces.
    """
    from src.game.vision import read_board_brightness, BOARD_ROWS, BOARD_COLS
    brightness = read_board_brightness(frame)

    # Only look at top rows where the falling piece should be
    top_brightness = brightness[:max_row]
    top_settled = settled_board[:max_row]

    # Falling piece cells: occupied AND bright (>85) AND not in settled board
    occupied = top_brightness > threshold
    bright = top_brightness > 85
    falling_mask = occupied & bright & ~top_settled

    if not falling_mask.any():
        # Fallback: just look for cells not in settled board (still top rows only)
        diff = occupied & ~top_settled
        if not diff.any():
            return None
        cols = np.where(diff.any(axis=0))[0]
        return cols.min() if len(cols) > 0 else None

    cols = np.where(falling_mask.any(axis=0))[0]
    return cols.min() if len(cols) > 0 else None

# ---------------------------------------------------------------------------
# Placement execution (NEW CV-driven version)
# ---------------------------------------------------------------------------

def execute_placement(core, input_server, piece, rotation, column, settled_board, frame_delay,
                      board_threshold=40):
    """Execute a placement: rotate, SENSE, and move. Does NOT hard drop."""
    frames = 0

    # 1. Wait for piece to fully spawn and become controllable
    wait_frames(core, SPAWN_WAIT, frame_delay)
    frames += SPAWN_WAIT

    # --- Rotation Loop ---
    rot_diff = rotation % ROTATION_COUNT[piece]
    if rot_diff > 0:
        logger.info("  - Rotating %dx...", rot_diff)
        for i in range(rot_diff):
            frame = core.read_screen()
            board_before_rot = read_board(frame, threshold=board_threshold)

            hold_button(core, input_server, "A_BUTTON", ROTATE_HOLD, frame_delay)
            wait_frames(core, ROTATE_RELEASE, frame_delay)
            frames += ROTATE_HOLD + ROTATE_RELEASE

            # Verification loop: poll for up to 30 frames, checking every frame
            rotated = False
            for poll_attempt in range(30):
                frame = core.read_screen()
                board_after_rot = read_board(frame, threshold=board_threshold)
                if not np.array_equal(board_before_rot, board_after_rot):
                    rotated = True
                    break
                wait_frames(core, 1, frame_delay)
                frames += 1

            if not rotated:
                logger.warning("  - Rotation %d/%d not detected via CV!", i + 1, rot_diff)


    # --- Movement Loop ---
    # First, sense our actual starting column after rotation
    frame = core.read_screen()
    board_with_piece = read_board(frame, threshold=board_threshold)
    current_col = find_piece_column(board_with_piece, settled_board)
    if current_col is None:
        logger.warning("  - Could not find piece for movement! Using prediction.")
        current_col = estimated_col_after_rotation(piece, rotation, SPAWN_COL[piece])

    delta = column - current_col
    if delta != 0:
        direction = "L_DPAD" if delta < 0 else "R_DPAD"
        logger.info("  - Moving from col %d to %d (delta %d)...", current_col, column, delta)
        for i in range(abs(delta)):
            hold_button(core, input_server, direction, TAP_HOLD, frame_delay)
            wait_frames(core, TAP_RELEASE, frame_delay)
            frames += TAP_HOLD + TAP_RELEASE

            moved = False
            # Poll up to 15 frames, checking every frame
            for poll_attempt in range(15):
                frame = core.read_screen()
                board_after_move = read_board(frame, threshold=board_threshold)
                next_col = find_piece_column(board_after_move, settled_board)
                if next_col is not None and next_col != current_col:
                    current_col = next_col
                    moved = True
                    break
                wait_frames(core, 1, frame_delay)
                frames += 1

            if not moved:
                logger.warning("  - Movement %d/%d not detected via CV!", i + 1, abs(delta))
                break # Exit movement loop if a step fails

    return frames


def commit_hard_drop(core, input_server, settled_board, frame_delay, board_threshold):
    """Attempt to hard drop, with CV verification that it locked.

    Returns (new_settled_board, lines_cleared, success).
    """
    # 1. Find the falling piece to monitor its disappearance.
    frame = core.read_screen()
    board_before_drop = read_board(frame, threshold=board_threshold)
    diff = board_before_drop & ~settled_board
    if not diff.any():
        logger.warning("Commit: No falling piece detected before drop.")
        # Still try to drop, but can't verify disappearance.
        hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, frame_delay)
        wait_frames(core, CV_SETTLE_FRAMES, frame_delay)
        frame = core.read_screen()
        raw_board = read_board(frame, threshold=board_threshold)
        new_board, lines_cleared = clear_full_rows(raw_board)
        return new_board, lines_cleared, False # Cannot confirm success

    falling_rows = np.where(diff.any(axis=1))[0]
    top_falling_row = falling_rows.min() if len(falling_rows) > 0 else -1

    # 2. Loop until the piece disappears or we time out
    attempts = 0
    # Poll for up to ~2 seconds (40 attempts * 3 frames/attempt), checking every 3rd frame
    for poll_attempt in range(40):
        attempts += 1
        # Send hard drop input
        hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, frame_delay)
        wait_frames(core, 2, frame_delay) # Small wait for input to process

        # Check if the piece is gone from its original position
        frame = core.read_screen()
        board_after_drop = read_board(frame, threshold=board_threshold)

        # The piece at its original top row should be gone.
        if top_falling_row != -1 and not board_after_drop[top_falling_row].any():
            # SUCCESS! The piece is gone. Now wait for settle.
            wait_frames(core, CV_SETTLE_FRAMES, frame_delay)
            frame = core.read_screen()
            raw_board = read_board(frame, threshold=board_threshold)
            new_board, lines_cleared = clear_full_rows(raw_board)
            logger.info("  - Hard drop confirmed via CV after %d attempts.", attempts)
            return new_board, lines_cleared, True

        # small wait before re-sending drop command
        wait_frames(core, 1, frame_delay)


    logger.warning("  - Hard drop FAILED to verify via CV after %d attempts.", attempts)
    frame = core.read_screen()
    raw_board = read_board(frame, threshold=board_threshold)
    new_board, lines_cleared = clear_full_rows(raw_board)
    return new_board, lines_cleared, False


# ---------------------------------------------------------------------------
# Input sanity test
# ---------------------------------------------------------------------------

def test_single_rotation(core, input_server, frame_delay):
    logger.info("=" * 60)
    logger.info("INPUT SANITY TEST: single A_BUTTON press")
    logger.info("=" * 60)

    settled = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    frame = core.read_screen()
    board_before = read_board(frame)
    diff_before = list(zip(*np.where(board_before & ~settled)))

    hold_button(core, input_server, "A_BUTTON", ROTATE_HOLD, frame_delay)
    wait_frames(core, ROTATE_RELEASE, frame_delay)

    frame = core.read_screen()
    board_after = read_board(frame)
    diff_after = list(zip(*np.where(board_after & ~settled)))

    before_cols = sorted(set(c for _, c in diff_before)) if diff_before else []
    after_cols = sorted(set(c for _, c in diff_after)) if diff_after else []
    before_rows = sorted(set(r for r, _ in diff_before)) if diff_before else []
    after_rows = sorted(set(r for r, _ in diff_after)) if diff_after else []

    changed = (before_cols != after_cols) or (before_rows != after_rows)
    if changed:
        logger.info("ROTATION REGISTERED!")
    else:
        logger.warning("ROTATION NOT DETECTED.")
    logger.info("=" * 60)
    return changed


# ---------------------------------------------------------------------------
# Placement verification
# ---------------------------------------------------------------------------

def verify_placement(board_before, board_after, piece, rotation, column):
    """Compare where piece actually landed vs where we expected.

    Returns (expected_cells, actual_new_cells, match).
    """
    from src.ai.board_evaluator import _simulate_drop

    cells = get_cells(piece, rotation)
    result = _simulate_drop(board_before, cells, column)
    if result is None:
        return None, None, False

    expected_board = result[0]

    # New cells = cells that appeared (after & ~before)
    actual_new = board_after & ~board_before
    expected_new = expected_board & ~board_before

    # Compare (ignoring top 4 rows where falling piece might be)
    actual_set = set(zip(*np.where(actual_new[4:])))
    expected_set = set(zip(*np.where(expected_new[4:])))

    match = actual_set == expected_set
    return expected_set, actual_set, match


# ---------------------------------------------------------------------------
# Reserve/Swap
# ---------------------------------------------------------------------------

SWAP_HOLD = 4            # frames to hold L_TRIG for swap
SWAP_SETTLE_FRAMES = 20  # frames to wait after swap for animation


def execute_swap(core, input_server, frame_delay, board_threshold):
    """Press L_TRIG to swap current piece with reserve, wait for animation.

    Returns the new reserve piece type read via CV (or None if unreadable).
    """
    hold_button(core, input_server, "L_TRIG", SWAP_HOLD, frame_delay)
    wait_frames(core, SWAP_SETTLE_FRAMES, frame_delay)
    frame = core.read_screen()
    return read_reserve_piece(frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    clear_screenshot_dir()
    parser = argparse.ArgumentParser(description="Watch CNN evaluator play on emulator")
    parser.add_argument("weights", help="Path to CNN weights (.pt)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-pieces", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--board-threshold", type=float, default=40,
                        help="Brightness threshold for board reading (default 40)")
    parser.add_argument("--monoblock-weight", type=float, default=0.0,
                        help="Weight for monoblock heuristic (0=disabled, try 0.5-2.0)")
    parser.add_argument("--tiebreaker", type=float, default=0.0,
                        help="CNN score tie threshold for color tiebreaker (0=disabled, 0.02=2%%)")
    parser.add_argument("--lookahead", type=int, default=0,
                        help="Lookahead depth (0=disabled, 1-3=N-piece lookahead)")
    parser.add_argument("--beam-width", type=int, default=10,
                        help="Beam width for lookahead pruning (default 10)")
    parser.add_argument("--color-weights", type=str, default=None,
                        help="Path to color-aware CNN weights (.pt). Replaces binary evaluator + monoblock heuristic.")
    parser.add_argument("--step", action="store_true",
                        help="Pause before each placement for manual review")
    parser.add_argument("--no-swap", action="store_true",
                        help="Disable reserve/swap piece feature")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
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
    board_threshold = args.board_threshold

    logger.info("Board brightness threshold: %.0f", board_threshold)

    use_color_eval = args.color_weights is not None
    color_evaluator = None
    evaluator = None
    mono_weight = args.monoblock_weight
    tiebreaker_threshold = args.tiebreaker
    if tiebreaker_threshold > 0:
        logger.info("Color tiebreaker enabled: threshold=%.3f", tiebreaker_threshold)
    lookahead_depth = args.lookahead
    beam_width = args.beam_width
    if lookahead_depth > 0:
        logger.info("Lookahead enabled: depth=%d, beam_width=%d", lookahead_depth, beam_width)

    if use_color_eval:
        logger.info("Loading COLOR board evaluator from %s (device=%s)",
                     args.color_weights, args.device)
        color_evaluator = ColorBoardEvaluator(args.color_weights, device=args.device)
        # Color evaluator replaces both binary CNN and monoblock heuristic
        mono_weight = 0.0
        empty_score = color_evaluator.evaluate(make_color_board())
        logger.info("Empty board score (color CNN): %.2f", empty_score)
    else:
        logger.info("Loading board evaluator from %s (device=%s)", args.weights, args.device)
        if mono_weight > 0:
            logger.info("Monoblock weight: %.2f", mono_weight)
        evaluator = BoardEvaluator(args.weights, device=args.device)
        empty_score = evaluator.evaluate(np.zeros((20, 10), dtype=bool))
        logger.info("Empty board score: %.2f", empty_score)

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

    # Load save state (resume before load — critical for inputs to register)
    logger.info("Loading save state (slot 1)...")
    core.resume()
    time.sleep(0.5)
    core.load_state(slot=1)
    time.sleep(0.5)
    core.pause()
    time.sleep(0.3)
    core.advance_frame()
    time.sleep(0.1)

    # Input sanity test
    if not args.skip_sanity:
        if not test_single_rotation(core, input_server, frame_delay):
            logger.error("Input sanity test FAILED. Aborting.")
            input_server.clear()
            core.resume()
            input_server.stop()
            return
        input_server.clear()
        core.resume()
        time.sleep(0.5)
        core.load_state(slot=1)
        time.sleep(0.5)
        core.pause()
        time.sleep(0.3)
        core.advance_frame()
        time.sleep(0.1)

    # Bootstrap: hard-drop first piece to get to a clean, known state
    logger.info("=" * 60)
    logger.info("Bootstrap: Synchronizing with emulator...")
    logger.info("=" * 60)

    # We don't know what the first piece is, so just drop it
    wait_frames(core, SPAWN_WAIT, frame_delay)
    hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, frame_delay)
    logger.info("Bootstrap: Dropped initial unknown piece.")

    # In the ghost-free window, read the board and clear lines
    wait_frames(core, CV_SETTLE_FRAMES, frame_delay)
    frame = core.read_screen()
    raw_board = read_board(frame, threshold=board_threshold)
    settled_board, lines_cleared = clear_full_rows(raw_board)
    logger.info("Bootstrap: Board settled, %d lines cleared.", lines_cleared)

    # Wait for next piece to spawn, then get the first reliable preview
    logger.info("Bootstrap: Waiting for first preview...")
    wait_frames(core, POST_CV_FRAMES, frame_delay)
    frame = core.read_screen()
    saved_preview = read_preview(frame)
    if not saved_preview or saved_preview[0] is None:
        logger.error("Bootstrap failed: could not read preview after static wait. Aborting.")
        return

    current_piece = saved_preview[0]
    logger.info("Bootstrap complete. First piece: %s, Preview: %s",
                pname(current_piece), pnames(saved_preview))

    # Initialize reserve piece tracking
    use_swap = not args.no_swap
    reserve_piece = read_reserve_piece(frame)
    if use_swap:
        logger.info("Reserve piece: %s (swap enabled)", pname(reserve_piece))
    else:
        logger.info("Reserve piece: %s (swap disabled)", pname(reserve_piece))

    # Initialize color board (bootstrap cells are UNKNOWN since we don't know their type)
    color_board = make_color_board()
    color_board[settled_board] = UNKNOWN

    # Vars for main loop
    pieces_placed = 0
    swaps_done = 0
    lines_cleared_total = lines_cleared
    verify_matches = 0
    verify_mismatches = 0
    piece_stats = {p: {"ok": 0, "fail": 0} for p in PieceType}

    limit_str = f"max {args.max_pieces}" if args.max_pieces > 0 else "unlimited"
    logger.info("=== Starting CV-first playback (current: %s, %s pieces) ===",
                pname(current_piece), limit_str)

    try:
        while True:
            if args.max_pieces > 0 and pieces_placed >= args.max_pieces:
                logger.info("Reached max pieces limit (%d). Stopping.", args.max_pieces)
                break

            if current_piece is None:
                logger.error("Lost track of current piece. Aborting.")
                break

            # 1. EVALUATE
            t0 = time.time()
            if use_color_eval:
                rotation, column, score = color_evaluator.find_best_placement(
                    color_board, current_piece
                )
                cnn_score, mono_score = score, 0.0
            elif mono_weight > 0:
                rotation, column, score, cnn_score, mono_score = \
                    find_best_placement_combined(
                        evaluator, settled_board, color_board,
                        current_piece, mono_weight
                    )
            else:
                if lookahead_depth > 0:
                    rotation, column, score = find_best_placement_lookahead(
                        evaluator, settled_board, current_piece,
                        preview=saved_preview[1:lookahead_depth + 1],
                        depth=lookahead_depth,
                        beam_width=beam_width,
                    )
                elif tiebreaker_threshold > 0:
                    rotation, column, score = evaluator.find_best_placement_with_tiebreaker(
                        settled_board, color_board, current_piece, threshold=tiebreaker_threshold
                    )
                else:
                    rotation, column, score = evaluator.find_best_placement(
                        settled_board, current_piece
                    )
                cnn_score, mono_score = score, 0.0
            t_eval = time.time()

            # 1b. SWAP DECISION
            swapped = False
            if use_swap and reserve_piece is not None:
                if use_color_eval:
                    rot2, col2, score2 = color_evaluator.find_best_placement(
                        color_board, reserve_piece
                    )
                else:
                    if lookahead_depth > 0:
                        rot2, col2, score2 = find_best_placement_lookahead(
                            evaluator, settled_board, reserve_piece,
                            preview=saved_preview[1:lookahead_depth + 1],
                            depth=lookahead_depth,
                            beam_width=beam_width,
                        )
                    elif tiebreaker_threshold > 0:
                        rot2, col2, score2 = evaluator.find_best_placement_with_tiebreaker(
                            settled_board, color_board, reserve_piece, threshold=tiebreaker_threshold
                        )
                    else:
                        rot2, col2, score2 = evaluator.find_best_placement(
                            settled_board, reserve_piece
                        )
                if score2 > score:
                    logger.info("Swap: current=%s score=%.2f, reserve=%s score=%.2f -> SWAP",
                                pname(current_piece), score, pname(reserve_piece), score2)
                    new_reserve = execute_swap(core, input_server, frame_delay, board_threshold)
                    # Update state: current piece becomes reserve, reserve becomes current
                    old_current = current_piece
                    current_piece = reserve_piece
                    reserve_piece = new_reserve if new_reserve is not None else old_current
                    rotation, column, score = rot2, col2, score2
                    swapped = True
                    swaps_done += 1
                else:
                    logger.info("Swap: current=%s score=%.2f, reserve=%s score=%.2f -> KEEP",
                                pname(current_piece), score, pname(reserve_piece), score2)
            elif use_swap and reserve_piece is None:
                # First swap opportunity: stash a bad piece if score is very low
                # For now, just log that reserve is empty
                logger.debug("Reserve empty, no swap possible.")

            # 2. STEP MODE (Review)
            skipped = False
            if args.step:
                spawn_col = SPAWN_COL[current_piece]
                est_col = estimated_col_after_rotation(current_piece, rotation, spawn_col)
                delta = column - est_col
                rot_diff = rotation % ROTATION_COUNT[current_piece]
                print("\n" + "=" * 50)
                print(f"  PIECE #{pieces_placed + 1}: {current_piece.name}")
                print(f"  Rotation: {rot_diff}x CW  (target rot={rotation})")
                print(f"  Spawn col: {spawn_col}  →  After rotation: col {est_col}")
                print(f"  Target col: {column}  →  Movement: {abs(delta)} {'LEFT' if delta < 0 else 'RIGHT' if delta > 0 else '(none)'}")
                if mono_weight > 0:
                    print(f"  CNN: {cnn_score:.1f}  Mono: {mono_score:.2f}  Combined: {score:.1f}")
                else:
                    print(f"  Score: {score:.1f}")
                print(f"  Preview: {pnames(saved_preview)}")
                print("=" * 50)
                print("\nCurrent board + planned placement (X = piece):")
                print(board_with_placement(settled_board, current_piece, rotation, column))
                try:
                    resp = input("[Enter]=execute  [s]=skip(hard drop)  [q]=quit → ").strip().lower()
                except EOFError:
                    resp = "q"
                if resp == "q":
                    logger.info("User quit in step mode.")
                    break
                if resp == "s":
                    skipped = True

            board_before_exec = settled_board.copy()

            # 3. EXECUTE (move only, no drop)
            t_exec = time.time()
            if skipped:
                logger.info("  Skipping move, will hard drop at spawn.")
            else:
                execute_placement(
                    core, input_server, current_piece,
                    rotation, column, settled_board, frame_delay,
                    board_threshold=board_threshold
                )
            t_move = time.time()

            # 4. VERIFY PLACEMENT (column-based)
            commit = False
            actual_ghost_cells = set()
            # Poll for up to 15 frames for ghost to appear, checking every frame
            for poll_attempt in range(15):
                frame = core.read_screen()
                board_with_ghost = read_board(frame, threshold=board_threshold)
                actual_ghost_cells = find_ghost_piece(board_with_ghost, settled_board)

                # If we found a valid 4-cell ghost, check its column
                if len(actual_ghost_cells) == 4:
                    actual_column = min({c for r, c in actual_ghost_cells})
                    if actual_column == column:
                        commit = True
                        break  # Match found, exit poll loop

                wait_frames(core, 1, frame_delay)  # Wait one frame before retrying

            if not commit and not skipped:
                logger.warning("  - PLACEMENT MISMATCH! Plan aborted. Dropping piece as is.")
                if len(actual_ghost_cells) == 4:
                    actual_column = min({c for r,c in actual_ghost_cells})
                    logger.warning("    Landed in col %d, expected %d", actual_column, column)
                else:
                    logger.warning("    Could not find valid ghost piece to verify column.")

            # 5. COMMIT (Hard Drop)
            settled_board, lines_cleared, drop_ok = commit_hard_drop(
                core, input_server, settled_board, frame_delay, board_threshold
            )
            if lines_cleared > 0:
                lines_cleared_total += lines_cleared
            t_cv = time.time()


            # 8. VERIFY STATE CHANGE
            board_has_changed = not np.array_equal(board_before_exec, settled_board)
            if not board_has_changed:
                logger.warning("Board state unchanged after drop. Resyncing and retrying same piece.")
                wait_frames(core, 30, frame_delay)
                # RESYNC: Re-read the board to get ground truth before retrying
                frame = core.read_screen()
                raw_board = read_board(frame, threshold=board_threshold)
                settled_board, _ = clear_full_rows(raw_board)
                continue

            # --- If we get here, the piece was successfully placed ---
            pieces_placed += 1

            # Update color board (needed for monoblock, tiebreaker, or color evaluator)
            if use_color_eval or mono_weight > 0 or tiebreaker_threshold > 0:
                color_result = simulate_drop_color(
                    color_board, current_piece, rotation, column
                )
                if color_result is not None:
                    color_board = color_result[0]
                else:
                    # Color sim failed (placement mismatch) — reset with UNKNOWN
                    logger.warning("Color board sim failed, resetting to UNKNOWN.")
                    color_board = make_color_board()
                    color_board[settled_board] = UNKNOWN

            # 9. SPAWN WAIT & READ PREVIEW
            wait_frames(core, POST_CV_FRAMES, frame_delay)
            frame = core.read_screen()
            new_preview = read_preview(frame)

            if not new_preview or new_preview[0] is None:
                logger.error("Could not read next piece from preview. Aborting.")
                break

            # Placement verification (for stats) uses the final board state
            _, _, placement_ok = verify_placement(
                board_before_exec, settled_board, current_piece, rotation, column
            )
            if placement_ok:
                verify_matches += 1
                piece_stats[current_piece]["ok"] += 1
            else:
                verify_mismatches += 1
                piece_stats[current_piece]["fail"] += 1

            # Log placement result
            if mono_weight > 0:
                logger.info("Piece #%d: %s rot=%d col=%d cnn=%.1f mono=%.2f combined=%.1f | %s | verify=%s",
                            pieces_placed, current_piece.name, rotation, column,
                            cnn_score, mono_score, score,
                            "ABORTED" if not commit and not skipped else "COMMITTED",
                            "OK" if (commit or skipped) and placement_ok else "MISMATCH")
            else:
                logger.info("Piece #%d: %s rot=%d col=%d score=%.1f | %s | verify=%s",
                            pieces_placed, current_piece.name, rotation, column, score,
                            "ABORTED" if not commit and not skipped else "COMMITTED",
                            "OK" if (commit or skipped) and placement_ok else "MISMATCH")
            if lines_cleared > 0:
                logger.info("  +%d lines (total: %d)", lines_cleared, lines_cleared_total)

            # 11. ADVANCE PIECE
            current_piece = saved_preview[0]
            saved_preview = new_preview

            # Re-read reserve piece (may have changed if we swapped)
            if use_swap:
                reserve_piece = read_reserve_piece(frame)

            # 12. GAME OVER CHECK
            if settled_board[:4].sum() > 30:
                logger.info("Board topped out (rows 0-3 occupied). Game over.")
                break

            if pieces_placed % 50 == 0:
                clear_screenshot_dir()

            if pieces_placed % 50 == 0:
                clear_screenshot_dir()

            if pieces_placed % 20 == 0:
                status_msg = "--- Status: %d pieces, %d lines, %d swaps, verify=%d/%d OK" % (
                    pieces_placed, lines_cleared_total, swaps_done,
                    verify_matches, verify_matches + verify_mismatches)
                if use_color_eval or mono_weight > 0:
                    status_msg += ", mono=%.2f" % monoblock_score(color_board)
                status_msg += " ---"
                logger.info(status_msg)
                logger.info("Board:\n%s", board_to_ascii(settled_board))

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        input_server.clear()
        logger.info("=== FINAL: %d pieces, %d lines, %d swaps, verify=%d/%d OK ===",
                    pieces_placed, lines_cleared_total, swaps_done,
                    verify_matches, verify_matches + verify_mismatches)
        # Per-piece-type breakdown
        logger.info("Per-piece accuracy:")
        for pt in PieceType:
            ok = piece_stats[pt]["ok"]
            fail = piece_stats[pt]["fail"]
            total = ok + fail
            if total > 0:
                logger.info("  %s: %d/%d (%.0f%%)", pt.name, ok, total, 100 * ok / total)
        logger.info("Final board:\n%s", board_to_ascii(settled_board))
        logger.info("Resuming emulator...")

        core.resume()
        input_server.stop()


if __name__ == "__main__":
    main()
