#!/usr/bin/env python3
"""Test spawn columns and rotation for all piece types encountered from save state.

For each piece in the save state sequence:
  1. Reload state, fast-forward to that piece
  2. Hard drop with NO rotation/movement → record landing columns
  3. Reload, fast-forward, rotate 1x CW, hard drop → record columns after rotation

This tells us:
  - spawn_col for each piece type (min_col of landing with no input)
  - column after 1 CW rotation (to validate rotation model)

Usage:
  .venv/bin/python3 tools/test_spawn_cols.py
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.pieces import PieceType, ROTATION_COUNT, get_cells, normalize_cells, PIECE_SHAPES
from src.game.vision import (
    read_board, read_board_clean, read_board_brightness, read_preview, board_to_ascii,
)

ROTATE_HOLD = 4
TAP_HOLD = 4
TAP_RELEASE = 3
HARD_DROP_HOLD = 2
LOCK_WAIT = 30
SPAWN_WAIT = 5


def hold_button(core, input_server, button, frames, fd):
    state = ControllerState()
    setattr(state, button, 1)
    for _ in range(frames):
        input_server.set_state(state)
        core.advance_frame()
        time.sleep(fd)
    input_server.clear()


def wait_frames(core, n, fd):
    for _ in range(n):
        core.advance_frame()
        time.sleep(fd)


def reload_state(core):
    core.resume()
    time.sleep(0.5)
    core.load_state(slot=1)
    time.sleep(2)
    core.pause()
    time.sleep(0.3)
    core.advance_frame()
    time.sleep(0.1)


def fast_drop(core, input_server, fd):
    """Hard drop the current piece as fast as possible (no wait/rotation)."""
    hold_button(core, input_server, "U_DPAD", HARD_DROP_HOLD, fd)
    wait_frames(core, LOCK_WAIT, fd)


def identify_piece(diff_cells):
    """Identify piece type and rotation from a set of 4 cells."""
    if len(diff_cells) != 4:
        return None, None
    min_r = min(r for r, c in diff_cells)
    min_c = min(c for r, c in diff_cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in diff_cells)

    for pt in PieceType:
        for rot in range(4):
            shape = normalize_cells(get_cells(pt, rot))
            if shape == normalized:
                return pt, rot
    return None, None


def get_diff_cells(board_before, board_after):
    """Get list of (row, col) cells that are new in board_after."""
    diff = board_after & ~board_before
    return list(zip(*np.where(diff))) if diff.any() else []


def main():
    fd = 1 / 60  # frame delay

    input_server = InputServer(port=8082)
    input_server.start()

    core = Mupen64PlusCore("lib/libmupen64plus.so.2.0.0", "lib", "lib/data")
    core.startup()
    core.load_rom("/mnt/c/code/n64/roms/New Tetris, The (USA).z64")
    core.attach_plugins(
        gfx="lib/mupen64plus-video-GLideN64.so",
        audio="lib/mupen64plus-audio-sdl.so",
        input="lib/mupen64plus-input-bot.so",
        rsp="lib/mupen64plus-rsp-hle.so",
    )
    core.execute()

    print("Waiting for emulator to boot...")
    time.sleep(5)

    NUM_PIECES = 8  # test first 8 pieces from save state

    # =====================================================================
    # PHASE 1: Determine piece sequence and spawn columns
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: SPAWN COLUMNS (no rotation, no movement)")
    print("=" * 60)

    spawn_results = {}  # piece_index -> (PieceType, rot_observed, min_col, cols, rows)

    for piece_idx in range(NUM_PIECES):
        reload_state(core)

        # Fast-forward: drop pieces 0..piece_idx-1 to get to piece_idx
        settled = np.zeros((20, 10), dtype=bool)
        for skip in range(piece_idx):
            wait_frames(core, SPAWN_WAIT, fd)
            fast_drop(core, input_server, fd)
            frame = core.read_screen()
            settled = read_board_clean(frame, threshold=65)

        # Now piece_idx is the active piece. Read settled board.
        frame = core.read_screen()
        settled = read_board_clean(frame, threshold=65)

        # Drop with spawn wait, no rotation, no movement
        wait_frames(core, SPAWN_WAIT, fd)
        fast_drop(core, input_server, fd)

        frame = core.read_screen()
        board_after = read_board_clean(frame, threshold=65)
        diff_cells = get_diff_cells(settled, board_after)

        pt, rot = identify_piece(diff_cells)
        if diff_cells:
            cols = sorted(set(c for r, c in diff_cells))
            rows = sorted(set(r for r, c in diff_cells))
            min_col = min(c for r, c in diff_cells)
        else:
            cols, rows, min_col = [], [], -1

        spawn_results[piece_idx] = (pt, rot, min_col, cols, rows)
        print(f"\n  Piece #{piece_idx}: type={pt.name if pt else '?'} rot={rot} "
              f"min_col={min_col} cols={cols} rows={rows} "
              f"({len(diff_cells)} new cells)")

    # =====================================================================
    # PHASE 2: Test rotation (1x CW) for each piece
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: ROTATION TEST (1x CW via A_BUTTON)")
    print("=" * 60)

    rot_results = {}

    for piece_idx in range(min(NUM_PIECES, 6)):  # test first 6
        reload_state(core)

        # Fast-forward to piece_idx
        settled = np.zeros((20, 10), dtype=bool)
        for skip in range(piece_idx):
            wait_frames(core, SPAWN_WAIT, fd)
            fast_drop(core, input_server, fd)
            frame = core.read_screen()
            settled = read_board_clean(frame, threshold=65)

        frame = core.read_screen()
        settled = read_board_clean(frame, threshold=65)

        # Rotate 1x CW, then hard drop
        wait_frames(core, SPAWN_WAIT, fd)
        hold_button(core, input_server, "A_BUTTON", ROTATE_HOLD, fd)
        wait_frames(core, TAP_RELEASE, fd)
        fast_drop(core, input_server, fd)

        frame = core.read_screen()
        board_after = read_board_clean(frame, threshold=65)
        diff_cells = get_diff_cells(settled, board_after)

        pt, rot = identify_piece(diff_cells)
        if diff_cells:
            cols = sorted(set(c for r, c in diff_cells))
            min_col = min(c for r, c in diff_cells)
        else:
            cols, min_col = [], -1

        rot_results[piece_idx] = (pt, rot, min_col, cols)

        # Get the spawn type from phase 1
        spawn_type = spawn_results[piece_idx][0]
        print(f"\n  Piece #{piece_idx}: spawn_type={spawn_type.name if spawn_type else '?'} "
              f"after 1xCW: type={pt.name if pt else '?'} rot={rot} "
              f"min_col={min_col} cols={cols}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Aggregate spawn cols by piece type
    type_spawn_cols = {}  # PieceType -> list of min_cols
    for idx, (pt, rot, min_col, cols, rows) in spawn_results.items():
        if pt is not None and min_col >= 0:
            type_spawn_cols.setdefault(pt, []).append(min_col)

    print("\nSpawn columns (no rotation, no movement):")
    for pt in PieceType:
        if pt in type_spawn_cols:
            observations = type_spawn_cols[pt]
            print(f"  {pt.name}: min_col = {observations}  (consistent={len(set(observations)) == 1})")
        else:
            print(f"  {pt.name}: not observed")

    print("\nRotation test (1x CW):")
    for idx, (pt, rot, min_col, cols) in rot_results.items():
        spawn_pt = spawn_results[idx][0]
        spawn_min = spawn_results[idx][2]
        print(f"  Piece #{idx} ({spawn_pt.name if spawn_pt else '?'}): "
              f"spawn_min_col={spawn_min} → after 1xCW: min_col={min_col} (delta={min_col - spawn_min if spawn_min >= 0 else '?'})")

    # Compute what our model predicts vs actual
    print("\nModel validation (estimated_col_after_rotation vs actual):")
    for idx, (pt_rot, rot_state, rot_min_col, rot_cols) in rot_results.items():
        spawn_pt = spawn_results[idx][0]
        spawn_min = spawn_results[idx][2]
        if spawn_pt is None or spawn_min < 0 or rot_min_col < 0:
            continue

        # Our center-preserving model prediction
        cells_r0 = get_cells(spawn_pt, 0)
        cells_r1 = get_cells(spawn_pt, 1)
        center0 = (min(c for _, c in cells_r0) + max(c for _, c in cells_r0)) / 2.0
        center1 = (min(c for _, c in cells_r1) + max(c for _, c in cells_r1)) / 2.0
        predicted = round(spawn_min + center0 - center1)

        match = "OK" if predicted == rot_min_col else f"WRONG (off by {rot_min_col - predicted})"
        print(f"  {spawn_pt.name}: spawn={spawn_min}, predicted_after_rot={predicted}, "
              f"actual={rot_min_col} → {match}")

    core.resume()
    input_server.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
