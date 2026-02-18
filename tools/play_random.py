#!/usr/bin/env python3
"""Play The New Tetris with random moves via CV + emulator control.

Proves the full pipeline end-to-end:
  Load save state -> read preview via CV -> pick random move ->
  inject inputs in real-time -> wait for settle -> repeat until game over.

Uses real-time input mode (time.sleep based), matching the proven pattern
from navigate_and_save.py. Only pauses briefly for screenshots.

Usage:
    DISPLAY=:0 GALLIUM_DRIVER=d3d12 .venv/bin/python3 tools/play_random.py
"""

import logging
import os
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.pieces import PieceType, ROTATION_COUNT, get_cells, get_width
from src.game.vision import read_board, read_preview, board_to_ascii

if "GALLIUM_DRIVER" not in os.environ:
    os.environ["GALLIUM_DRIVER"] = "d3d12"

logging.getLogger("src.emulator.core").setLevel(logging.WARNING)
logger = logging.getLogger("play_random")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

# --- Config ---
LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"

# In TNT, pieces spawn at approximately column 3, rotation 0
SPAWN_X = 3

# Real-time timing (seconds)
BUTTON_HOLD = 0.05   # how long to hold a button press
BUTTON_GAP = 0.10    # gap between consecutive presses
SETTLE_WAIT = 0.5    # wait after hard drop for piece to lock + new piece spawn

# Max pieces before we stop (safety limit)
MAX_PIECES = 200

# Game over detection
GAME_OVER_TOP_ROWS = 2
GAME_OVER_THRESHOLD = 5
MAX_IDENTICAL_BOARDS = 5


def press_button(input_server: InputServer, button: str,
                 hold: float = BUTTON_HOLD, gap: float = BUTTON_GAP):
    """Press a button in real-time (emulator must be running)."""
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold)
    input_server.clear()
    time.sleep(gap)


def capture_and_read(core: Mupen64PlusCore):
    """Pause emulator, take screenshot, read board + preview, resume.

    Returns (board, preview, frame).
    """
    core.pause()
    time.sleep(0.1)
    frame = core.read_screen()
    board = read_board(frame)
    preview = read_preview(frame)
    core.resume()
    time.sleep(0.1)
    return board, preview, frame


def detect_game_over(board: np.ndarray, prev_boards: list[np.ndarray]) -> str | None:
    """Check for game over conditions. Returns reason string or None."""
    top_occupied = int(board[:GAME_OVER_TOP_ROWS].sum())
    if top_occupied >= GAME_OVER_THRESHOLD:
        return f"topped_out ({top_occupied} cells in top {GAME_OVER_TOP_ROWS} rows)"

    if len(prev_boards) >= MAX_IDENTICAL_BOARDS:
        all_same = all(np.array_equal(board, pb) for pb in prev_boards[-MAX_IDENTICAL_BOARDS:])
        if all_same:
            return "stuck (identical boards)"

    return None


def main():
    logger.info("=== Random Tetris Player ===")

    # Start input server
    input_server = InputServer(port=8082)
    input_server.start()
    logger.info("Input server started")

    # Initialize emulator (following navigate_and_save.py pattern)
    core = Mupen64PlusCore(
        core_lib_path=str(LIB_DIR / "libmupen64plus.so.2.0.0"),
        plugin_dir=str(LIB_DIR),
        data_dir=str(DATA_DIR),
    )
    core.startup()
    core.load_rom(ROM_PATH)
    core.attach_plugins(
        gfx=str(LIB_DIR / "mupen64plus-video-GLideN64.so"),
        audio=None,  # no audio needed
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )
    logger.info("Emulator initialized")

    # Start emulation in real-time
    core.execute()
    logger.info("Emulation started, waiting for boot...")
    time.sleep(5)

    # Load save state (slot 1 = at game start, after countdown)
    core.pause()
    time.sleep(0.2)
    core.load_state(slot=1)
    logger.info("Save state loaded (slot 1)")
    core.resume()
    time.sleep(1.0)  # let it render a couple frames

    # Read initial state
    board, preview, _ = capture_and_read(core)
    logger.info("Initial board:\n%s", board_to_ascii(board))
    logger.info("Initial preview: %s", [p.name if p else "?" for p in preview])

    # Track state
    pieces_placed = 0
    prev_boards: list[np.ndarray] = []
    prev_preview_0: PieceType | None = None

    try:
        while pieces_placed < MAX_PIECES:
            # --- Read current state ---
            board, preview, _ = capture_and_read(core)

            # Game over check
            reason = detect_game_over(board, prev_boards)
            if reason:
                logger.info("GAME OVER: %s after %d pieces", reason, pieces_placed)
                break

            if all(p is None for p in preview):
                logger.info("GAME OVER: preview unreadable after %d pieces", pieces_placed)
                break

            prev_boards.append(board.copy())
            if len(prev_boards) > MAX_IDENTICAL_BOARDS + 1:
                prev_boards.pop(0)

            # --- Determine current piece ---
            if prev_preview_0 is not None:
                current_piece = prev_preview_0
            else:
                current_piece = random.choice(list(PieceType))

            prev_preview_0 = preview[0]

            # --- Pick random move ---
            num_rotations = ROTATION_COUNT[current_piece]
            rotation = random.randint(0, num_rotations - 1)
            piece_width = get_width(current_piece, rotation)
            max_col = 10 - piece_width
            target_col = random.randint(0, max_col)

            logger.info(
                "Piece #%d: %s rot=%d col=%d (preview: %s)",
                pieces_placed + 1,
                current_piece.name,
                rotation,
                target_col,
                [p.name if p else "?" for p in preview],
            )

            # --- Execute inputs in real-time ---
            # 1. Rotate (A button for CW)
            for _ in range(rotation):
                press_button(input_server, "A_BUTTON")

            # 2. Move horizontally from spawn
            dx = target_col - SPAWN_X
            if dx > 0:
                for _ in range(dx):
                    press_button(input_server, "R_DPAD")
            elif dx < 0:
                for _ in range(-dx):
                    press_button(input_server, "L_DPAD")

            # 3. Hard drop (U_DPAD in TNT)
            press_button(input_server, "U_DPAD")

            # 4. Wait for piece to settle and new piece to spawn
            time.sleep(SETTLE_WAIT)

            pieces_placed += 1

            # Show board every 5 pieces
            if pieces_placed % 5 == 0:
                board, _, _ = capture_and_read(core)
                logger.info("Board after %d pieces:\n%s",
                            pieces_placed, board_to_ascii(board))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # Final board
    board, _, _ = capture_and_read(core)
    logger.info("Final board after %d pieces:\n%s", pieces_placed, board_to_ascii(board))

    # Clean shutdown (stop, not shutdown — avoids segfault)
    logger.info("Shutting down...")
    core.stop()
    input_server.stop()
    logger.info("Done!")


if __name__ == "__main__":
    main()
