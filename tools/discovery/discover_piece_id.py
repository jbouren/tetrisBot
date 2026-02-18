#!/usr/bin/env python3
"""
Discovers piece type address by dropping one piece and analyzing the board.

Strategy:
1. Load the initial save state (empty board).
2. Take a snapshot of RDRAM (snap_before).
3. Hard drop one piece. The piece is now written to the board memory region.
4. Take another snapshot (snap_after).
5. Read the block of memory for the board from snap_after.
6. Find the non-zero values in the board data. This is the piece's color ID.
7. Search for that color ID value in the *entire* snap_before.
8. The address where it's found is a strong candidate for the current/next piece address.
"""
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import InputServer
from src.emulator.memory import MemoryReader, RDRAM_SIZE
from src.game import memory_map as mm

if "GALLIUM_DRIVER" not in os.environ:
    os.environ["GALLIUM_DRIVER"] = "d3d12"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def log(msg=""):
    print(msg, flush=True)

LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"

def press_button(input_server, button, hold_sec=0.15, pause_sec=0.5):
    """Press button in real-time."""
    from src.emulator.input_server import ControllerState
    log(f"  ... Pressing {button}")
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(pause_sec)

def main():
    log("=" * 60)
    log("Piece ID Address Discovery")
    log("=" * 60)

    input_server = InputServer(host="127.0.0.1", port=8082)
    input_server.start()

    core = Mupen64PlusCore(
        core_lib_path=str(LIB_DIR / "libmupen64plus.so.2.0.0"),
        plugin_dir=str(LIB_DIR),
        data_dir=str(DATA_DIR),
    )
    memory = MemoryReader(core, use_debug_api=False)

    core.startup()
    core.load_rom(ROM_PATH)
    core.attach_plugins(
        gfx=str(LIB_DIR / "mupen64plus-video-GLideN64.so"),
        audio=None,
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    log("\\n>>> Starting emulation...")
    core.execute()
    time.sleep(1)
    memory.refresh_pointer()
    log(">>> RDRAM pointer acquired\\n")

    log(">>> Loading save state from slot 1...")
    core.load_state(slot=1)
    time.sleep(1)
    core.pause()
    log(">>> Save state loaded.")

    log("\\n>>> Taking baseline snapshot (before drop)...")
    snap_before = memory.snapshot()

    log("\\n>>> Hard dropping the first piece...")
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.2, pause_sec=1.5) # Hard drop + ARE
    core.pause()

    log("\\n>>> Taking post-drop snapshot...")
    snap_after = memory.snapshot()

    log("\\n" + "=" * 60)
    log("ANALYSIS")
    log("=" * 60)

    board_offset = mm.ADDR_BOARD_BASE & 0x1FFFFFFF
    board_bytes = mm.BOARD_WIDTH * mm.BOARD_HEIGHT * mm.BOARD_CELL_SIZE
    board_data_after = snap_after[board_offset : board_offset + board_bytes]

    board_colors = set()
    for i in range(0, len(board_data_after), 2):
        cell_val = (board_data_after[i] << 8) | board_data_after[i+1]
        if cell_val != 0:
            # Simplistic assumption: color is in the high bits
            color = (cell_val >> 12) & 0xF
            if color != 0:
                board_colors.add(color)

    if not board_colors:
        log("!!! ERROR: No piece data found on the board after drop.")
        log("    This could mean the hard drop failed or parsing is wrong.")
    else:
        log(f"\\nFound piece color(s) on board: {board_colors}")
        for color_id in board_colors:
            log(f"\\n>>> Searching for value {color_id} in pre-drop memory...")
            candidates = []
            for i in range(RDRAM_SIZE):
                if snap_before[i] == color_id:
                    addr = 0x80000000 + i
                    # Exclude the board itself from the search
                    if not (mm.ADDR_BOARD_BASE <= addr < mm.ADDR_BOARD_BASE + board_bytes):
                        candidates.append(addr)

            log(f"  Found {len(candidates)} candidate addresses for piece color {color_id}:")
            for addr in candidates[:50]: # Limit output
                log(f"    - 0x{addr:08X}")


    log("\\n>>> Shutting down...")
    try:
        core.stop()
    except Exception:
        pass
    input_server.stop()
    log("Done.")

if __name__ == "__main__":
    main()
