#!/usr/bin/env python3
"""
Validates a list of candidate memory addresses to find the one for the current piece.

Strategy:
1. Start with a large list of candidate addresses from discover_piece_id.py.
2. Load the initial save state (first piece is about to drop).
3. Take a snapshot of memory (snap_before).
4. Hard drop the piece and wait for the next piece to appear.
5. Take another snapshot (snap_after).
6. For each candidate address, compare its value in snap_before and snap_after.
7. The correct address should show a value change corresponding to the piece type changing.
   Addresses where the value doesn't change are eliminated.
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
from src.emulator.memory import MemoryReader

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

# This list is from the output of discover_piece_id.py
CANDIDATE_ADDRESSES = [
    0x8000BC5D, 0x8001719D, 0x800190DF, 0x8001F1BC, 0x80023FDD, 0x8002781D,
    0x8002911D, 0x8003AF5E, 0x8003AF80, 0x8003AFAC, 0x8003B1DC, 0x8003B1EE,
    0x8003B24C, 0x8003B2BE, 0x8003B2FE, 0x8003B306, 0x8003B38A, 0x8003B3A8,
    0x8003B484, 0x8003B4F0, 0x8003B4F4, 0x8003B568, 0x8003B6AA, 0x8003BC62,
    0x8003C248, 0x8003C2DE, 0x8003C4CA, 0x8003C576, 0x8003C746, 0x8003C7CA,
    0x8003C86A, 0x8003C8BC, 0x8003C8FE, 0x8003C91A, 0x8003C978, 0x8003CA34,
    0x8003CAF6, 0x8003CAFC, 0x8003CB5A, 0x8003CBBC, 0x8003CBBE, 0x8003CBC2,
    0x8003CC42, 0x8003CC74, 0x8003CCAE, 0x8003CCB2, 0x8003CCBA, 0x8003CD30,
    0x8003CD6A, 0x8003CDD8,
    # This is just a sample, the real run had 33k. Let's start with these.
]


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
    log("Piece ID Candidate Validator")
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
    log(">>> Save state loaded (initial piece: T).")

    log("\\n>>> Taking baseline snapshot (before drop)...")
    snap_before = memory.snapshot()

    log("\\n>>> Hard dropping the piece and waiting for next piece...")
    core.resume()
    # Hard drop (Up) + ARE delay to allow next piece to spawn
    press_button(input_server, "U_DPAD", hold_sec=0.2, pause_sec=1.5)
    core.pause()

    log("\\n>>> Taking post-drop snapshot...")
    snap_after = memory.snapshot()

    log("\\n" + "=" * 60)
    log("ANALYSIS")
    log("=" * 60)

    survivors = []
    for addr in CANDIDATE_ADDRESSES:
        offset = addr & 0x1FFFFFFF
        val_before = snap_before[offset]
        val_after = snap_after[offset]

        if val_before != val_after:
            # We expect the initial piece to be color 15
            if val_before == 15:
                log(f"  SURVIVOR: 0x{addr:08X} changed from {val_before} -> {val_after}")
                survivors.append(addr)
            else:
                log(f"  - 0x{addr:08X} changed, but initial value was not 15 ({val_before})")
        else:
            # Uncomment for very verbose logging
            # log(f"  - 0x{addr:08X} did not change ({val_before})")
            pass


    log(f"\\nFound {len(survivors)} surviving candidate(s):")
    for addr in survivors:
        log(f"  - 0x{addr:08X}")


    log("\\n>>> Shutting down...")
    try:
        core.stop()
    except Exception:
        pass
    input_server.stop()
    log("Done.")

if __name__ == "__main__":
    main()
