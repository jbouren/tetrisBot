#!/usr/bin/env python3
"""
Tests the game state parser by loading a save state and printing the parsed state.

Includes a "warm-up" phase to synchronize emulator state with the discovery scripts,
which is necessary to resolve memory discrepancies (heisenbugs).
"""
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import InputServer, ControllerState
from src.emulator.memory import MemoryReader
from src.game.state import GameState

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
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(pause_sec)

def main():
    log("=" * 60)
    log("Game State Parser Test")
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
    time.sleep(2)
    memory.refresh_pointer()
    log(">>> RDRAM pointer acquired\\n")

    # --- Warm-up Phase ---
    # The value at the piece address is different without this.
    # We must perform at least one load/action/load cycle to match discovery.
    log(">>> Performing warm-up cycle to stabilize memory...")
    core.load_state(slot=1)
    time.sleep(0.5)
    core.pause()
    core.resume()
    time.sleep(0.5) # Let piece fall briefly
    core.pause()
    log(">>> Warm-up complete.")

    log("\\n>>> Loading final save state for parsing...")
    core.load_state(slot=1)
    time.sleep(1)
    core.pause()
    log(">>> Save state loaded.")

    log("\\n>>> Reading game state from memory...")
    game_state = GameState.from_memory(memory)

    log("\\n" + "=" * 60)
    log("PARSED GAME STATE")
    log("=" * 60)

    log(f"\\nSummary: {game_state.summary()}")
    log("\\nBoard:")
    log(game_state.board.to_ascii())

    log("\\n" + "=" * 60)

    log("\\n>>> Shutting down...")
    try:
        core.stop()
    except Exception:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
