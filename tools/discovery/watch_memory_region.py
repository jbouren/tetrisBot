#!/usr/bin/env python3
"""Continuously watch a region of memory while the game is running."""

import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
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
DATA_DIR = LIB_DIR / "lib/data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"

def _press_button(input_server: InputServer, button: str, hold_sec=0.05, pause_sec=0.5):
    """Press a button in real-time."""
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(pause_sec)

def _navigate_menu(core: Mupen64PlusCore, input_server: InputServer):
    """Send inputs to get from title screen to gameplay."""
    log(">>> Navigating menu...")
    core.resume()
    time.sleep(1)

    _press_button(input_server, "START_BUTTON", hold_sec=0.1, pause_sec=3.5)
    _press_button(input_server, "START_BUTTON", hold_sec=0.1, pause_sec=3.5)
    _press_button(input_server, "A_BUTTON", hold_sec=0.5, pause_sec=3.0)
    _press_button(input_server, "A_BUTTON", hold_sec=0.5, pause_sec=3.0)
    _press_button(input_server, "A_BUTTON", hold_sec=0.5, pause_sec=2.5)
    _press_button(input_server, "A_BUTTON", hold_sec=0.5, pause_sec=2.5)
    _press_button(input_server, "A_BUTTON", hold_sec=0.5, pause_sec=2.5)
    _press_button(input_server, "D_DPAD", hold_sec=0.1, pause_sec=2.0)
    _press_button(input_server, "R_DPAD", hold_sec=0.1, pause_sec=2.0)
    _press_button(input_server, "D_DPAD", hold_sec=0.1, pause_sec=2.0)
    _press_button(input_server, "A_BUTTON", hold_sec=0.1, pause_sec=2.0)

    log(">>> Waiting for game to start after menu navigation...")
    time.sleep(10)
    log(">>> Menu navigation complete.")
    core.pause() # Pause for controlled watching

def main():
    log("=" * 60)
    log("Watch Memory Region")
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
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    log("\\n>>> Starting emulation...")
    core.execute()
    log(">>> Loading save state...")
    core.load_state(slot=1)
    time.sleep(0.5)
    memory.refresh_pointer()


    # --- Watch a memory region in a loop ---
    log("\\n>>> Starting memory watch loop...")
    core.resume() # Let the game run freely
    input_server.clear()

    watch_addrs = {
        "X_CAND": 0x8010323E,
        "Y_CAND_A": 0x8010BE1C,
        "Y_CAND_B": 0x8011EF36,
        "Y_CAND_C": 0x80120A81,
        "Y_CAND_D": 0x800D3910,
    }

    try:
        while True:
            for name, addr in watch_addrs.items():
                u8 = memory.read_u8(addr)
                s8 = memory.read_s8(addr)
                u16 = memory.read_u16(addr)
                s16 = memory.read_s16(addr)
                u16_le = memory.read_u16_le(addr)
                s16_le = memory.read_s16_le(addr)
                log(
                    f"{name} (0x{addr:08X}): "
                    f"u8={u8:<3} s8={s8:<4} | "
                    f"u16={u16:<5} s16={s16:<6} | "
                    f"u16_le={u16_le:<5} s16_le={s16_le:<6}"
                )



            log("-" * 60)
            time.sleep(0.5)

    except KeyboardInterrupt:
        log("\\nUser interrupted.")


    log("\\n>>> Shutting down...")
    core.stop()
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
