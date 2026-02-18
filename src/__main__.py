"""Entry point: python -m src

Runs the Tetris bot with default configuration.
Adjust paths below to match your setup_env.sh output.
"""

import logging
import os
import sys
from pathlib import Path

from .bot import TetrisBot

# WSL2/WSLg: Use d3d12 GPU passthrough for OpenGL instead of llvmpipe software rendering
if "GALLIUM_DRIVER" not in os.environ:
    os.environ["GALLIUM_DRIVER"] = "d3d12"

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"

# Default configuration
CONFIG = {
    # Core library (built with DEBUGGER=1)
    "core_lib_path": str(LIB_DIR / "libmupen64plus.so.2.0.0"),
    "plugin_dir": str(LIB_DIR),
    "data_dir": str(DATA_DIR),
    # ROM
    "rom_path": "/mnt/c/code/n64/roms/New Tetris, The (USA).z64",
    # Plugins
    "gfx_plugin": str(LIB_DIR / "mupen64plus-video-GLideN64.so"),
    "audio_plugin": None,  # Disabled for performance
    "input_plugin": str(LIB_DIR / "mupen64plus-input-sdl.so"),
    "rsp_plugin": str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    # Input server (for mupen64plus-input-bot plugin)
    "input_host": "127.0.0.1",
    "input_port": 8082,
    # If True, use DebugMemRead* functions instead of direct RDRAM pointer.
    # Slower but guaranteed correct byte ordering.
    "use_debug_api": False,
    # Seconds to wait for game to boot before starting bot
    "boot_wait_seconds": 5,
}


def main():
    # Configure logging
    log_level = logging.DEBUG if "--debug" in sys.argv else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Check that core library exists
    core_path = Path(CONFIG["core_lib_path"])
    if not core_path.exists():
        print(f"ERROR: Core library not found at {core_path}")
        print("Run ./setup_env.sh first to build mupen64plus from source.")
        sys.exit(1)

    # Check that ROM exists
    rom_path = Path(CONFIG["rom_path"])
    if not rom_path.exists():
        print(f"ERROR: ROM not found at {rom_path}")
        sys.exit(1)

    # Override input plugin if mupen64plus-input-bot is available
    input_bot_path = LIB_DIR / "mupen64plus-input-bot.so"
    if input_bot_path.exists():
        CONFIG["input_plugin"] = str(input_bot_path)
        print(f"Using input-bot plugin: {input_bot_path}")
    else:
        print("WARNING: mupen64plus-input-bot.so not found.")
        print("Using mupen64plus-input-sdl.so with xdotool fallback.")

    bot = TetrisBot(CONFIG)
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
