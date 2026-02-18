#!/usr/bin/env python3
"""Capture frames from the emulator and save as PNG.

Tests read_screen() and benchmarks capture speed.
"""

import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from PIL import Image

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer

if "GALLIUM_DRIVER" not in os.environ:
    os.environ["GALLIUM_DRIVER"] = "d3d12"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"


def main():
    print("=== Screen Capture Test ===", flush=True)

    input_server = InputServer(host="127.0.0.1", port=8082)
    input_server.start()

    core = Mupen64PlusCore(
        core_lib_path=str(LIB_DIR / "libmupen64plus.so.2.0.0"),
        plugin_dir=str(LIB_DIR),
        data_dir=str(DATA_DIR),
    )

    core.startup()
    core.load_rom(ROM_PATH)
    core.attach_plugins(
        gfx=str(LIB_DIR / "mupen64plus-video-GLideN64.so"),
        audio=None,
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    print("Starting emulation...", flush=True)
    core.execute()
    time.sleep(2)

    print("Loading save state...", flush=True)
    core.load_state(slot=1)
    time.sleep(1)
    core.pause()
    time.sleep(0.2)

    # Capture initial frame
    print("\nCapturing initial frame (game start)...", flush=True)
    t0 = time.perf_counter()
    frame = core.read_screen()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Shape: {frame.shape}, Time: {elapsed:.0f}ms", flush=True)

    img = Image.fromarray(frame, "RGB")
    img.save("/tmp/tetris_game_start.png")
    print(f"  Saved /tmp/tetris_game_start.png", flush=True)

    # Drop a piece and capture
    print("\nDropping a piece...", flush=True)
    core.resume()
    time.sleep(0.05)
    state = ControllerState()
    state.U_DPAD = 1
    input_server.set_state(state)
    time.sleep(0.05)
    input_server.clear()
    time.sleep(2.0)
    core.pause()
    time.sleep(0.2)

    frame2 = core.read_screen()
    img2 = Image.fromarray(frame2, "RGB")
    img2.save("/tmp/tetris_after_drop.png")
    print(f"  Saved /tmp/tetris_after_drop.png", flush=True)

    # Benchmark
    print("\nBenchmarking 10 captures...", flush=True)
    times = []
    for i in range(10):
        core.advance_frame()  # advance so there's a new frame
        time.sleep(0.02)
        t0 = time.perf_counter()
        frame = core.read_screen()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    print(f"  Times: {[f'{t:.0f}ms' for t in times]}", flush=True)
    print(f"  Average: {np.mean(times):.0f}ms, Min: {np.min(times):.0f}ms, Max: {np.max(times):.0f}ms", flush=True)

    print("\nDone!", flush=True)
    try:
        core.stop()
    except:
        pass
    input_server.stop()


if __name__ == "__main__":
    main()
