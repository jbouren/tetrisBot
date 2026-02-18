#!/usr/bin/env python3
"""Run a longer game with random moves to find the J piece color.

Moves pieces left/right randomly before dropping to spread them out
and avoid topping out quickly.
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
from PIL import Image

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.game.pieces import PieceType, PIECE_NAMES
from src.game.vision import read_preview, sample_preview_slot, classify_color

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

# Known piece colors so far
KNOWN_COLORS = {
    "I": (40, 135, 190),
    "O": (146, 144, 142),
    "T": (186, 169, 24),
    "S": (65, 173, 87),
    "Z": (168, 62, 62),
    "L": (168, 58, 150),
}


def pname(p):
    return PIECE_NAMES[p] if p is not None else "???"


def is_known_color(rgb):
    """Check if this RGB is close to any known piece color."""
    for name, ref in KNOWN_COLORS.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(ref))
        if dist < 40:
            return name
    return None


def main():
    print("=== Find J Piece Color ===", flush=True)

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

    unknown_colors = []

    for drop_num in range(50):
        # Capture frame to read preview
        core.pause()
        time.sleep(0.1)
        frame = core.read_screen()
        preview = read_preview(frame)
        core.resume()

        # Check for unknown colors in preview
        for slot in range(3):
            rgb = sample_preview_slot(frame, slot)
            if rgb is not None:
                known = is_known_color(rgb)
                classified = classify_color(rgb)
                if known is None:
                    print(f"  *** DROP {drop_num} SLOT {slot}: UNKNOWN COLOR "
                          f"RGB=({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}) "
                          f"classified as {pname(classified)}", flush=True)
                    unknown_colors.append((drop_num, slot, rgb.copy()))
                    # Save this frame
                    img = Image.fromarray(frame, "RGB")
                    img.save(f"/mnt/c/code/tetrisBot/screenshots/j_candidate_{drop_num:02d}.png")

        if drop_num % 10 == 0:
            prev_str = f"[{pname(preview[0])}, {pname(preview[1])}, {pname(preview[2])}]"
            print(f"Drop {drop_num}: {prev_str}", flush=True)

        # Move piece randomly left or right, then hard drop
        time.sleep(0.05)
        state = ControllerState()

        # Random horizontal movement (spread pieces out)
        direction = random.choice(["left", "right", "none"])
        moves = random.randint(1, 4)
        for _ in range(moves):
            state = ControllerState()
            if direction == "left":
                state.L_DPAD = 1
            elif direction == "right":
                state.R_DPAD = 1
            input_server.set_state(state)
            time.sleep(0.05)
            input_server.clear()
            time.sleep(0.05)

        # Hard drop
        state = ControllerState()
        state.U_DPAD = 1
        input_server.set_state(state)
        time.sleep(0.05)
        input_server.clear()
        time.sleep(0.8)

    print(f"\n=== RESULTS ===")
    if unknown_colors:
        print(f"Found {len(unknown_colors)} unknown color observations:")
        for drop, slot, rgb in unknown_colors:
            print(f"  Drop {drop} slot {slot}: RGB=({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})")
        avg = np.mean([rgb for _, _, rgb in unknown_colors], axis=0)
        print(f"  Average: RGB=({avg[0]:.0f},{avg[1]:.0f},{avg[2]:.0f})")
        print(f"  This is likely J piece (purple)!")
    else:
        print("No unknown colors found. J may be very close to L (pink).")
        print("All preview colors matched known pieces.")

    print("\nDone!", flush=True)
    try:
        core.stop()
    except Exception:
        pass
    input_server.stop()


if __name__ == "__main__":
    main()
