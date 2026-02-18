#!/usr/bin/env python3
"""Discovery v15 — Confirm X at 0x801267E4 + smarter board search.

CONFIRMED from v14:
- 0x801267E4: consistent delta=2 per right press (X * 2?)
- Board is NOT a zero-initialized array

Strategy:
1. Verify 0x801267E4: range test (full left to full right)
2. Board: Look for ANY u32/u8 word changes (not just 0->nonzero) in a small diff window
3. Board: Follow piece placement — what EXACT memory locations change when a piece locks?
"""

import logging
import os
import struct
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.emulator.memory import MemoryReader, RDRAM_SIZE

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


def reload_save(core):
    core.load_state(slot=1)
    time.sleep(0.5)
    core.pause()
    time.sleep(0.1)


def read_word(core, addr):
    return core.debug_read_32(addr)


def read_float(core, addr):
    raw = core.debug_read_32(addr)
    return struct.unpack('f', struct.pack('I', raw))[0]


def set_button(input_server, button):
    state = ControllerState()
    if button:
        setattr(state, button, 1)
    input_server.set_state(state)


def advance_n(core, n):
    for _ in range(n):
        core.advance_frame()
        time.sleep(0.01)


def press_frame(input_server, core, button, hold=1, gap=1):
    """Press button for hold frames, then release for gap frames."""
    set_button(input_server, button)
    advance_n(core, hold)
    set_button(input_server, None)
    input_server.clear()
    advance_n(core, gap)


def main():
    log("=" * 60)
    log("Discovery v15 — Confirm X + Board Search")
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

    log("\n>>> Starting emulation...")
    core.execute()
    time.sleep(2)
    memory.refresh_pointer()

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 1: Verify X at 0x801267E4 — full range test
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Verify X position at 0x801267E4")
    log("=" * 60)

    X_ADDR = 0x801267E4

    reload_save(core)
    x_init = read_word(core, X_ADDR)
    y_init = read_float(core, 0x800D02CC)
    log(f"\n>>> Initial: X_raw={x_init} Y={y_init:.2f}")

    # Move left as far as possible (10 presses)
    log("\n>>> Moving LEFT to boundary:")
    for i in range(10):
        press_frame(input_server, core, "L_DPAD", hold=1, gap=1)
        x = read_word(core, X_ADDR)
        y = read_float(core, 0x800D02CC)
        log(f"  Left {i+1}: X_raw={x} Y={y:.2f}")

    # Now move right all the way across
    log("\n>>> Moving RIGHT across full board:")
    reload_save(core)
    # First move all the way left
    for _ in range(10):
        press_frame(input_server, core, "L_DPAD", hold=1, gap=1)

    x_leftmost = read_word(core, X_ADDR)
    log(f"  At left wall: X_raw={x_leftmost}")

    all_x = [x_leftmost]
    for i in range(15):
        press_frame(input_server, core, "R_DPAD", hold=1, gap=1)
        x = read_word(core, X_ADDR)
        y = read_float(core, 0x800D02CC)
        all_x.append(x)
        log(f"  Right {i+1}: X_raw={x} Y={y:.2f}")

    log(f"\n>>> X values across board: {all_x}")
    deltas = [all_x[i+1] - all_x[i] for i in range(len(all_x)-1)]
    log(f">>> Deltas: {deltas}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Isolate piece lock moment
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Capture the exact moment piece locks into board")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()

    # Take snapshot before hard drop
    snap_pre = memory.snapshot()
    x_pre = read_word(core, X_ADDR)
    y_pre = read_float(core, 0x800D02CC)
    log(f"\n>>> Pre-drop: X={x_pre} Y={y_pre:.2f}")

    # Hard drop and advance enough frames for piece to lock
    press_frame(input_server, core, "U_DPAD", hold=2, gap=0)
    # Now advance frame-by-frame and monitor when piece locks
    for frame in range(90):
        core.advance_frame()
        time.sleep(0.01)
        y = read_float(core, 0x800D02CC)
        if frame % 10 == 0 or (frame > 0 and abs(y - y_pre) > 3):
            log(f"  Frame {frame}: Y={y:.2f}")
        y_pre = y

    snap_post = memory.snapshot()

    # Now find ALL u32 words that changed (not just 0->nonzero)
    SCAN_START = 0x080000
    SCAN_END = 0x200000
    all_word_changes = []
    for phys in range(SCAN_START, SCAN_END, 4):
        wa = struct.unpack_from('<I', snap_pre, phys)[0]
        wb = struct.unpack_from('<I', snap_post, phys)[0]
        if wa != wb:
            addr = 0x80000000 + phys
            all_word_changes.append((addr, wa, wb))

    log(f"\n>>> Total u32 words changed: {len(all_word_changes)}")

    # Filter for interesting changes: small values, specific patterns
    # Look for words where the LOW byte changed to a small value (1-8)
    # while the rest stayed similar (board cell update)
    cell_like_changes = []
    for addr, old, new in all_word_changes:
        # Changed word where new value has a small component
        if new != 0 and old != new:
            new_byte0 = new & 0xFF
            new_byte3 = (new >> 24) & 0xFF
            old_byte0 = old & 0xFF
            old_byte3 = (old >> 24) & 0xFF
            # Check if a single byte within the word changed to a small value
            if (1 <= new_byte0 <= 8 and old_byte0 == 0 and (old >> 8) == (new >> 8)):
                cell_like_changes.append((addr, old, new, "byte0"))
            elif (1 <= new_byte3 <= 8 and old_byte3 == 0 and (old & 0x00FFFFFF) == (new & 0x00FFFFFF)):
                cell_like_changes.append((addr, old, new, "byte3"))

    log(f"\n>>> Cell-like changes (single byte 0->[1-8] within word): {len(cell_like_changes)}")
    for addr, old, new, which in cell_like_changes[:20]:
        log(f"  0x{addr:08X}: 0x{old:08X} -> 0x{new:08X} ({which})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Narrow board search — diff 1-piece vs 2-piece states
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: 1-piece vs 2-piece diff (narrowed)")
    log("=" * 60)

    # Drop first piece with real-time (reliable)
    reload_save(core)
    memory.refresh_pointer()
    core.resume()
    time.sleep(0.05)
    state = ControllerState()
    state.U_DPAD = 1
    input_server.set_state(state)
    time.sleep(0.05)
    input_server.clear()
    time.sleep(2.0)  # Wait longer for full settle
    core.pause()
    time.sleep(0.2)
    snap_1piece = memory.snapshot()
    log(f">>> 1 piece dropped, score: {core.debug_read_16(0x8011EED6)}")

    # Drop second piece
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
    snap_2piece = memory.snapshot()
    log(f">>> 2 pieces dropped, score: {core.debug_read_16(0x8011EED6)}")

    # Diff between 1-piece and 2-piece snapshots
    # Focus on NON-RENDERING regions (exclude 0x80130000+ which is display lists)
    word_diff = []
    for phys in range(SCAN_START, min(0x130000, SCAN_END), 4):
        wa = struct.unpack_from('<I', snap_1piece, phys)[0]
        wb = struct.unpack_from('<I', snap_2piece, phys)[0]
        if wa != wb:
            addr = 0x80000000 + phys
            word_diff.append((addr, wa, wb))

    log(f"\n>>> u32 words changed between 1 and 2 pieces (below 0x80130000): {len(word_diff)}")
    # Group by region
    regions = {}
    for addr, old, new in word_diff:
        region = (addr >> 12) << 12  # Group by 4KB pages
        if region not in regions:
            regions[region] = []
        regions[region].append((addr, old, new))

    for region in sorted(regions.keys()):
        entries = regions[region]
        if len(entries) <= 20:  # Only show small-change regions
            log(f"\n  Region 0x{region:08X} ({len(entries)} changes):")
            for addr, old, new in entries:
                log(f"    0x{addr:08X}: {old} -> {new}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Inspect 0x801267E0-0x80126A00 neighborhood
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: X position neighborhood (0x801267C0-0x80126A00)")
    log("=" * 60)

    reload_save(core)

    # Dump the full area
    log("\n>>> Initial state:")
    for addr in range(0x801267C0, 0x80126A00, 4):
        val = read_word(core, addr)
        if val != 0 and val != 0xFFFFFFFF:
            log(f"  0x{addr:08X}: {val} (0x{val:08X})")

    # After 1 piece drop
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

    log("\n>>> After 1 drop:")
    for addr in range(0x801267C0, 0x80126A00, 4):
        val = read_word(core, addr)
        if val != 0 and val != 0xFFFFFFFF:
            log(f"  0x{addr:08X}: {val} (0x{val:08X})")

    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v15 COMPLETE")
    log("=" * 60)

    try:
        core.stop()
    except:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
