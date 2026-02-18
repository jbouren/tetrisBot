#!/usr/bin/env python3
"""Discovery v14 — Surgical board search + immediate frame-precise moves.

Key learning: The board is NOT a simple byte array in the display records.
We need to search smarter.

Strategy:
1. Frame-precise moves IMMEDIATELY after save state (before piece falls)
2. Dump board struct region fully before/after drops
3. Search ENTIRE RDRAM for 200-byte regions where small # of cells flip 0->nonzero
4. Try u32-per-cell format (200 x 4 bytes = 800 bytes)
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


def main():
    log("=" * 60)
    log("Discovery v14 — Surgical Board Search")
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
    # EXPERIMENT 1: Immediate frame-precise moves
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Immediate moves after save state load")
    log("=" * 60)

    reload_save(core)

    # Read state immediately (no warmup frames)
    y0 = read_float(core, 0x800D02CC)
    bd64_0 = read_word(core, 0x8010BD64)
    bbec_0 = read_word(core, 0x8010BBEC)
    dffc_0 = read_word(core, 0x8010DFFC)
    log(f"\n>>> Immediately after save load: Y={y0:.2f} BD64={bd64_0} BBEC={bbec_0} DFFC={dffc_0}")

    # Do a single right press (1 frame on, 1 frame off) and read
    set_button(input_server, "R_DPAD")
    core.advance_frame()
    time.sleep(0.01)
    set_button(input_server, None)
    input_server.clear()
    core.advance_frame()
    time.sleep(0.01)

    y1 = read_float(core, 0x800D02CC)
    bd64_1 = read_word(core, 0x8010BD64)
    bbec_1 = read_word(core, 0x8010BBEC)
    log(f">>> After 1 RIGHT press (2 frames): Y={y1:.2f} BD64={bd64_1} BBEC={bbec_1}")

    # Another right
    set_button(input_server, "R_DPAD")
    core.advance_frame()
    time.sleep(0.01)
    set_button(input_server, None)
    input_server.clear()
    core.advance_frame()
    time.sleep(0.01)

    y2 = read_float(core, 0x800D02CC)
    bd64_2 = read_word(core, 0x8010BD64)
    log(f">>> After 2nd RIGHT press (4 frames): Y={y2:.2f} BD64={bd64_2}")

    # 3 more rights
    for i in range(3):
        set_button(input_server, "R_DPAD")
        core.advance_frame()
        time.sleep(0.01)
        set_button(input_server, None)
        input_server.clear()
        core.advance_frame()
        time.sleep(0.01)

    y5 = read_float(core, 0x800D02CC)
    bd64_5 = read_word(core, 0x8010BD64)
    log(f">>> After 5 RIGHT presses (10 frames): Y={y5:.2f} BD64={bd64_5}")

    # Now reload and try lefts
    reload_save(core)
    bd64_init = read_word(core, 0x8010BD64)
    log(f"\n>>> Reload for left: BD64={bd64_init}")

    for i in range(5):
        set_button(input_server, "L_DPAD")
        core.advance_frame()
        time.sleep(0.01)
        set_button(input_server, None)
        input_server.clear()
        core.advance_frame()
        time.sleep(0.01)

    y_left = read_float(core, 0x800D02CC)
    bd64_left = read_word(core, 0x8010BD64)
    log(f">>> After 5 LEFT presses: Y={y_left:.2f} BD64={bd64_left}")

    # Let's also try scanning ALL u32 in focused areas for consistent delta
    log("\n>>> Scanning 0x8010BB00-0x8010BE00 for address that changes by +1 per right press:")
    reload_save(core)

    scan_range = list(range(0x8010BB00, 0x8010BE00, 4))
    # Also include wider ranges
    scan_range += list(range(0x800D0280, 0x800D0340, 4))
    scan_range += list(range(0x80126700, 0x80126A00, 4))

    vals_before = {a: read_word(core, a) for a in scan_range}

    # Single right press
    set_button(input_server, "R_DPAD")
    core.advance_frame()
    time.sleep(0.01)
    set_button(input_server, None)
    input_server.clear()
    core.advance_frame()
    time.sleep(0.01)

    vals_after1 = {a: read_word(core, a) for a in scan_range}

    # Second right press
    set_button(input_server, "R_DPAD")
    core.advance_frame()
    time.sleep(0.01)
    set_button(input_server, None)
    input_server.clear()
    core.advance_frame()
    time.sleep(0.01)

    vals_after2 = {a: read_word(core, a) for a in scan_range}

    # Find addresses where delta1 == delta2 > 0
    for addr in scan_range:
        v0, v1, v2 = vals_before[addr], vals_after1[addr], vals_after2[addr]
        d1 = v1 - v0
        d2 = v2 - v1
        if d1 == d2 and d1 != 0 and abs(d1) < 100:
            log(f"  0x{addr:08X}: {v0} -> {v1} -> {v2} (consistent delta={d1})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Board struct full diff before/after drop
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Board struct region diff")
    log("=" * 60)

    reload_save(core)

    # Read full region 0x8012E220-0x8012F600 before drop
    region_start = 0x8012E220
    region_size = 0x13E0  # up to 0x8012F600
    before = [core.debug_read_8(region_start + i) for i in range(region_size)]

    # Hard drop via frame advance
    set_button(input_server, "U_DPAD")
    advance_n(core, 2)
    set_button(input_server, None)
    input_server.clear()
    # Wait for piece to land and settle
    advance_n(core, 60)

    after = [core.debug_read_8(region_start + i) for i in range(region_size)]

    changes = []
    for i in range(region_size):
        if before[i] != after[i]:
            addr = region_start + i
            changes.append((addr, before[i], after[i]))

    log(f"\n>>> Board struct region changes ({len(changes)} bytes changed):")
    for addr, old, new in changes[:50]:
        offset = addr - region_start
        log(f"  0x{addr:08X} (+0x{offset:04X}): {old} -> {new}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Search for small-value arrays after dropping 5 pieces
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Search for small-value arrays (board candidates)")
    log("=" * 60)

    # Drop 5 pieces total from save state using reliable real-time control
    reload_save(core)
    memory.refresh_pointer()
    snap_empty = memory.snapshot()

    # Drop 5 pieces
    for _ in range(5):
        core.resume()
        time.sleep(0.05)
        state = ControllerState()
        state.U_DPAD = 1
        input_server.set_state(state)
        time.sleep(0.05)
        input_server.clear()
        time.sleep(1.5)

    core.pause()
    time.sleep(0.1)
    snap_5pieces = memory.snapshot()

    SCAN_START = 0x080000
    SCAN_END = 0x200000

    # Strategy: Look for 32-bit words where value went from 0 to 1-10
    # Group them by proximity (within 1000 bytes)
    new_small_words = []
    for phys in range(SCAN_START, SCAN_END, 4):
        wa = struct.unpack_from('<I', snap_empty, phys)[0]
        wb = struct.unpack_from('<I', snap_5pieces, phys)[0]
        if wa == 0 and 1 <= wb <= 10:
            addr = 0x80000000 + phys
            new_small_words.append((addr, wb))

    log(f"\n>>> u32 words that went 0 -> [1-10] after 5 drops: {len(new_small_words)}")

    # Group into clusters (within 200 bytes of each other)
    if new_small_words:
        clusters = []
        current = [new_small_words[0]]
        for i in range(1, len(new_small_words)):
            if new_small_words[i][0] - current[-1][0] < 200:
                current.append(new_small_words[i])
            else:
                clusters.append(current)
                current = [new_small_words[i]]
        clusters.append(current)

        # Show clusters with 3+ entries (likely board-related)
        for cluster in clusters:
            if len(cluster) >= 3:
                log(f"\n  Cluster: {len(cluster)} words at 0x{cluster[0][0]:08X}-0x{cluster[-1][0]:08X}")
                for addr, val in cluster:
                    log(f"    0x{addr:08X}: {val}")

    # Also search for bytes that went 0 -> [1-8] and group them
    log("\n>>> Searching for byte-level board (0->[1-8] after 5 drops):")
    new_small_bytes = []
    for phys in range(SCAN_START, SCAN_END):
        if snap_empty[phys] == 0 and 1 <= snap_5pieces[phys] <= 8:
            virt = 0x80000000 + (phys ^ 3)
            new_small_bytes.append((virt, snap_5pieces[phys]))

    log(f"  Total: {len(new_small_bytes)} bytes")
    if new_small_bytes:
        # Group into clusters
        new_small_bytes.sort()
        clusters = []
        current = [new_small_bytes[0]]
        for i in range(1, len(new_small_bytes)):
            if new_small_bytes[i][0] - current[-1][0] < 500:
                current.append(new_small_bytes[i])
            else:
                clusters.append(current)
                current = [new_small_bytes[i]]
        clusters.append(current)

        for cluster in clusters:
            if 10 <= len(cluster) <= 50:  # 5 pieces * 4 blocks = 20 cells expected
                log(f"\n  Cluster: {len(cluster)} bytes at 0x{cluster[0][0]:08X}-0x{cluster[-1][0]:08X}")
                for addr, val in cluster:
                    log(f"    0x{addr:08X}: {val}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Check 0x80120E00-0x80121000 area
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Score/stats region 0x8011EE00-0x80121000")
    log("=" * 60)

    # This area is near the score (0x8011EED6). Maybe board stats are here.
    # Check what changed between empty and 5-piece states
    area_changes = []
    for phys in range(0x11EE00, 0x121000, 4):
        wa = struct.unpack_from('<I', snap_empty, phys)[0]
        wb = struct.unpack_from('<I', snap_5pieces, phys)[0]
        if wa != wb:
            addr = 0x80000000 + phys
            area_changes.append((addr, wa, wb))

    log(f"\n>>> Changed u32 words in 0x8011EE00-0x80121000: {len(area_changes)}")
    for addr, old, new in area_changes[:40]:
        log(f"  0x{addr:08X}: {old} -> {new}")

    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v14 COMPLETE")
    log("=" * 60)

    try:
        core.stop()
    except:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
