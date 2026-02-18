#!/usr/bin/env python3
"""Discovery v7 — Board grid and piece position.

Key insights from v5/v6:
- Piece values at 0x8010BBEC, 0x8010DFFC, 0x8010E0A4 are 32-bit words (0-7)
- DFFC = E0A4_prev + 1 (mod 8) — queue indices, not direct types
- Y position NOT found by byte search — likely stored as larger value (u16/u32)
  or as a float/fixed-point pixel position
- Board records at 0x800D10A9 (stride 0x9C) are display objects, not simple grid
- X candidates didn't increment cleanly

This script:
1. Searches for Y/X as u16 and u32 values (larger range)
2. Finds the actual board grid by dropping pieces at different columns
   and looking for a SMALL contiguous array (not scattered display objects)
3. Reads the piece struct area between drops to find next-piece queue
"""

import logging
import os
import struct
import sys
import time
from collections import Counter, defaultdict
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

SCAN_START = 0x080000
SCAN_END = 0x200000


def press_button(input_server, button, hold_sec=0.1, pause_sec=0.3):
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(pause_sec)


def reload_save(core):
    core.load_state(slot=1)
    time.sleep(0.5)
    core.pause()
    time.sleep(0.1)


def hexdump_debug(core, start_virt, count):
    data = [core.debug_read_8(start_virt + i) for i in range(count)]
    for i in range(0, len(data), 16):
        row = data[i:i + 16]
        addr = start_virt + i
        hex_str = " ".join(f"{b:02X}" for b in row)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        log(f"  {addr:08X}: {hex_str:<48} |{ascii_str}|")


def diff_words(snap_a, snap_b, start, end):
    """Diff 32-bit words (aligned). Returns {virt_addr: (old, new)}."""
    diffs = {}
    for phys in range(start, end, 4):
        wa = struct.unpack_from('<I', snap_a, phys)[0]
        wb = struct.unpack_from('<I', snap_b, phys)[0]
        if wa != wb:
            # For 32-bit words, phys offset = N64 virtual offset (no XOR needed)
            diffs[0x80000000 + phys] = (wa, wb)
    return diffs


def main():
    log("=" * 60)
    log("Discovery v7 — Board Grid & Piece Position")
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
    # EXPERIMENT 1: Find Y position as 32-bit word
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Y position via 32-bit word diffs")
    log("=" * 60)

    # Take multiple snapshots during piece fall
    reload_save(core)
    snap0 = memory.snapshot()

    core.resume()
    time.sleep(0.3)
    core.pause()
    time.sleep(0.1)
    snap1 = memory.snapshot()

    core.resume()
    time.sleep(0.3)
    core.pause()
    time.sleep(0.1)
    snap2 = memory.snapshot()

    core.resume()
    time.sleep(0.3)
    core.pause()
    time.sleep(0.1)
    snap3 = memory.snapshot()

    # Find 32-bit words that strictly increase across all intervals
    log("\n>>> Searching for increasing u32 values (possible Y position)...")
    for max_val in [100, 300, 1000, 10000]:
        increasing = []
        for phys in range(SCAN_START, SCAN_END, 4):
            w0 = struct.unpack_from('<I', snap0, phys)[0]
            w1 = struct.unpack_from('<I', snap1, phys)[0]
            w2 = struct.unpack_from('<I', snap2, phys)[0]
            w3 = struct.unpack_from('<I', snap3, phys)[0]
            if w0 < w1 < w2 < w3 and w3 < max_val:
                increasing.append((0x80000000 + phys, w0, w1, w2, w3))
        log(f"\n  Strictly increasing u32 (max={max_val}): {len(increasing)}")
        for addr, v0, v1, v2, v3 in increasing[:20]:
            dbg = core.debug_read_32(addr)
            log(f"    0x{addr:08X}: {v0} -> {v1} -> {v2} -> {v3}  [now={dbg}]")

    # Also try signed interpretation for values that could be negative
    log("\n>>> Searching for Y as float/fixed-point...")
    # Check if values could be IEEE float
    increasing_float = []
    for phys in range(SCAN_START, SCAN_END, 4):
        try:
            f0 = struct.unpack_from('<f', snap0, phys)[0]
            f1 = struct.unpack_from('<f', snap1, phys)[0]
            f2 = struct.unpack_from('<f', snap2, phys)[0]
            f3 = struct.unpack_from('<f', snap3, phys)[0]
            if 0 < f0 < f1 < f2 < f3 < 500 and f0 > 0.1:
                increasing_float.append((0x80000000 + phys, f0, f1, f2, f3))
        except (struct.error, ValueError):
            pass
    log(f"\n  Strictly increasing floats (0.1-500): {len(increasing_float)}")
    for addr, v0, v1, v2, v3 in increasing_float[:20]:
        raw = core.debug_read_32(addr)
        log(f"    0x{addr:08X}: {v0:.2f} -> {v1:.2f} -> {v2:.2f} -> {v3:.2f}  [raw=0x{raw:08X}]")

    # N64 uses big-endian floats! Try big-endian interpretation
    increasing_float_be = []
    for phys in range(SCAN_START, SCAN_END, 4):
        try:
            f0 = struct.unpack_from('>f', snap0, phys)[0]
            f1 = struct.unpack_from('>f', snap1, phys)[0]
            f2 = struct.unpack_from('>f', snap2, phys)[0]
            f3 = struct.unpack_from('>f', snap3, phys)[0]
            if 0 < f0 < f1 < f2 < f3 < 500 and f0 > 0.1:
                increasing_float_be.append((0x80000000 + phys, f0, f1, f2, f3))
        except (struct.error, ValueError):
            pass
    log(f"\n  Big-endian increasing floats (0.1-500): {len(increasing_float_be)}")
    for addr, v0, v1, v2, v3 in increasing_float_be[:20]:
        raw = core.debug_read_32(addr)
        log(f"    0x{addr:08X}: {v0:.2f} -> {v1:.2f} -> {v2:.2f} -> {v3:.2f}  [raw=0x{raw:08X}]")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: X position via 32-bit diffs (move right)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: X position via move right diffs")
    log("=" * 60)

    reload_save(core)
    snap_x0 = memory.snapshot()

    core.resume()
    time.sleep(0.05)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.1)
    core.pause()
    time.sleep(0.1)
    snap_x1 = memory.snapshot()

    # Find words that changed by a small consistent amount
    diffs_x = diff_words(snap_x0, snap_x1, SCAN_START, SCAN_END)
    log(f"\n  32-bit words that changed after 1 move right: {len(diffs_x)}")

    # Filter: values that increased or decreased by a small amount (1-20)
    x_candidates = []
    for addr in sorted(diffs_x.keys()):
        old, new = diffs_x[addr]
        delta = new - old  # Signed
        if -0x80000000 < delta < 0x80000000:  # Not a huge wrap
            signed_delta = delta if delta < 0x80000000 else delta - 0x100000000
            abs_d = abs(signed_delta)
            if 1 <= abs_d <= 20 and old < 1000 and new < 1000:
                x_candidates.append((addr, old, new, signed_delta))

    log(f"  Filtered (small delta, values < 1000): {len(x_candidates)}")
    for addr, old, new, delta in x_candidates[:30]:
        log(f"    0x{addr:08X}: {old} -> {new} (delta {delta:+d})")

    # Now move right 3 more times and track candidates
    if x_candidates:
        track_addrs = [a for a, _, _, _ in x_candidates[:20]]
        vals = {a: [core.debug_read_32(a)] for a in track_addrs}

        core.resume()
        time.sleep(0.05)
        for move in range(3):
            press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.1)
            core.pause()
            time.sleep(0.1)
            for a in track_addrs:
                vals[a].append(core.debug_read_32(a))
            if move < 2:
                core.resume()
                time.sleep(0.05)

        log("\n  X candidates across 4 right moves:")
        for a in track_addrs:
            vs = vals[a]
            deltas = [vs[i+1] - vs[i] for i in range(len(vs)-1)]
            consistent = all(d == deltas[0] for d in deltas) and deltas[0] != 0
            marker = " <-- CONSISTENT!" if consistent else ""
            log(f"    0x{a:08X}: {vs}{marker}")

    # Try floats for X too
    log("\n  Checking float values that change on right move...")
    float_x = []
    for phys in range(SCAN_START, SCAN_END, 4):
        f0 = struct.unpack_from('<f', snap_x0, phys)[0]
        f1 = struct.unpack_from('<f', snap_x1, phys)[0]
        if 0 < f0 < 500 and 0 < f1 < 500 and 0.1 < (f1 - f0) < 50:
            float_x.append((0x80000000 + phys, f0, f1))
    log(f"  LE floats with positive delta (0.1-50): {len(float_x)}")
    for addr, f0, f1 in float_x[:15]:
        log(f"    0x{addr:08X}: {f0:.2f} -> {f1:.2f} (delta {f1-f0:.2f})")

    float_x_be = []
    for phys in range(SCAN_START, SCAN_END, 4):
        f0 = struct.unpack_from('>f', snap_x0, phys)[0]
        f1 = struct.unpack_from('>f', snap_x1, phys)[0]
        if 0 < f0 < 500 and 0 < f1 < 500 and 0.1 < (f1 - f0) < 50:
            float_x_be.append((0x80000000 + phys, f0, f1))
    log(f"  BE floats with positive delta (0.1-50): {len(float_x_be)}")
    for addr, f0, f1 in float_x_be[:15]:
        log(f"    0x{addr:08X}: {f0:.2f} -> {f1:.2f} (delta {f1-f0:.2f})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Board grid — find by dropping at 3 positions
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Board grid via 32-bit word diffs")
    log("=" * 60)

    # Drop at center (default)
    reload_save(core)
    snap_bc = memory.snapshot()
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)
    snap_ac = memory.snapshot()
    diffs_center = diff_words(snap_bc, snap_ac, SCAN_START, SCAN_END)

    # Drop at far left
    reload_save(core)
    snap_bl = memory.snapshot()
    core.resume()
    time.sleep(0.1)
    for _ in range(5):
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.08)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)
    snap_al = memory.snapshot()
    diffs_left = diff_words(snap_bl, snap_al, SCAN_START, SCAN_END)

    # Drop at far right
    reload_save(core)
    snap_br = memory.snapshot()
    core.resume()
    time.sleep(0.1)
    for _ in range(5):
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.08)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)
    snap_ar = memory.snapshot()
    diffs_right = diff_words(snap_br, snap_ar, SCAN_START, SCAN_END)

    log(f"\n  Words changed - center: {len(diffs_center)}, left: {len(diffs_left)}, right: {len(diffs_right)}")

    # Find words that are 0->nonzero (potential board cells)
    # Focus on SMALL nonzero values (not pointers/render data)
    def filter_board_words(diffs, max_val=0xFF):
        """Find words that went from 0 to a small nonzero value."""
        result = {}
        for addr, (old, new) in diffs.items():
            if old == 0 and 0 < new <= max_val:
                result[addr] = new
        return result

    center_cells = filter_board_words(diffs_center)
    left_cells = filter_board_words(diffs_left)
    right_cells = filter_board_words(diffs_right)

    log(f"  0->small nonzero - center: {len(center_cells)}, left: {len(left_cells)}, right: {len(right_cells)}")

    # Addresses unique to each position
    c_addrs = set(center_cells.keys())
    l_addrs = set(left_cells.keys())
    r_addrs = set(right_cells.keys())

    c_only = c_addrs - l_addrs - r_addrs
    l_only = l_addrs - c_addrs - r_addrs
    r_only = r_addrs - c_addrs - l_addrs
    in_all = c_addrs & l_addrs & r_addrs

    log(f"\n  Unique to center: {len(c_only)}")
    log(f"  Unique to left: {len(l_only)}")
    log(f"  Unique to right: {len(r_only)}")
    log(f"  In all three: {len(in_all)}")

    # Board cells should be UNIQUE to each position (different piece lands at different columns)
    # Combine unique-to-position cells for stride analysis
    all_unique = sorted(c_only | l_only | r_only)
    log(f"\n  All position-unique cells: {len(all_unique)}")
    for addr in all_unique[:30]:
        in_c = "C" if addr in c_only else " "
        in_l = "L" if addr in l_only else " "
        in_r = "R" if addr in r_only else " "
        val = core.debug_read_32(addr)
        log(f"    0x{addr:08X}: [{in_l}{in_c}{in_r}] val={val}")

    if len(all_unique) >= 2:
        strides = [all_unique[i+1] - all_unique[i] for i in range(len(all_unique)-1)]
        stride_freq = Counter(strides)
        log(f"\n  Stride frequency:")
        for stride, count in stride_freq.most_common(10):
            log(f"    stride {stride} (0x{stride:X}): {count}x")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Board — try byte-level with 0x800D region
    # Look at entire 0x800D10A9-0x800D8B45 range (200 records × 0x9C)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Board record field analysis")
    log("=" * 60)

    RECORD_BASE = 0x800D10A9
    RECORD_STRIDE = 0x9C
    NUM_RECORDS = 200  # 10 cols × 20 rows

    # Read the first byte and byte+16 (the two that changed in v6) for all 200 records
    # across the 3 drops (center, left, right)
    reload_save(core)

    log("\n>>> First byte (offset+0) and flag byte (offset+16) for 200 records at save state:")
    records_save = []
    for r in range(NUM_RECORDS):
        addr = RECORD_BASE + r * RECORD_STRIDE
        b0 = core.debug_read_8(addr)
        b16 = core.debug_read_8(addr + 16)
        records_save.append((b0, b16))

    # Print in a 10×20 grid if possible (try both row-major and column-major)
    log("\n  If organized as 10 columns × 20 rows (column-major):")
    log("  Record | Col | Row | byte+0 | byte+16")
    for r in range(min(NUM_RECORDS, 50)):
        col = r // 20
        row = r % 20
        b0, b16 = records_save[r]
        marker = ""
        if b0 != 0 or b16 != 0:
            marker = f"  <-- b0={b0}, b16={b16}"
        if r < 5 or (r >= 20 and r < 25) or (r >= 40 and r < 45):
            log(f"  R{r:3d}   | C{col}  | R{row:2d}  | 0x{b0:02X}   | 0x{b16:02X}{marker}")

    # Check if byte+2 (X coord) and byte+4 (Y coord) show a grid pattern
    log("\n>>> Screen coordinates (offset+2 and +4) for first 30 records:")
    log("  Record | offset+2 (X?) | offset+4 (Y?) | offset+0")
    for r in range(30):
        addr = RECORD_BASE + r * RECORD_STRIDE
        b0 = core.debug_read_8(addr)
        # Read as 16-bit for better precision
        xy_x = core.debug_read_8(addr + 2)
        xy_y = core.debug_read_8(addr + 4)
        log(f"  R{r:3d}   | {xy_x:3d} (0x{xy_x:02X})    | {xy_y:3d} (0x{xy_y:02X})    | {b0}")

    # Now drop at center and see which records changed
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)

    log("\n>>> Records that changed after center drop:")
    changed_records = []
    for r in range(NUM_RECORDS):
        addr = RECORD_BASE + r * RECORD_STRIDE
        b0 = core.debug_read_8(addr)
        b16 = core.debug_read_8(addr + 16)
        old_b0, old_b16 = records_save[r]
        if b0 != old_b0 or b16 != old_b16:
            col = r // 20 if r < 200 else -1
            row = r % 20 if r < 200 else -1
            changed_records.append(r)
            log(f"  R{r:3d} (C{col},R{row}): b0 {old_b0}->{b0}, b16 {old_b16}->{b16}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Wider board search — scan EVERY 32-bit word
    # Look for region with 0x00000001-type values that appear
    # at board-like strides after a drop
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Wide board scan (32-bit words, value 1-8)")
    log("=" * 60)

    reload_save(core)
    snap_pre = memory.snapshot()
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)
    snap_post = memory.snapshot()

    # Find words going 0 -> small value (1-8)
    board_candidates = {}
    for phys in range(SCAN_START, SCAN_END, 4):
        w_old = struct.unpack_from('<I', snap_pre, phys)[0]
        w_new = struct.unpack_from('<I', snap_post, phys)[0]
        if w_old == 0 and 1 <= w_new <= 8:
            board_candidates[0x80000000 + phys] = w_new

    log(f"\n  Words going 0 -> (1-8): {len(board_candidates)}")

    # Group by address region (first 3 hex digits after 0x80)
    from collections import defaultdict
    regions = defaultdict(list)
    for addr, val in sorted(board_candidates.items()):
        region = (addr >> 12) & 0xFFF
        regions[region].append((addr, val))

    log(f"  Address regions:")
    for region in sorted(regions.keys()):
        addrs = regions[region]
        if len(addrs) >= 2:
            log(f"    0x{0x80000 + region:05X}xxx: {len(addrs)} hits")
            for addr, val in addrs[:5]:
                log(f"      0x{addr:08X}: -> {val}")

    # For each region with 3+ hits, check stride pattern
    for region in sorted(regions.keys()):
        addrs_list = [a for a, _ in regions[region]]
        if len(addrs_list) >= 3:
            strides = [addrs_list[i+1] - addrs_list[i] for i in range(len(addrs_list)-1)]
            if len(set(strides)) <= 3:  # Consistent stride
                log(f"\n  Region 0x{0x80000 + region:05X}xxx has consistent strides: {strides}")
                for addr, val in regions[region]:
                    log(f"    0x{addr:08X}: {val}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 6: Piece struct deep dive
    # Read the full piece struct area from BBEC to E0B0
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 6: Piece struct area dump")
    log("=" * 60)

    reload_save(core)

    # The piece structs seem to be at 0x8010BBE0-0x8010E0D0 range
    # Let's dump key parts and look for Y/X position within the struct
    log("\n>>> Piece struct A (0x8010BBD0-0x8010BC60):")
    hexdump_debug(core, 0x8010BBD0, 0x90)

    log("\n>>> Piece struct B (0x8010DFD0-0x8010E0E0):")
    hexdump_debug(core, 0x8010DFD0, 0x110)

    # Now let piece fall and check what changed in these structs
    core.resume()
    time.sleep(0.8)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After 0.8s of falling:")
    log(">>> Piece struct A (0x8010BBD0-0x8010BC60):")
    hexdump_debug(core, 0x8010BBD0, 0x90)

    log("\n>>> Piece struct B (0x8010DFD0-0x8010E0E0):")
    hexdump_debug(core, 0x8010DFD0, 0x110)

    # Move right and check
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.1)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After move right:")
    log(">>> Piece struct A (0x8010BBD0-0x8010BC60):")
    hexdump_debug(core, 0x8010BBD0, 0x90)

    # ── Score check ──────────────────────────────────────────
    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v7 COMPLETE")
    log("=" * 60)

    log("\n>>> Shutting down...")
    try:
        core.stop()
    except:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
