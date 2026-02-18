#!/usr/bin/env python3
"""Discovery v9 — Find X position and board data.

CONFIRMED from v8:
- Y position: float at 0x800D02CC (increases during fall, ~32 at top)
- Piece struct: 0x8010BBE0, board ptr: 0x8010BC5C -> 0x8012E220
- Piece queue entry: 0x8010DFFC (u32, values 0-7)

TODO:
- X position: check 0x800D0298 and surrounding addresses on left/right moves
- Board grid: follow pointers 0x8012F3D0 and 0x8012D050
- Rotation state: check what changes on A_BUTTON but NOT on move/fall
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


def read_float(core, addr):
    raw = core.debug_read_32(addr)
    return struct.unpack('f', struct.pack('I', raw))[0]


def read_word(core, addr):
    return core.debug_read_32(addr)


def main():
    log("=" * 60)
    log("Discovery v9 — X Position & Board Data")
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
    # EXPERIMENT 1: X position — read ALL values near Y on each move
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: X position search (0x800D0200-0x800D0300)")
    log("=" * 60)

    reload_save(core)

    # Read entire 0x800D0200-0x800D0300 region as u32 and float
    scan_addrs = list(range(0x800D0200, 0x800D0310, 4))

    log("\n>>> All values at save state:")
    for addr in scan_addrs:
        u32 = read_word(core, addr)
        fval = struct.unpack('f', struct.pack('I', u32))[0]
        if u32 != 0:
            log(f"  0x{addr:08X}: u32={u32:10d} (0x{u32:08X})  float={fval:.4f}")

    # Move right 1 time (without falling — pause immediately)
    vals_per_move = []
    reload_save(core)

    # Read initial values
    initial = {}
    for addr in scan_addrs:
        initial[addr] = read_word(core, addr)
    vals_per_move.append(initial)

    # Quickly move right 5 times, reading after each
    for move_num in range(5):
        core.resume()
        time.sleep(0.02)  # Minimal fall time
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
        core.pause()
        time.sleep(0.05)

        current = {}
        for addr in scan_addrs:
            current[addr] = read_word(core, addr)
        vals_per_move.append(current)

    log("\n>>> Addresses that changed on right moves:")
    for addr in scan_addrs:
        values = [v[addr] for v in vals_per_move]
        if len(set(values)) > 1:  # Changed at least once
            # Show as both u32 and float
            float_vals = [struct.unpack('f', struct.pack('I', v))[0] for v in values]
            log(f"  0x{addr:08X}: u32={values}  float={[f'{f:.2f}' for f in float_vals]}")

    # Now move LEFT 5 times from save state
    log("\n>>> After left moves:")
    reload_save(core)

    left_vals = [{}]
    for addr in scan_addrs:
        left_vals[0][addr] = read_word(core, addr)

    for move_num in range(5):
        core.resume()
        time.sleep(0.02)
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.05)
        core.pause()
        time.sleep(0.05)

        current = {}
        for addr in scan_addrs:
            current[addr] = read_word(core, addr)
        left_vals.append(current)

    log("\n>>> Addresses that changed on left moves:")
    for addr in scan_addrs:
        values = [v[addr] for v in left_vals]
        if len(set(values)) > 1:
            float_vals = [struct.unpack('f', struct.pack('I', v))[0] for v in values]
            log(f"  0x{addr:08X}: u32={values}  float={[f'{f:.2f}' for f in float_vals]}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Wider X search using snapshot diffs
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Wide X search — snapshot diff on right move")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()
    snap_before = memory.snapshot()

    # Single fast right move
    core.resume()
    time.sleep(0.02)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.02)
    core.pause()
    time.sleep(0.05)
    snap_after = memory.snapshot()

    # Diff u32 words that changed by a SMALL amount (1-30 as u32)
    SCAN_START = 0x080000
    SCAN_END = 0x200000
    small_diffs = []
    for phys in range(SCAN_START, SCAN_END, 4):
        wa = struct.unpack_from('<I', snap_before, phys)[0]
        wb = struct.unpack_from('<I', snap_after, phys)[0]
        if wa != wb:
            diff = wb - wa
            if 1 <= diff <= 30 or -30 <= diff <= -1:
                addr = 0x80000000 + phys
                small_diffs.append((addr, wa, wb, diff))

    log(f"\n  Words with small delta (+-1 to +-30): {len(small_diffs)}")
    for addr, old, new, delta in sorted(small_diffs)[:40]:
        fval_old = struct.unpack('f', struct.pack('I', old))[0]
        fval_new = struct.unpack('f', struct.pack('I', new))[0]
        log(f"    0x{addr:08X}: {old} -> {new} (delta {delta:+d})  float: {fval_old:.2f} -> {fval_new:.2f}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Rotation state
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Rotation state search")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()
    snap_rot0 = memory.snapshot()

    # Single rotation (A button)
    core.resume()
    time.sleep(0.02)
    press_button(input_server, "A_BUTTON", hold_sec=0.05, pause_sec=0.05)
    core.pause()
    time.sleep(0.05)
    snap_rot1 = memory.snapshot()

    # Second rotation
    core.resume()
    time.sleep(0.02)
    press_button(input_server, "A_BUTTON", hold_sec=0.05, pause_sec=0.05)
    core.pause()
    time.sleep(0.05)
    snap_rot2 = memory.snapshot()

    # Find words that cycle 0->1->2 or increment by 1 each rotation
    rot_candidates = []
    for phys in range(SCAN_START, SCAN_END, 4):
        w0 = struct.unpack_from('<I', snap_rot0, phys)[0]
        w1 = struct.unpack_from('<I', snap_rot1, phys)[0]
        w2 = struct.unpack_from('<I', snap_rot2, phys)[0]
        # Rotation should cycle: 0->1->2->3->0 or similar
        if w0 < 4 and w1 < 4 and w2 < 4 and w0 != w1 and w1 != w2:
            if (w1 - w0) % 4 == 1 and (w2 - w1) % 4 == 1:  # Incrementing mod 4
                addr = 0x80000000 + phys
                rot_candidates.append((addr, w0, w1, w2))

    log(f"\n  Rotation candidates (mod-4 incrementing): {len(rot_candidates)}")
    for addr, w0, w1, w2 in rot_candidates[:20]:
        current = read_word(core, addr)
        log(f"    0x{addr:08X}: {w0} -> {w1} -> {w2}  [now={current}]")

    # Also check near the piece struct
    log("\n>>> Piece struct area rotation check:")
    for addr in range(0x8010BBE0, 0x8010BC60, 4):
        w0 = struct.unpack_from('<I', snap_rot0, (addr & 0x1FFFFFFF))[0]
        w1 = struct.unpack_from('<I', snap_rot1, (addr & 0x1FFFFFFF))[0]
        w2 = struct.unpack_from('<I', snap_rot2, (addr & 0x1FFFFFFF))[0]
        if w0 != w1 or w1 != w2:
            log(f"    0x{addr:08X}: {w0} -> {w1} -> {w2}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Follow board pointers
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Follow board pointers")
    log("=" * 60)

    reload_save(core)

    # Board struct at 0x8012E220 has pointers:
    ptr1 = read_word(core, 0x8012E228)  # 0x8012F3D0
    ptr2 = read_word(core, 0x8012E22C)  # 0x8012D050

    log(f"\n>>> Board pointer 1: 0x{ptr1:08X}")
    if 0x80000000 <= ptr1 < 0x80800000:
        hexdump_debug(core, ptr1, 0x100)

    log(f"\n>>> Board pointer 2: 0x{ptr2:08X}")
    if 0x80000000 <= ptr2 < 0x80800000:
        hexdump_debug(core, ptr2, 0x200)

    # Also follow pointers from 0x8012E2C0-0x8012E2D0
    for offset in [0xC0, 0xC8, 0xD0, 0xD8]:
        ptr = read_word(core, 0x8012E200 + offset)
        if 0x80000000 <= ptr < 0x80800000:
            log(f"\n>>> Pointer at 0x{0x8012E200 + offset:08X} -> 0x{ptr:08X}:")
            hexdump_debug(core, ptr, 0x80)

    # Drop a piece and check these areas
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)

    log(f"\n>>> Board pointer 2 AFTER drop: 0x{ptr2:08X}")
    if 0x80000000 <= ptr2 < 0x80800000:
        hexdump_debug(core, ptr2, 0x200)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Search for 10x20 board array
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Search for 10x20 contiguous board array")
    log("=" * 60)

    # Reload and drop at different positions to find board cells
    # Strategy: drop at left, reload, drop at right
    # Board cells will be at different offsets within the same array
    # The stride between left and right cells gives us column stride

    reload_save(core)
    memory.refresh_pointer()
    snap_empty = memory.snapshot()

    # Drop at far left
    core.resume()
    time.sleep(0.1)
    for _ in range(5):
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)
    snap_left = memory.snapshot()

    # Find ALL u8 bytes that changed from 0 to small nonzero (1-8)
    left_cells = set()
    for phys in range(SCAN_START, SCAN_END):
        if snap_empty[phys] == 0 and 1 <= snap_left[phys] <= 8:
            virt = 0x80000000 + (phys ^ 3)  # XOR^3 correction
            left_cells.add(virt)

    # Drop at far right
    reload_save(core)
    memory.refresh_pointer()
    snap_empty2 = memory.snapshot()
    core.resume()
    time.sleep(0.1)
    for _ in range(5):
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)
    snap_right = memory.snapshot()

    right_cells = set()
    for phys in range(SCAN_START, SCAN_END):
        if snap_empty2[phys] == 0 and 1 <= snap_right[phys] <= 8:
            virt = 0x80000000 + (phys ^ 3)  # XOR^3 correction
            right_cells.add(virt)

    # Cells unique to each position
    left_only = sorted(left_cells - right_cells)
    right_only = sorted(right_cells - left_cells)

    log(f"\n  Left-drop unique cells (0->1-8): {len(left_only)}")
    for addr in left_only[:20]:
        val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {val}")

    log(f"\n  Right-drop unique cells (0->1-8): {len(right_only)}")
    for addr in right_only[:20]:
        val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {val}")

    # Look for pairs (left_addr, right_addr) with consistent stride
    if left_only and right_only:
        # The column stride = (right_addr - left_addr) for corresponding cells
        # If I-piece (horizontal), there are 4 cells in a row
        # Left drop: cells at columns 0-3, Right drop: cells at columns 6-9 (or similar)
        # Stride between corresponding cells = (right_col - left_col) * cell_size

        # Try to find matching patterns
        left_strides = [left_only[i+1] - left_only[i] for i in range(min(len(left_only)-1, 10))]
        right_strides = [right_only[i+1] - right_only[i] for i in range(min(len(right_only)-1, 10))]
        log(f"\n  Left cell strides: {left_strides[:10]}")
        log(f"  Right cell strides: {right_strides[:10]}")

    # ── Also do a byte search for something that looks like a row (10 values) ──
    log("\n>>> Searching for 10-byte patterns (board rows)...")
    # Look for 10 consecutive bytes where most are 0 but some are nonzero
    # after dropping a piece at center
    reload_save(core)
    memory.refresh_pointer()
    snap_pre_center = memory.snapshot()
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)
    snap_post_center = memory.snapshot()

    # Look for groups of 10 bytes where exactly 2-4 changed (part of a tetromino in one row)
    for stride in [1, 2, 4]:  # Try different cell sizes
        patterns = []
        for start_phys in range(SCAN_START, SCAN_END - 10 * stride, stride):
            changes = 0
            row_changes = []
            for col in range(10):
                phys = start_phys + col * stride
                if phys < SCAN_END and snap_pre_center[phys] != snap_post_center[phys]:
                    if snap_pre_center[phys] == 0 and snap_post_center[phys] != 0:
                        changes += 1
                        row_changes.append(col)
            if 2 <= changes <= 4 and len(row_changes) >= 2:
                # Check if the rest are still 0
                zeros = sum(1 for col in range(10) if
                           start_phys + col * stride < SCAN_END and
                           snap_post_center[start_phys + col * stride] == 0)
                if zeros >= 6:
                    virt = 0x80000000 + (start_phys ^ 3)
                    patterns.append((virt, changes, row_changes, stride))

        if patterns:
            log(f"\n  Stride={stride}: {len(patterns)} potential board rows")
            for addr, changes, cols, s in patterns[:10]:
                log(f"    0x{addr:08X}: {changes} changed at cols {cols}")

    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v9 COMPLETE")
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
