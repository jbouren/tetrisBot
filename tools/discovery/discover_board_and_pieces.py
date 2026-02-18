#!/usr/bin/env python3
"""Targeted discovery: board layout, next piece, and reserve piece (L swap).

Loads save state from slot 1. Runs focused experiments to confirm:
1. Board cell structure (0x800D10xx region, stride ~0x9C)
2. Next piece address
3. Reserve/hold piece address (swap with L_TRIG)
4. Current piece type address (0x8010E0A4 candidate)
"""

import logging
import os
import sys
import time
from collections import Counter
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

# Game data range (skip framebuffer)
SCAN_START = 0x080000
SCAN_END = 0x200000


def press_button(input_server, button, hold_sec=0.1, pause_sec=0.3):
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(pause_sec)


def snapshot_paused(core, memory):
    core.pause()
    time.sleep(0.1)
    return memory.snapshot()


def diff_snapshots(snap_a, snap_b):
    diffs = {}
    for i in range(SCAN_START, SCAN_END):
        if snap_a[i] != snap_b[i]:
            diffs[0x80000000 + i] = (snap_a[i], snap_b[i])
    return diffs


def reload_save(core, memory):
    core.load_state(slot=1)
    time.sleep(0.5)
    core.pause()
    time.sleep(0.1)
    return memory.snapshot()


def hexdump_debug(core, start_virt, count, highlight_addrs=None):
    """Hex dump with optional address highlighting."""
    if highlight_addrs is None:
        highlight_addrs = set()
    data = [core.debug_read_8(start_virt + i) for i in range(count)]
    for i in range(0, len(data), 16):
        row = data[i:i + 16]
        addr = start_virt + i
        hex_parts = []
        for j, b in enumerate(row):
            a = addr + j
            if a in highlight_addrs:
                hex_parts.append(f"[{b:02X}]")
            else:
                hex_parts.append(f" {b:02X} ")
        hex_str = "".join(hex_parts)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        log(f"  {addr:08X}: {hex_str}  |{ascii_str}|")


def read_u32_debug(core, addr):
    return core.debug_read_32(addr)


def read_u16_debug(core, addr):
    return core.debug_read_16(addr)


def read_u8_debug(core, addr):
    return core.debug_read_8(addr)


def main():
    log("=" * 60)
    log("Board & Pieces Discovery — The New Tetris")
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
    # EXPERIMENT 1: Board cell structure
    # Previous run found cells at 0x800D10BA, 0x800D1156, etc.
    # all going 0->0x02 with stride ~0x9C (156 bytes)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Board Cell Structure")
    log("=" * 60)

    # Dump the board candidate region BEFORE any drops (should be empty/zero)
    snap_clean = reload_save(core, memory)
    log("\n>>> Board region BEFORE drop (0x800D1000 - 0x800D1600):")
    log("  (Candidate cells: 0x800D10BA, 0x800D1156, 0x800D11F2, etc.)")
    hexdump_debug(core, 0x800D1000, 0x100)
    log("  ...")
    hexdump_debug(core, 0x800D1400, 0x100)

    # Drop one piece
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_drop1 = snapshot_paused(core, memory)

    log("\n>>> Board region AFTER 1st drop (0x800D1000 - 0x800D1600):")
    hexdump_debug(core, 0x800D1000, 0x100)
    log("  ...")
    hexdump_debug(core, 0x800D1400, 0x100)

    # Check which bytes changed in this region specifically
    board_region_changes = []
    for offset in range(0x0D1000, 0x0D1600):
        if snap_clean[offset] != snap_drop1[offset]:
            addr = 0x80000000 + offset
            board_region_changes.append((addr, snap_clean[offset], snap_drop1[offset]))

    log(f"\n  Changes in 0x800D1000-0x800D1600: {len(board_region_changes)}")
    for addr, old, new in board_region_changes:
        log(f"    0x{addr:08X}: 0x{old:02X} -> 0x{new:02X}")

    # Calculate strides between changed addresses
    if len(board_region_changes) >= 2:
        addrs = [a for a, _, _ in board_region_changes]
        strides = [addrs[i+1] - addrs[i] for i in range(len(addrs)-1)]
        log(f"  Strides: {strides}")

    # Drop second piece (don't reload — let board accumulate)
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_drop2 = snapshot_paused(core, memory)

    board_changes_2 = []
    for offset in range(0x0D1000, 0x0D1600):
        if snap_drop1[offset] != snap_drop2[offset]:
            addr = 0x80000000 + offset
            board_changes_2.append((addr, snap_drop1[offset], snap_drop2[offset]))
    log(f"\n  Changes after 2nd drop (in board region): {len(board_changes_2)}")
    for addr, old, new in board_changes_2:
        log(f"    0x{addr:08X}: 0x{old:02X} -> 0x{new:02X}")

    # Drop third piece
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_drop3 = snapshot_paused(core, memory)

    board_changes_3 = []
    for offset in range(0x0D1000, 0x0D1600):
        if snap_drop2[offset] != snap_drop3[offset]:
            addr = 0x80000000 + offset
            board_changes_3.append((addr, snap_drop2[offset], snap_drop3[offset]))
    log(f"\n  Changes after 3rd drop (in board region): {len(board_changes_3)}")
    for addr, old, new in board_changes_3:
        log(f"    0x{addr:08X}: 0x{old:02X} -> 0x{new:02X}")

    # Full dump after 3 drops
    log("\n>>> Board region AFTER 3 drops (0x800D1000 - 0x800D1600):")
    hexdump_debug(core, 0x800D1000, 0x600)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Current piece type confirmation
    # Candidate: 0x8010E0A4 (4->0->6 in previous run)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Piece Type / Next Piece")
    log("=" * 60)

    # Read piece type candidate at save state (initial piece)
    snap_fresh = reload_save(core, memory)
    piece_val = read_u8_debug(core, 0x8010E0A4)
    log(f"\n  0x8010E0A4 at save state: {piece_val}")

    # Dump the neighborhood to look for next piece, piece queue
    log("\n>>> Piece region (0x8010E080 - 0x8010E100):")
    hexdump_debug(core, 0x8010E080, 0x80)

    # Also check the other piece candidate from prev run
    log(f"  0x8010BBEC at save state: {read_u8_debug(core, 0x8010BBEC)}")
    log(f"  0x800D02B3 at save state: {read_u8_debug(core, 0x800D02B3)}")

    # Dump wider region around 0x8010E0A4 for context
    log("\n>>> Wider piece context (0x8010E000 - 0x8010E200):")
    hexdump_debug(core, 0x8010E000, 0x200)

    # Drop piece and check what changes near the piece type
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after = snapshot_paused(core, memory)

    piece_val_after = read_u8_debug(core, 0x8010E0A4)
    log(f"\n  0x8010E0A4 after drop: {piece_val_after}")
    log(f"  0x8010BBEC after drop: {read_u8_debug(core, 0x8010BBEC)}")
    log(f"  0x800D02B3 after drop: {read_u8_debug(core, 0x800D02B3)}")

    # Find what changed in the piece type neighborhood
    log("\n  Changes near piece candidates (0x8010E000-0x8010E200):")
    for offset in range(0x10E000, 0x10E200):
        if snap_fresh[offset] != snap_after[offset]:
            addr = 0x80000000 + offset
            log(f"    0x{addr:08X}: {snap_fresh[offset]} -> {snap_after[offset]}")

    # Drop a second piece to confirm
    snap_before2 = memory.snapshot()
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after2 = snapshot_paused(core, memory)

    log(f"\n  0x8010E0A4 after 2nd drop: {read_u8_debug(core, 0x8010E0A4)}")
    log(f"  0x8010BBEC after 2nd drop: {read_u8_debug(core, 0x8010BBEC)}")

    log("\n  Changes near piece (0x8010E000-0x8010E200) after 2nd drop:")
    for offset in range(0x10E000, 0x10E200):
        if snap_before2[offset] != snap_after2[offset]:
            addr = 0x80000000 + offset
            log(f"    0x{addr:08X}: {snap_before2[offset]} -> {snap_after2[offset]}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Reserve / Hold piece (L_TRIG swap)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Reserve Piece (L swap)")
    log("=" * 60)

    snap_pre_swap = reload_save(core, memory)
    piece_before = read_u8_debug(core, 0x8010E0A4)
    log(f"\n  Current piece before L swap: {piece_before}")

    # Resume and press L to swap with reserve
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.5)
    snap_post_swap = snapshot_paused(core, memory)

    piece_after_swap = read_u8_debug(core, 0x8010E0A4)
    log(f"  Current piece after L swap: {piece_after_swap}")

    # Find what changed — focus on small values (piece type 0-6)
    diffs_swap = diff_snapshots(snap_pre_swap, snap_post_swap)
    log(f"  Total changes from L swap: {len(diffs_swap)}")

    # Filter for values 0-6 that changed (piece type related)
    swap_piece_candidates = []
    for addr in sorted(diffs_swap.keys()):
        old, new = diffs_swap[addr]
        if 0 <= old <= 6 and 0 <= new <= 6 and old != new:
            swap_piece_candidates.append((addr, old, new))

    log(f"\n  L-swap changes with values 0-6: {len(swap_piece_candidates)}")
    for addr, old, new in swap_piece_candidates[:40]:
        log(f"    0x{addr:08X}: {old} -> {new}")

    # Addresses that changed in L-swap but NOT in fall/move/rotate
    # (from previous run we know fall/move/rot change sets — but we can
    # approximate by looking for addresses only in the swap diff)
    log(f"\n  Changes near piece region (0x8010E000-0x8010E200):")
    for offset in range(0x10E000, 0x10E200):
        if snap_pre_swap[offset] != snap_post_swap[offset]:
            addr = 0x80000000 + offset
            old = snap_pre_swap[offset]
            new = snap_post_swap[offset]
            marker = " <-- PIECE TYPE?" if (0 <= old <= 6 and 0 <= new <= 6) else ""
            log(f"    0x{addr:08X}: {old} -> {new}{marker}")

    # Now swap BACK (press L again) to confirm it's reversible
    snap_pre_swap2 = memory.snapshot()
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.5)
    snap_post_swap2 = snapshot_paused(core, memory)

    log(f"\n  Current piece after L swap back: {read_u8_debug(core, 0x8010E0A4)}")

    # Find addresses that REVERSED (old->new then new->old)
    diffs_swap2 = diff_snapshots(snap_pre_swap2, snap_post_swap2)
    reversed_addrs = []
    for addr in sorted(set(diffs_swap.keys()) & set(diffs_swap2.keys())):
        o1, n1 = diffs_swap[addr]
        o2, n2 = diffs_swap2[addr]
        if o1 == n2 and n1 == o2:  # Perfectly reversed
            reversed_addrs.append((addr, o1, n1))

    log(f"\n  Addresses that reversed on 2nd L swap: {len(reversed_addrs)}")
    piece_reversed = [(a, o, n) for a, o, n in reversed_addrs if 0 <= o <= 6 and 0 <= n <= 6]
    log(f"  Of those, with values 0-6: {len(piece_reversed)}")
    for addr, old, new in piece_reversed[:30]:
        log(f"    0x{addr:08X}: {old} <-> {new}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Next piece queue
    # Drop several pieces and track what the "next" value was
    # before each drop vs what becomes current after
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Next Piece Queue")
    log("=" * 60)

    snap_q = reload_save(core, memory)

    # Read candidates for current and next piece
    # Check values 0-6 in the 0x8010E0xx and 0x8010BBxx regions
    log("\n>>> Reading piece-related bytes at save state:")
    for addr in [0x8010E0A4, 0x8010E0A5, 0x8010E0A6, 0x8010E0A7,
                 0x8010E0A0, 0x8010E0A1, 0x8010E0A2, 0x8010E0A3,
                 0x8010E0A8, 0x8010E0A9, 0x8010E0AA, 0x8010E0AB,
                 0x8010BBEC, 0x8010BBED, 0x8010BBEE, 0x8010BBEF,
                 0x8010BBE8, 0x8010BBE9, 0x8010BBEA, 0x8010BBEB]:
        val = read_u8_debug(core, addr)
        if 0 <= val <= 6:
            log(f"  0x{addr:08X}: {val}  <-- possible piece type")
        else:
            log(f"  0x{addr:08X}: {val}")

    # Do 5 drops and track how values shift
    log("\n>>> Tracking piece values across 5 drops:")
    for drop_num in range(1, 6):
        snap_pre = memory.snapshot() if drop_num > 1 else snap_q

        # Read key candidates before drop
        vals_before = {}
        for addr in [0x8010E0A4, 0x8010BBEC, 0x800D02B3]:
            vals_before[addr] = read_u8_debug(core, addr)

        core.resume()
        press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
        snap_post = snapshot_paused(core, memory)

        vals_after = {}
        for addr in [0x8010E0A4, 0x8010BBEC, 0x800D02B3]:
            vals_after[addr] = read_u8_debug(core, addr)

        log(f"\n  Drop {drop_num}:")
        for addr in [0x8010E0A4, 0x8010BBEC, 0x800D02B3]:
            log(f"    0x{addr:08X}: {vals_before[addr]} -> {vals_after[addr]}")

    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("DISCOVERY COMPLETE")
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
