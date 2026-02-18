#!/usr/bin/env python3
"""Focused memory discovery v4 — find board, piece, and state addresses.

Loads save state (slot 1) created by navigate_and_save.py.
Reloads between each phase for clean comparisons.
Uses cross-phase intersection to isolate X, Y, rotation, piece type.
Scans only game data range (skips framebuffer/video noise).
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

# Game data lives here (known: score=0x11EED6, gamemode=0x0CFF60)
# Skip low RDRAM (stack/DMA) and high RDRAM (framebuffer)
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
    """Diff in game data range only. Returns {virt_addr: (old, new)}."""
    diffs = {}
    for i in range(SCAN_START, SCAN_END):
        if snap_a[i] != snap_b[i]:
            diffs[0x80000000 + i] = (snap_a[i], snap_b[i])
    return diffs


def reload_save(core, memory):
    """Reload save state and return a fresh snapshot."""
    core.load_state(slot=1)
    time.sleep(0.5)
    core.pause()
    time.sleep(0.1)
    return memory.snapshot()


def hexdump_debug(core, start_virt, count):
    data = [core.debug_read_8(start_virt + i) for i in range(count)]
    for i in range(0, len(data), 16):
        row = data[i:i + 16]
        hex_str = " ".join(f"{b:02X}" for b in row)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        log(f"  {start_virt + i:08X}: {hex_str:<48} |{ascii_str}|")


def main():
    log("=" * 60)
    log("Focused Memory Discovery v4 — The New Tetris")
    log(f"Scan range: 0x{SCAN_START + 0x80000000:08X} - 0x{SCAN_END + 0x80000000:08X}")
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

    # ── Phase 1: Natural fall (Y coordinate) ─────────────────
    log("\n>>> Phase 1: Let piece fall naturally (isolate Y)...")
    snap_before = reload_save(core, memory)
    core.resume()
    time.sleep(1.5)  # Let piece fall ~1.5 seconds
    snap_after_fall = snapshot_paused(core, memory)
    diffs_fall = diff_snapshots(snap_before, snap_after_fall)
    log(f"  Changes: {len(diffs_fall)}")

    # ── Phase 2: Move right (X coordinate) ───────────────────
    log("\n>>> Phase 2: Move piece right (isolate X)...")
    snap_before = reload_save(core, memory)
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.15)
    snap_after_move = snapshot_paused(core, memory)
    diffs_move = diff_snapshots(snap_before, snap_after_move)
    log(f"  Changes: {len(diffs_move)}")

    # ── Phase 3: Rotate (rotation state) ─────────────────────
    log("\n>>> Phase 3: Rotate piece (isolate rotation)...")
    snap_before = reload_save(core, memory)
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "A_BUTTON", hold_sec=0.05, pause_sec=0.15)
    snap_after_rot = snapshot_paused(core, memory)
    diffs_rot = diff_snapshots(snap_before, snap_after_rot)
    log(f"  Changes: {len(diffs_rot)}")

    # ── Phase 4: Hard drop + new piece (piece type change) ───
    log("\n>>> Phase 4: Hard drop piece (board + piece type change)...")
    snap_before = reload_save(core, memory)
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after_drop = snapshot_paused(core, memory)
    diffs_drop = diff_snapshots(snap_before, snap_after_drop)
    log(f"  Changes: {len(diffs_drop)}")

    # ── Phase 5: Second drop (different board cells) ─────────
    log("\n>>> Phase 5: Drop another piece from current state...")
    snap_before5 = memory.snapshot()
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after_drop2 = snapshot_paused(core, memory)
    diffs_drop2 = diff_snapshots(snap_before5, snap_after_drop2)
    log(f"  Changes: {len(diffs_drop2)}")

    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("CROSS-PHASE ANALYSIS")
    log("=" * 60)

    fall_addrs = set(diffs_fall.keys())
    move_addrs = set(diffs_move.keys())
    rot_addrs = set(diffs_rot.keys())
    drop_addrs = set(diffs_drop.keys())
    drop2_addrs = set(diffs_drop2.keys())

    # ── Y coordinate: changes during fall, NOT during move/rotate ─
    y_candidates = fall_addrs - move_addrs - rot_addrs
    log(f"\n  Y candidates (fall only, not move/rotate): {len(y_candidates)}")
    y_filtered = []
    for addr in sorted(y_candidates):
        old, new = diffs_fall[addr]
        delta = new - old
        if 0 < delta < 10 and old < 25:
            y_filtered.append((addr, old, new))
    log(f"  Y filtered (small increase, value < 25): {len(y_filtered)}")
    for addr, old, new in y_filtered[:30]:
        log(f"    0x{addr:08X}: {old} -> {new} (delta +{new - old})")

    # ── X coordinate: changes during move, NOT during fall/rotate ─
    x_candidates = move_addrs - fall_addrs - rot_addrs
    log(f"\n  X candidates (move only, not fall/rotate): {len(x_candidates)}")
    x_filtered = []
    for addr in sorted(x_candidates):
        old, new = diffs_move[addr]
        delta = new - old
        if abs(delta) == 1 and 0 <= min(old, new) and max(old, new) <= 12:
            x_filtered.append((addr, old, new))
    log(f"  X filtered (delta +-1, value 0-12): {len(x_filtered)}")
    for addr, old, new in x_filtered[:30]:
        log(f"    0x{addr:08X}: {old} -> {new}")

    # Also check addresses that change by exactly 1 (less strict)
    x_delta1 = []
    for addr in sorted(x_candidates):
        old, new = diffs_move[addr]
        if abs(new - old) == 1:
            x_delta1.append((addr, old, new))
    log(f"  X any delta=1: {len(x_delta1)}")
    for addr, old, new in x_delta1[:30]:
        log(f"    0x{addr:08X}: {old} -> {new}")

    # ── Rotation: changes during rotate, NOT during fall/move ─
    rot_candidates = rot_addrs - fall_addrs - move_addrs
    log(f"\n  Rot candidates (rotate only, not fall/move): {len(rot_candidates)}")
    rot_filtered = []
    for addr in sorted(rot_candidates):
        old, new = diffs_rot[addr]
        if 0 <= old <= 3 and 0 <= new <= 3 and old != new:
            rot_filtered.append((addr, old, new))
    log(f"  Rot filtered (values 0-3): {len(rot_filtered)}")
    for addr, old, new in rot_filtered[:30]:
        log(f"    0x{addr:08X}: {old} -> {new}")

    # ── Piece type: changes during drop, values 0-6 ──────────
    # Piece type should change after a drop (new piece spawns)
    # It should also appear in drop2 (another piece change)
    type_candidates = drop_addrs & drop2_addrs
    log(f"\n  Piece type candidates (changed in both drops): {len(type_candidates)}")
    type_filtered = []
    for addr in sorted(type_candidates):
        o1, n1 = diffs_drop[addr]
        o2, n2 = diffs_drop2[addr]
        # After drop: old piece type -> new piece type, both 0-6
        if all(0 <= v <= 6 for v in (o1, n1, o2, n2)):
            type_filtered.append((addr, o1, n1, o2, n2))
    log(f"  Piece type filtered (all values 0-6): {len(type_filtered)}")
    for addr, o1, n1, o2, n2 in type_filtered[:30]:
        log(f"    0x{addr:08X}: drop1 {o1}->{n1}, drop2 {o2}->{n2}")

    # ── Board cells: 0->nonzero after drop, unique per drop ──
    ztn_drop1 = {a for a, (o, n) in diffs_drop.items() if o == 0 and n != 0}
    ztn_drop2 = {a for a, (o, n) in diffs_drop2.items() if o == 0 and n != 0}
    board_drop1_only = ztn_drop1 - ztn_drop2
    board_drop2_only = ztn_drop2 - ztn_drop1
    board_both = ztn_drop1 & ztn_drop2

    log(f"\n  Board analysis:")
    log(f"    0->nonzero in drop 1: {len(ztn_drop1)}")
    log(f"    0->nonzero in drop 2: {len(ztn_drop2)}")
    log(f"    Unique to drop 1 (board cells): {len(board_drop1_only)}")
    log(f"    Unique to drop 2 (board cells): {len(board_drop2_only)}")
    log(f"    In both drops: {len(board_both)}")

    # Stride analysis on board cells
    all_board = sorted(board_drop1_only | board_drop2_only)
    if len(all_board) >= 2:
        strides = [all_board[i + 1] - all_board[i] for i in range(len(all_board) - 1)]
        stride_freq = Counter(strides)
        log(f"    Stride frequency (top 10):")
        for stride, count in stride_freq.most_common(10):
            log(f"      stride {stride} (0x{stride:X}): {count}x")

    log(f"\n  Board cells unique to drop 1:")
    for addr in sorted(board_drop1_only)[:20]:
        _, new = diffs_drop[addr]
        log(f"    0x{addr:08X}: 0x00 -> 0x{new:02X}")

    log(f"\n  Board cells unique to drop 2:")
    for addr in sorted(board_drop2_only)[:20]:
        _, new = diffs_drop2[addr]
        log(f"    0x{addr:08X}: 0x00 -> 0x{new:02X}")

    # ── Addresses changing in ALL phases (timers/counters) ────
    all_phases = fall_addrs & move_addrs & rot_addrs & drop_addrs
    log(f"\n  Changed in ALL phases (timers/counters): {len(all_phases)}")

    # ── Dump neighborhoods of best candidates ─────────────────
    if y_filtered:
        best_y = y_filtered[0][0]
        log(f"\n>>> Dumping around best Y candidate 0x{best_y:08X}:")
        hexdump_debug(core, best_y - 32, 96)

    if x_filtered:
        best_x = x_filtered[0][0]
        log(f"\n>>> Dumping around best X candidate 0x{best_x:08X}:")
        hexdump_debug(core, best_x - 32, 96)
    elif x_delta1:
        best_x = x_delta1[0][0]
        log(f"\n>>> Dumping around best X (delta=1) candidate 0x{best_x:08X}:")
        hexdump_debug(core, best_x - 32, 96)

    if rot_filtered:
        best_rot = rot_filtered[0][0]
        log(f"\n>>> Dumping around best Rot candidate 0x{best_rot:08X}:")
        hexdump_debug(core, best_rot - 32, 96)

    if type_filtered:
        best_type = type_filtered[0][0]
        log(f"\n>>> Dumping around best Piece Type candidate 0x{best_type:08X}:")
        hexdump_debug(core, best_type - 32, 96)

    if board_drop1_only:
        # Densest cluster
        sorted_b = sorted(board_drop1_only)
        best_start = sorted_b[0]
        best_count = 0
        for s in sorted_b:
            c = sum(1 for a in sorted_b if s <= a < s + 0x100)
            if c > best_count:
                best_count = c
                best_start = s
        log(f"\n>>> Densest board cluster: 0x{best_start:08X} ({best_count} cells in 0x100)")
        hexdump_debug(core, best_start, min(0x100, 256))

    # ── Score check ──────────────────────────────────────────
    log(f"\n>>> Score after experiments:")
    score = core.debug_read_16(0x8011EED6)
    log(f"  Score A: {score}")

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
