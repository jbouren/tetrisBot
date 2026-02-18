#!/usr/bin/env python3
"""Discovery v5 — Corrected byte ordering (XOR^3 fix).

Previous scripts had a critical bug: snapshot diffs reported raw physical
offsets as virtual addresses (0x80000000 + phys_offset), but debug_read_8()
internally does rdram_array[phys_offset ^ 3]. So snapshot address 0x8010E0A4
actually corresponds to debug_read_8(0x8010E0A7).

This script corrects all addresses by applying XOR^3 when converting from
raw snapshot offsets to N64 virtual addresses. All reported addresses are
directly usable with debug_read_8() / MemoryReader.read_u8().

Loads save state (slot 1) created by navigate_and_save.py.
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

# Game data range (skip stack/DMA and framebuffer)
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


def phys_to_virt(phys_offset):
    """Convert raw RDRAM physical offset to N64 virtual address for byte access.

    On little-endian host, mupen64plus stores RDRAM with 32-bit word granularity.
    A byte at raw array index `i` corresponds to N64 virtual address
    0x80000000 + (i ^ 3), because debug_read_8(addr) does rdram[phys ^ 3].
    """
    return 0x80000000 + (phys_offset ^ 3)


def virt_to_phys_byte(virt_addr):
    """Convert N64 virtual address to raw RDRAM array index for byte access."""
    return (virt_addr & 0x1FFFFFFF) ^ 3


def diff_snapshots(snap_a, snap_b):
    """Diff snapshots with XOR^3 correction.

    Returns {virt_addr: (old, new)} where virt_addr is directly usable
    with debug_read_8().
    """
    diffs = {}
    for phys in range(SCAN_START, SCAN_END):
        if snap_a[phys] != snap_b[phys]:
            virt = phys_to_virt(phys)
            diffs[virt] = (snap_a[phys], snap_b[phys])
    return diffs


def reload_save(core, memory):
    """Reload save state and return a fresh snapshot."""
    core.load_state(slot=1)
    time.sleep(0.5)
    core.pause()
    time.sleep(0.1)
    return memory.snapshot()


def read_snap_byte(snap, virt_addr):
    """Read a byte from snapshot at N64 virtual address (with XOR^3)."""
    phys = virt_to_phys_byte(virt_addr)
    if 0 <= phys < len(snap):
        return snap[phys]
    return None


def hexdump_debug(core, start_virt, count, highlight_addrs=None):
    """Hex dump using debug_read_8 (correct byte order)."""
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


def main():
    log("=" * 60)
    log("Discovery v5 — XOR^3 Corrected Byte Ordering")
    log(f"Scan range: 0x{SCAN_START + 0x80000000:08X} - 0x{SCAN_END + 0x80000000:08X}")
    log("=" * 60)

    # Verify XOR^3 correction on known address
    log("\n>>> XOR^3 sanity check:")
    log(f"  Score addr 0x8011EED6 -> phys offset 0x{virt_to_phys_byte(0x8011EED6):06X}")
    log(f"  Phys 0x10E0A4 -> virt 0x{phys_to_virt(0x10E0A4):08X} (was wrongly reported as 0x8010E0A4)")
    log(f"  Phys 0x10BBEC -> virt 0x{phys_to_virt(0x10BBEC):08X} (was wrongly reported as 0x8010BBEC)")
    log(f"  Phys 0x0D10BA -> virt 0x{phys_to_virt(0x0D10BA):08X} (was wrongly reported as 0x800D10BA)")

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
    # EXPERIMENT 1: Verify known addresses work with correction
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Verify XOR^3 with known addresses")
    log("=" * 60)

    snap_initial = reload_save(core, memory)

    # Score at 0x8011EED6 — read both ways and compare
    score_debug = core.debug_read_16(0x8011EED6)
    # For 16-bit: physical offset = (0x11EED6 & ~1) ^ 2 = 0x11EED4 (for the word)
    # Actually let's just read the bytes individually
    score_b0 = core.debug_read_8(0x8011EED6)
    score_b1 = core.debug_read_8(0x8011EED7)
    score_snap_b0 = snap_initial[virt_to_phys_byte(0x8011EED6)]
    score_snap_b1 = snap_initial[virt_to_phys_byte(0x8011EED7)]
    log(f"\n  Score (debug_read_16): {score_debug}")
    log(f"  Score byte 0 debug={score_b0} snap={score_snap_b0} {'MATCH' if score_b0 == score_snap_b0 else 'MISMATCH'}")
    log(f"  Score byte 1 debug={score_b1} snap={score_snap_b1} {'MATCH' if score_b1 == score_snap_b1 else 'MISMATCH'}")

    # Check the previously misidentified addresses with correction
    log("\n  Corrected piece type candidates:")
    # Old: 0x8010E0A4 (raw phys 0x10E0A4) → corrected: 0x8010E0A7
    corrected_piece = phys_to_virt(0x10E0A4)
    val_debug = core.debug_read_8(corrected_piece)
    val_snap = snap_initial[0x10E0A4]
    log(f"  0x{corrected_piece:08X} (was 0x8010E0A4): debug={val_debug} snap_raw={val_snap} {'MATCH' if val_debug == val_snap else 'MISMATCH'}")

    corrected_bbec = phys_to_virt(0x10BBEC)
    val_debug = core.debug_read_8(corrected_bbec)
    val_snap = snap_initial[0x10BBEC]
    log(f"  0x{corrected_bbec:08X} (was 0x8010BBEC): debug={val_debug} snap_raw={val_snap} {'MATCH' if val_debug == val_snap else 'MISMATCH'}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Fall / Move / Rotate / Drop with corrected addresses
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Cross-phase analysis (corrected)")
    log("=" * 60)

    # Phase A: Natural fall (Y coordinate)
    log("\n>>> Phase A: Let piece fall ~1.5s...")
    snap_before = reload_save(core, memory)
    core.resume()
    time.sleep(1.5)
    snap_after_fall = snapshot_paused(core, memory)
    diffs_fall = diff_snapshots(snap_before, snap_after_fall)
    log(f"  Changes: {len(diffs_fall)}")

    # Phase B: Move right (X coordinate)
    log("\n>>> Phase B: Move piece right...")
    snap_before = reload_save(core, memory)
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.15)
    snap_after_move = snapshot_paused(core, memory)
    diffs_move = diff_snapshots(snap_before, snap_after_move)
    log(f"  Changes: {len(diffs_move)}")

    # Phase C: Rotate (A button)
    log("\n>>> Phase C: Rotate piece...")
    snap_before = reload_save(core, memory)
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "A_BUTTON", hold_sec=0.05, pause_sec=0.15)
    snap_after_rot = snapshot_paused(core, memory)
    diffs_rot = diff_snapshots(snap_before, snap_after_rot)
    log(f"  Changes: {len(diffs_rot)}")

    # Phase D: Hard drop (piece type changes)
    log("\n>>> Phase D: Hard drop piece...")
    snap_before = reload_save(core, memory)
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after_drop = snapshot_paused(core, memory)
    diffs_drop = diff_snapshots(snap_before, snap_after_drop)
    log(f"  Changes: {len(diffs_drop)}")

    # Phase E: Second drop from same state (different board cells)
    log("\n>>> Phase E: Drop 2nd piece (from current state)...")
    snap_before5 = memory.snapshot()
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after_drop2 = snapshot_paused(core, memory)
    diffs_drop2 = diff_snapshots(snap_before5, snap_after_drop2)
    log(f"  Changes: {len(diffs_drop2)}")

    # ── Cross-phase analysis ──────────────────────────────────
    log("\n" + "-" * 40)
    log("CROSS-PHASE RESULTS (corrected addresses)")
    log("-" * 40)

    fall_addrs = set(diffs_fall.keys())
    move_addrs = set(diffs_move.keys())
    rot_addrs = set(diffs_rot.keys())
    drop_addrs = set(diffs_drop.keys())
    drop2_addrs = set(diffs_drop2.keys())

    # ── Y coordinate: changes during fall, NOT during move/rotate ─
    y_candidates = fall_addrs - move_addrs - rot_addrs
    y_filtered = []
    for addr in sorted(y_candidates):
        old, new = diffs_fall[addr]
        delta = new - old
        if 0 < delta < 10 and old < 25:
            y_filtered.append((addr, old, new))
    log(f"\n  Y candidates (fall only, small increase, val<25): {len(y_filtered)}")
    for addr, old, new in y_filtered[:30]:
        # Verify with debug_read_8
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {old} -> {new} (delta +{new - old})  [debug now={dbg_val}]")

    # ── X coordinate: changes during move, NOT during fall/rotate ─
    x_candidates = move_addrs - fall_addrs - rot_addrs
    x_filtered = []
    for addr in sorted(x_candidates):
        old, new = diffs_move[addr]
        delta = new - old
        if abs(delta) == 1 and 0 <= min(old, new) and max(old, new) <= 12:
            x_filtered.append((addr, old, new))
    log(f"\n  X candidates (move only, delta=+-1, val 0-12): {len(x_filtered)}")
    for addr, old, new in x_filtered[:30]:
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {old} -> {new}  [debug now={dbg_val}]")

    # ── Rotation: changes during rotate, NOT during fall/move ─
    rot_candidates = rot_addrs - fall_addrs - move_addrs
    rot_filtered = []
    for addr in sorted(rot_candidates):
        old, new = diffs_rot[addr]
        if 0 <= old <= 3 and 0 <= new <= 3 and old != new:
            rot_filtered.append((addr, old, new))
    log(f"\n  Rot candidates (rotate only, val 0-3): {len(rot_filtered)}")
    for addr, old, new in rot_filtered[:30]:
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {old} -> {new}  [debug now={dbg_val}]")

    # ── Piece type: changes after both drops, values 0-6 ─────
    type_candidates = drop_addrs & drop2_addrs
    type_filtered = []
    for addr in sorted(type_candidates):
        o1, n1 = diffs_drop[addr]
        o2, n2 = diffs_drop2[addr]
        if all(0 <= v <= 6 for v in (o1, n1, o2, n2)):
            type_filtered.append((addr, o1, n1, o2, n2))
    log(f"\n  Piece type candidates (both drops, val 0-6): {len(type_filtered)}")
    for addr, o1, n1, o2, n2 in type_filtered[:30]:
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: drop1 {o1}->{n1}, drop2 {o2}->{n2}  [debug now={dbg_val}]")

    # ── Board cells: 0->nonzero after drop ────────────────────
    ztn_drop1 = {a for a, (o, n) in diffs_drop.items() if o == 0 and n != 0}
    ztn_drop2 = {a for a, (o, n) in diffs_drop2.items() if o == 0 and n != 0}
    board_drop1_only = ztn_drop1 - ztn_drop2
    board_drop2_only = ztn_drop2 - ztn_drop1

    log(f"\n  Board (0->nonzero):")
    log(f"    Drop 1 unique: {len(board_drop1_only)}")
    log(f"    Drop 2 unique: {len(board_drop2_only)}")

    all_board = sorted(board_drop1_only | board_drop2_only)
    if len(all_board) >= 2:
        strides = [all_board[i + 1] - all_board[i] for i in range(len(all_board) - 1)]
        stride_freq = Counter(strides)
        log(f"    Stride frequency (top 10):")
        for stride, count in stride_freq.most_common(10):
            log(f"      stride {stride} (0x{stride:X}): {count}x")

    log(f"\n  Board cells from drop 1:")
    for addr in sorted(board_drop1_only)[:20]:
        _, new = diffs_drop[addr]
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: 0->0x{new:02X}  [debug now={dbg_val}]")

    log(f"\n  Board cells from drop 2:")
    for addr in sorted(board_drop2_only)[:20]:
        _, new = diffs_drop2[addr]
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: 0->0x{new:02X}  [debug now={dbg_val}]")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Reserve piece (L swap)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Reserve Piece (L swap)")
    log("=" * 60)

    snap_pre_swap = reload_save(core, memory)

    # Read key addresses at save state
    log("\n>>> Values at save state (corrected addresses):")
    for addr in sorted(set(a for a, _, _, _, _ in type_filtered[:10])):
        val = core.debug_read_8(addr)
        log(f"  0x{addr:08X}: {val}")

    # L swap
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.5)
    snap_post_swap = snapshot_paused(core, memory)
    diffs_swap = diff_snapshots(snap_pre_swap, snap_post_swap)
    log(f"\n  Total changes from L swap: {len(diffs_swap)}")

    # Filter for piece-like values (0-6)
    swap_piece_candidates = []
    for addr in sorted(diffs_swap.keys()):
        old, new = diffs_swap[addr]
        if 0 <= old <= 6 and 0 <= new <= 6 and old != new:
            swap_piece_candidates.append((addr, old, new))

    log(f"  L-swap candidates (values 0-6): {len(swap_piece_candidates)}")
    for addr, old, new in swap_piece_candidates[:40]:
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {old} -> {new}  [debug now={dbg_val}]")

    # L swap back — find reversed addresses
    snap_pre_swap2 = memory.snapshot()
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.5)
    snap_post_swap2 = snapshot_paused(core, memory)
    diffs_swap2 = diff_snapshots(snap_pre_swap2, snap_post_swap2)

    reversed_addrs = []
    for addr in sorted(set(diffs_swap.keys()) & set(diffs_swap2.keys())):
        o1, n1 = diffs_swap[addr]
        o2, n2 = diffs_swap2[addr]
        if o1 == n2 and n1 == o2:
            reversed_addrs.append((addr, o1, n1))

    log(f"\n  Perfectly reversed on 2nd L swap: {len(reversed_addrs)}")
    piece_reversed = [(a, o, n) for a, o, n in reversed_addrs if 0 <= o <= 6 and 0 <= n <= 6]
    log(f"  Of those, with values 0-6: {len(piece_reversed)}")
    for addr, old, new in piece_reversed[:30]:
        dbg_val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {old} <-> {new}  [debug now={dbg_val}]")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Next piece tracking across drops
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Next Piece Tracking")
    log("=" * 60)

    # Gather all interesting addresses from experiments above
    interesting = set()
    for addr, _, _ in y_filtered[:5]:
        interesting.add(addr)
    for addr, _, _ in x_filtered[:5]:
        interesting.add(addr)
    for addr, _, _ in rot_filtered[:5]:
        interesting.add(addr)
    for addr, _, _, _, _ in type_filtered[:10]:
        interesting.add(addr)
    for addr, _, _ in piece_reversed[:10]:
        interesting.add(addr)

    interesting = sorted(interesting)
    log(f"\n>>> Tracking {len(interesting)} interesting addresses across 5 drops")

    snap_q = reload_save(core, memory)

    # Show initial values
    log("\n  Initial values:")
    for addr in interesting:
        val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {val}")

    for drop_num in range(1, 6):
        core.resume()
        press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
        snapshot_paused(core, memory)

        log(f"\n  After drop {drop_num}:")
        for addr in interesting:
            val = core.debug_read_8(addr)
            log(f"    0x{addr:08X}: {val}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Board grid layout discovery
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Board Grid Layout")
    log("=" * 60)

    # Drop piece at center (default position — no movement)
    snap_center = reload_save(core, memory)
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after_center = snapshot_paused(core, memory)
    diffs_center = diff_snapshots(snap_center, snap_after_center)
    center_cells = {a for a, (o, n) in diffs_center.items() if o == 0 and n != 0}

    # Drop piece at far left
    snap_left = reload_save(core, memory)
    core.resume()
    time.sleep(0.1)
    for _ in range(5):
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.08)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after_left = snapshot_paused(core, memory)
    diffs_left = diff_snapshots(snap_left, snap_after_left)
    left_cells = {a for a, (o, n) in diffs_left.items() if o == 0 and n != 0}

    # Drop piece at far right
    snap_right = reload_save(core, memory)
    core.resume()
    time.sleep(0.1)
    for _ in range(5):
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.08)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    snap_after_right = snapshot_paused(core, memory)
    diffs_right = diff_snapshots(snap_right, snap_after_right)
    right_cells = {a for a, (o, n) in diffs_right.items() if o == 0 and n != 0}

    log(f"\n  Center drop 0->nonzero: {len(center_cells)}")
    for addr in sorted(center_cells)[:20]:
        _, n = diffs_center[addr]
        log(f"    0x{addr:08X}: 0->0x{n:02X}")

    log(f"\n  Left drop 0->nonzero: {len(left_cells)}")
    for addr in sorted(left_cells)[:20]:
        _, n = diffs_left[addr]
        log(f"    0x{addr:08X}: 0->0x{n:02X}")

    log(f"\n  Right drop 0->nonzero: {len(right_cells)}")
    for addr in sorted(right_cells)[:20]:
        _, n = diffs_right[addr]
        log(f"    0x{addr:08X}: 0->0x{n:02X}")

    # Compare center vs left vs right to find column stride
    all_cells = sorted(center_cells | left_cells | right_cells)
    log(f"\n  All board cell addresses ({len(all_cells)}):")
    for addr in all_cells[:40]:
        in_c = "C" if addr in center_cells else " "
        in_l = "L" if addr in left_cells else " "
        in_r = "R" if addr in right_cells else " "
        # Get the value
        val = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: [{in_l}{in_c}{in_r}] val={val}")

    if len(all_cells) >= 2:
        strides = [all_cells[i + 1] - all_cells[i] for i in range(len(all_cells) - 1)]
        stride_freq = Counter(strides)
        log(f"\n  Combined stride frequency:")
        for stride, count in stride_freq.most_common(10):
            log(f"    stride {stride} (0x{stride:X}): {count}x")

    # ── Dump neighborhood of best board candidates ────────────
    if all_cells:
        best = all_cells[0]
        log(f"\n>>> Hex dump around first board cell 0x{best:08X}:")
        hexdump_debug(core, best - 16, 64)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 6: Wider piece region dump
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 6: Piece Region Dumps")
    log("=" * 60)

    reload_save(core, memory)

    # Dump the neighborhood around piece type candidates
    if type_filtered:
        best_type = type_filtered[0][0]
        log(f"\n>>> Around best piece type candidate 0x{best_type:08X}:")
        hexdump_debug(core, (best_type & ~0xF) - 32, 96)

    if piece_reversed:
        for addr, old, new in piece_reversed[:3]:
            log(f"\n>>> Around reversed address 0x{addr:08X} ({old}<->{new}):")
            hexdump_debug(core, (addr & ~0xF) - 16, 64)

    # ── Score verification ────────────────────────────────────
    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v5 COMPLETE")
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
