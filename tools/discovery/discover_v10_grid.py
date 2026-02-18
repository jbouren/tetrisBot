#!/usr/bin/env python3
"""Discovery v10 — Confirm X, map board grid, piece types.

GOALS:
1. Confirm X position at 0x8010BD64 (track across multiple left/right moves)
2. Map the 0x9C-stride board records to 10x20 grid
3. Identify piece type numbers by dropping at known positions
4. Find next piece / reserve piece via L-swap
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


def press_button(input_server, button, hold_sec=0.05, pause_sec=0.15):
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


def read_word(core, addr):
    return core.debug_read_32(addr)


def read_float(core, addr):
    raw = core.debug_read_32(addr)
    return struct.unpack('f', struct.pack('I', raw))[0]


def main():
    log("=" * 60)
    log("Discovery v10 — Confirm X, Board Grid, Piece Types")
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
    # EXPERIMENT 1: Confirm X position at 0x8010BD64
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Confirm X position")
    log("=" * 60)

    # Also track nearby addresses and the piece struct coords
    x_candidates = [0x8010BD64, 0x8010BC24, 0x8010BD2C]
    coord_addrs = list(range(0x8010BBF8, 0x8010BC18, 4))  # 8 coord values

    reload_save(core)

    log("\n>>> Initial state:")
    for addr in x_candidates:
        log(f"  0x{addr:08X} = {read_word(core, addr)}")
    log(f"  Coords: {[read_word(core, a) for a in coord_addrs]}")
    log(f"  Y float: {read_float(core, 0x800D02CC):.2f}")
    log(f"  Piece BBEC: {read_word(core, 0x8010BBEC)}")
    log(f"  Piece DFFC: {read_word(core, 0x8010DFFC)}")

    # Move RIGHT 5 times, minimal fall
    log("\n>>> Moving RIGHT 5 times:")
    for i in range(5):
        core.resume()
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
        core.pause()
        time.sleep(0.05)
        vals = {hex(a): read_word(core, a) for a in x_candidates}
        coords = [read_word(core, a) for a in coord_addrs]
        log(f"  Right {i+1}: BD64={vals['0x8010bd64']}  BC24={vals['0x8010bc24']}  BD2C={vals['0x8010bd2c']}  coords={coords}")

    # Reload and move LEFT 5 times
    reload_save(core)
    log("\n>>> Moving LEFT 5 times:")
    init_bd64 = read_word(core, 0x8010BD64)
    log(f"  Initial BD64={init_bd64}")
    for i in range(5):
        core.resume()
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.05)
        core.pause()
        time.sleep(0.05)
        vals = {hex(a): read_word(core, a) for a in x_candidates}
        coords = [read_word(core, a) for a in coord_addrs]
        log(f"  Left {i+1}: BD64={vals['0x8010bd64']}  BC24={vals['0x8010bc24']}  BD2C={vals['0x8010bd2c']}  coords={coords}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Map board grid — drop pieces at known columns
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Map board grid via 0x9C-stride records")
    log("=" * 60)

    # The 0x9C-stride records seem to start near 0x800D10A9
    # 200 records (10 cols x 20 rows) at 0x9C stride = 0x7A80 bytes
    # Potential base addresses to check
    # First, let's find the exact base by reading the occupancy byte
    # We know 0x800D10B9 got value 1 on left-drop
    # Offset within record: 0x800D10B9 - record_base
    # If record 0 starts at 0x800D10A9, then offset = 0x10

    RECORD_STRIDE = 0x9C  # 156 bytes
    OCCUPANCY_OFFSET = 0x10  # byte within each record that holds occupancy
    # But let's verify this offset by checking a few records

    # First, dump the first record fully before and after a drop
    reload_save(core)
    memory.refresh_pointer()
    snap_empty = memory.snapshot()

    # Drop at center (no moves, just hard drop)
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    snap_drop1 = memory.snapshot()

    # Find ALL bytes that changed in the 0x800D1000-0x800D9000 range (board area)
    BOARD_SCAN_START = 0x0D1000  # physical
    BOARD_SCAN_END = 0x0D9000
    changed_bytes = {}
    for phys in range(BOARD_SCAN_START, BOARD_SCAN_END):
        if snap_empty[phys] != snap_drop1[phys]:
            virt = 0x80000000 + (phys ^ 3)
            changed_bytes[virt] = (snap_empty[phys], snap_drop1[phys])

    log(f"\n>>> Bytes changed in 0x800D1000-0x800D9000 after center drop: {len(changed_bytes)}")

    # Group by record (assuming base 0x800D10A9, stride 0x9C)
    RECORD_BASE = 0x800D10A9
    records_changed = {}
    for virt, (old, new) in sorted(changed_bytes.items()):
        offset_from_base = virt - RECORD_BASE
        if offset_from_base >= 0:
            record_idx = offset_from_base // RECORD_STRIDE
            byte_offset = offset_from_base % RECORD_STRIDE
            if record_idx not in records_changed:
                records_changed[record_idx] = []
            records_changed[record_idx].append((byte_offset, old, new, virt))

    log(f"\n>>> Records that changed (base=0x{RECORD_BASE:08X}, stride=0x{RECORD_STRIDE:X}):")
    for rec_idx in sorted(records_changed.keys())[:30]:
        changes = records_changed[rec_idx]
        rec_addr = RECORD_BASE + rec_idx * RECORD_STRIDE
        log(f"  Record {rec_idx:3d} (0x{rec_addr:08X}):")
        for byte_off, old, new, virt in changes:
            log(f"    offset +0x{byte_off:02X}: {old} -> {new}  (0x{virt:08X})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Determine board layout — drop left, center, right
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Board layout — which records for which columns?")
    log("=" * 60)

    def get_changed_records(snap_before, snap_after):
        """Return set of record indices that changed."""
        changed = set()
        for phys in range(BOARD_SCAN_START, BOARD_SCAN_END):
            if snap_before[phys] != snap_after[phys]:
                virt = 0x80000000 + (phys ^ 3)
                offset = virt - RECORD_BASE
                if offset >= 0:
                    rec = offset // RECORD_STRIDE
                    if 0 <= rec < 200:
                        changed.add(rec)
        return changed

    # Drop at far left
    reload_save(core)
    memory.refresh_pointer()
    snap_a = memory.snapshot()
    core.resume()
    time.sleep(0.05)
    for _ in range(5):
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    snap_b = memory.snapshot()
    left_recs = get_changed_records(snap_a, snap_b)

    # Note what piece and X were
    piece_left = read_word(core, 0x8010DFFC)

    # Drop at center (no moves)
    reload_save(core)
    memory.refresh_pointer()
    snap_a = memory.snapshot()
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    snap_b = memory.snapshot()
    center_recs = get_changed_records(snap_a, snap_b)
    piece_center = read_word(core, 0x8010DFFC)

    # Drop at far right
    reload_save(core)
    memory.refresh_pointer()
    snap_a = memory.snapshot()
    core.resume()
    time.sleep(0.05)
    for _ in range(5):
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    snap_b = memory.snapshot()
    right_recs = get_changed_records(snap_a, snap_b)
    piece_right = read_word(core, 0x8010DFFC)

    log(f"\n  Left drop records  (piece={piece_left}): {sorted(left_recs)}")
    log(f"  Center drop records (piece={piece_center}): {sorted(center_recs)}")
    log(f"  Right drop records (piece={piece_right}): {sorted(right_recs)}")

    # Analyze: if row-major (10 cols per row), record N = row(N//10), col(N%10)
    # if col-major (20 rows per col), record N = col(N//20), row(N%20)
    log("\n  If ROW-MAJOR (10 cols/row):")
    log(f"    Left:   {[(r//10, r%10) for r in sorted(left_recs)]}")
    log(f"    Center: {[(r//10, r%10) for r in sorted(center_recs)]}")
    log(f"    Right:  {[(r//10, r%10) for r in sorted(right_recs)]}")

    log("\n  If COL-MAJOR (20 rows/col):")
    log(f"    Left:   col,row = {[(r//20, r%20) for r in sorted(left_recs)]}")
    log(f"    Center: col,row = {[(r//20, r%20) for r in sorted(center_recs)]}")
    log(f"    Right:  col,row = {[(r//20, r%20) for r in sorted(right_recs)]}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Next piece & reserve piece (L-swap)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Next piece & reserve (L-swap)")
    log("=" * 60)

    reload_save(core)

    # Read all piece-related addresses
    piece_addrs = {
        "BBEC": 0x8010BBEC,
        "BC24": 0x8010BC24,
        "BD64": 0x8010BD64,
        "DFFC": 0x8010DFFC,
        "E0A4": 0x8010E0A4,
    }

    # Also scan wider for more piece fields
    wider_piece = list(range(0x8010DFC0, 0x8010E100, 4))

    log("\n>>> Before any action:")
    for name, addr in piece_addrs.items():
        log(f"  {name}: {read_word(core, addr)}")
    log(f"  Wider: {[(hex(a), read_word(core, a)) for a in wider_piece if read_word(core, a) != 0]}")

    # L-swap (swap current with reserve)
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.3)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After L-swap:")
    for name, addr in piece_addrs.items():
        log(f"  {name}: {read_word(core, addr)}")
    log(f"  Wider: {[(hex(a), read_word(core, a)) for a in wider_piece if read_word(core, a) != 0]}")

    # L-swap again (should swap back)
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.3)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After 2nd L-swap (should restore):")
    for name, addr in piece_addrs.items():
        log(f"  {name}: {read_word(core, addr)}")
    log(f"  Wider: {[(hex(a), read_word(core, a)) for a in wider_piece if read_word(core, a) != 0]}")

    # Now do a full RDRAM diff for L-swap to find reserve piece address
    reload_save(core)
    memory.refresh_pointer()
    snap_pre_swap = memory.snapshot()

    core.resume()
    time.sleep(0.05)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.3)
    core.pause()
    time.sleep(0.1)
    snap_post_swap = memory.snapshot()

    # Find u32 words that swapped values (current piece value and reserve value exchanged)
    SCAN_START = 0x080000
    SCAN_END = 0x200000
    swap_candidates = []
    for phys in range(SCAN_START, SCAN_END, 4):
        wa = struct.unpack_from('<I', snap_pre_swap, phys)[0]
        wb = struct.unpack_from('<I', snap_post_swap, phys)[0]
        if wa != wb and 0 <= wa <= 7 and 0 <= wb <= 7:
            addr = 0x80000000 + phys
            swap_candidates.append((addr, wa, wb))

    log(f"\n>>> Words with values 0-7 that changed on L-swap: {len(swap_candidates)}")
    for addr, old, new in swap_candidates[:30]:
        log(f"  0x{addr:08X}: {old} -> {new}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Piece type identification
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Piece type identification")
    log("=" * 60)

    # Read the piece type at save state, then look at the block coords
    # to figure out what shape it is
    reload_save(core)

    piece_val = read_word(core, 0x8010DFFC)
    coords = [read_word(core, a) for a in coord_addrs]
    log(f"\n>>> Piece value: {piece_val}")
    log(f"  Block coords (8 values): {coords}")

    # The 8 coords might be 4 (x,y) pairs
    pairs = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    log(f"  As (x,y) pairs: {pairs}")

    # Normalize to relative positions
    min_x = min(p[0] for p in pairs)
    min_y = min(p[1] for p in pairs)
    rel = [(p[0]-min_x, p[1]-min_y) for p in pairs]
    log(f"  Relative positions: {rel}")

    # Now rotate and check each rotation
    log("\n>>> Rotating 4 times to see all orientations:")
    for rot in range(4):
        core.resume()
        time.sleep(0.02)
        press_button(input_server, "A_BUTTON", hold_sec=0.05, pause_sec=0.1)
        core.pause()
        time.sleep(0.05)

        coords = [read_word(core, a) for a in coord_addrs]
        pairs = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        min_x = min(p[0] for p in pairs)
        min_y = min(p[1] for p in pairs)
        rel = sorted([(p[0]-min_x, p[1]-min_y) for p in pairs])
        log(f"  Rotation {rot+1}: coords={coords}  relative={rel}")

    # Drop and check the NEXT piece
    reload_save(core)
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    piece_val2 = read_word(core, 0x8010DFFC)
    coords2 = [read_word(core, a) for a in coord_addrs]
    pairs2 = [(coords2[i], coords2[i+1]) for i in range(0, 8, 2)]
    min_x2 = min(p[0] for p in pairs2)
    min_y2 = min(p[1] for p in pairs2)
    rel2 = sorted([(p[0]-min_x2, p[1]-min_y2) for p in pairs2])
    log(f"\n>>> After drop — next piece value: {piece_val2}")
    log(f"  Block coords: {coords2}  relative={rel2}")

    # Drop again for 3rd piece
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    piece_val3 = read_word(core, 0x8010DFFC)
    coords3 = [read_word(core, a) for a in coord_addrs]
    pairs3 = [(coords3[i], coords3[i+1]) for i in range(0, 8, 2)]
    min_x3 = min(p[0] for p in pairs3)
    min_y3 = min(p[1] for p in pairs3)
    rel3 = sorted([(p[0]-min_x3, p[1]-min_y3) for p in pairs3])
    log(f">>> 3rd piece value: {piece_val3}")
    log(f"  Block coords: {coords3}  relative={rel3}")

    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v10 COMPLETE")
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
