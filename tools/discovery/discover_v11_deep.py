#!/usr/bin/env python3
"""Discovery v11 — Deep board analysis + X float search.

Key insights from v10:
- Records 0-7 at 0x9C stride always change on first drop (pool allocation)
- Need to drop multiple pieces to see which records accumulate
- X position needs a float search over wider range
- Block coords in piece struct are not simple (x,y) grid pairs
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


def hexdump_debug(core, start_virt, count):
    data = [core.debug_read_8(start_virt + i) for i in range(count)]
    for i in range(0, len(data), 16):
        row = data[i:i + 16]
        addr = start_virt + i
        hex_str = " ".join(f"{b:02X}" for b in row)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        log(f"  {addr:08X}: {hex_str:<48} |{ascii_str}|")
    return data


def main():
    log("=" * 60)
    log("Discovery v11 — Deep Board + X Float Search")
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

    RECORD_BASE = 0x800D10A9
    RECORD_STRIDE = 0x9C

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 1: Drop 5 pieces sequentially, track record allocation
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Sequential drops — track record allocation")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()

    # Take snapshot of just the board area occupancy bytes
    # Check offset +0x10 of each record (confirmed as occupancy byte)
    def read_all_records(core):
        """Read occupancy byte (offset +0x10) for all 200 records."""
        vals = []
        for i in range(200):
            addr = RECORD_BASE + i * RECORD_STRIDE + 0x10
            vals.append(core.debug_read_8(addr))
        return vals

    initial_occ = read_all_records(core)
    log(f"\n>>> Initial occupancy: non-zero records: {[(i, v) for i, v in enumerate(initial_occ) if v != 0]}")

    # Drop 5 pieces at center (no left/right moves), tracking after each
    all_drops_occ = [initial_occ]
    for drop_num in range(5):
        core.resume()
        time.sleep(0.05)
        press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
        core.pause()
        time.sleep(0.1)

        occ = read_all_records(core)
        piece = read_word(core, 0x8010DFFC)
        score = core.debug_read_16(0x8011EED6)
        nonzero = [(i, v) for i, v in enumerate(occ) if v != 0]
        log(f"\n>>> After drop {drop_num + 1} (piece={piece}, score={score}):")
        log(f"  Non-zero records: {nonzero}")

        # Show NEW records (changed from previous)
        prev = all_drops_occ[-1]
        new_records = [(i, occ[i]) for i in range(200) if occ[i] != prev[i]]
        log(f"  Newly changed: {new_records}")

        all_drops_occ.append(occ)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Dump full record structure for first placed block
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Full record dump (records 0-3)")
    log("=" * 60)

    # Records should be populated from the drops above
    for rec_idx in range(4):
        rec_addr = RECORD_BASE + rec_idx * RECORD_STRIDE
        log(f"\n>>> Record {rec_idx} at 0x{rec_addr:08X}:")
        hexdump_debug(core, rec_addr, RECORD_STRIDE)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Look for grid position within records
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Find grid position within records")
    log("=" * 60)

    # Drop at far-left, check which records and their row/col fields
    reload_save(core)
    memory.refresh_pointer()

    # First, drop at far left
    core.resume()
    time.sleep(0.05)
    for _ in range(5):
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    left_occ = read_all_records(core)
    left_nonzero = [(i, v) for i, v in enumerate(left_occ) if v != 0]
    log(f"\n>>> After left drop: {left_nonzero}")

    # Dump first few occupied records fully
    for rec_idx, val in left_nonzero[:4]:
        rec_addr = RECORD_BASE + rec_idx * RECORD_STRIDE
        log(f"\n>>> Left-drop Record {rec_idx} (occ={val}) at 0x{rec_addr:08X}:")
        hexdump_debug(core, rec_addr, RECORD_STRIDE)

    # Now reload and drop at far right
    reload_save(core)
    core.resume()
    time.sleep(0.05)
    for _ in range(5):
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    right_occ = read_all_records(core)
    right_nonzero = [(i, v) for i, v in enumerate(right_occ) if v != 0]
    log(f"\n>>> After right drop: {right_nonzero}")

    for rec_idx, val in right_nonzero[:4]:
        rec_addr = RECORD_BASE + rec_idx * RECORD_STRIDE
        log(f"\n>>> Right-drop Record {rec_idx} (occ={val}) at 0x{rec_addr:08X}:")
        hexdump_debug(core, rec_addr, RECORD_STRIDE)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: X position as float — wide search
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: X position float search")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()

    # Snapshot before move
    snap_before = memory.snapshot()

    # Single fast right move
    core.resume()
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.02)
    core.pause()
    time.sleep(0.05)
    snap_after = memory.snapshot()

    # Find ALL float-valued words that changed
    # A right move should increase X float by some consistent delta
    SCAN_START = 0x080000
    SCAN_END = 0x200000
    float_changes = []
    for phys in range(SCAN_START, SCAN_END, 4):
        wa = struct.unpack_from('<I', snap_before, phys)[0]
        wb = struct.unpack_from('<I', snap_after, phys)[0]
        if wa == wb:
            continue
        fa = struct.unpack('f', struct.pack('I', wa))[0]
        fb = struct.unpack('f', struct.pack('I', wb))[0]
        # Look for reasonable float changes (small positive delta for right move)
        if not (abs(fa) < 10000 and abs(fb) < 10000):
            continue
        delta = fb - fa
        if 0.1 < delta < 20.0:  # Positive delta for right move
            addr = 0x80000000 + phys
            float_changes.append((addr, fa, fb, delta))

    log(f"\n>>> Floats with positive delta (0.1-20) on right move: {len(float_changes)}")
    for addr, old_f, new_f, delta in sorted(float_changes, key=lambda x: x[3]):
        log(f"  0x{addr:08X}: {old_f:.4f} -> {new_f:.4f} (delta={delta:.4f})")

    # Also check: second right move should give same delta
    snap_before2 = snap_after
    core.resume()
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.02)
    core.pause()
    time.sleep(0.05)
    snap_after2 = memory.snapshot()

    # For each candidate, verify consistent delta
    log("\n>>> Verifying consistent delta on 2nd right move:")
    for addr, old_f, new_f, delta in float_changes:
        phys = (addr & 0x1FFFFFFF)
        wb2 = struct.unpack_from('<I', snap_after2, phys)[0]
        fb2 = struct.unpack('f', struct.pack('I', wb2))[0]
        delta2 = fb2 - new_f
        if abs(delta2 - delta) < 0.5:  # Similar delta
            log(f"  0x{addr:08X}: {old_f:.4f} -> {new_f:.4f} -> {fb2:.4f}  (deltas: {delta:.4f}, {delta2:.4f})  CONSISTENT")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Search for X as u16 or i16
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: X as u16/i16 search")
    log("=" * 60)

    # Maybe X is stored as a 16-bit value within a struct
    reload_save(core)
    memory.refresh_pointer()
    snap_a = memory.snapshot()

    # Move right once
    core.resume()
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.02)
    core.pause()
    time.sleep(0.05)
    snap_b = memory.snapshot()

    # Search for u16 values that increased by exactly 1
    u16_candidates = []
    for phys in range(SCAN_START, SCAN_END, 2):
        va = struct.unpack_from('<H', snap_a, phys)[0]
        vb = struct.unpack_from('<H', snap_b, phys)[0]
        if vb - va == 1 and 0 <= va <= 20:  # Small value that incremented by 1
            addr_raw = 0x80000000 + phys
            # Apply XOR^2 for 16-bit access
            addr_corrected = 0x80000000 + (phys ^ 2)
            u16_candidates.append((addr_corrected, va, vb))

    log(f"\n>>> u16 values that increased by exactly 1 on right move (value 0-20): {len(u16_candidates)}")
    for addr, old, new in u16_candidates[:20]:
        log(f"  0x{addr:08X}: {old} -> {new}")

    # Also try negative (for left move effect)
    # Move left from current position
    snap_c = snap_b
    core.resume()
    press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.02)
    core.pause()
    time.sleep(0.05)
    snap_d = memory.snapshot()

    # Check which of our u16 candidates decreased by 1
    log("\n>>> Of those, which decreased by 1 on left move:")
    for addr, _, _ in u16_candidates:
        phys = (addr & 0x1FFFFFFF) ^ 2  # reverse XOR^2
        vc = struct.unpack_from('<H', snap_c, phys)[0]
        vd = struct.unpack_from('<H', snap_d, phys)[0]
        if vc - vd == 1:
            log(f"  0x{addr:08X}: {vc} -> {vd}  *** CONFIRMED X CANDIDATE ***")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 6: Explore board struct at 0x8012E220 more deeply
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 6: Board struct deep dive")
    log("=" * 60)

    reload_save(core)

    # Dump a wider range around the board struct
    log("\n>>> Board struct at 0x8012E220 (0x200 bytes):")
    hexdump_debug(core, 0x8012E220, 0x200)

    # Follow ALL pointers in the board struct
    log("\n>>> Following pointers in board struct:")
    for offset in range(0, 0x120, 4):
        addr = 0x8012E220 + offset
        val = read_word(core, addr)
        if 0x80000000 <= val < 0x80800000 and val != addr:
            # It's a pointer — read what it points to
            log(f"\n  0x{addr:08X} (+0x{offset:02X}) -> 0x{val:08X}:")
            # Read first 32 bytes at pointer target
            for i in range(0, 32, 4):
                w = read_word(core, val + i)
                log(f"    +0x{i:02X}: 0x{w:08X} ({w})")

    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v11 COMPLETE")
    log("=" * 60)

    try:
        core.stop()
    except:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
