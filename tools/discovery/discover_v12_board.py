#!/usr/bin/env python3
"""Discovery v12 — Find board data using multiple strategies.

Key insight from v11: The 0x9C-stride records 0-7 are the ACTIVE piece display,
NOT placed board cells. Records 47+ are UI elements. The actual board grid
data is stored elsewhere.

Strategies:
1. Check if piece struct coords (666-668) are indices into the 0x9C record array
   - Record 666 would be at 0x800D10A9 + 666*0x9C = 0x800EA6B1
   - We were scanning too narrow a range before!
2. Search for board as 20 x u16 bitmask (each row = 10 bits)
3. Full RDRAM diff between empty and 1-piece boards, looking for exactly 4 cells
4. Follow more board struct pointers (0x8010BAD0)
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
    log("Discovery v12 — Board Data Hunt")
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
    # EXPERIMENT 1: Check piece struct indices as record array indices
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Piece coord indices in 0x9C record array")
    log("=" * 60)

    reload_save(core)

    # Read the current piece's block indices
    coord_addrs = list(range(0x8010BBF8, 0x8010BC18, 4))
    coords = [read_word(core, a) for a in coord_addrs]
    log(f"\n>>> Piece block indices: {coords}")

    # Calculate record addresses for these indices
    for idx in sorted(set(coords)):
        rec_addr = RECORD_BASE + idx * RECORD_STRIDE
        log(f"\n>>> Record at index {idx} (0x{rec_addr:08X}):")
        hexdump_debug(core, rec_addr, RECORD_STRIDE)

    # Now hard-drop and check if these records change
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    # Re-read the same indices
    log("\n>>> After drop, same record indices:")
    for idx in sorted(set(coords)):
        rec_addr = RECORD_BASE + idx * RECORD_STRIDE
        occ_addr = rec_addr + 0x10
        occ_val = core.debug_read_8(occ_addr)
        log(f"  Index {idx}: occupancy byte at +0x10 = {occ_val}")

    # Also read new piece's indices
    coords_new = [read_word(core, a) for a in coord_addrs]
    log(f"\n>>> New piece indices: {coords_new}")

    # Check where the landed piece went
    # The piece drops to the bottom. The Y is tracked as float.
    # But the record indices might change to reflect the landed position
    log("\n>>> Board struct area after drop:")
    # Check 0x8012E334 (changed 0→6 in v8 after drop)
    log(f"  0x8012E334 = {read_word(core, 0x8012E334)}")
    log(f"  0x8012F3D0+0x0C = {read_word(core, 0x8012F3DC)}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Search for bitmask board (20 rows x 10-bit)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Bitmask board search")
    log("=" * 60)

    # Drop piece at far left (columns 0-1 for O piece or 0-3 for I piece)
    reload_save(core)
    memory.refresh_pointer()
    snap_empty = memory.snapshot()

    # Drop at center (no moves)
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    snap_1piece = memory.snapshot()

    piece_type = read_word(core, 0x8010DFFC)
    log(f"\n>>> Dropped piece type: {piece_type}")

    # Search for u32 words that went from 0 to a value with 2-4 bits set
    # (tetromino occupying 2-4 cells in a single row)
    SCAN_START = 0x080000
    SCAN_END = 0x200000

    bitmask_candidates = []
    for phys in range(SCAN_START, SCAN_END, 4):
        wa = struct.unpack_from('<I', snap_empty, phys)[0]
        wb = struct.unpack_from('<I', snap_1piece, phys)[0]
        if wa == 0 and wb != 0:
            bits = bin(wb).count('1')
            if 2 <= bits <= 4 and wb < 0x400:  # 10-bit max for 10 columns
                addr = 0x80000000 + phys
                bitmask_candidates.append((addr, wb, bits))

    log(f"\n>>> u32 words: 0 -> small bitmask (2-4 bits, <0x400): {len(bitmask_candidates)}")
    for addr, val, bits in sorted(bitmask_candidates):
        log(f"  0x{addr:08X}: 0x{val:04X} = {bin(val)} ({bits} bits)")

    # Also search u16
    u16_bitmask = []
    for phys in range(SCAN_START, SCAN_END, 2):
        va = struct.unpack_from('<H', snap_empty, phys)[0]
        vb = struct.unpack_from('<H', snap_1piece, phys)[0]
        if va == 0 and vb != 0:
            bits = bin(vb).count('1')
            if 2 <= bits <= 4 and vb < 0x400:
                addr = 0x80000000 + (phys ^ 2)  # XOR^2 for u16
                u16_bitmask.append((addr, vb, bits))

    log(f"\n>>> u16 words: 0 -> small bitmask (2-4 bits, <0x400): {len(u16_bitmask)}")
    # Filter to look for grouped addresses (multiple rows close together)
    if u16_bitmask:
        for addr, val, bits in sorted(u16_bitmask)[:30]:
            log(f"  0x{addr:08X}: 0x{val:04X} = {bin(val)} ({bits} bits)")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Full diff — groups of exactly 4 byte changes
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Find groups of exactly 4 changed bytes")
    log("=" * 60)

    # Drop at specific position for known layout
    reload_save(core)
    memory.refresh_pointer()
    snap_a = memory.snapshot()

    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    snap_b = memory.snapshot()

    # Find all changed bytes
    all_changes = []
    for phys in range(SCAN_START, SCAN_END):
        if snap_a[phys] != snap_b[phys]:
            virt = 0x80000000 + (phys ^ 3)
            all_changes.append((phys, virt, snap_a[phys], snap_b[phys]))

    log(f"\n>>> Total bytes changed: {len(all_changes)}")

    # Now drop 2nd piece
    snap_c = snap_b
    core.resume()
    time.sleep(0.05)
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    snap_d = memory.snapshot()

    # Changes between 1-piece and 2-pieces states
    piece2_changes = []
    for phys in range(SCAN_START, SCAN_END):
        if snap_c[phys] != snap_d[phys]:
            virt = 0x80000000 + (phys ^ 3)
            piece2_changes.append((phys, virt, snap_c[phys], snap_d[phys]))

    log(f">>> Bytes changed between 1-piece and 2-pieces: {len(piece2_changes)}")

    # Find bytes that ONLY changed in piece2 (not in piece1)
    piece1_phys = set(p for p, _, _, _ in all_changes)
    piece2_only = [(p, v, o, n) for p, v, o, n in piece2_changes if p not in piece1_phys]
    log(f">>> Bytes changed ONLY in 2nd drop (new board cells?): {len(piece2_only)}")

    # Group nearby changes
    if piece2_only:
        # Sort by physical address and find clusters
        piece2_only.sort()
        clusters = []
        current_cluster = [piece2_only[0]]
        for i in range(1, len(piece2_only)):
            if piece2_only[i][0] - current_cluster[-1][0] < 200:
                current_cluster.append(piece2_only[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [piece2_only[i]]
        clusters.append(current_cluster)

        log(f"\n>>> Clusters of nearby changes:")
        for cluster in clusters:
            if len(cluster) >= 3:  # At least 3 bytes in cluster
                log(f"\n  Cluster at 0x{cluster[0][1]:08X}-0x{cluster[-1][1]:08X} ({len(cluster)} bytes):")
                for phys, virt, old, new in cluster[:20]:
                    log(f"    0x{virt:08X} (phys 0x{phys:06X}): {old} -> {new}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Follow pointer 0x8010BAD0 (from board struct)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Follow remaining board pointers")
    log("=" * 60)

    reload_save(core)

    for ptr_addr_name, ptr_addr in [
        ("0x8010BAD0 (from D050+0x08)", 0x8010BAD0),
        ("0x80103520 (from D050+0x0C)", 0x80103520),
        ("0x80107608 (from F348+0x00)", 0x80107608),
        ("0x8010764C (from F348+0x04)", 0x8010764C),
    ]:
        log(f"\n>>> {ptr_addr_name}:")
        hexdump_debug(core, ptr_addr, 0x80)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Search for X position via debug_read approach
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: X position — targeted word search")
    log("=" * 60)

    reload_save(core)

    # Read a wide range of u32 values near known addresses
    # Read state BEFORE any move
    pre_move = {}
    # Scan piece struct area and nearby
    for addr in range(0x8010BB00, 0x8010BE00, 4):
        pre_move[addr] = read_word(core, addr)
    # Also scan 0x800D0200-0x800D0400
    for addr in range(0x800D0200, 0x800D0400, 4):
        pre_move[addr] = read_word(core, addr)
    # And board struct area
    for addr in range(0x8012E200, 0x8012E400, 4):
        pre_move[addr] = read_word(core, addr)
    # And 0x80126700-0x80126A00 (L-swap area from v10)
    for addr in range(0x80126700, 0x80126A00, 4):
        pre_move[addr] = read_word(core, addr)

    # Single quick right move
    core.resume()
    time.sleep(0.01)
    press_button(input_server, "R_DPAD", hold_sec=0.04, pause_sec=0.01)
    core.pause()
    time.sleep(0.05)

    post_move_r = {}
    for addr in pre_move:
        post_move_r[addr] = read_word(core, addr)

    log("\n>>> Words changed on RIGHT move (focused areas):")
    for addr in sorted(pre_move.keys()):
        if pre_move[addr] != post_move_r[addr]:
            old, new = pre_move[addr], post_move_r[addr]
            delta = new - old
            log(f"  0x{addr:08X}: {old} -> {new} (delta={delta})")

    # Now left move from same state
    pre_left = post_move_r.copy()
    for addr in pre_left:
        pre_left[addr] = read_word(core, addr)

    core.resume()
    time.sleep(0.01)
    press_button(input_server, "L_DPAD", hold_sec=0.04, pause_sec=0.01)
    core.pause()
    time.sleep(0.05)

    post_move_l = {}
    for addr in pre_left:
        post_move_l[addr] = read_word(core, addr)

    log("\n>>> Words changed on LEFT move:")
    for addr in sorted(pre_left.keys()):
        if pre_left[addr] != post_move_l[addr]:
            old, new = pre_left[addr], post_move_l[addr]
            delta = new - old
            log(f"  0x{addr:08X}: {old} -> {new} (delta={delta})")

    # Identify addresses that moved +N on right and -N on left (or similar)
    log("\n>>> Addresses with opposite deltas (right=+D, left=-D):")
    for addr in sorted(pre_move.keys()):
        if addr in post_move_r and addr in pre_left and addr in post_move_l:
            r_delta = post_move_r[addr] - pre_move[addr]
            l_delta = post_move_l[addr] - pre_left[addr]
            if r_delta != 0 and l_delta != 0 and r_delta * l_delta < 0:
                log(f"  0x{addr:08X}: right_delta={r_delta:+d}, left_delta={l_delta:+d}")

    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v12 COMPLETE")
    log("=" * 60)

    try:
        core.stop()
    except:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
