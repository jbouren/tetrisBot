#!/usr/bin/env python3
"""Discovery v16 — Board via proportional accumulation + piece swap + X revisited.

KEY INSIGHTS from v15:
- 0x801267E4 is NOT X (erratic across range test)
- Board not stored as 0->nonzero cells
- 0x801269B4 changed 7->2 after drop (piece type?)
- Neighborhood 0x801267C0-0x80126A00 has piece-related data

NEW STRATEGIES:
1. Board: Take snapshots after 0,1,2,3,4,5 drops. Find regions where
   # of changed bytes grows proportionally (~4 per piece).
2. Reserve piece: Press L (swap), see what changes.
3. X position: Search more broadly — maybe it's a signed i8/i16 or a float
   stored somewhere we haven't looked.
4. Piece type mapping: Drop pieces at center, read block coords + DFFC to map types.
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
    set_button(input_server, button)
    advance_n(core, hold)
    set_button(input_server, None)
    input_server.clear()
    advance_n(core, gap)


def drop_piece_realtime(core, input_server):
    """Drop a piece using real-time control (reliable)."""
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


def main():
    log("=" * 60)
    log("Discovery v16 — Board Accumulation + Piece Swap")
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
    # EXPERIMENT 1: Proportional board accumulation
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Proportional board accumulation (6 drops)")
    log("=" * 60)

    SCAN_START = 0x080000
    SCAN_END = 0x200000

    reload_save(core)
    memory.refresh_pointer()
    snap_base = memory.snapshot()

    snapshots = [snap_base]
    for drop_num in range(6):
        drop_piece_realtime(core, input_server)
        memory.refresh_pointer()
        snap = memory.snapshot()
        snapshots.append(snap)
        score = core.debug_read_16(0x8011EED6)
        log(f"  Drop {drop_num + 1}: score={score}")

    # For each snapshot, count how many u32 words differ from base
    # Group by 256-byte pages
    log("\n>>> Analyzing proportional changes per 256-byte page:")
    page_changes = {}  # page_addr -> [count_after_1, count_after_2, ...]
    for phys in range(SCAN_START, SCAN_END, 4):
        base_val = struct.unpack_from('<I', snap_base, phys)[0]
        page = (phys >> 8) << 8
        if page not in page_changes:
            page_changes[page] = [0] * 6
        for i in range(6):
            snap_val = struct.unpack_from('<I', snapshots[i + 1], phys)[0]
            if snap_val != base_val:
                page_changes[page][i] += 1

    # Find pages where changes grow roughly proportionally
    # Ideal: each drop adds ~1-4 new changes to this page
    log("\n>>> Pages with proportionally growing changes:")
    proportional_pages = []
    for page in sorted(page_changes.keys()):
        counts = page_changes[page]
        # Check if changes are monotonically increasing and roughly linear
        if counts[-1] < 3:
            continue  # Too few total changes
        if counts[-1] > 100:
            continue  # Too many (likely rendering data)
        monotonic = all(counts[i] <= counts[i+1] for i in range(5))
        if monotonic and counts[0] >= 1 and counts[-1] >= 6:
            addr = 0x80000000 + page
            growth_rate = counts[-1] / 6
            proportional_pages.append((addr, counts, growth_rate))
            log(f"  0x{addr:08X}: {counts} (avg {growth_rate:.1f}/drop)")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Byte-level proportional accumulation
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Byte-level proportional accumulation")
    log("=" * 60)

    # Count bytes (not words) that changed from base per 256-byte page
    byte_page_changes = {}
    for phys in range(SCAN_START, SCAN_END):
        base_val = snap_base[phys]
        page = (phys >> 8) << 8
        if page not in byte_page_changes:
            byte_page_changes[page] = [0] * 6
        for i in range(6):
            if snapshots[i + 1][phys] != base_val:
                byte_page_changes[page][i] += 1

    log("\n>>> Byte-level pages with proportional growth (monotonic, 4-80 changes):")
    for page in sorted(byte_page_changes.keys()):
        counts = byte_page_changes[page]
        if counts[-1] < 4 or counts[-1] > 80:
            continue
        monotonic = all(counts[i] <= counts[i+1] for i in range(5))
        if monotonic and counts[0] >= 1:
            addr = 0x80000000 + page
            log(f"  0x{addr:08X}: {counts}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Reserve piece (L swap)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Reserve piece (L button swap)")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()

    # Read key addresses before swap
    dffc = read_word(core, 0x8010DFFC)
    b4 = read_word(core, 0x801269B4)
    d4 = read_word(core, 0x801267D4)
    e0a4 = read_word(core, 0x8010E0A4)
    log(f"\n>>> Before L swap: DFFC={dffc} 69B4={b4} 67D4={d4} E0A4={e0a4}")

    # Take snapshot before swap
    snap_pre_swap = memory.snapshot()

    # Press L (shoulder button)
    press_frame(input_server, core, "L_TRIG", hold=2, gap=10)

    # Read after swap
    dffc2 = read_word(core, 0x8010DFFC)
    b4_2 = read_word(core, 0x801269B4)
    d4_2 = read_word(core, 0x801267D4)
    e0a4_2 = read_word(core, 0x8010E0A4)
    log(f">>> After L swap:  DFFC={dffc2} 69B4={b4_2} 67D4={d4_2} E0A4={e0a4_2}")

    snap_post_swap = memory.snapshot()

    # Find ALL words that changed between pre and post swap
    swap_changes = []
    for phys in range(SCAN_START, SCAN_END, 4):
        wa = struct.unpack_from('<I', snap_pre_swap, phys)[0]
        wb = struct.unpack_from('<I', snap_post_swap, phys)[0]
        if wa != wb:
            addr = 0x80000000 + phys
            swap_changes.append((addr, wa, wb))

    log(f"\n>>> Total u32 words changed after L swap: {len(swap_changes)}")

    # Focus on small changes (values < 100)
    log("\n>>> Small-value changes after L swap:")
    for addr, old, new in swap_changes:
        if old < 100 and new < 100:
            log(f"  0x{addr:08X}: {old} -> {new}")

    # Also show changes where only 1-2 bytes differ within the word
    log("\n>>> Single-byte changes within words after L swap:")
    for addr, old, new in swap_changes:
        diff_bytes = sum(1 for shift in range(0, 32, 8)
                        if ((old >> shift) & 0xFF) != ((new >> shift) & 0xFF))
        if diff_bytes == 1 and abs(old - new) < 256:
            log(f"  0x{addr:08X}: {old} -> {new} (0x{old:08X} -> 0x{new:08X})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Swap again to see if it reverses
    # ══════════════════════════════════════════════════════════
    log("\n>>> Second L swap (should swap back?):")
    press_frame(input_server, core, "L_TRIG", hold=2, gap=10)

    dffc3 = read_word(core, 0x8010DFFC)
    b4_3 = read_word(core, 0x801269B4)
    d4_3 = read_word(core, 0x801267D4)
    e0a4_3 = read_word(core, 0x8010E0A4)
    log(f">>> After 2nd swap: DFFC={dffc3} 69B4={b4_3} 67D4={d4_3} E0A4={e0a4_3}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: X position — wider search with all formats
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: X position — exhaustive search")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()
    snap_center = memory.snapshot()

    # 1 right press
    press_frame(input_server, core, "R_DPAD", hold=1, gap=3)
    snap_right1 = memory.snapshot()

    # 2nd right press
    press_frame(input_server, core, "R_DPAD", hold=1, gap=3)
    snap_right2 = memory.snapshot()

    # Search for u32 words where value increased by same delta twice
    log("\n>>> u32 words with consistent positive delta on 2 right presses:")
    for phys in range(SCAN_START, SCAN_END, 4):
        v0 = struct.unpack_from('<I', snap_center, phys)[0]
        v1 = struct.unpack_from('<I', snap_right1, phys)[0]
        v2 = struct.unpack_from('<I', snap_right2, phys)[0]
        d1 = v1 - v0
        d2 = v2 - v1
        if d1 == d2 and d1 > 0 and d1 < 100:
            addr = 0x80000000 + phys
            log(f"  0x{addr:08X}: {v0} -> {v1} -> {v2} (delta={d1})")

    # Search i16 with consistent delta (endian-aware)
    log("\n>>> i16 words with consistent positive delta:")
    for phys in range(SCAN_START, SCAN_END, 2):
        v0 = struct.unpack_from('<h', snap_center, phys)[0]
        v1 = struct.unpack_from('<h', snap_right1, phys)[0]
        v2 = struct.unpack_from('<h', snap_right2, phys)[0]
        d1 = v1 - v0
        d2 = v2 - v1
        if d1 == d2 and 0 < d1 < 50:
            addr = 0x80000000 + phys
            log(f"  0x{addr:08X}: {v0} -> {v1} -> {v2} (delta={d1}, i16)")

    # Search for floats that increased by roughly equal amounts
    log("\n>>> Floats with consistent positive delta:")
    for phys in range(SCAN_START, SCAN_END, 4):
        v0 = struct.unpack_from('<f', snap_center, phys)[0]
        v1 = struct.unpack_from('<f', snap_right1, phys)[0]
        v2 = struct.unpack_from('<f', snap_right2, phys)[0]
        # Check for reasonable float values and consistent delta
        if all(-100 < v < 100 for v in [v0, v1, v2]):
            d1 = v1 - v0
            d2 = v2 - v1
            if 0.1 < d1 < 10 and abs(d1 - d2) < 0.01:
                addr = 0x80000000 + phys
                log(f"  0x{addr:08X}: {v0:.3f} -> {v1:.3f} -> {v2:.3f} (delta={d1:.3f})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 6: Piece type mapping via block coordinates
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 6: Piece type mapping")
    log("=" * 60)

    # Drop 8 pieces, recording DFFC (type) and block coords each time
    reload_save(core)

    for drop in range(8):
        dffc = read_word(core, 0x8010DFFC)
        # Read block coords (8 u32 values at 0x8010BBF8-0x8010BC17)
        coords = []
        for i in range(8):
            coords.append(read_word(core, 0x8010BBF8 + i * 4))
        # Read the 0x801269B4 value too
        val_69b4 = read_word(core, 0x801269B4)
        val_67d4 = read_word(core, 0x801267D4)
        log(f"  Drop {drop}: DFFC={dffc} 69B4={val_69b4} 67D4={val_67d4} coords={coords}")

        drop_piece_realtime(core, input_server)

    log("\n" + "=" * 60)
    log("DISCOVERY v16 COMPLETE")
    log("=" * 60)

    try:
        core.stop()
    except:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
