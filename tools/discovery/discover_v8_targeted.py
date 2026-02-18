#!/usr/bin/env python3
"""Discovery v8 — Targeted: board grid at 0x8012E2xx, Y/X floats, piece struct.

Key findings from v7:
- Y position is likely float at 0x800D02CC (32.40→35.15 during fall)
- Piece struct at 0x8010BBE0 has pointer 0x8012E220 at offset 0x8010BC58
- 0x8012E2xx region had board-like hits (0→1 and 0→2 transitions)
- Board display objects at 0x800D10A9 (stride 0x9C) are NOT the actual grid

This script:
1. Dumps 0x8012E200-0x8012E400 before/after drops to find board grid
2. Confirms Y float and finds X float nearby
3. Explores piece struct to identify current/next piece type
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
    """Read IEEE float at N64 address. mupen64plus stores 32-bit words
    in host order, so debug_read_32 returns the correct bit pattern."""
    raw = core.debug_read_32(addr)
    return struct.unpack('f', struct.pack('I', raw))[0]


def read_word(core, addr):
    return core.debug_read_32(addr)


def main():
    log("=" * 60)
    log("Discovery v8 — Targeted Board & Position")
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
    # EXPERIMENT 1: Board grid at 0x8012E2xx
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Board grid (0x8012E200-0x8012E400)")
    log("=" * 60)

    reload_save(core)

    log("\n>>> Before any drops:")
    hexdump_debug(core, 0x8012E200, 0x200)

    # Drop piece at center
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After 1st drop (center):")
    hexdump_debug(core, 0x8012E200, 0x200)

    # Drop another piece
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After 2nd drop:")
    hexdump_debug(core, 0x8012E200, 0x200)

    # Wider area — also check 0x8012E000-0x8012E200 and 0x8012E400-0x8012E600
    log("\n>>> Wider area after 2 drops (0x8012E000-0x8012E200):")
    hexdump_debug(core, 0x8012E000, 0x200)

    log("\n>>> Wider area after 2 drops (0x8012E400-0x8012E600):")
    hexdump_debug(core, 0x8012E400, 0x200)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Y and X float position
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Y/X float position")
    log("=" * 60)

    reload_save(core)

    # Read float at 0x800D02CC and nearby addresses
    log("\n>>> Float values near piece Y at save state:")
    for offset in range(-0x20, 0x30, 4):
        addr = 0x800D02CC + offset
        try:
            fval = read_float(core, addr)
            uval = read_word(core, addr)
            if abs(fval) < 10000 and fval != 0.0:
                log(f"  0x{addr:08X}: float={fval:10.4f}  u32=0x{uval:08X}  <-- NONZERO FLOAT")
            else:
                log(f"  0x{addr:08X}: float={fval:10.4f}  u32=0x{uval:08X}")
        except:
            log(f"  0x{addr:08X}: ERROR reading")

    # Let piece fall and check Y float
    log("\n>>> Y float during fall (every 0.2s for 2s):")
    core.resume()
    for t in range(10):
        time.sleep(0.2)
        core.pause()
        time.sleep(0.05)
        y_val = read_float(core, 0x800D02CC)
        # Also check nearby addresses for X
        vals = {}
        for offset in [-16, -12, -8, -4, 0, 4, 8, 12, 16]:
            addr = 0x800D02CC + offset
            try:
                vals[offset] = read_float(core, addr)
            except:
                vals[offset] = None
        log(f"  t={t*0.2+0.2:.1f}s: Y(+0)={vals[0]:.2f}  " +
            " ".join(f"({offset:+d})={v:.2f}" for offset, v in sorted(vals.items()) if v is not None and offset != 0 and abs(v) < 1000 and v != 0))
        core.resume()

    core.pause()
    time.sleep(0.1)

    # Now test X: move right and check all nearby floats
    log("\n>>> Float values after move right (relative to 0x800D02CC):")
    reload_save(core)

    # Read before move
    before_floats = {}
    for offset in range(-0x40, 0x40, 4):
        addr = 0x800D02CC + offset
        try:
            before_floats[addr] = read_float(core, addr)
        except:
            pass

    core.resume()
    time.sleep(0.1)  # Brief fall
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.1)
    core.pause()
    time.sleep(0.1)

    log("\n  Changes after 1 right move (float values that changed):")
    for addr in sorted(before_floats.keys()):
        after = read_float(core, addr)
        before = before_floats[addr]
        if before != after and abs(before) < 10000 and abs(after) < 10000:
            log(f"    0x{addr:08X}: {before:.4f} -> {after:.4f} (delta {after-before:+.4f})")

    # Move right 3 more times
    log("\n>>> Tracking float changes across 4 right moves:")
    reload_save(core)

    # Find all float addresses between 0x800D0200-0x800D0400
    float_track = {}
    for offset in range(-0xCC, 0x134, 4):
        addr = 0x800D02CC + offset
        float_track[addr] = [read_float(core, addr)]

    core.resume()
    time.sleep(0.1)
    for move in range(4):
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.15)
        core.pause()
        time.sleep(0.05)
        for addr in float_track:
            float_track[addr].append(read_float(core, addr))
        if move < 3:
            core.resume()
            time.sleep(0.05)

    # Show floats that changed consistently
    log("\n  Float addresses that changed monotonically on right moves:")
    for addr in sorted(float_track.keys()):
        vals = float_track[addr]
        if all(abs(v) < 10000 for v in vals):
            deltas = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            # All deltas same sign and > 0.1
            if all(d > 0.1 for d in deltas) or all(d < -0.1 for d in deltas):
                log(f"    0x{addr:08X}: {[f'{v:.2f}' for v in vals]} deltas={[f'{d:.2f}' for d in deltas]}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Wider search for X float
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Wider X float search")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()
    snap_xr0 = memory.snapshot()

    core.resume()
    time.sleep(0.05)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
    press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.05)
    core.pause()
    time.sleep(0.1)
    snap_xr3 = memory.snapshot()

    log("\n>>> Floats that increased after 3 right moves (game data range):")
    SCAN_START = 0x080000
    SCAN_END = 0x200000
    x_float_candidates = []
    for phys in range(SCAN_START, SCAN_END, 4):
        f0 = struct.unpack_from('<f', snap_xr0, phys)[0]
        f1 = struct.unpack_from('<f', snap_xr3, phys)[0]
        if 0 < f0 < 500 and 0 < f1 < 500 and 1.0 < (f1 - f0) < 100:
            x_float_candidates.append((0x80000000 + phys, f0, f1))

    log(f"  Float candidates (increase 1-100 after 3 right moves): {len(x_float_candidates)}")
    for addr, f0, f1 in x_float_candidates[:30]:
        log(f"    0x{addr:08X}: {f0:.2f} -> {f1:.2f} (delta {f1-f0:+.2f})")

    # Also try decrease (some games use negative X direction)
    x_float_dec = []
    for phys in range(SCAN_START, SCAN_END, 4):
        f0 = struct.unpack_from('<f', snap_xr0, phys)[0]
        f1 = struct.unpack_from('<f', snap_xr3, phys)[0]
        if 0 < f0 < 500 and 0 < f1 < 500 and -100 < (f1 - f0) < -1.0:
            x_float_dec.append((0x80000000 + phys, f0, f1))

    log(f"\n  Float candidates (decrease 1-100 after 3 right moves): {len(x_float_dec)}")
    for addr, f0, f1 in x_float_dec[:30]:
        log(f"    0x{addr:08X}: {f0:.2f} -> {f1:.2f} (delta {f1-f0:+.2f})")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Piece struct — piece queue at 0x8010DFF0
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Piece queue in struct B")
    log("=" * 60)

    reload_save(core)

    # Struct B at 0x8010DFF0-0x8010E0B0 contains 8 entries, each 0x20 bytes
    # Format: ptr1, ptr2, 0, value, 0x08, ptr3, ptr4, ptr5
    # Values at DFF0+C, E010+C, E030+C, E050+C, E070+C, E090+C, E0A0+4, ...
    log("\n>>> Piece queue values at save state:")
    queue_addrs = []
    base = 0x8010DFFC
    for i in range(8):
        addr = base + i * 0x20
        if addr < 0x8010E100:
            val = read_word(core, addr)
            log(f"  Queue[{i}] at 0x{addr:08X}: {val}")
            queue_addrs.append(addr)

    # Actually, let me read the struct entries more carefully
    # Each entry is: ptr, ptr, u32(0), u32(VALUE), u32(8), ptr
    # Starting at 0x8010DFF0 with stride 0x20
    log("\n>>> Piece queue (full struct entries):")
    for i in range(8):
        base_addr = 0x8010DFF0 + i * 0x20
        ptr1 = read_word(core, base_addr)
        ptr2 = read_word(core, base_addr + 4)
        zero = read_word(core, base_addr + 8)
        value = read_word(core, base_addr + 12)
        eight = read_word(core, base_addr + 16)
        ptr3 = read_word(core, base_addr + 20)
        log(f"  [{i}] 0x{base_addr:08X}: ptr1=0x{ptr1:08X} ptr2=0x{ptr2:08X} " +
            f"zero={zero} VALUE={value} eight={eight} ptr3=0x{ptr3:08X}")

    # Drop 5 pieces and track queue values
    log("\n>>> Queue values across 5 drops:")
    for drop in range(5):
        core.resume()
        press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=2.0)
        core.pause()
        time.sleep(0.1)
        vals = [read_word(core, 0x8010DFF0 + i * 0x20 + 12) for i in range(8)]
        log(f"  Drop {drop+1}: {vals}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Current piece struct A — full decode
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Current piece struct decode")
    log("=" * 60)

    reload_save(core)

    log("\n>>> Piece struct A full decode at save state:")
    log(f"  0x8010BBE0: ptr1    = 0x{read_word(core, 0x8010BBE0):08X}")
    log(f"  0x8010BBE4: ptr2    = 0x{read_word(core, 0x8010BBE4):08X}  (0x800DB210 = shared game ptr)")
    log(f"  0x8010BBE8: field1  = {read_word(core, 0x8010BBE8)}")
    log(f"  0x8010BBEC: field2  = {read_word(core, 0x8010BBEC)}  (piece-related, values 0-7)")
    log(f"  0x8010BBF0: const8  = {read_word(core, 0x8010BBF0)}")
    log(f"  0x8010BBF4: self_ptr= 0x{read_word(core, 0x8010BBF4):08X}")

    # Coords area — 8 u32 values (4 pairs of positions?)
    log(f"  0x8010BBF8: coord0  = {read_word(core, 0x8010BBF8)}  ({read_word(core, 0x8010BBF8) & 0xFFFF})")
    log(f"  0x8010BBFC: coord1  = {read_word(core, 0x8010BBFC)}")
    log(f"  0x8010BC00: coord2  = {read_word(core, 0x8010BC00)}")
    log(f"  0x8010BC04: coord3  = {read_word(core, 0x8010BC04)}")
    log(f"  0x8010BC08: coord4  = {read_word(core, 0x8010BC08)}")
    log(f"  0x8010BC0C: coord5  = {read_word(core, 0x8010BC0C)}")
    log(f"  0x8010BC10: coord6  = {read_word(core, 0x8010BC10)}")
    log(f"  0x8010BC14: coord7  = {read_word(core, 0x8010BC14)}")

    log(f"  0x8010BC18: ptr3    = 0x{read_word(core, 0x8010BC18):08X}")
    log(f"  0x8010BC1C: ptr4    = 0x{read_word(core, 0x8010BC1C):08X}")
    log(f"  0x8010BC20: field3  = {read_word(core, 0x8010BC20)}")
    log(f"  0x8010BC24: field4  = {read_word(core, 0x8010BC24)}  (piece-related, values 0-7)")
    log(f"  0x8010BC28: const8b = {read_word(core, 0x8010BC28)}")
    log(f"  0x8010BC2C: ptr5    = 0x{read_word(core, 0x8010BC2C):08X}")

    # Pointers area (0x8010BC30-0x8010BC4C)
    for i in range(8):
        addr = 0x8010BC30 + i * 4
        log(f"  0x{addr:08X}: ptr     = 0x{read_word(core, addr):08X}")

    log(f"  0x8010BC50: ptr_gm  = 0x{read_word(core, 0x8010BC50):08X}")
    log(f"  0x8010BC54: field5  = {read_word(core, 0x8010BC54)}  (13=0x0D)")
    log(f"  0x8010BC58: self2   = 0x{read_word(core, 0x8010BC58):08X}")
    log(f"  0x8010BC5C: board?  = 0x{read_word(core, 0x8010BC5C):08X}  <-- BOARD POINTER?")

    # Follow the board pointer
    board_ptr = read_word(core, 0x8010BC5C)
    log(f"\n>>> Following board pointer 0x{board_ptr:08X}:")
    if 0x80000000 <= board_ptr < 0x80800000:
        hexdump_debug(core, board_ptr, 0x200)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 6: Scan piece struct area for X/Y as float
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 6: Piece area float scan")
    log("=" * 60)

    reload_save(core)

    # Read all floats in the piece struct area 0x8010BB00-0x8010BD00
    log("\n>>> All nonzero floats in piece struct area:")
    for addr in range(0x8010BB00, 0x8010BD00, 4):
        fval = read_float(core, addr)
        if abs(fval) > 0.01 and abs(fval) < 100000:
            uval = read_word(core, addr)
            log(f"  0x{addr:08X}: float={fval:.4f}  u32={uval}  (0x{uval:08X})")

    # Also check 0x800D0200-0x800D0400
    log("\n>>> All nonzero floats in 0x800D0200-0x800D0400:")
    for addr in range(0x800D0200, 0x800D0400, 4):
        fval = read_float(core, addr)
        if abs(fval) > 0.01 and abs(fval) < 100000:
            uval = read_word(core, addr)
            log(f"  0x{addr:08X}: float={fval:.4f}  u32={uval}  (0x{uval:08X})")

    # Let piece fall 1 second and check which floats changed
    core.resume()
    time.sleep(1.0)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After 1s fall, changed floats in 0x800D0200-0x800D0400:")
    for addr in range(0x800D0200, 0x800D0400, 4):
        fval = read_float(core, addr)
        if abs(fval) > 0.01 and abs(fval) < 100000:
            uval = read_word(core, addr)
            log(f"  0x{addr:08X}: float={fval:.4f}  u32={uval}")

    # ── Score check ──────────────────────────────────────────
    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v8 COMPLETE")
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
