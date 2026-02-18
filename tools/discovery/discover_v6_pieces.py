#!/usr/bin/env python3
"""Discovery v6 — Piece type, next piece, reserve piece, position tracking.

Uses corrected XOR^3 byte ordering throughout.
Tracks specific candidate addresses across multiple drops to identify:
- Current piece type
- Next piece(s)
- Reserve/hold piece
- Piece X, Y position
- Rotation state

Also examines the board record structure at 0x800D10A9 (stride 0x9C).
"""

import logging
import os
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


# ── Candidate addresses from v5 (all corrected with XOR^3) ────────────
PIECE_CANDIDATES = {
    # From cross-phase and L-swap analysis
    "0x8010E0A7": 0x8010E0A7,   # Was 0x8010E0A4 (uncorrected). Val=7 at save.
    "0x8010BBEF": 0x8010BBEF,   # Cycles 4,6,0,2,6,0 across drops. Y-candidate too.
    "0x8010BC27": 0x8010BC27,   # drop1 2->6, drop2 6->5
    "0x8010BD67": 0x8010BD67,   # Mirrors BC27
    "0x8010DFFF": 0x8010DFFF,   # drop1 5->0, drop2 0->6
    "0x800D02B0": 0x800D02B0,   # drop1 5->2, drop2 2->3
    "0x8011EEEA": 0x8011EEEA,   # Reversed on L-swap (3<->2)
    "0x801211BF": 0x801211BF,   # Reversed on L-swap (4<->2)
}

# Also read the full 32-bit word containing each candidate for context
WORD_ADDRS = {
    "E0A4 word": 0x8010E0A4,
    "BBEC word": 0x8010BBEC,
    "BC24 word": 0x8010BC24,
    "BD64 word": 0x8010BD64,
    "DFFC word": 0x8010DFFC,
    "D02B0 word": 0x800D02B0,  # Already aligned? Let's read aligned
    "EEEA word": 0x8011EEE8,
}


def read_candidates(core, label=""):
    """Read all piece candidate addresses and print them."""
    vals = {}
    for name, addr in sorted(PIECE_CANDIDATES.items(), key=lambda x: x[1]):
        val = core.debug_read_8(addr)
        vals[addr] = val
    if label:
        parts = [f"{v}" for _, v in sorted(vals.items())]
        log(f"  {label}: " + " | ".join(
            f"{name}={vals[addr]}" for name, addr in sorted(PIECE_CANDIDATES.items(), key=lambda x: x[1])
        ))
    return vals


def main():
    log("=" * 60)
    log("Discovery v6 — Piece & Position Tracking")
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
    # EXPERIMENT 1: Track candidates across 8 drops
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Piece candidates across 8 drops")
    log("=" * 60)

    reload_save(core)
    log("\n  Header: " + " | ".join(
        name for name, _ in sorted(PIECE_CANDIDATES.items(), key=lambda x: x[1])
    ))

    read_candidates(core, "save state")

    # Also read 32-bit words for context
    log("\n  32-bit words at save state:")
    for name, addr in sorted(WORD_ADDRS.items()):
        val = core.debug_read_32(addr & ~3)  # Align to 4
        log(f"    {name} (0x{addr & ~3:08X}): 0x{val:08X} = {val}")

    for drop in range(1, 9):
        core.resume()
        press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
        core.pause()
        time.sleep(0.1)
        read_candidates(core, f"drop {drop}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: L-swap tracking (current <-> reserve)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: L-swap (current <-> reserve)")
    log("=" * 60)

    reload_save(core)
    read_candidates(core, "save state")

    # L-swap
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.5)
    core.pause()
    time.sleep(0.1)
    read_candidates(core, "after L-swap 1")

    # L-swap back
    core.resume()
    time.sleep(0.1)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.5)
    core.pause()
    time.sleep(0.1)
    read_candidates(core, "after L-swap 2")

    # Drop then L-swap (so a different piece is current)
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)
    read_candidates(core, "after drop")

    core.resume()
    time.sleep(0.1)
    press_button(input_server, "L_TRIG", hold_sec=0.05, pause_sec=0.5)
    core.pause()
    time.sleep(0.1)
    read_candidates(core, "after L-swap 3")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Move tracking (X position)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: X position tracking")
    log("=" * 60)

    reload_save(core)

    # X candidate addresses (from v5 cross-phase analysis)
    x_candidates = [
        0x800D391B, 0x800E20B1, 0x801255BF,
        0x801352AC, 0x8013550C, 0x8013551C,
        0x801552BC, 0x8015551C, 0x8015552C,
    ]

    log("\n  X candidates at save state:")
    for addr in x_candidates:
        log(f"    0x{addr:08X}: {core.debug_read_8(addr)}")

    # Move right multiple times
    core.resume()
    time.sleep(0.1)
    for move in range(1, 5):
        press_button(input_server, "R_DPAD", hold_sec=0.05, pause_sec=0.15)
        core.pause()
        time.sleep(0.1)
        log(f"\n  After move right {move}:")
        for addr in x_candidates:
            log(f"    0x{addr:08X}: {core.debug_read_8(addr)}")
        if move < 4:
            core.resume()
            time.sleep(0.05)

    # Move left
    reload_save(core)
    core.resume()
    time.sleep(0.1)
    for move in range(1, 5):
        press_button(input_server, "L_DPAD", hold_sec=0.05, pause_sec=0.15)
        core.pause()
        time.sleep(0.1)
        log(f"\n  After move left {move}:")
        for addr in x_candidates:
            log(f"    0x{addr:08X}: {core.debug_read_8(addr)}")
        if move < 4:
            core.resume()
            time.sleep(0.05)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Fall tracking (Y position)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Y position tracking")
    log("=" * 60)

    reload_save(core)

    # Expanded Y candidate search: scan for values that increase
    # frame by frame as piece falls. Check piece candidates too.
    y_track = [0x8010BBEF, 0x80128907]  # From v5 Y candidates

    log("\n  Y candidates at save state:")
    for addr in y_track:
        log(f"    0x{addr:08X}: {core.debug_read_8(addr)}")

    # Let piece fall for short intervals and check
    core.resume()
    for t in range(1, 8):
        time.sleep(0.3)
        core.pause()
        time.sleep(0.05)
        log(f"\n  After falling {t * 0.3:.1f}s:")
        for addr in y_track:
            log(f"    0x{addr:08X}: {core.debug_read_8(addr)}")
        # Also check 16-bit and 32-bit reads nearby
        for addr in [0x8010BBEC, 0x8010BBF0]:
            val32 = core.debug_read_32(addr & ~3)
            log(f"    0x{addr & ~3:08X} (u32): 0x{val32:08X} = {val32}")
        core.resume()

    core.pause()
    time.sleep(0.1)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Wider search for Y position
    # Look for bytes/words that increment steadily as piece falls
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Systematic Y position search")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()
    snap0 = memory.snapshot()

    # Take snapshots at 3 time intervals
    core.resume()
    time.sleep(0.4)
    core.pause()
    time.sleep(0.1)
    snap1 = memory.snapshot()

    core.resume()
    time.sleep(0.4)
    core.pause()
    time.sleep(0.1)
    snap2 = memory.snapshot()

    core.resume()
    time.sleep(0.4)
    core.pause()
    time.sleep(0.1)
    snap3 = memory.snapshot()

    # Find bytes that strictly increase across all 3 intervals
    # (indicating Y position incrementing as piece falls)
    SCAN_START = 0x080000
    SCAN_END = 0x200000
    increasing_bytes = []
    for phys in range(SCAN_START, SCAN_END):
        v0, v1, v2, v3 = snap0[phys], snap1[phys], snap2[phys], snap3[phys]
        if v0 < v1 < v2 < v3 and v0 < 25 and v3 < 25:
            virt = 0x80000000 + (phys ^ 3)
            increasing_bytes.append((virt, v0, v1, v2, v3))

    log(f"\n  Strictly increasing bytes (val 0-24): {len(increasing_bytes)}")
    for addr, v0, v1, v2, v3 in increasing_bytes[:30]:
        dbg = core.debug_read_8(addr)
        log(f"    0x{addr:08X}: {v0} -> {v1} -> {v2} -> {v3}  [debug now={dbg}]")

    # Also search for 16-bit values that increase (check aligned pairs)
    # Using debug_read_16 equivalent from snapshots:
    # For a 16-bit value at N64 addr A (aligned to 2):
    #   phys = (A & 0x1FFFFFFF) ^ 2  -- this gives the start of the 2 bytes in snapshot
    # Actually, for 16-bit: snapshot stores in host order within 32-bit words.
    # Let me just use word-aligned 32-bit reads and check fields within.
    increasing_words = []
    for phys in range(SCAN_START, SCAN_END, 4):
        # Read as 32-bit little-endian from snapshot
        w0 = int.from_bytes(snap0[phys:phys+4], 'little')
        w1 = int.from_bytes(snap1[phys:phys+4], 'little')
        w2 = int.from_bytes(snap2[phys:phys+4], 'little')
        w3 = int.from_bytes(snap3[phys:phys+4], 'little')
        if w0 < w1 < w2 < w3 and w0 < 25 and w3 < 25 and w0 >= 0:
            virt = 0x80000000 + phys
            increasing_words.append((virt, w0, w1, w2, w3))

    log(f"\n  Strictly increasing 32-bit words (val 0-24): {len(increasing_words)}")
    for addr, v0, v1, v2, v3 in increasing_words[:30]:
        dbg = core.debug_read_32(addr)
        log(f"    0x{addr:08X}: {v0} -> {v1} -> {v2} -> {v3}  [debug now=0x{dbg:08X}]")

    # Check 16-bit halves of each word too
    increasing_halves = []
    for phys in range(SCAN_START, SCAN_END, 4):
        w0 = int.from_bytes(snap0[phys:phys+4], 'little')
        w1 = int.from_bytes(snap1[phys:phys+4], 'little')
        w2 = int.from_bytes(snap2[phys:phys+4], 'little')
        w3 = int.from_bytes(snap3[phys:phys+4], 'little')
        # High 16 bits (N64 addr + 0)
        h0, h1, h2, h3 = w0 >> 16, w1 >> 16, w2 >> 16, w3 >> 16
        if h0 < h1 < h2 < h3 and h0 < 25 and h3 < 25:
            virt = 0x80000000 + phys  # 32-bit aligned addr, high half
            increasing_halves.append((virt, "hi16", h0, h1, h2, h3))
        # Low 16 bits (N64 addr + 2)
        l0, l1, l2, l3 = w0 & 0xFFFF, w1 & 0xFFFF, w2 & 0xFFFF, w3 & 0xFFFF
        if l0 < l1 < l2 < l3 and l0 < 25 and l3 < 25:
            virt = 0x80000000 + phys + 2
            increasing_halves.append((virt, "lo16", l0, l1, l2, l3))

    log(f"\n  Strictly increasing 16-bit halves (val 0-24): {len(increasing_halves)}")
    for addr, half, v0, v1, v2, v3 in increasing_halves[:30]:
        dbg = core.debug_read_16(addr)
        log(f"    0x{addr:08X} ({half}): {v0} -> {v1} -> {v2} -> {v3}  [debug now={dbg}]")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 6: Board record structure dump
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 6: Board Record Structure")
    log("=" * 60)

    reload_save(core)

    # The board region has records at stride 0x9C starting around 0x800D10A9
    RECORD_BASE = 0x800D10A9
    RECORD_STRIDE = 0x9C  # 156 bytes
    NUM_RECORDS = 20  # Check 20 records (could be rows or cells)

    log(f"\n>>> Board records BEFORE drop (first 5 of {NUM_RECORDS}):")
    log(f"    Record base: 0x{RECORD_BASE:08X}, stride: 0x{RECORD_STRIDE:X}")
    for r in range(5):
        addr = RECORD_BASE + r * RECORD_STRIDE
        log(f"\n  Record {r} (0x{addr:08X}):")
        hexdump_debug(core, addr, RECORD_STRIDE)

    # Drop a piece (center, default position)
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    log(f"\n>>> Board records AFTER 1st drop (first 5):")
    for r in range(5):
        addr = RECORD_BASE + r * RECORD_STRIDE
        log(f"\n  Record {r} (0x{addr:08X}):")
        hexdump_debug(core, addr, RECORD_STRIDE)

    # Drop another piece
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    log(f"\n>>> Board records AFTER 2nd drop (first 5):")
    for r in range(5):
        addr = RECORD_BASE + r * RECORD_STRIDE
        log(f"\n  Record {r} (0x{addr:08X}):")
        hexdump_debug(core, addr, RECORD_STRIDE)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 7: Check near known addresses for piece struct
    # The piece type, position, rotation might be in a struct
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 7: Piece struct neighborhood")
    log("=" * 60)

    reload_save(core)

    # Dump neighborhoods around the strongest piece candidates
    for name, addr in [
        ("0x8010BBEC region", 0x8010BBD0),
        ("0x8010BC24 region", 0x8010BC10),
        ("0x8010DFFC region", 0x8010DFE0),
        ("0x8010E0A4 region", 0x8010E090),
    ]:
        log(f"\n>>> {name}:")
        hexdump_debug(core, addr, 64)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 8: Wider piece address scan
    # Look for struct-like patterns containing piece type value
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 8: Find piece struct by scanning for type value")
    log("=" * 60)

    reload_save(core)

    # At save state, what piece type value do we see?
    # 0x8010E0A7 = 7, but types should be 0-6. Let me check if 7
    # means something else, or if the mapping is different.
    # Let's just dump the region and look for patterns.

    # Read key values at save state
    log("\n>>> Key piece values at save state:")
    for addr in [0x8010E0A4, 0x8010E0A7, 0x8010BBEC, 0x8010BBEF,
                 0x8010BC24, 0x8010BC27, 0x8010BD64, 0x8010BD67,
                 0x8010DFFC, 0x8010DFFF]:
        val = core.debug_read_8(addr)
        w32 = core.debug_read_32(addr & ~3)
        log(f"    0x{addr:08X}: u8={val}  u32@aligned=0x{w32:08X}")

    # Now drop and move to see how values change together
    log("\n>>> Drop + check struct:")
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.05, pause_sec=1.5)
    core.pause()
    time.sleep(0.1)

    for addr in [0x8010E0A4, 0x8010E0A7, 0x8010BBEC, 0x8010BBEF,
                 0x8010BC24, 0x8010BC27, 0x8010BD64, 0x8010BD67,
                 0x8010DFFC, 0x8010DFFF]:
        val = core.debug_read_8(addr)
        w32 = core.debug_read_32(addr & ~3)
        log(f"    0x{addr:08X}: u8={val}  u32@aligned=0x{w32:08X}")

    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("DISCOVERY v6 COMPLETE")
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
