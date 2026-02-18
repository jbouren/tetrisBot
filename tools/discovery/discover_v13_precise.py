#!/usr/bin/env python3
"""Discovery v13 — Frame-precise control + column pointers + board hunt.

Key issues from v12:
- Timing contamination makes X/rotation unreliable
- Board data not in display records (0x9C stride) or record indices (666-668)
- 10 pointers at 0x80107658-76 might reference 10 board columns
- Need frame-precise control via advance_frame()

Strategy:
1. Use advance_frame() for single-frame button presses
2. Track X candidate (BD64) with frame precision
3. Follow the 10-pointer chain at 0x80107658
4. Search for board as array of u8 or u32 (200 cells)
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


def hexdump_debug(core, start_virt, count):
    data = [core.debug_read_8(start_virt + i) for i in range(count)]
    for i in range(0, len(data), 16):
        row = data[i:i + 16]
        addr = start_virt + i
        hex_str = " ".join(f"{b:02X}" for b in row)
        log(f"  {addr:08X}: {hex_str:<48}")
    return data


def set_button(input_server, button):
    """Set a single button (or None to clear)."""
    state = ControllerState()
    if button:
        setattr(state, button, 1)
    input_server.set_state(state)


def main():
    log("=" * 60)
    log("Discovery v13 — Frame-Precise Control")
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
    # EXPERIMENT 1: Frame-precise X tracking
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 1: Frame-precise X tracking with advance_frame()")
    log("=" * 60)

    reload_save(core)

    # Read initial state
    addrs_to_track = {
        "BD64": 0x8010BD64,
        "BC24": 0x8010BC24,
        "BBEC": 0x8010BBEC,
        "DFFC": 0x8010DFFC,
        "Y_flt": 0x800D02CC,
        "D02B0": 0x800D02B0,
        "BB14": 0x8010BB14,
    }

    def read_state():
        vals = {}
        for name, addr in addrs_to_track.items():
            if name == "Y_flt":
                vals[name] = f"{read_float(core, addr):.2f}"
            else:
                vals[name] = read_word(core, addr)
        return vals

    log("\n>>> Initial state:")
    state = read_state()
    for name, val in state.items():
        log(f"  {name}: {val}")

    # Try advance_frame approach: set button, advance N frames, clear, read
    log("\n>>> Testing advance_frame() - advance 10 frames with no input:")
    try:
        for i in range(10):
            core.advance_frame()
            time.sleep(0.02)  # Small delay to let frame process
        log("  advance_frame() works!")
        state = read_state()
        for name, val in state.items():
            log(f"  {name}: {val}")
        use_frame_advance = True
    except Exception as e:
        log(f"  advance_frame() FAILED: {e}")
        log("  Falling back to resume/pause timing")
        use_frame_advance = False

    if use_frame_advance:
        # Do precise right moves: set R_DPAD, advance 1 frame, clear, advance 5 frames, read
        log("\n>>> Frame-precise RIGHT moves:")
        reload_save(core)
        state = read_state()
        log(f"  Initial: BD64={state['BD64']} BC24={state['BC24']} Y={state['Y_flt']}")

        for move in range(8):
            # Set right button
            set_button(input_server, "R_DPAD")
            # Advance 2 frames (press)
            core.advance_frame()
            time.sleep(0.01)
            core.advance_frame()
            time.sleep(0.01)
            # Release
            set_button(input_server, None)
            input_server.clear()
            # Advance 4 frames (release + processing)
            for _ in range(4):
                core.advance_frame()
                time.sleep(0.01)

            state = read_state()
            log(f"  Right {move+1}: BD64={state['BD64']} BC24={state['BC24']} Y={state['Y_flt']} BBEC={state['BBEC']} DFFC={state['DFFC']}")

        # Now left moves
        log("\n>>> Frame-precise LEFT moves:")
        reload_save(core)
        state = read_state()
        log(f"  Initial: BD64={state['BD64']} BC24={state['BC24']} Y={state['Y_flt']}")

        for move in range(8):
            set_button(input_server, "L_DPAD")
            core.advance_frame()
            time.sleep(0.01)
            core.advance_frame()
            time.sleep(0.01)
            set_button(input_server, None)
            input_server.clear()
            for _ in range(4):
                core.advance_frame()
                time.sleep(0.01)

            state = read_state()
            log(f"  Left {move+1}: BD64={state['BD64']} BC24={state['BC24']} Y={state['Y_flt']} BBEC={state['BBEC']} DFFC={state['DFFC']}")

        # Rotation test
        log("\n>>> Frame-precise ROTATION (A button):")
        reload_save(core)
        state = read_state()
        coords = [read_word(core, a) for a in range(0x8010BBF8, 0x8010BC18, 4)]
        log(f"  Initial: BD64={state['BD64']} BBEC={state['BBEC']} coords={coords}")

        for rot in range(5):
            set_button(input_server, "A_BUTTON")
            core.advance_frame()
            time.sleep(0.01)
            core.advance_frame()
            time.sleep(0.01)
            set_button(input_server, None)
            input_server.clear()
            for _ in range(4):
                core.advance_frame()
                time.sleep(0.01)

            state = read_state()
            coords = [read_word(core, a) for a in range(0x8010BBF8, 0x8010BC18, 4)]
            log(f"  Rot {rot+1}: BD64={state['BD64']} BBEC={state['BBEC']} DFFC={state['DFFC']} coords={coords}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Follow 10-column pointers
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Follow 10-column pointer chain")
    log("=" * 60)

    reload_save(core)

    # From v12: 0x80107658-0x8010767C had 10 pointers:
    # 0x80132560, 0x80132558, 0x80132550, 0x80132548, 0x80132540,
    # 0x80132538, 0x80132530, 0x80132528, 0x80132520, 0x80132518
    log("\n>>> 10 pointers at 0x80107658:")
    ptrs_10 = []
    for i in range(10):
        addr = 0x80107658 + i * 4
        ptr = read_word(core, addr)
        ptrs_10.append(ptr)
        log(f"  [{i}] 0x{addr:08X} -> 0x{ptr:08X}")

    # Follow each pointer and dump what's there
    for i, ptr in enumerate(ptrs_10):
        if 0x80000000 <= ptr < 0x80800000:
            log(f"\n>>> Column {i}? pointer 0x{ptr:08X}:")
            hexdump_debug(core, ptr, 0x60)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Board search — drop 10 pieces, find accumulation
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Board accumulation — drop 10 pieces")
    log("=" * 60)

    reload_save(core)
    memory.refresh_pointer()
    snap_empty = memory.snapshot()

    # Drop 10 pieces at center using resume/pause (reliable)
    for drop in range(10):
        core.resume()
        time.sleep(0.05)
        # Hard drop
        state = ControllerState()
        state.U_DPAD = 1
        input_server.set_state(state)
        time.sleep(0.05)
        input_server.clear()
        time.sleep(1.5)
        core.pause()
        time.sleep(0.1)

    snap_10pieces = memory.snapshot()

    # Search for 200-cell arrays where 30-50 bytes became nonzero
    # (10 pieces x 4 blocks = 40 filled cells, but some might stack)
    SCAN_START = 0x080000
    SCAN_END = 0x200000

    # Scan for contiguous regions where many bytes changed from 0 to nonzero
    # Use sliding window of 200-400 bytes
    for window_size in [200, 400, 800]:
        for stride in [1, 2, 4]:
            best_count = 0
            best_start = 0
            for start in range(SCAN_START, SCAN_END - window_size, stride):
                count = 0
                for offset in range(0, window_size, stride):
                    phys = start + offset
                    if phys < SCAN_END:
                        if snap_empty[phys] == 0 and snap_10pieces[phys] != 0:
                            count += 1
                if count > best_count and 20 <= count <= 60:
                    best_count = count
                    best_start = start

            if best_count >= 20:
                virt = 0x80000000 + best_start
                log(f"\n  Window={window_size}, stride={stride}: best={best_count} new nonzero cells at 0x{virt:08X}")
                # Show the values
                log(f"  Content:")
                for offset in range(0, min(window_size, 200), stride):
                    phys = best_start + offset
                    va = snap_empty[phys]
                    vb = snap_10pieces[phys]
                    if va != vb:
                        cell_virt = 0x80000000 + (phys ^ 3) if stride == 1 else 0x80000000 + phys
                        log(f"    offset {offset//stride:3d} (0x{cell_virt:08X}): {va} -> {vb}")

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Check 0x80132518-0x80132568 area (column data?)
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Column data area 0x80132518-0x80132600")
    log("=" * 60)

    log("\n>>> Before drops (from save state):")
    reload_save(core)
    hexdump_debug(core, 0x80132500, 0x100)

    # Drop 3 pieces
    for drop in range(3):
        core.resume()
        time.sleep(0.05)
        state = ControllerState()
        state.U_DPAD = 1
        input_server.set_state(state)
        time.sleep(0.05)
        input_server.clear()
        time.sleep(1.5)
        core.pause()
        time.sleep(0.1)

    log("\n>>> After 3 drops:")
    hexdump_debug(core, 0x80132500, 0x100)

    # ══════════════════════════════════════════════════════════
    # EXPERIMENT 5: Look at display area around 0x800E2070
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Display struct area 0x800E1F80-0x800E2100")
    log("=" * 60)

    reload_save(core)
    log("\n>>> Before drop (active piece display):")
    hexdump_debug(core, 0x800E1F80, 0x180)

    # Drop
    core.resume()
    time.sleep(0.05)
    state = ControllerState()
    state.U_DPAD = 1
    input_server.set_state(state)
    time.sleep(0.05)
    input_server.clear()
    time.sleep(1.5)
    core.pause()
    time.sleep(0.1)

    log("\n>>> After drop (placed piece display?):")
    hexdump_debug(core, 0x800E1F80, 0x180)

    log(f"\n>>> Score: {core.debug_read_16(0x8011EED6)}")

    log("\n" + "=" * 60)
    log("DISCOVERY v13 COMPLETE")
    log("=" * 60)

    try:
        core.stop()
    except:
        pass
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
