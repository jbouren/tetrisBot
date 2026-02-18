#!/usr/bin/env python3
"""Automated memory address discovery for The New Tetris.

Launches the emulator in-process, takes RDRAM snapshots at different
game states, and analyzes diffs to find board, piece, score, and
game state addresses.

Usage:
    cd /mnt/c/code/tetrisBot
    DISPLAY=:0 .venv/bin/python tools/discover_addresses.py
"""

import ctypes as ct
import logging
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.emulator.memory import MemoryReader, RDRAM_SIZE

# Ensure d3d12 GPU driver on WSL2
if "GALLIUM_DRIVER" not in os.environ:
    os.environ["GALLIUM_DRIVER"] = "d3d12"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("discover")

LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"

# Known GameShark addresses (physical RDRAM offsets)
KNOWN_ADDRESSES = {
    "score_a": 0x0011EED6,  # 0x8011EED6 & 0x1FFFFFFF
    "score_b": 0x0011EEB6,  # 0x8011EEB6 & 0x1FFFFFFF
    "game_mode": 0x000CFF60,  # 0x800CFF60 & 0x1FFFFFFF
}


def diff_snapshots(snap_a: bytes, snap_b: bytes, min_val_change: int = 1):
    """Find all byte offsets that changed between two RDRAM snapshots.

    Returns dict of {offset: (old_val, new_val)}.
    """
    changes = {}
    for i in range(len(snap_a)):
        if snap_a[i] != snap_b[i]:
            changes[i] = (snap_a[i], snap_b[i])
    return changes


def find_value_in_snapshot(snap: bytes, value: int, size: int = 2):
    """Search for a specific value in RDRAM snapshot.

    Tries both big-endian and little-endian, with and without XOR swapping.
    size: 1, 2, or 4 bytes.
    """
    results = []

    if size == 1:
        val_byte = value & 0xFF
        for i in range(len(snap)):
            if snap[i] == val_byte:
                results.append(("raw", i))
    elif size == 2:
        be = struct.pack(">H", value & 0xFFFF)
        le = struct.pack("<H", value & 0xFFFF)
        for i in range(len(snap) - 1):
            if snap[i:i+2] == be:
                results.append(("BE", i))
            if snap[i:i+2] == le:
                results.append(("LE", i))
    elif size == 4:
        be = struct.pack(">I", value & 0xFFFFFFFF)
        le = struct.pack("<I", value & 0xFFFFFFFF)
        for i in range(len(snap) - 3):
            if snap[i:i+4] == be:
                results.append(("BE32", i))
            if snap[i:i+4] == le:
                results.append(("LE32", i))

    return results


def cluster_changes(changes: dict, max_gap: int = 16):
    """Group changed offsets into clusters (nearby changes likely belong
    to the same data structure)."""
    if not changes:
        return []

    offsets = sorted(changes.keys())
    clusters = []
    current_cluster = [offsets[0]]

    for i in range(1, len(offsets)):
        if offsets[i] - offsets[i-1] <= max_gap:
            current_cluster.append(offsets[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [offsets[i]]
    clusters.append(current_cluster)

    return clusters


def analyze_region(snap: bytes, start: int, length: int, label: str = ""):
    """Print a hex dump of a memory region."""
    print(f"\n{'='*60}")
    print(f"Region: {label} @ 0x{start:06X} (virt 0x{0x80000000 + start:08X}), {length} bytes")
    print(f"{'='*60}")
    for row_start in range(start, start + length, 16):
        hex_part = ""
        ascii_part = ""
        for i in range(16):
            offset = row_start + i
            if offset < start + length and offset < len(snap):
                b = snap[offset]
                hex_part += f"{b:02X} "
                ascii_part += chr(b) if 32 <= b < 127 else "."
            else:
                hex_part += "   "
                ascii_part += " "
        print(f"  {row_start:06X}: {hex_part} |{ascii_part}|")


def send_button(input_server, button, hold_sec=0.1, wait_sec=0.5):
    """Press and release a button."""
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(wait_sec)


def advance_frames(core, n):
    """Advance N frames while paused."""
    for _ in range(n):
        core.advance_frame()


def main():
    print("=" * 60)
    print("The New Tetris — Memory Address Discovery")
    print("=" * 60)

    # Initialize
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

    # Attach plugins (no audio for speed)
    core.attach_plugins(
        gfx=str(LIB_DIR / "mupen64plus-video-GLideN64.so"),
        audio=None,
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    print("\n>>> Starting emulation...")
    core.execute()
    time.sleep(2)

    # Acquire RDRAM pointer
    memory.refresh_pointer()
    print(">>> RDRAM pointer acquired")

    # ── Phase 1: Title screen snapshot ─────────────────────────────
    print("\n>>> Phase 1: Waiting on title screen (3 seconds)...")
    time.sleep(3)

    core.pause()
    time.sleep(0.1)
    snap_title = memory.snapshot()
    print(f"  Title screen snapshot: {len(snap_title)} bytes")

    # Check known GameShark addresses
    print("\n>>> Checking known GameShark addresses...")
    for name, phys_offset in KNOWN_ADDRESSES.items():
        # Read with XOR byte swap (N64 big-endian to host)
        if phys_offset + 1 < len(snap_title):
            # Raw bytes at offset (host byte order, word-swapped)
            raw_b0 = snap_title[phys_offset]
            raw_b1 = snap_title[phys_offset + 1] if phys_offset + 1 < len(snap_title) else 0
            # Also read with XOR3 swap
            xor_b0 = snap_title[phys_offset ^ 3] if (phys_offset ^ 3) < len(snap_title) else 0
            xor_b1 = snap_title[(phys_offset + 1) ^ 3] if ((phys_offset + 1) ^ 3) < len(snap_title) else 0
            # And XOR2 swap for halfword
            xor2_offset = phys_offset ^ 2
            xor2_b0 = snap_title[xor2_offset] if xor2_offset < len(snap_title) else 0
            xor2_b1 = snap_title[xor2_offset + 1] if xor2_offset + 1 < len(snap_title) else 0

            raw_val = (raw_b0 << 8) | raw_b1
            xor3_val = (xor_b0 << 8) | xor_b1
            xor2_val = (xor2_b0 << 8) | xor2_b1

            print(f"  {name} @ phys 0x{phys_offset:06X}:")
            print(f"    raw={raw_val:#06x}  xor3={xor3_val:#06x}  xor2={xor2_val:#06x}")
            analyze_region(snap_title, phys_offset - 8, 32, f"{name} neighborhood")

    # ── Phase 2: Navigate to gameplay ─────────────────────────────
    print("\n>>> Phase 2: Navigating menus...")
    core.resume()
    time.sleep(0.5)

    # Press START to pass title screen
    send_button(input_server, "START_BUTTON", hold_sec=0.15, wait_sec=2.0)
    print("  Pressed START")

    # Press A to select mode (Marathon)
    send_button(input_server, "A_BUTTON", hold_sec=0.15, wait_sec=1.5)
    print("  Pressed A (mode select)")

    # Press A to confirm
    send_button(input_server, "A_BUTTON", hold_sec=0.15, wait_sec=1.5)
    print("  Pressed A (confirm)")

    # Press A to start
    send_button(input_server, "A_BUTTON", hold_sec=0.15, wait_sec=3.0)
    print("  Pressed A (start game)")

    # Take snapshot at what should be gameplay
    core.pause()
    time.sleep(0.1)
    snap_gameplay1 = memory.snapshot()
    print(f"  Gameplay snapshot 1: {len(snap_gameplay1)} bytes")

    # ── Phase 3: Diff title vs gameplay ───────────────────────────
    print("\n>>> Phase 3: Diffing title vs gameplay...")
    changes_title_game = diff_snapshots(snap_title, snap_gameplay1)
    print(f"  Total changed bytes: {len(changes_title_game)}")

    clusters = cluster_changes(changes_title_game, max_gap=4)
    print(f"  Change clusters (gap<=4): {len(clusters)}")

    # Show clusters near known addresses
    for name, phys_offset in KNOWN_ADDRESSES.items():
        for cluster in clusters:
            if any(abs(o - phys_offset) < 32 for o in cluster):
                print(f"\n  Cluster near {name} (0x{phys_offset:06X}):")
                print(f"    Offsets: {[f'0x{o:06X}' for o in cluster[:20]]}")
                if len(cluster) > 20:
                    print(f"    ... and {len(cluster) - 20} more")
                break

    # ── Phase 4: Advance some frames and diff again ───────────────
    print("\n>>> Phase 4: Advancing 60 frames (1 second of gameplay)...")
    core.resume()
    time.sleep(0.1)
    core.pause()
    time.sleep(0.1)

    # Advance 60 frames
    for _ in range(60):
        core.advance_frame()

    snap_gameplay2 = memory.snapshot()
    changes_game12 = diff_snapshots(snap_gameplay1, snap_gameplay2)
    print(f"  Changed bytes in 60 frames: {len(changes_game12)}")

    # ── Phase 5: Search for board-like structures ─────────────────
    print("\n>>> Phase 5: Looking for board-like data structures...")

    # The board is 10x20 = 200 cells. With 1-2 bytes per cell, that's 200-400 bytes.
    # A newly started game should have mostly zeros (empty board) with a few
    # non-zero bytes for the current piece.

    # Find large regions of zeros that might be the empty board
    # Look for 200+ consecutive zero bytes in gameplay snapshot
    zero_runs = []
    run_start = None
    run_len = 0
    for i in range(0x80000, 0x200000):  # Search in a reasonable range
        if snap_gameplay1[i] == 0:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len >= 200:
                zero_runs.append((run_start, run_len))
            run_start = None
            run_len = 0

    print(f"  Found {len(zero_runs)} zero runs >= 200 bytes (potential empty board)")
    for start, length in zero_runs[:10]:
        print(f"    0x{start:06X} - 0x{start+length:06X} ({length} bytes)")

    # ── Phase 6: Look for score changes ───────────────────────────
    print("\n>>> Phase 6: Searching for score value (should be 0 at start)...")

    # During early gameplay, score should be 0 or very small.
    # Look for known score address region
    score_phys = KNOWN_ADDRESSES["score_a"]
    analyze_region(snap_gameplay1, score_phys - 32, 80, "Score A region (gameplay)")
    analyze_region(snap_gameplay1, KNOWN_ADDRESSES["score_b"] - 16, 48, "Score B region (gameplay)")
    analyze_region(snap_gameplay1, KNOWN_ADDRESSES["game_mode"] - 16, 48, "Game mode region (gameplay)")

    # ── Phase 7: Advance more and place a piece ──────────────────
    print("\n>>> Phase 7: Pressing DOWN to speed up piece drop...")
    core.resume()

    # Hold DOWN to drop the piece faster
    state = ControllerState()
    state.D_DPAD = 1
    input_server.set_state(state)
    time.sleep(2.0)
    input_server.clear()
    time.sleep(1.0)

    core.pause()
    time.sleep(0.1)
    snap_after_drop = memory.snapshot()
    print(f"  Snapshot after drop: {len(snap_after_drop)} bytes")

    # Diff gameplay vs after dropping a piece
    changes_drop = diff_snapshots(snap_gameplay1, snap_after_drop)
    print(f"  Changed bytes after drop: {len(changes_drop)}")

    clusters_drop = cluster_changes(changes_drop, max_gap=2)
    print(f"  Clusters (gap<=2): {len(clusters_drop)}")

    # Look for clusters of exactly 4 changes close together (a tetromino)
    print("\n  Clusters of 2-8 bytes (possible piece/position data):")
    small_clusters = [c for c in clusters_drop if 2 <= len(c) <= 8]
    for cluster in small_clusters[:30]:
        vals_old = [f"{snap_gameplay1[o]:02X}" for o in cluster]
        vals_new = [f"{snap_after_drop[o]:02X}" for o in cluster]
        virt_addrs = [f"0x{0x80000000 + o:08X}" for o in cluster]
        print(f"    offsets={[f'0x{o:06X}' for o in cluster]}")
        print(f"    virt   ={virt_addrs}")
        print(f"    old    ={vals_old}  new={vals_new}")

    # Look for clusters of 10-40 bytes (possible board row changes)
    print("\n  Clusters of 10-40 bytes (possible board row data):")
    row_clusters = [c for c in clusters_drop if 10 <= len(c) <= 40]
    for cluster in row_clusters[:15]:
        start = cluster[0]
        end = cluster[-1]
        length = end - start + 1
        print(f"    0x{start:06X}-0x{end:06X} ({length} bytes, {len(cluster)} changed)")
        analyze_region(snap_after_drop, start - 4, length + 8,
                      f"Board row candidate @ 0x{start:06X}")

    # ── Phase 8: Try reading with DebugMemRead for verification ───
    print("\n>>> Phase 8: Reading known addresses via DebugMemRead API...")
    try:
        score_a_val = core.debug_read_16(0x8011EED6)
        score_b_val = core.debug_read_16(0x8011EEB6)
        game_mode_val = core.debug_read_16(0x800CFF60)
        p1_input_val = core.debug_read_32(0x801101B0)
        print(f"  Score A (0x8011EED6): {score_a_val:#06x} ({score_a_val})")
        print(f"  Score B (0x8011EEB6): {score_b_val:#06x} ({score_b_val})")
        print(f"  Game mode (0x800CFF60): {game_mode_val:#06x} ({game_mode_val})")
        print(f"  P1 input (0x801101B0): {p1_input_val:#010x}")
    except Exception as e:
        print(f"  DebugMemRead failed: {e}")

    # ── Phase 9: Wider scan - look for piece type values ──────────
    print("\n>>> Phase 9: Scanning for piece type patterns...")
    # Piece types are likely 0-6 (I,J,L,O,S,T,Z)
    # Current piece should be somewhere. Search for bytes that were 0
    # in title and became 0-6 in gameplay
    piece_candidates = []
    for offset, (old, new) in changes_title_game.items():
        if 0x80000 <= offset <= 0x200000:  # Reasonable range
            if old == 0 and 0 <= new <= 6:
                piece_candidates.append((offset, new))

    print(f"  Candidates (was 0, now 0-6): {len(piece_candidates)}")
    # Group by value to find which piece type value is most common
    by_value = defaultdict(list)
    for offset, val in piece_candidates:
        by_value[val].append(offset)
    for val in sorted(by_value.keys()):
        count = len(by_value[val])
        samples = [f"0x{o:06X}" for o in by_value[val][:5]]
        print(f"    value={val}: {count} occurrences, e.g. {samples}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Title -> Gameplay: {len(changes_title_game)} bytes changed")
    print(f"Gameplay 60 frames: {len(changes_game12)} bytes changed")
    print(f"After piece drop: {len(changes_drop)} bytes changed")
    print(f"\nNext steps:")
    print(f"  1. Examine the hex dumps above for recognizable patterns")
    print(f"  2. Use tools/memory_scanner.py interactively for targeted searches")
    print(f"  3. Place specific pieces and diff to find board layout")

    # Cleanup
    print("\n>>> Shutting down...")
    try:
        core.shutdown()
    except Exception:
        pass
    input_server.stop()
    print("Done.")


if __name__ == "__main__":
    main()
