#!/usr/bin/env python3
"""Interactive memory scanner for discovering game state addresses in RDRAM.

Usage:
    python tools/memory_scanner.py

This tool connects to the mupen64plus core, starts the game, and provides
an interactive console for exploring memory. The workflow for finding
addresses:

1. Start the game and get to gameplay
2. Take a snapshot: `snap`
3. Perform a known action (place a piece, clear a line)
4. Take another snapshot: `snap`
5. Diff the snapshots: `diff`
6. Search for known values: `search <value>`
7. Watch an address: `watch <addr>`
8. Narrow down candidates by repeating

Commands:
    snap              - Take an RDRAM snapshot
    diff              - Diff the last two snapshots
    search <value>    - Search current snapshot for a value (decimal or 0x hex)
    search16 <value>  - Search for a 16-bit value
    search32 <value>  - Search for a 32-bit value
    read <addr>       - Read value at address (virtual 0x80XXXXXX)
    read16 <addr>     - Read 16-bit value
    read32 <addr>     - Read 32-bit value
    block <addr> <n>  - Read N bytes starting at address
    watch <addr> [n]  - Watch address, print value every n frames (default 30)
    pause             - Pause emulation
    resume            - Resume emulation
    step [n]          - Advance n frames (default 1)
    save <slot>       - Save state to slot (0-9)
    load <slot>       - Load state from slot
    score             - Read known score address
    help              - Show this help
    quit              - Exit
"""

import logging
import struct
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.memory import MemoryReader, RDRAM_SIZE
from src.game import memory_map as mm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"


class MemoryScanner:
    """Interactive memory exploration tool."""

    def __init__(self):
        self.core: Mupen64PlusCore | None = None
        self.memory: MemoryReader | None = None
        self.snapshots: list[bytes] = []

    def start_emulator(self):
        """Initialize and start the emulator."""
        core_path = str(LIB_DIR / "libmupen64plus.so.2.0.0")

        logger.info("Loading core: %s", core_path)
        self.core = Mupen64PlusCore(core_path, str(LIB_DIR), str(DATA_DIR))
        self.core.startup()

        logger.info("Loading ROM: %s", ROM_PATH)
        self.core.load_rom(ROM_PATH)

        # Attach plugins
        self.core.attach_plugins(
            gfx=str(LIB_DIR / "mupen64plus-video-rice.so"),
            audio=str(LIB_DIR / "mupen64plus-audio-sdl.so"),
            input=str(LIB_DIR / "mupen64plus-input-sdl.so"),
            rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
        )

        logger.info("Starting emulation...")
        self.core.execute()

        time.sleep(3)
        self.memory = MemoryReader(self.core)
        self.memory.refresh_pointer()
        logger.info("Ready! Use 'help' for commands.")

    def take_snapshot(self) -> int:
        """Take an RDRAM snapshot. Returns snapshot index."""
        data = self.memory.snapshot()
        self.snapshots.append(data)
        idx = len(self.snapshots) - 1
        logger.info("Snapshot #%d taken (%d bytes)", idx, len(data))
        return idx

    def diff_snapshots(self, idx_a: int = -2, idx_b: int = -1, max_show: int = 50):
        """Diff two snapshots and show changed addresses."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots. Use 'snap' first.")
            return

        a = self.snapshots[idx_a]
        b = self.snapshots[idx_b]
        changes = []

        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                changes.append((i, a[i], b[i]))

        print(f"Found {len(changes)} changed bytes")
        if len(changes) > max_show:
            print(f"Showing first {max_show}:")

        for offset, old, new in changes[:max_show]:
            virt = 0x80000000 + offset
            print(f"  0x{virt:08X} (offset 0x{offset:06X}): {old:3d} (0x{old:02X}) -> {new:3d} (0x{new:02X})")

        # Look for clusters of changes (potential arrays/structs)
        if changes:
            self._analyze_clusters(changes)

    def _analyze_clusters(self, changes: list[tuple[int, int, int]]):
        """Find clusters of changed bytes (potential game state structures)."""
        if len(changes) < 2:
            return

        offsets = [c[0] for c in changes]
        clusters = []
        cluster_start = offsets[0]
        cluster_end = offsets[0]

        for i in range(1, len(offsets)):
            if offsets[i] - offsets[i - 1] <= 16:  # Within 16 bytes = same cluster
                cluster_end = offsets[i]
            else:
                clusters.append((cluster_start, cluster_end))
                cluster_start = offsets[i]
                cluster_end = offsets[i]
        clusters.append((cluster_start, cluster_end))

        if clusters:
            print(f"\nClusters of changes ({len(clusters)}):")
            for start, end in clusters[:20]:
                size = end - start + 1
                virt_start = 0x80000000 + start
                virt_end = 0x80000000 + end
                count = sum(1 for o, _, _ in changes if start <= o <= end)
                print(f"  0x{virt_start:08X} - 0x{virt_end:08X} ({size} bytes, {count} changed)")

    def search_value(self, value: int, size: int = 1):
        """Search for a value in the most recent snapshot."""
        if not self.snapshots:
            print("No snapshots. Use 'snap' first.")
            return

        data = self.snapshots[-1]
        matches = []

        if size == 1:
            val_byte = value & 0xFF
            for i in range(len(data)):
                if data[i] == val_byte:
                    matches.append(i)
        elif size == 2:
            # Search both byte orders
            be = struct.pack(">H", value & 0xFFFF)
            le = struct.pack("<H", value & 0xFFFF)
            for i in range(len(data) - 1):
                chunk = data[i : i + 2]
                if chunk == be or chunk == le:
                    matches.append(i)
        elif size == 4:
            be = struct.pack(">I", value & 0xFFFFFFFF)
            le = struct.pack("<I", value & 0xFFFFFFFF)
            for i in range(len(data) - 3):
                chunk = data[i : i + 4]
                if chunk == be or chunk == le:
                    matches.append(i)

        print(f"Found {len(matches)} matches for {value} (0x{value:X}) as {size}-byte value")
        for offset in matches[:30]:
            virt = 0x80000000 + offset
            context = data[offset : offset + 8]
            hex_context = " ".join(f"{b:02X}" for b in context)
            print(f"  0x{virt:08X} (offset 0x{offset:06X}): {hex_context}")
        if len(matches) > 30:
            print(f"  ... and {len(matches) - 30} more")

    def read_address(self, addr: int, size: int = 1) -> int:
        """Read a value at a virtual address."""
        if size == 1:
            val = self.memory.read_u8(addr)
        elif size == 2:
            val = self.memory.read_u16(addr)
        elif size == 4:
            val = self.memory.read_u32(addr)
        else:
            val = 0
        return val

    def read_block(self, addr: int, count: int):
        """Read and display a block of bytes."""
        data = self.memory.read_block(addr, count)
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            virt = addr + i
            print(f"  0x{virt:08X}: {hex_part:<48s} {ascii_part}")

    def watch_address(self, addr: int, frames: int = 30, size: int = 2):
        """Watch an address, printing its value every N frames."""
        print(f"Watching 0x{addr:08X} every {frames} frames (Ctrl+C to stop)...")
        try:
            while True:
                val = self.read_address(addr, size)
                print(f"  0x{addr:08X} = {val} (0x{val:0{size*2}X})")
                for _ in range(frames):
                    self.core.advance_frame()
        except KeyboardInterrupt:
            print("Stopped watching.")

    def interactive_loop(self):
        """Main interactive command loop."""
        while True:
            try:
                line = input("\n[scanner] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            try:
                if cmd == "help":
                    print(__doc__)
                elif cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "snap":
                    self.take_snapshot()
                elif cmd == "diff":
                    self.diff_snapshots()
                elif cmd == "search" and len(parts) >= 2:
                    val = int(parts[1], 0)
                    self.search_value(val, size=1)
                elif cmd == "search16" and len(parts) >= 2:
                    val = int(parts[1], 0)
                    self.search_value(val, size=2)
                elif cmd == "search32" and len(parts) >= 2:
                    val = int(parts[1], 0)
                    self.search_value(val, size=4)
                elif cmd == "read" and len(parts) >= 2:
                    addr = int(parts[1], 0)
                    val = self.read_address(addr, 1)
                    print(f"  0x{addr:08X} = {val} (0x{val:02X})")
                elif cmd == "read16" and len(parts) >= 2:
                    addr = int(parts[1], 0)
                    val = self.read_address(addr, 2)
                    print(f"  0x{addr:08X} = {val} (0x{val:04X})")
                elif cmd == "read32" and len(parts) >= 2:
                    addr = int(parts[1], 0)
                    val = self.read_address(addr, 4)
                    print(f"  0x{addr:08X} = {val} (0x{val:08X})")
                elif cmd == "block" and len(parts) >= 3:
                    addr = int(parts[1], 0)
                    count = int(parts[2], 0)
                    self.read_block(addr, count)
                elif cmd == "watch" and len(parts) >= 2:
                    addr = int(parts[1], 0)
                    frames = int(parts[2]) if len(parts) >= 3 else 30
                    self.watch_address(addr, frames)
                elif cmd == "pause":
                    self.core.pause()
                    print("Paused.")
                elif cmd == "resume":
                    self.core.resume()
                    print("Resumed.")
                elif cmd == "step":
                    n = int(parts[1]) if len(parts) >= 2 else 1
                    for _ in range(n):
                        self.core.advance_frame()
                    print(f"Advanced {n} frame(s).")
                elif cmd == "save" and len(parts) >= 2:
                    slot = int(parts[1])
                    self.core.save_state(slot)
                    print(f"Saved to slot {slot}.")
                elif cmd == "load" and len(parts) >= 2:
                    slot = int(parts[1])
                    self.core.load_state(slot)
                    print(f"Loaded from slot {slot}.")
                elif cmd == "score":
                    a = self.read_address(mm.ADDR_SCORE_A, 2)
                    b = self.read_address(mm.ADDR_SCORE_B, 2)
                    print(f"  Score A (0x{mm.ADDR_SCORE_A:08X}) = {a}")
                    print(f"  Score B (0x{mm.ADDR_SCORE_B:08X}) = {b}")
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for usage.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    scanner = MemoryScanner()
    try:
        scanner.start_emulator()
        scanner.interactive_loop()
    except Exception:
        logger.exception("Fatal error")
    finally:
        if scanner.core:
            scanner.core.shutdown()


if __name__ == "__main__":
    main()
