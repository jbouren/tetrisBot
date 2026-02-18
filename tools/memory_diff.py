#!/usr/bin/env python3
"""Diff two RDRAM snapshot files to find changed addresses.

Usage:
    # Save snapshots from memory_scanner.py, then diff offline:
    python tools/memory_diff.py snapshot_before.bin snapshot_after.bin

    # Or use interactively via memory_scanner.py's 'snap' + 'diff' commands.

This tool is for offline analysis of saved RDRAM snapshots. It finds all
bytes that differ between two snapshots and groups them into clusters,
helping identify game state structures (board, score, piece, etc.).
"""

import struct
import sys
from pathlib import Path


def load_snapshot(path: str) -> bytes:
    """Load a raw RDRAM snapshot from disk."""
    data = Path(path).read_bytes()
    if len(data) != 0x800000:
        print(f"WARNING: Expected 8MB (0x800000), got {len(data)} bytes")
    return data


def diff_snapshots(before: bytes, after: bytes) -> list[tuple[int, int, int]]:
    """Find all byte offsets that changed.

    Returns list of (offset, old_value, new_value).
    """
    changes = []
    length = min(len(before), len(after))
    for i in range(length):
        if before[i] != after[i]:
            changes.append((i, before[i], after[i]))
    return changes


def find_clusters(changes: list[tuple[int, int, int]], gap: int = 16) -> list[list[tuple[int, int, int]]]:
    """Group changed bytes into clusters (within `gap` bytes of each other)."""
    if not changes:
        return []

    clusters = []
    current_cluster = [changes[0]]

    for i in range(1, len(changes)):
        if changes[i][0] - changes[i - 1][0] <= gap:
            current_cluster.append(changes[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [changes[i]]
    clusters.append(current_cluster)
    return clusters


def analyze_cluster(cluster: list[tuple[int, int, int]], before: bytes, after: bytes):
    """Print detailed analysis of a cluster of changes."""
    start = cluster[0][0]
    end = cluster[-1][0]
    virt_start = 0x80000000 + start
    virt_end = 0x80000000 + end

    print(f"\n  Cluster: 0x{virt_start:08X} - 0x{virt_end:08X} "
          f"({end - start + 1} bytes span, {len(cluster)} changed)")

    # Show context: 16-byte aligned block around the cluster
    block_start = (start // 16) * 16
    block_end = ((end // 16) + 1) * 16

    for row_offset in range(block_start, min(block_end, len(before)), 16):
        virt = 0x80000000 + row_offset
        before_hex = []
        after_hex = []
        for i in range(16):
            off = row_offset + i
            if off < len(before):
                b = before[off]
                a = after[off]
                marker = "*" if b != a else " "
                before_hex.append(f"{b:02X}{marker}")
                after_hex.append(f"{a:02X}{marker}")
            else:
                before_hex.append("   ")
                after_hex.append("   ")

        print(f"    0x{virt:08X} before: {' '.join(before_hex)}")
        print(f"    0x{virt:08X} after:  {' '.join(after_hex)}")

    # Check for stride patterns (potential arrays)
    if len(cluster) >= 3:
        offsets = [c[0] for c in cluster]
        strides = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        unique_strides = set(strides)
        if len(unique_strides) <= 2:
            print(f"    Stride pattern: {sorted(unique_strides)} "
                  f"(possible array with element size {min(unique_strides)})")


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/memory_diff.py <before.bin> <after.bin>")
        print()
        print("To create snapshots, use memory_scanner.py:")
        print("  1. Run memory_scanner.py")
        print("  2. Use 'snap' command to take snapshots")
        print("  3. Snapshots are saved as snapshot_N.bin")
        sys.exit(1)

    before_path = sys.argv[1]
    after_path = sys.argv[2]

    print(f"Loading: {before_path}")
    before = load_snapshot(before_path)
    print(f"Loading: {after_path}")
    after = load_snapshot(after_path)

    print("Diffing...")
    changes = diff_snapshots(before, after)
    print(f"Total changed bytes: {len(changes)}")

    if not changes:
        print("No changes found!")
        return

    # Summary by memory region
    regions = {
        "0x800-0x80F": (0x00080000, 0x000FFFFF),
        "0x810-0x81F": (0x00100000, 0x001FFFFF),
        "0x820-0x82F": (0x00200000, 0x002FFFFF),
        "0x830-0x83F": (0x00300000, 0x003FFFFF),
    }

    print("\nChanges by region:")
    for name, (lo, hi) in regions.items():
        count = sum(1 for o, _, _ in changes if lo <= o <= hi)
        if count > 0:
            print(f"  {name}: {count} bytes changed")

    # Cluster analysis
    clusters = find_clusters(changes)
    print(f"\nFound {len(clusters)} clusters of changes:")

    for cluster in clusters[:30]:
        analyze_cluster(cluster, before, after)

    if len(clusters) > 30:
        print(f"\n  ... and {len(clusters) - 30} more clusters")

    # Suggest likely candidates
    print("\n=== Suggestions ===")
    print("Small clusters (1-4 bytes) are likely: score, level, piece type, position")
    print("Medium clusters (8-40 bytes) are likely: board rows, piece state structs")
    print("Large clusters (100+ bytes) are likely: full board state, video/audio buffers")

    small = [c for c in clusters if len(c) <= 4]
    medium = [c for c in clusters if 5 <= len(c) <= 40]
    large = [c for c in clusters if len(c) > 40]

    if small:
        print(f"\nSmall clusters ({len(small)}) - likely game state variables:")
        for cluster in small[:10]:
            virt = 0x80000000 + cluster[0][0]
            vals_before = [c[1] for c in cluster]
            vals_after = [c[2] for c in cluster]
            print(f"  0x{virt:08X}: {vals_before} -> {vals_after}")


if __name__ == "__main__":
    main()
