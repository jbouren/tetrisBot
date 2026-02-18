"""High-performance memory reader using direct RDRAM pointer access.

The N64 is big-endian. mupen64plus stores RDRAM in host byte order with
32-bit word granularity. On little-endian x86, sub-word reads need byte
swapping. The XOR trick handles this:
  - byte read at offset N:  actual offset = N ^ 3
  - halfword read at offset N: actual offset = N ^ 2
  - word read at offset N: no XOR needed (aligned 32-bit)

If empirical testing shows the XOR trick is wrong for this build, set
use_debug_api=True to fall back to DebugMemRead* function calls.
"""

import ctypes as ct
import struct

from .core import Mupen64PlusCore

# RDRAM is 4MB on standard N64, 8MB with expansion pak.
# mupen64plus always allocates 8MB.
RDRAM_SIZE = 0x800000  # 8 MB


class MemoryReader:
    """Reads N64 RDRAM through either direct pointer or debug API."""

    def __init__(self, core: Mupen64PlusCore, use_debug_api: bool = False):
        self._core = core
        self._use_debug_api = use_debug_api
        self._rdram_ptr: ct.c_void_p | None = None
        self._rdram_array: ct.Array | None = None

    def refresh_pointer(self):
        """Re-acquire the RDRAM pointer. Call after emulation starts."""
        if self._use_debug_api:
            return
        self._rdram_ptr = self._core.get_rdram_pointer()
        self._rdram_array = ct.cast(
            self._rdram_ptr, ct.POINTER(ct.c_uint8 * RDRAM_SIZE)
        ).contents

    @staticmethod
    def virt_to_phys(addr: int) -> int:
        """Convert N64 virtual address to RDRAM physical offset.

        KSEG0: 0x80000000-0x9FFFFFFF -> phys = addr & 0x1FFFFFFF
        KSEG1: 0xA0000000-0xBFFFFFFF -> phys = addr & 0x1FFFFFFF
        """
        return addr & 0x1FFFFFFF

    # ── Read operations ─────────────────────────────────────────────────────

    def read_u8(self, addr: int) -> int:
        """Read unsigned byte from N64 virtual address (0x80XXXXXX)."""
        if self._use_debug_api:
            return self._core.debug_read_8(addr)
        offset = self.virt_to_phys(addr)
        return self._rdram_array[offset ^ 3]

    def read_u16(self, addr: int) -> int:
        """Read unsigned 16-bit value (big-endian) from N64 virtual address."""
        if self._use_debug_api:
            return self._core.debug_read_16(addr)
        offset = self.virt_to_phys(addr)
        # XOR with 2 for halfword access within a 32-bit word
        swapped = offset ^ 2
        b0 = self._rdram_array[swapped]
        b1 = self._rdram_array[swapped + 1]
        return (b0 << 8) | b1

    def read_u16_le(self, addr: int) -> int:
        """Read unsigned 16-bit value (little-endian) from N64 virtual address."""
        if self._use_debug_api:
            # Fallback needs to handle endianness
            val = self._core.debug_read_16(addr)
            return ((val & 0xFF) << 8) | (val >> 8)
        offset = self.virt_to_phys(addr)
        # XOR with 2 for halfword access within a 32-bit word
        swapped = offset ^ 2
        b0 = self._rdram_array[swapped]
        b1 = self._rdram_array[swapped + 1]
        return (b1 << 8) | b0


    def read_u32(self, addr: int) -> int:
        """Read unsigned 32-bit value from N64 virtual address."""
        if self._use_debug_api:
            return self._core.debug_read_32(addr)
        offset = self.virt_to_phys(addr)
        # 32-bit aligned reads: no XOR needed, but stored in host order
        return struct.unpack_from(
            "<I", bytes(self._rdram_array[offset : offset + 4])
        )[0]

    def read_s8(self, addr: int) -> int:
        """Read signed byte."""
        val = self.read_u8(addr)
        return val if val < 128 else val - 256

    def read_s16(self, addr: int) -> int:
        """Read signed 16-bit value."""
        val = self.read_u16(addr)
        return val if val < 32768 else val - 65536

    def read_s16_le(self, addr: int) -> int:
        """Read signed 16-bit little-endian value."""
        val = self.read_u16_le(addr)
        return val if val < 32768 else val - 65536


    def read_block(self, addr: int, size: int) -> bytes:
        """Read a contiguous block of bytes from RDRAM.

        Returns bytes in N64 (big-endian) byte order, with sub-word
        byte swapping applied.
        """
        if self._use_debug_api:
            # Slow path: read byte by byte
            return bytes(self._core.debug_read_8(addr + i) for i in range(size))
        offset = self.virt_to_phys(addr)
        result = bytearray(size)
        for i in range(size):
            result[i] = self._rdram_array[(offset + i) ^ 3]
        return bytes(result)

    def read_block_fast(self, addr: int, size: int) -> bytes:
        """Read a block without byte swapping (raw host-order bytes).

        Use this when you'll handle endianness yourself or when reading
        32-bit aligned, 32-bit-multiple sized blocks.
        """
        if self._use_debug_api:
            return self.read_block(addr, size)
        offset = self.virt_to_phys(addr)
        return bytes(self._rdram_array[offset : offset + size])

    # ── Snapshot for memory scanning ────────────────────────────────────────

    def snapshot(self) -> bytes:
        """Take a complete 8MB snapshot of RDRAM (raw, no byte swapping).

        Used by memory discovery tools to diff before/after states.
        """
        if self._use_debug_api:
            raise RuntimeError("snapshot() requires direct pointer access")
        return bytes(self._rdram_array)
