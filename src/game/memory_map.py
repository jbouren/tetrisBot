"""Memory addresses for The New Tetris (USA) - N64.

All addresses are N64 virtual addresses (KSEG0, 0x80XXXXXX).
Physical RDRAM offset = addr & 0x1FFFFFFF.

Known from GameShark codes:
  81YYYYYY ZZZZ  -> write 16-bit value ZZZZ to RDRAM offset YYYYYY
  D0YYYYYY 00ZZ  -> conditional: if byte at YYYYYY == ZZ, execute next code

Addresses marked TODO need to be discovered using tools/memory_scanner.py.
Run the game, perform known actions, and diff RDRAM snapshots.
"""

# ── Score ───────────────────────────────────────────────────────────────────
# From GameShark: 8111EED6 ???? and 8111EEB6 ????
# These appear to be score-related 16-bit values.
ADDR_SCORE_A = 0x8011EED6  # Primary score register (16-bit)
ADDR_SCORE_B = 0x8011EEB6  # Secondary score register (16-bit)

# ── Game Mode ───────────────────────────────────────────────────────────────
# From GameShark: 810CFF60 38D1 (turbo mode)
ADDR_GAME_MODE = 0x800CFF60  # 16-bit, game mode flags

# ── Controller Input State ──────────────────────────────────────────────────
# From GameShark conditional: D01101B3 0024 (if byte at 0x1101B3 == 0x24)
# 0x801101B0 region appears to be controller/input state
ADDR_P1_INPUT = 0x801101B0  # Player 1 input state (32-bit)

# ── Board State (TODO: discover with memory_scanner.py) ────────────────────
# The board is 10 columns x 20 rows.
# Each cell likely 1-2 bytes: color (4 bits), connection bits, broken flag.
#
# Discovery results from tools/discover_focused.py on 2026-02-09:
#   - Board base address: 0x80368D27
#   - Board cell stride: 2 bytes
#   - Piece Y position candidate: 0x8010DE86
#   - Piece type candidate: 0x8010BC24
#
# Discovery strategy:
#   1. Use tools/navigate_and_save.py to create a save state at game start.
#   2. Use tools/discover_focused.py to load the state, drop pieces, and
#      diff RDRAM snapshots.
#   3. Analyze stride of unique 0->nonzero memory changes to find the board.
#   4. Analyze values that change consistently with piece drops for position/type.
ADDR_BOARD_BASE = 0x80368D28  # Base address of the 10x20 board grid
BOARD_CELL_SIZE = 1           # Stride between cells is 1 byte
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# ── Current Piece (TODO) ───────────────────────────────────────────────────
# Piece type cycles through 7 values (I=0, J=1, L=2, O=3, S=4, T=5, Z=6)
# or possibly different encoding.
ADDR_CURRENT_PIECE = 0x800009B6  # Strong candidate from discover_v4.py
ADDR_PIECE_X = 0x8010DFFC       # u16_le, piece X position on board
ADDR_PIECE_Y = 0x8011EF36       # u16_le, piece Y position on board
ADDR_PIECE_ROT = 0x80000407     # TODO: rotation state (0-3)

# ── Next Piece (TODO) ──────────────────────────────────────────────────────
ADDR_NEXT_PIECE = 0x80000000    # TODO: discover

# ── Game State (TODO) ──────────────────────────────────────────────────────
ADDR_GAME_STATE = 0x800CFF60    # From GameShark code 810CFF60 38D1
ADDR_LEVEL = 0x80000000         # TODO: current level
ADDR_LINES = 0x80000000         # TODO: total lines cleared

# ── Game State Constants (TODO: verify values) ─────────────────────────────
GAME_STATE_MENU = 0
GAME_STATE_PLAYING = 14577     # Value observed when gameplay is active
GAME_STATE_PAUSED = 2
GAME_STATE_GAMEOVER = 3


def is_address_discovered(addr: int) -> bool:
    """Check if an address has been discovered (not placeholder)."""
    return addr != 0x80000000
