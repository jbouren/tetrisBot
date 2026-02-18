"""Tetris board representation and parsing from RDRAM.

The board is 10 columns wide and 20 rows tall. Row 0 is the top.
Each cell stores color and connection information for The New Tetris's
unique square-building mechanic.
"""

from __future__ import annotations

import numpy as np

from ..emulator.memory import MemoryReader
from . import memory_map as mm


class Cell:
    """A single cell on the Tetris board.

    In The New Tetris, each cell is a single byte.
    - Bits 7-4: color (0 = empty, 1-7 = piece colors, 15=silver, etc.)
    - Bit 3: unused
    - Bit 2: connection right
    - Bit 1: connection down
    - Bit 0: unused
    """

    __slots__ = ("raw", "color", "occupied", "conn_right", "conn_down")

    def __init__(self, raw: int = 0):
        self.raw = raw
        self.color = (raw >> 4) & 0xF
        self.conn_right = bool(raw & 0x4)  # Bit 2
        self.conn_down = bool(raw & 0x2)   # Bit 1
        self.occupied = self.color != 0

    def __repr__(self):
        if not self.occupied:
            return "."
        return str(self.color)


class Board:
    """10-wide by 20-tall Tetris board."""

    WIDTH = 10
    HEIGHT = 20

    def __init__(self, cells: list[list[Cell]] | None = None):
        if cells is not None:
            self.cells = cells
        else:
            self.cells = [
                [Cell() for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)
            ]

    @classmethod
    def from_memory(cls, mem: MemoryReader) -> Board:
        """Parse board state from RDRAM."""
        if not mm.is_address_discovered(mm.ADDR_BOARD_BASE):
            raise RuntimeError(
                "Board base address not discovered yet. "
                "Run tools/memory_scanner.py first."
            )

        total_bytes = cls.HEIGHT * cls.WIDTH * mm.BOARD_CELL_SIZE
        raw_data = mem.read_block(mm.ADDR_BOARD_BASE, total_bytes)

        cells = []
        for row in range(cls.HEIGHT):
            row_cells = []
            for col in range(cls.WIDTH):
                offset = (row * cls.WIDTH + col) * mm.BOARD_CELL_SIZE
                if mm.BOARD_CELL_SIZE == 2:
                    raw = (raw_data[offset] << 8) | raw_data[offset + 1]
                else:
                    raw = raw_data[offset]
                row_cells.append(Cell(raw))
            cells.append(row_cells)
        return cls(cells)

    @classmethod
    def from_occupancy(cls, grid: np.ndarray) -> Board:
        """Create a board from a 20x10 boolean occupancy grid.

        Used for testing AI without a real emulator.
        """
        cells = []
        for row in range(cls.HEIGHT):
            row_cells = []
            for col in range(cls.WIDTH):
                cell = Cell()
                if grid[row, col]:
                    cell.color = 1
                    cell.occupied = True
                row_cells.append(cell)
            cells.append(row_cells)
        return cls(cells)

    # ── Grid representations ────────────────────────────────────────────────

    def to_occupancy_grid(self) -> np.ndarray:
        """Return 20x10 boolean array. True = occupied."""
        grid = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
        for r in range(self.HEIGHT):
            for c in range(self.WIDTH):
                grid[r, c] = self.cells[r][c].occupied
        return grid

    def to_color_grid(self) -> np.ndarray:
        """Return 20x10 int array of color values (0 = empty)."""
        grid = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int8)
        for r in range(self.HEIGHT):
            for c in range(self.WIDTH):
                grid[r, c] = self.cells[r][c].color
        return grid

    # ── Board metrics ───────────────────────────────────────────────────────

    def column_heights(self) -> np.ndarray:
        """Height of each column (0 = empty, 20 = full).

        Height = number of rows from the bottom to the topmost occupied cell.
        """
        grid = self.to_occupancy_grid()
        heights = np.zeros(self.WIDTH, dtype=int)
        for c in range(self.WIDTH):
            for r in range(self.HEIGHT):
                if grid[r, c]:
                    heights[c] = self.HEIGHT - r
                    break
        return heights

    def count_holes(self) -> int:
        """Count empty cells with at least one occupied cell above them."""
        grid = self.to_occupancy_grid()
        holes = 0
        for c in range(self.WIDTH):
            found_block = False
            for r in range(self.HEIGHT):
                if grid[r, c]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def count_complete_lines(self) -> int:
        """Count rows that are completely filled."""
        grid = self.to_occupancy_grid()
        return int(grid.all(axis=1).sum())

    def get_complete_lines(self) -> list[int]:
        """Return row indices of complete lines."""
        grid = self.to_occupancy_grid()
        return [r for r in range(self.HEIGHT) if grid[r].all()]

    def clear_lines(self) -> tuple[Board, int]:
        """Return a new board with complete lines removed and count of lines cleared."""
        complete = self.get_complete_lines()
        if not complete:
            return Board([row[:] for row in self.cells]), 0

        # Keep non-complete rows
        remaining = [
            [Cell(self.cells[r][c].raw) for c in range(self.WIDTH)]
            for r in range(self.HEIGHT)
            if r not in complete
        ]
        # Add empty rows at top
        empty_rows = [
            [Cell() for _ in range(self.WIDTH)] for _ in range(len(complete))
        ]
        return Board(empty_rows + remaining), len(complete)

    def bumpiness(self) -> int:
        """Sum of absolute height differences between adjacent columns."""
        heights = self.column_heights()
        return int(np.abs(np.diff(heights)).sum())

    def aggregate_height(self) -> int:
        """Sum of all column heights."""
        return int(self.column_heights().sum())

    def max_height(self) -> int:
        """Height of the tallest column."""
        return int(self.column_heights().max())

    # ── Display ─────────────────────────────────────────────────────────────

    def to_ascii(self) -> str:
        """Render the board as an ASCII art string."""
        lines = []
        lines.append("+" + "-" * self.WIDTH + "+")
        for r in range(self.HEIGHT):
            row_str = ""
            for c in range(self.WIDTH):
                cell = self.cells[r][c]
                if cell.occupied:
                    row_str += "#" if cell.color else "?"
                else:
                    row_str += "."
            lines.append("|" + row_str + "|")
        lines.append("+" + "-" * self.WIDTH + "+")
        return "\n".join(lines)

    def __repr__(self):
        return self.to_ascii()
