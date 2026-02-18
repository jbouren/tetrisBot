"""Complete game state read from emulator memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import memory_map as mm
from .board import Board
from .n64_piece_map import n64_to_piece_type
from .pieces import PieceType

if TYPE_CHECKING:
    from ..emulator.memory import MemoryReader


@dataclass
class GameState:
    """Snapshot of the complete game state at a single frame."""

    board: Board
    current_piece: PieceType
    current_x: int
    current_y: int
    current_rotation: int
    next_piece: PieceType
    score: int
    lines: int
    level: int
    is_playing: bool
    is_game_over: bool

    @classmethod
    def from_memory(cls, mem: MemoryReader) -> GameState:
        """Read complete game state from RDRAM.

        Requires memory addresses to be discovered first.
        Falls back to defaults for undiscovered addresses.
        """
        # Board
        if mm.is_address_discovered(mm.ADDR_BOARD_BASE):
            board = Board.from_memory(mem)
        else:
            board = Board()

        # Current piece
        if mm.is_address_discovered(mm.ADDR_CURRENT_PIECE):
            n64_id = mem.read_u8(mm.ADDR_CURRENT_PIECE)
            current_piece = n64_to_piece_type(n64_id)
            if current_piece is None:
                # Log unknown piece ID?
                current_piece = PieceType.I # Fallback
        else:
            current_piece = PieceType.I

        # Piece position
        current_x = (
            mem.read_u16_le(mm.ADDR_PIECE_X)
            if mm.is_address_discovered(mm.ADDR_PIECE_X)
            else 4
        )
        current_y = (
            mem.read_u16_le(mm.ADDR_PIECE_Y)
            if mm.is_address_discovered(mm.ADDR_PIECE_Y)
            else 0
        )
        current_rotation = (
            mem.read_u8(mm.ADDR_PIECE_ROT)
            if mm.is_address_discovered(mm.ADDR_PIECE_ROT)
            else 0
        )

        # Next piece
        if mm.is_address_discovered(mm.ADDR_NEXT_PIECE):
            try:
                next_piece = PieceType(mem.read_u8(mm.ADDR_NEXT_PIECE))
            except ValueError:
                next_piece = PieceType.T
        else:
            next_piece = PieceType.T

        # Score (known from GameShark)
        score = mem.read_u16(mm.ADDR_SCORE_A)

        # Lines and level
        lines = (
            mem.read_u16(mm.ADDR_LINES)
            if mm.is_address_discovered(mm.ADDR_LINES)
            else 0
        )
        level = (
            mem.read_u8(mm.ADDR_LEVEL)
            if mm.is_address_discovered(mm.ADDR_LEVEL)
            else 0
        )

        # Game state
        if mm.is_address_discovered(mm.ADDR_GAME_STATE):
            gs = mem.read_u16(mm.ADDR_GAME_STATE)
            is_playing = gs == mm.GAME_STATE_PLAYING
            is_game_over = gs == mm.GAME_STATE_GAMEOVER
        else:
            is_playing = True
            is_game_over = False

        return cls(
            board=board,
            current_piece=current_piece,
            current_x=current_x,
            current_y=current_y,
            current_rotation=current_rotation,
            next_piece=next_piece,
            score=score,
            lines=lines,
            level=level,
            is_playing=is_playing,
            is_game_over=is_game_over,
        )

    def summary(self) -> str:
        """One-line summary of the game state."""
        return (
            f"Piece={self.current_piece.name} "
            f"@({self.current_x},{self.current_y}) "
            f"rot={self.current_rotation} "
            f"next={self.next_piece.name} "
            f"score={self.score} "
            f"lines={self.lines} "
            f"level={self.level}"
        )
