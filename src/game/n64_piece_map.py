"""Mapping from in-game N64 piece IDs to our internal PieceType enum."""

from .pieces import PieceType

# This mapping was discovered empirically using tools/manual_piece_mapper.py.
# The game seems to use a different set of IDs than the standard 0-6.
# A Heisenbug causes the value to be 8 during live play but 114 from a clean load.
N64_PIECE_MAP = {
    # Game Value -> Internal PieceType
    # Discovered via tools/manual_piece_mapper.py
    8: PieceType.T,    # Value observed during live play
    114: PieceType.T,  # Value observed from a clean save state load
    115: PieceType.S,
    116: PieceType.Z,
    117: PieceType.L,
    118: PieceType.J,
    119: PieceType.I,
    120: PieceType.O,
}

def n64_to_piece_type(n64_id: int) -> PieceType | None:
    """Convert an N64 piece ID to a PieceType, or None if unknown."""
    return N64_PIECE_MAP.get(n64_id)
