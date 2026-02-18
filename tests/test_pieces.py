"""Tests for piece definitions and rotation tables."""

import pytest

from src.game.pieces import (
    PIECE_SHAPES,
    ROTATION_COUNT,
    PieceType,
    get_cells,
    get_height,
    get_width,
    normalize_cells,
)


class TestPieceType:
    def test_all_seven_pieces(self):
        assert len(PieceType) == 7
        assert PieceType.I == 0
        assert PieceType.Z == 6

    def test_piece_shapes_all_defined(self):
        for piece in PieceType:
            assert piece in PIECE_SHAPES
            assert len(PIECE_SHAPES[piece]) == 4  # 4 rotation states

    def test_each_rotation_has_4_cells(self):
        for piece in PieceType:
            for rot in range(4):
                cells = get_cells(piece, rot)
                assert len(cells) == 4, f"{piece.name} rot {rot} has {len(cells)} cells"


class TestPieceDimensions:
    def test_i_piece_horizontal(self):
        assert get_width(PieceType.I, 0) == 4
        assert get_height(PieceType.I, 0) == 1

    def test_i_piece_vertical(self):
        assert get_width(PieceType.I, 1) == 1
        assert get_height(PieceType.I, 1) == 4

    def test_o_piece_all_rotations(self):
        for rot in range(4):
            assert get_width(PieceType.O, rot) == 2
            assert get_height(PieceType.O, rot) == 2

    def test_t_piece_dimensions(self):
        # T spawn: 3 wide, 2 tall
        assert get_width(PieceType.T, 0) == 3
        assert get_height(PieceType.T, 0) == 2

    def test_s_piece_dimensions(self):
        assert get_width(PieceType.S, 0) == 3
        assert get_height(PieceType.S, 0) == 2
        assert get_width(PieceType.S, 1) == 2
        assert get_height(PieceType.S, 1) == 3

    def test_z_piece_dimensions(self):
        assert get_width(PieceType.Z, 0) == 3
        assert get_height(PieceType.Z, 0) == 2
        assert get_width(PieceType.Z, 1) == 2
        assert get_height(PieceType.Z, 1) == 3


class TestRotationCount:
    def test_i_piece_2_rotations(self):
        assert ROTATION_COUNT[PieceType.I] == 2

    def test_o_piece_1_rotation(self):
        assert ROTATION_COUNT[PieceType.O] == 1

    def test_t_piece_4_rotations(self):
        assert ROTATION_COUNT[PieceType.T] == 4

    def test_s_z_pieces_4_rotations(self):
        assert ROTATION_COUNT[PieceType.S] == 4
        assert ROTATION_COUNT[PieceType.Z] == 4

    def test_j_l_pieces_4_rotations(self):
        assert ROTATION_COUNT[PieceType.J] == 4
        assert ROTATION_COUNT[PieceType.L] == 4


class TestGetCells:
    def test_wraps_rotation(self):
        """Rotation 4 should be same as rotation 0."""
        cells_0 = get_cells(PieceType.T, 0)
        cells_4 = get_cells(PieceType.T, 4)
        assert cells_0 == cells_4

    def test_all_cells_non_negative(self):
        """All cell offsets should be >= 0."""
        for piece in PieceType:
            for rot in range(4):
                cells = get_cells(piece, rot)
                for r, c in cells:
                    assert r >= 0, f"{piece.name} rot {rot}: negative row {r}"
                    assert c >= 0, f"{piece.name} rot {rot}: negative col {c}"

    def test_no_duplicate_cells(self):
        """Each rotation should have 4 unique cell positions."""
        for piece in PieceType:
            for rot in range(4):
                cells = get_cells(piece, rot)
                assert len(set(cells)) == 4, f"{piece.name} rot {rot} has duplicate cells"


class TestNormalizeCells:
    def test_already_normalized(self):
        cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert normalize_cells(cells) == sorted(cells)

    def test_shift_to_origin(self):
        cells = [(2, 3), (2, 4), (3, 3), (3, 4)]
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert normalize_cells(cells) == expected

    def test_preserves_shape(self):
        """Normalizing should not change the relative positions."""
        cells = [(5, 5), (5, 6), (5, 7), (5, 8)]
        result = normalize_cells(cells)
        assert result == [(0, 0), (0, 1), (0, 2), (0, 3)]


class TestSymmetricPieces:
    def test_i_piece_symmetry(self):
        """I-piece rotation 0 and 2 should be identical."""
        cells_0 = normalize_cells(get_cells(PieceType.I, 0))
        cells_2 = normalize_cells(get_cells(PieceType.I, 2))
        assert cells_0 == cells_2

    def test_o_piece_all_same(self):
        """O-piece should be the same in all rotations."""
        base = normalize_cells(get_cells(PieceType.O, 0))
        for rot in range(1, 4):
            assert normalize_cells(get_cells(PieceType.O, rot)) == base

    def test_s_piece_symmetry(self):
        """S-piece rotation 0 and 2 should be identical."""
        cells_0 = normalize_cells(get_cells(PieceType.S, 0))
        cells_2 = normalize_cells(get_cells(PieceType.S, 2))
        assert cells_0 == cells_2

    def test_z_piece_symmetry(self):
        """Z-piece rotation 0 and 2 should be identical."""
        cells_0 = normalize_cells(get_cells(PieceType.Z, 0))
        cells_2 = normalize_cells(get_cells(PieceType.Z, 2))
        assert cells_0 == cells_2
