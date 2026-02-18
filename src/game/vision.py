"""Computer vision module for reading game state from emulator screenshots.

Reads the 10x20 board grid, identifies piece colors, and detects the current
piece, next piece, and other UI elements from captured frames.
"""

import numpy as np
from collections import deque

from src.game.pieces import PieceType

# Board grid bounds (pixel coordinates at 640x480 resolution)
BOARD_LEFT = 135
BOARD_TOP = 52
BOARD_RIGHT = 305
BOARD_BOTTOM = 405
BOARD_COLS = 10
BOARD_ROWS = 20

CELL_WIDTH = (BOARD_RIGHT - BOARD_LEFT) / BOARD_COLS   # ~17.0
CELL_HEIGHT = (BOARD_BOTTOM - BOARD_TOP) / BOARD_ROWS   # ~17.65

# Brightness threshold: pixels above this in a cell = occupied
OCCUPIED_THRESHOLD = 40

# Sample region: fraction of cell size to sample around center (avoids edges)
SAMPLE_FRACTION = 0.4

# Preview box bounds (right side of screen, 3 next pieces stacked vertically)
PREVIEW_LEFT = 310
PREVIEW_TOP = 40
PREVIEW_RIGHT = 360
PREVIEW_BOTTOM = 200

# X range where actual piece pixels appear (excluding ornate border)
PREVIEW_PIECE_X0 = 316
PREVIEW_PIECE_X1 = 358

# Y ranges for each preview slot (3 slots)
PREVIEW_SLOTS = [
    (66, 88),    # Slot 0 (next piece)
    (118, 142),  # Slot 1
    (174, 198),  # Slot 2
]

# Reserve/swap piece box (left side of screen)
RESERVE_LEFT = 65
RESERVE_TOP = 55
RESERVE_RIGHT = 120
RESERVE_BOTTOM = 100

# Brightness threshold for preview piece pixels
PREVIEW_BRIGHTNESS_THRESHOLD = 60

# Reference colors for piece type identification (RGB).
# Measured from The New Tetris preview area at 640x480.
# Mapping confirmed via TNT64pieces.png reference.
PIECE_COLORS: dict[PieceType, np.ndarray] = {
    PieceType.I: np.array([40, 135, 190]),    # Cyan
    PieceType.O: np.array([146, 144, 142]),   # Gray/White
    PieceType.T: np.array([186, 169, 24]),    # Yellow/Gold
    PieceType.S: np.array([65, 173, 87]),     # Green
    PieceType.Z: np.array([168, 62, 62]),     # Red
    PieceType.L: np.array([168, 58, 150]),    # Pink (confirmed from drop 9)
    PieceType.J: np.array([108, 55, 168]),    # Purple/Blue (confirmed from j_candidate_19)
}


def get_cell_center(col: int, row: int) -> tuple[float, float]:
    """Get the pixel center of a board cell (col 0-9, row 0-19, top-down)."""
    cx = BOARD_LEFT + (col + 0.5) * CELL_WIDTH
    cy = BOARD_TOP + (row + 0.5) * CELL_HEIGHT
    return cx, cy


def sample_cell(frame: np.ndarray, col: int, row: int) -> np.ndarray:
    """Sample a small region around the center of a board cell.

    Returns the mean RGB values of the sampled region.
    """
    cx, cy = get_cell_center(col, row)
    half_w = CELL_WIDTH * SAMPLE_FRACTION / 2
    half_h = CELL_HEIGHT * SAMPLE_FRACTION / 2

    x0 = max(0, int(cx - half_w))
    x1 = min(frame.shape[1], int(cx + half_w))
    y0 = max(0, int(cy - half_h))
    y1 = min(frame.shape[0], int(cy + half_h))

    region = frame[y0:y1, x0:x1]
    return region.mean(axis=(0, 1))  # mean RGB


def read_board(frame: np.ndarray, threshold: float = OCCUPIED_THRESHOLD) -> np.ndarray:
    """Read the 10x20 board grid from a frame.

    Args:
        frame: RGB numpy array (480, 640, 3) from read_screen().
        threshold: Brightness threshold for occupied cells.
                   Default 40 catches ghost pieces too.
                   Use ~65-75 to filter ghost pieces.

    Returns:
        20x10 numpy bool array. True = occupied, False = empty.
        Row 0 is the top of the board.
    """
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            mean_rgb = sample_cell(frame, col, row)
            brightness = mean_rgb.mean()
            board[row, col] = brightness > threshold

    return board


def read_board_brightness(frame: np.ndarray) -> np.ndarray:
    """Read the brightness of every board cell. Returns 20x10 float array."""
    brightness = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=float)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            mean_rgb = sample_cell(frame, col, row)
            brightness[row, col] = mean_rgb.mean()
    return brightness


def read_board_clean(frame: np.ndarray, threshold: float = 65,
                     ghost_max: float = 65) -> np.ndarray:
    """Read board with ghost piece removal. Single-pass convenience function.

    Reads brightness, applies threshold, then strips any remaining dim cells.
    Default threshold=65 excludes ghost pieces (brightness 40-55) entirely.
    """
    brightness = read_board_brightness(frame)
    board = brightness > threshold
    return strip_ghost(board, brightness, ghost_max)


def strip_ghost(board: np.ndarray, brightness: np.ndarray,
                ghost_max: float = 65) -> np.ndarray:
    """Remove ghost piece cells from a board.

    Ghost pieces in TNT have brightness 40-55 and sit at the projected
    landing position. This strips ALL dim cells (brightness <= ghost_max)
    anywhere on the board, not just at the bottom — because ghosts can
    appear on top of existing settled pieces.

    Args:
        board: 20x10 bool array from read_board().
        brightness: 20x10 float array from read_board_brightness().
        ghost_max: Maximum brightness for ghost cells (cells brighter
                   than this are definitely real).

    Returns:
        20x10 bool array with ghost cells removed.
    """
    # Remove ALL dim cells — ghost can appear anywhere (above settled pieces)
    ghost_mask = board & (brightness <= ghost_max)
    return board & ~ghost_mask


def read_board_colors(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read board occupancy and color data.

    Returns:
        occupancy: 20x10 bool array (True = occupied)
        colors: 20x10x3 float array (mean RGB of each cell)
    """
    occupancy = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
    colors = np.zeros((BOARD_ROWS, BOARD_COLS, 3), dtype=float)

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            mean_rgb = sample_cell(frame, col, row)
            colors[row, col] = mean_rgb
            brightness = mean_rgb.mean()
            occupancy[row, col] = brightness > OCCUPIED_THRESHOLD

    return occupancy, colors


def board_to_ascii(board: np.ndarray) -> str:
    """Convert a 20x10 bool board to ASCII art for debugging."""
    lines = []
    lines.append("+" + "-" * BOARD_COLS + "+")
    for row in range(BOARD_ROWS):
        cells = ""
        for col in range(BOARD_COLS):
            cells += "#" if board[row, col] else "."
        lines.append("|" + cells + "|")
    lines.append("+" + "-" * BOARD_COLS + "+")
    return "\n".join(lines)


# --- Piece color classification ---

def classify_color(rgb: np.ndarray) -> PieceType | None:
    """Classify an RGB color to the nearest piece type.

    Returns None if the color doesn't match any known piece well enough.
    """
    best_piece = None
    best_dist = float("inf")

    for piece_type, ref_color in PIECE_COLORS.items():
        dist = np.linalg.norm(rgb - ref_color)
        if dist < best_dist:
            best_dist = dist
            best_piece = piece_type

    # Reject if too far from any reference (>80 RGB distance)
    if best_dist > 80:
        return None

    return best_piece


# --- Preview reading ---

def sample_preview_slot(frame: np.ndarray, slot: int) -> np.ndarray | None:
    """Sample the average color of bright pixels in a preview slot.

    Args:
        frame: RGB frame (480, 640, 3).
        slot: Preview slot index (0=next, 1, 2).

    Returns:
        Mean RGB of the piece pixels, or None if no piece visible.
    """
    y0, y1 = PREVIEW_SLOTS[slot]
    region = frame[y0:y1, PREVIEW_PIECE_X0:PREVIEW_PIECE_X1]

    brightness = region.mean(axis=2)
    bright_mask = brightness > PREVIEW_BRIGHTNESS_THRESHOLD

    if bright_mask.sum() < 15:
        return None

    return region[bright_mask].mean(axis=0)


def read_preview(frame: np.ndarray) -> list[PieceType | None]:
    """Read the 3 preview pieces from the frame.

    Returns:
        List of 3 piece types (or None for unreadable slots).
        Index 0 = next piece, 1 and 2 = further ahead.
    """
    result = []
    for slot in range(3):
        rgb = sample_preview_slot(frame, slot)
        if rgb is None:
            result.append(None)
        else:
            result.append(classify_color(rgb))
    return result


def read_reserve_piece(frame: np.ndarray) -> PieceType | None:
    """Read the reserve/swap piece from the left side of the screen.

    Returns:
        The piece type in the reserve slot, or None if empty/unreadable.
    """
    region = frame[RESERVE_TOP:RESERVE_BOTTOM, RESERVE_LEFT:RESERVE_RIGHT]
    brightness = region.mean(axis=2)
    bright_mask = brightness > PREVIEW_BRIGHTNESS_THRESHOLD

    if bright_mask.sum() < 15:
        return None

    avg_color = region[bright_mask].mean(axis=0)
    return classify_color(avg_color)


# --- Current piece detection ---

# Falling pieces are brighter than settled pieces in The New Tetris.
# Settled pieces dim to ~50-75 brightness, falling pieces are ~100-140.
FALLING_BRIGHTNESS_THRESHOLD = 85


def detect_falling_piece(
    current_board: np.ndarray,
    settled_board: np.ndarray,
    current_colors: np.ndarray,
) -> tuple[PieceType | None, list[tuple[int, int]]]:
    """Detect the falling piece by diffing current frame against settled board.

    Args:
        current_board: 20x10 bool occupancy from current frame.
        settled_board: 20x10 bool occupancy of only settled pieces (no falling piece).
        current_colors: 20x10x3 RGB colors from current frame.

    Returns:
        (piece_type, cells) where cells is list of (row, col) positions.
        piece_type may be None if detection fails.
    """
    new_cells = current_board & ~settled_board
    positions = list(zip(*np.where(new_cells)))

    if len(positions) == 0:
        return None, []

    # Get average color of the new cells
    piece_colors = np.array([current_colors[r, c] for r, c in positions])
    avg_color = piece_colors.mean(axis=0)
    piece_type = classify_color(avg_color)

    return piece_type, positions


def detect_falling_piece_by_brightness(
    occupancy: np.ndarray,
    colors: np.ndarray,
) -> tuple[PieceType | None, list[tuple[int, int]]]:
    """Detect the falling piece by brightness (falling pieces are brighter).

    In The New Tetris, settled pieces dim to ~50-75 brightness while the
    active falling piece stays at ~100-140. This avoids needing a separate
    settled board reference.

    Returns:
        (piece_type, cells) where cells is list of (row, col) positions.
    """
    brightness = colors.mean(axis=2)
    bright_mask = occupancy & (brightness > FALLING_BRIGHTNESS_THRESHOLD)

    positions = list(zip(*np.where(bright_mask)))

    if len(positions) == 0:
        return None, []

    # Filter to keep only the 4 brightest cells (a tetromino has exactly 4)
    if len(positions) > 4:
        cell_brightness = [brightness[r, c] for r, c in positions]
        # Sort by brightness descending, keep top 4
        sorted_idx = np.argsort(cell_brightness)[::-1]
        positions = [positions[i] for i in sorted_idx[:4]]

    piece_colors = np.array([colors[r, c] for r, c in positions])
    avg_color = piece_colors.mean(axis=0)
    piece_type = classify_color(avg_color)

    return piece_type, positions

def _find_connected_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Find all distinct, connected groups of cells using BFS."""
    components = []
    visited = set()

    for start_cell in cells:
        if start_cell in visited:
            continue

        component = set()
        q = deque([start_cell])
        visited.add(start_cell)
        component.add(start_cell)

        while q:
            r, c = q.popleft()

            # Check 4 neighbors (up, down, left, right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r + dr, c + dc)
                if neighbor in cells and neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    q.append(neighbor)

        components.append(component)

    return components


def find_ghost_piece(board_with_piece: np.ndarray, settled_board: np.ndarray) -> set[tuple[int, int]]:
    """Find the coordinates of the ghost piece using connected components.

    The ghost piece is the set of new cells on the board that are not part
    of the currently falling piece. We find it by finding all disconnected
    "islands" of new cells and identifying the lower one as the ghost.

    Returns:
        A set of (row, col) tuples for the 4 ghost cells, or an empty set.
    """
    diff = board_with_piece & ~settled_board
    if not diff.any():
        return set()

    new_cells = set(zip(*np.where(diff)))
    if not new_cells:
        return set()

    components = _find_connected_components(new_cells)

    # Expect exactly two components: the falling piece and the ghost.
    # If not, the board is in a weird state (e.g. mid-animation), so fail gracefully.
    if len(components) != 2:
        return set()

    # The component with the lower min row is the falling piece.
    # The other one is the ghost.
    components.sort(key=lambda c: min(r for r, _ in c))

    falling_piece = components[0]
    ghost_piece = components[1]

    # Final sanity check: a valid ghost piece should have 4 cells.
    if len(ghost_piece) != 4:
        return set()

    return ghost_piece
