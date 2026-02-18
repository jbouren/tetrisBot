#!/usr/bin/env bash
#
# Build mupen64plus from source with DEBUGGER=1 for the Tetris bot.
# All .so files are placed in ./lib/
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
LIB_DIR="$PROJECT_DIR/lib"
DATA_DIR="$LIB_DIR/data"

echo "=== Tetris Bot: mupen64plus build script ==="
echo "Project: $PROJECT_DIR"
echo "Build:   $BUILD_DIR"
echo "Output:  $LIB_DIR"
echo ""

# ── 1. Install system dependencies ──────────────────────────────────────────

echo ">>> Installing system dependencies..."
if sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential \
        git \
        nasm \
        pkg-config \
        libsdl2-dev \
        libpng-dev \
        libz-dev \
        libfreetype-dev \
        libjson-c-dev \
        libgl-dev \
        libglu1-mesa-dev \
        python3-dev \
        python3-pip \
        python3-venv \
        libminizip-dev \
        xvfb \
        2>/dev/null
else
    echo "  Skipping apt-get (no passwordless sudo). Install deps manually if needed:"
    echo "  sudo apt-get install -y build-essential git nasm pkg-config libsdl2-dev libpng-dev libz-dev libfreetype-dev libjson-c-dev libgl-dev libglu1-mesa-dev python3-dev python3-pip python3-venv libminizip-dev xvfb"
fi

# ── 2. Create directories ──────────────────────────────────────────────────

mkdir -p "$BUILD_DIR" "$LIB_DIR" "$DATA_DIR"

# ── 3. Clone repositories ──────────────────────────────────────────────────

REPOS=(
    "mupen64plus-core"
    "mupen64plus-ui-console"
    "mupen64plus-video-rice"
    "mupen64plus-audio-sdl"
    "mupen64plus-input-sdl"
    "mupen64plus-rsp-hle"
)

echo ""
echo ">>> Cloning mupen64plus repositories..."
for repo in "${REPOS[@]}"; do
    if [ -d "$BUILD_DIR/$repo" ]; then
        echo "  $repo: already cloned, pulling latest..."
        git -C "$BUILD_DIR/$repo" pull --quiet 2>/dev/null || true
    else
        echo "  $repo: cloning..."
        git clone --quiet --depth 1 \
            "https://github.com/mupen64plus/$repo.git" \
            "$BUILD_DIR/$repo"
    fi
done

# Also clone mupen64plus-input-bot for HTTP-based input injection
if [ -d "$BUILD_DIR/mupen64plus-input-bot" ]; then
    echo "  mupen64plus-input-bot: already cloned, pulling latest..."
    git -C "$BUILD_DIR/mupen64plus-input-bot" pull --quiet 2>/dev/null || true
else
    echo "  mupen64plus-input-bot: cloning..."
    git clone --quiet --depth 1 \
        "https://github.com/kevinhughes27/mupen64plus-input-bot.git" \
        "$BUILD_DIR/mupen64plus-input-bot" 2>/dev/null || {
        echo "  WARNING: mupen64plus-input-bot not found on GitHub."
        echo "  Will build a minimal custom input plugin instead."
    }
fi

# ── 4. Build core with DEBUGGER=1 ──────────────────────────────────────────

echo ""
echo ">>> Building mupen64plus-core (DEBUGGER=1)..."
make -C "$BUILD_DIR/mupen64plus-core/projects/unix" \
    DEBUGGER=1 \
    VULKAN=0 \
    PREFIX="" \
    DESTDIR="$LIB_DIR" \
    all -j"$(nproc)"

# Copy core library
cp "$BUILD_DIR/mupen64plus-core/projects/unix/"libmupen64plus.so* "$LIB_DIR/"

# Copy API headers (needed for input-bot build)
mkdir -p "$BUILD_DIR/api-headers"
cp "$BUILD_DIR/mupen64plus-core/src/api/"m64p_*.h "$BUILD_DIR/api-headers/"

# Copy data files (required by core)
if [ -d "$BUILD_DIR/mupen64plus-core/data" ]; then
    cp "$BUILD_DIR/mupen64plus-core/data/"* "$DATA_DIR/" 2>/dev/null || true
fi

# Copy video plugin data files (Rice .ini)
if [ -f "$BUILD_DIR/mupen64plus-video-rice/data/RiceVideoLinux.ini" ]; then
    cp "$BUILD_DIR/mupen64plus-video-rice/data/RiceVideoLinux.ini" "$DATA_DIR/"
fi

echo "  Verifying debug symbols..."
sync  # ensure file is flushed on Windows filesystem
if nm -D "$LIB_DIR/libmupen64plus.so.2.0.0" 2>/dev/null | grep -q DebugMemGetPointer; then
    echo "  OK: DebugMemGetPointer found ($(nm -D "$LIB_DIR/libmupen64plus.so.2.0.0" | grep -c DebugMem) DebugMem* symbols)"
else
    echo "  WARNING: DebugMemGetPointer not found in nm output."
    echo "  This may be a filesystem sync issue on WSL2. Checking build output directly..."
    if nm -D "$BUILD_DIR/mupen64plus-core/projects/unix/libmupen64plus.so.2.0.0" 2>/dev/null | grep -q DebugMemGetPointer; then
        echo "  OK: Found in build dir. Continuing."
    else
        echo "  ERROR: Debug build truly failed."
        exit 1
    fi
fi

# ── 5. Build plugins ───────────────────────────────────────────────────────

CORE_API="$BUILD_DIR/mupen64plus-core/projects/unix"

echo ""
echo ">>> Building mupen64plus-video-rice..."
make -C "$BUILD_DIR/mupen64plus-video-rice/projects/unix" \
    APIDIR="$BUILD_DIR/mupen64plus-core/src/api" \
    all -j"$(nproc)" 2>&1 || {
    echo "  WARNING: video-rice build failed, will try to continue."
}
cp "$BUILD_DIR/mupen64plus-video-rice/projects/unix/"mupen64plus-video-rice.so "$LIB_DIR/" 2>/dev/null || true

echo ">>> Building mupen64plus-audio-sdl..."
make -C "$BUILD_DIR/mupen64plus-audio-sdl/projects/unix" \
    APIDIR="$BUILD_DIR/mupen64plus-core/src/api" \
    all -j"$(nproc)" 2>&1 || {
    echo "  WARNING: audio-sdl build failed."
}
cp "$BUILD_DIR/mupen64plus-audio-sdl/projects/unix/"mupen64plus-audio-sdl.so "$LIB_DIR/" 2>/dev/null || true

echo ">>> Building mupen64plus-input-sdl..."
make -C "$BUILD_DIR/mupen64plus-input-sdl/projects/unix" \
    APIDIR="$BUILD_DIR/mupen64plus-core/src/api" \
    all -j"$(nproc)" 2>&1 || {
    echo "  WARNING: input-sdl build failed."
}
cp "$BUILD_DIR/mupen64plus-input-sdl/projects/unix/"mupen64plus-input-sdl.so "$LIB_DIR/" 2>/dev/null || true

echo ">>> Building mupen64plus-rsp-hle..."
make -C "$BUILD_DIR/mupen64plus-rsp-hle/projects/unix" \
    APIDIR="$BUILD_DIR/mupen64plus-core/src/api" \
    all -j"$(nproc)" 2>&1 || {
    echo "  WARNING: rsp-hle build failed."
}
cp "$BUILD_DIR/mupen64plus-rsp-hle/projects/unix/"mupen64plus-rsp-hle.so "$LIB_DIR/" 2>/dev/null || true

echo ">>> Building mupen64plus-ui-console..."
make -C "$BUILD_DIR/mupen64plus-ui-console/projects/unix" \
    APIDIR="$BUILD_DIR/mupen64plus-core/src/api" \
    COREDIR="$LIB_DIR" \
    all -j"$(nproc)" 2>&1 || {
    echo "  WARNING: ui-console build failed."
}
cp "$BUILD_DIR/mupen64plus-ui-console/projects/unix/"mupen64plus "$LIB_DIR/" 2>/dev/null || true

# ── 6. Build input-bot plugin (if available) ────────────────────────────────

if [ -d "$BUILD_DIR/mupen64plus-input-bot" ]; then
    echo ">>> Building mupen64plus-input-bot..."
    if [ -f "$BUILD_DIR/mupen64plus-input-bot/projects/unix/Makefile" ]; then
        make -C "$BUILD_DIR/mupen64plus-input-bot/projects/unix" \
            APIDIR="$BUILD_DIR/mupen64plus-core/src/api" \
            all -j"$(nproc)" 2>&1 || {
            echo "  WARNING: input-bot build failed."
        }
        cp "$BUILD_DIR/mupen64plus-input-bot/projects/unix/"mupen64plus-input-bot.so "$LIB_DIR/" 2>/dev/null || true
    elif [ -f "$BUILD_DIR/mupen64plus-input-bot/Makefile" ]; then
        make -C "$BUILD_DIR/mupen64plus-input-bot" \
            APIDIR="$BUILD_DIR/mupen64plus-core/src/api" \
            all -j"$(nproc)" 2>&1 || {
            echo "  WARNING: input-bot build failed."
        }
        find "$BUILD_DIR/mupen64plus-input-bot" -name "*.so" -exec cp {} "$LIB_DIR/" \; 2>/dev/null || true
    else
        echo "  WARNING: No Makefile found in mupen64plus-input-bot."
    fi
fi

# ── 7. Set up Python virtual environment ────────────────────────────────────

echo ""
echo ">>> Setting up Python virtual environment..."
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    python3 -m venv "$PROJECT_DIR/.venv"
fi
source "$PROJECT_DIR/.venv/bin/activate"
pip install -q -r "$PROJECT_DIR/requirements.txt"

# ── 8. Summary ──────────────────────────────────────────────────────────────

echo ""
echo "=== Build complete ==="
echo ""
echo "Libraries in $LIB_DIR:"
ls -la "$LIB_DIR/"*.so* 2>/dev/null || echo "  (none found)"
echo ""
echo "Console frontend:"
ls -la "$LIB_DIR/mupen64plus" 2>/dev/null || echo "  (not built)"
echo ""
echo "To test manually:"
echo "  $LIB_DIR/mupen64plus --plugindir $LIB_DIR --datadir $DATA_DIR \\"
echo "    --windowed --resolution 640x480 \\"
echo "    \"/mnt/c/code/n64/roms/New Tetris, The (USA).z64\""
echo ""
echo "To run the bot:"
echo "  source .venv/bin/activate"
echo "  python -m src"
