"""Python wrapper around the mupen64plus core shared library (libmupen64plus.so).

Loads the core via ctypes, manages the emulation lifecycle, and provides
frame-by-frame control with direct RDRAM access for memory reading.
"""

import ctypes as ct
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .types import (
    CORE_API_VERSION,
    DebugCallbackType,
    M64Command,
    M64CoreParam,
    M64DbgMemPtrType,
    M64EmuState,
    M64Error,
    M64MsgLevel,
    M64PluginType,
    StateCallbackType,
)

logger = logging.getLogger(__name__)


class Mupen64PlusError(Exception):
    """Raised when a mupen64plus API call fails."""

    def __init__(self, func_name: str, error_code: int):
        self.func_name = func_name
        self.error_code = error_code
        try:
            name = M64Error(error_code).name
        except ValueError:
            name = f"UNKNOWN({error_code})"
        super().__init__(f"{func_name} failed: {name}")


class Mupen64PlusCore:
    """Manages the mupen64plus core lifecycle and provides the main API.

    Usage:
        core = Mupen64PlusCore("lib/libmupen64plus.so.2.0.0", "lib/", "lib/data/")
        core.startup()
        core.load_rom("/path/to/rom.z64")
        core.attach_plugins(gfx="...", audio="...", input="...", rsp="...")
        core.execute()  # starts emulation in background thread
        core.pause()
        # ... read memory, set inputs, advance_frame() ...
        core.stop()
        core.shutdown()
    """

    def __init__(self, core_lib_path: str, plugin_dir: str, data_dir: str):
        self._core_lib_path = core_lib_path
        self._plugin_dir = plugin_dir
        self._data_dir = data_dir

        self._lib: ct.CDLL = ct.cdll.LoadLibrary(core_lib_path)
        self._setup_function_prototypes()

        self._emu_thread: threading.Thread | None = None
        self._running = False
        self._plugins: dict[M64PluginType, ct.CDLL] = {}

        # Prevent garbage collection of callbacks
        self._debug_cb: DebugCallbackType | None = None
        self._state_cb: StateCallbackType | None = None

        # Synchronization
        self._state_lock = threading.Lock()
        self._emu_state = M64EmuState.STOPPED

    def _setup_function_prototypes(self):
        """Set argtypes and restype for core API functions."""
        lib = self._lib

        # CoreStartup
        lib.CoreStartup.argtypes = [
            ct.c_int,       # APIVersion
            ct.c_char_p,    # ConfigPath
            ct.c_char_p,    # DataPath
            ct.c_void_p,    # DebugContext
            DebugCallbackType,
            ct.c_void_p,    # StateContext
            StateCallbackType,
        ]
        lib.CoreStartup.restype = ct.c_int

        # CoreShutdown
        lib.CoreShutdown.argtypes = []
        lib.CoreShutdown.restype = ct.c_int

        # CoreDoCommand
        lib.CoreDoCommand.argtypes = [ct.c_int, ct.c_int, ct.c_void_p]
        lib.CoreDoCommand.restype = ct.c_int

        # CoreAttachPlugin
        lib.CoreAttachPlugin.argtypes = [ct.c_int, ct.c_void_p]
        lib.CoreAttachPlugin.restype = ct.c_int

        # CoreDetachPlugin
        lib.CoreDetachPlugin.argtypes = [ct.c_int]
        lib.CoreDetachPlugin.restype = ct.c_int

        # Debug API
        lib.DebugMemGetPointer.argtypes = [ct.c_int]
        lib.DebugMemGetPointer.restype = ct.c_void_p

        lib.DebugMemRead32.argtypes = [ct.c_uint]
        lib.DebugMemRead32.restype = ct.c_uint

        lib.DebugMemRead16.argtypes = [ct.c_uint]
        lib.DebugMemRead16.restype = ct.c_ushort

        lib.DebugMemRead8.argtypes = [ct.c_uint]
        lib.DebugMemRead8.restype = ct.c_ubyte

        lib.DebugMemWrite32.argtypes = [ct.c_uint, ct.c_uint]
        lib.DebugMemWrite32.restype = None

        lib.DebugMemWrite16.argtypes = [ct.c_uint, ct.c_ushort]
        lib.DebugMemWrite16.restype = None

        lib.DebugMemWrite8.argtypes = [ct.c_uint, ct.c_ubyte]
        lib.DebugMemWrite8.restype = None

    def _check_error(self, rc: int, func_name: str):
        if rc != M64Error.SUCCESS:
            raise Mupen64PlusError(func_name, rc)

    # ── Callbacks ───────────────────────────────────────────────────────────

    @staticmethod
    def _on_debug(_context, level: int, message: bytes):
        """Called by the core for debug/log messages."""
        if message is None:
            return
        text = message.decode("utf-8", errors="replace")
        if level <= M64MsgLevel.ERROR:
            logger.error("[m64p] %s", text)
        elif level <= M64MsgLevel.WARNING:
            logger.warning("[m64p] %s", text)
        elif level <= M64MsgLevel.INFO:
            logger.info("[m64p] %s", text)
        else:
            logger.debug("[m64p] %s", text)

    def _on_state_change(self, _context, param: int, value: int):
        """Called by the core when emulation state changes."""
        if param == M64CoreParam.EMU_STATE:
            with self._state_lock:
                try:
                    self._emu_state = M64EmuState(value)
                except ValueError:
                    pass
            logger.debug("Emulation state -> %s (%d)", self._emu_state.name, value)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def startup(self):
        """Initialize the mupen64plus core."""
        self._debug_cb = DebugCallbackType(self._on_debug)
        self._state_cb = StateCallbackType(self._on_state_change)

        config_path = self._data_dir.encode("utf-8")
        data_path = self._data_dir.encode("utf-8")

        rc = self._lib.CoreStartup(
            CORE_API_VERSION,
            config_path,
            data_path,
            None,
            self._debug_cb,
            None,
            self._state_cb,
        )
        self._check_error(rc, "CoreStartup")
        logger.info("mupen64plus core started (API version 0x%06X)", CORE_API_VERSION)

    def shutdown(self):
        """Fully shut down the core. Call after stop() and detaching plugins."""
        self.stop()
        self._detach_all_plugins()
        self._close_rom()
        rc = self._lib.CoreShutdown()
        self._check_error(rc, "CoreShutdown")
        logger.info("mupen64plus core shut down")

    # ── ROM ─────────────────────────────────────────────────────────────────

    def load_rom(self, rom_path: str):
        """Load an N64 ROM file into the core."""
        rom_data = Path(rom_path).read_bytes()
        rom_buffer = ct.create_string_buffer(rom_data)
        rc = self._lib.CoreDoCommand(
            M64Command.ROM_OPEN, len(rom_data), ct.byref(rom_buffer)
        )
        self._check_error(rc, "ROM_OPEN")
        logger.info("ROM loaded: %s (%d bytes)", rom_path, len(rom_data))

    def _close_rom(self):
        try:
            self._lib.CoreDoCommand(M64Command.ROM_CLOSE, 0, None)
        except Exception:
            pass

    # ── Plugins ─────────────────────────────────────────────────────────────

    def attach_plugins(
        self,
        gfx: str | None = None,
        audio: str | None = None,
        input: str | None = None,
        rsp: str | None = None,
    ):
        """Load and attach plugins. Pass paths to .so files."""
        plugin_map = {
            M64PluginType.GFX: gfx,
            M64PluginType.AUDIO: audio,
            M64PluginType.INPUT: input,
            M64PluginType.RSP: rsp,
        }
        for ptype, path in plugin_map.items():
            if path is None:
                continue
            self._attach_plugin(ptype, path)

    def _attach_plugin(self, ptype: M64PluginType, path: str):
        handle = ct.cdll.LoadLibrary(path)

        # PluginStartup(CoreHandle, DebugContext, DebugCallback)
        handle.PluginStartup.argtypes = [ct.c_void_p, ct.c_void_p, DebugCallbackType]
        handle.PluginStartup.restype = ct.c_int
        rc = handle.PluginStartup(
            ct.c_void_p(self._lib._handle), None, self._debug_cb
        )
        self._check_error(rc, f"PluginStartup({ptype.name})")

        rc = self._lib.CoreAttachPlugin(ptype, ct.c_void_p(handle._handle))
        self._check_error(rc, f"CoreAttachPlugin({ptype.name})")

        self._plugins[ptype] = handle
        logger.info("Plugin attached: %s -> %s", ptype.name, path)

    def _detach_all_plugins(self):
        for ptype in list(self._plugins):
            try:
                self._lib.CoreDetachPlugin(ptype)
            except Exception:
                pass
            try:
                self._plugins[ptype].PluginShutdown()
            except Exception:
                pass
        self._plugins.clear()

    # ── Emulation control ───────────────────────────────────────────────────

    def execute(self):
        """Start emulation in a background thread.

        M64CMD_EXECUTE blocks until M64CMD_STOP is called, so we run it
        in a daemon thread.
        """
        if self._running:
            return
        self._running = True
        self._emu_thread = threading.Thread(
            target=self._run_emulation, name="m64p-emu", daemon=True
        )
        self._emu_thread.start()
        logger.info("Emulation started (background thread)")

    def _run_emulation(self):
        """Runs in background thread. Blocks until STOP."""
        self._lib.CoreDoCommand(M64Command.EXECUTE, 0, None)
        self._running = False
        logger.info("Emulation thread exited")

    def pause(self):
        """Pause emulation."""
        rc = self._lib.CoreDoCommand(M64Command.PAUSE, 0, None)
        self._check_error(rc, "PAUSE")

    def resume(self):
        """Resume emulation after pause."""
        rc = self._lib.CoreDoCommand(M64Command.RESUME, 0, None)
        self._check_error(rc, "RESUME")

    def advance_frame(self):
        """Advance exactly one frame. Must be paused first."""
        rc = self._lib.CoreDoCommand(M64Command.ADVANCE_FRAME, 0, None)
        self._check_error(rc, "ADVANCE_FRAME")

    def stop(self):
        """Stop emulation and wait for the thread to exit."""
        if not self._running:
            return
        try:
            self._lib.CoreDoCommand(M64Command.STOP, 0, None)
        except Exception:
            pass
        if self._emu_thread:
            self._emu_thread.join(timeout=5.0)
        self._running = False

    def reset(self, soft: bool = True):
        """Reset the emulated N64. soft=True for soft reset, False for hard."""
        rc = self._lib.CoreDoCommand(M64Command.RESET, int(soft), None)
        self._check_error(rc, "RESET")

    # ── State management ────────────────────────────────────────────────────

    def save_state(self, slot: int = 0):
        """Save emulator state to a slot (0-9)."""
        self._lib.CoreDoCommand(M64Command.STATE_SET_SLOT, slot, None)
        rc = self._lib.CoreDoCommand(M64Command.STATE_SAVE, 1, None)
        self._check_error(rc, "STATE_SAVE")

    def load_state(self, slot: int = 0):
        """Load emulator state from a slot (0-9)."""
        self._lib.CoreDoCommand(M64Command.STATE_SET_SLOT, slot, None)
        rc = self._lib.CoreDoCommand(M64Command.STATE_LOAD, 1, None)
        self._check_error(rc, "STATE_LOAD")

    # ── Memory access ───────────────────────────────────────────────────────

    def get_rdram_pointer(self) -> ct.c_void_p:
        """Get raw pointer to the 8MB RDRAM buffer.

        Returns None if emulation hasn't started or debug is unavailable.
        """
        ptr = self._lib.DebugMemGetPointer(M64DbgMemPtrType.RDRAM)
        if not ptr:
            raise RuntimeError(
                "DebugMemGetPointer returned NULL. "
                "Is DEBUGGER=1 enabled? Is emulation running?"
            )
        return ptr

    def debug_read_32(self, address: int) -> int:
        """Read a 32-bit value from N64 virtual address (handles endianness)."""
        return self._lib.DebugMemRead32(ct.c_uint(address))

    def debug_read_16(self, address: int) -> int:
        """Read a 16-bit value from N64 virtual address."""
        return self._lib.DebugMemRead16(ct.c_uint(address))

    def debug_read_8(self, address: int) -> int:
        """Read an 8-bit value from N64 virtual address."""
        return self._lib.DebugMemRead8(ct.c_uint(address))

    def debug_write_32(self, address: int, value: int):
        """Write a 32-bit value to N64 virtual address."""
        self._lib.DebugMemWrite32(ct.c_uint(address), ct.c_uint(value))

    def debug_write_16(self, address: int, value: int):
        """Write a 16-bit value to N64 virtual address."""
        self._lib.DebugMemWrite16(ct.c_uint(address), ct.c_ushort(value))

    def debug_write_8(self, address: int, value: int):
        """Write an 8-bit value to N64 virtual address."""
        self._lib.DebugMemWrite8(ct.c_uint(address), ct.c_ubyte(value))

    # ── State queries ───────────────────────────────────────────────────────

    # ── Screen capture ─────────────────────────────────────────────────────

    _screenshot_dir = Path.home() / ".local/share/mupen64plus/screenshot"

    def read_screen(self, max_retries=1) -> np.ndarray:
        """Capture the current frame via TAKE_NEXT_SCREENSHOT.

        Issues a screenshot command, advances one frame so the render thread
        captures it, then reads the PNG file back. Returns an RGB numpy array
        of shape (height, width, 3), uint8.

        The emulator should be paused before calling this.
        """
        for attempt in range(max_retries + 1):
            try:
                t0 = time.monotonic()

                # Get a list of existing screenshot files to ignore them
                self._screenshot_dir.mkdir(parents=True, exist_ok=True)
                existing_files = set(self._screenshot_dir.glob("newtetris-*.png"))

                # Tell the core to take a screenshot on the next rendered frame
                self._lib.CoreDoCommand(M64Command.TAKE_NEXT_SCREENSHOT, 0, None)
                self.advance_frame()

                t1 = time.monotonic()

                # Poll for the new file to appear
                new_path = None
                deadline = time.monotonic() + 2.0  # 2-second timeout
                while time.monotonic() < deadline:
                    try:
                        # Find any file that is not in the original set
                        files = self._screenshot_dir.glob("newtetris-*.png")
                        new_files = [p for p in files if p not in existing_files]
                        if new_files:
                            new_path = new_files[0]
                            break
                    except OSError as e:
                        logger.warning("Error polling for screenshot: %s", e)

                    time.sleep(0.01)

                if new_path is None:
                    raise RuntimeError("Screenshot file not created within timeout")

                t2 = time.monotonic()

                # Wait for file to be fully written by polling for size stabilization
                last_size = -1
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        size = new_path.stat().st_size
                        if size > 0 and size == last_size:
                            break
                        last_size = size
                    except FileNotFoundError:
                        # File may not be visible yet, wait a moment
                        pass
                    except OSError as e:
                        logger.warning("Error stat-ing screenshot file: %s", e)
                    time.sleep(0.005)

                # Read the image file
                bgr_image = cv2.imread(str(new_path))

                # Clean up the screenshot file immediately after reading
                try:
                    new_path.unlink()
                except OSError as e:
                    logger.warning("Failed to delete screenshot %s: %s", new_path, e)

                if bgr_image is None:
                    raise RuntimeError(f"Failed to read screenshot file {new_path}")

                t3 = time.monotonic()
                total_ms = (t3 - t0) * 1000
                if total_ms > 200:
                    logger.debug(
                        "read_screen slow: advance=%.0fms poll=%.0fms read=%.0fms TOTAL=%.0fms",
                        (t1 - t0) * 1000,
                        (t2 - t1) * 1000,
                        (t3 - t2) * 1000,
                        total_ms,
                    )

                return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            except RuntimeError as e:
                logger.warning("read_screen failed on attempt %d: %s", attempt + 1, e)
                if attempt < max_retries:
                    time.sleep(0.1)  # Wait 100ms before retrying
                else:
                    raise e
        raise RuntimeError("read_screen failed after all retries")

    @property
    def emu_state(self) -> M64EmuState:
        """Current emulation state (STOPPED, RUNNING, PAUSED)."""
        with self._state_lock:
            return self._emu_state

    @property
    def is_running(self) -> bool:
        return self._running
