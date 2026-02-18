"""ctypes definitions mirroring the mupen64plus C API headers."""

import ctypes as ct
from enum import IntEnum


# ── Error codes (m64p_types.h) ──────────────────────────────────────────────

class M64Error(IntEnum):
    SUCCESS = 0
    NOT_INIT = 1
    ALREADY_INIT = 2
    INCOMPATIBLE = 3
    INPUT_ASSERT = 4
    INPUT_INVALID = 5
    INPUT_NOT_FOUND = 6
    NO_MEMORY = 7
    FILES = 8
    INTERNAL = 9
    INVALID_STATE = 10
    PLUGIN_FAIL = 11
    SYSTEM_FAIL = 12
    UNSUPPORTED = 13
    WRONG_TYPE = 14


# ── Core commands (m64p_types.h) ────────────────────────────────────────────

class M64Command(IntEnum):
    NOP = 0
    ROM_OPEN = 1
    ROM_CLOSE = 2
    ROM_GET_HEADER = 3
    ROM_GET_SETTINGS = 4
    EXECUTE = 5
    STOP = 6
    PAUSE = 7
    RESUME = 8
    CORE_STATE_QUERY = 9
    STATE_LOAD = 10
    STATE_SAVE = 11
    STATE_SET_SLOT = 12
    SEND_SDL_KEYDOWN = 13
    SEND_SDL_KEYUP = 14
    SET_FRAME_CALLBACK = 15
    TAKE_NEXT_SCREENSHOT = 16
    CORE_STATE_SET = 17
    READ_SCREEN = 18
    RESET = 19
    ADVANCE_FRAME = 20


# ── Plugin types (m64p_types.h) ─────────────────────────────────────────────

class M64PluginType(IntEnum):
    NULL = 0
    RSP = 1
    GFX = 2
    AUDIO = 3
    INPUT = 4
    CORE = 5


# ── Message levels ──────────────────────────────────────────────────────────

class M64MsgLevel(IntEnum):
    ERROR = 1
    WARNING = 2
    INFO = 3
    STATUS = 4
    VERBOSE = 5


# ── Debug memory pointer types ──────────────────────────────────────────────

class M64DbgMemPtrType(IntEnum):
    RDRAM = 1
    PI_REG = 2
    SI_REG = 3
    VI_REG = 4
    RI_REG = 5
    AI_REG = 6


# ── Core parameter types (for state queries) ───────────────────────────────

class M64CoreParam(IntEnum):
    EMU_STATE = 1
    VIDEO_MODE = 2
    SAVESTATE_SLOT = 3
    SPEED_FACTOR = 4
    SPEED_LIMITER = 5
    VIDEO_SIZE = 6
    AUDIO_VOLUME = 7
    AUDIO_MUTE = 8
    INPUT_GAMESHARK = 9
    STATE_LOADCOMPLETE = 10
    STATE_SAVECOMPLETE = 11


# ── Emulation states ───────────────────────────────────────────────────────

class M64EmuState(IntEnum):
    STOPPED = 1
    RUNNING = 2
    PAUSED = 3


# ── Callback function types ────────────────────────────────────────────────

DebugCallbackType = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int, ct.c_char_p)
StateCallbackType = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int, ct.c_int)
FrameCallbackType = ct.CFUNCTYPE(None, ct.c_uint)


# ── N64 controller BUTTONS (m64p_plugin.h) ─────────────────────────────────

class BUTTONS(ct.Union):
    """N64 controller state as a union of bitfields and raw u32."""

    class _Bits(ct.LittleEndianStructure):
        _fields_ = [
            ("R_DPAD", ct.c_uint16, 1),
            ("L_DPAD", ct.c_uint16, 1),
            ("D_DPAD", ct.c_uint16, 1),
            ("U_DPAD", ct.c_uint16, 1),
            ("START_BUTTON", ct.c_uint16, 1),
            ("Z_TRIG", ct.c_uint16, 1),
            ("B_BUTTON", ct.c_uint16, 1),
            ("A_BUTTON", ct.c_uint16, 1),
            ("R_CBUTTON", ct.c_uint16, 1),
            ("L_CBUTTON", ct.c_uint16, 1),
            ("D_CBUTTON", ct.c_uint16, 1),
            ("U_CBUTTON", ct.c_uint16, 1),
            ("R_TRIG", ct.c_uint16, 1),
            ("L_TRIG", ct.c_uint16, 1),
            ("Reserved1", ct.c_uint16, 1),
            ("Reserved2", ct.c_uint16, 1),
        ]

    _fields_ = [
        ("Value", ct.c_uint32),
        ("bits", _Bits),
    ]
    _anonymous_ = ()


# ── API version we target ──────────────────────────────────────────────────

CORE_API_VERSION = 0x020001
PLUGIN_API_VERSION = 0x020000
CONFIG_API_VERSION = 0x020000
