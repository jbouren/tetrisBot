"""HTTP server for the mupen64plus-input-bot plugin.

The input-bot plugin polls an HTTP endpoint each frame to get the current
controller state. This module runs a lightweight HTTP server that responds
with JSON-encoded button states.

If mupen64plus-input-bot is unavailable, this module also supports a
fallback mode using xdotool to send keyboard events to the emulator window.
"""

import json
import logging
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """N64 controller button/axis state."""

    A_BUTTON: int = 0
    B_BUTTON: int = 0
    Z_TRIG: int = 0
    L_TRIG: int = 0
    R_TRIG: int = 0
    START_BUTTON: int = 0
    U_DPAD: int = 0
    D_DPAD: int = 0
    L_DPAD: int = 0
    R_DPAD: int = 0
    U_CBUTTON: int = 0
    D_CBUTTON: int = 0
    L_CBUTTON: int = 0
    R_CBUTTON: int = 0
    X_AXIS: int = 0
    Y_AXIS: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def clear(self):
        """Reset all buttons to unpressed."""
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)

    def is_empty(self) -> bool:
        """True if no buttons are pressed and axes are centered."""
        return all(getattr(self, f) == 0 for f in self.__dataclass_fields__)


class InputServer:
    """HTTP server polled by mupen64plus-input-bot each frame.

    Usage:
        server = InputServer(port=8082)
        server.start()
        server.set_button("A_BUTTON", 1)  # press A
        # ... emulator reads it on next frame ...
        server.set_button("A_BUTTON", 0)  # release A
        server.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8082):
        self.state = ControllerState()
        self._lock = threading.Lock()
        self._server = HTTPServer((host, port), self._make_handler())
        self._server.allow_reuse_address = True
        self._server.timeout = 0.1
        self._thread: threading.Thread | None = None
        self._host = host
        self._port = port

    def _make_handler(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with server_ref._lock:
                    body = server_ref.state.to_json().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass  # Suppress per-request logging

        return Handler

    def start(self):
        """Start the HTTP server in a background thread."""
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="input-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("Input server listening on %s:%d", self._host, self._port)

    def stop(self):
        """Stop the HTTP server."""
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Input server stopped")

    def set_button(self, button: str, value: int):
        """Set a single button/axis value."""
        with self._lock:
            setattr(self.state, button, value)

    def set_state(self, state: ControllerState):
        """Replace the entire controller state."""
        with self._lock:
            self.state = state

    def get_state(self) -> ControllerState:
        """Get a copy of the current state."""
        with self._lock:
            return ControllerState(**asdict(self.state))

    def clear(self):
        """Release all buttons."""
        with self._lock:
            self.state.clear()


# ── Fallback: xdotool keyboard input ───────────────────────────────────────

# Default mupen64plus keyboard mappings (scancode-based).
# These map N64 buttons to keyboard keys for use with xdotool.
KEYBOARD_MAP = {
    "A_BUTTON": "x",
    "B_BUTTON": "c",
    "Z_TRIG": "z",
    "L_TRIG": "a",
    "R_TRIG": "s",
    "START_BUTTON": "Return",
    "U_DPAD": "Up",
    "D_DPAD": "Down",
    "L_DPAD": "Left",
    "R_DPAD": "Right",
    "U_CBUTTON": "i",
    "D_CBUTTON": "k",
    "L_CBUTTON": "j",
    "R_CBUTTON": "l",
}


class XdotoolInput:
    """Fallback input injector using xdotool key events.

    Use this if mupen64plus-input-bot plugin is unavailable. Requires
    the emulator window to be focused and xdotool installed.
    """

    def __init__(self, window_name: str = "Mupen64Plus"):
        self._window_name = window_name
        self._window_id: str | None = None
        self._pressed: set[str] = set()

    def find_window(self) -> bool:
        """Find the emulator window by name."""
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", self._window_name],
                capture_output=True,
                text=True,
            )
            windows = result.stdout.strip().split("\n")
            if windows and windows[0]:
                self._window_id = windows[0]
                return True
        except FileNotFoundError:
            logger.warning("xdotool not found. Install with: sudo apt install xdotool")
        return False

    def press(self, button: str):
        """Press a button (send keydown)."""
        key = KEYBOARD_MAP.get(button)
        if not key or not self._window_id:
            return
        if button not in self._pressed:
            subprocess.run(
                ["xdotool", "keydown", "--window", self._window_id, key],
                capture_output=True,
            )
            self._pressed.add(button)

    def release(self, button: str):
        """Release a button (send keyup)."""
        key = KEYBOARD_MAP.get(button)
        if not key or not self._window_id:
            return
        if button in self._pressed:
            subprocess.run(
                ["xdotool", "keyup", "--window", self._window_id, key],
                capture_output=True,
            )
            self._pressed.discard(button)

    def release_all(self):
        """Release all pressed buttons."""
        for button in list(self._pressed):
            self.release(button)

    def send_state(self, state: ControllerState):
        """Apply a full controller state (press/release as needed)."""
        for button in KEYBOARD_MAP:
            value = getattr(state, button, 0)
            if value:
                self.press(button)
            else:
                self.release(button)
