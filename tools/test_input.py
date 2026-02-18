#!/usr/bin/env python3
"""Test input injection manually.

Usage:
    python tools/test_input.py

Starts the input HTTP server and lets you manually send button presses
to verify the mupen64plus-input-bot plugin is working. Run the emulator
separately with the input-bot plugin pointing at localhost:8082.

Commands:
    a / b / z / start   - Press and release a button
    left / right / up / down  - D-pad directions
    hold <button> <frames>    - Hold a button for N frames
    sequence                  - Run a test sequence (Start -> A -> A -> A)
    status                    - Show current button state
    clear                     - Release all buttons
    quit                      - Exit
"""

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.input_server import ControllerState, InputServer


BUTTON_ALIASES = {
    "a": "A_BUTTON",
    "b": "B_BUTTON",
    "z": "Z_TRIG",
    "l": "L_TRIG",
    "r": "R_TRIG",
    "start": "START_BUTTON",
    "up": "U_DPAD",
    "down": "D_DPAD",
    "left": "L_DPAD",
    "right": "R_DPAD",
    "cup": "U_CBUTTON",
    "cdown": "D_CBUTTON",
    "cleft": "L_CBUTTON",
    "cright": "R_CBUTTON",
}


def main():
    port = 8082
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    server = InputServer(host="127.0.0.1", port=port)
    server.start()
    print(f"Input server running on http://127.0.0.1:{port}")
    print("Type 'help' for commands. Ctrl+C to exit.")
    print()
    print("To test, run mupen64plus with the input-bot plugin:")
    print(f"  mupen64plus --input mupen64plus-input-bot.so <rom>")
    print()

    try:
        while True:
            try:
                line = input("[input] > ").strip().lower()
            except EOFError:
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "help":
                print(__doc__)
            elif cmd == "quit" or cmd == "exit":
                break
            elif cmd == "status":
                state = server.get_state()
                pressed = []
                for field in state.__dataclass_fields__:
                    val = getattr(state, field)
                    if val:
                        pressed.append(f"{field}={val}")
                if pressed:
                    print(f"  Pressed: {', '.join(pressed)}")
                else:
                    print("  All buttons released")
            elif cmd == "clear":
                server.clear()
                print("  All buttons released")
            elif cmd == "hold" and len(parts) >= 3:
                button_name = BUTTON_ALIASES.get(parts[1], parts[1].upper())
                frames = int(parts[2])
                frame_time = 1.0 / 60.0
                print(f"  Holding {button_name} for {frames} frames...")
                server.set_button(button_name, 1)
                time.sleep(frames * frame_time)
                server.set_button(button_name, 0)
                print(f"  Released {button_name}")
            elif cmd == "sequence":
                print("  Running menu navigation sequence...")
                menu = [
                    ("START_BUTTON", 1.0),
                    ("A_BUTTON", 1.0),
                    ("A_BUTTON", 1.0),
                    ("A_BUTTON", 2.0),
                ]
                for button, wait in menu:
                    print(f"    Press {button}")
                    server.set_button(button, 1)
                    time.sleep(0.1)
                    server.set_button(button, 0)
                    time.sleep(wait)
                print("  Sequence complete")
            elif cmd in BUTTON_ALIASES:
                button_name = BUTTON_ALIASES[cmd]
                print(f"  Press {button_name}")
                server.set_button(button_name, 1)
                time.sleep(0.1)
                server.set_button(button_name, 0)
                print(f"  Released {button_name}")
            else:
                # Try as a raw button name
                button_name = cmd.upper()
                if hasattr(ControllerState, button_name):
                    print(f"  Press {button_name}")
                    server.set_button(button_name, 1)
                    time.sleep(0.1)
                    server.set_button(button_name, 0)
                    print(f"  Released {button_name}")
                else:
                    print(f"  Unknown command: {cmd}. Type 'help' for usage.")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
