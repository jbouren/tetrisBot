"""Translates piece placements into sequences of controller inputs.

The New Tetris controls:
  - D-pad Left/Right: move piece horizontally (1 cell per tap)
  - A button: rotate clockwise
  - B button: rotate counter-clockwise
  - D-pad Down: soft drop (faster fall)
  - D-pad Up: hard drop (instant drop, no lock delay in some modes)

Input timing: each button press is 1 frame on + 1 frame off to avoid
triggering DAS (Delayed Auto Shift) auto-repeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..emulator.input_server import ControllerState

if TYPE_CHECKING:
    from .planner import Placement


@dataclass
class InputAction:
    """A single frame of controller input."""

    state: ControllerState
    description: str = ""  # For debugging


class InputController:
    """Converts placements to input sequences."""

    def __init__(self, hard_drop_button: str = "U_DPAD"):
        """
        Args:
            hard_drop_button: Which button triggers hard/firm drop.
                "U_DPAD" for The New Tetris (D-pad Up = hard drop).
        """
        self._hard_drop = hard_drop_button

    def placement_to_inputs(
        self,
        current_x: int,
        current_rotation: int,
        target: Placement,
    ) -> list[InputAction]:
        """Generate input sequence to move from current position to target.

        Strategy:
        1. Rotate to target rotation (shortest path CW or CCW)
        2. Move horizontally to target column
        3. Hard drop

        Returns a list of InputActions, one per frame.
        """
        inputs: list[InputAction] = []

        # Step 1: Rotations
        rot_diff = (target.rotation - current_rotation) % 4
        if rot_diff == 0:
            pass
        elif rot_diff <= 2:
            # Rotate CW (A button)
            for i in range(rot_diff):
                inputs.append(self._button_press("A_BUTTON", f"rotate CW {i+1}"))
                inputs.append(self._release(f"release after CW {i+1}"))
        else:
            # Rotate CCW (B button) - 1 press instead of 3 CW
            inputs.append(self._button_press("B_BUTTON", "rotate CCW"))
            inputs.append(self._release("release after CCW"))

        # Step 2: Horizontal movement
        dx = target.column - current_x
        if dx > 0:
            for i in range(dx):
                inputs.append(self._button_press("R_DPAD", f"move right {i+1}"))
                inputs.append(self._release(f"release after right {i+1}"))
        elif dx < 0:
            for i in range(-dx):
                inputs.append(self._button_press("L_DPAD", f"move left {i+1}"))
                inputs.append(self._release(f"release after left {i+1}"))

        # Step 3: Hard drop
        inputs.append(self._button_press(self._hard_drop, "hard drop"))
        inputs.append(self._release("release after drop"))

        # Extra frames to let the piece lock and new piece spawn
        for i in range(10):
            inputs.append(self._release(f"wait for lock {i+1}"))

        return inputs

    @staticmethod
    def _button_press(button: str, description: str = "") -> InputAction:
        """Create a single button press action."""
        state = ControllerState()
        setattr(state, button, 1)
        return InputAction(state=state, description=description)

    @staticmethod
    def _release(description: str = "") -> InputAction:
        """Create a release (all buttons up) action."""
        return InputAction(state=ControllerState(), description=description)

    def soft_drop_inputs(self, frames: int = 1) -> list[InputAction]:
        """Generate soft drop inputs (hold down)."""
        inputs = []
        for i in range(frames):
            state = ControllerState()
            state.D_DPAD = 1
            inputs.append(InputAction(state=state, description=f"soft drop {i+1}"))
        inputs.append(self._release("release soft drop"))
        return inputs

    def start_game_inputs(self, sprint_mode: bool = True) -> list[InputAction]:
        """Generate inputs to navigate from title screen to gameplay.

        See MEMORY.md for the exact sequence for Sprint Mode.
        """
        inputs: list[InputAction] = []

        # Helper to add a press-release-wait sequence
        def add_menu_action(button: str, desc: str, wait_frames: int):
            inputs.append(self._button_press(button, desc))
            # Hold for a few frames to ensure registration in menu
            for _ in range(3):
                inputs.append(self._button_press(button, f"{desc} hold"))
            inputs.append(self._release(f"release {desc}"))
            for _ in range(wait_frames):
                inputs.append(self._release(f"wait after {desc}"))

        # Sequence from MEMORY.md, using timings from the last working bot.py version
        # (seconds * 60 fps = frames)
        # START > START > A > A > A > Down > Right > Down > A
        add_menu_action("START_BUTTON", "skip intro", 210)  # 3.5s
        add_menu_action("START_BUTTON", "pass title screen", 210)  # 3.5s
        add_menu_action("A_BUTTON", "enter menu", 180)  # 3.0s
        add_menu_action("A_BUTTON", "confirm advance", 180)  # 3.0s
        add_menu_action("A_BUTTON", "select player", 150)  # 2.5s
        add_menu_action("D_DPAD", "nav to Sprint mode", 120)  # 2.0s
        add_menu_action("R_DPAD", "select within Sprint", 120)  # 2.0s
        add_menu_action("D_DPAD", "sub-selection", 120)  # 2.0s
        add_menu_action("A_BUTTON", "start game", 120)  # 2.0s

        # Wait for "3-2-1-GO" countdown before gameplay begins
        for _ in range(600):  # ~10 seconds
            inputs.append(self._release("wait for countdown"))

        return inputs
