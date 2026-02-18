"""Main Tetris bot orchestrator.

Ties together the emulator interface, game state parsing, AI planning,
and input injection into a single frame-by-frame bot loop.
"""

import logging
import time

# Suppress verbose emulator core logging
logging.getLogger("src.emulator.core").setLevel(logging.WARNING)

from .ai.controller import InputAction, InputController
from .ai.heuristic import HeuristicEvaluator
from .ai.planner import MovePlanner, Placement
from .emulator.core import Mupen64PlusCore
from .emulator.input_server import ControllerState, InputServer
from .emulator.memory import MemoryReader
from .game.pieces import PieceType
from .game.state import GameState

logger = logging.getLogger(__name__)


class TetrisBot:
    """Main bot that plays The New Tetris autonomously."""

    def __init__(self, config: dict):
        self.config = config

        # Emulator
        self.core = Mupen64PlusCore(
            core_lib_path=config["core_lib_path"],
            plugin_dir=config["plugin_dir"],
            data_dir=config["data_dir"],
        )
        self.memory = MemoryReader(
            self.core, use_debug_api=config.get("use_debug_api", False)
        )
        self.input_server = InputServer(
            host=config.get("input_host", "127.0.0.1"),
            port=config.get("input_port", 8082),
        )

        # AI
        self.evaluator = HeuristicEvaluator()
        self.planner = MovePlanner(self.evaluator)
        self.controller = InputController()

        # State tracking
        self._last_piece: PieceType | None = None
        self._last_piece_x: int | None = None
        self._input_queue: list[InputAction] = []
        self._frames_played = 0
        self._pieces_placed = 0
        self._frames_per_batch = config.get("frames_per_batch", 10)

    def start(self):
        """Initialize everything and enter the bot loop."""
        logger.info("=== Tetris Bot Starting ===")

        # Start input server
        self.input_server.start()

        # Initialize emulator
        logger.info("Initializing mupen64plus core...")
        self.core.startup()

        logger.info("Loading ROM: %s", self.config["rom_path"])
        self.core.load_rom(self.config["rom_path"])

        logger.info("Attaching plugins...")
        self.core.attach_plugins(
            gfx=self.config.get("gfx_plugin"),
            audio=self.config.get("audio_plugin"),
            input=self.config.get("input_plugin"),
            rsp=self.config.get("rsp_plugin"),
        )

        logger.info("Starting emulation...")
        self.core.execute()

        # Wait for the game to boot. 18s is required to get past the intro cinematics.
        boot_wait = self.config.get("boot_wait_seconds", 18)
        logger.info("Waiting %d seconds for game to boot...", boot_wait)
        time.sleep(boot_wait)

        # Acquire RDRAM pointer
        self.memory.refresh_pointer()
        logger.info("RDRAM pointer acquired")

        # Navigate menu to start gameplay
        logger.info("Navigating menu...")
        self._navigate_menu()

        # Enter main bot loop
        logger.info("Entering bot loop...")
        try:
            self._bot_loop()
        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
        except Exception:
            logger.exception("Bot loop error")

        # Report final stats
        logger.info(
            "=== Bot Finished === Frames: %d, Pieces: %d",
            self._frames_played,
            self._pieces_placed,
        )

    def _press_button(self, button: str, hold_sec=0.05, pause_sec=0.5):
        """Press a button in real-time."""
        logger.debug(
            "Pressing %s (hold %.2fs, pause %.2fs)", button, hold_sec, pause_sec
        )
        state = ControllerState()
        setattr(state, button, 1)
        self.input_server.set_state(state)
        time.sleep(hold_sec)
        self.input_server.clear()
        time.sleep(pause_sec)

    def _navigate_menu(self):
        """Send inputs to get from title screen to gameplay."""
        # Sequence from MEMORY.md, timings from navigate_and_save.py
        self.core.resume()  # Ensure emulation is running for real-time input
        time.sleep(1)

        self._press_button("START_BUTTON", hold_sec=0.1, pause_sec=3.5)
        self._press_button("START_BUTTON", hold_sec=0.1, pause_sec=3.5)
        self._press_button("A_BUTTON", hold_sec=0.5, pause_sec=3.0)
        self._press_button("A_BUTTON", hold_sec=0.5, pause_sec=3.0)
        self._press_button("A_BUTTON", hold_sec=0.5, pause_sec=2.5)
        self._press_button("A_BUTTON", hold_sec=0.5, pause_sec=2.5)
        self._press_button("A_BUTTON", hold_sec=0.5, pause_sec=2.5)
        self._press_button("D_DPAD", hold_sec=0.1, pause_sec=2.0)
        self._press_button("R_DPAD", hold_sec=0.1, pause_sec=2.0)
        self._press_button("D_DPAD", hold_sec=0.1, pause_sec=2.0)
        self._press_button("A_BUTTON", hold_sec=0.1, pause_sec=2.0)

        logger.info("Waiting for game to start after menu navigation...")
        time.sleep(10)  # Wait for 3-2-1-GO countdown
        logger.info("Menu navigation complete.")

    def _bot_loop(self):
        """Main decision loop. Runs frame-by-frame after pausing."""
        # Pause emulation for frame-by-frame control
        self.core.pause()
        time.sleep(0.1)

        while True:
            # Read current game state
            state = GameState.from_memory(self.memory)

            if state.is_game_over:
                logger.info(
                    "GAME OVER! Score: %d, Lines: %d, Pieces: %d",
                    state.score,
                    state.lines,
                    self._pieces_placed,
                )
                break

            # If the input queue is empty, it's time to plan the next move.
            if not self._input_queue:
                # Before planning, ensure the game is actually in a playable state.
                # This prevents planning during screen transitions, pauses, etc.
                if not state.is_playing:
                    # If not playing, just wait a frame and check again.
                    # Clear any lingering input state.
                    self.input_server.clear()
                else:
                    # It's playing and time to plan.
                    self._pieces_placed += 1
                    placement = self.planner.find_best_placement(
                        state.board,
                        state.current_piece,
                        next_piece=state.next_piece,
                    )

                    if placement is not None:
                        logger.debug(
                            "Piece #%d: %s -> col=%d rot=%d (score=%.2f)",
                            self._pieces_placed,
                            state.current_piece.name,
                            placement.column,
                            placement.rotation,
                            placement.score,
                        )
                        self._input_queue = self.controller.placement_to_inputs(
                            current_x=state.current_x,
                            current_rotation=state.current_rotation,
                            target=placement,
                        )
                    else:
                        logger.warning("No valid placement found!")
                        # Wait for a bit if no placement is found, maybe the
                        # state is weird.
                        for _ in range(30):
                            self._input_queue.append(
                                self.controller._release("no valid placement")
                            )

            # Execute next input from queue
            if self._input_queue:
                action = self._input_queue.pop(0)
                self.input_server.set_state(action.state)

            # Always advance one frame
            self.core.advance_frame()
            self._frames_played += 1

            # Rate-limit the loop to ~60 FPS
            time.sleep(1 / 60)

            # Periodic status logging
            if self._frames_played > 0 and self._frames_played % 300 == 0:
                logger.info(
                    "Frame %d | %s",
                    self._frames_played,
                    state.summary(),
                )

    def stop(self):
        """Clean shutdown."""
        try:
            self.core.shutdown()
        except Exception:
            logger.exception("Error during core shutdown")
        self.input_server.stop()
