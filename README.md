# N64 Tetris Bot

This project is an AI bot that plays "The New Tetris" on the Nintendo 64 using the mupen64plus emulator. It uses a hybrid approach, combining direct emulator control for inputs and frame-perfect advancement with a computer vision (CV) system to read the game state from screenshots.

The bot uses a pre-trained convolutional neural network (CNN) from the [fischly/tetris-ai](https://github.com/fischly/tetris-ai) project to evaluate board states and determine the optimal placement for the current piece.

## Current Status

The bot is highly functional and plays the game reliably.

- **Plays Sprint Mode**: Successfully navigates the game menus to start a Sprint (40-line clear) game.
- **High Accuracy**: Achieves ~95.5% placement accuracy in live emulator gameplay.
- **CV-First Architecture**: The control loop is driven entirely by computer vision, reading the board in a "ghost-free" window and tracking pieces via the preview queue. This makes it resilient to timing variations.
- **Performance**: Has successfully cleared over 30 lines in a single run, limited primarily by the game speed.
- **Emulator Integration**: Runs on WSL2, interfacing with mupen64plus for frame-perfect control and input injection.

## Architecture

The project is broken down into several key components:

- `src/emulator/`: A Python wrapper around the mupen64plus C core library (`libmupen64plus.so`). This allows for direct control over the emulator's execution, including pausing, advancing frame-by-frame, and loading/saving states.
- `src/game/`: Contains the game logic, piece definitions (`pieces.py`), and the computer vision module (`vision.py`). The CV module handles reading the 10x20 board grid and the upcoming pieces from emulator screenshots.
- `src/ai/`: The AI decision-making component. `board_evaluator.py` wraps the pre-trained PyTorch CNN model, providing a function to find the best placement (rotation, column) for a given piece and board state.
- `tools/`: Utility scripts for running the bot, training models, and calibration.
  - `play_eval.py`: The main script to run the bot.
  - `calibrate_inputs.py`: A script used to determine the correct frame timings for button presses.
  - `navigate_and_save.py`: A one-off script to navigate the game menus and create the initial save state.
- `lib/`: Contains the compiled mupen64plus core and plugin `.so` files.

## Key Technical Decisions

- **Hybrid CV/Emulator Control**: Early attempts to read the game state directly from the N64's memory (RDRAM) proved too difficult due to the complex way "The New Tetris" stores its board data. The project pivoted to a more robust computer vision approach, which has been very successful.
- **Deterministic, Frame-Perfect Timing**: The bot does not rely on `time.sleep()` for its main loop. Instead, it tells the emulator exactly when to advance to the next frame. This allows for precise, repeatable actions.
- **CV in a "Ghost-Free Window"**: A key insight was that for a few frames after a piece locks, the board is drawn with the newly-settled piece but *before* the next piece's ghost appears. By reading the screen in this window, we get a perfect, clean board state without needing complex ghost-filtering logic.
- **Sense-Plan-Act Feedback Loop**: The bot doesn't just send a blind sequence of commands. For each piece, it rotates, then uses CV to *sense* the piece's new column, then *plans* the necessary movement, and finally *acts*. This feedback loop makes it resilient to dropped inputs.

## Setup

### 1. Environment

- This project is designed to run on **WSL2** with GPU passthrough enabled.
- It requires a Python 3.12 environment.
- `binutils-dev` is required for building the mupen64plus debugger module.

### 2. Build Mupen64Plus

The `setup_env.sh` script automates the process of cloning and building the required mupen64plus components from source.

```bash
./setup_env.sh
```

This will clone the necessary repositories, apply a small patch to the input-bot plugin, and build all `.so` files, placing them in the `lib/` directory.

### 3. Python Dependencies

Create a virtual environment and install the required Python packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. ROM

You must provide your own ROM for "The New Tetris (USA)". Place it at: `/mnt/c/code/n64/roms/New Tetris, The (USA).z64`.

### 5. AI Model

Download the pre-trained CNN weights from [fischly/tetris-ai](https://github.com/fischly/tetris-ai) or another source. The recommended model is `good-cnn-2.pt`. Place it in the `models/` directory.

### 6. Initial Save State

The bot relies on a save state (slot 1) that is positioned at the very beginning of a Sprint mode game, right as the first piece is spawning. Run the `navigate_and_save.py` script once to create this state.

```bash
.venv/bin/python3 tools/navigate_and_save.py
```

This will launch the emulator, automatically press the buttons to navigate the menu, and save the state.

## How to Run

With the setup complete, you can run the bot using the `play_eval.py` script.

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the bot
python3 tools/play_eval.py models/good-cnn-2.pt
```

### Command-Line Options

- `--step`: Pause before each piece placement, showing the planned move and waiting for user input. This is excellent for debugging.
- `--speed <float>`: Adjust the playback speed. `2.0` is double speed, `0.5` is half speed.
- `--board-threshold <int>`: The brightness threshold (0-255) for detecting occupied cells on the board. Default is 40.

The bot will automatically clear out the emulator's screenshot directory (`~/.local/share/mupen64plus/screenshot`) at the start of each run and periodically during gameplay to avoid hitting the 1000-screenshot limit.
