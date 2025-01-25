# Minesweeper Hint

The Minesweeper Hint is a Python program designed to assist players by providing hints for solving the [Minesweeper Online](https://minesweeper.online/) game. It analyzes the game board visually, evaluates safe moves, and highlights tiles to guide players toward solving the puzzle more efficiently.

The app shares most of the logic with https://github.com/SamuelSoNChino/Minesweeper-Bot/

### Features

- **Hint Generation**: The application identifies safe tiles to click and marks suspected mines, reducing the need for random guesses.

- **Visual Detection**: Uses color filters and edge recognition to detect the game board and grid tiles.

- **Interactive Overlay**: Highlights safe moves, mines, and unknown areas directly on the game board.

## How It Works

1. **Field Detection**: The application identifies the game field using color-based masking and edge detection.

2. **Tile Grid Mapping**: It maps the positions of individual tiles on the grid.

3. **Tile Analysis**: Each tile is evaluated to determine whether it is safe to click, a mine, or unknown.

4. **Hint Overlay**: Using visual feedback, the application overlays hints directly on the game screen.

5. **Re-evaluation**: After every action, the application updates its knowledge of the board and continues until the game is won or a random guess fails.

## Setup & Usage
Prerequisites

- Install the required Python libraries:

        pip install pyautogui opencv-python-headless numpy

- A desktop environment with Minesweeper running in a visible browser window.

### Steps to Run

1. Open Minesweeper Online and start a new game.

2. Run the script in your Python environment.

3. The application will provide hints by analyzing the game screen and overlaying visual guides on the board.


## Troubleshooting

If the application does not work as expected, try the following:

1. **Adjust Zoom and Brightness**:
    
    - Experiment with browser zoom levels or screen brightness to ensure accurate detection.

2. **Change the `MAIN_COLOR`**:
    - If detection issues persist, update the `MAIN_COLOR` constant to match the primary shade of gray in your minefield.
        - Use [Image Color Picker](https://imagecolorpicker.com/) to find the RGB code of the gray area.
        - Convert the RGB value into a single grayscale value for `MAIN_COLOR`.

## Program Explanation

The application relies on the following steps:

1. **Visual Detection**:

    - Color filters and edge detection isolate the minefield from the screenshot.
    
    - The `task2` function identifies the field’s bounding box.
    
    - The `task1` function maps the grid positions of all tiles.

2. **Tile Classification**:
    - Each tile is analyzed using the `detect_tile` function, which identifies numbers, mines, and empty spaces based on pixel colors.

3. **Logic Processing**:
    - A simple algorithm evaluates neighbors and assigns safety or danger scores to unknown tiles.

4. **Hint Overlay**:
    - The application interacts with tiles based on the evaluation:

        - Clicks safe tiles.
        
        - Flags suspected mines.
        
        - Ignores tiles marked as unknown.

5. **Reiteration**:
    - The process repeats until the game is solved or fails due to a random guess.

## Customization

- **Change `MAIN_COLOR`**:

    Update the primary gray shade for better field detection on custom or altered game boards.
