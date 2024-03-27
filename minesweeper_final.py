import time
import pyautogui
import cv2 as cv
import numpy as np
from typing import Tuple, List

Position = Tuple[int, int]
MAIN_COLOR = 198


def task2(image: np.ndarray) -> Tuple[Position, Position]:
    mask = cv.inRange(image, np.array([MAIN_COLOR, MAIN_COLOR, MAIN_COLOR]),
                      np.array([MAIN_COLOR, MAIN_COLOR, MAIN_COLOR]))
    grayscale_image = cv.cvtColor(cv.bitwise_and(image, image, mask=mask),
                                  cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(grayscale_image, cv.RETR_TREE, 2)
    field_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(field_contour)
    field_position = ((x, y), (x + w, y + h))
    return field_position


def task1(image: np.ndarray, minefield: Tuple[Position, Position]) \
        -> Tuple[np.ndarray, np.ndarray]:
    image = image[minefield[0][1]:minefield[1][1],
                  minefield[0][0]:minefield[1][0]]
    mask = cv.inRange(image, np.array([MAIN_COLOR, MAIN_COLOR, MAIN_COLOR]),
                      np.array([MAIN_COLOR, MAIN_COLOR, MAIN_COLOR]))
    grayscale_image = cv.cvtColor(cv.bitwise_and(image, image, mask=mask),
                                  cv.COLOR_BGR2GRAY)

    contours, hierarchy = cv.findContours(grayscale_image, cv.RETR_TREE, 2)

    max_area = 0.1
    second_max_area = 0.0
    largest_contour = -1
    second_largest_contour = None

    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > max_area:
            second_max_area = max_area
            second_largest_contour = largest_contour
            max_area = area
            largest_contour = i
        elif area > second_max_area:
            second_max_area = area
            second_largest_contour = i

    child_index = int(hierarchy[0][second_largest_contour][2])
    children_count = 0
    same_size = True
    first_size = (contours[child_index][2][0][0] -
                  contours[child_index][0][0][0])
    max_size = 0
    while child_index != -1:
        children_count += 1
        size = contours[child_index][2][0][0] - contours[child_index][0][0][0]
        if size != first_size:
            same_size = False
            if size > first_size:
                max_size = size
            else:
                max_size = first_size
        child_index = int(hierarchy[0][child_index][0])

    if same_size:
        first_child = contours[int(hierarchy[0][second_largest_contour][2])]
        first_child_size = first_child[2][0][0] - first_child[0][0][0]
        second_child = contours[
            int(hierarchy[0][int(hierarchy[0][second_largest_contour][2])][0])]
        second_child_size = second_child[2][0][0] - second_child[0][0][0]
        difference = first_child[2][0][0] - second_child[0][0][0]
        if second_child_size + first_child_size < difference - 2:
            tile_size = first_size + (
                difference - second_child_size - first_child_size) // 2 + 4
        else:
            tile_size = second_child_size + 3
    else:
        tile_size = max_size + 3

    if second_largest_contour is not None:
        x, y, w, h = cv.boundingRect(contours[second_largest_contour])
    else:
        x, y, w, h = (0, 0, 0, 0)

    x += 6
    y += 6
    w -= 10
    h -= 10

    x_cords = []
    y_cords = []

    i = 0

    while i <= w + 5:
        x_cords.append(int(i) + x + int(minefield[0][0]))
        i += tile_size

    i = 0

    while i <= h + 5:
        y_cords.append(int(i) + y + int(minefield[0][1]))
        i += tile_size

    return np.array(x_cords), np.array(y_cords)


def detect_tile(tile: np.ndarray) -> str:
    unique_colors = np.unique(tile.reshape(-1, tile.shape[2]), axis=0)
    three = False
    five = False
    black = False
    for unique_color in unique_colors:
        if cv.inRange(unique_color, np.array([210, 0, 0], dtype=np.uint8),
                      np.array([255, 30, 30], dtype=np.uint8)).all():
            return "1"
        elif cv.inRange(unique_color, np.array([0, 90, 0], dtype=np.uint8),
                        np.array([30, 140, 30], dtype=np.uint8)).all():
            return "2"
        elif cv.inRange(unique_color, np.array([0, 0, 210], dtype=np.uint8),
                        np.array([30, 30, 255], dtype=np.uint8)).all():
            three = True
        elif cv.inRange(unique_color, np.array([90, 0, 0], dtype=np.uint8),
                        np.array([140, 30, 30], dtype=np.uint8)).all():
            return "4"
        elif cv.inRange(unique_color, np.array([0, 0, 50], dtype=np.uint8),
                        np.array([40, 40, 170], dtype=np.uint8)).all():
            five = True
        elif cv.inRange(unique_color, np.array([50, 50, 0], dtype=np.uint8),
                        np.array([170, 170, 40], dtype=np.uint8)).all():
            return "6"
        elif cv.inRange(unique_color, np.array([0, 0, 0], dtype=np.uint8),
                        np.array([50, 50, 50], dtype=np.uint8)).all():
            black = True
    if black and three:
        return "M"
    elif black:
        return "7"
    elif three:
        return "3"
    elif five:
        return "5"
    mask = cv.inRange(tile, np.array([230, 230, 230], dtype=np.uint8),
                      np.array([255, 255, 255], dtype=np.uint8))
    pixel_count = cv.countNonZero(mask)
    total_pixels = tile.shape[0] * tile.shape[1]
    ratio = pixel_count / total_pixels
    if ratio > 0.1:
        return "?"
    mask = cv.inRange(tile, np.array([70, 70, 70], dtype=np.uint8),
                      np.array([180, 180, 180], dtype=np.uint8))
    pixel_count = cv.countNonZero(mask)
    ratio = pixel_count / total_pixels
    if ratio > 0.5:
        return "8"
    return "0"


def task3(image: np.ndarray, field_grid: Tuple[np.ndarray, np.ndarray]) \
        -> List[str]:
    game_state = []
    tile_size = field_grid[0][1] - field_grid[0][0]
    for y in field_grid[1][:-1]:
        line = ""
        for x in field_grid[0][:-1]:
            tile = image[y:y + tile_size, x:x + tile_size]
            tile_symbol = detect_tile(tile)
            line += tile_symbol
        game_state.append(line)
    return game_state


def evaluate_tile(tile_index: Tuple[int, int],
                  minefield: List[str],
                  hints: List[str]) -> List[Tuple[Tuple[int, int], str]]:
    changes = []
    number_of_mines = int(minefield[tile_index[0]][tile_index[1]])
    number_of_flags = 0
    number_of_unknowns = 0
    number_of_dangerous = 0
    neighbours = []
    number_of_correct_flags = 0
    number_of_safe = 0
    for i in range(max(0, tile_index[0] - 1),
                   min(tile_index[0] + 2, len(minefield))):
        for j in range(max(0, tile_index[1] - 1),
                       min(tile_index[1] + 2, len(minefield[0]))):
            if i != tile_index[0] or j != tile_index[1]:
                neighbours.append((i, j))
                if minefield[i][j] == "M":
                    number_of_flags += 1
                elif minefield[i][j] == "?":
                    number_of_unknowns += 1
                if hints[i][j] == "3" or hints[i][j] == "4":
                    number_of_dangerous += 1
                if hints[i][j] == "2":
                    number_of_safe += 1

    if number_of_flags > number_of_mines:
        for neighbour in neighbours:
            if minefield[neighbour[0]][neighbour[1]] == "M" and \
                    hints[neighbour[0]][neighbour[1]] == "0":
                changes.append((neighbour, "1"))
    else:
        for neighbour in neighbours:
            if minefield[neighbour[0]][neighbour[1]] == "M" and \
                    hints[neighbour[0]][neighbour[1]] == "0":
                number_of_correct_flags += 1

    if number_of_correct_flags == number_of_mines:
        for neighbour in neighbours:
            if minefield[neighbour[0]][neighbour[1]] == "?" and \
                    hints[neighbour[0]][neighbour[1]] != "2":
                changes.append((neighbour, "2"))
    if number_of_correct_flags + number_of_dangerous == number_of_mines:
        for neighbour in neighbours:
            if minefield[neighbour[0]][neighbour[1]] == "?" and \
                    hints[neighbour[0]][neighbour[1]] == "0":
                changes.append((neighbour, "2"))

    if (number_of_unknowns - number_of_safe +
            number_of_flags - number_of_correct_flags ==
            number_of_mines - number_of_correct_flags):
        for neighbour in neighbours:
            if minefield[neighbour[0]][neighbour[1]] == "?" and \
                    hints[neighbour[0]][neighbour[1]] == "0":
                changes.append((neighbour, "3"))
            elif minefield[neighbour[0]][neighbour[1]] == "M" and \
                    hints[neighbour[0]][neighbour[1]] == "1":
                changes.append((neighbour, "4"))
    return changes


def mine_check(tile_index: Tuple[int, int],
               minefield: List[str],
               hints: List[str]) -> List[Tuple[Tuple[int, int], str]]:
    changes = []
    number_of_mines = int(minefield[tile_index[0]][tile_index[1]])
    number_of_flags = 0
    neighbours = []
    for i in range(max(0, tile_index[0] - 1),
                   min(tile_index[0] + 2, len(minefield))):
        for j in range(max(0, tile_index[1] - 1),
                       min(tile_index[1] + 2, len(minefield[0]))):
            if i != tile_index[0] or j != tile_index[1]:
                neighbours.append((i, j))
                if minefield[i][j] == "M":
                    number_of_flags += 1
    if number_of_flags > number_of_mines:
        for neighbour in neighbours:
            if minefield[neighbour[0]][neighbour[1]] == "M" and \
                    hints[neighbour[0]][neighbour[1]] == "0":
                changes.append((neighbour, "1"))
    return changes


def task(minefield: List[str]) -> List[str]:
    helper = [len(row) * "0" for row in minefield]
    non_important = ("?", "0", "M")
    for y in range(len(minefield)):
        for x in range(len(minefield[0])):
            if minefield[y][x] not in non_important:
                for change in mine_check((y, x), minefield, helper):
                    helper[change[0][0]] = helper[change[0][0]][
                        :change[0][1]] + change[1] + \
                        helper[change[0][0]][
                        change[0][1] + 1:]

    change_count = 1
    while change_count != 0:
        change_count = 0
        for y in range(len(minefield)):
            for x in range(len(minefield[0])):
                if minefield[y][x] not in non_important:
                    for change in evaluate_tile((y, x), minefield, helper):
                        change_count += 1
                        helper[change[0][0]] = helper[change[0][0]][
                            :change[0][1]] + change[1] + \
                            helper[change[0][0]][
                            change[0][1] + 1:]
    return helper


def final(image: np.ndarray, helper: List[str],
          field_grid: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    x_tile = 0
    y_tile = 0
    colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    tile_size = field_grid[0][1] - field_grid[0][0]
    overlay = np.zeros_like(image, dtype=np.uint8)

    for y in field_grid[1][:-1]:
        for x in field_grid[0][:-1]:
            index = int(helper[y_tile][x_tile])
            if index != 0:
                cv.rectangle(overlay, (x, y), (x + tile_size, y + tile_size),
                             colors[index - 1], thickness=-1)
            x_tile += 1
        x_tile = 0
        y_tile += 1
    image = cv.addWeighted(image, 1, overlay, 0.5, 0)
    return image


while True:
    screenshot = np.array(pyautogui.screenshot())
    try:
        right_side = screenshot[:, len(screenshot[0]) // 2:]
        right_side = cv.cvtColor(right_side, cv.COLOR_RGB2BGR)
        field = task2(right_side)
        grid = task1(right_side, field)
        hint = task(task3(right_side, grid))
        output = final(right_side, hint, grid)[field[0][1]:field[1][1],
                                               field[0][0]:field[1][0]]
        cv.imshow("Result", output)
    except (IndexError, ValueError):
        print("Turn on: https://minesweeper.online/ in the right part of the screen.")
        time.sleep(0.1)
    if cv.waitKey(1) == ord("q"):
        break
