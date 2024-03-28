# Minesweeper-Hint
The hint is based on visual detection of the minesweeper field at https://minesweeper.online/.
To use it, start a game in the right half of the screen and turn on the code, you should see color hint in another widnow.
To exit the program simple press Q when focused on the hint widnow.

TROUUBLESHOOTING:

If the hint doesn't open, try to play with different zoom and birghtness settings. If even this doesn't help, you can try to replace
the MAIN_COLOR constant with the main shade of grey your minefield has (the biggest areas of the frame are of this color). You can find 
it out by pasting screenshot into https://imagecolorpicker.com/, and you will get the RGB code of that color, than try to play with different zoom values.

PROGRAM EXPLAINED:

The program is based on visual recoginition of the field, its grid, tiles using color filters and edge recognition. Then a simple algorithm assigns different
colors to tiles, green - safe, red - mine, yellow - wrong flag and purple - mine, but there are some yellow tiles nearby. 
