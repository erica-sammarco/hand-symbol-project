import pygame
from pygame.locals import *

# Constants for colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Key Colors
RED = (208, 0, 0)
GREEN = (32, 191, 85)
BLUE = (30, 150, 252)
YELLOW = (252, 243, 0)
PINK = (230, 55, 191)
PURPLE = (141, 0, 201)
ICE_BLUE = (155, 243, 240)
LIGHT_GREEN = (179, 255, 179)
ORANGE = (255, 87, 20)

width, height = 800, 600

# Define the keyboard layout
keyboard_layout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

# Define the color options
colors = [RED, GREEN, BLUE, YELLOW, PINK, ICE_BLUE, PURPLE, LIGHT_GREEN, ORANGE, WHITE]

# Function to draw the keyboard
def draw_keyboard(screen, font):
    key_width = 60
    key_height = 60
    key_gap = 10
    y_offset = height - (len(keyboard_layout) * (key_height + key_gap)) - 20

    for row_idx, row in enumerate(keyboard_layout):
        x_offset = (width - (len(row) * (key_width + key_gap))) // 2
        for col_idx, key in enumerate(row):
            rect = pygame.Rect(x_offset + col_idx * (key_width + key_gap), y_offset + row_idx * (key_height + key_gap), key_width, key_height)
            pygame.draw.rect(screen, BLACK, rect)  # Fill the key with color
            color_idx = col_idx % len(colors)
            pygame.draw.rect(screen, colors[color_idx], rect, 2, 5)  # Draw outline for the key
            text_surface = font.render(key, True, WHITE)
            text_rect = text_surface.get_rect(center=rect.center)
            screen.blit(text_surface, text_rect)

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Custom Color Keyboard')
font = pygame.font.Font(None, 36)

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    screen.fill(BLACK)
    # draw_images(screen)
    draw_keyboard(screen, font)
    pygame.display.flip()

pygame.quit()