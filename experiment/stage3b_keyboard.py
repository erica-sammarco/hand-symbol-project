from enum import Enum
import pygame
import sys
import cv2
import datetime
import copy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from common import *
from generate_sequence import generate
from experiment_values import *

# How long of a gap should there be between 
# the user pressing space and the next image being displayed?
BREAK_TIME = 2

# Poses used in this trial
CURR_POSES = POSES[0:NUM_POSES]

image_types = ["*.jpg", "*.jpeg", "*.png"]
filepath = './images/{}'.format(PARTICIPANT_ID)
files = glob_filetypes(filepath, "*.jpg", "*.jpeg", "*.png")
files.sort()
print(files)

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 850
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Hand Pose Keyboard Trial")

# Set up font
target_letter = pygame.font.Font(None, 150)
title = pygame.font.Font(None, 40)
subtitle = pygame.font.Font(None, 30)
long_text = pygame.font.Font(None, 24)
line_space = 40

pose_width = 100

# Load images
images = []
for image_idx in CURR_POSES:
    original_image = pygame.image.load(files[image_idx])

    # Calculate scaling factor
    scale_factor = pose_width / original_image.get_height()

    # Apply scaling factors to maintain aspect ratio
    image = pygame.transform.scale(original_image, (int(original_image.get_width() * scale_factor),
                                                    int(original_image.get_height() * scale_factor)))
    
    images.append(image)

# Define the keyboard layout
keyboard_layout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

# Define the color options
colors = [red, green, blue, yellow, pink, ice_blue, purple, light_green, orange, white]

# Initialize State
State = Enum('State', ['START', 'PROMPT', 'BREAK', 'DONE', 'PAUSE'])
current_state = State.START
sequence = generate(True)
seq_index = -1
phrase_index = -1

# Initialize Clock
clock = pygame.time.Clock()

# Initialize Camera
cap = open_camera()

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='../hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)

# Time Keeping
last_event = 0
start = 0

result_doc = open("./keyboard-results/a_result_record.txt", "a")
result_file = None

result_doc_read = open("./keyboard-results/a_result_record.txt", "r")
prev_trials = result_doc_read.read().count(PARTICIPANT_ID)

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='../hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)

# Get the color of the given key based on the current point in the sequence
def get_key_color(key, seq_idx):
    if seq_idx == -1:
        return white
    else:
        key_pose = sequence[seq_index][key]
        color_idx = CURR_POSES.index(key_pose)
        return colors[color_idx]

key_width = 60
key_height = 60
key_gap = 15
keyboard_y_offset = height - (len(keyboard_layout) * (key_height + key_gap)) - 60

# Function to draw the keyboard
def draw_keyboard(screen, font, seq_idx):
    for row_idx, row in enumerate(keyboard_layout):
        x_offset = (width - (len(row) * (key_width + key_gap))) // 2
        for col_idx, key in enumerate(row):
            rect = pygame.Rect(x_offset + col_idx * (key_width + key_gap), keyboard_y_offset + row_idx * (key_height + key_gap), key_width, key_height)
            pygame.draw.rect(screen, black, rect)  # Fill the key with color
            color = get_key_color(key, seq_idx)
            pygame.draw.rect(screen, color, rect, 3, 5)  # Draw outline for the key
            text_surface = font.render(key, True, white)
            text_rect = text_surface.get_rect(center=rect.center)
            screen.blit(text_surface, text_rect)

pose_gap = 30
poses_x_offset = (width - (len(CURR_POSES) * (pose_width + pose_gap))) // 2
poses_y_offset = keyboard_y_offset - pose_width - pose_gap

def draw_poses(screen):
    for idx, p in enumerate(CURR_POSES):
        image = images[idx] 
        x_pos = poses_x_offset + idx * (pose_width + pose_gap)
        screen.blit(image, (x_pos, poses_y_offset))
        rect = pygame.Rect(x_pos, poses_y_offset, pose_width, pose_width)
        color = colors[idx]
        pygame.draw.rect(screen, color, rect, 3, 5)

def draw_text(screen, seq_idx, target):
    if seq_idx != -1:
        text_surface = subtitle.render(PHRASE, True, white)
        top_left_x = (width - text_surface.get_width())//2
        top_left_y = 265
        text_rect = text_surface.get_rect(topleft=(top_left_x, top_left_y))

        # Render the text up to the sequence index
        so_far_surface = subtitle.render(PHRASE[:seq_idx], True, grey)
        screen.blit(so_far_surface, text_rect)

        # Render the second half of the text with color2
        # second_half_surface = title.render(PHRASE[seq_idx+1:], True, black)
        # second_half_rect = second_half_surface.get_rect(topleft=(x + text_surface.get_width() // 2, y))
        # screen.blit(second_half_surface, second_half_rect)

    if target:
        # Render the current target letter
        target_surface = target_letter.render(PHRASE[seq_idx].upper(), True, white)
        target_rect = target_surface.get_rect(center=(width//2, poses_y_offset - (target_letter.get_height() // 1.5)))
        screen.blit(target_surface, target_rect)

def increment_phrase_idx(phrase_idx):
    next_idx = phrase_idx + 1
    if next_idx >= len(PHRASE):
        return next_idx
    if PHRASE[next_idx] == " ":
        return increment_phrase_idx(next_idx)
    else:
        return next_idx

while True:
    # Clear the screen
    screen.fill(black)

    if current_state is not State.DONE:
        # Display video
        ret, frame = cap.read()
        if ret:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(img)

            annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
        
            converted = pygame.surfarray.make_surface(annotated_image.swapaxes(0, 1))
            converted = pygame.transform.scale(converted, (converted.get_width()/5, converted.get_height()/5))
            screen.blit(converted, (width // 2 - converted.get_width() // 2, 20))

    if current_state is State.START:
        # Wrap the paragraph onto new lines
        wrapped_lines = wrap_text("When the next letter is displayed, find and make the correct hand position and press space as fast as possible.", long_text, width - 40)

        # Render each line onto a surface
        text_surfaces = [long_text.render(line, True, blue) for line in wrapped_lines]
        text = title.render("Press space to begin trial.", True, blue)        
        text_surfaces.insert(0, text)
        
        # Blit each text surface onto the window
        for i, text_surface in enumerate(text_surfaces):
            text_rect = text_surface.get_rect(center=(width // 2, height // 2.5 + i * line_space))
            screen.blit(text_surface, text_rect)
        
        draw_keyboard(screen, title, -1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_state = State.BREAK
                    last_event = pygame.time.get_ticks()
                    start = last_event + (BREAK_TIME * 1000)
                    result_doc.write('Date/Time: {} , Participant ID: {}\n'.format(datetime.datetime.now(), PARTICIPANT_ID))

                    result_file = open('./keyboard-results/{}_{}.txt'.format(PARTICIPANT_ID, prev_trials), 'a')
                    result_file.write('Trial Parameters :\n')
                    result_file.write(' POSES: {}\n'.format(CURR_POSES))
                    result_file.write(' BREAK_TIME: {}\n'.format(BREAK_TIME))
                    result_file.write(' USE_SEED: {}\n'.format(USE_SEED))
                    result_file.write(' SEED: {}\n'.format(SEED))
                    result_file.write(' PHRASE: {}\n'.format(PHRASE))
                    test_trg = "["
                    for idx, char in enumerate(char.upper() for char in PHRASE if char != ' '):
                        test_trg = test_trg + str(sequence[idx][char]) + ", "
                    test_trg += "]"
                    result_file.write(' SEQUENCE: {}\n'.format(test_trg))

                    ticks = pygame.time.get_ticks()
                    result_file.write('-------- Trial Begin : {}, {}---------\n'.format(datetime.datetime.now(), ticks))

    if current_state is State.BREAK: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    current_state = State.PAUSE

        draw_text(screen, phrase_index, False)
        draw_poses(screen)
        draw_keyboard(screen, title, -1)

        curr_trial = title.render('Trial {} of {}'.format(seq_index + 1, len(sequence)), True, grey)
        curr_trial_rect = curr_trial.get_rect(center=(width // 2, height - 30))
        screen.blit(curr_trial, curr_trial_rect)

        ticks = pygame.time.get_ticks()
        frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": -1, "time": datetime.datetime.now(), "ticks": ticks}
        result_file.write(str(frame_result)+"\n")

        if(ticks >= start):
            seq_index += 1
            if phrase_index == -1:
                phrase_index = increment_phrase_idx(phrase_index)
            current_state = State.PROMPT
            last_event = ticks
        
    if current_state is State.PROMPT:
        if seq_index < len(sequence): 
            draw_text(screen, phrase_index, True)
            draw_poses(screen)
            draw_keyboard(screen, title, seq_index)

            curr_trial = title.render('Trial {} of {}'.format(seq_index + 1, len(sequence)), True, grey)
            curr_trial_rect = curr_trial.get_rect(center=(width // 2, height - 30))
            screen.blit(curr_trial, curr_trial_rect)

            frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": sequence[seq_index][PHRASE[phrase_index].upper()], "time": datetime.datetime.now(), "ticks": ticks}
            result_file.write(str(frame_result)+"\n")
            
        elif seq_index >= len(sequence):
            current_state = State.DONE
            result_file.write('-------- Trial End : {}---------\n'.format(datetime.datetime.now()))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_state = State.BREAK
                    phrase_index = increment_phrase_idx(phrase_index)
                    last_event = pygame.time.get_ticks()
                    start = last_event + (BREAK_TIME * 1000)
                if event.key == pygame.K_p:
                    current_state = State.PAUSE
    if current_state is State.PAUSE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_state = State.BREAK
                    last_event = pygame.time.get_ticks()
                    start = last_event + (BREAK_TIME * 1000)

        frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": -2, "time": datetime.datetime.now(), "ticks": ticks}
        result_file.write(str(frame_result)+"\n")

        pause_txt = title.render("Experiment paused. Press space to continue.", True, blue) 
        pause_txt_rect = pause_txt.get_rect(center=(width // 2, height // 2+line_space))
        screen.blit(pause_txt, pause_txt_rect)
    if current_state is State.DONE:
        text = title.render("Trial complete", True, blue)
        text2 = title.render("Press any key to quit.", True, blue) 
        text_rect = text.get_rect(center=(width // 2, height // 2))
        text2_rect = text2.get_rect(center=(width // 2, height // 2+line_space))
        screen.blit(text, text_rect)
        screen.blit(text2, text2_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                pygame.quit()
                sys.exit()
    
    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(30)