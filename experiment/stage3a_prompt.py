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
from experiment_values import PARTICIPANT_ID, POSES, NUM_POSES, USE_SEED

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
width, height = 600, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Hand Pose Prompt Trial")

# Set up font
title = pygame.font.Font(None, 36)
subtitle = pygame.font.Font(None, 30)
long_text = pygame.font.Font(None, 24)
line_space = 40

# Load images
images = []
for image_idx in CURR_POSES:
    original_image = pygame.image.load(files[image_idx])

    # Desired size for the image
    desired_size = height // 2

    # Calculate scaling factor
    scale_factor = desired_size / original_image.get_height()

    # Apply scaling factors to maintain aspect ratio
    image = pygame.transform.scale(original_image, (int(original_image.get_width() * scale_factor),
                                                    int(original_image.get_height() * scale_factor)))
    
    images.append(image)

# Initialize State
State = Enum('State', ['START', 'PROMPT', 'BREAK', 'DONE'])
current_state = State.START
sequence = generate(False)
seq_index = -1

# Initialize Clock
clock = pygame.time.Clock()

# Initialize Camera
cap = cv2.VideoCapture(0)

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='../hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)

# Time Keeping
last_event = 0
start = 0

result_doc = open("./prompt-results/a_result_record.txt", "a")
result_file = None

result_doc_read = open("./prompt-results/a_result_record.txt", "r")
prev_trials = result_doc_read.read().count(PARTICIPANT_ID)

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='../hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)

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
        wrapped_lines = wrap_text("When the image is displayed, make the hand position and press space as fast as possible.", long_text, width - 40)

        # Render each line onto a surface
        text_surfaces = [long_text.render(line, True, blue) for line in wrapped_lines]
        text = title.render("Press space to begin trial.", True, blue)        
        text_surfaces.insert(0, text)
        
        # Blit each text surface onto the window
        for i, text_surface in enumerate(text_surfaces):
            text_rect = text_surface.get_rect(center=(width // 2, height // 2 + i * line_space))
            screen.blit(text_surface, text_rect)

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

                    result_file = open('./prompt-results/{}_{}.txt'.format(PARTICIPANT_ID, prev_trials), 'a')
                    result_file.write('Trial Parameters :\n')
                    result_file.write(' POSES: {}\n'.format(CURR_POSES))
                    result_file.write(' USE_SEED: {}\n'.format(USE_SEED))
                    result_file.write(' SEQUENCE: {}\n'.format(sequence))

                    ticks = pygame.time.get_ticks()
                    result_file.write('-------- Trial Begin : {}, {}---------\n'.format(datetime.datetime.now(), ticks))

    if current_state is State.BREAK: 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        ticks = pygame.time.get_ticks()
        frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": -1, "time": datetime.datetime.now(), "ticks": ticks}
        result_file.write(str(frame_result)+"\n")

        if(ticks >= start):
            seq_index += 1
            current_state = State.PROMPT
            last_event = ticks
        
    if current_state is State.PROMPT:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_state = State.BREAK
                    last_event = pygame.time.get_ticks()
                    start = last_event + (BREAK_TIME * 1000)
        
        if seq_index < len(sequence) : 
            image_idx = CURR_POSES.index(sequence[seq_index])
            image = images[image_idx]
            ticks = pygame.time.get_ticks()
            
            screen.blit(image, (width // 2 - image.get_width() // 2, ((height - converted.get_height()/5) // 2 - image.get_height() // 2) + (2 * converted.get_height()/5) + 40))
            frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": sequence[seq_index], "time": datetime.datetime.now(), "ticks": ticks}
            result_file.write(str(frame_result)+"\n")
            
        else :
            current_state = State.DONE
            result_file.write('-------- Trial End : {}---------\n'.format(datetime.datetime.now()))
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