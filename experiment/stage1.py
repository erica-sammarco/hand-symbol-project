import pygame
import sys
import cv2
import copy
import random
import string
import os
import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from enum import Enum
from common import *

# CONSTANTS:
NUM_POSES = 6
NUM_ITERATIONS = 7
PREP_TIME = 1.5
TRIAL_LENGTH = 1
TRIAL_GAP_LENGTH = 1
ACQUIRE_IMAGES = True
# If you would like to use previously loaded images for calibration
# store the filepath to the images folder here:
IMG_FILEPATH = "./images/original_poses"

# Initialize Pygame
pygame.init()

# Set up display
width, height = 600, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Experiment Set-Up, Calibration")

# Set up font
title = pygame.font.Font(None, 36)
subtitle = pygame.font.Font(None, 30)
long_text = pygame.font.Font(None, 24)
line_space = 40

# Initialize State
State = Enum('State', ['START', 'ACQUIRE', 'CAL_START', 'CALIBRATE', 'DONE', 'PAUSE'])
current_state = State.START if ACQUIRE_IMAGES else State.CAL_START
old_state = current_state
participant_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
img_counter = 0
initialized = False

# Create the images folder if it doesn't exist
if(ACQUIRE_IMAGES): 
    IMG_FILEPATH = './images/{}'.format(participant_id)
    os.makedirs(IMG_FILEPATH, exist_ok=True)

# Initialize Clock
clock = pygame.time.Clock()

# Initialize Camera
cap = cv2.VideoCapture(0)

# Define the top-left and bottom-right coordinates of the square
left_top = (175, 200)
right_top = (1200, 200)
sq_size = 600
top_left = None
bottom_right = None

def draw_crop_square(frame):
    return cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 4)

desired_height = 215
desired_width = 385
def scale_img(img):
    # Calculate scaling factor
    height_scale_factor = desired_height / img.get_height()
    width_scale_factor = desired_width / img.get_width()

    # Apply scaling factors to maintain aspect ratio
    image = pygame.transform.scale(img, (int(img.get_width() * width_scale_factor),
                                        int(img.get_height() * height_scale_factor)))
    
    return image


# Set up CALIBRATE state
sound_played = False
image_index = 0
image_order = []
images = []

# Time Keeping
last_event = 0
play_sound = 0
stop_image = 0
start = 0
sound_time = 0
end = 0

# Load sound
pygame.mixer.init()
sound = pygame.mixer.Sound("../beep-2.mp3") 

result_filename = participant_id+".txt"
result_doc = open("./calibration-results/a_result_record.txt", "a")
result_file = None

paricipant_id_doc = open("./participants.txt", "a")
paricipant_id_doc.write('Date/Time: {} , Participant ID: {}\n'.format(datetime.datetime.now(), participant_id))

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
        ret, original_frame = cap.read()
        if ret:
            flipped = cv2.flip(original_frame, 1)
            frame = np.copy(flipped)
            if current_state is State.ACQUIRE and top_left and bottom_right:
                frame = draw_crop_square(frame)
            if current_state is State.CALIBRATE or current_state is State.CAL_START or current_state is State.PAUSE:
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(img)

                annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
            
                converted = pygame.surfarray.make_surface(annotated_image.swapaxes(0, 1))
                converted = scale_img(converted)
                screen.blit(converted, (width // 2 - converted.get_width() // 2, 20))

            else:
                converted = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
                converted = scale_img(converted)
                screen.blit(converted, (width // 2 - converted.get_width() // 2, 20))
    if current_state is State.START:
        text = title.render("Press L or R to begin experiment set up.", True, blue)
        text2 = subtitle.render("L or R indicates which hand will be used.", True, blue) 
        text_rect = text.get_rect(center=(width // 2, height // 2))
        text2_rect = text2.get_rect(center=(width // 2, height // 2+line_space))
        screen.blit(text, text_rect)
        screen.blit(text2, text2_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    top_left = left_top
                    bottom_right = (top_left[0] + sq_size, top_left[1] + sq_size)
                    current_state = State.ACQUIRE
                if event.key == pygame.K_r:
                    top_left = right_top
                    bottom_right = (top_left[0] + sq_size, top_left[1] + sq_size)
                    current_state = State.ACQUIRE
    if current_state is State.ACQUIRE:
        script = "You will now take photos of ten hand positions you feel comfortable making. Center your hand in the red square and press space when you are ready to take the photo. Position suggestions include: thumbs up, open palm, 'i love you', numbers..."
        # Wrap the paragraph onto new lines
        wrapped_lines = wrap_text(script, long_text, width - 40)

        # Render each line onto a surface
        text_surfaces = [long_text.render(line, True, blue) for line in wrapped_lines]
        count = title.render('Remaining Positions: {}'.format(NUM_POSES - img_counter), True, green)
        text_surfaces.insert(0, count)
        
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
                    x = top_left[0]
                    y = top_left[1]
                    cropped_frame = flipped[y:y+sq_size, x:x+sq_size]
                    cv2.imwrite('./images/{}/pose_{}.png'.format(participant_id, img_counter), cropped_frame)
                    img_counter += 1
                    if img_counter == NUM_POSES:
                        current_state = State.CAL_START
    if current_state is State.CAL_START:
        text = long_text.render("When the image of your hand pose is displayed, copy the position yourself.", True, blue)
        text2 = title.render("Press space to begin.", True, blue) 
        text_rect = text.get_rect(center=(width // 2, height // 2))
        text2_rect = text2.get_rect(center=(width // 2, height // 2+line_space))
        screen.blit(text, text_rect)
        screen.blit(text2, text2_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_state = State.CALIBRATE
    if current_state is State.CALIBRATE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    old_state = current_state
                    current_state = State.PAUSE
        
        curr_trial = title.render('Trial {} of {}'.format(image_index + 1, len(image_order)), True, blue)
        curr_trial_rect = curr_trial.get_rect(center=(width // 2, height - 50))
        screen.blit(curr_trial, curr_trial_rect)

        if not initialized:
            # Accessing pose files
            files = glob_filetypes(IMG_FILEPATH, "*.jpg", "*.jpeg", "*.png")
            files.sort()
            print(files)

            # Load images
            for image in files:
                original_image = pygame.image.load(image)

                # Desired size for the image
                desired_size = height // 2

                # Calculate scaling factor
                scale_factor = desired_size / original_image.get_height()

                # Apply scaling factors to maintain aspect ratio
                image = pygame.transform.scale(original_image, (int(original_image.get_width() * scale_factor),
                                                                int(original_image.get_height() * scale_factor)))
                
                images.append(image)
            
            # Set up image order
            values = list(range(0, len(files)))
            

            for r in range(0, int(NUM_ITERATIONS)):
                random.shuffle(values)
                image_order = image_order + values
            print(image_order)

            result_doc.write('Date/Time: {} , Filename: {}\n'.format(datetime.datetime.now(), result_filename))
            result_file = open("./calibration-results/" + result_filename, 'a')

            result_file.write('Trial Parameters :\n')
            result_file.write(' NUM_POSES: {}\n'.format(NUM_POSES))
            result_file.write(' NUM_ITERATIONS: {}\n'.format(NUM_ITERATIONS))
            result_file.write(' PREP_TIME: {}\n'.format(PREP_TIME))
            result_file.write(' TRIAL_LENGTH: {}\n'.format(TRIAL_LENGTH))
            result_file.write(' TRIAL_GAP_LENGTH: {}\n'.format(TRIAL_GAP_LENGTH))
            result_file.write(' ACQUIRE_IMAGES: {}\n'.format(ACQUIRE_IMAGES))

            result_file.write('\nImages Used :\n')
            for idx, f in enumerate(files):
                result_file.write(' {}: {}/{}\n'.format(idx, IMG_FILEPATH,f))
            result_file.write('-------- Trial Begin : {}---------\n'.format(datetime.datetime.now()))

            last_event = pygame.time.get_ticks()
            play_sound = last_event + (PREP_TIME * 1000)
            stop_image = last_event + ((PREP_TIME + TRIAL_LENGTH) * 1000)
            start = last_event
            
            initialized = True
        if image_index < len(image_order) and current_state is not State.PAUSE : 
            image = images[image_order[image_index]]
            ticks = pygame.time.get_ticks()
            if(ticks >= start) :
                screen.blit(image, (width // 2 - image.get_width() // 2, ((height - converted.get_height()/5) // 2 - image.get_height() // 2) + (2 * converted.get_height()/5) + 40))
                status = image_order[image_index] if sound_played  else -1
                frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": status, "time": ticks}
                result_file.write(str(frame_result)+"\n")
            else : 
                frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": -1, "time": ticks}
                result_file.write(str(frame_result)+"\n")
            if(ticks >= play_sound and sound_played == False) :
                pygame.mixer.Sound.play(sound)
                sound_played = True
                sound_time = ticks
            elif(ticks >= stop_image) :
                image_index += 1
                sound_played = False 
                end =  ticks
                print('Trial Start: {} ms'.format(start))
                print('Sound Played: {} ms'.format(sound_time))
                print('Trial End: {} ms'.format(end))
                last_event = ticks
                start = last_event + (TRIAL_GAP_LENGTH * 1000)
                play_sound = start + (PREP_TIME * 1000)
                stop_image = start + ((PREP_TIME + TRIAL_LENGTH) * 1000)
        elif image_index >= len(image_order):
            current_state = State.DONE
            result_file.write('-------- Trial End : {}---------\n'.format(datetime.datetime.now()))
    if current_state is State.PAUSE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_state = old_state
                    last_event = pygame.time.get_ticks()
                    start = last_event + (TRIAL_GAP_LENGTH * 1000)
                    play_sound = start + (PREP_TIME * 1000)
                    stop_image = start + ((PREP_TIME + TRIAL_LENGTH) * 1000)
                    sound_played = False

        status = -2
        frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": status, "time": ticks}
        result_file.write(str(frame_result)+"\n")

        pause_txt = title.render("Calibration paused. Press space to continue.", True, blue) 
        pause_txt_rect = pause_txt.get_rect(center=(width // 2, height // 2+line_space))
        screen.blit(pause_txt, pause_txt_rect)
    if current_state is State.DONE:
        text = title.render("Experiment set-up complete", True, blue)
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