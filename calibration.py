import pygame
import sys
import json
import glob
import os
import random
import string
import cv2
import datetime
import copy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# file with timestamp for result file
# file with result-HASH.txt

# Opening JSON file
constants = open('./constants.json')
trial_info = json.load(constants)
print(trial_info)

# Accessing pose files
def glob_filetypes(root_dir, *patterns):
    return [path
            for pattern in patterns
            for path in glob.glob(os.path.join(root_dir, pattern))]

image_types = ["*.jpg", "*.jpeg", "*.png"]
filepath = trial_info["POSE_FILES"]
files = glob_filetypes(filepath, "*.jpg", "*.jpeg", "*.png")
files.sort()
print(files)

# Initialize Pygame
pygame.init()

# Set up display
width, height = 600, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Calibration Trial")

# Set up font
font = pygame.font.Font(None, 36)

# Load images
images = []
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
    
# Load sound
pygame.mixer.init()
sound = pygame.mixer.Sound(trial_info["PROMPT_SOUND"]) 

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (17, 21, 255)
green = (56, 196, 65)
red = (245, 64, 43)

# Set initial state
trial_running = False
sound_played = False
image_index = 0

# Time Keeping
last_event = 0
play_sound = 0
stop_image = 0
start = 0
sound_time = 0
end = 0

# Initialize Clock
clock = pygame.time.Clock()

# Initialize Camera
cap = cv2.VideoCapture(0)

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)

MARGIN = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  #print(rgb_image)
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    # print(annotated_image)
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

result_filename = ""
result_doc = open("./results/a_result_record.txt", "a")
results = []

RIGHT_HAND = "Right"
LEFT_HAND = "Left"

def process_results(results):
    result_file.write('-------- Result Matrix ---------\n')
    result_file.write(' 0 = LEFT, 1 = RIGHT\n')
    result_file.write('HAND - 63 NORMALIZED LANDMARKS - TRIAL STATUS - TIME\n')
    matrix = []
    for feature in results:
        hand_landmarker = feature['handLandmarker']
        handedness = hand_landmarker.handedness
        landmarks = hand_landmarker.hand_landmarks
        for idx in range(0, len(handedness)):
            row = []
            if handedness[idx][0].display_name == RIGHT_HAND:
                row.append(1)
            elif handedness[idx][0].display_name == LEFT_HAND:
                row.append(0)
            for point in landmarks[idx]:
                row.append(point.x)
                row.append(point.y)
                row.append(point.z)
            row.append(feature['trialStatus'])
            row.append(feature['time'])
            matrix.append(row)
    result_file.write(str(matrix)+"\n")


            
            

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            result_file.write('-------- Program Quit : {} ---------\n'.format(datetime.datetime.now()))
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if (trial_running):
                    trial_running = False
                    sound_played = False
                    last_event, play_sound, stop_image, start, end, sound_played = 0, 0, 0, 0, 0, 0
                    result_file.write('-------- Trial Interrupted : {} ---------\n'.format(datetime.datetime.now()))
                else:
                    # Set up image order
                    values = list(range(0, len(files)))
                    image_order = []

                    for r in range(0, int(trial_info["NUM_ITERATIONS"])):
                        random.shuffle(values)
                        image_order = image_order + values
                    print(image_order)

                    # Set up result document
                    result_filename = 'result_{}.txt'.format(''.join(random.choices(string.ascii_letters + string.digits, k=6)))
                    result_file = open("./results/" + result_filename, 'a')
                    result_doc.write('Date/Time: {} , Filename: {}\n'.format(datetime.datetime.now(), result_filename))

                    result_file.write('Trial Parameters :\n')
                    for val in trial_info:
                        result_file.write(' {} : {}\n'.format(val, trial_info[val]))
                    result_file.write('\nImages Used :\n')
                    for idx, f in enumerate(files):
                        result_file.write(' {}: {}\n'.format(idx, f))
                    result_file.write('-------- Trial Begin : {}---------\n'.format(datetime.datetime.now()))

                    trial_running = True
                    last_event = pygame.time.get_ticks()
                    play_sound = last_event + (trial_info["PREP_TIME"] * 1000)
                    stop_image = last_event + (trial_info["TRIAL_LENGTH"] * 1000)
                    start = last_event
                

    # Clear the screen
    screen.fill(black)

    # Display video
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(img)

        annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
        #frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # cv2.imshow("Live Feed", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        converted = pygame.surfarray.make_surface(annotated_image.swapaxes(0, 1))
        # frame = pygame.surfarray.make_surface(frame)
        converted = pygame.transform.scale(converted, (converted.get_width()/5, converted.get_height()/5))
        screen.blit(converted, (width // 2 - converted.get_width() // 2, 20))

    # Display "Press Space to Begin" message
    if not trial_running:
        text = font.render("Press space to begin trial", True, blue)
        text_rect = text.get_rect(center=(width // 2, height // 2))
        screen.blit(text, text_rect)

    else:
        if image_index < len(image_order) : 
            image = images[image_order[image_index]]
            ticks = pygame.time.get_ticks()
            if(ticks >= start) :
                screen.blit(image, (width // 2 - image.get_width() // 2, ((height - converted.get_height()/5) // 2 - image.get_height() // 2) + (2 * converted.get_height()/5) + 40))
                frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": image_order[image_index], "time": ticks}
                results.append(frame_result)
                result_file.write(str(frame_result)+"\n")
            else : 
                frame_result = {"handLandmarker": copy.deepcopy(detection_result), "trialStatus": -1, "time": ticks}
                results.append(frame_result)
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
                start = last_event + (trial_info["TRIAL_GAP_LENGTH"] * 1000)
                play_sound = start + (trial_info["PREP_TIME"] * 1000)
                stop_image = start + (trial_info["TRIAL_LENGTH"] * 1000)

        else :
            trial_running = False
            sound_played = False 
            image_index = 0
            result_file.write('-------- Trial End : {}---------\n'.format(datetime.datetime.now()))
            process_results(results)

            

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(30)
