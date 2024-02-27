import glob
import os
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
grey = (145, 145, 145)
red = (208, 0, 0)
green = (32, 191, 85)
blue = (30, 150, 252)
yellow = (252, 243, 0)
pink = (230, 55, 191)
purple = (141, 0, 201)
ice_blue = (155, 243, 240)
light_green = (179, 255, 179)
orange = (255, 87, 20)

# Function to wrap text onto new lines
def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        if font.size(current_line + ' ' + word)[0] <= max_width:
            current_line += ' ' + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def glob_filetypes(root_dir, *patterns):
                return [path
                        for pattern in patterns
                        for path in glob.glob(os.path.join(root_dir, pattern))]

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
    # annotated_image = cv2.flip(annotated_image, 1)
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
