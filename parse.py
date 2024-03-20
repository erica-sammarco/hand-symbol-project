import re

import numpy as np


class Category:
    def __init__(self, index, score, display_name, category_name):
        self.index = index
        self.score = score
        self.display_name = display_name
        self.category_name = category_name


class Landmark:
    def __init__(self, x, y, z, visibility, presence):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence

    @classmethod
    def to_numpy(cls, landmark_list):
        return np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmark_list])


class NormalizedLandmark(Landmark):
    pass


def parse(data_string):
    handedness = []
    normalized_landmarks = []
    world_landmarks = []
    # Extract handedness data using regex
    handedness_match = re.search(r'handedness=\[\[.*?\]\]', data_string)
    if handedness_match:
        handedness_data = handedness_match.group()

        # Extract hand_landmarks data using regex
        hand_landmarks_match = re.search(r'hand_landmarks=\[\[.*?\]\]',
                                         data_string)
        hand_landmarks_data = hand_landmarks_match.group()
        # print(hand_landmarks_data)

        # Extract hand_world_landmarks data using regex
        hand_world_landmarks_match = re.search(
            r'hand_world_landmarks=\[\[.*?\]\]', data_string)
        hand_world_landmarks_data = hand_world_landmarks_match.group()

        # Parse handedness data
        category_entries = re.findall(r'Category\(.*?\)', handedness_data)
        for entry in category_entries:
            category_match = re.search(
                r"index=(.*?), score=(.*?), display_name='(.*?)', category_name='(.*?)",
                entry)
            index = int(category_match.group(1))
            score = float(category_match.group(2))
            display_name = category_match.group(3)
            category_name = category_match.group(4)
            category = Category(index, score, display_name, category_name)
            handedness.append(category)

        # Parse hand_landmarks data
        landmark_entries = re.findall(r'NormalizedLandmark\(.*?\)',
                                      hand_landmarks_data)

        hand_landmarks = []
        for idx, entry in enumerate(landmark_entries):
            if len(hand_landmarks) == 21:
                normalized_landmarks.append(hand_landmarks)
                hand_landmarks = []

            match = re.search(
                r'x=(.*?), y=(.*?), z=(.*?), visibility=(.*?), presence=(.*?)\)',
                entry)
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            visibility = float(match.group(4))
            presence = float(match.group(5))
            normalized_landmark = NormalizedLandmark(x, y, z, visibility,
                                                     presence)
            hand_landmarks.append(normalized_landmark)

        normalized_landmarks.append(hand_landmarks)
        hand_landmarks = []

        # Parse hand_world_landmarks data
        landmark_entries = re.findall(r'Landmark\(.*?\)',
                                      hand_world_landmarks_data)
        for entry in landmark_entries:
            if len(hand_landmarks) == 21:
                world_landmarks.append(hand_landmarks)
                hand_landmarks = []

            match = re.search(
                r'x=(.*?), y=(.*?), z=(.*?), visibility=(.*?), presence=(.*?)\)',
                entry)
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            visibility = float(match.group(4))
            presence = float(match.group(5))
            world_landmark = NormalizedLandmark(x, y, z, visibility, presence)
            hand_landmarks.append(normalized_landmark)

        world_landmarks.append(hand_landmarks)
        hand_landmarks = []

    return {
        "handedness": handedness,
        "normalized": normalized_landmarks,
        "world": world_landmarks
    }
