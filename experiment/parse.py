import re
from collections import defaultdict

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


def parse_line(line):
    handedness = []
    normalized_landmarks = []
    world_landmarks = []
    # Extract handedness data using regex
    handedness_match = re.search(r'handedness=\[\[.*?\]\]', line)
    if handedness_match:
        handedness_data = handedness_match.group()

        # Extract hand_landmarks data using regex
        hand_landmarks_match = re.search(r'hand_landmarks=\[\[.*?\]\]',
                                         line)
        hand_landmarks_data = hand_landmarks_match.group()
        # print(hand_landmarks_data)

        # Extract hand_world_landmarks data using regex
        hand_world_landmarks_match = re.search(
            r'hand_world_landmarks=\[\[.*?\]\]', line)
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


def parse(s):
    """ Returns most common hand observed, as an array
    """
    landmark = defaultdict(list)
    landmark_world = defaultdict(list)
    trial_status = defaultdict(list)

    count = 0
    # Read the file and parse all hand landmark data
    for line_idx, line in enumerate(s.split('\n')):
        if not line or line[0] != '{':
            continue
        match = re.search(r"'trialStatus': (.*?),", line)
        trial = int(match.group(1))

        count += 1
        landmark_results = parse_line(line)

        for idx, hand in enumerate(landmark_results["handedness"]):
            # saw Left and left somewhere, just get lowercase
            name = hand.display_name.lower()

            # store
            landmark[name].append(landmark_results["normalized"][idx])
            landmark_world[name].append(landmark_results["world"][idx])
            trial_status[name].append(trial)

    # find which hand was most common in trial (we'll use this one)
    hand_common = max(landmark, key=lambda x: len(landmark.get(x)))

    # to numpy
    lmk = np.stack([Landmark.to_numpy(lmk_list)
                    for lmk_list in landmark[hand_common]], axis=2)
    lmk_world = np.stack([Landmark.to_numpy(lmk_list)
                          for lmk_list in landmark_world[hand_common]], axis=2)
    trigger = np.array(trial_status[hand_common])

    return lmk, lmk_world, trigger


def split_trial(trigger):
    """ gets index of each trial start and min length across all trials

    Args:
        trigger (np.array): (n) -1 where no trials are active. otherwise has
            index of trial

    Returns:
        idx_trial_start (np.array): (num_trial) the first index of each trial
        y (np.array): label of trial
        trial_length (int): number of indices contained in each trial
    """
    # we append a leading and trailing -1 to trial_labels
    trigger = np.insert(trigger, 0, -1)
    trigger = np.append(trigger, -1)

    # must be int (np.diff w/ input boolean outputs boolean?  lame)
    is_trial = (trigger != -1).astype(np.int8)
    idx_trial_start = np.where(np.diff(is_trial) == 1)[0]
    idx_trial_end = np.where(np.diff(is_trial) == -1)[0]
    trial_length = idx_trial_end - idx_trial_start

    # we add one to compensate for insertion of leading -1 value
    y = trigger[idx_trial_start + 1]

    # note: np.diff discards first index, but we append a leading -1 to all trial_labels,
    # these effects negate each other
    return idx_trial_start, y, min(trial_length)


if __name__ == '__main__':
    # quick test case
    idx_trial_start, y, trial_length = split_trial(
        trigger=[2, 2, 2, -1, -1, 10, 10, 10, -1, 1, 1, 1, 1])
    np.testing.assert_allclose(idx_trial_start, [0, 5, 9])
    np.testing.assert_allclose(y, [2, 10, 1])
    assert trial_length == 3
