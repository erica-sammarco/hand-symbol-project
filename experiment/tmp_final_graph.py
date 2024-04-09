import pathlib
import re

from train import get_trained_classifier, process_file_data
import pandas as pd
import matplotlib.pyplot as plt



folder_key = pathlib.Path(__file__).parent / 'keyboard-results'

participant_id = '0u6vBd'

file_dict = dict()
for file in folder_key.glob(f'{participant_id}*.txt'):
    s = open(file, 'r').read()
    match = re.search('\n POSES:.*\n', s)
    if match is None:
        raise Exception('file doesnt contain poses')
    pose_list = [int(x) for x in
                 match.group().split(':')[-1].strip()[1:-1].split(', ')]
    num_pose = len(pose_list)
    file_dict[num_pose] = file

cls = get_trained_classifier(participant_id, n_neighbors=1)

df_list = list()
for num_pose, file in file_dict.items():
    df = pd.DataFrame()
    x, y, trial_length = process_file_data(file)
    df['trial_length'] = trial_length
    df['num_pose'] = num_pose
    df['target_pose'] = y
    # df['participant_id'] = participant_id
    df_list.append(df)

df = pd.concat(df_list)
df['time (sec)'] = df['trial_length'] / 30

df_mean_per_num_pose = df.groupby('num_pose').mean()

import seaborn as sns

sns.scatterplot(data=df_mean_per_num_pose, y='time (sec)', x='num_pose')
plt.show()

