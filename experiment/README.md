To run the experiment:
- First run stage1.py. 
    - This will collect 10 images from the participant, 
        then ask them to repeat each poses 3 times in a random order, 
        collecting sufficient data to be used to classify the hand poses.
    - The Participant ID will be recorded in participants.txt. You will need this value for the remaining stages. 
    - Results of stage 1 are recorded in the calibration-results folder, and images are stored in ./images under a folder with the appropriate participant ID. 
- Next, open stage2.ipynb.Find the Participant ID for your trial from 
    - At the top of the notebook, change the participant ID to the current participant.
    - Run all cells. 
    - At the bottom of the notebook, the ordered list of most identifiable hand poses will be outputted.
- Copy the Participant ID (PARTICIPANT_ID) and the list of best combinations (POSES) into the appropriate fields of experiment_values.txt
- Determine how many poses you would like to use for the trials 
    i.e. how many poses the participant should be prompted with
    Enter this value as NUM_POSES field of the experiment_values.txt
- Run stage3a_prompt.py to run the trial in which the user will be prompted via images of the hand poses.
- Run stage3b_keyboard.py to run the trial in which the user will be asked to type a phrase, prompted by letters and the poses associated with a colored keyboard. 