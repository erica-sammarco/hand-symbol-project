from string import ascii_uppercase
import numpy as np

from experiment_values import USE_SEED, POSES

# What phrase will be used to test the user's typing?
PHRASE = "the quick brown fox jumps over the lazy dog"
# What should the seed start at?
SEED = 25

# Generate the sequence of poses for the above phrase
# parameter - phrase: Indicates if you want the full set of alphabetical values 
#                       (false if you just want the target values)
def generate(phrase: bool):
    seed = SEED

    sequence = [char.upper() for char in PHRASE if char != ' ']

    keyboard_values = []
    target_values = []

    for char in sequence:
        if USE_SEED:
            rng = np.random.default_rng(seed=seed)
        else: 
            rng = np.random.default_rng()
        
        user_symbol_idx = rng.integers(low=0, high=len(POSES), size=len(ascii_uppercase))
        user_symbols_all = [POSES[idx] for idx in user_symbol_idx]
        all_letters = dict(zip(ascii_uppercase, user_symbols_all))
        keyboard_values.append(all_letters)
        target_values.append(all_letters[char])
        seed += 1

    # print(target_values)

    # test_trg = "["
    # for idx, char in enumerate(sequence):
    #     test_trg = test_trg + str(keyboard_values[idx][char]) + ", "
    # test_trg += "]"

    # print(test_trg)

    if phrase:
        return keyboard_values
    else:
        return target_values
    
# generate(USE_SEED, True, POSES)