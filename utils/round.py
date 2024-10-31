import numpy as np


def banker_round(x):

    half = 0.5
    one = 1
    two = 2

    rounded = np.ceil(x - half)
    bankers_mask = one - (np.ceil(x + half) - np.floor(x + half))
    non_even = np.abs(np.mod(rounded, two))
    return rounded + (bankers_mask * non_even)
