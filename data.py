import pandas as pd
import numpy as np
import os
import pickle
import torch

import json
import itertools
import re
from collections import defaultdict

import config

#DEVICE = config.DEVICE
MAX_LENGTH = 18

def create_opus_pairs(input_path):
    data = []

    """Loads Opusparcus data, get the sentence pairs"""
    with open(input_path) as file:
        for line in file:
            input_text = line
            #re.sub(r"en-.+ ","",input_text)
            #print(input_text)

            data.append(input_text.split("\t"))

    opus_pairs = []

    opus_pairs = np.array([pair[1:3] for pair in data])

    return opus_pairs

#op = create_opus_pairs("data/opusparcus/en/test/en-test.txt")

