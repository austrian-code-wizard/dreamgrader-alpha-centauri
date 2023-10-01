import os
import ast
import csv
import json
import torch
import itertools
import collections

import pandas as pd
import torch
from torchvision.io import read_image
import gymnasium as gym
import numpy as np
from PIL import Image
from gym import spaces

import render
import meta_exploration
from envs.miniwob.inbox import EmailInboxObservation
from envs.miniwob.constants import NUM_INSTANCES
from envs.miniwob.fake_inbox_scroll import FakeInboxScrollMetaEnv, NUM_EMAILS, SIZES, EMAIL_1, TEXT_MAX_LENGTH, ASCII_CHARSET, INBOX_UP


class FakeInboxScrollVectorizedMetaEnv(FakeInboxScrollMetaEnv):
    MAX_STEPS = None
    NUM_TRAIN = None
    NUM_TEST = None
    DATA_DIR = None
    USE_SYMBOL_QUERIES = None
    USE_BACK_ACTION = None
    ENV_ID_SCHEDULE = None
    NUM_DEMOS = None

    ITER = None
    NUM_ACTIONS_WITH_BACK = 6
    NUM_ACTIONS_NO_BACK = 5
    DEFAULT_DATA_DIR = "/scr-ssd/moritzst/data_envs_scroll"


    def __init__(self, env_id, _):
        super().__init__(env_id, None)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Sequence(
                gym.spaces.Dict({
                    'screenshot': gym.spaces.Box(low=np.array([0] * 6), high=np.array([1, 3, 7, 3, 6, 2]), dtype=np.int),
                    'question': gym.spaces.Text(min_length=0, max_length=TEXT_MAX_LENGTH, charset=ASCII_CHARSET)
                })
            ),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([type(self).NUM_TRAIN + type(self).NUM_TEST + 1]),
                dtype=np.int)
        })


    def _get_state(self, idx: int):
        """
        We are representing the state as a vector with the following dimensions:
        [
            0-1 # If in inbox or email view
            0-3 # If in inbox view, which scroll position are we in
            0-7 # If in email view, which email is selected
            0-3 # If in email view, which size are we seeing
            0-6 # Which email are we asking about
            0-2 # Which size are we asking about
        ]
        """
        vector_state = np.zeros((6))
        
        # Set if in inbox or email view
        if self.cur_states[idx] >= EMAIL_1:
            vector_state[0] = 1

        # Set if in inbox view, which scroll position are we in
        if self.cur_states[idx] < EMAIL_1:
            vector_state[1] = self.cur_states[idx] + 1

        # Set if in email view, which email is selected
        if self.cur_states[idx] >= EMAIL_1:
            vector_state[2] = self.cur_states[idx] - EMAIL_1 + 1

        # Set info on current size
        if self.cur_states[idx] >= EMAIL_1:
            email_index = self.cur_states[idx] - EMAIL_1
            emails = json.loads(self.DF.iloc[self._env_numbers[idx], 1])
            email_size = SIZES.index(emails[email_index]["font_size"])
            vector_state[3] = email_size + 1

        # Set which email we are asking about
        vector_state[4] = self._email_indices[idx]

        # Set which size we are asking about
        vector_state[5] = self._email_sizes[idx]

        return {
            "screenshot": vector_state,
            "question": self._questions[idx],
            "dom": "None"
        }

