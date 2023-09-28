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
from envs.miniwob.fake_inbox_scroll import FakeInboxScrollMetaEnv, NUM_EMAILS, SIZES, EMAIL_1, TEXT_MAX_LENGTH, ASCII_CHARSET


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
                    'screenshot': gym.spaces.Box(low=0, high=1, shape=(NUM_EMAILS * len(SIZES),), dtype=np.uint8),
                    'question': gym.spaces.Text(min_length=0, max_length=TEXT_MAX_LENGTH, charset=ASCII_CHARSET)
                })
            ),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([type(self).NUM_TRAIN + type(self).NUM_TEST + 1]),
                dtype=np.int)
        })
    

    def _step(self, action):
        if self.exploitation:
            states = [{
                "screenshot": np.zeros((NUM_EMAILS * len(SIZES))),
                "question": "None",
                "dom": "None"
            } for _ in range(NUM_INSTANCES)]
            reward = [0] * NUM_INSTANCES
            info = [None] * NUM_INSTANCES
            done = [True] * NUM_INSTANCES
        else:
            self.cur_states = [self._get_next_state(cur_state, a) for cur_state, a in zip(self.cur_states, action)]
            states = []
            for i, (idx, state) in enumerate(zip(self._env_numbers, self.cur_states)):
                vector_state = np.zeros((NUM_EMAILS * len(SIZES)))
                if self.cur_states[i] - EMAIL_1 >= 0:
                    email_index = self.cur_states[i] - EMAIL_1
                    emails = json.loads(self.df.iloc[self._env_numbers[i], 1])
                    email_size = SIZES.index(emails[email_index]["font_size"])
                    vector_state[email_index * len(SIZES) + email_size] = 1
                states.append({
                    "screenshot": vector_state,
                    "question": self._questions[i],
                    "dom": "None"
                })
            reward = [0] * NUM_INSTANCES
            info = [None] * NUM_INSTANCES
            done = [False] * NUM_INSTANCES
            self._steps += 1
            done = done if self._steps < type(self).MAX_STEPS else [True]*NUM_INSTANCES
        return states, reward, done, info

    def _reset(self):
        # old hack but messes up evaluation of correct answer
        self._steps = 0
        self.cur_states = [0 for _ in range(NUM_INSTANCES)]
        obs = [{
            "screenshot": np.zeros((NUM_EMAILS * len(SIZES))),
            "question": self._questions[i],
            "dom": "None"
        } for i, (idx, state) in enumerate(zip(self._env_numbers, self.cur_states))]
        return obs