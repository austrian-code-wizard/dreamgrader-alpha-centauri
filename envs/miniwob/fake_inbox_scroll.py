import os
import json

import torch
import numpy as np
import pandas as pd
from gym import spaces
import gymnasium as gym
from torchvision.io import read_image
from torchvision.transforms import ToPILImage

import render
import meta_exploration
from envs.miniwob.inbox import EmailInboxObservation
from envs.miniwob.constants import TASK_HEIGHT, TASK_WIDTH, ASCII_CHARSET, TEXT_MAX_LENGTH, SYMBOLS, SIZES


# Constants
NUM_EMAILS = 7

# Actions
SCROLL_DOWN = 0
SCROLL_UP = 1
CLICK_UP = 2
CLICK_MID = 3
CLICK_DOWN = 4
BACK = 5

# States
INBOX_UP = 0
INBOX_MID = 1
INBOX_DOWN = 2
EMAIL_1 = 3
EMAIL_2 = 4
EMAIL_3 = 5
EMAIL_4 = 6
EMAIL_5 = 7
EMAIL_6 = 8
EMAIL_7 = 9


DEMOS = {
    INBOX_UP: {
        0: CLICK_UP,
        1: CLICK_MID,
        2: CLICK_DOWN,
        3: SCROLL_DOWN,
        4: SCROLL_DOWN,
        5: SCROLL_DOWN,
        6: SCROLL_DOWN,
    },
    INBOX_MID: {
        0: SCROLL_UP,
        1: SCROLL_UP,
        2: CLICK_UP,
        3: CLICK_MID,
        4: CLICK_DOWN,
        5: SCROLL_DOWN,
        6: SCROLL_DOWN
    },
    INBOX_DOWN: {
        0: SCROLL_UP,
        1: SCROLL_UP,
        2: SCROLL_UP,
        3: SCROLL_UP,
        4: CLICK_UP,
        5: CLICK_MID,
        6: CLICK_DOWN
    }
}

# Formatted as current_state: desired_state: action
DEMOS_WITH_BACK = {
    INBOX_UP: {
        0: CLICK_UP,
        1: CLICK_MID,
        2: CLICK_DOWN,
        3: SCROLL_DOWN,
        4: SCROLL_DOWN,
        5: SCROLL_DOWN,
        6: SCROLL_DOWN,
    },
    INBOX_MID: {
        0: SCROLL_UP,
        1: SCROLL_UP,
        2: CLICK_UP,
        3: CLICK_MID,
        4: CLICK_DOWN,
        5: SCROLL_DOWN,
        6: SCROLL_DOWN
    },
    INBOX_DOWN: {
        0: SCROLL_UP,
        1: SCROLL_UP,
        2: SCROLL_UP,
        3: SCROLL_UP,
        4: CLICK_UP,
        5: CLICK_MID,
        6: CLICK_DOWN
    },
    EMAIL_1: {
        1: BACK,
        2: BACK,
        3: BACK,
        4: BACK,
        5: BACK,
        6: BACK
    },
    EMAIL_2: {
        0: BACK,
        2: BACK,
        3: BACK,
        4: BACK,
        5: BACK,
        6: BACK
    },
    EMAIL_3: {
        0: BACK,
        1: BACK,
        3: BACK,
        4: BACK,
        5: BACK,
        6: BACK
    },
    EMAIL_4: {
        0: BACK,
        1: BACK,
        2: BACK,
        4: BACK,
        5: BACK,
        6: BACK
    },
    EMAIL_5: {
        0: BACK,
        1: BACK,
        2: BACK,
        3: BACK,
        5: BACK,
        6: BACK
    },
    EMAIL_6: {
        0: BACK,
        1: BACK,
        2: BACK,
        3: BACK,
        4: BACK,
        6: BACK
    },
    EMAIL_7: {
        0: BACK,
        1: BACK,
        2: BACK,
        3: BACK,
        4: BACK,
        5: BACK
    }
}


TRANSITIONS = {
    INBOX_UP: {
        SCROLL_DOWN: INBOX_MID,
        CLICK_UP: EMAIL_1,
        CLICK_MID: EMAIL_2,
        CLICK_DOWN: EMAIL_3
    },
    INBOX_MID: {
        SCROLL_DOWN: INBOX_DOWN,
        SCROLL_UP: INBOX_UP,
        CLICK_UP: EMAIL_3,
        CLICK_MID: EMAIL_4,
        CLICK_DOWN: EMAIL_5
    },
    INBOX_DOWN: {
        SCROLL_UP: INBOX_MID,
        CLICK_UP: EMAIL_5,
        CLICK_MID: EMAIL_6,
        CLICK_DOWN: EMAIL_7
    },
    EMAIL_1: {
        BACK: INBOX_UP
    },
    EMAIL_2: {
        BACK: INBOX_UP
    },
    EMAIL_3: {
        BACK: INBOX_UP
    },
    EMAIL_4: {
        BACK: INBOX_MID
    },
    EMAIL_5: {
        BACK: INBOX_MID
    },
    EMAIL_6: {
        BACK: INBOX_DOWN
    },
    EMAIL_7: {
        BACK: INBOX_DOWN
    }
}


class InstructionWrapper(meta_exploration.InstructionWrapper):
    def __init__(self, env, exploration_trajectory, seed=None,
                 test=False, first_episode_no_instruction=False,
                 first_episode_no_optimization=False,
                 fixed_instructions=False, exploitation=False):
        super().__init__(
            env,
            exploration_trajectory,
            seed=seed, test=test,
            first_episode_no_instruction=first_episode_no_instruction,
            first_episode_no_optimization=first_episode_no_optimization,
            fixed_instructions=fixed_instructions)
        self._exploitation = exploitation
        self.env.exploitation = exploitation
        if exploitation:
            self.action_space = spaces.Discrete(3)

    def _instruction_observation_space(self):
        return gym.spaces.Box(np.array([0]), np.array([2]), dtype=np.int)

    def _reward(self, instruction_state, action, original_reward):
        return original_reward, False

    def _generate_instructions(self, test=False):
        return np.array([0]) # dummy unused instruction

    def step(self, action):
        if self._exploitation:
            done = [True] * len(action)
            reward = []
            for a, label in zip(action, self.env_id):
                reward.append(1 if (a == label).item() else -0.1)
            # Take dummy action, since existing action may be out of
            # bounds
            # Bypass parent class
            state, _, _, info = self.env.step([0] * len(action))
            return state, reward, done, info
        # Bypass parent class
        return self.env.step(action)


class FakeInboxScrollMetaEnv(meta_exploration.MetaExplorationEnv):
    MAX_STEPS = None
    NUM_TRAIN = None
    NUM_TEST = None
    DATA_DIR = None
    USE_SYMBOL_QUERIES = None
    USE_BACK_ACTION = None
    ENV_ID_SCHEDULE = None
    NUM_DEMOS = None
    USE_CACHE = None
    USE_SCREENSHOTS = None
    USE_DOMS = None
    USE_SCROLL_STATE = None
    USE_CLASSIFICATION = None

    ITER = None
    SCREENSHOT_CACHE = {}
    DOM_CACHE = {}
    NUM_ACTIONS_WITH_BACK = 6
    NUM_ACTIONS_NO_BACK = 5
    DEFAULT_DATA_DIR = "/scr-ssd/moritzst/data_envs_scroll"
    DF = None

    def __init__(self, env_id, _):
        super().__init__(env_id, EmailInboxObservation)
        self._steps = 0
        self.cur_states = [0 for _ in range(len(env_id))]
        self._env_numbers = None
        self._email_indices = None
        self._email_sizes = None

        obs_space = {}
        if not type(self).USE_CLASSIFICATION:
            obs_space['question'] = gym.spaces.Box(low=np.array([0] * 2), high=np.array([6, 2]), dtype=np.int)
        else:
            obs_space['question'] = gym.spaces.Box(low=np.array([0]), high=np.array([6]), dtype=np.int)

        if type(self).USE_SCREENSHOTS:
            obs_space['screenshot'] = gym.spaces.Box(low=0, high=255, shape=(TASK_HEIGHT, TASK_WIDTH, 1), dtype=np.uint8)
        if type(self).USE_DOMS:
            obs_space['dom'] = gym.spaces.Text(min_length=0, max_length=TEXT_MAX_LENGTH, charset=ASCII_CHARSET)
        if type(self).USE_SCROLL_STATE:
            obs_space['scroll_state'] = gym.spaces.Box(low=np.array([0]), high=np.array([2]), dtype=np.int)
 
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Sequence(
                gym.spaces.Dict(obs_space)
            ),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([1 if not type(self).USE_CLASSIFICATION else 2]),
                dtype=np.int)
        })
        self.action_space = gym.spaces.Discrete(type(self).NUM_ACTIONS_WITH_BACK if type(self).USE_BACK_ACTION else type(self).NUM_ACTIONS_NO_BACK)
        self.exploitation = False
        if type(self).DF is None:
            type(self).DF = pd.read_csv(os.path.abspath(f"{self.DATA_DIR}/inbox_samples.csv"))
        
        self.set_underlying_env_id(env_id)

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    @classmethod
    def load_config(cls, config: dict = None):
        cls.USE_SYMBOL_QUERIES = config.get("use_symbol_queries", False)
        cls.DATA_DIR = config.get("data_dir", cls.DEFAULT_DATA_DIR)
        cls.MAX_STEPS = config.get("max_steps", 4)
        cls.NUM_TRAIN = config.get("num_train", 100)
        cls.NUM_TEST = config.get("num_test", 10)
        cls.USE_BACK_ACTION = config.get("use_back_action", False)
        cls.ENV_ID_SCHEDULE = config.get("env_id_schedule", None)
        cls.NUM_DEMOS = config.get("num_demos", 0)
        cls.USE_CACHE = config.get("use_cache", False)
        cls.USE_SCREENSHOTS = config.get("use_screenshots", True)
        cls.USE_DOMS = config.get("use_doms", True)
        cls.USE_SCROLL_STATE = config.get("use_scroll_state", False)
        cls.USE_CLASSIFICATION = config.get("use_classification", False)


    @classmethod
    def set_iter(cls, iter):
        cls.ITER = iter


    def is_demo(self):
        return True if self.ITER is not None and self.ITER < self.NUM_DEMOS else False


    def get_demo(self):
        if self.exploitation:
            return torch.tensor(self.env_id).to("cpu")
        actions = []
        demos = DEMOS_WITH_BACK if type(self).USE_BACK_ACTION else DEMOS
        for i in range(len(self.env_id)):
            if self.cur_states[i] in demos and self._email_indices[i] in demos[self.cur_states[i]]:
                a = demos[self.cur_states[i]][self._email_indices[i]]
            else:
                a = np.random.randint(type(self).NUM_ACTIONS_WITH_BACK if type(self).USE_BACK_ACTION else type(self).NUM_ACTIONS_NO_BACK)
            actions.append(a)
        return actions


    @classmethod
    def env_ids(cls):
        train_ids = list(range(cls.NUM_TRAIN))
        if cls.ENV_ID_SCHEDULE is not None and cls.ITER is not None:
            """Schedules are of format:
            {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                100: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                200: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                300: "all"
                ...
            }
            """
            key = sorted([k for k in cls.ENV_ID_SCHEDULE.keys() if int(k) <= cls.ITER], key=lambda x: int(x))[-1]
            if cls.ENV_ID_SCHEDULE.get(key) != "all":
                train_ids = cls.ENV_ID_SCHEDULE.get(key)
                assert max(train_ids) < cls.NUM_TRAIN, "Cannot have train id outside of num train range"
        return train_ids, list(range(cls.NUM_TRAIN, cls.NUM_TRAIN + cls.NUM_TEST))

    @property
    def env_id(self):
        return self._labels

    @property
    def questions(self):
        return self._questions


    def _get_next_state(self, cur_state, action):
        if cur_state in TRANSITIONS and action in TRANSITIONS[cur_state]:
            return TRANSITIONS[cur_state][action]
        return cur_state


    def _get_screenshot(self, env_number, cur_state):
        if (env_number, cur_state) in type(self).SCREENSHOT_CACHE:
            img = type(self).SCREENSHOT_CACHE[(env_number, cur_state)]
        else:
            path = f"{self.DATA_DIR}/inboxes/{env_number}/{cur_state}.png"
            if not os.path.exists(path):
                suffix = '' if cur_state == 0 else f"-{cur_state - 1}"
                path = f"{self.DATA_DIR}/inboxes/{env_number}{suffix}.png"
            if not os.path.exists(path):
                raise Exception(f"Screenshot {path} does not exist")
            img = read_image(path).permute(1, 2, 0)
        
        if type(self).USE_CACHE and (env_number, cur_state) not in type(self).SCREENSHOT_CACHE:
            type(self).SCREENSHOT_CACHE[(env_number, cur_state)] = img

        if torch.cuda.is_available():
            img = img.cuda()
        return img
    

    def _get_dom(self, env_number, cur_state):
        if (env_number, cur_state) in type(self).DOM_CACHE:
            dom = type(self).DOM_CACHE[(env_number, cur_state)]
        else:
            path = f"{self.DATA_DIR}/doms/{env_number}/{cur_state}.txt"
            if not os.path.exists(path):
                suffix = '' if cur_state == 0 else f"-{cur_state - 1}"
                path = f"{self.DATA_DIR}/doms/{env_number}{suffix}.txt"
            if not os.path.exists(path):
                raise Exception(f"DOM {path} does not exist")
            with open(path, "r") as f:
                dom = f.read()

            if cur_state > INBOX_DOWN:
                emails = json.loads(self.DF.iloc[env_number, 1])
                size = emails[cur_state - EMAIL_1]["font_size"]
                dom = dom.replace("<div class=email-body>", f"<div class=email-body size={size}>")
        
        if type(self).USE_CACHE and (env_number, cur_state) not in type(self).DOM_CACHE:
            type(self).DOM_CACHE[(env_number, cur_state)] = dom

        return dom
    

    def _get_scroll_state(self, _, cur_state):
        vector_state = np.array([0])
        if cur_state == INBOX_MID:
            vector_state[0] = 1
        elif cur_state == INBOX_DOWN:
            vector_state[0] = 2
        return vector_state
    

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
        if not type(self).USE_CLASSIFICATION:
            vector_state = np.zeros((2))

            # Set which email we are asking about
            vector_state[0] = self._email_indices[idx]

            # Set which size we are asking about
            vector_state[1] = self._email_sizes[idx]
        else:
            vector_state = np.zeros((1))
            vector_state[0] = self._email_indices[idx]

        return {
            "screenshot": self._get_screenshot(self._env_numbers[idx], self.cur_states[idx]) if type(self).USE_SCREENSHOTS else None,
            "question": vector_state,
            "dom": self._get_dom(self._env_numbers[idx], self.cur_states[idx]) if type(self).USE_DOMS else None,
            "scroll_state": self._get_scroll_state(self._env_numbers[idx], self.cur_states[idx]) if type(self).USE_SCROLL_STATE else None
        }


    def _generate_question_and_label(self, env_number, email_number, email_size):
        emails = json.loads(self.DF.iloc[env_number, 1])
        font_size = SIZES[email_size]
        
        # Only activate if using symbol queries
        if FakeInboxScrollMetaEnv.USE_SYMBOL_QUERIES:
            symbol = SYMBOLS[email_number]
            symbol_order = [e["symbol"] for e in emails]
            email_number = symbol_order.index(symbol)
        
        if not type(self).USE_CLASSIFICATION:
            question = f"Is the {'1st' if email_number == 0 else '2nd' if email_number == 1 else '3rd' if email_number == 2 else f'{email_number+1}th'} email body {font_size}?"
            label = emails[email_number]["font_size"] == font_size
        else:
            question = f"What is the font size of the {'1st' if email_number == 0 else '2nd' if email_number == 1 else '3rd' if email_number == 2 else f'{email_number+1}th'} email body?"
            label = SIZES.index(emails[email_number]["font_size"])
        return question, label, email_number
    

    def _step(self, action):
        if not self.exploitation:
                self.cur_states = [self._get_next_state(cur_state, a) for cur_state, a in zip(self.cur_states, action)]
        states = [self._get_state(i) for i in range(len(action))]
        reward = [0] * len(action)
        info = [None] * len(action)
        done = [False] * len(action)
        self._steps += 1
        done = done if self._steps < type(self).MAX_STEPS else [True]*len(action)
        return states, reward, done, info

    def _reset(self):
        # old hack but messes up evaluation of correct answer
        self._steps = 0
        self.cur_states = [INBOX_UP for _ in range(len(self.env_id))]
        obs = [self._get_state(i) for i in range(len(self.env_id))]
        return obs

    def render(self, mode=None):
        imgs = []
        for i in range(len(self.env_id)):
            img = ToPILImage()(self._get_screenshot(self._env_numbers[i], self.cur_states[i]).permute(2, 0, 1))
            img = render.Render(img)
            img.write_text("Underlying env ID: {}".format(self._env_id[i]))
            question = self._questions[i]
            if type(self).USE_SYMBOL_QUERIES and not type(self).USE_CLASSIFICATION:
                emails = json.loads(self.DF.iloc[self._env_numbers[i], 1])
                symbol = emails[self._email_indices[i]]["symbol"]
                question = question.split()
                question.pop(2)
                question.insert(2, symbol)
                question.insert(3, f"({self._email_indices[i]+1})")
                question = " ".join(question)
            img.write_text(f"Q: {question}")
            if not type(self).USE_CLASSIFICATION:
                img.write_text(f"A: {self._labels[i]}")
            else:
                img.write_text(f"A: {SIZES[self._labels[i]]}")
            imgs.append(img)
        return imgs
    
    @property
    def underlying_env_id(self):
        return self._env_id

    def set_underlying_env_id(self, id):
        self._env_id = id
        
        if not type(self).USE_CLASSIFICATION:
            self._env_numbers = [idx // (NUM_EMAILS * len(SIZES)) for idx in id]
            self._email_indices = [(idx % (NUM_EMAILS * len(SIZES))) // len(SIZES) for idx in id]
            self._email_sizes = [(idx % (NUM_EMAILS * len(SIZES))) % len(SIZES) for idx in id]
        else:
            self._env_numbers = [idx // NUM_EMAILS for idx in id]
            self._email_indices = [idx % NUM_EMAILS for idx in id]
            self._email_sizes = [0 for _ in range(len(id))]
            
        question_labels = [self._generate_question_and_label(env_number, email_number, email_size) for env_number, email_number, email_size in zip(self._env_numbers, self._email_indices, self._email_sizes)]
        self._questions = [q for (q, _, _) in question_labels]
        self._labels = [l for (_, l, _) in question_labels]
        self._email_indices = [i for (_, _, i) in question_labels]
