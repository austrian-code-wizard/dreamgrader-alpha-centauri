import os
import re
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
from envs.miniwob.constants import TASK_HEIGHT, TASK_WIDTH, ASCII_CHARSET, TEXT_MAX_LENGTH, SYMBOLS, SIZES, PEOPLE_NAMES, LOREM_WORDS


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

# Features
FEATURES = {
    "symbol": {
        "values": SYMBOLS,
        "extractor": lambda email: SYMBOLS.index(email["symbol"])
    },
    "body_size": {
        "values": SIZES,
        "extractor": lambda email: SIZES.index(email["font_size"])
    },
    "sender_name": {
        "values": PEOPLE_NAMES,
        "extractor": lambda email: PEOPLE_NAMES.index(email["raw_name"])
    },
    "subject_first_word": {
        "values": LOREM_WORDS,
        "extractor": lambda email: LOREM_WORDS.index(re.sub(r'[^a-z]', '', email["subject"].split()[0].lower()))
    },
    "body_last_word": {
        "values": LOREM_WORDS,
        "extractor": lambda email: LOREM_WORDS.index(re.sub(r'[^a-z]', '', email["body"].split()[-1].lower()))
    },
    "index": {
        "values": [str(i) for i in range(NUM_EMAILS)],
        "extractor": lambda email: email["idx"]
    }
}

"""
Demos are formatted as follows:

current state: {
    desired state: action
}
"""
DEMOS = {
    INBOX_UP: {
        INBOX_UP: SCROLL_UP,
        INBOX_MID: SCROLL_DOWN,
        INBOX_DOWN: SCROLL_DOWN,
        EMAIL_1: CLICK_UP,
        EMAIL_2: CLICK_MID,
        EMAIL_3: CLICK_DOWN,
        EMAIL_4: SCROLL_DOWN,
        EMAIL_5: SCROLL_DOWN,
        EMAIL_6: SCROLL_DOWN,
        EMAIL_7: SCROLL_DOWN,
    },
    INBOX_MID: {
        INBOX_UP: SCROLL_UP,
        INBOX_DOWN: SCROLL_DOWN,
        EMAIL_1: SCROLL_UP,
        EMAIL_2: SCROLL_UP,
        EMAIL_3: CLICK_UP,
        EMAIL_4: CLICK_MID,
        EMAIL_5: CLICK_DOWN,
        EMAIL_6: SCROLL_DOWN,
        EMAIL_7: SCROLL_DOWN
    },
    INBOX_DOWN: {
        INBOX_UP: SCROLL_UP,
        INBOX_MID: SCROLL_UP,
        INBOX_DOWN: SCROLL_DOWN,
        EMAIL_1: SCROLL_UP,
        EMAIL_2: SCROLL_UP,
        EMAIL_3: SCROLL_UP,
        EMAIL_4: SCROLL_UP,
        EMAIL_5: CLICK_UP,
        EMAIL_6: CLICK_MID,
        EMAIL_7: CLICK_DOWN
    }
}

DEMOS_WITH_BACK = {
    INBOX_UP: {
        INBOX_UP: SCROLL_UP,
        INBOX_MID: SCROLL_DOWN,
        INBOX_DOWN: SCROLL_DOWN,
        EMAIL_1: CLICK_UP,
        EMAIL_2: CLICK_MID,
        EMAIL_3: CLICK_DOWN,
        EMAIL_4: SCROLL_DOWN,
        EMAIL_5: SCROLL_DOWN,
        EMAIL_6: SCROLL_DOWN,
        EMAIL_7: SCROLL_DOWN,
    },
    INBOX_MID: {
        INBOX_UP: SCROLL_UP,
        INBOX_DOWN: SCROLL_DOWN,
        EMAIL_1: SCROLL_UP,
        EMAIL_2: SCROLL_UP,
        EMAIL_3: CLICK_UP,
        EMAIL_4: CLICK_MID,
        EMAIL_5: CLICK_DOWN,
        EMAIL_6: SCROLL_DOWN,
        EMAIL_7: SCROLL_DOWN
    },
    INBOX_DOWN: {
        INBOX_UP: SCROLL_UP,
        INBOX_MID: SCROLL_UP,
        INBOX_DOWN: SCROLL_DOWN,
        EMAIL_1: SCROLL_UP,
        EMAIL_2: SCROLL_UP,
        EMAIL_3: SCROLL_UP,
        EMAIL_4: SCROLL_UP,
        EMAIL_5: CLICK_UP,
        EMAIL_6: CLICK_MID,
        EMAIL_7: CLICK_DOWN
    },
    EMAIL_1: {
        INBOX_UP: BACK,
        INBOX_MID: BACK,
        INBOX_DOWN: BACK,
        EMAIL_2: BACK,
        EMAIL_3: BACK,
        EMAIL_4: BACK,
        EMAIL_5: BACK,
        EMAIL_6: BACK,
        EMAIL_7: BACK
    },
    EMAIL_2: {
        INBOX_UP: BACK,
        INBOX_MID: BACK,
        INBOX_DOWN: BACK,
        EMAIL_1: BACK,
        EMAIL_3: BACK,
        EMAIL_4: BACK,
        EMAIL_5: BACK,
        EMAIL_6: BACK,
        EMAIL_7: BACK
    },
    EMAIL_3: {
        INBOX_UP: BACK,
        INBOX_MID: BACK,
        INBOX_DOWN: BACK,
        EMAIL_1: BACK,
        EMAIL_2: BACK,
        EMAIL_4: BACK,
        EMAIL_5: BACK,
        EMAIL_6: BACK,
        EMAIL_7: BACK
    },
    EMAIL_4: {
        INBOX_UP: BACK,
        INBOX_MID: BACK,
        INBOX_DOWN: BACK,
        EMAIL_1: BACK,
        EMAIL_2: BACK,
        EMAIL_3: BACK,
        EMAIL_5: BACK,
        EMAIL_6: BACK,
        EMAIL_7: BACK
    },
    EMAIL_5: {
        INBOX_UP: BACK,
        INBOX_MID: BACK,
        INBOX_DOWN: BACK,
        EMAIL_1: BACK,
        EMAIL_2: BACK,
        EMAIL_3: BACK,
        EMAIL_4: BACK,
        EMAIL_6: BACK,
        EMAIL_7: BACK
    },
    EMAIL_6: {
        INBOX_UP: BACK,
        INBOX_MID: BACK,
        INBOX_DOWN: BACK,
        EMAIL_1: BACK,
        EMAIL_2: BACK,
        EMAIL_3: BACK,
        EMAIL_4: BACK,
        EMAIL_5: BACK,
        EMAIL_7: BACK
    },
    EMAIL_7: {
        INBOX_UP: BACK,
        INBOX_MID: BACK,
        INBOX_DOWN: BACK,
        EMAIL_1: BACK,
        EMAIL_2: BACK,
        EMAIL_3: BACK,
        EMAIL_4: BACK,
        EMAIL_5: BACK,
        EMAIL_6: BACK
    }
}

# Describes which screenshot state to show after a given action
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
            unique_feature_values = []
            for f in FakeInboxScrollMulticlassMetaEnv.TARGET_FEATURES:
                if FEATURES[f]["values"] not in unique_feature_values:
                    unique_feature_values.append(FEATURES[f]["values"])
            self.action_space = spaces.Discrete(sum(len(f) for f in unique_feature_values))

    def _instruction_observation_space(self):
        return gym.spaces.Box(np.array([0]), np.array([1]), dtype=np.int)

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


class FakeInboxScrollMulticlassMetaEnv(meta_exploration.MetaExplorationEnv):
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
    QUERY_FEATURES = None
    TARGET_FEATURES = None

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
        self.states = [{
            "current_state": INBOX_UP,
            "target_state": None,
            "query_feature": None,
            "query_feature_value": None,
            "target_feature": None,
            "target_feature_value": None,
            "question": None,
            "label": None,
            "inbox_id": None,
            "correct_answer": None
        } for _ in range(len(env_id))]

        obs_space = {}

        # Question space is [query feature index, query feature value, target feature index]
        obs_space['question'] = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([
            len(type(self).QUERY_FEATURES) - 1,
            max([len(FEATURES[f]["values"]) for f in type(self).QUERY_FEATURES]) - 1,
            len(type(self).TARGET_FEATURES) - 1
        ]), dtype=np.int)

        if type(self).USE_SCREENSHOTS:
            obs_space['screenshot'] = gym.spaces.Box(low=0, high=255, shape=(TASK_HEIGHT, TASK_WIDTH, 1), dtype=np.uint8)
        if type(self).USE_DOMS:
            obs_space['dom'] = gym.spaces.Text(min_length=0, max_length=TEXT_MAX_LENGTH, charset=ASCII_CHARSET)
        if type(self).USE_SCROLL_STATE:
            obs_space['scroll_state'] = gym.spaces.Box(low=np.array([0]), high=np.array([2]), dtype=np.int)

        unique_feature_values = []
        for f in type(self).TARGET_FEATURES:
            if FEATURES[f]["values"] not in unique_feature_values:
                unique_feature_values.append(FEATURES[f]["values"])
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Sequence(
                gym.spaces.Dict(obs_space)
            ),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([sum(len(f) for f in unique_feature_values) - 1]),
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

        cls.QUERY_FEATURES = config.get("query_features", [])
        assert len(cls.QUERY_FEATURES) > 0, "Must specify query features"
        assert all(f in FEATURES for f in cls.QUERY_FEATURES), "Invalid query feature name"
        cls.TARGET_FEATURES = config.get("target_features")
        assert len(cls.TARGET_FEATURES) > 0, "Must specify target features"
        assert all(f in FEATURES for f in cls.TARGET_FEATURES), "Invalid target feature name"


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
            state = self.states[i]
            if state["current_state"] in demos and state["target_state"] in demos[state["current_state"]]:
                a = demos[state["current_state"]][state["target_state"]]
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
        return [state["correct_answer"] for state in self.states]

    @property
    def questions(self):
        return [state["question"] for state in self.states]


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
        state = self.states[idx]
        question = np.array([state["query_feature"], state["query_feature_value"], state["target_feature"]])

        return {
            "screenshot": self._get_screenshot(state["inbox_id"], state["current_state"]) if type(self).USE_SCREENSHOTS else [0],
            "question": question,
            "dom": self._get_dom(state["inbox_id"], state["current_state"]) if type(self).USE_DOMS else None,
            "scroll_state": self._get_scroll_state(state["inbox_id"], state["current_state"]) if type(self).USE_SCROLL_STATE else None
        }


    def _step(self, action):
        if not self.exploitation:
                for state, a in zip(self.states, action):
                    state["current_state"] = self._get_next_state(state["current_state"], a)

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
        for s in self.states:
            s["current_state"] = INBOX_UP
        obs = [self._get_state(i) for i in range(len(self.env_id))]
        return obs

    def render(self, mode=None):
        imgs = []
        for i in range(len(self.env_id)):
            state = self.states[i]
            img = ToPILImage()(self._get_screenshot(state["inbox_id"], state["current_state"]).permute(2, 0, 1))
            img = render.Render(img)
            img.write_text("Underlying env ID: {}".format(self._env_id[i]))
            question = self.states[i]["question"]
            img.write_text(f"Q: {question}")
            label = self.states[i]["label"]
            img.write_text(f"A: {label}")
            imgs.append(img)
        return imgs
    
    @property
    def underlying_env_id(self):
        return self._env_id
    

    def _configure_env(self, id, env):

        # We have N inboxes where each inbox has M emails with Q queries with T targets
        ids_per_inbox = NUM_EMAILS * len(type(self).QUERY_FEATURES) * len(type(self).TARGET_FEATURES)
        env["inbox_id"] = id // ids_per_inbox

        # Fetch emails as JSON
        emails = json.loads(self.DF.iloc[env["inbox_id"], 1])

        # To get the ID of the email, we know that we can mod it by the number of emails per inbox,
        # then divide by the number of queries times targets per email
        ids_per_email = len(type(self).QUERY_FEATURES) * len(type(self).TARGET_FEATURES)
        env["target_email"] = (id % ids_per_inbox) // ids_per_email

        # To get the query feature, we can mode by the number of elements per email and divide by
        # the nuber of elements per query
        ids_per_query = len(type(self).TARGET_FEATURES)
        env["query_feature"] = (id % ids_per_email) // ids_per_query

        # To get the target feature, we mod by the number of elements per query
        env["target_feature"] = id % ids_per_query

        # Set the desired final state of exploration
        env["target_state"] = EMAIL_1 + env["target_email"] if type(self).TARGET_FEATURES[env["target_feature"]] in ["body_size", "body_last_word"] else env["target_email"] // 3

        # Set the query feature value of the target email
        env["query_feature_value"] = FEATURES[type(self).QUERY_FEATURES[env["query_feature"]]]["extractor"](emails[env["target_email"]])

        # Set the value of the target feature
        env["target_feature_value"] = FEATURES[type(self).TARGET_FEATURES[env["target_feature"]]]["extractor"](emails[env["target_email"]])

        # Env set the question in human language
        env["question"] = f"What is the {type(self).TARGET_FEATURES[env['target_feature']]} of the email where {type(self).QUERY_FEATURES[env['query_feature']]} is {FEATURES[type(self).QUERY_FEATURES[env['query_feature']]]['values'][env['query_feature_value']]}?"

        # Set the answer to the question in human language
        env["label"] = FEATURES[type(self).TARGET_FEATURES[env['target_feature']]]['values'][env['target_feature_value']]

        # Set the correct answer
        unique_feature_values = []
        for f in type(self).TARGET_FEATURES[:env["target_feature"]]:
            if FEATURES[f]["values"] not in unique_feature_values and FEATURES[f]["values"] != FEATURES[type(self).TARGET_FEATURES[env["target_feature"]]]["values"]:
                unique_feature_values.append(FEATURES[f]["values"])
        env["correct_answer"] = sum(
            len(f) for f in unique_feature_values
        ) + env["target_feature_value"]

        return env


    def set_underlying_env_id(self, id):
        self._env_id = id
        self.states = [self._configure_env(env_id, env_info) for env_id, env_info in zip(id, self.states)]
