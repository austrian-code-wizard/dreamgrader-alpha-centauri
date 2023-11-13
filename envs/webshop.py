import time
from string import ascii_letters, digits, punctuation

import torch
import numpy as np
from gym import spaces
import gymnasium as gym

import render
import meta_exploration
from envs.miniwob.constants import NUM_INSTANCES
from web_agent_site.envs.web_agent_dream_env import WebAgentDreamEnv
from web_agent_site.envs.web_agent_dream_env_dom import WebAgentDreamDOMEnv


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
            self.action_space = spaces.Discrete(WebShopMetaEnv.NUM_ITEMS)

    def _instruction_observation_space(self):
        return gym.spaces.Box(np.array([0]), np.array([1]), dtype=int)

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
            state, _, _, info = self.env.step([3] * len(action))
            return state, reward, done, info
        # Bypass parent class
        return self.env.step(action)


class WebShopMetaEnv(meta_exploration.MetaExplorationEnv):
    MAX_STEPS = None
    NUM_TRAIN = None
    NUM_TEST = None
    WINDOW_WIDTH = None
    WINDOW_HEIGHT = None
    NUM_ITEMS = None
    QUANTIZE = None
    SCROLL_AMOUNT = None
    SCROLL_TIME = None
    NUM_ACTIONS = None
    ITER = None
    NUM_DEMOS = None

    def __init__(self, env_id, _):
        assert NUM_INSTANCES == 1, "Only supporting 1 concurrent env with webshop at the moment"
        super().__init__(env_id, WebshopObservation)
        self._steps = 0
        
        self._env = WebAgentDreamDOMEnv(window_height=self.WINDOW_HEIGHT, window_width=self.WINDOW_WIDTH, scroll_amount=self.SCROLL_AMOUNT, scroll_time=self.SCROLL_TIME)

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Text(min_length=0, max_length=100000, charset=ascii_letters + digits + punctuation),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([self.NUM_ITEMS]), # TODO: check if this actually matters
                dtype=int)
        })
        # self._env.reset() #TODO: this is done explicitly in training loop
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.exploitation = False

    @classmethod
    def instruction_wrapper(cls):
        return InstructionWrapper

    @classmethod
    def load_config(cls, config=None):
        cls.MAX_STEPS = config.get("max_steps", 10)
        cls.NUM_TRAIN = config.get("num_train", 1000000)
        cls.NUM_TEST = config.get("num_test", 1000)
        cls.WINDOW_WIDTH = config.get("window_width", 960)
        cls.WINDOW_HEIGHT = config.get("window_height", 540)
        cls.NUM_ITEMS = config.get("num_items", 15)
        cls.QUANTIZE = config.get("quantize", None)
        cls.SCROLL_AMOUNT = config.get("scroll_amount", 180)
        cls.SCROLL_TIME = config.get("scroll_time", 150)
        cls.NUM_ACTIONS = config.get("num_actions", 4)
        cls.USE_SCREENSHOT = config.get("use_screenshot", False)
        cls.NUM_DEMOS = config.get("num_demos", 0)


    @classmethod
    def env_ids(cls):
        return list(range(cls.NUM_TRAIN)), list(range(cls.NUM_TRAIN, cls.NUM_TRAIN + cls.NUM_TEST))

    @property
    def env_id(self):
        return self._correct_answers
    
    @property
    def questions(self):
        return self._questions

    @classmethod
    def set_iter(cls, iter):
        cls.ITER = iter

    def is_demo(self):
        return True if self.ITER is not None and self.ITER < self.NUM_DEMOS else False

    def get_demo(self):
        if self.exploitation:
            return torch.tensor(self.env_id).to("cpu") # Return correct answer
        elif self._steps == 0:
            return [1] # Search items on initial step
        return [0] # End episode on results page

    def _step(self, action):
        # print(f"Action: {action}")
        assert len(action) == 1, "Only supporting 1 concurrent env with webshop at the moment"
        start = time.time()
        state, reward, done, info = self._env.step(action[0])
        # print(f"Time to step: {time.time() - start}")
        self._steps += 1
        done = done if self._steps < type(self).MAX_STEPS else True
        state = state["html"]
        return [{
            "observation": state,
            "question": self._questions[0]
        }], [reward], [done], [info]

    def _reset(self):
        if self.exploitation:
            return [{"observation": "", "question": ""}]
        self._steps = 0
        start = time.time()
        state, _ = self._env.reset(seed=self._env_id[0])
        # print(f"Time to reset: {time.time() - start}")
        self._questions = [state["instruction_text"]]

        # self._correct_answers = [[p["index"] for p in best_products]]
        self._correct_answers = [state["best_products"][0]["index"]]
        state = state["html"]
        return [{
            "observation": state,
            "question": self._questions[0]
        }]

    def render(self, mode=None):
        imgs = []
        img = self._env.screenshot
        img = render.Render(img)
        img.write_text("Underlying env ID: {}".format(self._env_id[0]))
        img.write_text(f"Q: {self.questions[0]}")
        img.write_text(f"A: {self.env_id[0]}")
        imgs.append(img)
        return imgs
    
    @property
    def underlying_env_id(self):
        return self._env_id

    def set_underlying_env_id(self, id):
        self._env_id = id
        self._reset()


class WebshopObservation:
    def __init__(self, observation):
        self._observation = observation["observation"]
        self._question = observation["question"]

    @property
    def is_cuda(self):
        return False

    @property
    def observation(self):
        return self._observation

    @property
    def question(self):
        return self._question

    def cpu(self):
        # Hacky way to accomodate cpu/cuda switching in observation buffer
        return self

    def pin_memory(self):
        return self

    def cuda(self, **kwargs):
        return self