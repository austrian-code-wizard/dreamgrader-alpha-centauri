import os
os.environ["TRANSFORMERS_CACHE"] = "/iris/u/moritzst/.cache"

import ast
import csv
import json
import time
import torch
import itertools
import collections
from typing import Literal

import torch
import numpy as np
from PIL import Image
from gym import spaces
import gymnasium as gym
from transformers import BitsAndBytesConfig
from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor

import render
import meta_exploration
from envs.miniwob.constants import NUM_INSTANCES
from web_agent_site.envs.web_agent_dream_env import WebAgentDreamEnv


def scale_image(img: Image, desired_width = 1920, desired_height=1080):
  ratio = img.width / img.height
  desired_ratio = desired_width / desired_height
  if ratio > desired_ratio:
    # width is larger than desired width
    new_width = desired_width
    new_height = int(desired_width / ratio)
  else:
    new_width = int(desired_height * ratio)
    new_height = desired_height
  return img.resize((new_width, new_height))


class WebAgentDreamEnvFuyu(WebAgentDreamEnv):
    pretrained_path = "adept/fuyu-8b"

    model = None
    processor = None

    def __init__(self, *args, quantize: Literal[None, "4b", "8b"] = None, **kwargs) -> None:

        # load model, tokenizer, and processor
        if WebAgentDreamEnvFuyu.processor is None:
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
            image_processor = FuyuImageProcessor(target_height=WebShopMetaEnv.WINDOW_HEIGHT, target_width=WebShopMetaEnv.WINDOW_WIDTH)
            WebAgentDreamEnvFuyu.processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)

        if WebAgentDreamEnvFuyu.model is None:
            if quantize == "4b":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                )
            elif quantize == "8b":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                quantization_config = None

            WebAgentDreamEnvFuyu.model = FuyuForCausalLM.from_pretrained(self.pretrained_path, quantization_config=quantization_config, device_map="cuda:0")
        
        super().__init__(*args, **kwargs)

    @property
    def state(self):
        state = super().state
        text = state["instruction_text"] + "\n" + "Actions: " + ", ".join(state["click_actions"]) + "\n"
        image = state["screenshot"]
        image = scale_image(image, desired_width=WebShopMetaEnv.WINDOW_WIDTH, desired_height=WebShopMetaEnv.WINDOW_HEIGHT)

        start = time.time()
        model_inputs = self.processor(text=text, images=[image], device="cuda:0")
        for k, v in model_inputs.items():
            model_inputs[k] = v.to("cuda:0")
        with torch.no_grad():
            output = self.model(**model_inputs, output_hidden_states=True)
            output = output.hidden_states[-1][0]
        print(f"Time to run model: {time.time() - start}")
        return output, state["instruction_text"], state["best_products"]


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


    def __init__(self, env_id, _):
        assert NUM_INSTANCES == 1, "Only supporting 1 concurrent env with webshop at the moment"
        super().__init__(env_id, lambda x: x)
        self._steps = 0
        
        # TODO: change back to original email inbox env once exp is done
        env = WebAgentDreamEnvFuyu(quantize=self.QUANTIZE, window_height=self.WINDOW_HEIGHT, window_width=self.WINDOW_WIDTH, scroll_amount=self.SCROLL_AMOUNT, scroll_time=self.SCROLL_TIME)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.WINDOW_HEIGHT, self.WINDOW_WIDTH)),
            "env_id": gym.spaces.Box(np.array([0]),
                np.array([3]), # TODO: check if this actually matters
                dtype=int)
        })
        self._env = env
        self._env.reset()
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


    @classmethod
    def env_ids(cls):
        return list(range(cls.NUM_TRAIN)), list(range(cls.NUM_TRAIN, cls.NUM_TRAIN + cls.NUM_TEST))

    @property
    def env_id(self):
        return self._correct_answers
    
    @property
    def questions(self):
        return self._questions

    def _step(self, action):
        assert len(action) == 1, "Only supporting 1 concurrent env with webshop at the moment"
        start = time.time()
        (state, _, __), reward, done, info = self._env.step(action[0])
        print(f"Time to step: {time.time() - start}")
        self._steps += 1
        done = done if self._steps < type(self).MAX_STEPS else [True]*len(action)
        return [state], [reward], [done], [info]

    def _reset(self):
        # old hack but messes up evaluation of correct answer
        self._steps = 0
        start = time.time()
        (obs, question, best_products), _ = self._env.reset(seed=self._env_id[0])
        print(f"Time to reset: {time.time() - start}")
        self._questions = [question]

        # self._correct_answers = [[p["index"] for p in best_products]]
        self._correct_answers = [best_products[0]["index"]]
        return [obs]

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
