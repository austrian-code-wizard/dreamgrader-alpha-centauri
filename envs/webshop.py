import time
from string import ascii_letters, digits, punctuation

import re
import torch
import numpy as np
from gym import spaces
import gymnasium as gym
from transformers import MarkupLMProcessor, MarkupLMModel

import render
import meta_exploration
from envs.miniwob.constants import NUM_INSTANCES
from web_agent_site.envs.web_agent_dream_env import WebAgentDreamEnv
from web_agent_site.envs.web_agent_dream_env_dom import WebAgentDreamDOMEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MarkupLMWrapper(gym.Wrapper):
    model = None
    processor = None

    def __init__(self, env, path=None):
        super().__init__(env)
        if MarkupLMWrapper.processor is None:
            MarkupLMWrapper.processor = MarkupLMProcessor.from_pretrained(path)

        if MarkupLMWrapper.model is None:
            MarkupLMWrapper.model = MarkupLMModel.from_pretrained(path).to(device)
            MarkupLMWrapper.model.eval()

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(MarkupLMWrapper.model.config.hidden_size,),
            dtype=np.float32
        )

    def _clean_dom(self, dom):
        # Remove best product label
        dom = re.sub(r'<div id="best-products".*?</div>\n', "", dom, flags=re.DOTALL)

        # Remove instruction text
        dom = re.sub(r'<div id="instruction-text".*?</div>\n', "", dom, flags=re.DOTALL)

        # Remove head
        return re.sub(r'<head>.*?</head>\n', "", dom, flags=re.DOTALL)


    def _get_instruction(self, dom):
        return re.findall(r'<div id="instruction-text".*?>(.*?)</div>', dom, flags=re.DOTALL)[0]

    def _get_embeddings(self, dom, question):

        dom = self._clean_dom(dom)

        encoding = self.processor(html_strings=[dom], questions=[question], padding=True, max_length=512, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encoding)

        outputs = outputs["last_hidden_state"][0]
        
        # List of 1 x S x D tensors or 1 x D tensors (pooled)
        return outputs.unsqueeze(0)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state["html"] = self._get_embeddings(state["html"], state["instruction_text"])
        return state, reward, done, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        state["html"] = self._get_embeddings(state["html"], state["instruction_text"])
        return state, info


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
                reward.append(1 if any((a == l).item() for l in label) else -0.1)
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
    TEST = False
    NUM_DEMOS = None
    EMBED_STATES = None
    EMBED_PATH = None
    RETURN_N = None
    NUM_RANDOM = None
    SHUFFLE_PRODUCTS = None

    def __init__(self, env_id, _):
        assert NUM_INSTANCES == 1, "Only supporting 1 concurrent env with webshop at the moment"
        super().__init__(env_id, WebshopObservation)
        self._steps = 0
        
        self._env = WebAgentDreamDOMEnv(
            window_height=self.WINDOW_HEIGHT,
            window_width=self.WINDOW_WIDTH,
            scroll_amount=self.SCROLL_AMOUNT,
            scroll_time=self.SCROLL_TIME,
            return_n=self.RETURN_N,
            num_random=self.NUM_RANDOM,
            shuffle_products=self.SHUFFLE_PRODUCTS)

        if self.EMBED_STATES:
            self._env = MarkupLMWrapper(self._env, path=self.EMBED_PATH)

        self.observation_space = gym.spaces.Dict({
            "observation": self._env.observation_space,
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
        cls.QUANTIZE = config.get("quantize", None)
        cls.SCROLL_AMOUNT = config.get("scroll_amount", 180)
        cls.SCROLL_TIME = config.get("scroll_time", 150)
        cls.NUM_ACTIONS = config.get("num_actions", 4)
        cls.USE_SCREENSHOT = config.get("use_screenshot", False)
        cls.NUM_DEMOS = config.get("num_demos", 0)
        cls.EMBED_STATES = config.get("embed_states", False)
        cls.EMBED_PATH = config.get("embed_path", None)
        cls.RETURN_N = config.get("return_n", 1)
        cls.NUM_RANDOM = config.get("num_random", 0)
        cls.NUM_ITEMS = cls.RETURN_N + cls.NUM_RANDOM
        cls.SHUFFLE_PRODUCTS = config.get("shuffle_products", True)


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

    @classmethod
    def close(_):
        WebAgentDreamDOMEnv.close()

    @classmethod
    def set_test(cls, test):
        cls.TEST = test

    def is_demo(self):
        return True if self.ITER < self.NUM_DEMOS else False

    def get_demo(self):
        if self.exploitation:
            if isinstance(self.env_id[0], int):
                return torch.tensor(self.env_id).to("cpu")
            return torch.tensor([id[0] for id in self.env_id]).to("cpu") # Return correct answer
        elif self._steps == 0:
            return [0] # Search items on initial step
        if isinstance(self.env_id[0], int):
            return [id + 2 for id in self.env_id] # Look at the correct answer on subsequent steps
        return [id[0] + 2 for id in self.env_id] # Look at the correct answer on subsequent steps

    def _step(self, action):
        # print(f"Action: {action}")
        assert len(action) == 1, "Only supporting 1 concurrent env with webshop at the moment"
        if self.exploitation:
            return [{"observation": torch.zeros((1)) if self.EMBED_STATES else "", "question": ""}], [0], [True], [None]
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
            return [{"observation": torch.zeros((1)) if self.EMBED_STATES else "", "question": ""}]
        self._steps = 0
        start = time.time()
        state, _ = self._env.reset(goal_id=self._env_id[0], seed=self.ITER, is_test=self.TEST)
        # print(f"Time to reset: {time.time() - start}")
        self._questions = [state["instruction_text"]]

        self._correct_answers = [[p["index"] for p in state["best_products"]]]

        # Need to shuffle correct answers for training
        for i in range(len(self._correct_answers)):
            np.random.shuffle(self._correct_answers[i])
    
        state = state["html"]
        return [{
            "observation": state,
            "question": self._questions[0]
        }]

    def render(self, mode=None):
        imgs = []
        img = self._env.render()
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
        if isinstance(self._observation, torch.Tensor):
            return self._observation.is_cuda
        return False

    @property
    def observation(self):
        return self._observation

    @property
    def question(self):
        return self._question

    def cpu(self):
        # Hacky way to accomodate cpu/cuda switching in observation buffer
        if not isinstance(self._observation, torch.Tensor):
            return self
        return WebshopObservation({
            "observation": self._observation.detach().cpu(),
            "question": self._question
        })

    def pin_memory(self):
        if not isinstance(self._observation, torch.Tensor):
            return self
        return WebshopObservation({
            "observation": self._observation.pin_memory(),
            "question": self._question
        })

    def cuda(self, **kwargs):
        if not torch.cuda.is_available():
            return self
        if not isinstance(self._observation, torch.Tensor):
            return self
        return WebshopObservation({
            "observation": self._observation.cuda(**kwargs),
            "question": self._question
        })
