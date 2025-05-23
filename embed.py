import os

import abc
import math
import time
import collections
import numpy as np
import torch
import re
from torch import nn, Tensor
from torch import distributions as td
from torch.nn import functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import MarkupLMProcessor, MarkupLMModel

from envs import bounce
from envs import grid
from envs import miniwob
from envs import webshop
from envs.miniwob.constants import QUESTIONS, PEOPLE_NAMES, LOREM_WORDS, HTML_TOKENS, SYMBOLS
import relabel
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Embedder(abc.ABC, nn.Module):
    """Defines the embedding of an object in the forward method.

    Subclasses should register to the from_config method.
    """

    def __init__(self, embed_dim):
        """Sets the embed dim.

        Args:
            embed_dim (int): the dimension of the outputted embedding.
        """
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        """Returns the dimension of the output (int)."""
        return self._embed_dim

    @classmethod
    def from_config(cls, config):
        """Constructs and returns Embedder from config.

        Args:
            config (Config): parameters for constructing the Embedder.

        Returns:
            Embedder
        """
        config_type = config.get("type")
        if config_type == "simple_grid_state":
            return SimpleGridStateEmbedder.from_config(config)
        elif config_type == "fixed_vocab":
            return FixedVocabEmbedder.from_config(config)
        elif config_type == "linear":
            return LinearEmbedder.from_config(config)
        else:
            raise ValueError("Config type {} not supported".format(config_type))


def get_state_embedder(env):
    """Returns the appropriate type of embedder given the environment type."""
    env = env.unwrapped
    if isinstance(env.unwrapped, grid.GridEnv):
        return SimpleGridStateEmbedder
    elif isinstance(env.unwrapped, bounce.BounceBinaryMetaEnv) and env.IMG_STATE:
        return BounceImageEmbedder
    elif isinstance(env.unwrapped, bounce.BounceMetaEnv):
        return BounceEmbedder
    elif isinstance(env.unwrapped, webshop.WebShopMetaEnv):
        return WebshopEmbedder
    elif isinstance(env.unwrapped, miniwob.fake_inbox_scroll_vectorized.FakeInboxScrollVectorizedMetaEnv):
        return MiniWobVectorizedEmbedderV2
    elif isinstance(env.unwrapped, miniwob.inbox.InboxMetaEnv) or isinstance(env.unwrapped, miniwob.fake_inbox.FakeInboxMetaEnv) or isinstance(env.unwrapped, miniwob.fake_inbox_scroll.FakeInboxScrollMetaEnv) or isinstance(env.unwrapped, miniwob.fake_inbox_scroll_multiclass.FakeInboxScrollMulticlassMetaEnv):
        return MiniWobEmbedder
    # Dependencies on OpenGL, so only load if absolutely necessary
    from envs.miniworld import sign
    if isinstance(env, sign.MiniWorldSign):
        return MiniWorldEmbedder

    raise ValueError()


class TransitionEmbedder(Embedder):
    def __init__(self, state_embedder, action_embedder, reward_embedder, embed_dim):
        super().__init__(embed_dim)

        self._state_embedder = state_embedder
        self._action_embedder = action_embedder
        self._reward_embedder = reward_embedder
        reward_embed_dim = (
                0 if reward_embedder is None else reward_embedder.embed_dim)

        self._transition_embedder = nn.Sequential(
                nn.Linear(
                    self._state_embedder.embed_dim * 2 +
                    self._action_embedder.embed_dim + reward_embed_dim,
                    128),
                nn.ReLU(),
                nn.Linear(128, embed_dim)
        )

    def forward(self, experiences):
        state_embeds = self._state_embedder(
                [exp.state.observation for exp in experiences])
        next_state_embeds = self._state_embedder(
                [exp.next_state.observation for exp in experiences])
        action_embeds = self._action_embedder([exp.action for exp in experiences])
        embeddings = [state_embeds, next_state_embeds, action_embeds]
        if self._reward_embedder is not None:
            embeddings.append(self._reward_embedder(
                    [exp.next_state.prev_reward for exp in experiences]))
        transition_embeds = self._transition_embedder(torch.cat(embeddings, -1))
        return transition_embeds

    @classmethod
    def from_config(cls, config, env):
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                config.get("experience_embedder").get("state_embed_dim"), config.get("experience_embedder").get("state_embed_config"))
        action_embedder = FixedVocabEmbedder(
                env.action_space.n,
                config.get("experience_embedder").get("action_embedder").get("embed_dim"))
        return cls(state_embedder, action_embedder, config.get("embed_dim"))


class TrajectoryEmbedder(Embedder, relabel.RewardLabeler):
    def __init__(self, transition_embedder, id_embedder, penalty, embed_dim, decoder_output_dim=2):
        super().__init__(embed_dim)

        self._transition_embedder = transition_embedder
        self._id_embedder = id_embedder
        self._transition_lstm = nn.LSTM(transition_embedder.embed_dim, 128)
        self._transition_fc_layer = nn.Linear(128, 128)
        self._transition_output_layer = nn.Linear(128, embed_dim)
        # Outputs binary prediction
        # TODO: make this configurable based on whether doing binary or multiclass
        self._decoder_head = nn.Linear(embed_dim, decoder_output_dim)
        self._penalty = penalty
        self._use_ids = True

    def use_ids(self, use):
        self._use_ids = use

    def _compute_contexts(self, trajectories):
        """Returns contexts and masks.

        Args:
            trajectories (list[list[Experience]]): see forward().

        Returns:
            id_contexts (torch.FloatTensor): tensor of shape (batch_size)
                embedding the id's in the trajectories.
            all_transition_contexts (torch.FloatTensor): tensor of shape
                (batch_size, max_len + 1, embed_dim) embedding the sequences of states
                and actions in the trajectories.
            transition_contexts (torch.FloatTensor): tensor of shape
                (batch_size, embed_dim) equal to the last unpadded value in
                all_transition_contexts.
            mask (torch.BoolTensor): tensor of shape (batch_size, max_len + 1).
                The value is False if the trajectory_contexts value should be masked.
        """
        # trajectories: (batch_size, max_len)
        # mask: (batch_size, max_len)
        padded_trajectories, mask = utils.pad(trajectories)
        sequence_lengths = torch.tensor([len(traj) for traj in trajectories], device="cpu").long()

        # (batch_size * max_len, embed_dim)
        transition_embed = self._transition_embedder(
                [exp for traj in padded_trajectories for exp in traj])

        # Sorted only required for ONNX
        padded_transitions = nn.utils.rnn.pack_padded_sequence(
                transition_embed.reshape(mask.shape[0], mask.shape[1], -1),
                sequence_lengths, batch_first=True, enforce_sorted=False)

        transition_hidden_states = self._transition_lstm(padded_transitions)[0]
        # (batch_size, max_len, hidden_dim)
        transition_hidden_states, hidden_lengths = nn.utils.rnn.pad_packed_sequence(
                transition_hidden_states, batch_first=True)
        initial_hidden_states = torch.zeros(
                transition_hidden_states.shape[0], 1,
                transition_hidden_states.shape[-1])
        # (batch_size, max_len + 1, hidden_dim)
        transition_hidden_states = torch.cat(
                (initial_hidden_states, transition_hidden_states), 1)
        transition_hidden_states = F.relu(
                self._transition_fc_layer(transition_hidden_states))
        # (batch_size, max_len + 1, embed_dim)
        all_transition_contexts = self._transition_output_layer(
                transition_hidden_states)

        # (batch_size, 1, embed_dim)
        # Don't need to subtract 1 off of hidden_lengths as transition_contexts is
        # padded with init hidden state at the beginning.
        indices = hidden_lengths.unsqueeze(-1).unsqueeze(-1).expand(
                hidden_lengths.shape[0], 1, all_transition_contexts.shape[2]).to(
                        all_transition_contexts.device)
        transition_contexts = all_transition_contexts.gather(1, indices).squeeze(1)

        # (batch_size,)
        # HACK: This is just the env_ids now
        if isinstance(trajectories[0][0].state.env_id, int):
            env_ids = [traj[0].state.env_id for traj in trajectories]
            id_contexts = torch.tensor(env_ids)
        elif isinstance(trajectories[0][0].state.env_id, list):
            # Assume that correct env_ids are always shuffled
            env_ids = [traj[0].state.env_id for traj in trajectories]
            max_id = max([max(ids) for ids in env_ids])
            # Create a one-hot encoding of the env_ids
            id_contexts = torch.zeros(len(env_ids), max_id + 1)
            for i, ids in enumerate(env_ids):
                id_contexts[i, ids] = 1
        else:
            raise ValueError("Unsupported env_id type {}".format(
                trajectories[0][0].state.env_id))

        #id_contexts = self._id_embedder(
        #        torch.tensor([traj[0].state.env_id for traj in trajectories]))

        # don't mask the initial hidden states (batch_size, max_len + 1)
        mask = torch.cat(
                (torch.ones(transition_contexts.shape[0], 1).bool(), mask), -1)
        return id_contexts, all_transition_contexts, transition_contexts, mask

    def _compute_losses(
            self, trajectories, id_contexts, all_transition_contexts,
            transition_contexts, mask):
        """Computes losses based on the return values of _compute_contexts.

        Args:
            See return values of _compute_contexts.

        Returns:
            losses (dict(str: torch.FloatTensor)): see forward().
        """
        del trajectories

        # (batch_size, seq_len, 2)
        batch, seq_len, _ = all_transition_contexts.shape
        decoder_logits = self._decoder_head(all_transition_contexts)
        # We want to maximize \sum_t E[log q(z | tau^exp_{:t})]
        # Repeat to be (batch_size, seq_len)
        if len(id_contexts.shape) == 1:
            decoder_distribution = td.Categorical(logits=decoder_logits)
            id_contexts = id_contexts.unsqueeze(-1).expand(
                    -1, all_transition_contexts.shape[1])
            # (batch_size, seq_len)
            decoder_loss = -decoder_distribution.log_prob(id_contexts)
        else:
            # (batch_size, seq_len, env_id_dim)
            id_contexts = id_contexts.unsqueeze(1).expand(
                    -1, all_transition_contexts.shape[1], -1)

            # Compute loss for each id
            decoder_logits = decoder_logits[:,:,:id_contexts.shape[-1]]
            decoder_loss = -F.log_softmax(decoder_logits, dim=-1)
            decoder_loss = torch.sum(decoder_loss * id_contexts, dim=-1)

        decoder_loss = (decoder_loss * mask).sum() / mask.sum()

        losses = {
            "decoder_loss": decoder_loss,
            # No need for info bottleneck
            #"id_context_loss": torch.max((id_contexts ** 2).sum(-1), cutoff).mean()
        }
        return losses

    def forward(self, trajectories):
        """Embeds a batch of trajectories.

        Args:
            trajectories (list[list[Experience]]): batch of trajectories, where each
                trajectory comes from the same episode.

        Returns:
            embedding (torch.FloatTensor): tensor of shape (batch_size, embed_dim)
                embedding the trajectories. This embedding is based on the ids if
                use_ids is True, otherwise based on the transitions.
            losses (dict(str: torch.FloatTensor)): maps auxiliary loss names to their
                values.
        """
        # HACK: This now just returns the decoder label predictions
        id_contexts, all_transition_contexts, transition_contexts, mask = (
                self._compute_contexts(trajectories))
        # (batch_size, 2)
        decoder_logits = self._decoder_head(transition_contexts)
        #decoder_distribution = td.Categorical(logits=decoder_logits)
        ## (batch_size)
        #decoder_predictions = decoder_distribution.sample()

        losses = self._compute_losses(
                trajectories, id_contexts, all_transition_contexts,
                transition_contexts, mask)
        #return decoder_predictions, losses
        return decoder_logits, losses

    def label_rewards(self, trajectories):
        """Computes rewards for each experience in the trajectory.

        Args:
            trajectories (list[list[Experience]]): batch of trajectories.

        Returns:
            rewards (torch.FloatTensor): of shape (batch_size, max_seq_len) where
                rewards[i][j] is the rewards for the experience trajectories[i][j].
                This is padded with zeros and is detached from the graph.
            distances (torch.FloatTensor): of shape (batch_size, max_seq_len + 1)
                equal to ||f(e) - g(\tau^e_{:t})|| for each t.
        """
        id_contexts, all_transition_contexts, _, mask = self._compute_contexts(
                trajectories)

        batch, seq_len, _ = all_transition_contexts.shape
        decoder_logits = self._decoder_head(all_transition_contexts)
        # (batch_size, seq_len)
        if len(id_contexts.shape) == 1:
            # Compute rewards as E[log q(y | tau_{:t + 1}) - log q(y | tau_{:t})]
            # (batch_size, seq_len, 2)
            decoder_distribution = td.Categorical(logits=decoder_logits)
            decoder_log_probs = decoder_distribution.log_prob(
                    id_contexts.unsqueeze(-1).expand(-1, decoder_logits.shape[1]))
        else:
            id_contexts = id_contexts.clone()

            # Need to ignore masked env ids
            id_contexts[id_contexts == 0] = torch.inf
            id_contexts = id_contexts.unsqueeze(1).expand(
                    -1, all_transition_contexts.shape[1], -1)

            # Calculate log probs for each id
            decoder_logits = decoder_logits[:,:,:id_contexts.shape[-1]]
            decoder_log_probs = F.log_softmax(decoder_logits, dim=-1)

            # We use the max prob over correct env ids
            decoder_log_probs = torch.max((decoder_log_probs * id_contexts), dim=-1).values

        # Add penalty
        rewards = (decoder_log_probs[:, 1:] -
                   decoder_log_probs[:, :-1] - self._penalty)

        #distances = (
        #        (all_transition_contexts - id_contexts.unsqueeze(1).expand_as(
        #         all_transition_contexts).detach()) ** 2).sum(-1)
        #rewards = distances[:, :-1] - distances[:, 1:] - self._penalty
        #return (rewards * mask[:, 1:]).detach(), distances
        return (rewards * mask[:, 1:]).detach(), decoder_log_probs.detach()


class InstructionPolicyEmbedder(Embedder):
    """Embeds (s, i, \tau^e) where:

        - s is the current state
        - i is the current instruction
        - \tau^e is an exploration trajectory (s_0, a_0, s_1, ..., s_T)
    """

    def __init__(self, trajectory_embedder, obs_embedder, instruction_embedder,
                             embed_dim):
        """Constructs around embedders for each component.

        Args:
            trajectory_embedder (TrajectoryEmbedder): embeds batches of \tau^e
                (list[list[rl.Experience]]).
            obs_embedder (Embedder): embeds batches of states s.
            instruction_embedder (Embedder): embeds batches of instructions i.
            embed_dim (int): see Embedder.
        """
        super().__init__(embed_dim)

        self._obs_embedder = obs_embedder
        self._instruction_embedder = instruction_embedder
        self._trajectory_embedder = trajectory_embedder
        self._fc_layer = nn.Linear(
                obs_embedder.embed_dim + self._trajectory_embedder.embed_dim, 256)
        self._final_layer = nn.Linear(256, embed_dim)

    def forward(self, states, hidden_state):
        # obs_embed, hidden_state = self._obs_embedder(states, hidden_state)
        trajectory_embed, _ = self._trajectory_embedder(
                [state[0].trajectory for state in states])
        # This is just the decoder prediction
        #return trajectory_embed, hidden_state
        return trajectory_embed, None

        if len(obs_embed.shape) > 2:
            trajectory_embed = trajectory_embed.unsqueeze(1).expand(
                    -1, obs_embed.shape[1], -1)

        hidden = F.relu(self._fc_layer(
                torch.cat((obs_embed, trajectory_embed), -1)))
        return self._final_layer(hidden), hidden_state

    def aux_loss(self, experiences):
        _, aux_losses = self._trajectory_embedder(
                [exp[0].state.trajectory for exp in experiences])
        return aux_losses

    @classmethod
    def from_config(cls, config, env):
        """Returns a configured InstructionPolicyEmbedder.

        Args:
            config (Config): see Embedder.from_config.
            env (gym.Wrapper): the environment to run on. Expects this to be wrapped
                with an InstructionWrapper.

        Returns:
            InstructionPolicyEmbedder: configured according to config.
        """
        obs_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                config.get("obs_embedder").get("embed_dim"), config.get("obs_embedder").get("state_embed_config"))
        # Use SimpleGridEmbeder since these are just discrete vars
        instruction_embedder = SimpleGridStateEmbedder(
                env.observation_space["instructions"],
                config.get("instruction_embedder").get("embed_dim"))
        # Exploitation recurrence is not observing the rewards
        exp_embedder = ExperienceEmbedder(
                obs_embedder, instruction_embedder, None, None, None,
                obs_embedder.embed_dim)
        obs_embedder = RecurrentStateEmbedder(exp_embedder, obs_embedder.embed_dim)

        transition_config = config.get("transition_embedder")
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                transition_config.get("state_embed_dim"), transition_config.get("state_embed_config"))
        # This needs to cover embedding of the exploration time env...
        action_embedder = FixedVocabEmbedder(
                env.unwrapped.action_space.n, transition_config.get("action_embed_dim"))
        reward_embedder = None
        if transition_config.get("reward_embed_dim") is not None:
            reward_embedder = LinearEmbedder(
                    1, transition_config.get("reward_embed_dim"))
        transition_embedder = TransitionEmbedder(
                state_embedder, action_embedder, reward_embedder,
                transition_config.get("embed_dim"))
        id_embedder = IDEmbedder(
                env.observation_space["env_id"].high,
                config.get("transition_embedder").get("embed_dim"))
        if config.get("trajectory_embedder").get("type") == "ours":
            trajectory_embedder = TrajectoryEmbedder(
                    transition_embedder, id_embedder,
                    config.get("trajectory_embedder").get("penalty"),
                    transition_embedder.embed_dim,
                    decoder_output_dim=config.get("trajectory_embedder").get("decoder_output_dim"))
        else:
            raise ValueError("Unsupported trajectory embedder {}".format(
                config.get("trajectory_embedder")))
        return cls(trajectory_embedder, obs_embedder, instruction_embedder,
                             config.get("embed_dim"))


class RecurrentAndTaskIDEmbedder(Embedder):
    """Embedding used by IMPORT.

    Compute both:
        - g(\tau_{:t}) recurrently
        - f(e)

    Full embedding is:
        \phi(s_t, z), where z is randomly chosen from g(\tau_{:t}) and f(e).
    """

    def __init__(
            self, recurrent_state_embedder, id_embedder, state_embedder, embed_dim):
        super().__init__(embed_dim)
        assert id_embedder.embed_dim == recurrent_state_embedder.embed_dim
        self._recurrent_state_embedder = recurrent_state_embedder
        self._id_embedder = id_embedder
        self._state_embedder = state_embedder
        self._final_layer = nn.Linear(
                id_embedder.embed_dim + state_embedder.embed_dim, embed_dim)
        self._use_id = False

    def use_ids(self, use):
        self._use_id = use

    def _compute_embeddings(self, states, hidden_state=None):
        # (batch_size, seq_len, embed_dim)
        recurrent_embedding, hidden_state = self._recurrent_state_embedder(
                states, hidden_state)
        # (batch_size, embed_dim)
        id_embedding = self._id_embedder(
                torch.tensor([seq[0].env_id for seq in states]))

        if len(recurrent_embedding.shape) > 2:
            id_embedding = id_embedding.unsqueeze(1).expand_as(recurrent_embedding)
        return recurrent_embedding, id_embedding, hidden_state

    def forward(self, states, hidden_state=None):
        recurrent_embedding, id_embedding, hidden_state = self._compute_embeddings(
                states, hidden_state)

        history_embed = recurrent_embedding
        if self._use_id:
            history_embed = id_embedding

        # (batch_size, seq_len, state_embed_dim) or (batch_size, state_embed_dim)
        state_embeds = self._state_embedder(
                [state for seq in states for state in seq])
        if len(history_embed.shape) > 2:
            state_embeds = state_embeds.reshape(
                    history_embed.shape[0], history_embed.shape[1], -1)
        return self._final_layer(
                F.relu(torch.cat((history_embed, state_embeds), -1))), hidden_state

    def aux_loss(self, trajectories):
        # (batch_size, max_seq_len)
        trajectories, mask = utils.pad(trajectories)

        # (batch_size, max_seq_len, embed_dim)
        recurrent_embeddings, id_embeddings, hidden_state = self._compute_embeddings(
                [[exp.state for exp in traj] for traj in trajectories],
                [traj[0].agent_state for traj in trajectories])

        return {
            "embedding_distance": (
                    ((recurrent_embeddings - id_embeddings.detach()) ** 2)
                    .mean(0).sum())
        }

    @classmethod
    def from_config(cls, config, env):
        recurrent_state_embedder = RecurrentStateEmbedder.from_config(
                config.get("recurrent_embedder"), env)
        state_embed_config = config.get("state_embedder")
        state_embedder = get_state_embedder(env)(
            env.observation_space["observation"],
            state_embed_config.get("embed_dim"), state_embed_config.get("state_embed_config"))
        instruction_embedder = SimpleGridStateEmbedder(
            env.observation_space["instructions"],
            state_embed_config.get("embed_dim"))
        state_embedder = StateInstructionEmbedder(
                state_embedder, instruction_embedder,
                state_embed_config.get("embed_dim"))

        id_embed_config = config.get("id_embedder")
        id_embedder = IDEmbedder(
                env.observation_space["env_id"].high,
                id_embed_config.get("embed_dim"))
        return cls(
                recurrent_state_embedder, id_embedder, state_embedder,
                config.get("embed_dim"))


class VariBADEmbedder(Embedder):
    """Embedding used by VariBAD.

    Computes:
        - g(\tau_{:t}) recurrently and applies fully connected heads on top to
            produce q(z_t | \tau_{:t}) = N(head1(g(\tau_{:t})), head2(g(\tau_{:t})))
        - embedding = \phi(z_t.detach(), embed(s_t))

    Decoding auxiliary loss:
        - \sum_t \sum_i ||decoder(z_i, e(s_t), e(a_t)) - r_t||_2^2
        - \sum_t \sum_i ||decoder(z_i, e(s_t), e(a_t)) - s_{t + 1}||_2^2
    """

    def __init__(
            self, recurrent_state_embedder, z_dim, state_embedder, action_embedder,
            state_dim, embed_dim, predict_state=True):
        super().__init__(embed_dim)
        self._recurrent_state_embedder = recurrent_state_embedder
        self._fc_mu = nn.Linear(recurrent_state_embedder.embed_dim, z_dim)
        self._fc_logvar = nn.Linear(recurrent_state_embedder.embed_dim, z_dim)
        self._state_embedder = state_embedder
        self._phi = nn.Linear(
                z_dim + state_embedder.embed_dim, embed_dim)
        self._action_embedder = action_embedder
        self._decoder = nn.Sequential(
            nn.Linear(z_dim + state_embedder.embed_dim + action_embedder.embed_dim,
                                128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Predicts reward / state
        self._reward_head = nn.Linear(128, 1)
        self._state_head = nn.Linear(128, state_dim)

        # If False, does not do state prediction
        self._predict_state = predict_state
        self._z_dim = z_dim

    def _compute_z_distr(self, states, hidden_state=None):
        embeddings, hidden_state = self._recurrent_state_embedder(
                states, hidden_state=hidden_state)

        # (batch_size, sequence_length, embed_dim)
        mu = embeddings
        std = torch.ones_like(mu) * 1e-6

        q = td.Independent(td.Normal(mu, std), 1)
        return q, hidden_state

    def forward(self, states, hidden_state=None):
        q, hidden_state = self._compute_z_distr(states, hidden_state)
        # Don't backprop through encoder
        z = q.rsample()

        # (batch_size, seq_len, state_embed_dim) or (batch_size, state_embed_dim)
        state_embeds = self._state_embedder(
                [state for seq in states for state in seq])
        if len(z.shape) > 2:
            state_embeds = state_embeds.reshape(z.shape[0], z.shape[1], -1)
        return self._phi(F.relu(torch.cat((z, state_embeds), -1))), hidden_state

    def aux_loss(self, trajectories):
        # The trajectories that we will try to decode
        # (batch_size, max_trajectory_len)
        trajectories_to_predict, predict_mask = utils.pad(
                [traj[0].trajectory for traj in trajectories])

        # The trajectories we're using to encode z
        # They differ when we sample not the full trajectory
        # (batch_size, max_sequence_len)
        padded_trajectories, mask = utils.pad(trajectories)

        q = self._compute_z_distr(
                [[exp.state for exp in traj] for traj in padded_trajectories],
                [traj[0].agent_state for traj in padded_trajectories])[0]
        # (batch_size, max_sequence_len, z_dim)
        z = q.rsample()
        # (batch_size, max_trajectory_len, max_sequence_len, z_dim)
        z = z.unsqueeze(1).expand(-1, predict_mask.shape[1], -1, -1)

        # (batch_size, max_trajectory_len, embed_dim)
        # e(s)
        state_embeds = self._state_embedder(
                [exp.state for trajectory in trajectories_to_predict
                 for exp in trajectory]).reshape(z.shape[0], z.shape[1], -1)
        # e(a)
        action_embeds = self._action_embedder(
                [exp.action for trajectory in trajectories_to_predict
                 for exp in trajectory]).reshape(z.shape[0], z.shape[1], -1)

        # (batch_size, max_trajectory_len, max_sequence_len, embed_dim)
        state_embeds = state_embeds.unsqueeze(2).expand(-1, -1, z.shape[2], -1)
        action_embeds = action_embeds.unsqueeze(2).expand(-1, -1, z.shape[2], -1)

        decoder_input = torch.cat((z, state_embeds, action_embeds), -1)
        decoder_embed = self._decoder(decoder_input)

        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        predicted_rewards = self._reward_head(F.relu(decoder_embed))

        # (batch_size, max_trajectory_len)
        true_rewards = torch.tensor(
                [[exp.next_state.prev_reward for exp in trajectory]
                 for trajectory in trajectories_to_predict])

        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        true_rewards = true_rewards.unsqueeze(-1).unsqueeze(-1).expand_as(
                predicted_rewards)

        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        reward_decoding_loss = ((predicted_rewards - true_rewards) ** 2)

        predict_mask = predict_mask.unsqueeze(2).expand(-1, -1, mask.shape[-1])
        mask = mask.unsqueeze(1).expand_as(predict_mask)
        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        aggregate_mask = (predict_mask * mask).unsqueeze(-1)
        reward_decoding_loss = ((reward_decoding_loss * aggregate_mask).sum() /
                                                        reward_decoding_loss.shape[0])

        state_decoding_loss = torch.tensor(0).float()
        if self._predict_state:
            # (batch_size, max_trajectory_len, max_sequence_len, state_dim)
            predicted_states = self._state_head(F.relu(decoder_embed))

            # (batch_size, max_trajectory_len, state_dim)
            next_states_to_predict = torch.stack(
                    [torch.stack([exp.next_state.observation for exp in trajectory])
                     for trajectory in trajectories_to_predict])

            # (batch_size, max_trajectory_len, max_sequence_len, state_dim)
            next_states_to_predict = next_states_to_predict.unsqueeze(2).expand_as(
                    predicted_states)

            # (batch_size, max_trajectory_len, max_sequence_len, state_dim)
            state_decoding_loss = ((predicted_states - next_states_to_predict) ** 2)
            state_decoding_loss = ((state_decoding_loss * aggregate_mask).sum() /
                                                         state_decoding_loss.shape[0])

        #kl_loss = td.kl_divergence(q, self._prior(mask.shape[0], mask.shape[1]))
        return {
            "reward_decoding_loss": reward_decoding_loss,
            "state_decoding_loss": state_decoding_loss * 0.01,
            #"kl_loss": kl_loss * 0.1,
        }

    def _prior(self, batch_size, sequence_len):
        mu = torch.zeros(batch_size, sequence_len, self._z_dim)
        std = torch.ones_like(mu)
        return td.Independent(td.Normal(mu, std), 1)

    @classmethod
    def from_config(cls, config, env):
        recurrent_state_embedder = RecurrentStateEmbedder.from_config(
                config.get("recurrent_embedder"), env)
        state_embed_config = config.get("state_embedder")
        state_embedder = get_state_embedder(env)(
            env.observation_space["observation"],
            state_embed_config.get("embed_dim"), state_embed_config.get("state_embed_config"))
        instruction_embedder = SimpleGridStateEmbedder(
            env.observation_space["instructions"],
            state_embed_config.get("embed_dim"))
        state_embedder = StateInstructionEmbedder(
                state_embedder, instruction_embedder,
                state_embed_config.get("embed_dim"))

        action_embed_config = config.get("action_embedder")
        action_embedder = FixedVocabEmbedder(
                env.action_space.n, action_embed_config.get("embed_dim"))
        state_dim = len(env.observation_space["observation"].high)
        return cls(
                recurrent_state_embedder, config.get("z_dim"), state_embedder,
                action_embedder, state_dim, config.get("embed_dim"),
                config.get("predict_states"))


class RecurrentStateEmbedder(Embedder):
    """Applies an LSTM on top of a state embedding."""

    def __init__(self, state_embedder, embed_dim):
        super().__init__(embed_dim)

        self._state_embedder = state_embedder
        self._lstm_cell = nn.LSTMCell(state_embedder.embed_dim, embed_dim)

    def forward(self, states, hidden_state=None):
        """Embeds a batch of sequences of contiguous states.

        Args:
            states (list[list[np.array]]): of shape
                (batch_size, sequence_length, state_dim).
            hidden_state (list[object] | None): batch of initial hidden states
                to use with the LSTM. During inference, this should just be the
                previously returned hidden state.

        Returns:
            embedding (torch.tensor): shape (batch_size, sequence_length, embed_dim)
            hidden_state (object): hidden state after embedding every element in the
                sequence.
        """
        batch_size = len(states)
        sequence_len = len(states[0])

        # Stack batched hidden state
        if hidden_state is not None and all(h is None for h in hidden_state):
            hidden_state = None

        if hidden_state is not None:
            hs = []
            cs = []
            for hidden in hidden_state:
                if hidden is None:
                    hs.append(torch.zeros(1, self.embed_dim))
                    cs.append(torch.zeros(1, self.embed_dim))
                else:
                    hs.append(hidden[0])
                    cs.append(hidden[1])
            hidden_state = (torch.cat(hs, 0), torch.cat(cs, 0))

        flattened = [state for seq in states for state in seq]

        # (batch_size * sequence_len, embed_dim)
        state_embeds = self._state_embedder(flattened)
        state_embeds = state_embeds.reshape(batch_size, sequence_len, -1)
        embeddings = []
        for seq_index in range(sequence_len):
            hidden_state = self._lstm_cell(
                state_embeds[:, seq_index, :], hidden_state)

            # (batch_size, 1, embed_dim)
            embeddings.append(hidden_state[0].unsqueeze(1))

        # (batch_size, sequence_len, embed_dim)
        # squeezed to (batch_size, embed_dim) if sequence_len == 1
        embeddings = torch.cat(embeddings, 1).squeeze(1)

        # Detach to save GPU memory.
        detached_hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        return embeddings, detached_hidden_state

    @classmethod
    def from_config(cls, config, env):
        experience_embed_config = config.get("experience_embedder")
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                experience_embed_config.get("state_embed_dim"), experience_embed_config.get("state_embed_config"))
        action_embedder = FixedVocabEmbedder(
                env.action_space.n + 1, experience_embed_config.get("action_embed_dim"))
        instruction_embedder = None
        if experience_embed_config.get("instruction_embed_dim") is not None:
            # Use SimpleGridEmbedder since these are just discrete vars
            instruction_embedder = SimpleGridStateEmbedder(
                    env.observation_space["instructions"],
                    experience_embed_config.get("instruction_embed_dim"))

        reward_embedder = None
        if experience_embed_config.get("reward_embed_dim") is not None:
            reward_embedder = LinearEmbedder(
                    1, experience_embed_config.get("reward_embed_dim"))

        done_embedder = None
        if experience_embed_config.get("done_embed_dim") is not None:
            done_embedder = FixedVocabEmbedder(
                    2, experience_embed_config.get("done_embed_dim"))

        experience_embedder = ExperienceEmbedder(
                state_embedder, instruction_embedder, action_embedder,
                reward_embedder, done_embedder,
                experience_embed_config.get("embed_dim"))
        return cls(experience_embedder, config.get("embed_dim"))


class StateInstructionEmbedder(Embedder):
    """Embeds instructions and states and applies a linear layer on top."""

    def __init__(self, state_embedder, instruction_embedder, embed_dim):
        super().__init__(embed_dim)
        self._state_embedder = state_embedder
        self._instruction_embedder = instruction_embedder
        if instruction_embedder is not None:
            self._final_layer = nn.Linear(
                    state_embedder.embed_dim + instruction_embedder.embed_dim, embed_dim)
            assert self._state_embedder.embed_dim == embed_dim

    def forward(self, states):
        state_embeds = self._state_embedder([state.observation for state in states])
        if self._instruction_embedder is not None:
            instruction_embeds = self._instruction_embedder(
                    [torch.tensor(state.instructions) for state in states])
            return self._final_layer(
                    F.relu(torch.cat((state_embeds, instruction_embeds), -1)))
        return state_embeds


def init(module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module


class Flatten(nn.Module):
        def forward(self, x):
                return x.view(x.size(0), -1)


class MiniWorldEmbedder(Embedder):
    """Embeds 80x60 MiniWorld inputs.

    Network taken from gym-miniworld/.
    """
    def __init__(self, observation_space, embed_dim):
        super().__init__(embed_dim)

        # Architecture from gym-miniworld
        # For 80x60 input
        num_inputs = observation_space.shape[0]

        self._network = nn.Sequential(
                nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2),
                nn.ReLU(),

                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.ReLU(),

                nn.Conv2d(32, 32, kernel_size=4, stride=2),
                nn.ReLU(),

                Flatten(),

                nn.Linear(32 * 7 * 5, embed_dim),
        )

    def forward(self, obs):
        # (batch_size, 80, 60, 3)
        tensor = torch.stack(obs) / 255.
        return self._network(tensor)


class BounceEmbedder(Embedder):
    def __init__(self, observation_space, embed_dim):
        super().__init__(embed_dim)

        hidden_size = 128
        self._network = nn.Sequential(
                nn.Linear(observation_space.shape[0], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, embed_dim),
        )

    def forward(self, obs):
        obs = torch.stack(obs).float()
        return self._network(obs)


class BounceImageEmbedder(Embedder):
    """Embeds 84x84x1 bounce images."""
    def __init__(self, observation_space, embed_dim):
        super().__init__(embed_dim)

        # Architecture from DQN
        # For 84x84 input

        self._network = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, padding=(1, 1), kernel_size=3, stride=1),
                nn.Flatten(),
                nn.Linear(10 * 10 * 64, embed_dim))

    def forward(self, obs):
        # (batch_size, 80, 60, 3)
        tensor = torch.stack(obs) / 255.
        # This is hacking in the channel dim
        tensor = tensor.permute(0, 3, 2, 1)
        return self._network(tensor)


class MiniWobScreenshotEmbedder(Embedder):
    "Embeds screenshots using architecture from https://proceedings.mlr.press/v162/humphreys22a/humphreys22a.pdf"

    def __init__(self, observation_space, embed_dim=512):
        super().__init__(embed_dim)

        self._network = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, embed_dim, kernel_size=3, stride=2)
        )

    def forward(self, obs):
        # (batch_size, 80, 60, 3)
        tensor = obs / 255.
        # This is hacking in the channel dim
        tensor = tensor.permute(0, 3, 2, 1)
        result = self._network(tensor)
        return result.reshape(result.shape[0], self._embed_dim, -1).permute(0, 2, 1)


class Residual(nn.Module):  #@save
    """The Residual block of ResNet models. From https://d2l.ai/chapter_convolutional-modern/resnet.html"""
    def __init__(self, num_in, num_out, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
                                   stride=strides)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(num_out, num_out, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if use_1x1conv:
            self.conv3 = nn.Conv2d(num_in, num_out, kernel_size=1,
                                       stride=strides)
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)



class MiniWobLanguageEmbedder(Embedder):
    VOCABULARY = HTML_TOKENS + [w.lower() for w in PEOPLE_NAMES] + LOREM_WORDS + SYMBOLS

    def __init__(self, observation_space, embed_dim=256):
        super().__init__(embed_dim)

        self.tokenizer = self._tokenize
        self.vocab = build_vocab_from_iterator([type(self).VOCABULARY], specials=["<unk>", "<pad>", "<bos>"])               
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.embed = nn.Embedding(len(self.vocab), self.embed_dim)
        self.pos_enc = PositionalEncoding(self.embed_dim, max_len=300)
        if torch.cuda.is_available():
            self.embed = self.embed.cuda()


    @staticmethod
    def _tokenize(text):
        text = text.lower()
        
        # Remove punctuation
        text = text.replace(",", "")
        text = text.replace(".", "")
        text = text.replace("<", "")
        text = text.replace(">", "")

        # Split into tokens
        text = text.split()
        
        # Remove unknown words
        return [t for t in text if t in MiniWobLanguageEmbedder.VOCABULARY]
        

    def forward(self, obs):
        """Expects shape (batch_size, 1)"""
        obs = [torch.tensor([self.vocab["<bos>"]] + self.vocab(self.tokenizer(item))) for item in obs]
        # Pad to max length
        obs = nn.utils.rnn.pad_sequence(obs, batch_first=True, padding_value=self.vocab["<pad>"]).to(device)
        # Generate padding mask
        src_pad_mask = (obs == self.vocab["<pad>"]).to(device)
        obs = obs.permute(1, 0)
        embeddings = self.embed(obs)
        embeddings = self.pos_enc(embeddings)
        return embeddings, src_pad_mask


class MiniWobLanguageTransformer(Embedder):
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout_prob = 0.1  # dropout probability
    d_vocab = 128

    def __init__(self, observation_space, embed_dim=256):
        super().__init__(embed_dim)

        self.query = nn.Parameter(torch.randn((1, embed_dim)))
        self.pos_encoding = PositionalEncoding(self.d_vocab)
        self.encoder = nn.MultiheadAttention(
            embed_dim, 
            self.nhead,
            dropout=self.dropout_prob,
            kdim=self.d_vocab, 
            vdim=self.d_vocab,
            batch_first=False)
        self.norm1 = nn.LayerNorm(self.d_vocab, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        
    def forward(self, obs, query=None, pad_mask=None):
        if query is None:
            query = torch.repeat_interleave(self.query.unsqueeze(0), obs.shape[1], dim=1)
        else:
            # expected input to be (batch, 1, embed)
            query = query.permute(1, 0, 2)
        # obs = self.pos_encoding(obs)
        obs = self.norm1(obs)
        query = self.norm3(query)
        embeddings = query + self.encoder(query, obs, obs, key_padding_mask=pad_mask)[0]
        embeddings = embeddings.reshape(obs.shape[1], -1)
        embeddings = embeddings + self.dropout(self.fc1(self.norm2(embeddings)))
        return embeddings


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

"""class MiniWobQuestionEmbedder(Embedder):
    d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    transf_embed_dim = 64
    
    def __init__(self, observation_space, embed_dim=256, use_dom=False):
        super().__init__(embed_dim)

        self.tokenizer = get_tokenizer('basic_english')
        phrases = QUESTIONS + [" ".join(LOREM_WORDS), " ".join(PEOPLE_NAMES), "."]
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, phrases), specials=["<unk>", "<pad>", "<bos>"])
        if use_dom:
            for t in HTML_TOKENS:
                if t not in self.vocab:
                    self.vocab.append_token(t)
               
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.model = TransformerEmbedder(len(self.vocab),self.transf_embed_dim, self.nhead, self.d_hid, self.nlayers, self.dropout)
        self.output_proj = nn.Linear(self.transf_embed_dim, embed_dim)

    def forward(self, obs):
        # Expects shape (batch_size, 1)
        obs = [torch.tensor([self.vocab["<bos>"]] + self.vocab(self.tokenizer(item))) for item in obs]
        # Pad to max length
        obs = nn.utils.rnn.pad_sequence(obs, batch_first=True, padding_value=self.vocab["<pad>"]).to(device)
        # Generate padding mask
        src_pad_mask = (obs == self.vocab["<pad>"]).to(device)
        obs = obs.permute(1, 0)
        src_mask = self.model.generate_square_subsequent_mask(len(obs)).to(device)
        embeddings = self.model(obs, src_mask, src_pad_mask)

        # Mean pool while taking into account mask
        attn_mask = ~src_pad_mask.bool()
        num_tokens = attn_mask.sum(axis=-1).unsqueeze(-1)
        sum = (embeddings.permute(1, 0, 2) * attn_mask.unsqueeze(-1)).sum(axis=1)
        pooled_embedding = sum / num_tokens
        return F.relu(self.output_proj(pooled_embedding))"""


"""class TransformerEmbedder(nn.Module):
    # Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, src_pad_mask: Tensor = None) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_pad_mask)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)"""


class MiniWobVectorizedEmbedderV2(Embedder):
    """Embedder for SimpleGridEnv states.

    Concretely, embeds (x, y) separately with different embeddings for each cell.
    """

    def __init__(self, observation_space, embed_dim, use_dom=False):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (spaces.Box): limits for the observations to embed.
        """
        super().__init__(embed_dim)

        assert all(dim == 0 for dim in observation_space.feature_space["screenshot"].low)
        assert observation_space.feature_space["screenshot"].dtype == np.int

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim + 1, hidden_size) for dim in observation_space.feature_space["screenshot"].high])
        self._fc_layer = nn.Linear(hidden_size * len(observation_space.feature_space["screenshot"].high), 256)
        self._final_fc_layer = nn.Linear(256, embed_dim)

    def forward(self, obs):
        if isinstance(obs, list):
            question = [o.question for o in obs]
            dom = [o.dom for o in obs]
            screenshot = torch.stack([o.screenshot for o in obs])
        else:
            question = [obs.question]
            dom = [obs.dom]
            screenshot = obs.screenshot.unsqueeze(0)
        
        # Check batch size
        assert len(question) == screenshot.shape[0], "Batch size mismatch"
        B = len(question)

        tensor = screenshot.int().to(device)
        embeds = []
        for i in range(tensor.shape[1]):
            try:
                embeds.append(self._embedders[i](tensor[:, i]))
            except IndexError as e:
                print(f"IndexError at {i}")
                print(f"Vals: {tensor[:, i]}")
                print(f"embedder dim {self._embedders[i].num_embeddings}")
                raise e
        res = self._final_fc_layer(F.relu(self._fc_layer(torch.cat(embeds, -1))))
        return res
    

class MiniWobQuestionEmbedder(Embedder):
    """Embedder for SimpleGridEnv states.

    Concretely, embeds (x, y) separately with different embeddings for each cell.
    """

    def __init__(self, observation_space, embed_dim):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (spaces.Box): limits for the observations to embed.
        """
        super().__init__(embed_dim)

        assert all(dim == 0 for dim in observation_space.low)
        assert observation_space.dtype == np.int

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim + 1, hidden_size) for dim in observation_space.high])
        self._fc_layer = nn.Linear(hidden_size * len(observation_space.high), 256)
        self._final_fc_layer = nn.Linear(256, embed_dim)

    def forward(self, obs):

        tensor = obs.int().to(device)
        embeds = []
        for i in range(tensor.shape[1]):
            embeds.append(self._embedders[i](tensor[:, i]))
        res = self._final_fc_layer(F.relu(self._fc_layer(torch.cat(embeds, -1))))
        return res


class MiniWobEmbedder(Embedder):
    # nlayers = 8
    # nhead = 8
    nlayers = 2 #6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.1  # dropout probability
    raw_embed_dim = 128
    
    def __init__(self, observation_space, embed_dim=256):
        super().__init__(embed_dim)

        self.question_embedder = MiniWobQuestionEmbedder(observation_space.feature_space["question"], type(self).raw_embed_dim)
        if observation_space.feature_space.get("dom") is not None:
            self.dom_embedder = MiniWobLanguageEmbedder(observation_space.feature_space["dom"], embed_dim=type(self).raw_embed_dim)
        else:
            self.dom_embedder = None

        if observation_space.feature_space.get("screenshot") is not None:
            self.screenshot_embedder = MiniWobScreenshotEmbedder(observation_space.feature_space["screenshot"], embed_dim=type(self).raw_embed_dim)
        else:
            self.screenshot_embedder = None

        if observation_space.feature_space.get("scroll_state") is not None:
            self.scroll_state_embedder = MiniWobQuestionEmbedder(observation_space.feature_space["scroll_state"], embed_dim=type(self).raw_embed_dim)
        else:
            self.screenshot_embedder = None

        encoder_layers = nn.TransformerEncoderLayer(type(self).raw_embed_dim, self.nhead, type(self).raw_embed_dim, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)
        self.linear = nn.Linear(type(self).raw_embed_dim, embed_dim)

    def forward(self, obs):
        if isinstance(obs, list):
            question = torch.tensor(np.stack([o.question for o in obs]))
            dom = [o.dom for o in obs]
            screenshot = torch.stack([o.screenshot for o in obs])
            scroll_state = torch.tensor(np.stack([o.scroll_state for o in obs]))
        else:
            question = torch.tensor([obs.question])
            dom = [obs.dom]
            screenshot = obs.screenshot.unsqueeze(0)
            scroll_state = torch.tensor([obs.scroll_state])
        
        # Check batch size
        assert len(question) == screenshot.shape[0], "Batch size mismatch"
        B = len(question)

        question_embedding = self.question_embedder(question).unsqueeze(0)
        pad_mask = torch.zeros((B, 1), dtype=torch.bool).to(device)
        # print(f"Q embedding size {question_embedding.shape}")
        # print(f"Pad mask: {pad_mask.shape}")

        if self.dom_embedder is not None:
            dom_embedding, dom_pad_mask = self.dom_embedder(dom)
            # print(f"DOM embedding size {dom_embedding.shape}")
            question_embedding = torch.cat([question_embedding, dom_embedding], dim=0)
            # print(f"Q embedding size {question_embedding.shape}")
            pad_mask = torch.cat([pad_mask, dom_pad_mask], dim=1)
            # print(f"Pad mask: {pad_mask.shape}")

        if self.screenshot_embedder is not None:
            screenshot_embedding = self.screenshot_embedder(screenshot).permute(1, 0, 2)
            # print(f"Screenshot embedding size {screenshot_embedding.shape}")
            question_embedding = torch.cat([question_embedding, screenshot_embedding], dim=0)
            # print(f"Q embedding size {question_embedding.shape}")
            pad_mask = torch.cat([pad_mask, torch.zeros((B, screenshot_embedding.shape[0]), dtype=torch.bool).to(device)], dim=1)
            # print(f"Pad mask: {pad_mask.shape}")

        if self.scroll_state_embedder is not None:
            scroll_state_embedding = self.scroll_state_embedder(scroll_state).unsqueeze(0)
            question_embedding = torch.cat([question_embedding, scroll_state_embedding], dim=0)
            # print(f"Q embedding size {question_embedding.shape}")
            pad_mask = torch.cat([pad_mask, torch.zeros((B, scroll_state_embedding.shape[0]), dtype=torch.bool).to(device)], dim=1)
            # print(f"Pad mask: {pad_mask.shape}")

        multi_embedding = self.transformer_encoder(question_embedding, src_key_padding_mask=pad_mask)
        # print(f"Multi embedding size {multi_embedding.shape}")
        return self.linear(multi_embedding[0,:,:])


class WebshopEmbedder(Embedder):
    pretrained_path = "microsoft/markuplm-base"

    _model = None
    processor = None

    EMBEDDING_CACHE = {}
    
    def __init__(self, observation_space, embed_dim=256, config={}):
        super().__init__(embed_dim)

        self.use_pooled = config.get("use_pool", True)
        self.use_buffer = config.get("use_buffer", True)
        self.final_relu = config.get("final_relu", True)
        self.n_layers = config.get("n_layers", 1)
        self.heads = config.get("heads", 4)
        self.dropout = config.get("dropout", 0.0)
        self.unfreeze_layers = config.get("unfreeze_layers", [])
        self.is_already_embedded = observation_space.dtype == np.float32

        if not self.is_already_embedded:
            if WebshopEmbedder.processor is None:
                WebshopEmbedder.processor = MarkupLMProcessor.from_pretrained(WebshopEmbedder.pretrained_path)

            if WebshopEmbedder._model is None and not self.unfreeze_layers:
                WebshopEmbedder._model = MarkupLMModel.from_pretrained(WebshopEmbedder.pretrained_path).to(device)
                WebshopEmbedder._model.eval()

            if not self.unfreeze_layers:
                self.model = WebshopEmbedder._model

            if self.unfreeze_layers:
                self.model = MarkupLMModel.from_pretrained(WebshopEmbedder.pretrained_path).to(device)
                if not self.use_pooled:
                    self.model.pooler = None
                # Freeze all layers
                for name, param in self.model.named_parameters():
                    if any([layer in name for layer in self.unfreeze_layers]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                print(f"Num trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

            lm_embedding_size = self.model.config.hidden_size
        else:
            lm_embedding_size = observation_space.shape[0]

        if not self.use_pooled and not self.unfreeze_layers:
            self.cls_embedding = nn.Embedding(2, lm_embedding_size)
            encoder_layers = nn.TransformerEncoderLayer(lm_embedding_size, self.heads, lm_embedding_size, self.dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.n_layers)

            self.fc1 = nn.Linear(lm_embedding_size, embed_dim)
        elif self.use_pooled and not self.unfreeze_layers:
            self.fc1 = nn.Linear(lm_embedding_size, lm_embedding_size)
            self.fc2 = nn.Linear(lm_embedding_size, embed_dim)
            self.fc3 = nn.Linear(embed_dim, embed_dim)
        else:
            self.fc1 = nn.Linear(WebshopEmbedder.lm_embedding_size, embed_dim)


    def _clean_dom(self, dom):
        # Remove best product label
        dom = re.sub(r'<div id="best-products".*?</div>\n', "", dom, flags=re.DOTALL)

        # Remove instruction text
        dom = re.sub(r'<div id="instruction-text".*?</div>\n', "", dom, flags=re.DOTALL)

        # Remove head
        return re.sub(r'<head>.*?</head>\n', "", dom, flags=re.DOTALL)


    def _get_instruction(self, dom):
        return re.findall(r'<div id="instruction-text".*?>(.*?)</div>', dom, flags=re.DOTALL)[0]


    def forward(self, obs):

        # Turn into B x S x D tensor
        if not isinstance(obs, list):
            obs = [obs]

        questions = [o.question for o in obs]
        obs = [o.observation for o in obs]

        if not self.is_already_embedded:
        
            obs = [self._clean_dom(o) for o in obs]

            # Separate cached / not cached
            not_cached = [(o, q) for o, q in zip(obs, questions) if q+o not in WebshopEmbedder.EMBEDDING_CACHE]
            not_cached_obs = [o for o, _ in not_cached]
            not_cached_questions = [q for _, q in not_cached]
            
            if len(not_cached_obs) > 0:
                encoding = self.processor(html_strings=not_cached_obs, questions=not_cached_questions, padding=True, max_length=512, truncation=True, return_tensors="pt")

                if not self.unfreeze_layers:
                    with torch.no_grad():
                        outputs = self.model(**encoding)
                else:
                    outputs = self.model(**encoding)

                if self.use_pooled:
                    outputs = outputs["pooler_output"]
                else:
                    outputs = outputs["last_hidden_state"]
                
                # List of 1 x S x D tensors or 1 x D tensors (pooled)
                outputs = [o.unsqueeze(0) for o in outputs]

                # Cache outputs
                if self.use_buffer:
                    for o, q, output in zip(not_cached_obs, not_cached_questions, outputs):
                        WebshopEmbedder.EMBEDDING_CACHE[q+o] = output

            # Get cached outputs
            if self.use_buffer:
                outputs = [WebshopEmbedder.EMBEDDING_CACHE[q+o] for o, q in zip(obs, questions)]

        else:
            outputs = [o.detach() for o in obs]

        if not self.use_pooled and not self.unfreeze_layers:
            max_len = max([o.shape[1] for o in outputs])
            
            # compute mask of shape B x S
            pre_mask = [torch.zeros((o.shape[0], o.shape[1]), dtype=torch.bool).to(device) for o in outputs]
            pre_mask = [F.pad(o, (0, max_len - o.shape[1]), "constant", 1) for o in pre_mask]
            src_pad_mask = torch.cat(pre_mask, dim=0)

            # Now pad all sequences to max length
            outputs = [F.pad(o, (0, 0, 0, max_len - o.shape[1]), "constant", 0) for o in outputs]

            # Outputs is now B x S x D
            outputs = torch.cat(outputs, dim=0)
            # Add cls token
            cls_embedding = self.cls_embedding(torch.zeros((outputs.shape[0], 1), dtype=torch.long).to(device))
            outputs = torch.cat([cls_embedding, outputs], dim=1).permute(1, 0, 2)

            # Add zeros to pad mask
            src_pad_mask = torch.cat([torch.zeros((outputs.shape[1], 1), dtype=torch.bool).to(device), src_pad_mask], dim=1)
            outputs = self.transformer_encoder(outputs, src_key_padding_mask=src_pad_mask).permute(1, 0, 2)
            outputs = outputs[:,0,:]
            outputs = self.fc1(outputs)
        elif self.use_pooled and not self.unfreeze_layers:
            outputs = torch.cat(outputs, dim=0)
            outputs = F.relu(self.fc1(outputs))
            outputs = F.relu(self.fc2(outputs))
            outputs = self.fc3(outputs)
        elif self.use_pooled and self.unfreeze_layers:
            outputs = torch.cat(outputs, dim=0)
            outputs = self.fc1(outputs)
        elif self.use_pooled and self.unfreeze_layers:
            outputs = [o[:,0,:] for o in outputs]
            outputs = torch.cat(outputs, dim=0)
            outputs = self.fc1(outputs)
        else:
            raise NotImplementedError("This should not happen")

        if self.final_relu:
            return F.relu(outputs)
        return outputs


class SimpleGridStateEmbedder(Embedder):
    """Embedder for SimpleGridEnv states.

    Concretely, embeds (x, y) separately with different embeddings for each cell.
    """

    def __init__(self, observation_space, embed_dim):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (spaces.Box): limits for the observations to embed.
        """
        super().__init__(embed_dim)

        assert all(dim == 0 for dim in observation_space.low)
        assert observation_space.dtype == int

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim, hidden_size) for dim in observation_space.high])
        self._fc_layer = nn.Linear(hidden_size * len(observation_space.high), 256)
        self._final_fc_layer = nn.Linear(256, embed_dim)

    def forward(self, obs):
        tensor = torch.stack(obs)
        embeds = []
        for i in range(tensor.shape[1]):
            embeds.append(self._embedders[i](tensor[:, i]))
        return self._final_fc_layer(F.relu(self._fc_layer(torch.cat(embeds, -1))))


class IDEmbedder(Embedder):
    """Embeds N-dim IDs by embedding each component and applying a linear
    layer."""

    def __init__(self, observation_space, embed_dim):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (np.array): discrete max limits for each dimension of the
                state (expects min is 0).
        """
        super().__init__(embed_dim)

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim, hidden_size) for dim in observation_space])
        self._fc_layer = nn.Linear(hidden_size * len(observation_space), embed_dim)

    @classmethod
    def from_config(cls, config, observation_space):
        return cls(observation_space, config.get("embed_dim"))

    def forward(self, obs):
        tensor = obs
        if len(tensor.shape) == 1:  # 1-d IDs
            tensor = tensor.unsqueeze(-1)

        embeds = []
        for i in range(tensor.shape[1]):
            embeds.append(self._embedders[i](tensor[:, i]))
        return self._fc_layer(torch.cat(embeds, -1))


class FixedVocabEmbedder(Embedder):
    """Wrapper around nn.Embedding obeying the Embedder interface."""

    def __init__(self, vocab_size, embed_dim):
        """Constructs.

        Args:
            vocab_size (int): number of unique embeddings.
            embed_dim (int): dimension of output embedding.
        """
        super().__init__(embed_dim)

        self._embedder = nn.Embedding(vocab_size, embed_dim)

    @classmethod
    def from_config(cls, config):
        return cls(config.get("vocab_size"), config.get("embed_dim"))

    def forward(self, inputs):
        """Embeds inputs according to the underlying nn.Embedding.

        Args:
            inputs (list[int]): list of inputs of length batch.

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        tensor_inputs = torch.tensor(np.stack(inputs)).long()
        return self._embedder(tensor_inputs)


class LinearEmbedder(Embedder):
    """Wrapper around nn.Linear obeying the Embedder interface."""

    def __init__(self, input_dim, embed_dim):
        """Wraps a nn.Linear(input_dim, embed_dim).

        Args:
            input_dim (int): dimension of inputs to embed.
            embed_dim (int): dimension of output embedding.
        """
        super().__init__(embed_dim)

        self._embedder = nn.Linear(input_dim, embed_dim)

    @classmethod
    def from_config(cls, config):
        return cls(config.get("input_dim"), config.get("embed_dim"))

    def forward(self, inputs):
        """Embeds inputs according to the underlying nn.Linear.

        Args:
            inputs (list[np.array]): list of inputs of length batch.
                Each input is an array of shape (input_dim).

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        inputs = np.stack(inputs)
        if len(inputs.shape) == 1:
            inputs = np.expand_dims(inputs, 1)
        tensor_inputs = torch.tensor(inputs).float()
        return self._embedder(tensor_inputs)


class ExperienceEmbedder(Embedder):
    """Optionally embeds each of:

        - state s
        - instructions i
        - actions a
        - rewards r
        - done d

    Then passes a single linear layer over their concatenation.
    """

    def __init__(self, state_embedder, instruction_embedder, action_embedder,
                             reward_embedder, done_embedder, embed_dim):
        """Constructs.

        Args:
            state_embedder (Embedder | None)
            instruction_embedder (Embedder | None)
            action_embedder (Embedder | None)
            reward_embedder (Embedder | None)
            done_embedder (Embedder | None)
            embed_dim (int): dimension of the output
        """
        super().__init__(embed_dim)

        self._embedders = collections.OrderedDict()
        if state_embedder is not None:
            self._embedders["state"] = state_embedder
        if instruction_embedder is not None:
            self._embedders["instruction"] = instruction_embedder
        if action_embedder is not None:
            self._embedders["action"] = action_embedder
        if reward_embedder is not None:
            self._embedders["reward"] = reward_embedder
        if done_embedder is not None:
            self._embedders["done"] = done_embedder

        # Register the embedders so they get gradients
        self._register_embedders = nn.ModuleList(self._embedders.values())
        self._final_layer = nn.Linear(
                sum(embedder.embed_dim for embedder in self._embedders.values()),
                embed_dim)

    def forward(self, instruction_states):
        """Embeds the components for which this has embedders.

        Args:
            instruction_states (list[InstructionState]): batch of states.

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        def get_inputs(key, states):
            if key == "state":
                return [state.observation for state in states]
            elif key == "instruction":
                return [torch.tensor(state.instructions) for state in states]
            elif key == "action":
                actions = np.array(
                        [state.prev_action if state.prev_action is not None else -1
                         for state in states])
                return actions + 1
            elif key == "reward":
                return [state.prev_reward for state in states]
            elif key == "done":
                return [state.done for state in states]
            else:
                raise ValueError("Unsupported key: {}".format(key))

        embeddings = []
        for key, embedder in self._embedders.items():
            inputs = get_inputs(key, instruction_states)
            embeddings.append(embedder(inputs))
        return self._final_layer(F.relu(torch.cat(embeddings, -1)))
