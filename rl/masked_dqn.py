from stable_baselines3 import DQN
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from utilities import get_invalid_actions

import pdb

import random
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import torch

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

class MaskedDQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_final_eps=exploration_final_eps,
            exploration_initial_eps=exploration_initial_eps,
            exploration_fraction=exploration_fraction,
            max_grad_norm=max_grad_norm,
            _init_setup_model=_init_setup_model,
        )
        self.predict_count = 0
        self.sample_count = 0
        self.cdct = 0
        self.ttt = []

    def _reset_exploration_schedule(self, exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.2):
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _reset_counters(self):
        self.predict_count = 0
        self.sample_count = 0
        self.cdct = 0
        self.ttt = []
        self.learning_starts = 0

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        self.predict_count += 1
        #self.ttt.append(observation[0][0])
        #if(self.predict_count>40):
        #    print(self.ttt)
        #    self.trouble_number_whaterver += 1 
        rval = np.random.rand()
        if not deterministic and rval < self.exploration_rate:
            #print(f"Dont know wtf is going on. Deterministic: {deterministic}, Exploration Rate: {self.exploration_rate}, Rval: {rval}")
            if self.policy.is_vectorized_observation(observation):
                n_batch = observation.shape[0]
            else:
                n_batch = 1
            action = self._sample_valid_action(observation, n_batch)
        else:
            #print("Predicting Action Using Model")
            action, state = self.policy.predict(observation, state, episode_start, deterministic)

        return action, state
    
    def _sample_valid_action(self, observations, n_batch):
        self.sample_count += 1
        invalid_actions = get_invalid_actions(torch.tensor(observations))
        actions = []
        for i in range(observations.shape[0]):
            valid_actions = torch.nonzero(~invalid_actions[i]).flatten()
            index = torch.randint(0, valid_actions.size(0), (1,))
            actions.append(valid_actions[index].item())
        return np.array(actions)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                #pdb.set_trace()
                #next_q_values = self.negate_invalid_action_values(next_q_values, replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                if (target_q_values<-10).any():
                    pdb.set_trace()
                    next_q_values.must_break_now()

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def negate_invalid_action_values(self, q_values, observations):
        # q_values.shape = torch.Size([32, 43])
        # observations.shape = torch.Size([32, 81])
        NEGATIVE_VAL = -10

        # Invalidate (set to high negative) q values for action if board space used up
        for i in range(32):
            mask = observations[:,i+4] != 0
            q_values[:, i+11] = torch.where(mask, NEGATIVE_VAL, q_values[:, i+11])

        # Invalidate q values for action where no card number to play
        for i in range(8):
            mask = observations[:, i+36] == 0
            for offset in [0, 8, 16, 24]:
                q_values[:, i+11+offset] = torch.where(mask, NEGATIVE_VAL, q_values[:, i+11+offset])
        
        # Invalidate q values for actions based on game phase
        
        # Discard phase
        mask = observations[:, 1] == 1
        fill_tensor = torch.full((q_values.shape[0], 43-8), NEGATIVE_VAL).to(self.device)
        q_values[:, 8:43] = torch.where(mask[:, None], fill_tensor, q_values[:, 8:43])

        # Bet phase
        mask = observations[:, 2] == 1
        fill_tensor = torch.full((q_values.shape[0], 8-0), NEGATIVE_VAL).to(self.device)
        q_values[:, 0:8] = torch.where(mask[:, None], fill_tensor, q_values[:, 0:8])
        fill_tensor = torch.full((q_values.shape[0], 43-11), NEGATIVE_VAL).to(self.device)
        q_values[:, 11:43] = torch.where(mask[:, None], fill_tensor, q_values[:, 11:43])

        # Play phase
        mask = observations[:, 3] == 1
        fill_tensor = torch.full((q_values.shape[0], 11-0), NEGATIVE_VAL).to(self.device)
        q_values[:, 0:11] = torch.where(mask[:, None], fill_tensor, q_values[:, 0:11])

        return q_values

