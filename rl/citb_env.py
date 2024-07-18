import pdb
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
from models.round_env import RoundEnv
from utilities import visualize_observation as vo

class CitbEnv(gym.Env):
    metadata = {"render_modes": ["console"]}
    
    def __init__(self, render_mode="console", model=None, verbose='INFO'):
        super(CitbEnv, self).__init__()
        self.render_mode = render_mode
        self.num_steps = 0
        self.action_numbers_stepped = [0]*10
        self.action_for_number = np.zeros((10, 8 + 3 + 32))

        n_actions = 8 + 3 + 32 # discard number 1-8, bet 1-3, play 1-32
        self.action_space = Discrete(n_actions)

        # Observation Space
        obs_space_list = []
        obs_space_list += [10] # action_number
        obs_space_list += ([2]*3) # phases
        obs_space_list += ([5]*32) # board places
        obs_space_list += ([6]*8) # in hand - count per card number
        obs_space_list += ([6]*8) # removed from hand - count per card number
        obs_space_list += ([2]*16) # colors
        obs_space_list += ([4]*4) # bets
        obs_space_list += ([9]*4) # collected sets
        obs_space_list += ([33]*4) # cards played
        obs_space_list += [4] # starting player
        self.obs_space_list = obs_space_list
        self.observation_space = MultiDiscrete(obs_space_list)

        self.model = model
        self.verbose = verbose
        self.round_env = RoundEnv(model=model, verbose=verbose)

    def set_observation_space(self):
        obs_space = self.round_env.get_agent_observation_space()
        self._check_observation(obs_space)
        self.observation_space = obs_space

    def _check_observation(self, observation):
        return 
        # for i, (value, n) in enumerate(zip(observation, self.obs_space_list)):
        #     if value >= n:
        #         print(f"Out of bounds value detected! Index: {i}, Value: {value}, Max: {n}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Create new round
        self.round_env = RoundEnv(model=self.model, verbose=self.verbose)

        # Distribute cards
        self.round_env.distribute_cards()

        # Update obs space
        self.set_observation_space()

        return self.observation_space, {}  # empty info dict
    
    def step(self, action):
        """ Returns returns the next observation, the immediate reward, whether new state is a terminal state (episode is finished), 
        whether the max number of timesteps is reached (episode is artificially finished), and additional information
        """
        self.num_steps += 1

        action_number = self.observation_space[0]
        self.action_numbers_stepped[action_number] += 1
        self.action_for_number[action_number, action] += 1

        #Check phases
        if action_number == 0:
            # should discard
            if action not in range(0,8) or self.round_env.invalid_agent_discard(action+1):
                # invalid action, should penalize
                return self.observation_space, -1000, False, True, {}
            
            self.round_env.discard_cards(action+1)
            self.set_observation_space()
            if self.observation_space[0] == 0:
                pdb.set_trace()
            return self.observation_space, 0, False, False, {}
        
        elif action_number == 1:
            # should bet
            if action not in range(8,11):
                # invalid action, should penalize
                return self.observation_space, -1000, False, True, {}
            
            self.round_env.set_bets(action-7)
            self.set_observation_space()
            return self.observation_space, 0, False, False, {}
        
        else:
            # should play
            move = action - 11
            if move < 0:
                # invalid action, should penalize
                reward = -1000 
                return self.observation_space, reward, False, True, {}
            
            if self.round_env.is_invalid_agent_play(move):
                # played a card, but an invalid card
                turn_advancement_reward = (action_number-2)*10
                reward = -150 + turn_advancement_reward
                pdb.set_trace()
                return self.observation_space, reward, False, True, {}
            
            finished, score = self.round_env.agent_play_card(move)

            self.set_observation_space()
            # TLDR: action_number in obs space is max 10 values, but when finished, action_number could be 10
            # Long: 
            # Target q_value is set based on a network used on the 'next observations' after an action. If the 
            # episode is finished, the target q_value is not used (only reward is used), but apparently it is still calculated.
            # And in this case, I let the action_number go to 10 which implies the 'next observation' has an action_number of 10
            # which is out of bounds. Since Ive already trained the model for quite long, do not want to change the obs space
            # to make it a max val of 11 instead as that would require creating a new MLP network.
            if finished:
                if self.observation_space[0] == 10:
                    self.observation_space[0] = 9

            return self.observation_space, score, finished, False, {}
                       
    def render(self):
        print('xoxoxoxoxoxo')

    def close(self):
        pass
