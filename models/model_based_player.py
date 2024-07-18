from models.player import Player
from rl.masked_dqn import MaskedDQN
import wandb
import pdb

class ModelBasedPlayer(Player):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def discard_one(self, observation):
        action = self.get_next_action(observation)
        self.remove_card_from_hand(action+1)
        self.action_num += 1

    def set_bet(self, observation=None, bet=None):
        action = self.get_next_action(observation)
        self.bet = action - 7
        self.action_num += 1

    def play_card(self, played_moves, valid_playable_moves, observation=None):
        playable_moves = list(set(self.valid_moves) & set(valid_playable_moves))
        if(len(playable_moves) == 0):
            return -1

        action = self.get_next_action(observation)
        if(action == 0):
            pdb.set_trace()
        move = action - 11

        first_move = observation[-5]
        base_color = -1
        if first_move != 32:
            base_color = int(first_move / 8)
        self.update_after_play(move, base_color)

        return move

    def get_next_action(self, obs):
        obs[0] = self.action_num
        if self.action_num == 0:
            obs[1:4] = [1,0,0] # discard phase
        elif self.action_num == 1:
            obs[1:4] = [0,1,0] # bet phase
        else:
            obs[1:4] = [0,0,1] # play phase
        action, _states = self.model.predict(obs, deterministic=True)
        # if(self.action_num == 0):
        #     if(action>7):
        #         pdb.set_trace()
        # elif (self.action_num == 1):
        #     if(action<8) or (action>10):
        #         pdb.set_trace()
        # else:
        #     if(action<11):
        #         pdb.set_trace()
        return action
