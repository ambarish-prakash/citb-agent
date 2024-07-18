import random
import numpy as np
from models.board import Board
from models.player import Player
from models.model_based_player import ModelBasedPlayer
from utilities import move_to_readable
import pdb

class RoundEnv():
    UNPLAYED_CARD = 32
    AGENT_PLAYER_NUMBER = 0

    def __init__(self, model=None, verbose='INFO'): 
        self.board = Board()
        self.players = [Player()]
        for i in range(3):
            if(model):
                pl = ModelBasedPlayer(model)
            else:
                pl = Player()
            pl.set_name(f'Bot Player {i+1}')
            self.players.append(pl)

        self.starting_player = random.randint(0,3)
        self.current_player = self.starting_player
        self.played_moves = []
        self.agent = self.players[0]
        self.agent_idx = 0
        self.verbose = verbose

    def distribute_cards(self):
        cards = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8]
        random.shuffle(cards)
        for i,player in enumerate(self.players):
            player.set_hand(cards[i*10:i*10+10])

    def get_agent_observation_space(self):
        return self.get_observation_space(self.agent_idx)

    def get_observation_space(self, player_idx):
        player = self.players[player_idx]
        player_idx_diff = player_idx - self.agent_idx
        
        action_num = player.action_num
        obs_space_list = [player.action_num]
        if action_num == 0:
            obs_space_list += [1,0,0] # discard phase
        elif action_num == 1:
            obs_space_list += [0,1,0] # bet phase
        else:
            obs_space_list += [0,0,1] # play phase
        
        # board
        board_places = self.board.get_places()
        board_places_flattened = [item for sublist in board_places for item in sublist]
        board_places_flattened = [((x-1-player_idx_diff)%4)+1 if x !=0 else 0 for x in board_places_flattened]
        obs_space_list += board_places_flattened

        # hand cards
        cards_in_hand = player.get_hand()
        for i in range(1,9):
            obs_space_list.append(cards_in_hand.count(i))
        
        # played / discarded cards
        played_cards = player.get_played_cards()
        for i in range(1,9):
            obs_space_list.append(played_cards.count(i))
        
        # colors, bets and sets won (player stats)
        player_colors = []
        player_bets = []
        player_collected_sets = []
        for i in range(len(self.players)):
            ith_player = self.players[(i+player_idx_diff)%4]
            player_colors += ith_player.get_colors()
            player_bets += [ith_player.get_bet()]
            player_collected_sets += [ith_player.get_sets_won()]

        obs_space_list += player_colors
        obs_space_list += player_bets
        obs_space_list += player_collected_sets

        # Current set played cards
        current_played_moves = self.played_moves.copy()
        current_played_moves.extend([self.UNPLAYED_CARD] * (4-len(self.played_moves)))
        obs_space_list += current_played_moves

        # Starting Player
        obs_space_list += [(self.starting_player-player_idx_diff)%4]

        return np.array(obs_space_list)
        
    def invalid_agent_discard(self, card_number):
        return card_number not in self.agent.get_hand()

    def discard_cards(self, agent_card_number):
        self.agent.discard_one(card_number=agent_card_number)

        for i in range(1,4):
            self.players[i].discard_one(observation=self.get_observation_space(player_idx=i))
    
    def set_bets(self, agent_bet_number):
        self.agent.set_bet(bet=agent_bet_number)

        for i in range(1,4):
            self.players[i].set_bet(observation=self.get_observation_space(player_idx=i))

        self.play_cards()

    def four_cards_played(self):
        return len(self.played_moves) == 4

    def play_cards(self):
        """ Keeps playing cards until its the agent's turn
            Returns if the round is over either due to paradox or 
            (if all 8 sets have been played (based on action number)
        """
        if self.four_cards_played():
            winning_card_idx = self.calculate_winner(self.played_moves)
            winning_player = (self.starting_player + winning_card_idx ) % 4
            self.players[winning_player].win_set()

            if self.agent.action_num == 10:
                # Eight sets have been played, round is over
                self.calculate_player_scores()
                return True

            self.starting_player = winning_player
            self.current_player = self.starting_player
            self.played_moves = []
            self.play_cards()

        valid_board_moves = self.board.get_valid_places(self.starting_player == self.current_player)
        if self.current_player == self.AGENT_PLAYER_NUMBER:
            # Check if the agent cant play any moves (aka paradox)
            playable_moves = list(set(self.agent.valid_moves).intersection(valid_board_moves))
            if len(playable_moves) == 0:
                self.agent.caused_paradox()
                self.calculate_player_scores()
                return True

            return False
        
        current_player = self.players[self.current_player]
        obs = self.get_observation_space(player_idx=self.current_player)
        move = current_player.play_card(self.played_moves, valid_board_moves, observation=obs)
        
        if(move == -1):
            current_player.caused_paradox()
            self.calculate_player_scores()
            return True
        
        self.play_move(move)
        return self.play_cards()

    def play_move(self, move):
        # print(f'{self.current_player} played {move} - {move_to_readable(move)}.')
        # Place the piece on the board
        self.board.place_piece(move, self.current_player+1)

        # Remove it from the valid moves in the board as well as all players
        for player in self.players:
            player.remove_valid_move(move)

        # Add it to the list of played moves
        self.played_moves.append(move)
        self.current_player = (self.current_player+1)%4

    def is_invalid_agent_play(self,move):
        valid_board_moves = self.board.get_valid_places(self.starting_player == self.current_player)
        valid_player_moves = self.agent.valid_moves
        return move not in valid_board_moves or move not in valid_player_moves

    def get_base_color(self):
        if len(self.played_moves) == 0:
            return -1
        return int(self.played_moves[0] / 8)

    def agent_play_card(self, move):
        self.agent.update_after_play(move, self.get_base_color())
        self.play_move(move)
        finished = self.play_cards()
        agent_score = 0
        if finished:
            agent_score = self.get_agent_score()
        return finished, agent_score
          
    def calculate_winner(self, played_moves):
        if(self.verbose == 'DEBUG'):
            pms = []
            for i in played_moves:
                pms.append(move_to_readable(i))
            print(f'played moves: {" ".join(pms)}')

        # Case 1 - red is played - then highest red
        reds = [x for x in played_moves if x<8]
        if(len(reds) > 0):
            highest_red = max(reds)
            return played_moves.index(highest_red)
        
        # Case 2 - no red - highest suite of base color
        base_color = int(played_moves[0] / 8)
        colors = [x for x in played_moves if int(x/8) == base_color]
        highest_color = max(colors)
        return played_moves.index(highest_color)

    def calculate_player_scores(self):
        for i,player in enumerate(self.players):
            bonus = 0
            if player.bet == player.collected_sets:
                bonus = self.board.get_bonus(i+1)
            player.calc_score(bonus)

    def get_agent_score(self):
        return self.agent.get_score()
