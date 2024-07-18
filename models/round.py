import random
from models.board import Board
from models.player import Player
from utilities import move_to_readable


class Round:
    color_map = {0: 'Red', 1: 'Blue', 2: 'Green', 3: 'Yellow'}
    def __init__(self):
        self.board = Board()
        self.players = []
        for i in range(4):
            pl = Player()
            pl.set_name(f'Bot Player {i+1}')
            self.players.append(pl)

        self.starting_player = 0
    
    def play_set(self):
        # each player plays a card, set winner is calculated, their set total is updated and they become the current player
        played_moves = []
        for i in range(4):
            current_player_idx = (self.starting_player + i) % 4
            current_player = self.players[current_player_idx]
            valid_moves = self.board.get_valid_places(i == 0)
            move = current_player.play_card(played_moves, valid_moves)
            if(move == -1):
                #handle paradox case
                print(f'{current_player.nickname} caused Paradox!')
                print(f'{current_player.nickname}: Hand remaining - {current_player.hand}. Color Set: {current_player.color_set_readable()}')
                current_player.caused_paradox()
                #pdb.set_trace()
                return True

            # Place the piece on the board
            self.board.place_piece(move, current_player_idx+1)
            print(f'{current_player.nickname} played {move} - {move_to_readable(move)}. \tColor Set - {current_player.color_set_readable()}')

            # Remove it from the valid moves in the board as well as all players
            for player in self.players:
                player.remove_valid_move(move)

            # Add it to the list of played moves
            played_moves.append(move)

        winning_card_idx = self.calculate_winner(played_moves)
        winning_player = (self.starting_player + winning_card_idx ) % 4
        self.players[winning_player].win_set()
        print(f'{self.players[winning_player].nickname} won the set!')
        self.starting_player = winning_player

        return False

    def calculate_winner(self, played_moves):
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

    def play_round(self):
        # Deal out cards to players here
        cards = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8]
        random.shuffle(cards)
        for i,player in enumerate(self.players):
            #clone an array
            player.reset_round()
            player.set_hand(cards[i*10:i*10+10])
            print(f'{player.nickname} - {player.hand}')
            
        # then they each discard one
        for player in self.players:
            player.discard_one()
        
        # then they bet
        for player in self.players:
            player.set_bet()
            print(f'{player.nickname} bet {player.bet}')
        
        # then we play sets
        for i in range(8):
            paradox = self.play_set()
            if paradox:
                break

        # calc final scores
        for i,player in enumerate(self.players):
            bonus = 0
            if player.bet == player.collected_sets:
                bonus = self.board.get_bonus(i+1)
            player.calc_score(bonus)

        #print results
        for player in self.players:
            print(f'{player.nickname} - {player.get_score()} points')
