from utilities import adjacent_indices

class Board:
    def __init__(self):
        self.places = [[0 for _ in range(8)] for _ in range(4)]
        self.valid_places = [x for x in range(32)]
        self.red_played = False

    def get_places(self):
        return self.places

    def place_piece(self, move, player_number):
        row, col = int(move/8), move % 8
        self.places[row][col] = player_number 
        self.valid_places.remove(move)
        if(row == 0):
            self.red_played = True
        #print(f'{move}({move_to_readable(move)}) remove from valid places: {self.valid_places}')

    def get_valid_places(self, is_starting):
        if not is_starting or self.red_played:
            return self.valid_places
        
        #If you are starting and Red has not been played yet, you cannot start with red
        first_valid_actions = [x for x in self.valid_places if x > 7]
        return first_valid_actions
    
    def get_bonus(self, player_num):
        indices = [[i, j] for i, row in enumerate(self.places) for j, elem in enumerate(row) if elem == player_num]
        visited = []
        max_val = 0
        for idx in indices:
            if idx in visited:
                continue
            
            ct = 0
            to_visit = [idx]
            while len(to_visit) > 0:
                idx1 = to_visit[0]
                visited.append(idx1)
                to_visit.remove(idx1)
                ct+=1
                for idx2 in indices:
                    if(idx2 in visited):
                        continue

                    if(adjacent_indices(idx1, idx2)):
                        to_visit.append(idx2)

            max_val = max(max_val, ct)

        return max_val

    def display(self):
        disp_mapping = {0: '.', 1: 'X', 2:'O', 3: '#', 4: '@'}
        print('\t', end='')
        for x in [1,2,3,4,5,6,7,8]:
            print(f'{x}\t', end='')
        print('\nRed:\t', end='')
        for val in self.places[0]:
            print(f'{disp_mapping[val]}\t', end='')
        print('\nBlue:\t', end='')
        for val in self.places[1]:
            print(f'{disp_mapping[val]}\t', end='')
        print('\nGreen:\t', end='')
        for val in self.places[2]:
            print(f'{disp_mapping[val]}\t', end='')
        print('\nYellow:\t', end='')
        for val in self.places[3]:
            print(f'{disp_mapping[val]}\t', end='')
