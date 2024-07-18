import numpy as np
import torch

COLOR_MAP = {0:'Red', 1:'Blue', 2:'Green', 3:'Yellow'}

def move_to_readable(move):
    return COLOR_MAP[int(move/8)] + ' ' + str(move%8+1)

def adjacent_indices(idx1, idx2):
    if(idx1[0] == idx2[0]):
        return abs(idx1[1] - idx2[1]) == 1

    if(idx1[1] == idx2[1]):
        return abs(idx1[0] - idx2[0]) == 1
    
def bfs(idx, indices, visited, count):
    count += 1
    visited.append(idx)
    indices.remove(idx)

    for idx2 in indices:
        if(adjacent_indices(idx, idx2)):
            count += bfs(idx2, indices, visited, count)
    return count

def visualize_observation(obs: np.ndarray):
    print(f'Action Number: {obs[0]}')
    _print_board(obs[4:36])
    _print_agent_hand_and_discard(obs[36:52])
    _print_player_colors(obs[52:68])
    _print_played_moves(obs[76:80])
    _print_won_sets(obs[72:76])
    print(f"Starting Player: {obs[80]}")

def _print_won_sets(won_sets):
    disp = "Won sets: "
    for i,x in enumerate(won_sets):
        disp += f'Player {i+1}-{x}, '
    print(disp)


def _print_played_moves(moves):
    readable_moves = []
    for move in moves:
        if move == 32:
            break
        readable_moves.append(move_to_readable(move))
    
    if len(readable_moves) == 0:
        print("No moves played so far")
    else:
        print(f"Moves played: {readable_moves}")
    

def _print_player_colors(colors):
    disp = ""
    for i in range(4):
        pl_colors = colors[i*4: i*4 + 4]
        rem_colors = [COLOR_MAP[x] for x in range(4) if pl_colors[x] != 0]
        disp += f"Player {i+1}: {rem_colors}, "
    print(disp)

def _print_agent_hand_and_discard(places):
    hand = []
    for i,num in enumerate(places[:8]):
        for j in range(num):
            hand.append(i+1)
    print(f"Agent Hand: {hand}")

    discard = []
    for i,num in enumerate(places[8:16]):
        for j in range(num):
            discard.append(i+1)
    print(f"Agent Discard: {discard}")

def _print_board(places):
    disp_mapping = {0: '.', 1: '1', 2:'2', 3: '3', 4: '4'}
    print('\t', end='')
    for x in [1,2,3,4,5,6,7,8]:
        print(f'{x}\t', end='')
    print('\nRed:\t', end='')
    for val in places[0:8]:
        print(f'{disp_mapping[val]}\t', end='')
    print('\nBlue:\t', end='')
    for val in places[8:16]:
        print(f'{disp_mapping[val]}\t', end='')
    print('\nGreen:\t', end='')
    for val in places[16:24]:
        print(f'{disp_mapping[val]}\t', end='')
    print('\nYellow:\t', end='')
    for val in places[24:32]:
        print(f'{disp_mapping[val]}\t', end='')
    print("")

def get_invalid_actions(observations: torch.Tensor) -> torch.Tensor:
        N = observations.shape[0]
        invalid_actions = torch.zeros((N, 43) )
        INVALID_VAL = 1

        # Invalidate (set to high negative) q values for action if board space used up
        for i in range(32):
            mask = observations[:,i+4] != 0
            invalid_actions[:, i+11] = torch.where(mask, INVALID_VAL, invalid_actions[:, i+11])

        # Where no card number to play Invalidate q values for discard and play_card action
        for i in range(8):
            mask = observations[:, i+36] == 0
            # Discard card
            invalid_actions[:, i] = torch.where(mask, INVALID_VAL, invalid_actions[:, i])
            # Play card
            for offset in [0, 8, 16, 24]:
                invalid_actions[:, i+11+offset] = torch.where(mask, INVALID_VAL, invalid_actions[:, i+11+offset])

        # Invalidate q values for action where the color is not available
        for i in range(4):
            mask = observations[:, i+52] == 0
            fill_tensor = torch.full((invalid_actions.shape[0], 19-11), INVALID_VAL)
            invalid_actions[:, 11+i*8:19+i*8] = torch.where(mask[:,None], fill_tensor, invalid_actions[:, 11+i*8:19+i*8])

        # Invalidate q values for action if starting with red when you cannot
        mask1 = observations[:, 80] == 0
        mask2 = torch.all(observations[:, 4:12] == 0, dim=1)
        final_mask = torch.logical_and(mask1, mask2).unsqueeze(1)
        fill_tensor = torch.full((invalid_actions.shape[0], 19-11), INVALID_VAL)
        invalid_actions[:, 11:19] = torch.where(final_mask, fill_tensor, invalid_actions[:, 11:19])
        
        
        # Invalidate q values for actions based on game phase
        # Discard phase
        mask = observations[:, 1] == 1
        fill_tensor = torch.full((invalid_actions.shape[0], 43-8), INVALID_VAL)
        invalid_actions[:, 8:43] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 8:43])

        # Bet phase
        mask = observations[:, 2] == 1
        fill_tensor = torch.full((invalid_actions.shape[0], 8-0), INVALID_VAL)
        invalid_actions[:, 0:8] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 0:8])
        fill_tensor = torch.full((invalid_actions.shape[0], 43-11), INVALID_VAL)
        invalid_actions[:, 11:43] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 11:43])

        # Play phase
        mask = observations[:, 3] == 1
        fill_tensor = torch.full((invalid_actions.shape[0], 11-0), INVALID_VAL)
        invalid_actions[:, 0:11] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 0:11])

        return invalid_actions.bool()

def get_invalid_actions2(observations: torch.Tensor, device='cpu') -> torch.Tensor:
    N = observations.shape[0]
    invalid_actions = torch.zeros((N, 43), dtype=torch.bool).to(device)
    INVALID_VAL = True

    # Invalidate (set to high negative) q values for action if board space used up
    mask_board_space_used = observations[:, 4:36] != 0
    invalid_actions[:, 11:43] = (mask_board_space_used).to(device)

    # Where no card number to play Invalidate q values for discard and play_card action
    mask_no_card_number = (observations[:, 36:44] == 0).to(device)
    invalid_actions[:, :8] = torch.where(mask_no_card_number, INVALID_VAL, invalid_actions[:, :8])
    for offset in [0, 8, 16, 24]:
        invalid_actions[:, 11 + offset:19 + offset] = torch.where(mask_no_card_number, INVALID_VAL, invalid_actions[:, 11 + offset:19 + offset])

    # Invalidate q values for action where the color is not available
    mask_color_not_available = (observations[:, 52:56] == 0).to(device)
    # Repeat fill_tensor to match the required shape
    fill_tensor = torch.full((N, 32), INVALID_VAL).to(device)
    mask_repeated = mask_color_not_available.repeat_interleave(8, dim=1).to(device)
    invalid_actions[:, 11:43] = torch.where(mask_repeated, fill_tensor, invalid_actions[:, 11:43])

    # Invalidate q values for action if starting with red when you cannot
    mask1 = (observations[:, 80] == 0).to(device)
    mask2 = torch.all(observations[:, 4:12] == 0, dim=1).to(device)
    final_mask = torch.logical_and(mask1, mask2).unsqueeze(1).to(device)
    fill_tensor = torch.full((invalid_actions.shape[0], 8), INVALID_VAL).to(device)
    invalid_actions[:, 11:19] = torch.where(final_mask, fill_tensor, invalid_actions[:, 11:19])

    # Invalidate q values for actions based on game phase
    # Discard phase
    mask = (observations[:, 1] == 1).to(device)
    fill_tensor = torch.full((invalid_actions.shape[0], 43 - 8), INVALID_VAL).to(device)
    invalid_actions[:, 8:43] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 8:43])

    # Bet phase
    mask = (observations[:, 2] == 1).to(device)
    fill_tensor = torch.full((invalid_actions.shape[0], 8), INVALID_VAL).to(device)
    invalid_actions[:, 0:8] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 0:8]).to(device)
    fill_tensor = torch.full((invalid_actions.shape[0], 43 - 11), INVALID_VAL).to(device)
    invalid_actions[:, 11:43] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 11:43])

    # Play phase
    mask = (observations[:, 3] == 1).to(device)
    fill_tensor = torch.full((invalid_actions.shape[0], 11), INVALID_VAL).to(device)
    invalid_actions[:, 0:11] = torch.where(mask[:, None], fill_tensor, invalid_actions[:, 0:11])

    return invalid_actions
