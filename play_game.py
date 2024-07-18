from rl.citb_env import CitbEnv
from rl.masked_dqn import MaskedDQN
import wandb
import pdb
from utilities import visualize_observation
import numpy as np

LOAD_MODEL = True
USE_MODEL_BASED_OPPONENTS = True

#api = wandb.Api()

run_id = 'gw4g1n20'  # Replace with your run ID
#old_run = api.run(f"ambi-personal/CatInTheBox/{run_id}")

model_file_name = 'model.zip'
#filee = old_run.file(model_file_name)
#filee.download(replace=True)

print(f"Loading model from run {run_id}")
model = MaskedDQN.load(model_file_name)

ce = CitbEnv(model=model, verbose='DEBUG')

def parse_move(user_input):
    color_map = {'r':0, 'b':1, 'g':2, 'y':3}
    i = color_map[user_input[0]]
    j = int(user_input[1]) - 1
    move = i*8 + j
    return move + 11

def play(ce):
    obs, _ = ce.reset()
    visualize_observation(obs)

    user_input = input(">> Discard Card: ")
    obs, rew, terminated, truncated, _  = ce.step(int(user_input)-1)

    if(terminated or truncated):
        print("Game Over")
        return
    
    visualize_observation(obs)
    user_input = input(">> Set Bet: ")
    obs, rew, terminated, truncated, _  = ce.step(int(user_input)+7)
    if(terminated or truncated):
        print("Game Over")
        return
    
    print("\n Bets:")
    for i in range(4):
        print(f"Player {i+1} - {ce.round_env.players[i].get_bet()}")
    
    visualize_observation(obs)
    for i in range(8):
        user_input = input(">> Play Card: ")
        action = parse_move(user_input)
        obs, rew, terminated, truncated, _  = ce.step(action)
        if(terminated or truncated):
            players = ce.round_env.players
            for i,pl in enumerate(players):
                if pl.paradox == True:
                    if i == 0:
                        print(">> You caused a paradox! <<")
                    else:
                        print(f">> Player {i+1} caused a paradox! <<")
            print("Scores:")
            for i in range(4):
                print(f"Player {i+1} - {players[i].get_score()}")
            return
        visualize_observation(obs)

play(ce)