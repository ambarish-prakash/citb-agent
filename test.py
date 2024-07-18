from rl.citb_env import CitbEnv
from rl.masked_dqn import MaskedDQN
from rl.masked_dqn_policy import MaskedDQNPolicy
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import wandb
import pdb
from wandb.integration.sb3 import WandbCallback
import time
import statistics
from utilities import visualize_observation
import numpy as np

LOAD_MODEL = True
USE_MODEL_BASED_OPPONENTS = True

config = {
    "policy_type": MaskedDQNPolicy,
    #"policy_type": "MlpPolicy",
    "total_timesteps": 120000,
}

opponent_model = None
if USE_MODEL_BASED_OPPONENTS:
    api = wandb.Api()

    run_id = 'sn1q82qy'  # Replace with your run ID
    old_run = api.run(f"ambi-personal/CatInTheBox/{run_id}")

    model_file_name = 'model.zip'
    filee = old_run.file(model_file_name)
    filee.download(replace=True)

    print(f"Loading opponent model from run {run_id}")
    opponent_model = MaskedDQN.load(model_file_name)

# Instantiate the env
vec_env = make_vec_env(CitbEnv, n_envs=1, env_kwargs={'model': opponent_model})
vec_env.training = False

# Instantiate the model
model = MaskedDQN(config['policy_type'], vec_env, verbose=1,
            exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05,
            train_freq=10,)

#pdb.set_trace()

if LOAD_MODEL:
    api = wandb.Api()

    run_id = 'gw4g1n20'  # Replace with your run ID
    old_run = api.run(f"ambi-personal/CatInTheBox/{run_id}")

    model_file_name = 'model.zip'
    filee = old_run.file(model_file_name)
    filee.download(replace=True)

    print(f"Loading model from run {run_id}")
    model = MaskedDQN.load(model_file_name, vec_env)
    model._reset_counters()
    model._reset_exploration_schedule(0.0, 0.0, 0.0)


EPISODE_COUNT = 1000
ep_ct = 0
ep_rews = []
ep_final_action_num = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
terminated_ct, truncated_ct = 0,0
pos_rew = 0
while ep_ct < EPISODE_COUNT:
    ce = CitbEnv(model=opponent_model)
    obs,state = ce.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        obs2 = obs.copy()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = ce.step(action)
    
    if terminated:
        terminated_ct += 1
    if truncated:
        pdb.set_trace()
        truncated_ct += 1
    ep_final_action_num[obs[0]] += 1
    ep_rews.append(reward)
    if reward > 0:
        pos_rew += 1
    ep_ct += 1
    if ep_ct%250 == 0:
        print(ep_ct)

avg = statistics.mean(ep_rews)
print(avg)
print(ep_final_action_num)
print(pos_rew)
print(f'{terminated_ct} terminated, {truncated_ct} truncated, {ep_ct} episodes')

ee = np.array(ep_rews)
v,vcs = np.unique(ee, return_counts=True)
vcounts = dict(zip(v, vcs))
print(vcounts)

#pdb.set_trace()

# Results
# Base Masked: -288/-302
# Trained (vhbpfajc): -177
# Trained (y3krahtn): -15 / -20
# 

