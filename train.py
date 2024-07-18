from rl.citb_env import CitbEnv
from rl.masked_dqn import MaskedDQN
from rl.masked_dqn_policy import MaskedDQNPolicy
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import wandb
import pdb
from wandb.integration.sb3 import WandbCallback
import time
import random
import cProfile
import pstats
import io


USE_WANDB = True
LOAD_MODEL = True
USE_MODEL_BASED_OPPONENTS = True
PROFILE_CODE = False

config = {
    "policy_type": MaskedDQNPolicy,
    #"policy_type": "MlpPolicy",
    "total_timesteps": 600000,
}

tensorboard_log=f"runs/{random.randint(1000, 9999)}"

# Setup WandB
if USE_WANDB:
    run = wandb.init(
        project="CatInTheBox",
        entity="ambi-personal",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    tensorboard_log=f"runs/{run.id}"

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
vec_env = make_vec_env(CitbEnv, n_envs=4, env_kwargs={'model': opponent_model})

# Instantiate the model
model = MaskedDQN(config['policy_type'], vec_env, verbose=1,
            exploration_fraction=0.2, exploration_initial_eps=1.0, exploration_final_eps=0.1,
            tensorboard_log=tensorboard_log, train_freq=10, learning_starts=0)

#pdb.set_trace()

if LOAD_MODEL:
    api = wandb.Api()

    run_id = 'sn1q82qy'  # Replace with your run ID
    old_run = api.run(f"ambi-personal/CatInTheBox/{run_id}")

    model_file_name = 'model.zip'
    filee = old_run.file(model_file_name)
    filee.download(replace=True)

    print(f"Loading model from run {run_id}")
    model = MaskedDQN.load(model_file_name, vec_env)
    model._reset_counters()
    model._reset_exploration_schedule(1.0, 0.05, 0.2)

#pdb.set_trace()

# Train the model
def train_model():
    if USE_WANDB:
        model.learn(total_timesteps=config["total_timesteps"], log_interval=1000, progress_bar=True,
                callback=WandbCallback(
                    gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2,
                ))
        run.finish()
    else:
        model.learn(total_timesteps=config["total_timesteps"], log_interval=1000, progress_bar=True)

print("Training model...")
start_time = time.time()
if PROFILE_CODE:
    pr = cProfile.Profile()
    pr.enable()
    train_model()
    pr.disable()
else:
    train_model()
end_time = time.time()
execution_time = end_time - start_time

minutes = execution_time // 60
seconds = execution_time % 60

print(f"Execution time: {minutes} minutes, {seconds} seconds")

if PROFILE_CODE:
    # Create an output stream to save the profiling results
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    # Save profiling results to a file
    with open("profile_results.txt", "w") as f:
        f.write(s.getvalue())

    print("Profiling complete. Results are saved in 'profile_results.txt'.")

for env in model.env.envs:
  print('Environment:')
  print(env.num_steps)
  print(env.action_numbers_stepped)
  #print(env.action_for_number)

print(model.predict_count)
print(model.sample_count)
print(model.cdct)


print("=========")
e1 = model.env.envs[0]
afn1 = e1.action_for_number.astype(int)
print(afn1)

#pdb.set_trace()
