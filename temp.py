from utilities import get_invalid_actions, get_invalid_actions2
import torch
from rl.citb_env import CitbEnv
import time

env = CitbEnv()
obs, _ = env.reset()
tobs = torch.reshape(torch.tensor(obs), (1,81))
ooo = env.step(0)
too = torch.reshape(torch.tensor(ooo[0]), (1,81))
tobs = torch.cat((tobs, too), 0)
ooo = env.step(9)
too = torch.reshape(torch.tensor(ooo[0]), (1,81))
tobs = torch.cat((tobs, too), 0)

for i in range(29):
    ce = CitbEnv()
    oo,_ = ce.reset()
    too = torch.reshape(torch.tensor(oo), (1,81))
    tobs = torch.cat((tobs, too), 0)

start = time.time()
val = get_invalid_actions(tobs)
end = time.time()
t1 = end - start
print(t1)

start = time.time()
val2 = get_invalid_actions2(tobs)
end = time.time()
t2 = end - start
print(t2)

print(torch.all(val == val2))
print(round(t1/t2,4))

