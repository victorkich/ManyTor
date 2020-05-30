import manipulator as tor
from tqdm import tqdm
import time

epochs = 300
max_steps = 200
obj_number = 10

env = tor.Environment(obj_number)
obs = env.reset(returnable=True)
env.render()
epochs_time = []
rendering = False
epoch = 0
timer = time.time()
for i in range(1, epochs):
	time_epoch = time.time()
	for p in tqdm(range(max_steps)):
		action = env.action_sample()
		obs2, reward, done = env.step(action)
		if done:
			break
	epoch += 1
	epochs_time.append([i, time.time()-time_epoch])
	print('Total Reward: ', env.total_reward)
	print('Epoch: ', epoch)
	env.reset()

total_time = time.time()-timer
print('Total Time: ', total_time)
print('Time per Epoch: ', epochs_time)
env.render(stop_render=True)
