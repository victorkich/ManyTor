import manipulator as tor
from tqdm import tqdm
import time

# Multienv has a matrix representation, (2, 2) is 2x2 = 4 simultaneous environments
env_shape = (3, 2)
epochs = 50
max_steps = 50
obj_number = 7

multienv = tor.Multienv(env_shape=env_shape, obj_number=obj_number)
obs = multienv.reset(returnable=True)
env_number = env_shape[0]*env_shape[1]
epochs_time = []
epoch = 0
timer = time.time()
for i in range(1, epochs):
	time_epoch = time.time()
	for p in tqdm(range(max_steps)):
		action = multienv.action_sample()
		obs2, reward, done = multienv.step(action)
		if done == True:
			break

	if not i % 10:
		multienv.render()
	elif multienv.rendering:
		multienv.render(stop_render=True)

	epoch += 1
	epochs_time.append([i, time.time()-time_epoch])
	print('Total Reward: ', [multienv.environment[i].total_reward for i in range(env_number)])
	print('Epoch: ', epoch)
	multienv.reset()

total_time = time.time()-timer
print('Total Time: ', total_time)
print('Time per Epoch: ', epochs_time)
multienv.render(stop_render=True)
