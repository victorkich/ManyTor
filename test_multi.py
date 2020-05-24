import manipulator as tor
from tqdm import tqdm
import time

# Multienv has a matrix representation, (2, 2) is 2x2 = 4 simultaneous environments
env_shape = (1, 1)
epochs = 50
max_steps = 200
obj_number = 8

multienv = tor.Multienv(env_shape=env_shape, obj_number=obj_number)
obs = multienv.reset(returnable=True)

tempo_epocas = []
rendering = False
epoch = 0
tempo = time.time()
for i in range(1, epochs):
	time_epoch = time.time()
	for p in tqdm(range(max_steps)):
		action = multienv.action_sample()
		obs2, reward, done = multienv.step(action)
		if done == True:
			break

	if not i % 10:
		multienv.render()
		rendering = True
	elif rendering:
		multienv.render(stop_render=True)
		rendering = False
	epoch += 1
	tempo_epocas.append([i, time.time()-time_epoch])
	print('Total Reward: ', [multienv.environment[i].total_reward for i in range(env_shape[0]*env_shape[1])])
	print('Total Epochs: ', epoch)
	multienv.reset()

tempo_total = time.time()-tempo
print('Total Time: ', tempo_total)
print('Time per Epoch: ', tempo_epocas)
multienv.render(stop_render=True)
