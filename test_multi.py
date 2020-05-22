import manipulator as tor
from tqdm import tqdm
import time

env_number = 2
epochs = 3
max_steps = 100
obj_number = 5

multienv = tor.Multienv(env_number, obj_number)
obs = multienv.reset(returnable=True)
multienv.render()

tempo_epocas = []
epoch = 0
tempo = time.time()
for i in range(epochs):
	time_epoch = time.time()
	for p in tqdm(range(max_steps)):
		action = multienv.action_sample()
		obs2, reward, done = multienv.step(action)
		if done == True:
			break

	epoch += 1
	tempo_epocas.append([i, time.time()-time_epoch])
	print('Total Reward: ', [multienv.environment[i].total_reward for i in range(env_number)])
	print('Total Epochs: ', epoch)
	multienv.reset()

tempo_total = time.time()-tempo
print('Tempo total: ', tempo_total)
print('Tempo por epoca: ', tempo_epocas)
multienv.render(stop_render=True)
