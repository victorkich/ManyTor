import manipulator as tor
from tqdm import tqdm
import time

# Multienv has a matrix representation, (2, 2) is 2x2 = 4 simultaneous environments
env_shape = (2, 2)
epochs = 4
max_steps = 50
obj_number = 12

multienv = tor.Multienv(env_shape=(2, 2), obj_number=obj_number)
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
	print('Total Reward: ', [multienv.environment[i].total_reward for i in range(env_shape[0]*env_shape[1])])
	print('Total Epochs: ', epoch)
	multienv.reset()

tempo_total = time.time()-tempo
print('Tempo total: ', tempo_total)
print('Tempo por epoca: ', tempo_epocas)
multienv.render(stop_render=True)
