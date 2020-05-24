import manipulator as tor
from tqdm import tqdm
import time

epochs = 300
max_steps = 100
obj_number = 5

env = tor.Environment(obj_number)
obs = env.reset(returnable=True)

tempo_epocas = []
rendering = False
epoch = 0
tempo = time.time()
for i in range(1, epochs):
	time_epoch = time.time()
	for p in tqdm(range(max_steps)):
		action = env.action_sample()
		obs2, reward, done = env.step(action)
		if done:
			break

	if not i % 10:
		env.render()
		rendering = True
	elif rendering:
		env.render(stop_render=True)
		rendering = False

	epoch += 1
	tempo_epocas.append([i, time.time()-time_epoch])
	print('Total Reward: ', env.total_reward)
	print('Total Epochs: ', epoch)
	env.reset()

tempo_total = time.time()-tempo
print('Total Time: ', tempo_total)
print('Time per Epoch: ', tempo_epocas)
env.render(stop_render=True)
