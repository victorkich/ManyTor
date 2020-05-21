import manipulator as tor
import time

epochs = 3
max_steps = 100
obj_number = 5

env = tor.Environment(obj_number)
obs = env.reset(returnable=True)
env.render()

tempo_epocas = []
epoch = 0
tempo = time.time()
for i in range(epochs):
	time_epoch = time.time()
	for p in range(max_steps):
		action = env.action_sample()
		print('Action: ', action)
		obs2, reward, done = env.step(action)
		print('Observation 2: ', obs2)
		print('Rewards: ', reward)
		print('Done: ', done)
		print('Step: ', p)
		print('Epoch:', epoch)
		if done:
			break

	epoch += 1
	tempo_epocas.append([i, time.time()-time_epoch])
	print('Total Reward: ', env.total_reward)
	print('Total Epochs: ', epoch)
	env.reset()

tempo_total = time.time()-tempo
print('Tempo total: ', tempo_total)
print('Tempo por epoca: ', tempo_epocas)
env.render(stop_render=True)
