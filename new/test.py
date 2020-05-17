import manipulator as tor
import time

obj_number = 5
env = tor.Environment(obj_number)
obs = env.reset(True)
env.render()
epoch = 0
epocas = []
tempo = time.time()
for i in range(3):
	epoch += 1
	print('Epoch:', epoch)
	time_epoch = time.time()
	for p in range(50):
		action = env.action_sample()
		print('Action: '+ str(action))
		obs2, reward, done = env.step(action)
		print('Observation 2: '+ str(obs2))
		print('Rewards: '+ str(reward))
		print('Done: '+ str(done))
		print('Step: ', p)
		print('Epoch:', epoch)
		if done:
			break
	epocas.append([i, time.time()-time_epoch])
	print('Total Reward: ' + str(env.total_reward))
	print('Total Epochs: ' + str(epoch))
	print('Reset Value: ' + str(env.reset(True)))

tempo_total = time.time() - tempo
print('Tempo total: '+ str(tempo_total))
print('Tempo por epoch: ')
print(epocas)
env.render(stop_render=True)
