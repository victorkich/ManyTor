import threading
import numpy as np
import time
import math
import vispy

#-------------------------------------------------FUNCOES TOP-----------------------------------------------------------

def r_t(v1, v2):
	d = [abs(v1[i] - v2[i]) for i in range(3)]
	h_l = math.sqrt(d[0]**2 + d[1]**2)
	r = math.degrees(math.atan2(d[0],d[1]))
	theta = math.degrees(math.atan2(h_l,d[2]))
	return r, theta

def dh(a, alfa, d, theta):
	"""Builds the Homogeneous Transformation matrix	corresponding to each line of the Denavit-Hartenberg parameters.
	"""
	m = np.array([[np.cos(theta), -np.sin(theta) * np.cos(alfa), np.sin(theta) * np.sin(alfa), a * np.cos(theta)],
		[np.sin(theta), np.cos(theta) * np.cos(alfa), -np.cos(theta) * np.sin(alfa), a * np.sin(theta)],
		[0, np.sin(alfa), np.cos(alfa), d],
		[0, 0, 0, 1]])
	return m

def fk(mode, goals):
	"""Forward Kinematics.
	"""
	# Convert angles from degrees to radians
	t = [math.radians(x) for x in goals]
	# Register the DH parameters
	hs = [dh(0, -np.pi / 2, 4.3, t[0])]
	if mode >= 2:
		hs.append(dh(0, np.pi / 2, 0.0, t[1]))
	if mode >= 3:
		hs.append(dh(0, -np.pi / 2, 24.3, t[2]))
	if mode == 4:
		hs.append(dh(27.0, np.pi / 2, 0.0, t[3] - np.pi / 2))

	m = np.eye(4)
	for h in hs:
		m = m.dot(h)
	return m


#-------------------------------------------------CÓDIGO NOVO-----------------------------------------------------------


class Multienv:
	"""Function for start and render multiples Environments with our respective individual Agents. Therefore, start
		this sending: [agent_number:tuple, max_steps:int, obj_number:int]
	"""

	def __init__(self, agent_number, max_steps, obj_number):
		self.environment = [Environment(max_steps, obj_number) for i in range(agent_number)]
		self.agent = [Agent(self.environment[i]) for i in range(agent_number)]

	def render(self):
		for env in self.environment:
			env.render()


class Environment:
	"""Start environment sending [max_steps:int, obj_number:int]. If you want to create a custom model for your own
		manipulator change the get_action(self) and get_observation(self) functions.
	"""

	def __init__(self, max_steps, obj_number):
		self.steps_left = max_steps
		self.obj_number = obj_number
		self.actual_epoch = 0
		self.actual_step = 0
		self.goals = np.zeros(4)
		self.boolplot = np.array([True for i in range(self.obj_number)])
		self.trajectory = np.array([])
		self.joints_coordinates = np.array([])
		self.points = np.array([])

	def get_observations(self):
		distances = np.array([])
		for p in range(self.obj_number):
			# Computing all distances between the terminal and the objective points.
			mod_dist = np.array(
				[(abs(self.joints_coordinates[3, i] - self.points[p, i])) for i in range(3)])  # old self.points
			euc_dist = np.array(math.sqrt(math.sqrt(mod_dist[:, 0] ** 2 + mod_dist[:, 1] ** 2) ** 2 + mod_dist[:, 2] ** 2))
			distances = np.vstack((distances, euc_dist))
		
		v1 = self.joints_coordinates[3, :]
		obs = np.array([])
		obs = np.vstack((obs, distances))
		for p in range(self.obj_number):
			v2 = self.points[p, :]
			obs = obs.vstack((obs, r_t(v1, v2)))
		
		return obs

	def is_done(self):
		# Check and validate if the epoch is done.
		done = False
		for p in range(self.obj_number):
			validation_test = []
			for a in range(3):
				# Check if terminal (x, y and z) is close to each objective (x, y and z)
				if math.isclose(self.joints_coordinates[3, a], self.points[p, a], abs_tol=0.75):
					validation_test.append(True)
				else:
					validation_test.append(False)
			# If the three coordinates is close, the point is reached
			if all(validation_test):
				# points[p, :] = 0.0
				self.boolplot[p] = False

		if not self.boolplot.any():
			done = True

		if not done:
			done = self.steps_left == 0
		return done

	def action(self, action, obs):
		negative_reward = False

		# Generating route to manipulator plot
		route = np.linspace(self.goals, action, num=100)
		for p in range(100):
			self.goals = route[p, :]

			# Modes -> 1 = first joint / 2 = second joint
			#          3 = third joint / 4 = fourth joint
			joints_coordinates = np.array([fk(mode=i, goals=self.goals)[0:3, 3] for i in range(2, 5)])
			self.joints_coordinates = np.vstack((np.zeros(3), joints_coordinates))  # old self.df
			self.trajectory = np.vstack((self.trajectory, self.joints_coordinates[3, :]))  # old self.trajectory
			if self.joints_coordinates[3, 2] < 0:
				negative_reward = True
			time.sleep(0.005)

		obs2 = self.get_observations()

		if self.is_done():
			raise Exception("Game is Over")

		self.steps_left -= 1
		reward = self.get_reward(obs, obs2, negative_reward)
		return reward, obs2

	def action_sample(self):
		sample = [np.random.randint(low=-180, high=180, size=1)[0] for i in range(4)]
		return sample

	def reset(self):
		self.goals = np.zeros(4)
		self.boolplot = np.array([True for i in range(self.obj_number)])
		self.trajectory = np.array([])
		points = np.array([])

		cont = 0
		while cont < self.obj_number:
			# Generating points
			points_coordinates = [np.random.uniform(-51.3, 51.3) for i in range(3)]
			if points_coordinates[2] >= 0:
				# Sphere formula
				value = math.sqrt(
					math.sqrt(points_coordinates[0] ** 2 + points_coordinates[1] ** 2) ** 2 + points_coordinates[2] ** 2)
				if value <= 51.3:
					# Check distance in relation of the center
					points = np.vstack((points, points_coordinates))
					cont = cont + 1

		self.points = points
		obs = self.get_observations()
		return obs

	def get_reward(self, obs, obs2, negative_reward):
		min_dist = min(obs[:self.obj_number])
		min_dist2 = min(obs2[:self.obj_number])
		reward = np.tanh(min_dist-min_dist2)
		if negative_reward:
			reward = -1
		return reward
	
	def render(self):
		pass

class Agent:
	"""This class control the data flow of the agent using the forward kinematics	functions. If you want to create a
		custom model for your own manipulator change the forward kinematics function.
	"""

	def __init__(self, environment):
		self.total_reward = 0.0
		self.env = environment

	def step(self, action):
		obs = self.env.get_observations()
		reward, obs2 = self.env.action(action, obs)
		done = self.env.is_done()
		self.total_reward += reward
		return obs, obs2, reward, done


# ------------------------------------------------------CÓDIGO VELHO---------------------------------------------------

'''
def animate(self, i):
	x, y, z = [np.array(i) for i in [self.df.x, self.df.y, self.df.z]]
	self.ax.clear()
	self.ax.plot3D(x, y, z, 'gray', label='Links', linewidth=5)
	self.ax.scatter3D(x, y, z, color='black', label='Joints')
	self.ax.scatter3D(x[3], y[3], zs=0, zdir='z', label='Projection', color='red')
	# self.ax.scatter3D(0, 0, 4.3, plotnonfinite=False, s=135000, norm=1, alpha=0.2, lw=0)
	x, y, z = [np.array(i) for i in [self.trajectory.x, self.trajectory.y, self.trajectory.z]]
	self.ax.plot3D(x, y, z, c='b', label='Trajectory')

	if self.plotpoints == True:
		x, y, z, = [], [], []
		n_x, n_y, n_z = [np.array(i) for i in [self.points.x, self.points.y, self.points.z]]
		for i in range(self.obj_number[0]):
			if self.boolplot[i]:
				x.append(n_x[i])
				y.append(n_y[i])
				z.append(n_z[i])

		legend = 'Objectives: ' + str(self.obj_remaining[0]) + '/' + str(self.obj_number[0])
		self.ax.scatter3D(x, y, z, color='green', label=legend)

	title = 'Epoch: ' + str(self.actual_epoch) + ' Step: ' + str(self.actual_step)
	self.ax.set_title(title, size=10)
	self.ax.legend(loc=2, prop={'size': 7})
	self.ax.set_xlabel('x')
	self.ax.set_ylabel('y')
	self.ax.set_zlabel('z')
	self.ax.set_xlim([-60, 60])
	self.ax.set_ylim([-60, 60])
	self.ax.set_zlim([0, 60])
'''
