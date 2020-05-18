from subprocess import call
import numpy as np
import threading
import socket
import time
import math

HOST = 'localhost'     # Endereco IP do Servidor
PORT = 5000            # Porta que o Servidor esta

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def plot_vispy():
    call(["python3", "plotting.py"])

def r_theta(v1, v2):
	d = [abs(v1[i] - v2[i]) for i in range(3)]
	h_l = math.sqrt(d[0]**2 + d[1]**2)
	r = math.degrees(math.atan2(d[0], d[1]))
	theta = math.degrees(math.atan2(h_l, d[2]))
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


class Multienv:
	"""Function for start and render multiples Environments with our respective individual Agents. Therefore, start
		this sending: [agent_number:tuple, obj_number:int]
	"""

	def __init__(self, env_number, obj_number):
		self.environment = [Environment(obj_number) for i in range(env_number)]

	def render(self):
		for env in self.environment:
			env.render()


class Environment:
	"""Start environment sending [obj_number:int]. If you want to create a custom model for your own
		manipulator change the get_action(self) and get_observation(self) functions.
	"""

	def __init__(self, obj_number):
		self.obj_number = obj_number
		self.actual_epoch = 0
		self.actual_step = 0
		self.goals = np.zeros(4)
		self.alives = np.array([True for i in range(self.obj_number)])
		self.trajectory = np.array([0.0, 0.0, 51.3])
		self.joints_coordinates = np.array([])
		self.points = np.array([])
		self.total_reward = 0.0
		self.rendering = False

	def get_observations(self):
		obs = np.array([])
		j_c = self.joints_coordinates[2, :]
		for p in range(self.obj_number):
			# Computing all distances between the terminal and the objective points.
			if not self.alives[p]:
				obs = np.concatenate((obs, [0.0, 0.0, 0.0]), axis=0)
			else:
				mod_dist = np.array([(abs(j_c[i] - self.points[p, i])) for i in range(3)])
				euc_dist = np.array(math.sqrt(math.sqrt(mod_dist[0] ** 2 + mod_dist[1] ** 2) ** 2 + mod_dist[2] ** 2))
				obs = np.concatenate((obs, [euc_dist], r_theta(j_c, self.points[p, :])), axis=0)
		return obs

	def is_done(self):
		# Check and validate if the epoch is done.
		done = False
		for p in range(self.obj_number):
			validation_test = []
			for a in range(3):
				# Check if terminal (x, y and z) is close to each objective (x, y and z)
				if math.isclose(self.joints_coordinates[2, a], self.points[p, a], abs_tol=0.75):
					validation_test.append(True)
				else:
					validation_test.append(False)
			# If the three coordinates is close, the point is reached
			if all(validation_test):
				self.alives[p] = False

		if not self.alives.any():
			done = True

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
			self.joints_coordinates = np.vstack((np.zeros(3), joints_coordinates))
			self.trajectory = np.vstack((self.trajectory, self.joints_coordinates[2, :]))
			if self.joints_coordinates[3, 2] < 0:
				negative_reward = True

			# Send to vispy
			if self.rendering:
				shapes = [self.joints_coordinates.shape[0], self.points.shape[0], 1 if self.trajectory.shape[0] == 1 else 0]
				msg = np.vstack((shapes, self.joints_coordinates, self.points, self.trajectory[-1, :]))
				msg = msg.tobytes()
				self.udp.sendto(msg, self.dest)
				time.sleep(0.0015)

		obs2 = self.get_observations()
		ob = obs[::3]
		ob2 = obs2[::3]
		min_dist = min([ob[i] for i in range(ob.size) if (ob > 0)[i]])
		min_dist2 = min([ob2[i] for i in range(ob2.size) if (ob2 > 0)[i]])
		reward = np.tanh(min_dist - min_dist2)
		if negative_reward:
			reward = -1
		return reward, obs2

	def action_sample(self):
		sample = [np.random.randint(low=-180, high=180, size=1)[0] for i in range(4)]
		return sample

	def reset(self, returnable = False):
		self.goals = np.zeros(4)
		self.total_reward = 0.0
		self.alives = np.array([True for i in range(self.obj_number)])
		self.trajectory = np.array([0.0, 0.0, 51.3])
		joints_coordinates = np.array([fk(mode=i, goals=self.goals)[0:3, 3] for i in range(2, 5)])
		self.joints_coordinates = np.vstack((np.zeros(3), joints_coordinates))
		points = np.zeros(3)

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

		self.points = points[1:, :]
		obs = self.get_observations()

		if returnable:
			return obs
	
	def step(self, action):
		obs = self.get_observations()
		reward, obs2 = self.action(action, obs)
		self.total_reward += reward
		done = self.is_done()
		return obs2, reward, done
	
	def render(self, stop_render=False):
		if not stop_render:
			self.processThread = StoppableThread(target=plot_vispy)
			self.processThread.start()
			self.rendering = True
			self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			self.dest = (HOST, PORT)
		else:
			msg = str.encode('close')
			self.udp.sendto(msg, self.dest)
			time.sleep(0.5)
			self.udp.close()
			self.processThread.stop()
