from vispy.scene import visuals, SceneCanvas
from vispy import app, scene
import numpy as np
import threading
import socket
import json
import time
import sys

HOST = 'localhost'  # Server ip address
PORT = 5000  		  # Server port


def receive_data(sock, canvas):
	global joints_coordinates
	global points
	global trajectory
	global env_number
	global point
	global joint
	global traject
	global view
	global stop
	stop = False
	obj_number = 0

	while not stop:
		msg_byte, user = udp.recvfrom(1024)
		msg_list = json.loads(msg_byte)
		msg = np.array(msg_list)

		if msg[2] == 2:
			stop = True
			app.quit()
			sock.close()
		elif msg[2] == 3:
			env_shape = msg[0]
			env_number = int(env_shape[0]*env_shape[1])
			obj_number = int(msg[1])
			joints_coordinates = np.empty((env_number, 4, 3))
			points = np.empty((env_number, msg[1], 3))
			trajectory = [np.empty((1, 3)) for _ in range(env_number)]

			view = canvas.central_widget.add_view()
			camera = scene.cameras.TurntableCamera(fov=100)
			camera.set_range((90, -90, -90), (-90, 90, -90), (90, -90, -90))
			view.camera = camera

			threads = []
			line, column = 0, 0

			for i in range(env_number):
				point.append(visuals.Markers())
				joint.append(visuals.LinePlot())
				traject.append(visuals.Markers())

				x = column*80
				y = line*80
				threads.append(threading.Thread(name='Plot ' + str(i), target=update, args=(i, x, y)))

				if column < (env_shape[1]-1):
					column += 1
				else:
					line += 1
					column = 0

				view.add(point[i])
				view.add(joint[i])
				view.add(traject[i])
				threads[i].setDaemon(True)
				threads[i].start()

		else:
			ide = int(msg[0])
			msg = msg.reshape(-1, 3)
			index = msg[0, :]
			joints_coordinates[ide] = msg[1:5, :]
			points[ide] = msg[5:5+obj_number, :]
			if index[2] == 1:
				trajectory[ide] = np.array([0.0, 0.0, 51.3])
			trajectory[ide] = np.vstack((trajectory[ide], msg[5+obj_number:, :]))

		time.sleep(0.001)
		

def update(i, x, y):
	while True:
		poi = points[i].copy()
		poi[:, 0] = points[i][:, 0] + x
		poi[:, 1] = points[i][:, 1] + y
		j_c = joints_coordinates[i].copy()
		j_c[:, 0] = joints_coordinates[i][:, 0] + x
		j_c[:, 1] = joints_coordinates[i][:, 1] + y
		traj = trajectory.copy()
		traj[:, 0] = trajectory[i][:, 0] + x
		traj[:, 1] = trajectory[i][:, 1] + y
		traject[i].set_data(traj, edge_color='w', face_color='white', size=1)
		point[i].set_data(poi, edge_color='w', face_color='green', size=3)
		joint[i].set_data(j_c, color='gray', marker_size=5, face_color='red', edge_color='red')
		time.sleep(0.005)


udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
orig = (HOST, PORT)
udp.bind(orig)

canvas = SceneCanvas(show=True, size=(800, 600), resizable=False, keys="interactive")
point, joint, traject = [], [], []

dfs = threading.Thread(name='data_from_socket', target=receive_data, args=(udp, canvas, ))
dfs.setDaemon(True)
dfs.start()

time.sleep(0.02)
canvas.show()
canvas.app.reuse()
if sys.flags.interactive == 0:
	app.run()
