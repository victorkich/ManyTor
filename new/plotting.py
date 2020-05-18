from vispy.scene import visuals, SceneCanvas
import manipulator as tor
from vispy import app, scene
import numpy as np
import threading
import socket
import time
import sys

HOST = 'localhost'          # Endereco IP do Servidor
PORT = 5000         		# Porta que o Servidor esta

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
orig = (HOST, PORT)
udp.bind(orig)

joints_coordinates = np.array([tor.fk(mode=i, goals=np.zeros(4))[0:3, 3] for i in range(2, 5)])
joints_coordinates = np.vstack((np.zeros(3), joints_coordinates))
points = np.array([])
trajectory = np.array([0.0, 0.0, 51.3])

def receive_data(udp):
	global joints_coordinates
	global points
	global trajectory

	while True:
		msg_byte, _ = udp.recvfrom(1024)
		msg = np.frombuffer(msg_byte, dtype=np.float64)
		msg = np.array(msg.tolist())
		msg = msg.reshape(-1, 3)
		shapes = msg[0, :]
		joints_coordinates = msg[1:int(shapes[0]),:]
		points = msg[int(shapes[0]):int(shapes[0]+shapes[1]), :]
		if int(shapes[2]):
			trajectory = np.array([0.0, 0.0, 51.3])
		trajectory = np.vstack((trajectory, msg[int(shapes[0]+shapes[1]):, :]))
		print(msg)
		
		if msg == 'close':
			app.quit()
			udp.close()
			break

dfs = threading.Thread(name = 'data_from_socket', target = receive_data, args=(udp, ))
dfs.setDaemon(True)
dfs.start()

canvas = SceneCanvas(show=True, size=(800, 600),resizable=False,keys="interactive")

view = canvas.central_widget.add_view()
#view.camera = 'turntable'
camera = scene.cameras.TurntableCamera(fov=60)
camera.set_range((-4,4), (-4,4), (-4,4))
view.camera = camera
#xax = scene.Axis()
axis = visuals.XYZAxis(parent=view.scene)
grid = visuals.GridLines(parent=view.scene)

point = visuals.Markers()
joint = visuals.LinePlot()

view.add(point)
view.add(joint)
view.add(axis)




'''
grid = visuals.GridLines()
joint_base = visuals.LinePlot(parent=view.scene)
joint_one = visuals.LinePlot(parent=joint_base)
joint_two = visuals.LinePlot(parent=joint_one)
joint_three = visuals.LinePlot(parent=joint_two)

view.add(grid)
view.add(joint_base)
view.add(joint_one)
view.add(joint_two)
view.add(joint_three)
'''

timer = app.Timer()
def update(ev):
	'''
	print('Update Coordinates: ')
	print('Coordinates[0]: ', joints_coordinates[0, :])
	print('Coordinates[1]: ', joints_coordinates[1, :])
	print('Coordinates[2]: ', joints_coordinates[2, :])
	print('Coordinates[3]: ', joints_coordinates[3, :])
	'''
	print('Points: ', points)
	joint.set_data(data=joints_coordinates, color='gray', marker_size='10', width='20.0', face_color='red', edge_color='white')
	point.set_data(data=points, edge_color='green', face_color=(1, 1, 1, .5), size=10)

	#joint_base.set_data(data=joints_coordinates[0], color='gray', marker_size='10', width='20.0', face_color='red', edge_color='white')
	#joint_one.set_data(data=joints_coordinates[1], color='blue', marker_size='10', width='4') #parent=joint_base, 
	#joint_two.set_data(data=joints_coordinates[2], color='red', marker_size='10', width='4')
	#joint_three.set_data(data=joints_coordinates[3], color='green', marker_size='10', width='4')
	#scatter_j.set_data(joints_coordinates, edge_color=None, face_color=(1, 1, 1, .5), size=10) #deu certo?

timer.connect(update)
timer.start(0)

if __name__ == '__main__':
	canvas.show()
	canvas.app.reuse()
	if sys.flags.interactive == 0:
		app.run()
	print('FIM')