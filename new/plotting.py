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
points = np.array([[4.,5.,0.]])
trajectory = np.array([0.0, 0.0, 51.3])

def receive_data(udp):
	global joints_coordinates
	global points
	global trajectory

	while True:
		msg_byte, _ = udp.recvfrom(2048)
		msg = np.frombuffer(msg_byte, dtype=np.float64)
		msg = np.array(msg.tolist())
		msg = msg.reshape(-1, 3)
		shapes = msg[0, :]
		joints_coordinates = msg[1:int(shapes[0]),:]
		points = np.array(msg[int(shapes[0]):int(shapes[0]+shapes[1]), :])
		if int(shapes[2]):
			trajectory = np.array([0.0, 0.0, 51.3])
		trajectory = np.vstack((trajectory, msg[int(shapes[0]+shapes[1]):, :]))
		#print(point)
		print(joints_coordinates)
		
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
#grid = visuals.GridLines(parent=view.scene)

point = visuals.Markers()
joint = visuals.LinePlot()
joint2 = visuals.LinePlot()
traject = visuals.Markers()

view.add(point)
view.add(joint)
view.add(joint2)
view.add(axis)
view.add(traject)

timer = app.Timer()
def update(ev):
	#traject.set_data(trajectory, edge_color='w', face_color='blue', size=1)
	point.set_data(points, edge_color='w', face_color='green', size=3)

	joint.set_data(data=joints_coordinates, color='gray', marker_size='5', width='20.0', face_color='red', edge_color='white')
	joint2.set_data(data=j2, color='blue', marker_size='5', width='20.0', face_color='white', edge_color='white')

timer.connect(update)
timer.start(0)

if __name__ == '__main__':
	canvas.show()
	canvas.app.reuse()
	if sys.flags.interactive == 0:
		app.run()
	print('FIM')
