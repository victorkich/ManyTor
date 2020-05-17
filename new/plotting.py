from vispy.scene import visuals
from vispy import app
import numpy as np
import vispy.scene
import threading
import socket
import vispy
import time
import sys


HOST = 'localhost'          # Endereco IP do Servidor
PORT = 5005         		# Porta que o Servidor esta

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
orig = (HOST, PORT)
udp.bind(orig)

def receive_data(udp):
	while True:
		msg_byte, _ = udp.recvfrom(1024)
		msg = msg_byte.decode()
		if msg == 'close':
			app.quit()
			udp.close()
			break

dfs = threading.Thread(name = 'data_from_socket', target = receive_data, args=(udp, ))
dfs.setDaemon(True)
dfs.start()

canvas = vispy.scene.SceneCanvas(show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

def solver(t):
    pos = np.array([[0.5 + t/10000, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]])
    return pos

scatter = visuals.Markers()
view.add(scatter)
axis = visuals.XYZAxis(parent=view.scene)

timer = app.Timer()
import time
t = 0.0
def update(ev):
    global scatter
    global t
    global timer
    t += 1.0
    scatter.set_data(solver(t), edge_color=None, face_color=(1, 1, 1, .5), size=np.random.randint(0,30))

timer.connect(update)
timer.start(0)

if __name__ == '__main__':
    canvas.show()
    canvas.app.reuse()
    if sys.flags.interactive == 0:
        app.run()
    print('FIM')

