# from lilab.timecode_tag.netcoder import Netcoder
import numpy as np
import socket
import time


class Netcoder(object):
    def __init__(self, ip='10.50.7.109'):
        self.net_ip_port = (ip, 20173)
        self.lastTime = time.time()
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect(self.net_ip_port)
        self.code_num = -1
        self.send_data_byte = "get_status\n".encode("utf-8")

    def __call__(self) -> int:
        timenow = time.time()
        if timenow - self.lastTime > 0.016:
            self.tcp_socket.send(self.send_data_byte)
            code = self.tcp_socket.recv(1024).decode("utf-8")[:3]
            self.code_num = int(code) if code.isdigit() else -1
            self.lastTime = timenow

        return self.code_num

    def getTimeDelay(self, num) -> float:
        num2 = self()
        diff = (num2+1000 - num) % 100
        diffcorr = np.nan if (num == -1 or diff > 20) else float(diff) 
        return diffcorr
