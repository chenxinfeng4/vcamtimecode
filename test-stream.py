#%%
import ffmpegcv
import socket
from timetag_model.decoder import Detector
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time


tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# serve_ip = '10.50.5.83'
serve_ip = '10.50.7.109'
serve_port = 20173
tcp_socket.connect((serve_ip, serve_port))
#%%

def send_read(send_data):
    send_data_byte = send_data.encode("utf-8")
    tcp_socket.send(send_data_byte)

    from_server_msg = tcp_socket.recv(1024)
    return from_server_msg.decode("utf-8")


def get_global_code():
    code:str = send_read("get_status\n")
    code = code[:3]
    if code.isdigit():
        return int(code)
    else:
        print('code is ', code)
        return None


det = Detector('/home/liying_lab/chenxf/ml-project/timetag_model/resnet18_120_norm.engine')


url = 'rtsp://10.50.60.6:8554/mystream_usv'
vid = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, url, pix_fmt='gray') #, crop_xywh=(0,0,120,60)
time.sleep(3)
delay = []
for _ in tqdm.trange(2000):
    num1 = get_global_code()
    ret, frame = vid.read()
    frame = np.ascontiguousarray(frame[:60, :120])
    num, pvalue = det(frame.squeeze())
    dt = (num1 + 1000 - num) % 1000
    delay.append(dt)
vid.release()

delay = np.array(delay)
plt.hist(delay, bins=np.arange(12)+0.5)
plt.title(f'delay={np.mean(delay)}, max={np.max(delay)}') 
pickle.dump({'delay_frames':delay}, open('/home/liying_lab/chenxf/ml-project/论文图表/实时性能测试/delay_stream_3840_2400_br10M.pkl', 'wb'))

