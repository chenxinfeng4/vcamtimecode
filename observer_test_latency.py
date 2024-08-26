import ffmpegcv
import tqdm
import numpy as np
import time
import cvdecoder
import netdecoder
import cv2

cv_dec = cvdecoder.Detector('/home/liying_lab/chenxf/ml-project/timetag_model/shufflenet_120_norm.engine')

observer_ip = '10.50.7.109'
net_dec = netdecoder.Netcoder(observer_ip)

stream_url = 'rtsp://10.50.60.6:8554/mystream_usv'
vid = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, stream_url) #, crop_xywh=(0,0,120,60)
time.sleep(3)

iter_bar = tqdm.trange(2000)
for _ in iter_bar:
    net_timestamp = net_dec()
    ret, frame = vid.read()
    if not ret: break
    frame = np.ascontiguousarray(frame[:60, :120, 0])
    cv_timestamp, pvalue = cv_dec(frame.squeeze())
    dt = net_dec.getTimeDelay(cv_timestamp)
    iter_bar.set_description(f"dt={dt:>2.0f}")

    # show the frame using opencv
    cv2.imshow('frame', frame)
vid.release()
cv2.destroyAllWindows()
